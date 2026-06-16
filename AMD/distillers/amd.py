

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def _cfg_get(cfg: Any, name: str, default: Any):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _to_device(x, device):
    return x.to(device) if (x is not None and hasattr(x, "to")) else x


def move_pack(d: Dict[str, torch.Tensor], device):
    return {k: _to_device(v, device) for k, v in d.items()}


def fill(x: Dict[str, torch.Tensor], device):

    z = torch.tensor([0.0], device=device)

    for k in ["joint_feat", "image_feat", "text_feat", "joint_logits", "image_logits", "text_logits",]:
        if k not in x or x[k] is None:
            x[k] = z

    return x


def _supervised_loss_per_sample(logits: torch.Tensor, targets: torch.Tensor,) -> torch.Tensor:
   
    if targets.dim() == logits.dim():
        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction="none",
        ).mean(dim=-1)

    return F.cross_entropy(
        logits,
        targets.long(),
        reduction="none",
    )


def _kl_per_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float,
) -> torch.Tensor:
    
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)

    kl = F.kl_div(
        log_p_s,
        p_t,
        reduction="none",
    ).sum(dim=-1)

    return kl * (T * T)


def _proj_mse_per_sample(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    B = student_feat.size(0)

    Sd = student_feat.float().view(B, -1)
    Td = teacher_feat.float().view(B, -1)

    with torch.no_grad():
        Ds = Sd.size(1)
        eye = torch.eye(Ds, device=Sd.device, dtype=Sd.dtype)

        A = Sd.t() @ Sd + ridge * eye
        Bm = Sd.t() @ Td

        try:
            W = torch.linalg.solve(A, Bm)
        except RuntimeError:
            W = torch.linalg.pinv(A) @ Bm

    T_hat = Sd @ W

    return F.mse_loss(
        T_hat,
        Td,
        reduction="none",
    ).mean(dim=-1)


def _classification_modality_weights(
    student_dict: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    gamma: float,
    bounds: Tuple[float, float],
) -> Dict[str, torch.Tensor]:
    candidates = []

    for n in ["joint", "image", "text"]:
        has_feat = (
            f"{n}_feat" in student_dict
            and student_dict[f"{n}_feat"] is not None
            and student_dict[f"{n}_feat"].numel() > 1
        )
        has_logits = (
            f"{n}_logits" in student_dict
            and student_dict[f"{n}_logits"] is not None
            and student_dict[f"{n}_logits"].numel() > 1
        )
        if has_feat or has_logits:
            candidates.append(n)

    if len(candidates) == 0:
        raise RuntimeError("No valid modality branch is available for AMD.")

    diffs = []
    for n in candidates:
        loss = _supervised_loss_per_sample(
            student_dict[f"{n}_logits"],
            targets,
        )
        diffs.append(loss.detach())

    D = torch.stack(diffs, dim=1)
    mean_D = D.mean(dim=1, keepdim=True).clamp_min(1e-12)
    raw = (D / mean_D).pow(gamma)

    low, high = bounds
    raw = raw.clamp(low, high)
    raw = raw * (
        len(candidates) / raw.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )

    return {candidates[i]: raw[:, i] for i in range(len(candidates))}


def _cross_modal_directional_entropies(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    temperature: float,
    normalize_entropy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sample-wise cross-modal entropy.

    The diagonal is retained because it represents the matched image-text pair,
    rather than trivial within-modality self-similarity.
    """
    if image_feat.dim() != 2 or text_feat.dim() != 2:
        raise ValueError(
            "Expected image/text features with shape [B, D], got "
            f"{tuple(image_feat.shape)} and {tuple(text_feat.shape)}."
        )
    if image_feat.size(0) != text_feat.size(0):
        raise ValueError("Image and text batches must have the same size.")
    if image_feat.size(0) < 2:
        raise ValueError("Cross-modal entropy requires batch size >= 2.")
    if temperature <= 0:
        raise ValueError("retrieval_proxy_temperature must be positive.")

    image_feat = F.normalize(image_feat.float(), dim=-1)
    text_feat = F.normalize(text_feat.float(), dim=-1)

    logits_i2t = image_feat @ text_feat.t() / temperature
    logits_t2i = logits_i2t.t()

    log_p_i2t = F.log_softmax(logits_i2t, dim=1)
    log_p_t2i = F.log_softmax(logits_t2i, dim=1)

    p_i2t = log_p_i2t.exp()
    p_t2i = log_p_t2i.exp()

    entropy_i2t = -(p_i2t * log_p_i2t).sum(dim=1)
    entropy_t2i = -(p_t2i * log_p_t2i).sum(dim=1)

    if normalize_entropy:
        denom = entropy_i2t.new_tensor(float(image_feat.size(0))).log()
        entropy_i2t = entropy_i2t / denom.clamp_min(1e-12)
        entropy_t2i = entropy_t2i / denom.clamp_min(1e-12)

    return entropy_i2t, entropy_t2i


def _retrieval_modality_weights(
    student_dict: Dict[str, torch.Tensor],
    gamma: float,
    bounds: Tuple[float, float],
    temperature: float,
    normalize_entropy: bool,
) -> Dict[str, torch.Tensor]:
    """
    Image branch uses I2T entropy; text branch uses T2I entropy.
    Larger entropy indicates higher student learning demand.
    """
    entropy_i2t, entropy_t2i = _cross_modal_directional_entropies(
        student_dict["image_feat"],
        student_dict["text_feat"],
        temperature,
        normalize_entropy,
    )

    D = torch.stack(
        [entropy_i2t.detach(), entropy_t2i.detach()],
        dim=1,
    )

    mean_D = D.mean(dim=1, keepdim=True).clamp_min(1e-12)
    raw = (D / mean_D).pow(gamma)

    low, high = bounds
    raw = raw.clamp(low, high)
    raw = raw * (
        2.0 / raw.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )

    return {
        "image": raw[:, 0],
        "text": raw[:, 1],
    }


def _layer_weights(
    logit_or_align_loss: torch.Tensor,
    feat_or_recon_loss: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    use_level_adapt = bool(_cfg_get(config, "use_level_adapt", True))

    gamma_level = float(_cfg_get(config, "gamma_level", 3.0))
    low, high = _cfg_get(config, "level_bounds", (0.5, 2.0))

    lambda_logits = float(_cfg_get(config, "lambda_logits", 0.25))
    lambda_feat = float(_cfg_get(config, "lambda_feat", 0.1))

    if not use_level_adapt:
        total = lambda_logits + lambda_feat + 1e-12
        w_first = logit_or_align_loss.new_tensor(lambda_logits / total)
        w_second = logit_or_align_loss.new_tensor(lambda_feat / total)

        if logit_or_align_loss.dim() > 0:
            w_first = w_first.expand_as(logit_or_align_loss)
            w_second = w_second.expand_as(feat_or_recon_loss)

        return w_first, w_second

    D_first = (
        torch.abs(logit_or_align_loss.detach() - feat_or_recon_loss.detach())
        + logit_or_align_loss.detach()
    )
    D_second = (
        torch.abs(logit_or_align_loss.detach() - feat_or_recon_loss.detach())
        + feat_or_recon_loss.detach()
    )

    D_mean = (D_first + D_second) / 2.0 + 1e-12

    R_first = (D_first / D_mean).pow(gamma_level).clamp(low, high)
    R_second = (D_second / D_mean).pow(gamma_level).clamp(low, high)

    S = R_first + R_second + 1e-12

    return R_first / S, R_second / S


def _classification_teacher_weights(
    teacher_outputs_list,
    targets,
):
    t1, t2 = teacher_outputs_list

    loss1 = _supervised_loss_per_sample(t1, targets)
    loss2 = _supervised_loss_per_sample(t2, targets)
    proxy = torch.stack([loss1, loss2], dim=0)

    weights = torch.softmax(-proxy, dim=0)
    return weights[0], weights[1]


def _retrieval_teacher_weights(
    teacher1_dict: Dict[str, torch.Tensor],
    teacher2_dict: Dict[str, torch.Tensor],
    branch: str,
    temperature: float,
    normalize_entropy: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute branch-specific teacher credibility from cross-modal entropy.

    Image branch uses I2T entropy; text branch uses T2I entropy.
    Lower entropy indicates higher teacher credibility.
    """
    if branch not in ("image", "text"):
        raise ValueError(
            f"Retrieval teacher weighting supports image/text only, got {branch}."
        )

    t1_i2t, t1_t2i = _cross_modal_directional_entropies(
        teacher1_dict["image_feat"],
        teacher1_dict["text_feat"],
        temperature,
        normalize_entropy,
    )
    t2_i2t, t2_t2i = _cross_modal_directional_entropies(
        teacher2_dict["image_feat"],
        teacher2_dict["text_feat"],
        temperature,
        normalize_entropy,
    )

    if branch == "image":
        proxy1, proxy2 = t1_i2t.detach(), t2_i2t.detach()
    else:
        proxy1, proxy2 = t1_t2i.detach(), t2_t2i.detach()

    proxy = torch.stack([proxy1, proxy2], dim=0)
    weights = torch.softmax(-proxy, dim=0)

    return weights[0], weights[1]


def compute_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    device=None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Symmetric image-text InfoNCE loss.
    """
    image_embeddings = F.normalize(image_embeddings.float(), dim=1)
    text_embeddings = F.normalize(text_embeddings.float(), dim=1)

    logits = image_embeddings @ text_embeddings.t() / temperature

    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_i2t + loss_t2i)


def cosine_distillation_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Mean cosine distillation loss.
    Kept for compatibility.
    """
    s = F.normalize(student_embeddings.float(), dim=1)
    t = F.normalize(teacher_embeddings.float(), dim=1)

    return (1.0 - (s * t).sum(dim=1)).mean()


def cosine_distillation_loss_per_sample(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample cosine distillation loss.

    Returns:
        Tensor with shape [B].
    """
    s = F.normalize(student_embeddings.float(), dim=1)
    t = F.normalize(teacher_embeddings.float(), dim=1)

    return 1.0 - (s * t).sum(dim=1)


def compute_amd_loss(
    outputs_t1: Dict[str, torch.Tensor],
    outputs_t2: Dict[str, torch.Tensor],
    outputs_s: Dict[str, torch.Tensor],
    targets=None,
    device: str = "cpu",
    config: Any = None,
) -> torch.Tensor:
    T = float(_cfg_get(config, "T", 2))
    ridge = float(_cfg_get(config, "ridge", 5e-4))

    w_cdist = float(_cfg_get(config, "cosine_loss_weight", 4.0))
    w_contrast = float(_cfg_get(config, "contrastive_loss_weight", 7.5))
    temperature = float(_cfg_get(config, "temperature", 0.06))
    retrieval_proxy_temperature = float(
        _cfg_get(config, "retrieval_proxy_temperature", temperature)
    )
    normalize_retrieval_entropy = bool(
        _cfg_get(config, "normalize_retrieval_entropy", False)
    )

    # Backward compatibility with the original first file:
    # targets may be passed explicitly, or stored in config.targets / config["targets"].
    if targets is None:
        targets = _cfg_get(config, "targets", None)

    targets = _to_device(targets, device) if targets is not None else None

    t1 = fill(move_pack(outputs_t1, device), device)
    t2 = fill(move_pack(outputs_t2, device), device)
    s = fill(move_pack(outputs_s, device), device)

    total_loss = s["image_feat"].new_tensor(0.0)

    is_classification = targets is not None

    if is_classification:
        modality_gamma = float(_cfg_get(config, "amb_gamma", 3.0))
        modality_bounds = _cfg_get(config, "amb_bounds", (0.5, 2.0))

        w_adapt = _classification_modality_weights(
            s,
            targets,
            modality_gamma,
            modality_bounds,
        )

        for br in ["joint", "image", "text"]:
            # Teacher-aware weights: w_{b,1}, w_{b,2}
            w1, w2 = _classification_teacher_weights(
                [t1[f"{br}_logits"], t2[f"{br}_logits"]],
                targets,
            )  # [B], [B]

            # Logit-level losses: L^{logit}_{b,m}
            kd1 = _kl_per_sample(s[f"{br}_logits"], t1[f"{br}_logits"], T, )  # [B]

            kd2 = _kl_per_sample(s[f"{br}_logits"], t2[f"{br}_logits"], T,)  # [B]

            # Feature-level losses: L^{feat}_{b,m}
            f1 = _proj_mse_per_sample(s[f"{br}_feat"], t1[f"{br}_feat"], ridge,)  # [B]

            f2 = _proj_mse_per_sample(s[f"{br}_feat"], t2[f"{br}_feat"], ridge,)  # [B]

            # Level-aware weights for each teacher-branch pair.
            # These are w^l_{b,m}, not branch-level weights.
            w_logit_1, w_feat_1 = _layer_weights(kd1, f1, config)
            w_logit_2, w_feat_2 = _layer_weights(kd2, f2, config)

            # Level aggregation: L_{b,m}
            loss_t1 = w_logit_1 * kd1 + w_feat_1 * f1  # [B]
            loss_t2 = w_logit_2 * kd2 + w_feat_2 * f2  # [B]

            # Teacher aggregation: L_b
            branch_distill = w1 * loss_t1 + w2 * loss_t2  # [B]

            # Modality aggregation: w_b L_b
            wb = w_adapt[br]  # [B]
            total_loss = total_loss + (wb * branch_distill).mean()

    else:
        modality_gamma = float(_cfg_get(config, "amb_gamma", 3.0))
        modality_bounds = _cfg_get(config, "amb_bounds", (0.5, 2.0))

        w_adapt = _retrieval_modality_weights(
            s,
            modality_gamma,
            modality_bounds,
            retrieval_proxy_temperature,
            normalize_retrieval_entropy,
        )

        for br in ["image", "text"]:
            # Teacher-aware weights based on feature uncertainty.
            w1, w2 = _retrieval_teacher_weights(
                t1,
                t2,
                branch=br,
                temperature=retrieval_proxy_temperature,
                normalize_entropy=normalize_retrieval_entropy,
            )  # [B], [B]

            # Alignment-level losses.
            align1 = cosine_distillation_loss_per_sample(s[f"{br}_feat"], t1[f"{br}_feat"], )  # [B]

            align2 = cosine_distillation_loss_per_sample(s[f"{br}_feat"], t2[f"{br}_feat"], )  # [B]

            # Reconstruction-level losses.
            recon1 = _proj_mse_per_sample(s[f"{br}_feat"], t1[f"{br}_feat"], ridge, )  # [B]

            recon2 = _proj_mse_per_sample(s[f"{br}_feat"], t2[f"{br}_feat"], ridge, )  # [B]

            # Level aggregation for retrieval:
            # align/recon are treated as two representation levels.
            w_align_1, w_recon_1 = _layer_weights(align1, recon1, config)
            w_align_2, w_recon_2 = _layer_weights(align2, recon2, config)

            loss_t1 = w_align_1 * align1 + w_recon_1 * recon1
            loss_t2 = w_align_2 * align2 + w_recon_2 * recon2

            # Teacher aggregation.
            branch_distill = w1 * loss_t1 + w2 * loss_t2  # [B]

            # Modality aggregation.
            wb = w_adapt[br]  # [B]
            total_loss = total_loss + w_cdist * (wb * branch_distill).mean()

        contrastive_loss = compute_contrastive_loss(
            s["image_feat"],
            s["text_feat"],
            device=device,
            temperature=temperature,
        )

        total_loss = total_loss + w_contrast * contrastive_loss

    return total_loss