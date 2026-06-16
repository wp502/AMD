# -*- coding: utf-8 -*-
import os
import math
import copy
import argparse
import json
import random
import numpy as np
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import contextmanager

from evaluation import evaluate_classification, evaluate_retrieval
from open_clip import tokenize
from load_data import load_dataset
from load_model import build_model


# ===================== Utils =====================

def log_print(msg, log_path):
    print(msg, flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                pt = targets * xs_pos + (1 - targets) * xs_neg
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                focal_weight = (1 - pt) ** one_sided_gamma
            loss *= focal_weight
        else:
            pt = targets * xs_pos + (1 - targets) * xs_neg
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            focal_weight = (1 - pt) ** one_sided_gamma
            loss *= focal_weight

        return -loss.mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1.0 - d))


def cosine_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@contextmanager
def autocast_ctx(device: torch.device):
    if torch.cuda.is_available() and device.type == 'cuda':
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def _fmt_recall_dict(d: Dict[str, Any], ks=(1, 5, 10, 50, 100)) -> str:
    return " | ".join([f"R@{k}: {float(d.get(f'R@{k}', 0.0)):.4f}" for k in ks])


def _fmt_recall_triplet(eval_result: Dict[str, Any], ks=(1, 5, 10, 50, 100)) -> str:
    i2t = eval_result.get("I2T", {})
    t2i = eval_result.get("T2I", {})
    mean = eval_result.get("Mean", {})
    return (
        f"I2T {_fmt_recall_dict(i2t, ks)} || "
        f"T2I {_fmt_recall_dict(t2i, ks)} || "
        f"Mean {_fmt_recall_dict(mean, ks)}"
    )

@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader, device, fuse_alpha: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list, labels_list = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        text_raw = batch["text"]
        if hasattr(model, "get_tokenizer") and model.get_tokenizer() is not None:
            tok = model.get_tokenizer()
            text_tokens = tok(text_raw, padding=True, truncation=True, return_tensors="pt")
            text_tokens = text_tokens.to(images.device) if hasattr(text_tokens, "to") else text_tokens
        else:
            text_tokens = tokenize(text_raw).to(device, non_blocking=True)

        outputs = model(images, text_tokens)
        labels = batch["label"].float().to(device, non_blocking=True)

        if fuse_alpha is None:
            if "joint_logits" in outputs:
                logits = outputs["joint_logits"]
            elif "text_logits" in outputs and "image_logits" in outputs:
                logits = 0.7 * outputs["text_logits"] + 0.3 * outputs["image_logits"]
            elif "text_logits" in outputs:
                logits = outputs["text_logits"]
            else:
                logits = outputs["image_logits"]
        else:
            tlog = outputs.get("text_logits", None)
            ilog = outputs.get("image_logits", None)
            jlog = outputs.get("joint_logits", None)
            if jlog is not None:
                logits = jlog
            else:
                assert tlog is not None and ilog is not None, "需要 text_logits 和 image_logits 才能 late-fusion"
                logits = fuse_alpha * tlog + (1.0 - fuse_alpha) * ilog

        logits_list.append(logits.detach())
        labels_list.append(labels.detach())

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    bce = nn.BCEWithLogitsLoss()
    T_candidates = torch.linspace(0.5, 3.0, steps=26, device=logits.device)
    best_T, best_loss = 1.0, float("inf")
    for T in T_candidates:
        loss = bce(logits / T, labels).item()
        if loss < best_loss:
            best_loss, best_T = loss, float(T.item())
    return best_T


def compute_micro_macro_f1(pred: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    eps = 1e-12
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    micro_p = tp / (tp + fp + eps)
    micro_r = tp / (tp + fn + eps)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)

    C = labels.size(1)
    f1_sum = 0.0
    for c in range(C):
        y_true = labels[:, c]
        y_pred = pred[:, c]
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        f1_sum += f1
    macro_f1 = f1_sum / C
    return micro_f1, macro_f1


def per_class_threshold_search(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    thresholds = torch.zeros(probs.size(1), device=probs.device)
    grid = torch.linspace(0.05, 0.95, steps=19, device=probs.device)
    for c in range(probs.size(1)):
        best_f1, best_t = -1.0, 0.5
        pc = probs[:, c]
        yc = labels[:, c]
        for t in grid:
            pred = (pc >= t).float()
            tp = ((pred == 1) & (yc == 1)).sum().item()
            fp = ((pred == 1) & (yc == 0)).sum().item()
            fn = ((pred == 0) & (yc == 1)).sum().item()
            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            f1 = 2 * p * r / (p + r + 1e-12)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t.item())
        thresholds[c] = best_t
    return thresholds


def train_teacher_model(args, train_loader, val_loader, test_loader, num_category, model, log_path, save_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()
    base_lr = args.learning_rate
    head_lr_mult = args.head_lr_mult
    wd = 0.01
    norm_wd = 0.0

    patience = args.patience
    no_improve = 0

    decay_params, nodecay_params, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = n.lower()
        is_head = any(k in lname for k in ["proj", "projection", "head", "logit_scale", "classifier"])
        is_norm = any(k in n for k in ("bias", "LayerNorm.weight", "layer_norm.weight", "bn.weight", "bn1.weight", "bn2.weight"))
        if is_head:
            head_params.append(p)
        elif is_norm:
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optim_groups = [
        {"params": decay_params,    "lr": base_lr,                 "weight_decay": wd},
        {"params": nodecay_params,  "lr": base_lr,                 "weight_decay": norm_wd},
        {"params": head_params,     "lr": base_lr * head_lr_mult,  "weight_decay": wd},
    ]
    optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.98), eps=1e-8)

    optim_steps_per_epoch = max(1, len(train_loader) // max(1, args.accumulate_steps))
    total_steps = args.epoch * optim_steps_per_epoch
    warmup_steps = max(100, int(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cosine_warmup(step, total_steps, warmup_steps)
    )

    use_cuda = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_cuda,
        init_scale=2.0**10,
        growth_interval=2000
    )
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    # ====== Loss（分类） ======
    if args.loss_type == "asl":
        loss_fn = AsymmetricLoss(gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    if hasattr(model, "logit_scale"):
        with torch.no_grad():
            model.logit_scale.fill_(math.log(1/0.07))

    best_score = -1e9
    best_epoch = 0
    best_eval_result: Dict[str, Any] = {}
    best_state_dict = None

    for epoch in range(1, args.epoch + 1):
        model.train()
        total_loss = 0.0
        log_print(f"[LR] Epoch {epoch} start | lr={optimizer.param_groups[0]['lr']:.2e}", log_path)

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = batch["image"].to(device, non_blocking=True)
            text_raw = batch["text"]

            if hasattr(model, "get_tokenizer") and model.get_tokenizer() is not None:
                tok = model.get_tokenizer()
                text_tokens = tok(text_raw, padding=True, truncation=True, return_tensors="pt")
                text_tokens = text_tokens.to(images.device) if hasattr(text_tokens, "to") else text_tokens
            else:
                text_tokens = tokenize(text_raw).to(device, non_blocking=True)

            with autocast_ctx(device):
                outputs = model(images, text_tokens)

                if num_category:
                    if args.dataset.lower() == "vqav2":
                        # VQA: soft CE
                        soft = batch.get("soft_label", batch["label"].float()).to(device, non_blocking=True)
                        logits = outputs.get("joint_logits", outputs["image_logits"])
                        loss = soft_cross_entropy(logits, soft)
                    else:
                        labels = batch["label"].float().to(device, non_blocking=True)

                        if args.use_logit_adjust and hasattr(train_loader.dataset, "class_priors"):
                            priors = torch.as_tensor(train_loader.dataset.class_priors, device=device)
                            adjust = args.logit_adjust_tau * torch.log(torch.clamp(priors, min=1e-8))
                        else:
                            adjust = 0.0

                        joint_logits = outputs.get("joint_logits", None)
                        image_logits = outputs.get("image_logits", None)
                        text_logits  = outputs.get("text_logits",  None)

                        w_joint, w_img, w_txt = args.w_joint, args.w_img, args.w_txt

                        with torch.cuda.amp.autocast(enabled=False):
                            labels_f32 = labels.float()
                            loss = 0.0
                            if joint_logits is not None:
                                loss = loss + w_joint * loss_fn(joint_logits.float() - adjust, labels_f32)
                            if image_logits is not None:
                                loss = loss + w_img   * loss_fn(image_logits.float() - adjust, labels_f32)
                            if text_logits is not None:
                                loss = loss + w_txt   * loss_fn(text_logits.float() - adjust, labels_f32)
                else:
                    img_feat = F.normalize(outputs["image_feat"], dim=-1)
                    txt_feat = F.normalize(outputs["text_feat"], dim=-1)

                    scale_i2t = (1.0 / args.contrastive_tau_i2t)
                    scale_t2i = (1.0 / args.contrastive_tau_t2i)

                    sim_i2t = scale_i2t * (img_feat @ txt_feat.T)
                    sim_t2i = scale_t2i * (txt_feat @ img_feat.T)

                    bsz = img_feat.size(0)
                    labels_contrastive = torch.arange(bsz, device=img_feat.device)

                    with torch.no_grad():
                        eye = torch.eye(bsz, device=img_feat.device).bool()
                        k = min(8, bsz - 1)

                        sim_i2t_neg = sim_i2t.masked_fill(eye, float('-inf'))
                        _, hard_idx_i2t = torch.topk(sim_i2t_neg, k=k, dim=1)

                        sim_t2i_neg = sim_t2i.masked_fill(eye, float('-inf'))
                        _, hard_idx_t2i = torch.topk(sim_t2i_neg, k=k, dim=1)

                        def hardness_weight(sim_row, hard_idx):
                            hv = sim_row.gather(1, hard_idx).mean(dim=1)
                            return (torch.sigmoid(hv - hv.detach().mean()) * 0.5 + 0.5).detach()

                        w_i2t = hardness_weight(sim_i2t, hard_idx_i2t)
                        w_t2i = hardness_weight(sim_t2i, hard_idx_t2i)

                    loss_i2t = (F.cross_entropy(sim_i2t, labels_contrastive, reduction='none') * w_i2t).mean()
                    loss_t2i = (F.cross_entropy(sim_t2i, labels_contrastive, reduction='none') * w_t2i).mean()

                    alpha = args.contrastive_alpha_t2i
                    loss = (loss_i2t + alpha * loss_t2i) / (1.0 + alpha)

            loss = loss / max(1, args.accumulate_steps)
            if not torch.isfinite(loss):
                log_print(f"[Warn] skip batch due to non-finite loss: {loss.item()}", log_path)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (step + 1) % max(1, args.accumulate_steps) == 0:
                scaler.unscale_(optimizer)
                if args.clip_grad_norm > 0:
                    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()  

                if hasattr(model, "logit_scale"):
                    with torch.no_grad():
                        model.logit_scale.clamp_(min=math.log(1.0), max=math.log(1000.0))

                if ema is not None:
                    ema.update(model)

            total_loss += loss.item() * max(1, args.accumulate_steps)

        avg_loss = total_loss / max(1, len(train_loader))
        log_print(f"[Train] Epoch {epoch} | Loss: {avg_loss:.4f}", log_path)

        # ====== Eval ======
        model.eval()
        with torch.no_grad():
            if num_category:
                if args.dataset.lower() == "vqav2":
                    eval_result = evaluate_classification(model if ema is None else ema.ema,
                                                          val_loader, dataset=args.dataset)
                    vqa_acc = float(eval_result["vqa_acc"])
                    log_print(f"[Eval-VQA] Epoch {epoch} | Acc: {vqa_acc:.4f}", log_path)
                    current_score = vqa_acc
                else:
                    eval_result = evaluate_classification(model if ema is None else ema.ema,
                                                          val_loader, dataset=args.dataset)
                    log_print(
                        f"[Eval-Classification] Epoch {epoch} | "
                        f"Micro-F1: {eval_result['micro_f1']:.4f} | "
                        f"Macro-F1: {eval_result['macro_f1']:.4f} | "
                        f"Example-Acc: {eval_result['example_accuracy']:.4f}",
                        log_path,
                    )
                    current_score = eval_result["macro_f1"] if args.select_metric == "macro_f1" else eval_result["micro_f1"]

                    if args.calibrated_eval:
                        use_model = model if ema is None else ema.ema
                        best_val_score = -1.0
                        best_alpha = None
                        alpha_candidates = [args.fuse_alpha] if args.fuse_alpha >= 0 else [0.5, 0.6, 0.7, 0.8]
                        for alpha in alpha_candidates:
                            logits_v, labels_v = collect_logits_labels(
                                use_model, val_loader, device,
                                fuse_alpha=alpha if args.fuse_alpha >= 0 else alpha
                            )
                            T = fit_temperature(logits_v, labels_v) if args.use_temperature else 1.0
                            probs_v = torch.sigmoid(logits_v / T)
                            thresholds = per_class_threshold_search(probs_v, labels_v)
                            pred_v = (probs_v >= thresholds).float()
                            mic, mac = compute_micro_macro_f1(pred_v, labels_v)
                            if mac > best_val_score:
                                best_val_score = mac
                                best_alpha = alpha
                                eval_result["_calib_micro_f1"] = mic
                                eval_result["_calib_macro_f1"] = mac
                                eval_result["_calib_T"] = T
                                eval_result["_calib_thresholds"] = thresholds.detach().cpu().tolist()
                                eval_result["_calib_alpha"] = alpha if args.fuse_alpha >= 0 else best_alpha
                        log_print(
                            f"[Eval-Calibration] Epoch {epoch} | "
                            f"Macro-F1*: {eval_result.get('_calib_macro_f1', 0):.4f} "
                            f"(alpha={eval_result.get('_calib_alpha', best_alpha)}, T={eval_result.get('_calib_T', 1.0):.2f})",
                            log_path
                        )
                        if args.select_by_calibrated:
                            current_score = max(eval_result.get("_calib_macro_f1", -1.0), current_score)
            else:
                eval_result = evaluate_retrieval(model if ema is None else ema.ema, val_loader)
                log_print(
                    f"[Eval-Retrieval] Epoch {epoch} | " + _fmt_recall_triplet(eval_result, ks=(1, 5, 10, 50, 100)),
                    log_path
                )
                mean = eval_result.get("Mean", {})
                current_score = float(mean.get("R@1", 0.0))

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_eval_result = eval_result
            src = model if ema is None else ema.ema
            best_state_dict = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log_print(f"[EarlyStop] No improvement for {patience} epochs. Stop at Epoch {epoch}.", log_path)
                break

    assert best_state_dict is not None, "No best_state_dict recorded."
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base, ext = os.path.splitext(save_path)
    best_path = base + "_best" + ext
    torch.save(best_state_dict, best_path)

    log_print(f"✅ Best model saved to: {best_path}", log_path)
    log_print("=" * 60, log_path)
    log_print(f"🏁 Best teacher performance at Epoch {best_epoch}", log_path)

    if num_category:
        if "vqa_acc" in best_eval_result:
            log_print(f"VQA-Acc: {best_eval_result['vqa_acc']:.4f}", log_path)
        else:
            for k, v in best_eval_result.items():
                if isinstance(v, (int, float)):
                    log_print(f"{k}: {v:.4f}", log_path)
    else:
        log_print(_fmt_recall_triplet(best_eval_result, ks=(1, 5, 10, 50, 100)), log_path)

    log_print("=" * 60, log_path)

    if num_category and args.calibrated_test:
        try:
            calib_T = best_eval_result.get("_calib_T", 1.0)
            calib_th = best_eval_result.get("_calib_thresholds", None)
            calib_alpha = best_eval_result.get("_calib_alpha", None)

            use_model = model
            use_model.load_state_dict(best_state_dict, strict=False)
            use_model.to(device).eval()

            def forward_logits(b):
                images = b["image"].to(device, non_blocking=True)
                text_raw = b["text"]
                if hasattr(use_model, "get_tokenizer") and use_model.get_tokenizer() is not None:
                    tok = use_model.get_tokenizer()
                    text_tokens = tok(text_raw, padding=True, truncation=True, return_tensors="pt")
                    text_tokens = text_tokens.to(images.device) if hasattr(text_tokens, "to") else text_tokens
                else:
                    text_tokens = tokenize(text_raw).to(device, non_blocking=True)
                o = use_model(images, text_tokens)
                if "joint_logits" in o:
                    return o["joint_logits"]
                elif calib_alpha is not None and "text_logits" in o and "image_logits" in o:
                    return float(calib_alpha) * o["text_logits"] + (1.0 - float(calib_alpha)) * o["image_logits"]
                elif "text_logits" in o:
                    return o["text_logits"]
                else:
                    return o["image_logits"]

            logits_list, labels_list = [], []
            for b in test_loader:
                with torch.no_grad():
                    lg = forward_logits(b)
                    lb = b["label"].float().to(device, non_blocking=True)
                logits_list.append(lg)
                labels_list.append(lb)

            logits_t = torch.cat(logits_list, 0)
            labels_t = torch.cat(labels_list, 0)

            probs_t = torch.sigmoid(logits_t / float(calib_T))
            if calib_th is None:
                thresholds = torch.full((probs_t.size(1),), 0.5, device=probs_t.device)
            else:
                thresholds = torch.tensor(calib_th, device=probs_t.device, dtype=probs_t.dtype)

            pred_t = (probs_t >= thresholds).float()
            mic_t, mac_t = compute_micro_macro_f1(pred_t, labels_t)

            log_print(f"[Test-Calibrated] Micro-F1: {mic_t:.4f} | Macro-F1: {mac_t:.4f}", log_path)
            return mac_t, mic_t, thresholds, calib_T, calib_alpha
        except Exception as e:
            log_print(f"[Warn] Calibrated test failed: {repr(e)}", log_path)
            return None
    return None


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='flickr-30k',
                        choices=['mmimdb', 'vqav2', 'flickr-30k', 'ms-coco'])

    # model
    parser.add_argument('--teacher_model', default='clip-RN101',
                        choices=['clip-ViT-B-16', 'clip-ViT-L-14', 'clip-RN101', 'resnet18', 'bert-base'])
    parser.add_argument('--project_dim', type=int, default=512)

    # training
    parser.add_argument('--epoch', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=5e-5)   
    parser.add_argument('--warmup_ratio', type=float, default=0.10)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--head_lr_mult', type=float, default=3.0)

    # loss (classification)
    parser.add_argument('--loss_type', type=str, default='asl', choices=['asl', 'bce'])
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0)
    parser.add_argument('--asl_gamma_neg', type=float, default=3.0)
    parser.add_argument('--asl_clip', type=float, default=0.02)

    # multi-branch weights (classification)
    parser.add_argument('--w_joint', type=float, default=1.0)
    parser.add_argument('--w_img', type=float, default=0.5)
    parser.add_argument('--w_txt', type=float, default=0.5)

    # selection & EMA
    parser.add_argument('--select_metric', type=str, default='macro_f1', choices=['macro_f1', 'micro_f1'])
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.9995)

    # calibrated eval & fusion (classification only)
    parser.add_argument('--calibrated_eval', action='store_true')
    parser.add_argument('--use_temperature', action='store_true')
    parser.add_argument('--select_by_calibrated', action='store_true')
    parser.add_argument('--fuse_alpha', type=float, default=-1.0)

    # experiment detail
    parser.add_argument('--gpu', type=str, default='0', choices=['0', '1', '2', '3', '4', '5', '6', '7'])

    # early stop / seed / accumulation
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accumulate_steps', type=int, default=1)

    # calibrated test & thresholds save
    parser.add_argument('--calibrated_test', action='store_true')
    parser.add_argument('--save_thresholds', type=str, default='')

    # long-tail prior adjustment
    parser.add_argument('--use_logit_adjust', action='store_true')
    parser.add_argument('--logit_adjust_tau', type=float, default=1.0)

    parser.add_argument('--contrastive_alpha_t2i', type=float, default=1.3, help='T2I loss weight')
    parser.add_argument('--contrastive_tau_i2t', type=float, default=0.07)
    parser.add_argument('--contrastive_tau_t2i', type=float, default=0.05)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    set_seed(args.seed)

 
    save_dir = 'raw_models/teachers/train_new'
    model_name = args.teacher_model.replace("/", "-")
    save_path = os.path.join(save_dir, f"{model_name}_{args.dataset}_{args.project_dim}.pth")
    log_path = os.path.join(save_dir, f"{model_name}_{args.dataset}_log_{args.project_dim}.txt")

    train_loader, val_loader, test_loader, num_category = load_dataset(
        dataset=args.dataset, batch_size=args.batch_size
    )

    teacher_model = build_model(args.teacher_model, num_category, args.project_dim)

    # Warmup eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device).eval()
    if num_category:
        eval_result = evaluate_classification(teacher_model, val_loader, dataset=args.dataset)
        if args.dataset.lower() == "vqav2":
            log_print(f"[Warmup-VQA] Acc: {eval_result['vqa_acc']:.4f}", log_path)
        else:
            log_print(
                f"Micro-F1: {eval_result['micro_f1']:.4f} | "
                f"Macro-F1: {eval_result['macro_f1']:.4f} | "
                f"Example-Acc: {eval_result['example_accuracy']:.4f}", log_path
            )
    else:
        eval_result = evaluate_retrieval(teacher_model, val_loader)
        log_print(
            f"[Eval-Retrieval] Warmup | " + _fmt_recall_triplet(eval_result, ks=(1, 5, 10, 50, 100)),
            log_path
        )

  
    test_calib_result = train_teacher_model(
        args, train_loader, val_loader, test_loader, num_category, teacher_model, log_path, save_path
    )

    base, ext = os.path.splitext(os.path.join('raw_models/teachers/train_R', f"{model_name}_{args.dataset}.pth"))
    metrics_path = base + "_metrics.json"
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_obj = json.load(f)
    except Exception:
        metrics_obj = {}

    if test_calib_result is not None:
        mac_t, mic_t, thresholds, T, alpha = test_calib_result
        if 'best_eval_result' not in metrics_obj:
            metrics_obj['best_eval_result'] = {}
        metrics_obj['best_eval_result']['test_calib_macro_f1'] = float(mac_t)
        metrics_obj['best_eval_result']['test_calib_micro_f1'] = float(mic_t)
        metrics_obj['best_eval_result']['_calib_T'] = float(T)
        metrics_obj['best_eval_result']['_calib_alpha'] = float(alpha) if alpha is not None else None
        metrics_obj['best_eval_result']['_calib_thresholds'] = [float(x) for x in thresholds.detach().cpu().tolist()]

        if args.save_thresholds:
            try:
                with open(args.save_thresholds, "w", encoding="utf-8") as fth:
                    json.dump({
                        "alpha": float(alpha) if alpha is not None else None,
                        "temperature": float(T),
                        "thresholds": [float(x) for x in thresholds.detach().cpu().tolist()]
                    }, fth, indent=2, ensure_ascii=False)
                log_print(f"💾 Saved calibrated thresholds to: {args.save_thresholds}", log_path)
            except Exception as e:
                log_print(f"[Warn] Save thresholds failed: {repr(e)}", log_path)

    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, indent=4, ensure_ascii=False)
        log_print(f"📄Evaluation metrics saved to: {metrics_path}", log_path)
    except Exception as e:
        log_print(f"[Warn] Write metrics failed: {repr(e)}", log_path)


if __name__ == "__main__":
    main()
