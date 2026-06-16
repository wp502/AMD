
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

try:
    from open_clip import tokenize as _oc_tokenize
except Exception:
    _oc_tokenize = None

def _unwrap_model(model):
    """Handle DataParallel/DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def _get_device(model):
    """Robustly get the device of a model."""
    m = _unwrap_model(model)
    try:
        return next(m.parameters()).device
    except StopIteration:
        for buf in m.buffers():
            return buf.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_get_tokenizer(model):
    return model.get_tokenizer() if hasattr(model, "get_tokenizer") and model.get_tokenizer() is not None else None
@torch.no_grad()
def _forward_logits(model, batch, device, prefer="auto", fuse_alpha=None):

    images = batch["image"].to(device, non_blocking=True)
    text_raw = batch.get("text", None)

    tok = _maybe_get_tokenizer(model)
    if text_raw is not None:
        if tok is not None:
            text_tokens = tok(text_raw, padding=True, truncation=True, return_tensors="pt").to(device)
        else:
            assert _oc_tokenize is not None, "open_clip.tokenize is required when model has no tokenizer."
            text_tokens = _oc_tokenize(text_raw).to(device)
    else:
        text_tokens = None

    outputs = model(images, text_tokens)

    has_joint = "joint_logits" in outputs
    has_text  = "text_logits"  in outputs
    has_img   = "image_logits" in outputs

    if prefer == "joint" and has_joint:
        return outputs["joint_logits"]
    if prefer == "text" and has_text:
        return outputs["text_logits"]
    if prefer == "image" and has_img:
        return outputs["image_logits"]

    if has_joint:
        return outputs["joint_logits"]
    if has_text and has_img:
        a = 0.7 if fuse_alpha is None else float(fuse_alpha)
        return a * outputs["text_logits"] + (1.0 - a) * outputs["image_logits"]
    if has_text:
        return outputs["text_logits"]
    if has_img:
        return outputs["image_logits"]
    raise ValueError("Model outputs must include one of logits: joint/text/image.")


@torch.no_grad()
def _collect_logits_labels(model, loader, device, prefer="auto", fuse_alpha=None):
    logits_list, labels_list = [], []
    for batch in tqdm(loader, desc="Collecting logits/labels"):
        labels = batch["label"].float().to(device, non_blocking=True)
        logits = _forward_logits(model, batch, device, prefer=prefer, fuse_alpha=fuse_alpha)
        logits_list.append(logits.detach())
        labels_list.append(labels.detach())
    return torch.cat(logits_list, 0), torch.cat(labels_list, 0)


def _fit_temperature_lbfgs(logits, labels, init_T=1.0, max_iter=50):
    T = torch.tensor([init_T], device=logits.device, requires_grad=True)
    bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = bce(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def _per_class_thresholds(probs, labels, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)
    C = probs_np.shape[1]
    thr = np.zeros(C, dtype=np.float32)
    for c in range(C):
        best_f1, best_t = -1.0, 0.5
        pc = probs_np[:, c]
        yc = labels_np[:, c]
        for t in grid:
            pred = (pc >= t).astype(int)
            f1 = f1_score(yc, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr


def _apply_thresholds(probs, thresholds):
    # probs: [N, C] torch; thresholds: [C] np or torch
    if isinstance(thresholds, np.ndarray):
        thr = torch.from_numpy(thresholds).to(probs.device)
    else:
        thr = thresholds.to(probs.device)
    return (probs >= thr.view(1, -1)).int()

@torch.no_grad()
def evaluate_classification(
    model,
    val_loader,
    dataset: str = None,
    thresholds: np.ndarray = None,      
    topk: int = 1,
    prefer: str = "auto",              
    fuse_alpha: float = None,         
    calibrated: bool = True,           
    return_calib: bool = True,       
):
    model.eval()
    device = _get_device(model)

    if dataset is not None and dataset.lower() == "vqav2":
        total = 0
        acc_sum = 0.0
        for batch in tqdm(val_loader, desc="Evaluating (VQA-SoftAcc)"):
            images = batch["image"].to(device, non_blocking=True)
            soft = batch.get("soft_label", batch["label"].float()).to(device)

            tok = _maybe_get_tokenizer(model)
            text_raw = batch.get("text", None)
            if text_raw is not None:
                if tok is not None:
                    text_tokens = tok(text_raw, padding=True, truncation=True, return_tensors="pt").to(device)
                else:
                    assert _oc_tokenize is not None, "open_clip.tokenize is required when model has no tokenizer."
                    text_tokens = _oc_tokenize(text_raw).to(device)
            else:
                text_tokens = None

            outputs = model(images, text_tokens)
            if "joint_logits" in outputs:
                logits = outputs["joint_logits"]
            elif "image_logits" in outputs:
                logits = outputs["image_logits"]
            else:
                raise ValueError("Need 'joint_logits' or 'image_logits' for VQA evaluation.")

            probs = F.softmax(logits, dim=-1)

            if topk == 1:
                pred = probs.argmax(dim=-1)
                gather_soft = soft.gather(1, pred.view(-1, 1)).squeeze(1)
                scores = torch.clamp(gather_soft * (10.0 / 3.0), max=1.0)
            else:
                topk_idx = probs.topk(k=topk, dim=-1).indices
                gather_soft = soft.gather(1, topk_idx)
                best = gather_soft.max(dim=1).values
                scores = torch.clamp(best * (10.0 / 3.0), max=1.0)

            acc_sum += scores.sum().item()
            total += images.size(0)
        return {"vqa_acc": acc_sum / max(total, 1)}

    logits, labels = _collect_logits_labels(model, val_loader, device, prefer=prefer, fuse_alpha=fuse_alpha)

    T = 1.0
    if calibrated:
        T = _fit_temperature_lbfgs(logits, labels, init_T=1.0, max_iter=50)
    probs = torch.sigmoid(logits / T)

    if thresholds is not None:
        if np.isscalar(thresholds):
            thr_used = np.full((probs.size(1),), float(thresholds), dtype=np.float32)
        else:
            thr_used = np.asarray(thresholds, dtype=np.float32)
    else:
        thr_used = _per_class_thresholds(probs, labels) if calibrated else np.full((probs.size(1),), 0.5, dtype=np.float32)

    y_true = labels.int().cpu().numpy()
    y_pred = _apply_thresholds(probs, thr_used).cpu().numpy()

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    example_acc = float((y_pred == y_true).all(axis=1).mean())

    result = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "example_accuracy": example_acc,
    }
    if return_calib:
        result.update({
            "_calib_temperature": float(T),
            "_calib_thresholds": thr_used.tolist() if isinstance(thr_used, np.ndarray) else thr_used,
            "_calib_alpha": fuse_alpha,
        })
    return result


@torch.no_grad()
def evaluate_retrieval(
    model,
    val_loader,
    image_tta: str = "none",          
    text_templates: list = None,      
    max_text_templates: int = 4,       
):
    model.eval()
    device = _get_device(model)
    tokenizer = _maybe_get_tokenizer(model)

    def _image_views(x, mode: str):
        # x: [B, 3, H, W] on device
        views = [x]
        if mode in ("hflip", "hflip+shift"):
            views.append(torch.flip(x, dims=[3]))
        if mode in ("shift", "hflip+shift"):
            pad = 4
            xpad = F.pad(x, pad=(pad, pad, pad, pad), mode="replicate")
            H, W = x.shape[-2:]
            shifts = [
                xpad[:, :, 0:H, pad:pad+W],            
                xpad[:, :, 2*pad:2*pad+H, pad:pad+W],   
                xpad[:, :, pad:pad+H, 0:W],            
                xpad[:, :, pad:pad+H, 2*pad:2*pad+W],  
            ]
            xshift = torch.stack(shifts, dim=0).mean(0)
            views.append(xshift)
        return views

    def _encode_text_ensemble(texts: list, template_list: list):
        if not template_list or len(template_list) == 0:
            if tokenizer is not None:
                text_tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            else:
                assert _oc_tokenize is not None, "open_clip.tokenize is required when model has no tokenizer."
                text_tokens = _oc_tokenize(texts).to(device)
            dummy_img = torch.zeros(1, 3, 224, 224, device=device)
            out = model(dummy_img, text_tokens)
            txt = F.normalize(out["text_feat"].detach().cpu(), dim=-1)
            return txt

        tmpl = template_list[:max_text_templates]
        feats = []
        for t in tmpl:
            cap_t = [t.format(c) for c in texts]
            if tokenizer is not None:
                text_tokens = tokenizer(cap_t, padding=True, truncation=True, return_tensors="pt").to(device)
            else:
                assert _oc_tokenize is not None, "open_clip.tokenize is required when model has no tokenizer."
                text_tokens = _oc_tokenize(cap_t).to(device)
            dummy_img = torch.zeros(1, 3, 224, 224, device=device)
            out = model(dummy_img, text_tokens)
            feats.append(F.normalize(out["text_feat"], dim=-1).detach().cpu())
        txt_feat = torch.stack(feats, dim=0).mean(0)
        txt_feat = F.normalize(txt_feat, dim=-1)
        return txt_feat

    default_templates = ["{}", "a photo of {}"]
    template_list = text_templates if text_templates is not None else default_templates
    if template_list is not None and len(template_list) == 1 and template_list[0] == "{}":
        template_list = []

    img_feat_list, cap_feat_list = [], []
    img_id_list, cap_img_id_list = [], []

    for batch in tqdm(val_loader, desc="Evaluating (Retrieval + TTA)"):
        images = batch["image"].to(device, non_blocking=True)  # [B,3,H,W]
        texts  = batch["text"]
        img_ids = batch["image_id"]  # [B]

        txt_feat = _encode_text_ensemble(texts, template_list)  # [B_cap, D] (CPU)

        # 图像（TTA）
        views = _image_views(images, image_tta) if image_tta != "none" else [images]
        img_feats_each_view = []
        for v in views:
            if tokenizer is not None:
                text_tokens = tokenizer([""], padding=True, truncation=True, return_tensors="pt").to(device)
            else:
                assert _oc_tokenize is not None, "open_clip.tokenize is required when model has no tokenizer."
                text_tokens = _oc_tokenize([""]).to(device)
            out = model(v, text_tokens)
            img_feat_view = F.normalize(out["image_feat"], dim=-1).detach().cpu()
            img_feats_each_view.append(img_feat_view)
        img_feat = torch.stack(img_feats_each_view, dim=0).mean(0)
        img_feat = F.normalize(img_feat, dim=-1)

        img_feat_list.append(img_feat)
        cap_feat_list.append(txt_feat)
        img_id_list.extend([int(i) for i in img_ids])
        cap_img_id_list.extend([int(i) for i in img_ids])

    cap_feats = torch.cat(cap_feat_list, dim=0)                # [C, D] (CPU)
    cap_img_ids = torch.tensor(cap_img_id_list, dtype=torch.long)  # [C]

    all_img_feats = torch.cat(img_feat_list, dim=0)            # [N, D] (CPU)
    all_img_ids   = torch.tensor(img_id_list, dtype=torch.long)
    uniq_ids, inv = torch.unique(all_img_ids, return_inverse=True)  # uniq_ids: [M]
    M, D = uniq_ids.numel(), all_img_feats.size(1)
    img_feats = torch.zeros(M, D)
    img_counts = torch.zeros(M, 1)
    img_feats.index_add_(0, inv, all_img_feats)
    img_counts.index_add_(0, inv, torch.ones_like(all_img_feats[:, :1]))
    img_feats = F.normalize(img_feats / img_counts.clamp_min(1.0), dim=-1)  # [M, D]

    id2row = {int(i): idx for idx, i in enumerate(uniq_ids.tolist())}
    cap_gt_rows = torch.tensor([id2row[int(i)] for i in cap_img_ids.tolist()], dtype=torch.long)  # [C]

    sim = img_feats @ cap_feats.T   # [M, C]

    from collections import defaultdict
    img_row_to_cap_cols = defaultdict(list)
    for col, img_id in enumerate(cap_img_ids.tolist()):
        row = id2row[int(img_id)]
        img_row_to_cap_cols[row].append(col)

    def recall_i2t_at_k(k: int) -> float:
        topk_cols = sim.topk(k=min(k, sim.size(1)), dim=1, largest=True).indices  # [M, k]
        hits = []
        for row in range(img_feats.size(0)):
            gt_cols = set(img_row_to_cap_cols[row])
            pred_cols = set(topk_cols[row].tolist())
            hits.append(1.0 if (gt_cols & pred_cols) else 0.0)
        return float(np.mean(hits)) if hits else 0.0

    def recall_t2i_at_k(k: int) -> float:
        topk_rows = sim.topk(k=min(k, sim.size(0)), dim=0, largest=True).indices  # [k, C]
        hits = (topk_rows == cap_gt_rows.unsqueeze(0)).any(dim=0).float()
        return float(hits.mean().item()) if cap_feats.size(0) > 0 else 0.0

    i2t_r1,  i2t_r5,  i2t_r10   = recall_i2t_at_k(1),  recall_i2t_at_k(5),  recall_i2t_at_k(10)
    t2i_r1,  t2i_r5,  t2i_r10  = recall_t2i_at_k(1),  recall_t2i_at_k(5),  recall_t2i_at_k(10)
    mean_r1, mean_r5, mean_r10 = (i2t_r1+t2i_r1)/2, (i2t_r5+t2i_r5)/2, (i2t_r10+t2i_r10)/2
    #i2t_r1,  i2t_r5,  i2t_r10,  i2t_r50,  i2t_r100   = recall_i2t_at_k(1),  recall_i2t_at_k(5),  recall_i2t_at_k(10),  recall_i2t_at_k(50),  recall_i2t_at_k(100)
    #t2i_r1,  t2i_r5,  t2i_r10,  t2i_r50,  t2i_r100  = recall_t2i_at_k(1),  recall_t2i_at_k(5),  recall_t2i_at_k(10),  recall_t2i_at_k(50),  recall_t2i_at_k(100)
    #mean_r1, mean_r5, mean_r10, mean_r50, mean_r100 = (i2t_r1+t2i_r1)/2, (i2t_r5+t2i_r5)/2, (i2t_r10+t2i_r10)/2, (i2t_r50+t2i_r50)/2, (i2t_r100+t2i_r100)/2

    return {
        "I2T": {"R@1": i2t_r1, "R@5": i2t_r5, "R@10": i2t_r10},
        "T2I": {"R@1": t2i_r1, "R@5": t2i_r5, "R@10": t2i_r10},
        "Mean": {"R@1": mean_r1, "R@5": mean_r5, "R@10": mean_r10},
        "num_images": int(img_feats.size(0)),
        "num_captions": int(cap_feats.size(0)),
        "tta": image_tta,
        "templates": (template_list if template_list else ["{}"]),
    }
