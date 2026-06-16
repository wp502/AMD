import os
import re
import math
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from open_clip import tokenize
from torch.optim.lr_scheduler import LambdaLR
import math


from distillers.msd import compute_msd_loss
from distillers.dsmd import KDModel, AdaptiveLossWeighting, compute_dsmd_loss
from distillers.g2d import compute_g2d_loss
from distillers.dclip import compute_dclip_loss
from distillers.kdmcse import MCSE, compute_kdmcse_loss
from distillers.amd import compute_amd_loss

from evaluation import evaluate_classification, evaluate_retrieval

class RandomDecayLRScheduler:
    def __init__(self, optimizer, decay_rate=0.98, max_noise_pct=0.05):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.max_noise_pct = max_noise_pct

    def step(self):
        for group in self.optimizer.param_groups:
            lr = group["lr"]

            lr = lr * self.decay_rate

            noise = (2 * torch.rand(1).item() - 1) * self.max_noise_pct
            lr = lr * (1 + noise)

            if "initial_lr" in group:
                lr = min(lr, group["initial_lr"])

            group["lr"] = lr

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]



def train_model(args, train_loader, val_loader, num_category,
                teacher_model_1, teacher_model_2, student_model):
    
    def _slug(s: str) -> str:
        s = s.lower().strip().replace(" ", "-")
        s = re.sub(r"[^a-z0-9_\-]+", "", s)
        return s

    def _save_root_students(args, dataset_slug):
        root = getattr(args, "save_dir_students", "raw_models/students")
        return os.path.join(root, distiller_tag, dataset_slug)

    def _save_root_teachers(args, dataset_slug):
        root = getattr(args, "save_dir_teachers", "raw_models/teachers")
        return os.path.join(root, distiller_tag, dataset_slug)

    s_tag = _slug(getattr(args, "student_model", "student"))
    t1_tag = _slug(getattr(args, "teacher_model_1", "teacher1"))
    t2_tag = _slug(getattr(args, "teacher_model_2", "teacher2"))
    distiller_tag = _slug(getattr(args, "distiller", "none"))

    best_score = .0
    best_epoch = 0
    best_eval_result = {}

    need_teacher = args.distiller in {'msd', 'dsmd', 'kdmcse', 'g2d', 'dclip', 'amd'}
    if need_teacher:
        if teacher_model_1 is None or teacher_model_2 is None:
            raise RuntimeError(f"distiller={args.distiller} 需要 teacher_model_1/2，但收到 None。")
        teacher_model_1.cuda().eval()
        teacher_model_2.cuda().eval()

    student_model.cuda()
    student_model.train()

    if args.distiller == 'dsmd':
        dsmd_kdmodel = KDModel(args)
        dsmd_adapter = AdaptiveLossWeighting(4)
    if args.distiller == 'kdmcse':
        kdmcse_kdmodel = MCSE().cuda()
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.learning_rate)
    total_epochs = args.epoch
    warmup_epochs = max(1, int(0.1 * total_epochs))  # 前 10% epoch 做 warmup

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    if not num_category:
        _init_eval = evaluate_retrieval(student_model, val_loader)
        _i2t = _init_eval.get("I2T", {})
        _t2i = _init_eval.get("T2I", {})
        _mean = _init_eval.get("Mean", {})
        print(
            f"I2T R@1: {float(_i2t.get('R@1',0.0)):.4f} | R@5: {float(_i2t.get('R@5',0.0)):.4f} | R@10: {float(_i2t.get('R@10',0.0)):.4f} || "
            f"T2I R@1: {float(_t2i.get('R@1',0.0)):.4f} | R@5: {float(_t2i.get('R@5',0.0)):.4f} | R@10: {float(_t2i.get('R@10',0.0)):.4f} || "
            f"Mean R@1: {float(_mean.get('R@1',0.0)):.4f} | R@5: {float(_mean.get('R@5',0.0)):.4f} | R@10: {float(_mean.get('R@10',0.0)):.4f}"
        )
    for epoch in range(total_epochs):
        print(f"[LR] Epoch {epoch+1} start | lr={optimizer.param_groups[0]['lr']:.2e}")

        student_model.train()
        total_loss = .0
        batch_count = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):

            images = batch["image"].cuda()     # [B, C, H, W]
            text_raw = batch["text"]           # list[str]

            if hasattr(student_model, "get_tokenizer"):
                tokenizer_s = student_model.get_tokenizer()
                use_bert_tokenizer = True
            else:
                tokenizer_s = tokenize
                use_bert_tokenizer = False

            if use_bert_tokenizer:
                text_tokens_s = tokenizer_s(text_raw, padding=True, truncation=True, return_tensors="pt")
                if isinstance(text_tokens_s, dict):
                    for k in text_tokens_s:
                        text_tokens_s[k] = text_tokens_s[k].cuda()
                else:
                    try:
                        text_tokens_s = {k: v.cuda() for k, v in text_tokens_s.items()}
                    except Exception:
                        raise RuntimeError("Student tokenizer must return a dict with input_ids & attention_mask for ALBEF.")
            else:
                text_tokens_s = tokenizer_s(text_raw).cuda()

            if args.distiller in {'msd', 'dsmd', 'kdmcse', 'g2d', 'dclip', 'amd'}:
                tokenizer_t = tokenize
                text_tokens_t = tokenizer_t(text_raw).cuda()
                with torch.no_grad():
                    outputs_t1 = teacher_model_1(images, text_tokens_t)
                    outputs_t2 = teacher_model_2(images, text_tokens_t)
            else:
                outputs_t1 = outputs_t2 = None

            outputs_s = student_model(images, text_tokens_s)

            if num_category:
                labels = batch["label"].cuda()    # [B]
                loss_cls_joint = F.binary_cross_entropy_with_logits(outputs_s["joint_logits"], labels)
                loss_cls_img = F.binary_cross_entropy_with_logits(outputs_s["image_logits"], labels)
                loss_cls_txt = F.binary_cross_entropy_with_logits(outputs_s["text_logits"], labels)
            else:
                img_feat_s = F.normalize(outputs_s["image_feat"], dim=-1)
                txt_feat_s = F.normalize(outputs_s["text_feat"], dim=-1)

                if "logit_scale" in outputs_s:
                    logit_scale = outputs_s["logit_scale"].exp()
                elif hasattr(student_model, "logit_scale"):
                    logit_scale = student_model.logit_scale.exp()
                else:
                    if not hasattr(student_model, "_logit_scale"):
                        student_model._logit_scale = torch.nn.Parameter(
                            torch.ones([], device=img_feat_s.device) * math.log(1/0.07)
                        )
                    def _param_in_optimizer(opt, p):
                        pid = id(p)
                        for g in opt.param_groups:
                            for q in g["params"]:
                                if id(q) == pid:
                                    return True
                        return False
                    if hasattr(student_model, "_logit_scale"):
                        if not _param_in_optimizer(optimizer, student_model._logit_scale):
                            optimizer.add_param_group({"params": [student_model._logit_scale]})
                    logit_scale = student_model._logit_scale.exp()

                logits_per_image = logit_scale * (img_feat_s @ txt_feat_s.T)
                logits_per_text = logits_per_image.T

                with torch.no_grad():
                    if hasattr(student_model, "logit_scale"):
                        student_model.logit_scale.clamp_((0), math.log(100))
                    elif hasattr(student_model, "_logit_scale"):
                        student_model._logit_scale.clamp_(0, math.log(100))
                bsz = img_feat_s.size(0)
                labels_contrastive = torch.arange(bsz).to(img_feat_s.device)
                loss_i2t = F.cross_entropy(logits_per_image, labels_contrastive)
                loss_t2i = F.cross_entropy(logits_per_text, labels_contrastive)

            if num_category:
                if args.distiller == 'none':
                    loss_distill = .0
                elif args.distiller == 'msd':
                    loss_distill = compute_msd_loss(outputs_t1, outputs_t2, outputs_s, batch.get("label", None), "classification")
                elif args.distiller == 'dsmd':
                    loss_distill = compute_dsmd_loss(dsmd_kdmodel, dsmd_adapter, outputs_t1, outputs_t2, outputs_s)
                elif args.distiller == 'kdmcse':
                    loss_distill = compute_kdmcse_loss(outputs_t1, outputs_t2, outputs_s, kdmcse_kdmodel)
                elif args.distiller == 'g2d':
                    loss_distill = compute_g2d_loss(outputs_t1, outputs_t2, outputs_s, config=args)
                elif args.distiller == 'dclip':
                    loss_distill = compute_dclip_loss(outputs_t1, outputs_t2, outputs_s, device='cpu', config=args)
                elif args.distiller == 'amd':
                    loss_distill = compute_amd_loss(
                        outputs_t1,
                        outputs_t2,
                        outputs_s,
                        batch.get("label", None),
                        device=images.device.type,
                        config=args,
                    )
                else:
                    raise ValueError(f"Unknown distiller: {args.distiller}")

            else:
                if args.distiller == 'none':
                    loss_distill = .0
                elif args.distiller == 'msd':
                    loss_distill = compute_msd_loss(outputs_t1, outputs_t2, outputs_s, batch.get("label", None), "retrieval")
                elif args.distiller == 'dsmd':
                    loss_distill = compute_dsmd_loss(dsmd_kdmodel, dsmd_adapter, outputs_t1, outputs_t2, outputs_s)
                elif args.distiller == 'kdmcse':
                    loss_distill = compute_kdmcse_loss(outputs_t1, outputs_t2, outputs_s, kdmcse_kdmodel)
                elif args.distiller == 'g2d':
                    loss_distill = compute_g2d_loss(outputs_t1, outputs_t2, outputs_s, config=args)
                elif args.distiller == 'dclip':
                    loss_distill = compute_dclip_loss(outputs_t1, outputs_t2, outputs_s, device='cpu', config=args)
                elif args.distiller == 'amd':
                    loss_distill = compute_amd_loss(outputs_t1, outputs_t2, outputs_s, batch.get("label", None), device=images.device.type, config=args)
                else:
                    loss_distill = .0

            if num_category:
                loss = loss_cls_joint + loss_distill
            else:
                loss = loss_i2t + loss_t2i + loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / max(1, batch_count)
        print(f"[Train] Epoch {epoch} | Loss: {avg_loss:.4f}")
      
        ds_name = str(getattr(args, "dataset", "") or getattr(getattr(val_loader, "dataset", None), "name", "")).lower()

        if num_category:
            if "vqav2" in ds_name or ds_name == "vqa" or "vqa" in ds_name:
                eval_result = evaluate_classification(student_model, val_loader, dataset="vqav2")
                vqa_acc = float(eval_result.get("vqa_acc", 0.0))
                print(f"[Eval-VQAv2] Epoch {epoch} | Overall: {vqa_acc:.4f}")
                metric_for_stop = vqa_acc
            else:
                eval_result = evaluate_classification(student_model, val_loader, dataset=None, calibrated=True, return_calib=True)
                micro = float(eval_result.get("micro_f1", 0.0))
                macro = float(eval_result.get("macro_f1", 0.0))
                exacc = float(eval_result.get("example_accuracy", 0.0))
                print(f"[Eval-Classification] Epoch {epoch} | Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f} | Example-Acc: {exacc:.4f}")
                metric_for_stop = micro
        else:
            eval_result = evaluate_retrieval(student_model, val_loader)
            _i2t  = eval_result.get("I2T", {})
            _t2i  = eval_result.get("T2I", {})
            _mean = eval_result.get("Mean", {})
            print(
                f"[Eval-Retrieval] Epoch {epoch+1} | "
                f"I2T R@1: {float(_i2t.get('R@1',0.0)):.4f} | R@5: {float(_i2t.get('R@5',0.0)):.4f} | R@10: {float(_i2t.get('R@10',0.0)):.4f} || "
                f"T2I R@1: {float(_t2i.get('R@1',0.0)):.4f} | R@5: {float(_t2i.get('R@5',0.0)):.4f} | R@10: {float(_t2i.get('R@10',0.0)):.4f} || "
                f"Mean R@1: {float(_mean.get('R@1',0.0)):.4f} | R@5: {float(_mean.get('R@5',0.0)):.4f} | R@10: {float(_mean.get('R@10',0.0)):.4f}"
            )
            metric_for_stop = float(_mean.get("R@1", 0.0))

        better = metric_for_stop > best_score
        if better:
            best_score = metric_for_stop
            best_epoch = epoch
            best_eval_result = eval_result.copy()

            _ds = ("vqav2" if ("vqav2" in ds_name or ds_name == "vqa" or "vqa" in ds_name) else
                   _slug(getattr(args, "dataset", "") or getattr(getattr(val_loader, "dataset", None), "name", "") or ("retrieval" if not num_category else "classification")))

            save_dir_s = _save_root_students(args, _ds)
            save_dir_t = _save_root_teachers(args, _ds)
            os.makedirs(save_dir_s, exist_ok=True)
            os.makedirs(save_dir_t, exist_ok=True)

            best_path = os.path.join(save_dir_s, f"{s_tag}_{_ds}_{distiller_tag}_t1-{t1_tag}_t2-{t2_tag}_best.pth")
            torch.save(student_model.state_dict(), best_path)
            print(f"✅ Best student model saved to: {best_path}")

            metrics_path = os.path.join(save_dir_s, f"{s_tag}_{_ds}_{distiller_tag}_t1-{t1_tag}_t2-{t2_tag}_metrics.json")
            metrics_to_save = {
                "best_epoch": int(best_epoch),
                "student_tag": s_tag,
                "teacher_model_1_tag": t1_tag,
                "teacher_model_2_tag": t2_tag,
                "dataset": _ds,
                "distiller": distiller_tag,
            }
            if num_category and ("vqav2" in ds_name or ds_name == "vqa" or "vqa" in ds_name):
                metrics_to_save["vqa_acc"] = float(eval_result.get("vqa_acc", 0.0))
            elif num_category:
                metrics_to_save.update({
                    "micro_f1": float(eval_result.get("micro_f1", 0.0)),
                    "macro_f1": float(eval_result.get("macro_f1", 0.0)),
                    "example_accuracy": float(eval_result.get("example_accuracy", 0.0)),
                })
                if "_calib_temperature" in eval_result:
                    metrics_to_save["_calib_temperature"] = float(eval_result["_calib_temperature"])
            else:
                _i2t  = eval_result.get("I2T", {})
                _t2i  = eval_result.get("T2I", {})
                _mean = eval_result.get("Mean", {})
                metrics_to_save.update({
                    "I2T": {"R@1": float(_i2t.get("R@1", 0.0)), "R@5": float(_i2t.get("R@5", 0.0)), "R@10": float(_i2t.get("R@10", 0.0))},
                    "T2I": {"R@1": float(_t2i.get("R@1", 0.0)), "R@5": float(_t2i.get("R@5", 0.0)), "R@10": float(_t2i.get("R@10", 0.0))},
                    "Mean": {"R@1": float(_mean.get("R@1", 0.0)), "R@5": float(_mean.get("R@5", 0.0)), "R@10": float(_mean.get("R@10", 0.0))},
                })
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)
            print(f"Evaluation metrics saved to: {metrics_path}")

        scheduler.step()
        student_model.train()

    print("============================================================")
    if num_category:
        ds_name = str(getattr(args, "dataset", "") or getattr(getattr(val_loader, "dataset", None), "name", "")).lower()
        if "vqav2" in ds_name or ds_name == "vqa" or "vqa" in ds_name:
            print(f"🏁 Best performance achieved at Epoch {best_epoch}")
            print(f"Overall(VQA): {best_eval_result.get('vqa_acc', 0.0):.4f}")
        else:
            print(f"🏁 Best student performance at Epoch {best_epoch}")
            print(f"micro_f1: {best_eval_result.get('micro_f1', 0.0):.4f}")
            print(f"macro_f1: {best_eval_result.get('macro_f1', 0.0):.4f}")
            print(f"example_accuracy: {best_eval_result.get('example_accuracy', 0.0):.4f}")
            if "_calib_temperature" in best_eval_result:
                print(f"_calib_temperature: {best_eval_result['_calib_temperature']:.4f}")
    else:
        __i2t  = best_eval_result.get("I2T", {})
        __t2i  = best_eval_result.get("T2I", {})
        __mean = best_eval_result.get("Mean", {})
        print(f"🏁 Best retrieval performance at Epoch {best_epoch}")
        print(
            f"I2T R@1: {float(__i2t.get('R@1',0.0)):.4f} | R@5: {float(__i2t.get('R@5',0.0)):.4f} | R@10: {float(__i2t.get('R@10',0.0)):.4f} || "
            f"T2I R@1: {float(__t2i.get('R@1',0.0)):.4f} | R@5: {float(__t2i.get('R@5',0.0)):.4f} | R@10: {float(__t2i.get('R@10',0.0)):.4f} || "
            f"Mean R@1: {float(__mean.get('R@1',0.0)):.4f} | R@5: {float(__mean.get('R@5',0.0)):.4f} | R@10: {float(__mean.get('R@10',0.0)):.4f}"
        )
    print("============================================================")
    
    return student_model
