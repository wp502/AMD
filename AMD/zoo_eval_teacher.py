# eval_teacher.py
# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch

from evaluation import evaluate_classification, evaluate_retrieval
from load_data import load_dataset
from load_model import build_model

def log_print(msg, log_file=None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def log_retrieval(eval_result, prefix="", log_file=None):
    if isinstance(eval_result, dict) and "I2T" in eval_result:
        i2t = eval_result["I2T"]; t2i = eval_result["T2I"]; mean = eval_result["Mean"]
        log_print(
            f"{prefix}I2T - R@1: {i2t['R@1']:.4f} | R@5: {i2t['R@5']:.4f} | R@10: {i2t['R@10']:.4f}",
            log_file
        )
        log_print(
            f"{prefix}T2I - R@1: {t2i['R@1']:.4f} | R@5: {t2i['R@5']:.4f} | R@10: {t2i['R@10']:.4f}",
            log_file
        )
        log_print(
            f"{prefix}Mean - R@1: {mean['R@1']:.4f} | R@5: {mean['R@5']:.4f} | R@10: {mean['R@10']:.4f}",
            log_file
        )
        if "tta" in eval_result or "templates" in eval_result:
            log_print(f"{prefix}TTA: {eval_result.get('tta', 'n/a')} | Templates: {eval_result.get('templates', 'n/a')}", log_file)
    else:
        log_print(
            f"{prefix}I2T - R@1: {eval_result['R@1']:.4f} | R@5: {eval_result['R@5']:.4f} | R@10: {eval_result['R@10']:.4f}",
            log_file
        )

@torch.no_grad()
def run_eval(args):
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    train_loader, val_loader, test_loader, num_category = load_dataset(
        dataset=args.dataset, batch_size=args.batch_size
    )
    model = build_model(args.teacher_model, num_category, args.project_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    msd = model.state_dict()
    filtered_ckpt = {k: v for k, v in ckpt.items() if k in msd and msd[k].shape == v.shape}
    unexpected = [k for k in ckpt.keys() if k not in msd]
    missing = [k for k in msd.keys() if k not in filtered_ckpt]
    if unexpected:
        log_print(f"[load] ignore unexpected keys: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}", args.log_path)
    if missing:
        log_print(f"[load] missing keys (keep model default): {missing[:8]}{' ...' if len(missing)>8 else ''}", args.log_path)

    model.load_state_dict(filtered_ckpt, strict=False)
    model.eval()

    log_print("=" * 60, args.log_path)
    log_print(f"Evaluate teacher: {args.teacher_model} on {args.dataset}", args.log_path)
    log_print(f"Checkpoint: {args.ckpt}", args.log_path)
    log_print(f"Batch size: {args.batch_size}", args.log_path)
    if not num_category:
        log_print(f"TTA: {args.eval_tta} | Templates: {args.eval_templates if args.eval_templates else ['{}']}", args.log_path)
    log_print("=" * 60, args.log_path)

    results = {}
    def _eval_split(split_name, loader):
        if loader is None:
            return None
        if num_category:
            out = evaluate_classification(model, loader)
            log_print(
                f"[{split_name}] Micro-F1: {out['micro_f1']:.4f} | "
                f"Macro-F1: {out['macro_f1']:.4f} | "
                f"Example-Acc: {out['example_accuracy']:.4f}",
                args.log_path
            )
        else:
            try:
                out = evaluate_retrieval(
                    model, loader,
                    image_tta=args.eval_tta,
                    text_templates=args.eval_templates,
                    max_text_templates=args.max_text_templates
                )
            except TypeError:
                log_print("[warn] evaluate_retrieval 不支持 image_tta/text_templates，已回退旧接口。", args.log_path)
                out = evaluate_retrieval(model, loader)
            log_retrieval(out, prefix=f"[{split_name}] ", log_file=args.log_path)
        return out

    # 只评估用户选择的 split（或两者都评）
    if args.split in ("val", "both"):
        results["val"] = _eval_split("Val", val_loader)
    if args.split in ("test", "both"):
        results["test"] = _eval_split("Test", test_loader)

    # ====== 保存JSON ======
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log_print("-" * 60, args.log_path)
    log_print(f"Metrics saved to: {args.save_json}", args.log_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset/model
    parser.add_argument("--dataset", type=str, default="mmimdb",
                        choices=["mmimdb", "vqav2", "flickr-30k", "ms-coco"])
    parser.add_argument("--teacher_model", default="clip-ViT-B-16",
                        choices=["clip-ViT-B-16", "clip-ViT-L-14", "clip-RN101"])
    parser.add_argument("--project_dim", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=256)  
    parser.add_argument("--ckpt", type=str, required=True, help="path to .pth weights")
    parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"])

    parser.add_argument('--eval_tta', type=str, default="hflip",
                        choices=["none", "hflip", "shift", "hflip+shift"],
                        help="测试时图像 TTA 方案（检索有效）")
    parser.add_argument('--eval_templates', type=str, nargs="*",
                        default=["{}"],  # ✅ 默认仅 '{}'
                        help='文本模板集成（检索有效），使用 {} 作为占位符；只传 "{}" 表示关闭集成')
    parser.add_argument('--max_text_templates', type=int, default=4,
                        help='最多使用多少个模板，防止显存激增')

    args = parser.parse_args()
    save_dir = "raw_models/teachers/eval"
    os.makedirs(save_dir, exist_ok=True)

    model_name = args.teacher_model.replace("/", "-")
    args.log_path = os.path.join(save_dir, f"{model_name}_{args.dataset}_eval_log.txt")
    args.save_json = os.path.join(save_dir, f"{model_name}_{args.dataset}_eval_metrics.json")

   
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_eval(args)
