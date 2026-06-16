import torch
import torch.nn.functional as F

def cosine_distillation_loss(student_embeddings, teacher_embeddings):
    """
    Cosine similarity loss: 1 - cos(student, teacher)
    """
    student_norm = F.normalize(student_embeddings, dim=1)
    teacher_norm = F.normalize(teacher_embeddings, dim=1)
    cos_sim = torch.sum(student_norm * teacher_norm, dim=1)  # [B]
    return torch.mean(1.0 - cos_sim)

def compute_contrastive_loss(image_embeddings, text_embeddings, device=None, temperature=0.05):
    """
    InfoNCE contrastive loss between image and text embeddings
    """
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    batch_size = image_embeddings.size(0)
    labels = torch.arange(batch_size, device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0

def compute_dclip_loss(outputs_t1, outputs_t2, outputs_s, device=None, config=None):
    """
    DCLIP distillation loss:
    - Each multimodal teacher independently supervises student (no averaging)
    - Cosine distillation: teacher_image -> student_image, teacher_text -> student_text
    - Contrastive loss: student_image <-> student_text
    """

    # === 提取学生特征 ===
    s_img = outputs_s["image_feat"]      # [B, D]
    s_txt = outputs_s["text_feat"]       # [B, D]

    # === 提取两个教师的模态特征 ===
    t1_img = outputs_t1["image_feat"]
    t1_txt = outputs_t1["text_feat"]
    t2_img = outputs_t2["image_feat"]
    t2_txt = outputs_t2["text_feat"]

    # === 教师1蒸馏 ===
    loss_img_t1 = cosine_distillation_loss(s_img, t1_img)
    loss_txt_t1 = cosine_distillation_loss(s_txt, t1_txt)

    # === 教师2蒸馏 ===
    loss_img_t2 = cosine_distillation_loss(s_img, t2_img)
    loss_txt_t2 = cosine_distillation_loss(s_txt, t2_txt)

    # === 图文对比损失（学生自身对齐） ===
    contrastive_loss = compute_contrastive_loss(s_img, s_txt, device=device,
                                                temperature=getattr(config, "temperature", 0.05))

    # === 权重配置 ===
    w_cdist = getattr(config, "cosine_loss_weight", 1.0)
    w_contrast = getattr(config, "contrastive_loss_weight", 1.0)

    # === 总蒸馏损失 ===
    total_loss = w_cdist * (loss_img_t1 + loss_txt_t1 + loss_img_t2 + loss_txt_t2) \
                 + w_contrast * contrastive_loss

    return total_loss