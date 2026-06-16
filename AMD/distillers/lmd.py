# distill_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_lmd_loss(teacher_outputs_1, teacher_outputs_2, student_outputs, labels=None, task='retrieval', alpha=0.5, T=1.0):

    loss_fn_kl = nn.KLDivLoss(reduction='batchmean')

    def kl_logits_loss(student_logit, teacher_logit):
        return loss_fn_kl(
            F.log_softmax(student_logit / T, dim=-1),
            F.softmax(teacher_logit.detach() / T, dim=-1)
        )

    if task == 'classification':
        # 对两个教师分别计算 KL loss
        losses = []
        for teacher in [teacher_outputs_1, teacher_outputs_2]:
            loss_joint = kl_logits_loss(student_outputs["joint_logits"], teacher["joint_logits"])
            loss_img = kl_logits_loss(student_outputs["image_logits"], teacher["image_logits"])
            loss_txt = kl_logits_loss(student_outputs["text_logits"], teacher["text_logits"])
            losses.append(0.5 * loss_joint + 0.25 * loss_img + 0.25 * loss_txt)
        distill_loss = sum(losses) / len(losses) 

    elif task == 'retrieval':
        losses = []
        for teacher in [teacher_outputs_1, teacher_outputs_2]:
            student_img = F.normalize(student_outputs["image_feat"], dim=-1)
            student_txt = F.normalize(student_outputs["text_feat"], dim=-1)
            teacher_img = F.normalize(teacher["image_feat"], dim=-1)
            teacher_txt = F.normalize(teacher["text_feat"], dim=-1)

            loss_feat_img = F.mse_loss(student_img, teacher_img)
            loss_feat_txt = F.mse_loss(student_txt, teacher_txt)

            student_logits = student_img @ student_txt.T
            teacher_logits = teacher_img @ teacher_txt.T

            loss_logits = loss_fn_kl(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits.detach() / T, dim=-1)
            )

            losses.append(0.25 * loss_feat_img + 0.25 * loss_feat_txt + 0.5 * loss_logits)
        distill_loss = sum(losses) / len(losses)  

    else:
        raise ValueError(f"Unknown task type: {task}")

    return distill_loss
