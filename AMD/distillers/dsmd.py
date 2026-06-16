import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Triplet（Hard Negative）损失
# -----------------------------
class TriptLoss(nn.Module):
    def __init__(self, margin: float = 0.05, max_violation: bool = True):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # im: (N, C), s: (N, C)
        scores = get_sim(im, s)  # (N, N) 点积相似度
        N = scores.size(0)
        device = scores.device

        diagonal = scores.diag().view(N, 1)  # (N, 1)
        d1 = diagonal.expand_as(scores)      # (N, N)
        d2 = diagonal.t().expand_as(scores)  # (N, N)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # 去掉对角项（自身配对）
        mask = torch.eye(N, dtype=torch.bool, device=device)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        if self.max_violation:
            cost_s = cost_s.max(dim=1)[0]   # 每行最大违规
            cost_im = cost_im.max(dim=0)[0] # 每列最大违规

        return cost_s.sum() + cost_im.sum()


def get_sim(images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
    # (N, C) x (C, N) -> (N, N)
    return images.mm(captions.t())


# -----------------------------
# KDModel（包含对比/队列/多损失）
# -----------------------------
class KDModel(nn.Module):
    def __init__(self, args=None, K: int = 8192, T: float = 0.05):
        """
        args: 可选，不再依赖 args.student_model 来推断维度。
        K: 队列长度
        T: 温度
        """
        super().__init__()
        self.T = T
        self.K = K

        # 先注册“空” buffer；实际维度在第一次使用时根据学生特征懒初始化
        self.register_buffer("i_queue", torch.empty(0, 0))     # (C_img(stu), K)
        self.register_buffer("t_queue", torch.empty(0, 0))     # (C_txt(stu), K)
        self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))

        # 损失
        self.contrastive_loss = self.ContrastiveLoss(self, t=self.T)
        self.tript_loss = TriptLoss()
        self.l1 = nn.L1Loss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.t1_img_adapt = None
        self.t1_txt_adapt = None
        self.t2_img_adapt = None
        self.t2_txt_adapt = None

    # 供外部调用（与原接口一致）
    def forward(self, outputs_s):
        return outputs_s["image_feat"], outputs_s["text_feat"]

    # -------- 懒初始化队列到真实维度 --------
    def _maybe_init_queues(self, img_dim: int, txt_dim: int, device: torch.device, dtype: torch.dtype):
        # 若尚未初始化（空张量），则基于当前 batch 的学生特征维度初始化
        if self.i_queue.numel() == 0:
            i_q = torch.randn(img_dim, self.K, device=device, dtype=dtype)
            i_q = F.normalize(i_q, dim=0)
            self.i_queue = i_q  # 仍作为 buffer
        if self.t_queue.numel() == 0:
            t_q = torch.randn(txt_dim, self.K, device=device, dtype=dtype)
            t_q = F.normalize(t_q, dim=0)
            self.t_queue = t_q
     # 懒初始化“教师→学生”线性适配器
    def _maybe_init_adapters(self, stu_img_dim, stu_txt_dim,
                             t1_img_dim, t1_txt_dim,
                            t2_img_dim, t2_txt_dim, device):
        if self.t1_img_adapt is None:
            self.t1_img_adapt = nn.Linear(t1_img_dim, stu_img_dim, bias=False).to(device)
            self.t1_txt_adapt = nn.Linear(t1_txt_dim, stu_txt_dim, bias=False).to(device)
        if self.t2_img_adapt is None:
            self.t2_img_adapt = nn.Linear(t2_img_dim, stu_img_dim, bias=False).to(device)
            self.t2_txt_adapt = nn.Linear(t2_txt_dim, stu_txt_dim, bias=False).to(device)

    # 供外部复用：把第 k 位教师特征适配到学生空间
    def adapt_teacher_feats(self, tea_img, tea_txt, teacher_id: int):
        if teacher_id == 1:
            return self.t1_img_adapt(tea_img), self.t1_txt_adapt(tea_txt)
        else:
            return self.t2_img_adapt(tea_img), self.t2_txt_adapt(tea_txt)        

    # -------- 对比损失（包含队列负样本）--------
    class ContrastiveLoss(nn.Module):
        def __init__(self, model, t: float = 0.05):
            super().__init__()
            self.T = t
            self.model = model

        def forward(
            self,
            stu_img: torch.Tensor,  # (N, C_img)
            stu_txt: torch.Tensor,  # (N, C_txt)
            tea_img: torch.Tensor,  # (N, C_img)
            tea_txt: torch.Tensor,  # (N, C_txt)
        ) -> torch.Tensor:

            # 归一化，稳定尺度
            stu_img = F.normalize(stu_img, dim=1)
            tea_img = F.normalize(tea_img, dim=1)
            stu_txt = F.normalize(stu_txt, dim=1)
            tea_txt = F.normalize(tea_txt, dim=1)

            # === 关键修复：用队列的克隆副本参与前向，避免后续原地更新引发 version 冲突 ===
            i_queue_snapshot = self.model.i_queue.clone()  # or .clone().detach()
            t_queue_snapshot = self.model.t_queue.clone()

            # 正样本相似度（逐样本）
            img_pos = torch.einsum('nc,nc->n', stu_img, tea_img).unsqueeze(-1)  # (N,1)
            txt_pos = torch.einsum('nc,nc->n', stu_txt, tea_txt).unsqueeze(-1)  # (N,1)

            # 负样本：与队列进行点积
            img_neg = torch.einsum('nc,ck->nk', stu_img, i_queue_snapshot)  # (N,K)
            txt_neg = torch.einsum('nc,ck->nk', stu_txt, t_queue_snapshot)  # (N,K)

            # 拼 logits：Nx(1+K)，并按温度缩放
            img_logits = torch.cat([img_pos, img_neg], dim=1) / self.T
            txt_logits = torch.cat([txt_pos, txt_neg], dim=1) / self.T

            device = img_logits.device
            labels = torch.zeros(img_logits.size(0), dtype=torch.long, device=device)
            ce = nn.CrossEntropyLoss()
            img_loss = ce(img_logits, labels)
            txt_loss = ce(txt_logits, labels)
            return img_loss + txt_loss

    # -------- 队列入队（环形写入，兼容任意 batch 大小）--------
    @torch.no_grad()
    def _dequeue_and_enqueue(self, i_keys: torch.Tensor, t_keys: torch.Tensor):
        """
        i_keys: (N, C_img), t_keys: (N, C_txt) —— 建议外部调用前已做 normalize + detach
        """
        # 若队列还未初始化（极端情况），根据 keys 的维度临时初始化
        self._maybe_init_queues(img_dim=i_keys.size(1), txt_dim=t_keys.size(1), device=i_keys.device, dtype=i_keys.dtype)

        N = i_keys.shape[0]

        # --- 写 i_queue ---
        ptr_i = int(self.i_queue_ptr)
        end = ptr_i + N
        if end <= self.K:
            self.i_queue[:, ptr_i:end] = i_keys.t()
        else:
            first = self.K - ptr_i
            self.i_queue[:, ptr_i:] = i_keys[:first].t()
            self.i_queue[:, :end - self.K] = i_keys[first:].t()
        self.i_queue_ptr[0] = end % self.K

        # --- 写 t_queue ---
        ptr_t = int(self.t_queue_ptr)
        end = ptr_t + N
        if end <= self.K:
            self.t_queue[:, ptr_t:end] = t_keys.t()
        else:
            first = self.K - ptr_t
            self.t_queue[:, ptr_t:] = t_keys[:first].t()
            self.t_queue[:, :end - self.K] = t_keys[first:].t()
        self.t_queue_ptr[0] = end % self.K

    # -------- 汇总四种损失 --------
    def forward_loss(
        self,
        stu_img: torch.Tensor,
        stu_txt: torch.Tensor,
        tea_img: torch.Tensor,
        tea_txt: torch.Tensor,
        teacher_id: int = 1,
    ):
        
        # 懒初始化队列 + 适配器
        self._maybe_init_queues(stu_img.size(1), stu_txt.size(1), stu_img.device, stu_img.dtype)
        self._maybe_init_adapters(stu_img.size(1), stu_txt.size(1),
                                  tea_img.size(1), tea_txt.size(1),
                                  tea_img.size(1), tea_txt.size(1),
                                  stu_img.device)

        # 教师→学生空间对齐
        tea_img_s, tea_txt_s = self.adapt_teacher_feats(tea_img, tea_txt, teacher_id)
        # 1) 对比损失（含队列负样本）
        con_loss = self.contrastive_loss(stu_img, stu_txt, tea_img_s, tea_txt_s)

        # 2) L1
        l1_img_loss = self.l1(F.normalize(stu_img,1), F.normalize(tea_img_s,1))
        l1_txt_loss = self.l1(F.normalize(stu_txt,1), F.normalize(tea_txt_s,1))
        l1_loss = l1_img_loss + l1_txt_loss

        # 3) Cosine（转成 [0,1] 相似度后取 1-mean）
        cos_img_loss = 1 - ((self.cos(F.normalize(stu_img,1), F.normalize(tea_img_s,1)).mean() + 1) / 2)
        cos_txt_loss = 1 - ((self.cos(F.normalize(stu_txt,1), F.normalize(tea_txt_s,1)).mean() + 1) / 2)
        cos_loss = cos_img_loss + cos_txt_loss

        # 4) Hard negative Triplet
        s_img_n, s_txt_n = F.normalize(stu_img,1), F.normalize(stu_txt,1)
        t_img_n, t_txt_n = F.normalize(tea_img_s,1), F.normalize(tea_txt_s,1)
        tript_loss = ( self.tript_loss(s_img_n, t_img_n)
                     + self.tript_loss(s_txt_n, t_txt_n)
                     + self.tript_loss(s_img_n, t_txt_n)
                     + self.tript_loss(s_txt_n, t_img_n) )

        return con_loss, l1_loss, cos_loss, tript_loss


# -----------------------------
# 自适应加权（保留原逻辑，修正设备对齐）
# -----------------------------
class AdaptiveLossWeighting(nn.Module):
    def __init__(self, num_losses=4, temperature=1.0, k=10.0, clamp_min=0.25, clamp_max=4.0):
        super().__init__()
        self.num_losses = num_losses
        self.prev_losses = None
        self.register_buffer("weights_buf", torch.ones(num_losses))
        self.t = temperature
        self.k = k
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        current_losses: shape (num_losses,)
        论文式：量纲归一 -> 学习速率(上一时刻/当前) -> softmax 归一（温度）
        """
        device = current_losses.device
        with torch.no_grad():
            cur = current_losses.detach()
            # (15) 量纲归一：除以均值，避免量纲差异
            cur = cur / (cur.mean() + 1e-8)

            if self.prev_losses is None:
                self.prev_losses = cur
                return self.weights_buf.to(device)

            prev = self.prev_losses
            # (16) 学习速率：prev / cur，下降更快 -> 比值更大 -> 更高权重
            lrates = prev / (cur + 1e-8)
            # (17) softmax（带温度/缩放系数）
            logits = self.k * (lrates / max(self.t, 1e-6))
            w = torch.softmax(logits, dim=0) * self.num_losses
            w = torch.clamp(w, self.clamp_min, self.clamp_max)

            self.prev_losses = cur
            return w.to(device)


# -----------------------------
# 组合两位教师的 DSMD 损失
# -----------------------------
def compute_dsmd_loss(
    dsmd_kdmodel: KDModel,
    dsmd_adapter: AdaptiveLossWeighting,
    outputs_t1: dict,
    outputs_t2: dict,
    outputs_s: dict,
):
    """
    outputs_*["image_feat"], ["text_feat"] : (N, C)
    dsmd_adapter: 自适应加权模块（保持梯度）
    """
    # 与教师 1
    con1, l11, cos1, hnd1 = dsmd_kdmodel.forward_loss(
        outputs_s["image_feat"], outputs_s["text_feat"],
        outputs_t1["image_feat"], outputs_t1["text_feat"], teacher_id=1
    )
    current1 = torch.stack([l11, cos1, con1, hnd1])
    weights1 = dsmd_adapter(current1)

    # 与教师 2
    con2, l12, cos2, hnd2 = dsmd_kdmodel.forward_loss(
        outputs_s["image_feat"], outputs_s["text_feat"],
        outputs_t2["image_feat"], outputs_t2["text_feat"], teacher_id=2
    )
    current2 = torch.stack([l12, cos2, con2, hnd2])
    weights2 = dsmd_adapter(current2)

    # 加权求和（保持在同设备）
    loss1 = l11 * weights1[0] + cos1 * weights1[1] + con1 * weights1[2] + hnd1 * weights1[3]
    loss2 = l12 * weights2[0] + cos2 * weights2[1] + con2 * weights2[2] + hnd2 * weights2[3]

    # 入队：把“对齐到学生空间的教师特征”写入队列（论文设定）
    with torch.no_grad():
        # 先把两位教师各自对齐到学生空间
        t1_img_s, t1_txt_s = dsmd_kdmodel.adapt_teacher_feats(outputs_t1["image_feat"], outputs_t1["text_feat"], teacher_id=1)
        t2_img_s, t2_txt_s = dsmd_kdmodel.adapt_teacher_feats(outputs_t2["image_feat"], outputs_t2["text_feat"], teacher_id=2)
        # 可选：两位教师取平均作为单次入队键（最小改动，保持步幅）
        tea_img_s = 0.5 * (t1_img_s + t2_img_s)
        tea_txt_s = 0.5 * (t1_txt_s + t2_txt_s)
        dsmd_kdmodel._dequeue_and_enqueue(
            F.normalize(tea_img_s.detach(), dim=1),
            F.normalize(tea_txt_s.detach(), dim=1),
        )

    return (loss1 + loss2) / 2
