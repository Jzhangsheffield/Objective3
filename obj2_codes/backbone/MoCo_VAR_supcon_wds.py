# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist


class MoCo3D(nn.Module):
    """
    MoCo-style 3D encoder + queue, adapted for *supervised contrastive* learning with WebDataset.

    与你旧版的区别（关键）：
    - 不再依赖 (index -> label_labeled_queue) 的预初始化流程。
    - 直接从 forward(...) 接收 batch 的 labels，并把 labels 同步入队列：
        queue:        [D, K]
        queue_labels: [K]
    - 因此训练脚本无需再跑一遍 init_model_loader 来填充 label_labeled_queue。

    Forward 输出：
    - features: [B + B + K, D]    (q, k, queue)
    - target:   [B + B + K]      (labels_q, labels_k, labels_queue)
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, enable_kcl_loss: bool=False, num_positive=0, exclude_invalid_queue=True):
        super().__init__()
        self.K = int(K)
        self.m = float(m)
        self.T = float(T)
        self.enable_kcl_loss = enable_kcl_loss
        
        # KCL 相关
        # num_positive = 0  -> 使用 queue 中“所有同类样本”作为 positives
        # num_positive > 0  -> 从 queue 的同类样本中随机采样最多 num_positive 个 positives
        if enable_kcl_loss:
            self.num_positive = int(num_positive)
            print(f"kcl loss num_positive: {self.num_positive}")
            
            # 是否把 queue 中 label = -1 的未初始化位置从分母里排除
            # 我建议 True，比原始 TSC 代码更稳
            self.exclude_invalid_queue = bool(exclude_invalid_queue)
            print(f"kcl loss enable exclude_invalid_queue: {self.exclude_invalid_queue}")

        # encoders
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            print(f"using projection head. dim={dim}")
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim),
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim),
            )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # queue: [D, K]
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # queue labels: [K]，-1 表示未知（例如刚初始化时）
        self.register_buffer("queue_labels", torch.full((self.K,), -1, dtype=torch.long))

        # queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, labels: torch.Tensor):
        """
        入队 keys 和对应 labels（都需要 all_gather 后再写入队列，以保证 DDP 下全局一致）。
        keys:   [B, D]
        labels: [B]
        """
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 旧版要求 K % batch_size == 0；这里为了稳妥保留该假设（否则会跨界写入两段）
        assert self.K % batch_size == 0, f"K({self.K}) must be divisible by batch_size({batch_size})"

        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_labels[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: torch.Tensor):
        """
        Batch shuffle：为了让 BN 看到来自不同 GPU 的混合样本（MoCo 经典做法）。
        当 world_size==1 时，直接返回原 x。
        """
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
            idx_unshuffle = torch.arange(x.shape[0], device=x.device)
            return x, idx_unshuffle

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all, device=x.device)
        dist.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: torch.Tensor, idx_unshuffle: torch.Tensor):
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
            return x

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    
    def compute_kcl_loss(self, q: torch.Tensor, k: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Non-targeted KCL loss.

        q:      [B, D]  normalized query features
        k:      [B, D]  normalized key features
        labels: [B]     class labels

        正样本：
        1) 当前样本对应的 key view（即 k_i）
        2) queue 中与 labels[i] 相同的样本（可选：最多采样 self.num_positive 个）

        负样本：
        - queue 中标签不同的样本
        """
        device = q.device
        B = q.shape[0]

        labels = labels.contiguous().view(-1, 1)  # [B,1]

        # 当前 memory queue（不反传梯度）
        queue = self.queue.clone().detach()                      # [D, K]
        queue_labels = self.queue_labels.clone().detach().view(1, -1)  # [1, K]

        # -------------------------------------------------------
        # 1) logits
        #    l_pos: q_i · k_i
        #    l_neg: q_i · queue_j
        # -------------------------------------------------------
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(1)   # [B,1] [B, D] 和 [B, D] 对应的行逐元素相乘并相加.
        l_neg = torch.einsum("nc,ck->nk", [q, queue])           # [B,K]

        logits = torch.cat([l_pos, l_neg], dim=1)               # [B,1+K]
        logits = logits / self.T                                # [B,1+K]

        # 数值稳定
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # -------------------------------------------------------
        # 2) queue 内同类样本 mask
        # -------------------------------------------------------
        pos_mask_queue = torch.eq(labels, queue_labels).to(q.dtype)  # [B,K]

        # 如果只采样固定数量 positives，就在每个 anchor 上随机采样
        if self.num_positive > 0:
            sampled_pos_mask_queue = torch.zeros_like(pos_mask_queue)
            for i in range(B):
                pos_idx = torch.nonzero(pos_mask_queue[i] > 0, as_tuple=False).squeeze(1) # 第 i 行, 正样本的索引
                if pos_idx.numel() == 0:
                    continue

                if pos_idx.numel() > self.num_positive:
                    perm = torch.randperm(pos_idx.numel(), device=device)[:self.num_positive] # 如果正样本数大于设定的所需正样本数, 则将索引打乱, 并取前num_positive 个.
                    pos_idx = pos_idx[perm]

                sampled_pos_mask_queue[i, pos_idx] = 1.0
        else:
            sampled_pos_mask_queue = pos_mask_queue

        # 当前 view positive（k_i）永远是正样本
        pos_mask = torch.cat(
            [
                torch.ones((B, 1), device=device, dtype=q.dtype),
                sampled_pos_mask_queue,
            ],
            dim=1,
        )  # [B,1+K]

        # -------------------------------------------------------
        # 3) valid mask（可选：屏蔽 queue 中 label=-1 的未初始化位置）
        # -------------------------------------------------------
        if self.exclude_invalid_queue:
            valid_queue_mask = (queue_labels != -1).to(q.dtype).expand(B, -1)  # [B,K]
        else:
            valid_queue_mask = torch.ones((B, self.K), device=device, dtype=q.dtype)

        valid_mask = torch.cat(
            [
                torch.ones((B, 1), device=device, dtype=q.dtype),
                valid_queue_mask,
            ],
            dim=1,
        )  # [B,1+K]

        # -------------------------------------------------------
        # 4) log_prob
        #    只在 valid_mask 上做 softmax 分母
        # -------------------------------------------------------
        exp_logits = torch.exp(logits) * valid_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # -------------------------------------------------------
        # 5) multi-positive average
        # -------------------------------------------------------
        num_pos = pos_mask.sum(dim=1).clamp_min(1.0)  # 至少有当前 view positive
        loss = -((pos_mask * log_prob).sum(dim=1) / num_pos).mean()

        return loss
        

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor, labels: torch.Tensor):
        """
        Input:
            im_q:   [B, ...]
            im_k:   [B, ...]
            labels: [B]
            compute_kcl_loss:
                - True : 在模型内部计算并返回 KCL 损失
                - False: 不计算 KCL，返回 loss_kcl=None

        Output:
            features: [B + B + K, D]   (保留，方便后续 SupLoss / prototype / 可视化)
            target:   [B + B + K]      (保留)
            loss_kcl: scalar or None
            q:        [B, D]
            k:        [B, D]
        """
        labels = labels.long().contiguous().view(-1)
        
        # query encoder
        q = self.encoder_q(im_q)                 # [B, D]
        q = nn.functional.normalize(q, dim=1)

        # key encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # 计算 kcl 损失
        loss_kcl = None
        if self.enable_kcl_loss:
            loss_kcl = self.compute_kcl_loss(q=q, k=k, labels=labels)
        
        # 拼接：q, k, queue
        features = torch.cat((q, k, self.queue.clone().detach().T), dim=0)  # [B+B+K, D]

        # 拼接 label：labels_q, labels_k, labels_queue
        target = torch.cat((labels, labels, self.queue_labels.clone().detach()), dim=0)  # [B+B+K]

        # enqueue
        self._dequeue_and_enqueue(k, labels)

        return features, target, loss_kcl, q, k


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """DDP all_gather；world_size==1 时直接返回。"""
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)
