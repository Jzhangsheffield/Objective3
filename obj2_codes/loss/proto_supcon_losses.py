# ref PCL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np


class SupLoss(nn.Module):
    def __init__(self, temperature=1.0, base_temperature=None, K=128):
        super(SupLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K

    def forward(self, features, labels=None): # features: [B + B + K, D] labels: [B + B + K]
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0] # ss: [B + B+ K]
        batch_size = (features.shape[0] - self.K) // 2 # batch_size = (B+ B+ K - K) // 2 = B

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device) # mask: [B, B + B + K]

        # TODO: label is -1
        # mask[torch.where(labels < 0)[0]] = 0

        # compute logits
        anchor_dot_contrast = torch.div(  
            torch.matmul(features[:batch_size], features.T),
            self.temperature)  # anchor_dot_contrast: [B, B + B + K]

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device), # 拼接时 q 在前所以只需要对前 batch_size 个进行屏蔽.
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12) # 其实就是 log(exp(i) / (exp(0) + exp(1) + exp(n))), 等于 log(exp(i)) - log((exp(0) + exp(1) + exp(n)) 而 log(exp(i)) == i

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    
class PrototypeRelaLoss(nn.Module):
    """
    Prototype 相对距离损失：
    - 每个 rank 只用自己 batch 的 features/labels 计算 class mean
    - 构造 prototype_new_local（带梯度）来算相对距离损失
    - 用 no_grad 把 prototype memory 更新成新的（local 版本）

    改进：
    - 只对“本 batch 出现的类别集合 S”内部的类别对计算 penalty
    """
    
    def __init__(self, num_classes: int=10, feat_dim: int=512,  momentum: float = 0.9):
        super().__init__()
        self.momentum = float(momentum)
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.register_buffer("prototypes", torch.randn(num_classes, feat_dim))
        
    
    def _compute_class_means(self, features: torch.Tensor, labels: torch.Tensor):
        B, D = features.shape
        C = self.num_classes
        device = features.device
        dtype = features.dtype

        sums = torch.zeros(C, D, device=device, dtype=dtype)
        counts = torch.zeros(C, 1, device=device, dtype=dtype)

        sums.index_add_(0, labels, features) # 这是对 sums 的原地操作: i 从 0 到 C-1, 取 c = labels[i], sums[c] += features[i] (第 i 行的 feature 要加到 sums 的哪一行）
        ones = torch.ones(B, 1, device=device, dtype=dtype)
        counts.index_add_(0, labels, ones) # 对 counts 原地操作： i 从 0 到 C-1, 取 c = labels[i], counts[c] += ones[i]

        class_means = sums / counts.clamp(min=1.0)
        return class_means, counts
    
    
    @staticmethod
    def _pairwise_dist(x: torch.Tensor):
        return torch.cdist(x, x, p=2)
    
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        features: [B, D]  = encoder_q avgpool features (q_avgpool_features)
        labels:   [B]     = ground-truth class labels for this batch (NOT queue labels)

        return: scalar loss
        """
        # 用于 防止使用AMP 可能出现的问题
        features = features.float()
        
        assert features.dim() == 2 and features.size(1) == self.feat_dim, \
            f"features should be [B, {self.feat_dim}]"
        assert labels.dim() == 1, "labels should be [B]"
        # labels 必须是 long 用于 index_add_
        labels = labels.long()

        # 1) 旧 prototype（常量基准，不带梯度）
        with torch.no_grad():
            P_old = self.prototypes.clone()  # [C, D]

        # 2) 本 batch 的 class mean（带梯度）
        class_means, counts = self._compute_class_means(features, labels)
        present = (counts.squeeze(1) > 0)  # [C] bool

        # 3) 构造 P_new_local（带梯度）：只更新本 batch 出现的类别
        P_new = P_old.clone()  # 这个张量会与 class_means 建图（因为下面会赋值带梯度的 updated）

        if present.any():
            old_present = P_old[present]              # [C_present, D] 常量
            mean_present = class_means[present]       # [C_present, D] 可反传
            updated_present = (1.0 - self.momentum) * old_present + self.momentum * mean_present
            P_new[present] = updated_present          # 使 P_new 对 features 有梯度

        # 4) 相对距离损失：只惩罚“变近”
        D_old = self._pairwise_dist(P_old)  # [C, C] 常量
        D_new = self._pairwise_dist(P_new)  # [C, C] 对 features 有梯度

        delta = D_old - D_new  # >0 表示变近
        # 去掉对角线
        diag = torch.diag(delta.diag())
        delta = delta - diag

        # 只惩罚变近
        penalty = F.relu(delta)

        C = self.num_classes
        loss = penalty.sum() / (C * (C - 1) + 1e-6)

        # 5) 方案 A：用 no_grad 做 local prototype memory 更新（与 loss 的梯度无关）
        with torch.no_grad():
            self.prototypes.copy_(P_new.detach())

        return loss

        
        
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

    
    
        

        
    
    