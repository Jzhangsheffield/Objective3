import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (softmax version).

    公式（per-sample）：
      FL = - alpha_y * (1 - p_t)^gamma * log(p_t)
    其中 p_t = softmax(logits)[range(B), y]

    参数：
      gamma: focusing 参数，常用 1~3，默认 2
      alpha: 类别权重，形状 [C] 的 Tensor（可选）
             - 若你想“focal + 类别不均衡”，通常用你 counts 得到的 class weights 作为 alpha
      reduction: 'mean' | 'sum' | 'none'
      eps: 数值稳定
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C]
        targets: [B]  int64
        """
        if logits.ndim != 2:
            raise ValueError(f"FocalLoss expects logits [B,C], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"FocalLoss expects targets [B], got {tuple(targets.shape)}")

        # log_probs: [B,C]
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # 取出每个样本的 p_t / log(p_t)
        idx = torch.arange(logits.size(0), device=logits.device)
        pt = probs[idx, targets].clamp(min=self.eps, max=1.0)           # [B]
        log_pt = log_probs[idx, targets]                                # [B]

        # alpha_y
        if self.alpha is None:
            alpha_t = 1.0
        else:
            if self.alpha.numel() != logits.size(1):
                raise ValueError(f"alpha length {self.alpha.numel()} != num_classes {logits.size(1)}")
            alpha_t = self.alpha[targets]                               # [B]

        loss = -alpha_t * ((1.0 - pt) ** self.gamma) * log_pt          # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
