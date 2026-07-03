#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prototype_directional_loss.py

============================================================
本文件实现：
1) 用当前 batch 的 q 特征，构造一个“可微分的 EMA 更新后 prototype bank 预览版”
2) 基于“方向性约束（directional loss）”计算损失：
      - 同类 prototype 变远 -> 惩罚
      - 不同类 prototype 变近 -> 惩罚
3) 在 optimizer.step() 之后，对真实 prototype_bank 执行 no_grad 的 EMA 更新
4) 支持 DDP：真实更新时会 all_reduce，保证各 rank 的 prototype_bank 一致
5) 兼容“每个类别 prototype 数量不同”的 dense-bank 表达方式

这个版本的关键点：
--------------------
prototype_bank 仍然使用固定形状 [C, M_max, D] 保存，便于广播和 checkpoint。
但是，真正有效的 prototype 个数由 class_num_prototypes[c] 指定。
因此：

    - build_differentiable_ema_bank(...)
    - differentiable_ema_directional_loss(...)
    - ema_update_prototype_bank_(...)

都支持额外传入 class_num_prototypes，
并在内部严格过滤无效 padded 槽位。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ============================================================
# 0) 基础小工具
# ============================================================

def _safe_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """对张量做 L2 normalize，并显式提供 eps，避免全 0 向量时数值不稳定。"""
    return F.normalize(x, dim=dim, eps=eps)



def _flatten_proto_bank(
    prototype_bank: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    将 prototype_bank 从 [C, M, D] 展平成 [P, D]，其中 P = C * M。

    返回：
    ------
    proto_flat:
        [P, D]，展平后的 prototype 并做了 L2 normalize。

    valid_mask:
        [P]，哪些 prototype 不是全 0（范数 > 1e-12）。
        这通常对应于“真正被初始化过”的 prototype。
    """
    assert prototype_bank.ndim == 3, f"Expect [C, M, D], got {tuple(prototype_bank.shape)}"

    C, M, D = prototype_bank.shape
    raw_flat = prototype_bank.view(C * M, D).float()
    valid_mask = raw_flat.norm(dim=1) > 1e-12
    proto_flat = _safe_normalize(raw_flat, dim=1)

    return proto_flat, valid_mask, C, M, D



def _flatten_class_proto_ids(
    labels: torch.Tensor,
    proto_ids: torch.Tensor,
    num_prototypes: int,
) -> torch.Tensor:
    """
    将 (class_id, proto_id) 映射到一个全局唯一的 flat_id。

    若：
        class_id in [0, C-1]
        proto_id in [0, M-1]

    则：
        flat_id = class_id * M + proto_id
        flat_id in [0, C*M - 1]
    """
    return labels.long() * int(num_prototypes) + proto_ids.long()



def _build_valid_sample_mask(
    labels: torch.Tensor,
    proto_ids: torch.Tensor,
    num_classes: int,
    num_prototypes_max: int,
    class_num_prototypes: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    统一构造“当前 batch 中哪些样本拥有合法 proto_id”的布尔 mask。

    这样做的原因是：
    ----------------
    在 varproto 版本中，不能只检查 proto_ids >= 0，
    还必须进一步检查：

        proto_ids < class_num_prototypes[labels]

    否则 padded 槽位可能会被错误地当成有效 prototype 来更新。

    参数：
    ------
    labels / proto_ids:
        [B]

    num_classes:
        C

    num_prototypes_max:
        M_max，即 dense bank 的第二维大小。

    class_num_prototypes:
        [C] 或 None。
        - None：退化为固定 prototype 数量版本，合法范围是 [0, M_max)
        - 非 None：按每个类别的有效 prototype 个数动态过滤
    """
    valid_label_mask = (labels >= 0) & (labels < num_classes)

    if class_num_prototypes is None:
        return valid_label_mask & (proto_ids >= 0) & (proto_ids < num_prototypes_max)

    if class_num_prototypes.ndim != 1 or class_num_prototypes.shape[0] != num_classes:
        raise ValueError(
            f"class_num_prototypes must have shape [{num_classes}], "
            f"got {tuple(class_num_prototypes.shape)}"
        )

    class_num_prototypes = class_num_prototypes.to(device=labels.device).long()
    labels_safe = labels.clone().long()
    labels_safe[~valid_label_mask] = 0
    per_anchor_k = class_num_prototypes[labels_safe]

    return valid_label_mask & (proto_ids >= 0) & (proto_ids < per_anchor_k)



def compute_prototype_distance_matrix(
    prototype_bank: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 prototype 两两之间的 cosine distance matrix。

    返回：
    ------
    dist_mat:
        [P, P]，其中 dist(i, j) = 1 - cos(proto_i, proto_j)

    valid_mask:
        [P]，哪些 prototype 有效（非全 0）
    """
    proto_flat, valid_mask, _, _, _ = _flatten_proto_bank(prototype_bank)

    sim_mat = torch.matmul(proto_flat, proto_flat.t()).clamp(-1.0, 1.0)
    dist_mat = 1.0 - sim_mat

    return dist_mat, valid_mask


# ============================================================
# 1) 构造“可微分的 EMA 更新后 prototype bank 预览版”
# ============================================================

def build_differentiable_ema_bank(
    old_prototype_bank: torch.Tensor,
    q: torch.Tensor,
    labels: torch.Tensor,
    proto_ids: torch.Tensor,
    preview_ema_momentum: float = 0.90,
    class_num_prototypes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    用当前 batch 的 q 特征，构造一个“可微分的 EMA 更新后 new bank 预览版”。

    为什么需要这个函数？
    --------------------
    真实 prototype bank 更新通常是在 no_grad 下做的，因此不能用于反传。
    为了让基于 prototype-bank 的损失能够对 q / encoder_q 回传梯度，
    这里构造一个 differentiable surrogate：

        new_proto = normalize(
            preview_ema_momentum * old_proto
            + (1 - preview_ema_momentum) * batch_mean_q
        )

    其中：
    - old_proto 使用 detach()，作为历史 / teacher 状态
    - batch_mean_q 来自当前 batch，保留梯度

    额外说明（varproto 版）：
    -------------------------
    若传入 class_num_prototypes，则只有满足

        proto_ids < class_num_prototypes[label]

    的样本才会参与 prototype preview 更新。
    """
    assert old_prototype_bank.ndim == 3
    assert q.ndim == 2
    assert labels.ndim == 1
    assert proto_ids.ndim == 1
    assert q.shape[0] == labels.shape[0] == proto_ids.shape[0]

    device = q.device
    q = _safe_normalize(q.float(), dim=1)

    old_flat, _, C, M, D = _flatten_proto_bank(old_prototype_bank)
    P = C * M

    valid_sample_mask = _build_valid_sample_mask(
        labels=labels,
        proto_ids=proto_ids,
        num_classes=C,
        num_prototypes_max=M,
        class_num_prototypes=class_num_prototypes,
    )
    if valid_sample_mask.sum().item() == 0:
        return (
            old_prototype_bank.detach().clone(),
            torch.zeros((C, M), device=device, dtype=torch.bool),
        )

    labels_v = labels[valid_sample_mask].long()
    proto_ids_v = proto_ids[valid_sample_mask].long()
    q_v = q[valid_sample_mask]

    flat_ids = _flatten_class_proto_ids(labels_v, proto_ids_v, M)

    new_flat = old_flat.detach().clone()
    updated_mask = torch.zeros((P,), device=device, dtype=torch.bool)

    unique_flat_ids = torch.unique(flat_ids)
    for pid in unique_flat_ids.tolist():
        pid = int(pid)

        mask = flat_ids == pid
        q_mean = q_v[mask].mean(dim=0, keepdim=True)
        q_mean = _safe_normalize(q_mean, dim=1).squeeze(0)

        old_p = old_flat[pid].detach()
        new_p = preview_ema_momentum * old_p + (1.0 - preview_ema_momentum) * q_mean
        new_p = _safe_normalize(new_p.unsqueeze(0), dim=1).squeeze(0)

        new_flat[pid] = new_p
        updated_mask[pid] = True

    new_bank = new_flat.view(C, M, D)
    updated_mask_2d = updated_mask.view(C, M)

    return new_bank, updated_mask_2d


# ============================================================
# 2) 旧/新 prototype 距离变化方向约束损失
# ============================================================

def prototype_directional_loss(
    old_prototype_bank: torch.Tensor,
    new_prototype_bank: torch.Tensor,
    updated_mask_2d: torch.Tensor,
    same_margin: float = 0.0,
    diff_margin: float = 0.0,
    same_weight: float = 1.0,
    diff_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算基础方向项（directional loss）。

    目标：
    ------
    1) 同类 prototype 如果变远了 -> 惩罚
    2) 不同类 prototype 如果变近了 -> 惩罚

    数学形式：
    ----------
    设：
        D_old(i,j) = old prototype 距离
        D_new(i,j) = new preview prototype 距离

    同类项：
        loss_same = mean( relu(D_new - D_old - same_margin) )

    异类项：
        loss_diff = mean( relu(D_old - D_new - diff_margin) )
    """
    device = old_prototype_bank.device
    dtype = old_prototype_bank.dtype

    old_dist, old_valid = compute_prototype_distance_matrix(old_prototype_bank)
    new_dist, new_valid = compute_prototype_distance_matrix(new_prototype_bank)

    updated_mask = updated_mask_2d.view(-1)
    valid_mask = old_valid & new_valid

    updated_ids = torch.nonzero(updated_mask, as_tuple=False).squeeze(1)
    valid_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

    if updated_ids.numel() == 0 or valid_ids.numel() <= 1:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "num_updated_proto": torch.tensor(0, device=device),
            "num_valid_proto": torch.tensor(int(valid_ids.numel()), device=device),
            "num_same_pairs": torch.tensor(0, device=device),
            "num_diff_pairs": torch.tensor(0, device=device),
            "loss_same": zero.detach(),
            "loss_diff": zero.detach(),
            "same_margin": torch.tensor(float(same_margin), device=device),
            "diff_margin": torch.tensor(float(diff_margin), device=device),
        }

    old_rows = old_dist[updated_ids][:, valid_ids]
    new_rows = new_dist[updated_ids][:, valid_ids]

    C, M, _ = old_prototype_bank.shape

    updated_cls = torch.div(updated_ids, M, rounding_mode="floor")
    valid_cls = torch.div(valid_ids, M, rounding_mode="floor")

    same_class_mask = updated_cls.unsqueeze(1).eq(valid_cls.unsqueeze(0))
    diff_class_mask = ~same_class_mask

    self_mask = updated_ids.unsqueeze(1).eq(valid_ids.unsqueeze(0))
    same_class_mask = same_class_mask & (~self_mask)

    same_delta = new_rows - old_rows
    same_bad = F.relu(same_delta - same_margin)
    same_vals = same_bad[same_class_mask]

    if same_vals.numel() > 0:
        loss_same = same_vals.mean()
        same_shift_mean = (new_rows[same_class_mask] - old_rows[same_class_mask]).mean().detach()
    else:
        loss_same = torch.zeros((), device=device, dtype=old_rows.dtype)
        same_shift_mean = torch.zeros((), device=device, dtype=old_rows.dtype)

    diff_bad = F.relu((old_rows - new_rows) - diff_margin)
    diff_vals = diff_bad[diff_class_mask]

    if diff_vals.numel() > 0:
        loss_diff = diff_vals.mean()
        diff_shift_mean = (new_rows[diff_class_mask] - old_rows[diff_class_mask]).mean().detach()
    else:
        loss_diff = torch.zeros((), device=device, dtype=old_rows.dtype)
        diff_shift_mean = torch.zeros((), device=device, dtype=old_rows.dtype)

    loss_dir = same_weight * loss_same + diff_weight * loss_diff

    stats = {
        "num_updated_proto": torch.tensor(int(updated_ids.numel()), device=device),
        "num_valid_proto": torch.tensor(int(valid_ids.numel()), device=device),
        "num_same_pairs": torch.tensor(int(same_vals.numel()), device=device),
        "num_diff_pairs": torch.tensor(int(diff_vals.numel()), device=device),
        "loss_same": loss_same.detach(),
        "loss_diff": loss_diff.detach(),
        "same_shift_mean": same_shift_mean,
        "diff_shift_mean": diff_shift_mean,
        "same_margin": torch.tensor(float(same_margin), device=device),
        "diff_margin": torch.tensor(float(diff_margin), device=device),
        "same_weight": torch.tensor(float(same_weight), device=device),
        "diff_weight": torch.tensor(float(diff_weight), device=device),
    }
    return loss_dir, stats



def differentiable_ema_directional_loss(
    old_prototype_bank: torch.Tensor,
    q: torch.Tensor,
    labels: torch.Tensor,
    proto_ids: torch.Tensor,
    preview_ema_momentum: float = 0.90,
    same_margin: float = 0.0,
    diff_margin: float = 0.0,
    same_weight: float = 1.0,
    diff_weight: float = 1.0,
    class_num_prototypes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    一步完成两件事：

    1) 用当前 batch q + old bank，构造“可微分的 EMA 更新后 new bank preview”
    2) 基于 old/new bank 计算基础方向项损失

    若传入 class_num_prototypes，则 preview 更新只会使用合法的类内 prototype id。
    """
    new_bank_preview, updated_mask_2d = build_differentiable_ema_bank(
        old_prototype_bank=old_prototype_bank,
        q=q,
        labels=labels,
        proto_ids=proto_ids,
        preview_ema_momentum=preview_ema_momentum,
        class_num_prototypes=class_num_prototypes,
    )

    loss_dir, stats = prototype_directional_loss(
        old_prototype_bank=old_prototype_bank,
        new_prototype_bank=new_bank_preview,
        updated_mask_2d=updated_mask_2d,
        same_margin=same_margin,
        diff_margin=diff_margin,
        same_weight=same_weight,
        diff_weight=diff_weight,
    )

    return loss_dir, new_bank_preview, updated_mask_2d, stats


# ============================================================
# 3) 真正的 prototype EMA 更新（no_grad, DDP-safe）
# ============================================================

@torch.no_grad()
def ema_update_prototype_bank_(
    prototype_bank: torch.Tensor,
    q: torch.Tensor,
    labels: torch.Tensor,
    proto_ids: torch.Tensor,
    bank_ema_momentum: float = 0.99,
    class_num_prototypes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    对真实保存的 prototype_bank 做原地 EMA 更新。

    这个函数应该在 optimizer.step() 之后调用。

    DDP 下如何保证各 rank 一致？
    ----------------------------
    1) 每个 rank 先统计：
        - 每个 prototype 对应的 q 的和（sums）
        - 每个 prototype 对应的样本数（counts）
    2) 对 sums / counts 执行 all_reduce
    3) 每个 rank 根据全局 sums / counts 得到相同的 batch_mean
    4) 每个 rank 按同样规则更新本地 prototype_bank

    额外说明（varproto 版）：
    -------------------------
    若传入 class_num_prototypes，则只更新真正有效的 prototype 槽位。
    padded 槽位永远不会被错误写入。
    """
    assert prototype_bank.ndim == 3
    assert q.ndim == 2
    assert labels.ndim == 1
    assert proto_ids.ndim == 1
    assert q.shape[0] == labels.shape[0] == proto_ids.shape[0]

    device = prototype_bank.device
    q = _safe_normalize(q.detach().float(), dim=1)

    _, _, C, M, D = _flatten_proto_bank(prototype_bank)
    P = C * M

    valid_sample_mask = _build_valid_sample_mask(
        labels=labels,
        proto_ids=proto_ids,
        num_classes=C,
        num_prototypes_max=M,
        class_num_prototypes=class_num_prototypes,
    )
    if valid_sample_mask.sum().item() == 0:
        return torch.zeros((C, M), device=device, dtype=torch.bool)

    labels_v = labels[valid_sample_mask].long()
    proto_ids_v = proto_ids[valid_sample_mask].long()
    q_v = q[valid_sample_mask]

    flat_ids = _flatten_class_proto_ids(labels_v, proto_ids_v, M)

    sums = torch.zeros((P, D), device=device, dtype=torch.float32)
    counts = torch.zeros((P, 1), device=device, dtype=torch.float32)

    unique_flat_ids = torch.unique(flat_ids)
    for pid in unique_flat_ids.tolist():
        pid = int(pid)
        mask = flat_ids == pid
        sums[pid] = q_v[mask].sum(dim=0)
        counts[pid, 0] = float(mask.sum().item())

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    updated_mask = counts.squeeze(1) > 0
    if updated_mask.sum().item() == 0:
        return torch.zeros((C, M), device=device, dtype=torch.bool)

    batch_means = sums[updated_mask] / counts[updated_mask].clamp_min(1.0)
    batch_means = _safe_normalize(batch_means, dim=1)

    raw_flat = prototype_bank.view(P, D)
    old_rows = _safe_normalize(raw_flat[updated_mask].float(), dim=1)
    new_rows = bank_ema_momentum * old_rows + (1.0 - bank_ema_momentum) * batch_means
    new_rows = _safe_normalize(new_rows, dim=1)

    raw_flat[updated_mask] = new_rows.to(raw_flat.dtype)

    return updated_mask.view(C, M)


# ============================================================
# 4) （可选）一个便于日志打印的小工具
# ============================================================

def directional_stats_to_str(stats: Dict[str, torch.Tensor]) -> str:
    """
    将 directional loss 的 stats 字典格式化成一行可读字符串，
    便于训练日志打印。
    """
    def _v(key: str, default: float = 0.0) -> float:
        val = stats.get(key, None)
        if val is None:
            return default
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return float(val.detach().cpu().item())
            return default
        return float(val)

    return (
        f"updated_proto={_v('num_updated_proto'):.0f}, "
        f"valid_proto={_v('num_valid_proto'):.0f}, "
        f"same_pairs={_v('num_same_pairs'):.0f}, "
        f"diff_pairs={_v('num_diff_pairs'):.0f}, "
        f"loss_same={_v('loss_same'):.6f}, "
        f"loss_diff={_v('loss_diff'):.6f}, "
        f"same_shift_mean={_v('same_shift_mean'):.6f}, "
        f"diff_shift_mean={_v('diff_shift_mean'):.6f}"
    )
