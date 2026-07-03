#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prorotype_contrastive_loss_wds_varproto.py

本文件实现“样本 -> prototype”的对比损失，并支持：
1) 每个类别拥有不同数量的 prototype；
2) dense prototype_bank 采用固定形状 [C, M_max, D]；
3) 通过 class_num_prototypes 显式屏蔽 padded 槽位；
4) 对无效 anchor 自动跳过，而不是直接索引越界或把 -1 送入损失。
"""

from typing import Optional

import torch
import torch.nn.functional as F



def prototype_contrastive_loss(
    q: torch.Tensor,
    labels: torch.Tensor,
    proto_ids: torch.Tensor,
    prototype_bank: torch.Tensor,
    class_num_prototypes: torch.Tensor,
    temperature: float = 0.07,
    use_prototype_temperature_scaling: bool = False,
    proto_rel_temperature_bank: Optional[torch.Tensor] = None,
    temperature_eps: float = 1e-6,
) -> torch.Tensor:
    """
    计算“样本 -> prototype”的对比损失。

    输入张量：
    ---------
    q:
        [B, D]，当前 batch 的 query 特征，默认应已归一化。

    labels:
        [B]，样本类别标签。

    proto_ids:
        [B]，样本在其所属类别内部被分配到的 prototype id。
        若某个样本当前没有合法 prototype，则应设为 -1。

    prototype_bank:
        [C, M_max, D]。
        这里的 M_max 是所有类别中最大的 prototype 数量。

    class_num_prototypes:
        [C]，记录每个类别真正有效的 prototype 数量。
        对于类别 c，只有 bank[c, :class_num_prototypes[c]] 是有效槽位。

    核心设计：
    ----------
    1) 对外仍然使用一个固定形状的 dense bank [C, M_max, D]，便于广播和 checkpoint。
    2) 但在 loss 内部，必须显式屏蔽掉 padded 槽位，避免它们进入 softmax 分母。
    3) 对 anchor 的合法性过滤，也必须根据 class_num_prototypes[label] 动态判断。

    正样本定义：
    ----------
    对于第 b 个样本，其正 prototype 为：
        class = labels[b]
        proto = proto_ids[b]

    负样本定义：
    ----------
    所有“异类且有效”的 prototype。

    同类别中其他 prototype 会被从分母屏蔽掉。
    这样做的好处是：不会把“同类但不是当前分配原型”的 prototype 当成硬负样本。
    """
    if q.ndim != 2:
        raise ValueError(f"q must be [B, D], got shape={tuple(q.shape)}")
    if labels.ndim != 1 or proto_ids.ndim != 1:
        raise ValueError(
            f"labels/proto_ids must be 1D, got labels={tuple(labels.shape)}, proto_ids={tuple(proto_ids.shape)}"
        )
    if prototype_bank.ndim != 3:
        raise ValueError(
            f"prototype_bank must be [C, M_max, D], got shape={tuple(prototype_bank.shape)}"
        )
    if class_num_prototypes.ndim != 1:
        raise ValueError(
            f"class_num_prototypes must be [C], got shape={tuple(class_num_prototypes.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    B, D = q.shape
    C, M_max, D_bank = prototype_bank.shape
    if D != D_bank:
        raise ValueError(
            f"Feature dim mismatch: q has D={D}, prototype_bank has D={D_bank}"
        )
    if labels.shape[0] != B or proto_ids.shape[0] != B:
        raise ValueError(
            f"Batch size mismatch: q={B}, labels={labels.shape[0]}, proto_ids={proto_ids.shape[0]}"
        )
    if class_num_prototypes.shape[0] != C:
        raise ValueError(
            f"class_num_prototypes length mismatch: expect {C}, got {class_num_prototypes.shape[0]}"
        )

    device = q.device
    class_num_prototypes = class_num_prototypes.to(device=device).long()

    # ------------------------------------------------------------
    # 1) 过滤无效 anchor
    # ------------------------------------------------------------
    # 有效 anchor 需要同时满足：
    # - label 合法
    # - proto_id >= 0
    # - proto_id < 该类别当前真正有效的 prototype 数量
    valid_label_mask = (labels >= 0) & (labels < C)
    labels_safe = labels.clone().long()
    labels_safe[~valid_label_mask] = 0

    per_anchor_k = class_num_prototypes[labels_safe]
    valid_anchor_mask = valid_label_mask & (proto_ids >= 0) & (proto_ids < per_anchor_k)

    if valid_anchor_mask.sum().item() == 0:
        # 若当前 batch 没有合法 anchor，则返回 0，避免 loss/索引报错。
        return q.new_zeros(())

    q_valid = q[valid_anchor_mask]
    labels_valid = labels[valid_anchor_mask].long()
    proto_ids_valid = proto_ids[valid_anchor_mask].long()

    # ------------------------------------------------------------
    # 2) 将 prototype bank 展平为 [C*M_max, D]
    # ------------------------------------------------------------
    # 虽然每个类别的 prototype 数不同，但为了便于广播和 checkpoint，
    # bank 仍然保存为 dense 形式 [C, M_max, D]。
    # 因此这里仍然按照固定 stride=M_max 展平。
    proto_flat = prototype_bank.view(C * M_max, D)
    pos_flat_idx = labels_valid * M_max + proto_ids_valid

    sim = torch.matmul(q_valid, proto_flat.t())

    # ------------------------------------------------------------
    # 3) 构造全局“有效 prototype 槽位” mask
    # ------------------------------------------------------------
    # active_proto_mask_2d[c, m] = True 表示第 c 类第 m 个 prototype 槽位是真实有效的；
    # False 表示 padded 槽位，不应参与 softmax 竞争。
    proto_slot_ids = torch.arange(M_max, device=device).unsqueeze(0).expand(C, M_max)
    active_proto_mask_2d = proto_slot_ids < class_num_prototypes.unsqueeze(1)
    active_proto_mask = active_proto_mask_2d.reshape(C * M_max)

    # ------------------------------------------------------------
    # 4) 温度缩放
    # ------------------------------------------------------------
    if use_prototype_temperature_scaling:
        if proto_rel_temperature_bank is None:
            raise ValueError(
                "use_prototype_temperature_scaling=True, but proto_rel_temperature_bank is None"
            )
        if proto_rel_temperature_bank.shape != (C, M_max):
            raise ValueError(
                f"proto_rel_temperature_bank shape mismatch: expect {(C, M_max)}, "
                f"got {tuple(proto_rel_temperature_bank.shape)}"
            )

        rel_temp_flat = proto_rel_temperature_bank.to(device=device, dtype=sim.dtype).view(C * M_max)
        rel_temp_flat = rel_temp_flat.clamp_min(temperature_eps)
        tau_eff = temperature * rel_temp_flat
        logits = sim / tau_eff.unsqueeze(0)
    else:
        logits = sim / temperature

    # ------------------------------------------------------------
    # 5) 构造 mask
    # ------------------------------------------------------------
    # 我们保留：
    # - 所有“异类且有效”的 prototype 作为负样本
    # - 当前样本自己的正 prototype
    # 我们屏蔽：
    # - 同类中其他 prototype
    # - 所有 padded prototype 槽位
    proto_class = torch.arange(C, device=device).unsqueeze(1).expand(C, M_max).reshape(-1)
    same_class_mask = proto_class.unsqueeze(0).eq(labels_valid.unsqueeze(1))
    active_mask = active_proto_mask.unsqueeze(0).expand(q_valid.size(0), -1)

    valid_mask = active_mask & (~same_class_mask)
    valid_mask.scatter_(1, pos_flat_idx.unsqueeze(1), True)

    logits = logits.masked_fill(~valid_mask, float("-inf"))
    loss = F.cross_entropy(logits, pos_flat_idx)
    return loss
