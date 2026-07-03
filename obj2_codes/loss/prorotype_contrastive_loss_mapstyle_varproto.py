#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn.functional as F


def prototype_contrastive_loss_single_positive(
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

    这个版本适配“每个类别的 prototype 数量不同”的场景。

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
    2) 但在 loss 内部，必须显式屏蔽掉 padded 槽位，避免它们进入分母。
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
    valid_label_mask = (labels >= 0) & (labels < C)
    labels_safe = labels.clone().long()
    labels_safe[~valid_label_mask] = 0

    per_anchor_k = class_num_prototypes[labels_safe]
    valid_anchor_mask = valid_label_mask & (proto_ids >= 0) & (proto_ids < per_anchor_k)

    if valid_anchor_mask.sum().item() == 0:
        return q.new_zeros(())

    q_valid = q[valid_anchor_mask]
    labels_valid = labels[valid_anchor_mask].long()
    proto_ids_valid = proto_ids[valid_anchor_mask].long()

    # ------------------------------------------------------------
    # 2) 将 prototype bank 展平为 [C*M_max, D]
    # ------------------------------------------------------------
    proto_flat = prototype_bank.view(C * M_max, D)
    pos_flat_idx = labels_valid * M_max + proto_ids_valid

    sim = torch.matmul(q_valid, proto_flat.t())

    # ------------------------------------------------------------
    # 3) 构造全局有效 prototype mask
    # ------------------------------------------------------------
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
    # 5) 构造 mask：
    #    - 保留所有“异类且有效”的 prototype 作为负样本
    #    - 屏蔽掉同类其他 prototype
    #    - 再把正 prototype 单独放回来
    # ------------------------------------------------------------
    proto_class = torch.arange(C, device=device).unsqueeze(1).expand(C, M_max).reshape(-1)
    same_class_mask = proto_class.unsqueeze(0).eq(labels_valid.unsqueeze(1))
    active_mask = active_proto_mask.unsqueeze(0).expand(q_valid.size(0), -1)

    valid_mask = active_mask & (~same_class_mask)
    valid_mask.scatter_(1, pos_flat_idx.unsqueeze(1), True)

    logits = logits.masked_fill(~valid_mask, float("-inf"))

    loss = F.cross_entropy(logits, pos_flat_idx)
    return loss


from typing import Optional

import torch
import torch.nn.functional as F


def prototype_contrastive_loss_all_positive(
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
    计算“样本 -> prototype”的对比损失（log 外求和版本）。

    这个版本适配“每个类别的 prototype 数量不同”的场景，并且将：
        同类所有有效 prototype
    都视为正样本。

    损失形式：
    ----------
    对于第 i 个样本，记：
        P_i = 该样本同类的所有有效 prototype 集合
        A_i = 所有有效 prototype 集合
        z_{ij} = 样本 i 与 prototype j 的 logit

    则本函数计算：
        L_i = -(1 / |P_i|) * sum_{p in P_i} log( exp(z_{ip}) / sum_{a in A_i} exp(z_{ia}) )

    也就是：
        - 对“每个正 prototype”分别计算一项 log-softmax loss
        - 然后对这些正项做平均

    与 “-log(sum exp(pos) / sum exp(all))” 不同，
    这里的求和是在 log 外面，因此它会更明确地约束样本接近每一个同类 prototype。

    输入张量：
    ---------
    q:
        [B, D]，当前 batch 的 query 特征，通常应已归一化。

    labels:
        [B]，样本类别标签。

    proto_ids:
        [B]，样本在其所属类别内部被分配到的 prototype id。
        本版本中：
            - 它仍用于过滤“哪些样本可参与 loss”
            - 但 loss 的正样本不再只用这个 assigned prototype，
              而是使用“同类所有有效 prototype”

        如果你后续希望“不依赖 proto_ids 也能参与 loss”，
        可以单独调整 valid_anchor_mask 的定义。

    prototype_bank:
        [C, M_max, D]。
        C 为类别数，M_max 为所有类别中最大的 prototype 数量。

    class_num_prototypes:
        [C]，每个类别真实有效的 prototype 数量。
        对于类别 c，只有 prototype_bank[c, :class_num_prototypes[c]] 是有效的。

    temperature:
        全局温度系数，必须 > 0。

    use_prototype_temperature_scaling:
        是否对不同 prototype 使用相对温度缩放。

    proto_rel_temperature_bank:
        [C, M_max]，每个 prototype 的相对温度倍率。
        仅当 use_prototype_temperature_scaling=True 时使用。

    temperature_eps:
        温度下界，防止除零或过小数值。

    返回：
    -----
    一个标量 loss（0-dim Tensor）。
    如果当前 batch 中没有合法 anchor，则返回 0。
    """
    # ------------------------------------------------------------
    # 0) 基本输入检查
    # ------------------------------------------------------------
    if q.ndim != 2:
        raise ValueError(f"q must be [B, D], got shape={tuple(q.shape)}")

    if labels.ndim != 1 or proto_ids.ndim != 1:
        raise ValueError(
            f"labels/proto_ids must be 1D, got labels={tuple(labels.shape)}, "
            f"proto_ids={tuple(proto_ids.shape)}"
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
    labels = labels.to(device=device).long()
    proto_ids = proto_ids.to(device=device).long()
    class_num_prototypes = class_num_prototypes.to(device=device).long()
    prototype_bank = prototype_bank.to(device=device)

    # ------------------------------------------------------------
    # 1) 过滤无效 anchor
    #
    # 保留你原本的逻辑：
    #   只有 label 合法，且 proto_ids 在该类有效 prototype 范围内，
    #   才认为这个样本是有效 anchor。
    # ------------------------------------------------------------
    valid_label_mask = (labels >= 0) & (labels < C)

    labels_safe = labels.clone()
    labels_safe[~valid_label_mask] = 0

    per_anchor_k = class_num_prototypes[labels_safe]  # [B]
    valid_anchor_mask = valid_label_mask & (proto_ids >= 0) & (proto_ids < per_anchor_k)

    if valid_anchor_mask.sum().item() == 0:
        return q.new_zeros(())

    q_valid = q[valid_anchor_mask]                     # [Bv, D]
    labels_valid = labels[valid_anchor_mask]          # [Bv]
    # 注意：proto_ids_valid 在这个版本里不再参与“正样本定义”，
    # 但保留 valid_anchor_mask 过滤逻辑时，前面仍需要 proto_ids。
    # 这里不再单独取 proto_ids_valid 也可以。

    Bv = q_valid.shape[0]

    # ------------------------------------------------------------
    # 2) 将 prototype bank 展平为 [C*M_max, D]
    # ------------------------------------------------------------
    proto_flat = prototype_bank.view(C * M_max, D)    # [C*M_max, D]

    # 相似度矩阵：每个样本对所有 prototype 的相似度
    sim = torch.matmul(q_valid, proto_flat.t())       # [Bv, C*M_max]

    # ------------------------------------------------------------
    # 3) 构造全局有效 prototype mask
    #
    # active_proto_mask_2d[c, m] = True 表示该槽位是有效 prototype
    # ------------------------------------------------------------
    proto_slot_ids = torch.arange(M_max, device=device).unsqueeze(0).expand(C, M_max)   # [C, M_max]
    active_proto_mask_2d = proto_slot_ids < class_num_prototypes.unsqueeze(1)            # [C, M_max]
    active_proto_mask = active_proto_mask_2d.reshape(C * M_max)                          # [C*M_max]

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

        rel_temp_flat = proto_rel_temperature_bank.to(
            device=device, dtype=sim.dtype
        ).view(C * M_max)

        rel_temp_flat = rel_temp_flat.clamp_min(temperature_eps)
        tau_eff = temperature * rel_temp_flat                     # [C*M_max]

        logits = sim / tau_eff.unsqueeze(0)                      # [Bv, C*M_max]
    else:
        logits = sim / temperature

    # ------------------------------------------------------------
    # 5) 构造正样本 mask 和有效分母 mask
    #
    # positive_mask:
    #   当前样本“同类且有效”的所有 prototype，全部作为正样本
    #
    # valid_mask:
    #   所有有效 prototype 都进入 softmax 分母
    # ------------------------------------------------------------
    proto_class = (
        torch.arange(C, device=device)
        .unsqueeze(1)
        .expand(C, M_max)
        .reshape(-1)
    )  # [C*M_max]

    same_class_mask = proto_class.unsqueeze(0).eq(labels_valid.unsqueeze(1))   # [Bv, C*M_max]
    active_mask = active_proto_mask.unsqueeze(0).expand(Bv, -1)                # [Bv, C*M_max]

    positive_mask = same_class_mask & active_mask                              # [Bv, C*M_max]
    valid_mask = active_mask                                                   # [Bv, C*M_max]

    # 理论上对于有效 anchor，其 positive_mask 至少会有一个 True；
    # 这里加一个保护，避免极端情况下出现非法样本。
    positive_count = positive_mask.sum(dim=1)                                  # [Bv]
    safe_row_mask = positive_count > 0

    if safe_row_mask.sum().item() == 0:
        return q.new_zeros(())

    logits = logits[safe_row_mask]
    positive_mask = positive_mask[safe_row_mask]
    positive_count = positive_count[safe_row_mask]

    # 无效 prototype 不进入 softmax 分母
    logits = logits.masked_fill(~valid_mask[safe_row_mask], float("-inf"))

    # ------------------------------------------------------------
    # 6) log 外求和版本
    #
    # log_prob[b, j] = log softmax over all valid prototypes
    #
    # 然后：
    #   loss_b = - (1 / num_pos_b) * sum_{j in positives} log_prob[b, j]
    #
    # 注意不能直接写：
    #   (log_prob * positive_mask.float()).sum(...)
    # 因为 -inf * 0 可能产生 nan。
    # 所以使用 masked_fill 把非正位置安全置 0。
    # ------------------------------------------------------------
    log_prob = F.log_softmax(logits, dim=1)                                    # [Bv_safe, C*M_max]

    positive_log_prob_sum = log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)   # [Bv_safe]
    loss_per_sample = - positive_log_prob_sum / positive_count.to(log_prob.dtype)   # [Bv_safe]

    loss = loss_per_sample.mean()
    return loss