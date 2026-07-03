#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .mapstype_dataloader_with_index_mindrove_modified_varlen import (
    PackedMultiModalConfig,
    load_label_map_json,
    build_packed_mapstyle_dataset,
    build_packed_mapstyle_loader_from_dataset,
)


@dataclass
class PrototypeRefreshConfig:
    """
    控制 prototype 刷新过程的配置。
    """

    tier_mode: str = "tier1"
    num_prototypes_per_class: Optional[List[int]] = None
    default_num_prototypes: int = 3

    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    verify_paths_on_init: bool = False

    device: Optional[torch.device] = None
    require_main_process_only: bool = True

    enable_prototype_temperature_scaling: bool = False
    proto_base_temperature: float = 0.07
    proto_temperature_eps: float = 1e-6

    # ---------------- MindRove variable-length / packing ----------------
    # refresh 阶段必须与训练阶段使用相同的 EMG / IMU 重采样长度。
    # 若 EMG / IMU 长度不同，进入单分支 ResNet1D 前会按训练脚本相同策略
    # 重采样到统一长度后再在通道维拼接。
    mindrove_emg_target_len: Optional[int] = None
    mindrove_imu_target_len: Optional[int] = None
    mindrove_pack_length_policy: str = "max"
    mindrove_pack_target_len: Optional[int] = None

    # ---------------- MindRove normalization ----------------
    # refresh 阶段若希望与训练输入分布一致，
    # 就需要在“重采样后、进入 encoder_k 前”使用与训练阶段完全相同的标准化配置。
    mindrove_apply_normalization: bool = False

    mindrove_left_emg_mean: Optional[tuple] = None
    mindrove_left_emg_std: Optional[tuple] = None
    mindrove_right_emg_mean: Optional[tuple] = None
    mindrove_right_emg_std: Optional[tuple] = None

    mindrove_left_imu_mean: Optional[tuple] = None
    mindrove_left_imu_std: Optional[tuple] = None
    mindrove_right_imu_mean: Optional[tuple] = None
    mindrove_right_imu_std: Optional[tuple] = None


# ============================================================
# 基础工具
# ============================================================

def is_main_process() -> bool:
    """判断当前进程是否为主进程。单进程模式下始终返回 True。"""
    if (not dist.is_available()) or (not dist.is_initialized()):
        return True
    return dist.get_rank() == 0



def _resolve_label_map_path(args) -> Path:
    """
    解析 label_map.json 的绝对路径。

    支持两种写法：
    1) 直接传绝对路径
    2) 传相对于 dataset_root 的相对路径
    """
    path = Path(args.label_map_json)
    if path.is_absolute():
        return path
    return Path(args.dataset_root) / path



def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """若模型被 DDP 包裹，则返回 model.module；否则原样返回。"""
    return model.module if hasattr(model, "module") else model



def build_mindrove_expected_keys(args) -> List[str]:
    """
    根据当前 hand / signal / merge_hands 配置，确定训练与 refresh 共用的
    MindRove dict 拼接顺序。

    规则：
    1) 若 mindrove_merge_hands=True：
       dataloader 会输出按 signal 合并后的 key，例如 ["emg"]、["imu"] 或 ["emg","imu"]

    2) 若 mindrove_merge_hands=False：
       dataloader 会保留 hand+signal 级别的 key，例如：
           ["left_emg", "left_imu", "right_emg", "right_imu"]

    这个顺序必须在训练脚本和 prototype refresh 脚本中保持一致，
    否则同一个样本在 refresh 与训练阶段的输入通道语义会不一致。
    """
    hands = [str(x).lower() for x in args.mindrove_hands]
    signals = [str(x).lower() for x in args.mindrove_signals]

    if args.mindrove_merge_hands:
        return list(signals)

    return [f"{h}_{s}" for h in hands for s in signals]



def _infer_signal_from_mindrove_key(key: str) -> Optional[str]:
    """从 MindRove key 中推断 signal 类型：emg / imu。"""
    key = str(key).lower()
    if key == "emg" or key.endswith("_emg"):
        return "emg"
    if key == "imu" or key.endswith("_imu"):
        return "imu"
    return None


def _resample_bcl_if_needed(x: torch.Tensor, target_len: int, key: str) -> torch.Tensor:
    """将 [B,C,L] 按需线性重采样到 target_len。"""
    if x.ndim != 3:
        raise ValueError(f"MindRove view['{key}'] must be [B,C,L], got shape={tuple(x.shape)}")
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if int(x.shape[2]) == target_len:
        return x.contiguous()
    return F.interpolate(
        x.float(),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).to(dtype=x.dtype).contiguous()


def _resolve_pack_target_len(
    view: Dict[str, torch.Tensor],
    expected_keys: List[str],
    args,
) -> int:
    """
    为单分支 ResNet1D 解析最终拼接长度。

    dataloader varlen 可以输出 EMG / IMU 不同长度，但当前 MoCo1D backbone 仍然只接收
    单个 [B,C,L] Tensor。因此在通道维拼接前必须把所有 key 对齐到同一个 L。
    """
    lengths = [int(view[k].shape[2]) for k in expected_keys]
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) == 1:
        return unique_lengths[0]

    policy = str(getattr(args, "mindrove_pack_length_policy", "max")).lower()
    fixed_target = getattr(args, "mindrove_pack_target_len", None)

    if fixed_target is not None:
        return int(fixed_target)
    if policy == "fixed":
        raise ValueError("mindrove_pack_length_policy='fixed' requires --mindrove_pack_target_len")
    if policy == "error":
        raise ValueError(
            "MindRove sequence lengths differ across keys: "
            f"{dict((k, int(view[k].shape[2])) for k in expected_keys)}. "
            "Set --mindrove_pack_length_policy max/min/emg/imu/fixed or use one signal only."
        )
    if policy == "max":
        return max(unique_lengths)
    if policy == "min":
        return min(unique_lengths)
    if policy == "emg":
        if getattr(args, "mindrove_emg_target_len", None) is not None:
            return int(args.mindrove_emg_target_len)
        for k in expected_keys:
            if _infer_signal_from_mindrove_key(k) == "emg":
                return int(view[k].shape[2])
        raise ValueError("mindrove_pack_length_policy='emg' but no EMG key is present")
    if policy == "imu":
        if getattr(args, "mindrove_imu_target_len", None) is not None:
            return int(args.mindrove_imu_target_len)
        for k in expected_keys:
            if _infer_signal_from_mindrove_key(k) == "imu":
                return int(view[k].shape[2])
        raise ValueError("mindrove_pack_length_policy='imu' but no IMU key is present")

    raise ValueError(
        f"Unsupported mindrove_pack_length_policy={policy!r}; "
        "choose from: max, min, emg, imu, fixed, error"
    )


def pack_mindrove_batched_view(
    view: Any,
    expected_keys: List[str],
    args=None,
) -> torch.Tensor:
    """
    将一个 MindRove 视图整理成单个 Tensor[B,C,L]。

    支持：
    1) Tensor[B,C,L]：直接返回。
    2) dict[str, Tensor[B,C,L]]：按 expected_keys 固定顺序在通道维拼接。

    当 EMG / IMU 长度不同而 args 不为 None 时，会先按
    --mindrove_pack_length_policy / --mindrove_pack_target_len 把每个 key 重采样到统一 L，
    再执行 torch.cat(dim=1)。这保证训练阶段与 prototype refresh 阶段输入语义一致。
    """
    if torch.is_tensor(view):
        if view.ndim != 3:
            raise ValueError(f"Expected Tensor[B,C,L], got shape={tuple(view.shape)}")
        return view.contiguous()

    if not isinstance(view, dict):
        raise TypeError(f"Expected dict or Tensor, got {type(view)}")

    missing = [k for k in expected_keys if k not in view]
    extra = [k for k in view.keys() if k not in expected_keys]

    if missing:
        raise KeyError(f"Missing MindRove keys: {missing}; available={sorted(view.keys())}")
    if extra:
        raise KeyError(f"Unexpected MindRove keys: {extra}; expected={expected_keys}")

    pieces = []
    batch_size = None
    target_len = _resolve_pack_target_len(view, expected_keys, args) if args is not None else None
    seq_len = None

    for k in expected_keys:
        x = view[k]
        if not torch.is_tensor(x):
            raise TypeError(f"MindRove view['{k}'] must be Tensor, got {type(x)}")
        if x.ndim != 3:
            raise ValueError(f"MindRove view['{k}'] must be [B,C,L], got shape={tuple(x.shape)}")

        if batch_size is None:
            batch_size = int(x.shape[0])
        elif int(x.shape[0]) != batch_size:
            raise ValueError(f"Batch size mismatch for key '{k}', shape={tuple(x.shape)}")

        if target_len is not None:
            x = _resample_bcl_if_needed(x, target_len=target_len, key=k)
        else:
            if seq_len is None:
                seq_len = int(x.shape[2])
            elif int(x.shape[2]) != seq_len:
                raise ValueError(f"Sequence length mismatch for key '{k}', shape={tuple(x.shape)}")

        pieces.append(x)

    if len(pieces) == 1:
        return pieces[0].contiguous()

    return torch.cat(pieces, dim=1).contiguous()


def _extract_single_view_and_labels_and_indices(
    batch: Dict[str, Any],
    tier_mode: str,
    expected_keys: List[str],
    args=None,
):
    """
    从 prototype refresh 阶段的 batch 中提取：
        - 单个 MindRove view
        - labels
        - global indices

    预期输入：
        batch["mindrove"] -> Dict[str, Tensor[B,C,L]]
        或直接是 Tensor[B,C,L]

    返回：
    ------
    x:
        形状 [B,C,L] 的单视图 MindRove 输入
    labels:
        当前 tier 的标签，形状 [B]
    indices:
        样本在整个 map-style 数据集中的 global index，形状 [B]
    """
    mr = batch["mindrove"]
    x = pack_mindrove_batched_view(mr, expected_keys, args=args)

    tier_ids = batch["tier_ids"]
    labels = tier_ids[tier_mode] if isinstance(tier_ids, dict) else tier_ids

    if "global_index" in batch:
        indices = batch["global_index"]
    elif "idx" in batch:
        indices = batch["idx"]
    elif "sample_id" in batch:
        indices = batch["sample_id"]
    else:
        raise KeyError(
            "Batch does not contain a usable global id field. "
            "Expected one of: global_index / idx / sample_id"
        )

    return x, labels, indices



# ============================================================
# 构建 refresh 阶段专用 loader
# ============================================================

def build_feature_extraction_loader(
    args,
    cfg: PrototypeRefreshConfig,
) -> torch.utils.data.DataLoader:
    """
    构建 refresh 阶段专用的 map-style DataLoader。

    与训练阶段 loader 的关键区别：
    --------------------------------
    1) rgb_two_views=False
       refresh 只需要单个 clip，不需要 MoCo 的双视角输出。

    2) is_train=False
       refresh 不使用随机增强，但仍可使用与训练一致的 MindRove normalization。

    3) shuffle=False, drop_last=False
       refresh 必须扫完整个训练集，并保持 global_index 对齐。
    """
    label_map_path = _resolve_label_map_path(args)
    label_map = load_label_map_json(label_map_path)

    cfg_loader = PackedMultiModalConfig(
        rgb_two_views=False,
        use_modalities=("mindrove",),
        missing_policy="skip",
        load_labels=True,
        tier_mode=cfg.tier_mode,
        is_train=False,
        label_map_path=str(label_map_path),

        # refresh 必须是单视图
        mindrove_two_views=False,
        mindrove_target_len=args.mindrove_target_len,
        mindrove_emg_target_len=getattr(args, "mindrove_emg_target_len", None),
        mindrove_imu_target_len=getattr(args, "mindrove_imu_target_len", None),
        mindrove_hands=tuple(args.mindrove_hands),
        mindrove_signals=tuple(args.mindrove_signals),
        mindrove_merge_hands=args.mindrove_merge_hands,

        # refresh 不做随机增强
        mindrove_apply_augmentation=False,

        # refresh 与训练保持一致的 normalization
        mindrove_apply_normalization=cfg.mindrove_apply_normalization,
        mindrove_left_emg_mean=cfg.mindrove_left_emg_mean,
        mindrove_left_emg_std=cfg.mindrove_left_emg_std,
        mindrove_right_emg_mean=cfg.mindrove_right_emg_mean,
        mindrove_right_emg_std=cfg.mindrove_right_emg_std,
        mindrove_left_imu_mean=cfg.mindrove_left_imu_mean,
        mindrove_left_imu_std=cfg.mindrove_left_imu_std,
        mindrove_right_imu_mean=cfg.mindrove_right_imu_mean,
        mindrove_right_imu_std=cfg.mindrove_right_imu_std,
    )

    dataset = build_packed_mapstyle_dataset(
        dataset_root=args.dataset_root,
        manifest_name=args.train_manifest_name,
        cfg=cfg_loader,
        label_map=label_map,
        verify_paths_on_init=cfg.verify_paths_on_init,
    )

    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        sampler=None,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
    )
    return loader


# ============================================================
# 全量提特征
# ============================================================

@torch.no_grad()
def extract_features_momentum(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tier_mode: str,
    args,
    main_process_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    使用 MoCo 的 momentum encoder（encoder_k）对整个 refresh loader 提取特征。

    当前版本针对 MindRove + ResNet1D 路径设计：
    - loader 输出单视图 MindRove
    - 输入在进入 encoder_k 前统一整理为 [B,C,L]
    - 不再进行 RGB/3D 路径中的 [B,T,3,H,W] -> [B,3,T,H,W] 维度置换

    返回：
    ------
    {
        "indices": LongTensor[N],
        "labels":  LongTensor[N],
        "feats":   FloatTensor[N, D],
    }

    说明：
    ------
    1) 这里提取的是 momentum encoder，即 encoder_k 的输出。
    2) 特征会在返回前做 L2 normalize。
    3) 当前实现默认推荐只在主进程执行。
    4) expected_keys 用于保证多路 MindRove dict 的拼接顺序
       与训练阶段保持一致。
    """
    if main_process_only and (not is_main_process()):
        return {}

    model_u = _unwrap_model(model)
    if not hasattr(model_u, "encoder_k"):
        raise AttributeError("Model does not have encoder_k; cannot refresh prototypes")

    expected_keys = build_mindrove_expected_keys(args)

    was_training = model_u.training
    model_u.eval()

    all_indices: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_feats: List[torch.Tensor] = []

    for batch in loader:
        x, labels, indices = _extract_single_view_and_labels_and_indices(
            batch=batch,
            tier_mode=tier_mode,
            expected_keys=expected_keys,
            args=args,
        )

        x = x.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        indices = indices.to(device, non_blocking=True).long()

        # MindRove refresh 路径约定输入为 [B,C,L]
        if x.ndim != 3:
            raise ValueError(f"MindRove refresh input must be [B,C,L], got shape={tuple(x.shape)}")

        x = x.contiguous()

        feats = model_u.encoder_k(x)
        feats = F.normalize(feats.float(), dim=1)

        all_indices.append(indices.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_feats.append(feats.detach().cpu())

    if was_training:
        model_u.train()

    if len(all_feats) == 0:
        raise RuntimeError("No features were extracted during prototype refresh")

    indices = torch.cat(all_indices, dim=0)
    labels = torch.cat(all_labels, dim=0)
    feats = torch.cat(all_feats, dim=0)

    return {
        "indices": indices,
        "labels": labels,
        "feats": feats,
    }


# ============================================================
# 按类别分组
# ============================================================

def group_features_by_class(
    indices: torch.Tensor,
    labels: torch.Tensor,
    feats: torch.Tensor,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    将全量样本按 class_id 分组，便于后续“每类单独做 KMeans”。

    返回格式：
    ----------
    grouped[class_id] = {
        "indices": LongTensor[n_c],
        "labels":  LongTensor[n_c],
        "feats":   FloatTensor[n_c, D],
    }
    """
    if indices.ndim != 1 or labels.ndim != 1 or feats.ndim != 2:
        raise ValueError(
            f"Expect indices=[N], labels=[N], feats=[N,D], got shapes "
            f"{tuple(indices.shape)}, {tuple(labels.shape)}, {tuple(feats.shape)}"
        )
    if not (indices.shape[0] == labels.shape[0] == feats.shape[0]):
        raise ValueError("indices, labels, feats must have the same first dimension")

    grouped: Dict[int, Dict[str, torch.Tensor]] = {}
    unique_labels = torch.unique(labels)

    for cls in unique_labels.tolist():
        mask = labels == cls
        grouped[int(cls)] = {
            "indices": indices[mask].clone(),
            "labels": labels[mask].clone(),
            "feats": feats[mask].clone(),
        }

    return grouped


# ============================================================
# KMeans 与 prototype 温度估计
# ============================================================

def _pad_1d_to_fixed_k(values: np.ndarray, target_k: int) -> np.ndarray:
    """
    将一维数组补齐到固定长度 target_k。

    在 varproto 版本中，prototype_bank 的空间形状仍然采用 [C, M_max, D]，
    其中 M_max = max(num_prototypes_per_class)。

    因此，对于某个类别，若它实际只使用 k_eff 个 prototype，
    那么它的相对温度数组也需要补齐到 M_max，才能与 bank 对齐。

    补齐策略：
        - 若已有值，则重复最后一个值
        - 若本身为空，则调用方不应走到这里
    """
    k_eff = values.shape[0]
    if k_eff == target_k:
        return values
    if k_eff <= 0:
        raise ValueError("values must contain at least one element")

    pad_num = target_k - k_eff
    pad = np.repeat(values[-1:], repeats=pad_num, axis=0)
    return np.concatenate([values, pad], axis=0)



def _compute_proto_relative_temperatures(
    z_c: torch.Tensor,
    centers_t: torch.Tensor,
    assign_t: torch.Tensor,
    target_k: int,
    enable_scaling: bool,
    eps: float,
    base_temperature: float = 0.07,
    clamp_percentile_low: float = 10.0,
    clamp_percentile_high: float = 90.0,
    log_count_bias: float = 10.0,
) -> torch.Tensor:
    """
    为当前类别中的每个 prototype 计算相对温度缩放系数。

    这里采用与 PCL density 逻辑兼容的方式：
        1) 先估计每个簇的 density
        2) 对 density 做分位裁剪
        3) 将 density 的均值缩放到 base_temperature
        4) 再转换成“相对倍率”返回

    返回值含义：
    ------------
    返回的是 rel_temp，而不是最终温度本身。

    若训练时在 prototype loss 中使用：
        tau_eff = temperature * rel_temp

    并且传入的 temperature == base_temperature，
    那么 tau_eff 就等于这里估计出来的 density 温度。

    若不启用 scaling，则直接返回全 1。

    注意：
    ------
    这里的 target_k 不是全局的 M_max，
    而是“该类别真正有效的 prototype 个数 k_eff”。
    之后再由调用方将结果写入前 k_eff 个槽位。
    """
    if not enable_scaling:
        return torch.ones((target_k,), dtype=torch.float32)

    if base_temperature <= 0:
        raise ValueError(f"base_temperature must be > 0, got {base_temperature}")

    if z_c.numel() == 0 or centers_t.numel() == 0:
        return torch.ones((target_k,), dtype=torch.float32)

    z_t = z_c.float()
    centers_t = centers_t.float()
    assign_t = assign_t.long()

    n_c = z_t.shape[0]
    k_eff = centers_t.shape[0]
    if n_c <= 0 or k_eff <= 0:
        return torch.ones((target_k,), dtype=torch.float32)

    dist_sq_all = torch.cdist(z_t, centers_t, p=2) ** 2
    assigned_dist_sq = dist_sq_all[torch.arange(n_c), assign_t]

    density = torch.zeros((k_eff,), dtype=torch.float32)

    for m in range(k_eff):
        mask_m = assign_t == m
        num_m = int(mask_m.sum().item())
        if num_m > 1:
            dist_m = assigned_dist_sq[mask_m]
            d_m = dist_m.sqrt().mean() / torch.log(
                torch.tensor(float(num_m + log_count_bias), dtype=torch.float32)
            )
            density[m] = d_m

    dmax = density.max()
    if dmax.item() <= 0:
        dmax = torch.tensor(1.0, dtype=torch.float32)

    for m in range(k_eff):
        mask_m = assign_t == m
        num_m = int(mask_m.sum().item())
        if num_m <= 1:
            density[m] = dmax

    density_np = density.cpu().numpy()
    low = np.percentile(density_np, clamp_percentile_low)
    high = np.percentile(density_np, clamp_percentile_high)
    density_np = np.clip(density_np, low, high)

    mean_density = float(density_np.mean())
    if mean_density <= eps:
        density_np = np.ones_like(density_np, dtype=np.float32) * float(base_temperature)
    else:
        density_np = float(base_temperature) * density_np / mean_density

    rel_temp_np = density_np / float(base_temperature)
    return torch.from_numpy(rel_temp_np).float()



def run_per_class_kmeans(
    grouped: Dict[int, Dict[str, torch.Tensor]],
    num_classes: int,
    feat_dim: int,
    num_prototypes_per_class: List[int],
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    enable_prototype_temperature_scaling: bool = False,
    base_temperature: float = 0.07,
    proto_temperature_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    对每个类别单独执行 KMeans，并构建训练阶段所需的 prototype state。

    这个 varproto 版本的核心设计是：
    --------------------------------
    1) 对外仍然返回一个致密张量 prototype_bank，形状固定为 [C, M_max, D]
       其中 M_max = max(num_prototypes_per_class)

    2) 但每个类别真正有效的 prototype 个数由 class_num_prototypes[cls] 指定
       只有 bank[cls, :class_num_prototypes[cls]] 才是有效槽位

    3) 所有下游模块（prototype loss / directional loss / EMA update）
       都应该使用 class_num_prototypes 来过滤 padded 槽位

    返回的核心字段：
    ------------------
    prototype_bank:
        [C, M_max, D]，致密存储的 prototype bank。

    proto_rel_temperature_bank:
        [C, M_max]，每个 prototype 对应的相对温度倍率。
        对于 padded 槽位，默认保持为 1。

    class_num_prototypes:
        [C]，记录每个类别当前真正有效的 prototype 数量。

    sample_to_proto:
        [N_slot]，记录每个 global_index 属于该类别内部的哪个 prototype。
        注意这里存的是“类内 prototype id”，范围是 [0, class_num_prototypes[cls)-1]。

    sample_to_class:
        [N_slot]，记录每个 global_index 的类别 id。

    valid_sample_mask:
        [N_slot]，标记哪些 global_index 在 refresh 数据集中实际有效。
    """
    if len(num_prototypes_per_class) != num_classes:
        raise ValueError(
            f"num_prototypes_per_class length mismatch: expect {num_classes}, "
            f"got {len(num_prototypes_per_class)}"
        )
    if any(int(v) <= 0 for v in num_prototypes_per_class):
        raise ValueError(f"All prototype counts must be > 0, got {num_prototypes_per_class}")

    all_indices = [item["indices"] for item in grouped.values()] # [[N], [N], [N], ..., [N]]
    if len(all_indices) == 0:
        raise RuntimeError("grouped is empty; cannot run KMeans")

    max_index = int(torch.cat(all_indices, dim=0).max().item())
    num_slots = max_index + 1

    requested_num_prototypes_per_class = [int(x) for x in num_prototypes_per_class]
    max_num_prototypes = int(max(requested_num_prototypes_per_class))

    prototype_bank = torch.zeros((num_classes, max_num_prototypes, feat_dim), dtype=torch.float32)
    proto_rel_temperature_bank = torch.ones((num_classes, max_num_prototypes), dtype=torch.float32)
    class_num_prototypes = torch.zeros((num_classes,), dtype=torch.long)

    sample_to_proto = torch.full((num_slots,), -1, dtype=torch.long)
    sample_to_class = torch.full((num_slots,), -1, dtype=torch.long)
    valid_sample_mask = torch.zeros((num_slots,), dtype=torch.bool)

    counts_per_class: Dict[int, int] = {}
    cluster_meta: Dict[int, Dict[str, Any]] = {}

    for cls in range(num_classes):
        requested_k = requested_num_prototypes_per_class[cls]

        if cls not in grouped:
            counts_per_class[cls] = 0
            class_num_prototypes[cls] = 0
            cluster_meta[cls] = {
                "n_samples": 0,
                "requested_k": int(requested_k),
                "k_eff": 0,
                "note": "no samples for this class",
                "rel_temperatures": [],
            }
            continue

        idx_c = grouped[cls]["indices"].long()
        y_c = grouped[cls]["labels"].long()
        z_c = grouped[cls]["feats"].float()

        n_c = int(z_c.shape[0])
        counts_per_class[cls] = n_c

        sample_to_class[idx_c] = y_c
        valid_sample_mask[idx_c] = True

        if n_c <= 0:
            class_num_prototypes[cls] = 0
            cluster_meta[cls] = {
                "n_samples": 0,
                "requested_k": int(requested_k),
                "k_eff": 0,
                "note": "empty class",
                "rel_temperatures": [],
            }
            continue

        # 某一类的有效 prototype 个数不能超过该类样本数。
        k_eff = min(int(requested_k), n_c)
        class_num_prototypes[cls] = int(k_eff)

        z_np = z_c.numpy()

        if n_c == 1:
            centers = z_np.copy()
            assign = np.zeros((1,), dtype=np.int64)
            inertia = 0.0
        else:
            km = KMeans(
                n_clusters=k_eff,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
            )
            assign = km.fit_predict(z_np)
            centers = km.cluster_centers_
            inertia = float(km.inertia_)

        centers_t = torch.from_numpy(centers).float()
        centers_t = F.normalize(centers_t, dim=1)

        prototype_bank[cls, :k_eff] = centers_t

        assign_t = torch.from_numpy(assign).long()
        sample_to_proto[idx_c] = assign_t

        rel_temp_eff = _compute_proto_relative_temperatures(
            z_c=z_c,
            centers_t=centers_t,
            assign_t=assign_t,
            target_k=k_eff,
            enable_scaling=enable_prototype_temperature_scaling,
            eps=proto_temperature_eps,
            base_temperature=base_temperature,
            clamp_percentile_low=10.0,
            clamp_percentile_high=90.0,
            log_count_bias=10.0,
        )
        proto_rel_temperature_bank[cls, :k_eff] = rel_temp_eff

        cluster_meta[cls] = {
            "n_samples": int(n_c),
            "requested_k": int(requested_k),
            "k_eff": int(k_eff),
            "inertia": inertia,
            "note": "kmeans" if n_c > 1 else "single-sample class",
            "rel_temperatures": [float(x) for x in rel_temp_eff.tolist()],
        }

    # 只对真正有效的槽位做 normalize。
    # padded 槽位保持为 0，不参与训练。
    for cls in range(num_classes):
        k_eff = int(class_num_prototypes[cls].item())
        if k_eff > 0:
            prototype_bank[cls, :k_eff] = F.normalize(prototype_bank[cls, :k_eff], dim=1)

    return {
        "prototype_bank": prototype_bank,
        "proto_rel_temperature_bank": proto_rel_temperature_bank,
        "class_num_prototypes": class_num_prototypes,
        "max_num_prototypes": max_num_prototypes,
        "sample_to_proto": sample_to_proto,
        "sample_to_class": sample_to_class,
        "valid_sample_mask": valid_sample_mask,
        "counts_per_class": counts_per_class,
        "cluster_meta": cluster_meta,
    }


# ============================================================
# 高层接口：刷新 prototypes
# ============================================================

def refresh_prototypes(
    model: torch.nn.Module,
    args,
    cfg: PrototypeRefreshConfig,
) -> Dict[str, Any]:
    """
    刷新 prototype 的高层入口。

    流程：
    ------
    1) 构建 refresh 专用 loader
    2) 用 momentum encoder 对整个训练集提特征
    3) 按 class_id 分组
    4) 对每个类别单独做 KMeans
    5) 返回训练阶段真正需要的 prototype state

    返回字段：
    ----------
    prototype_bank:
        [C, M_max, D]

    proto_rel_temperature_bank:
        [C, M_max]

    class_num_prototypes:
        [C]

    sample_to_proto:
        [N_slot]

    sample_to_class:
        [N_slot]
        这是按你的要求保留的输出字段。
        它在当前训练主循环中不是必须的，但对调试和后续检查很有帮助。

    valid_sample_mask:
        [N_slot]

    counts_per_class / cluster_meta:
        便于打印和调试
    """
    if cfg.require_main_process_only and (not is_main_process()):
        return {}

    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = load_label_map_json(_resolve_label_map_path(args))
    if cfg.tier_mode not in label_map:
        raise KeyError(f"tier_mode={cfg.tier_mode} not found in label_map.json")

    num_classes = len(label_map[cfg.tier_mode])
    if num_classes <= 0:
        raise ValueError(f"No classes found for tier={cfg.tier_mode}")

    if cfg.num_prototypes_per_class is None:
        num_prototypes_per_class = [int(cfg.default_num_prototypes)] * num_classes
    else:
        num_prototypes_per_class = [int(x) for x in cfg.num_prototypes_per_class]
        if len(num_prototypes_per_class) != num_classes:
            raise ValueError(
                f"num_prototypes_per_class length mismatch: expect {num_classes}, "
                f"got {len(num_prototypes_per_class)}"
            )
        if any(x <= 0 for x in num_prototypes_per_class):
            raise ValueError(f"All prototype counts must be > 0, got {num_prototypes_per_class}")

    feat_loader = build_feature_extraction_loader(args, cfg)

    extracted = extract_features_momentum(
        model=model,
        loader=feat_loader,
        device=device,
        tier_mode=cfg.tier_mode,
        args=args,
        main_process_only=cfg.require_main_process_only,
    )

    indices = extracted["indices"].long()
    labels = extracted["labels"].long()
    feats = extracted["feats"].float()

    if feats.ndim != 2 or feats.shape[0] <= 0:
        raise RuntimeError(f"Invalid extracted feature shape: {tuple(feats.shape)}")

    feat_dim = int(feats.shape[1]) # D
    grouped = group_features_by_class(indices=indices, labels=labels, feats=feats)

    clustered = run_per_class_kmeans(
        grouped=grouped,
        num_classes=num_classes,
        feat_dim=feat_dim,
        num_prototypes_per_class=num_prototypes_per_class,
        random_state=cfg.random_state,
        n_init=cfg.n_init,
        max_iter=cfg.max_iter,
        enable_prototype_temperature_scaling=cfg.enable_prototype_temperature_scaling,
        base_temperature=cfg.proto_base_temperature,
        proto_temperature_eps=cfg.proto_temperature_eps,
    )

    return {
        "prototype_bank": clustered["prototype_bank"],
        "proto_rel_temperature_bank": clustered["proto_rel_temperature_bank"],
        "class_num_prototypes": clustered["class_num_prototypes"],
        "sample_to_proto": clustered["sample_to_proto"],
        "sample_to_class": clustered["sample_to_class"],
        "valid_sample_mask": clustered["valid_sample_mask"],
        "counts_per_class": clustered["counts_per_class"],
        "cluster_meta": clustered["cluster_meta"],
        "tier_mode": cfg.tier_mode,
        "num_classes": num_classes,
        "requested_num_prototypes_per_class": num_prototypes_per_class,
        "max_num_prototypes": clustered["max_num_prototypes"],
        "feat_dim": feat_dim,
        "enable_prototype_temperature_scaling": cfg.enable_prototype_temperature_scaling,
    }


# ============================================================
# 辅助打印与广播
# ============================================================

def summarize_proto_state(proto_state: Dict[str, Any]) -> str:
    """
    将 prototype state 格式化成一行简洁字符串，便于日志打印。
    """
    if not proto_state:
        return "[prototype_refresh] empty state"

    num_classes = proto_state["num_classes"]
    max_num_prototypes = proto_state["max_num_prototypes"]
    feat_dim = proto_state["feat_dim"]
    counts_per_class = proto_state["counts_per_class"]
    requested_num_prototypes_per_class = proto_state["requested_num_prototypes_per_class"]
    use_temp_scaling = proto_state.get("enable_prototype_temperature_scaling", False)

    valid_classes = sum(1 for _, c in counts_per_class.items() if c > 0)
    total_samples = sum(int(c) for c in counts_per_class.values())

    return (
        f"[prototype_refresh] total_samples={total_samples}, "
        f"valid_classes={valid_classes}/{num_classes}, "
        f"max_num_prototypes={max_num_prototypes}, "
        f"requested_per_class={requested_num_prototypes_per_class}, "
        f"feat_dim={feat_dim}, proto_temp_scaling={use_temp_scaling}"
    )



def broadcast_proto_state(
    proto_state: Optional[dict],
    device: torch.device,
    rank: int,
) -> Optional[dict]:
    """
    将训练阶段真正需要的 prototype 状态广播到所有 rank。

    广播字段：
    ----------
    - prototype_bank
    - proto_rel_temperature_bank
    - class_num_prototypes
    - sample_to_proto
    - valid_sample_mask
    - enable_prototype_temperature_scaling

    另外，为了保留调试便利性，本实现也会把 sample_to_class 一并带上。
    训练本身并不依赖 sample_to_class，但保留它不会影响主流程。

    说明：
    ------
    1) 单进程模式下，只做 device 搬运。
    2) 多进程模式下，要求 rank 0 的 proto_state 非空，其余 rank 可传 None。
    """
    if proto_state is None:
        return None

    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
        out = dict(proto_state)
        out["prototype_bank"] = out["prototype_bank"].to(device)
        out["proto_rel_temperature_bank"] = out["proto_rel_temperature_bank"].to(device)
        out["class_num_prototypes"] = out["class_num_prototypes"].to(device).long()
        out["sample_to_proto"] = out["sample_to_proto"].to(device).long()
        out["sample_to_class"] = out["sample_to_class"].to(device).long()
        out["valid_sample_mask"] = out["valid_sample_mask"].to(device).bool()
        return out

    obj_list = [None]
    if rank == 0:
        obj_list[0] = {
            "prototype_bank": proto_state["prototype_bank"].cpu(),
            "proto_rel_temperature_bank": proto_state["proto_rel_temperature_bank"].cpu(),
            "class_num_prototypes": proto_state["class_num_prototypes"].cpu(),
            "sample_to_proto": proto_state["sample_to_proto"].cpu(),
            "sample_to_class": proto_state["sample_to_class"].cpu(),
            "valid_sample_mask": proto_state["valid_sample_mask"].cpu(),
            "enable_prototype_temperature_scaling": proto_state.get(
                "enable_prototype_temperature_scaling", False
            ),
        }

    dist.broadcast_object_list(obj_list, src=0)
    out = obj_list[0]
    out["prototype_bank"] = out["prototype_bank"].to(device, non_blocking=True)
    out["proto_rel_temperature_bank"] = out["proto_rel_temperature_bank"].to(device, non_blocking=True)
    out["class_num_prototypes"] = out["class_num_prototypes"].to(device, non_blocking=True).long()
    out["sample_to_proto"] = out["sample_to_proto"].to(device, non_blocking=True).long()
    out["sample_to_class"] = out["sample_to_class"].to(device, non_blocking=True).long()
    out["valid_sample_mask"] = out["valid_sample_mask"].to(device, non_blocking=True).bool()
    return out
