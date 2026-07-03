#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mapstype_dataloader_with_index_mindrove.py

在原有 map-style RGB / Depth dataloader 基础上，新增对 MindRove EMG / IMU 数据的读取、
重采样与样本级增强支持。

本文件的设计目标
================
1) 保持 RGB / Depth 的现有读取、采样、增强、输出逻辑不变
2) 新增 MindRove 模态，支持：
   - 仅输出 EMG / 仅输出 IMU / 同时输出 EMG+IMU
   - 仅输出左手 / 仅输出右手 / 同时输出左右手
   - 可选：将左右手同类信号在通道维拼接后输出
   - 支持 EMG / IMU 使用不同的重采样目标长度
   - 每一种 signal 内部统一重采样到固定长度，以便同一 key 可以 batch stack
   - 可选 single-view / two-view
3) MindRove 的 two-view 逻辑与 RGB 不同：
   - 不是从原始长序列中裁两个时间片段
   - 而是先对完整序列按 signal 各自的目标长度重采样
   - 再对同一个完整片段独立应用两次增强，得到两个 view
4) MindRove 增强已接入独立的 tensor-native 模块：
   - aug.mindrove_augmentation_tensor.apply_mindrove_augmentation
   - 输入为 Dict[str, Tensor[C,L]]
   - 输出为同结构 Dict[str, Tensor[C,L]]
5) 非训练模式（cfg.is_train=False）下：
   - MindRove 只进行读取与重采样
   - 自动关闭 MindRove 增强
   - 若仍开启 mindrove_two_views，则会返回两份相同的重采样结果；
     通常 validation / test 建议同时设置 mindrove_two_views=False

MindRove 输出约定
=================
MindRove 输出统一采用 channel-first，便于直接送入 Conv1d：

- 单视图:
    out["mindrove"] = {
        "left_emg":  Tensor[C,L_emg],
        "left_imu":  Tensor[C,L_imu],
        "right_emg": Tensor[C,L_emg],
        "right_imu": Tensor[C,L_imu],
        ...
    }

  注意：不同 key 的最后一维长度可以不同，例如 EMG 为 L_emg，IMU 为 L_imu。
  collate_fn 会按 dict key 分别 stack，因此同一个 key 在 batch 内长度一致即可。

- 双视图:
    out["mindrove"] = (
        {"left_emg": Tensor[C,L], ...},   # view1
        {"left_emg": Tensor[C,L], ...},   # view2
    )

若开启 mindrove_merge_hands=True，则会把左右手同类信号在通道维拼接：
- EMG: [8,L_emg] + [8,L_emg] -> [16,L_emg]
- IMU: [6,L_imu] + [6,L_imu] -> [12,L_imu]

例如：
    out["mindrove"] = {
        "emg": Tensor[16,L],
        "imu": Tensor[12,L],
    }

MindRove 流程
=============
对每个样本中的 MindRove 数据，处理顺序为：

1) 从 mindrove.pt 读取所请求的 hand / signal
2) 将每条原始流从 [L,C] 按 signal 各自目标长度重采样，并转成 [C,L_signal]
3) 若 cfg.mindrove_apply_normalization=True：
   按左右手 + 模态各自提供的 per-channel mean/std 做标准化
4) 若 cfg.is_train=True 且 cfg.mindrove_apply_augmentation=True：
   调用独立增强模块进行样本级增强
5) 若 cfg.mindrove_two_views=True：
   对同一个“已重采样且已标准化”的片段独立增强两次，得到两个 view
6) 若 cfg.mindrove_merge_hands=True：
   在标准化 / 增强之后按 signal 合并左右手

注意
====
1) 当前实现严格检查输入格式，不做模糊兜底
2) 若启用 mindrove_merge_hands=True，则必须同时请求左右手
3) 若 missing_policy="skip"，缺失手或缺失文件会直接报错
4) 若 missing_policy="pad"，会用全零序列补齐缺失 MindRove 分支
5) validation / test 模式建议：
   - is_train=False
   - mindrove_two_views=False
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as Fv2
import random
import numpy as np

# 你现有的空间增强文件（RGB）
from aug.spatial_augmentation import TemporallyConsistentSpatialAugmentation, ValidationAugmentation
# 你现有的时间采样文件（RGB / Depth）
from aug.temporal_augmentation_adaptive import sample_indices_strict, sample_two_views_indices
# mindrove 增强文件
from aug.mindrove_augmentation_tensor_varlen import apply_mindrove_augmentation as MindRoveAugmentation


# ============================================================
# 1) 配置
# ============================================================

@dataclass
class PackedMultiModalConfig:
    # ---------------- RGB / Depth 时间采样 ----------------
    n_frames: int = 16

    # RGB 是否 two-view（对比学习）
    rgb_two_views: bool = False

    # 启用哪些模态，可选：rgb / depth / mindrove
    use_modalities: Tuple[str, ...] = ("rgb", "depth")

    # 缺失策略：skip / pad
    missing_policy: str = "skip"

    # ---------------- labels / tier ----------------
    load_labels: bool = True
    label_map_path: Optional[str] = None

    # "all" / "tier1" / "tier2" / "tier3"
    tier_mode: str = "all"

    # ---------------- train / val ----------------
    is_train: bool = True

    # -------- RGB 空间增强参数 --------
    rgb_out_hw: Tuple[int, int] = (224, 224)

    # RandomResizedCrop
    rrc_scale: Tuple[float, float] = (0.6, 1.0)
    rrc_ratio: Tuple[float, float] = (0.75, 1.3333333333)

    # 是否启用训练阶段随机 spatial augmentation
    # 注意：这里不新增无随机增强 transform。
    # 如果设为 False，则在 build_packed_mapstyle_dataset 中把各增强概率置 0，
    # 但仍然使用 TemporallyConsistentSpatialAugmentation。
    rgb_apply_spatial_aug: bool = True

    # Flip
    rgb_hflip_p: float = 0.5
    rgb_vflip_p: float = 0.5

    # ColorJitter
    rgb_jitter_p: float = 0.5
    rgb_jitter_brightness: float = 0.24
    rgb_jitter_contrast: float = 0.24
    rgb_jitter_saturation: float = 0.24
    rgb_jitter_hue: float = 0.16

    # Grayscale
    rgb_gray_p: float = 0.2

    # GaussianBlur
    rgb_blur_p: float = 0.5
    rgb_blur_kernel: int = 7
    rgb_blur_sigma: Tuple[float, float] = (0.1, 1.0)

    # RGB normalize 参数（用于可视化反归一化）
    rgb_mean: Tuple[float, float, float] = (0.356, 0.363, 0.367)
    rgb_std: Tuple[float, float, float] = (0.288, 0.271, 0.270)

    # ---------------- Depth 输出尺寸 ----------------
    depth_out_hw: Tuple[int, int] = (224, 224)

    # ---------------- 默认 pad 尺寸 ----------------
    default_rgb_hw: Tuple[int, int] = (256, 256)
    default_depth_hw: Tuple[int, int] = (224, 224)

    # Depth pad dtype
    default_depth_dtype: str = "int32"
    
    # 由 build_packed_mapstyle_dataset(...) 在运行时挂进去
    rgb_transform: Optional[Any] = field(default=None, repr=False, compare=False)

    # ---------------- MindRove 参数 ----------------
    # 是否输出两个增强视角
    mindrove_two_views: bool = False

    # MindRove 默认目标长度（与 RGB 的 n_frames 含义不同，不复用）。
    # 兼容旧配置：如果不单独设置 mindrove_emg_target_len / mindrove_imu_target_len，
    # EMG 和 IMU 都会使用这个长度。
    mindrove_target_len: int = 256

    # 可选：按 signal 单独指定目标长度。
    # 典型场景：
    #   mindrove_emg_target_len = 512
    #   mindrove_imu_target_len = 256
    # 如果为 None，则回退到 mindrove_target_len。
    #
    # 注意：
    # - 同一种 signal 内部，left/right 必须使用相同长度，才能在通道维合并。
    # - 不同 signal 之间允许长度不同，例如 emg: [C,512], imu: [C,256]。
    mindrove_emg_target_len: Optional[int] = None
    mindrove_imu_target_len: Optional[int] = None

    # 请求哪只手，可选：left / right
    mindrove_hands: Tuple[str, ...] = ("left", "right")

    # 请求哪种信号，可选：emg / imu
    mindrove_signals: Tuple[str, ...] = ("emg", "imu")

    # 是否把左右手同类信号合并输出
    mindrove_merge_hands: bool = False

    # 是否启用 MindRove 增强占位逻辑
    # 目前占位逻辑为 identity，不改数值；后续你可直接替换对应函数
    mindrove_apply_augmentation: bool = True

    # 是否对 MindRove 做标准化。
    # 标准化在“重采样之后、增强之前”执行，公式为：
    #     x_norm = (x - mean) / std
    # 这里的 mean / std 必须由外部显式提供，且按“左右手 + 模态”分别配置。
    mindrove_apply_normalization: bool = False

    # ---------------- MindRove normalization stats ----------------
    # 约定：
    # - EMG 的 mean/std 长度必须为 8
    # - IMU 的 mean/std 长度必须为 6
    # - 若启用 mindrove_apply_normalization=True，则对所有被请求的 hand+signal，
    #   都必须显式提供对应的 mean 与 std；否则直接报错。
    mindrove_left_emg_mean: Optional[Tuple[float, ...]] = None
    mindrove_left_emg_std: Optional[Tuple[float, ...]] = None
    mindrove_right_emg_mean: Optional[Tuple[float, ...]] = None
    mindrove_right_emg_std: Optional[Tuple[float, ...]] = None

    mindrove_left_imu_mean: Optional[Tuple[float, ...]] = None
    mindrove_left_imu_std: Optional[Tuple[float, ...]] = None
    mindrove_right_imu_mean: Optional[Tuple[float, ...]] = None
    mindrove_right_imu_std: Optional[Tuple[float, ...]] = None

    # ---------------- MindRove augmentation params ----------------
    mindrove_time_warp_prob: float = 0.5
    mindrove_time_warp_sigma: float = 0.2
    mindrove_time_warp_num_knots: int = 4
    mindrove_time_warp_num_splines: int = 150

    mindrove_emg_scaling_prob: float = 0.5
    mindrove_emg_scaling_sigma: float = 0.1
    mindrove_emg_noise_prob: float = 0.8
    mindrove_emg_noise_sigma: float = 0.05

    # Drift：尽量对齐 tsaug.Drift
    mindrove_emg_drift_prob: float = 0.0
    mindrove_emg_drift_max: Union[float, Tuple[float, float]] = 0.0
    mindrove_emg_drift_n_points: Union[int, List[int]] = 3
    mindrove_emg_drift_kind: str = "additive"
    mindrove_emg_drift_per_channel: bool = False
    mindrove_emg_drift_normalize: bool = True

    mindrove_imu_scaling_prob: float = 0.5
    mindrove_imu_scaling_sigma: float = 0.1
    mindrove_imu_noise_prob: float = 0.8
    mindrove_imu_noise_sigma: float = 0.05

    mindrove_imu_drift_prob: float = 0.0
    mindrove_imu_drift_max: Union[float, Tuple[float, float]] = 0.0
    mindrove_imu_drift_n_points: Union[int, List[int]] = 3
    mindrove_imu_drift_kind: str = "additive"
    mindrove_imu_drift_per_channel: bool = False
    mindrove_imu_drift_normalize: bool = False

    mindrove_emg_negate_prob: float = 0.0
    mindrove_imu_negate_prob: float = 0.0
    mindrove_emg_channel_dropout_prob: float = 0.0
    mindrove_emg_channel_dropout_max_channels: int = 1
    mindrove_imu_channel_dropout_prob: float = 0.0
    mindrove_imu_channel_dropout_max_channels: int = 1


# ============================================================
# 2) label_map 加载与 tier 映射
# ============================================================

def load_label_map_json(path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    读取 label_map.json：
    {
      "tier1": {"cap": 0, ...},
      "tier2": {"cap_pen": 0, ...},
      "tier3": {"cap_red_pen": 0, ...}
    }
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    out: Dict[str, Dict[str, int]] = {}
    for tier in ("tier1", "tier2", "tier3"):
        mp = obj.get(tier, {}) or {}
        if not isinstance(mp, dict):
            raise ValueError(f"label_map.json: '{tier}' must be a dict, got {type(mp)}")
        out[tier] = {str(k): int(v) for k, v in mp.items()}
    return out


def build_label_map_from_manifest(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    如果没有提供外部 label_map.json，就从 manifest 动态构造。
    注意：
    - 这种做法适合单 split 调试
    - 正式 train/val/test 最好还是用统一 label_map.json
    """
    out: Dict[str, Dict[str, int]] = {}
    for tier in ("tier1", "tier2", "tier3"):
        names = []
        for rec in records:
            v = rec.get(tier, None)
            if v is not None:
                names.append(str(v))
        uniq = sorted(set(names))
        out[tier] = {name: i for i, name in enumerate(uniq)}
    return out


def get_required_tiers(tier_mode: str) -> List[str]:
    if tier_mode in ("tier1", "tier2", "tier3"):
        return [tier_mode]
    return ["tier1", "tier2", "tier3"]


def map_tier_actions_to_ids(
    labels: Dict[str, Any],
    label_map: Dict[str, Dict[str, int]],
    tiers: List[str],
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    tier_actions: Dict[str, Optional[str]] = {}
    tier_ids: Dict[str, int] = {}

    for t in tiers:
        a = labels.get(t, None)
        if a is None:
            tier_actions[t] = None
            tier_ids[t] = -1
            continue
        a_str = str(a)
        tier_actions[t] = a_str
        tier_ids[t] = int(label_map.get(t, {}).get(a_str, -1))

    return tier_actions, tier_ids


# ============================================================
# 3) 配置与工具函数
# ============================================================
def _seed_worker(worker_id: int) -> None:
    """
    为每个 DataLoader worker 单独设置随机种子。

    说明
    ----
    1) PyTorch 会先给每个 worker 分配一个独立的 torch seed
    2) 这里再把这个 seed 同步给 numpy 和 python.random
    3) 这样可保证：
       - torch / numpy / random 在同一 worker 内保持一致
       - 不同 worker 之间不会共享同样的 numpy 随机状态

    注意
    ----
    torch.initial_seed() 返回的是当前 worker 已经被 PyTorch 设置好的 seed。
    为了兼容 numpy 的 seed 范围，这里取低 32 位。
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)



MINDROVE_SIGNAL_CHANNELS: Dict[str, int] = {
    "emg": 8,
    "imu": 6,
}


def _get_mindrove_target_len(cfg: "PackedMultiModalConfig", signal: str) -> int:
    """
    返回某一种 MindRove signal 的目标重采样长度。

    兼容逻辑
    --------
    1) 如果 signal == "emg" 且 cfg.mindrove_emg_target_len 不为 None，
       则使用 mindrove_emg_target_len。
    2) 如果 signal == "imu" 且 cfg.mindrove_imu_target_len 不为 None，
       则使用 mindrove_imu_target_len。
    3) 否则回退到 cfg.mindrove_target_len。

    这样旧脚本只设置 mindrove_target_len 时行为完全不变；
    新脚本可以显式设置 EMG / IMU 不同长度。
    """
    signal = str(signal).strip().lower()
    if signal == "emg" and cfg.mindrove_emg_target_len is not None:
        return int(cfg.mindrove_emg_target_len)
    if signal == "imu" and cfg.mindrove_imu_target_len is not None:
        return int(cfg.mindrove_imu_target_len)
    if signal not in MINDROVE_SIGNAL_CHANNELS:
        raise KeyError(f"Unsupported signal: {signal}")
    return int(cfg.mindrove_target_len)


def _validate_mindrove_channel_stats(
    name: str,
    value: Any,
    expected_channels: int,
) -> Tuple[float, ...]:
    """
    严格校验单组 MindRove 标准化统计量。

    要求
    ----
    1) value 必须是 list 或 tuple
    2) 长度必须与该信号通道数一致
    3) 所有元素都必须是 int / float
    """
    if value is None:
        raise ValueError(f"{name} must be provided when MindRove normalization is enabled.")

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be list or tuple, got {type(value)}")

    if len(value) != expected_channels:
        raise ValueError(
            f"{name} must have length {expected_channels}, got length {len(value)}: {value}"
        )

    out: List[float] = []
    for i, v in enumerate(value):
        if not isinstance(v, (int, float)):
            raise TypeError(f"{name}[{i}] must be int or float, got {type(v)}")
        out.append(float(v))
    return tuple(out)


def _normalize_modalities(use_modalities: Tuple[str, ...]) -> Tuple[str, ...]:
    x = tuple(str(m).strip().lower() for m in use_modalities if str(m).strip() != "")
    valid = {"rgb", "depth", "mindrove"}
    bad = [m for m in x if m not in valid]
    if bad:
        raise ValueError(f"Unsupported modalities: {bad}. Supported: rgb, depth, mindrove")
    if len(x) == 0:
        raise ValueError("At least one modality must be enabled.")
    return x


def _normalize_hands(hands: Tuple[str, ...]) -> Tuple[str, ...]:
    x = tuple(str(h).strip().lower() for h in hands if str(h).strip() != "")
    valid = {"left", "right"}
    bad = [h for h in x if h not in valid]
    if bad:
        raise ValueError(f"Unsupported mindrove_hands: {bad}. Supported: left, right")
    if len(x) == 0:
        raise ValueError("mindrove_hands cannot be empty.")
    return x


def _normalize_signals(signals: Tuple[str, ...]) -> Tuple[str, ...]:
    x = tuple(str(s).strip().lower() for s in signals if str(s).strip() != "")
    valid = {"emg", "imu"}
    bad = [s for s in x if s not in valid]
    if bad:
        raise ValueError(f"Unsupported mindrove_signals: {bad}. Supported: emg, imu")
    if len(x) == 0:
        raise ValueError("mindrove_signals cannot be empty.")
    return x


def _validate_config(cfg: PackedMultiModalConfig) -> None:
    cfg.use_modalities = _normalize_modalities(cfg.use_modalities)

    if cfg.missing_policy not in ("skip", "pad"):
        raise ValueError(f"missing_policy must be 'skip' or 'pad', got {cfg.missing_policy}")

    # ---------------- RGB spatial augmentation 参数检查 ----------------
    if "rgb" in cfg.use_modalities:
        if not isinstance(cfg.rgb_apply_spatial_aug, bool):
            raise TypeError(
                f"rgb_apply_spatial_aug must be bool, got {type(cfg.rgb_apply_spatial_aug)}"
            )

        if not isinstance(cfg.rgb_out_hw, tuple) or len(cfg.rgb_out_hw) != 2:
            raise TypeError(f"rgb_out_hw must be tuple[int, int], got {cfg.rgb_out_hw}")
        if not all(isinstance(v, int) and v > 0 for v in cfg.rgb_out_hw):
            raise ValueError(f"rgb_out_hw must contain positive ints, got {cfg.rgb_out_hw}")

        if not isinstance(cfg.rrc_scale, tuple) or len(cfg.rrc_scale) != 2:
            raise TypeError(f"rrc_scale must be tuple[float, float], got {cfg.rrc_scale}")
        if not (0.0 < float(cfg.rrc_scale[0]) <= float(cfg.rrc_scale[1]) <= 1.0):
            raise ValueError(
                f"rrc_scale must satisfy 0 < min <= max <= 1, got {cfg.rrc_scale}"
            )

        if not isinstance(cfg.rrc_ratio, tuple) or len(cfg.rrc_ratio) != 2:
            raise TypeError(f"rrc_ratio must be tuple[float, float], got {cfg.rrc_ratio}")
        if not (0.0 < float(cfg.rrc_ratio[0]) <= float(cfg.rrc_ratio[1])):
            raise ValueError(
                f"rrc_ratio must satisfy 0 < min <= max, got {cfg.rrc_ratio}"
            )

        for name in (
            "rgb_hflip_p",
            "rgb_vflip_p",
            "rgb_jitter_p",
            "rgb_gray_p",
            "rgb_blur_p",
        ):
            value = getattr(cfg, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be int or float, got {type(value)}")
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        for name in (
            "rgb_jitter_brightness",
            "rgb_jitter_contrast",
            "rgb_jitter_saturation",
        ):
            value = getattr(cfg, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be int or float, got {type(value)}")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if not isinstance(cfg.rgb_jitter_hue, (int, float)):
            raise TypeError(f"rgb_jitter_hue must be int or float, got {type(cfg.rgb_jitter_hue)}")
        if not (0.0 <= float(cfg.rgb_jitter_hue) <= 0.5):
            raise ValueError(f"rgb_jitter_hue must be in [0, 0.5], got {cfg.rgb_jitter_hue}")

        if not isinstance(cfg.rgb_blur_kernel, int):
            raise TypeError(f"rgb_blur_kernel must be int, got {type(cfg.rgb_blur_kernel)}")
        if cfg.rgb_blur_kernel < 3 or cfg.rgb_blur_kernel % 2 == 0:
            raise ValueError(
                f"rgb_blur_kernel must be an odd integer >= 3, got {cfg.rgb_blur_kernel}"
            )

        if not isinstance(cfg.rgb_blur_sigma, tuple) or len(cfg.rgb_blur_sigma) != 2:
            raise TypeError(f"rgb_blur_sigma must be tuple[float, float], got {cfg.rgb_blur_sigma}")
        if not (0.0 < float(cfg.rgb_blur_sigma[0]) <= float(cfg.rgb_blur_sigma[1])):
            raise ValueError(
                f"rgb_blur_sigma must satisfy 0 < min <= max, got {cfg.rgb_blur_sigma}"
            )

        if not isinstance(cfg.rgb_mean, tuple) or len(cfg.rgb_mean) != 3:
            raise TypeError(f"rgb_mean must be tuple[float, float, float], got {cfg.rgb_mean}")
        if not isinstance(cfg.rgb_std, tuple) or len(cfg.rgb_std) != 3:
            raise TypeError(f"rgb_std must be tuple[float, float, float], got {cfg.rgb_std}")
        if any(float(v) <= 0.0 for v in cfg.rgb_std):
            raise ValueError(f"rgb_std must contain only positive values, got {cfg.rgb_std}")
        
    if "mindrove" in cfg.use_modalities:
        cfg.mindrove_hands = _normalize_hands(cfg.mindrove_hands)
        cfg.mindrove_signals = _normalize_signals(cfg.mindrove_signals)

        if not isinstance(cfg.mindrove_target_len, int) or cfg.mindrove_target_len <= 0:
            raise ValueError(f"mindrove_target_len must be positive int, got {cfg.mindrove_target_len}")

        # 可选的 per-signal 目标长度。
        # 允许 EMG / IMU 使用不同的重采样长度；未设置时回退到 mindrove_target_len。
        for attr_name in ("mindrove_emg_target_len", "mindrove_imu_target_len"):
            attr_value = getattr(cfg, attr_name)
            if attr_value is not None:
                if not isinstance(attr_value, int) or attr_value <= 0:
                    raise ValueError(f"{attr_name} must be positive int or None, got {attr_value}")

        if cfg.mindrove_merge_hands and set(cfg.mindrove_hands) != {"left", "right"}:
            raise ValueError(
                "mindrove_merge_hands=True requires mindrove_hands to contain both 'left' and 'right'."
            )

        if not isinstance(cfg.mindrove_apply_normalization, bool):
            raise TypeError("mindrove_apply_normalization must be bool")

        if cfg.mindrove_apply_normalization:
            for hand in cfg.mindrove_hands:
                for signal in cfg.mindrove_signals:
                    expected_c = MINDROVE_SIGNAL_CHANNELS[signal]

                    mean_attr = f"mindrove_{hand}_{signal}_mean"
                    std_attr = f"mindrove_{hand}_{signal}_std"

                    mean_value = _validate_mindrove_channel_stats(
                        name=mean_attr,
                        value=getattr(cfg, mean_attr),
                        expected_channels=expected_c,
                    )
                    std_value = _validate_mindrove_channel_stats(
                        name=std_attr,
                        value=getattr(cfg, std_attr),
                        expected_channels=expected_c,
                    )

                    if any(v <= 0.0 for v in std_value):
                        raise ValueError(f"{std_attr} must contain only positive values, got {std_value}")

                    setattr(cfg, mean_attr, mean_value)
                    setattr(cfg, std_attr, std_value)

        if cfg.mindrove_apply_augmentation:
            if not (0.0 <= cfg.mindrove_time_warp_prob <= 1.0):
                raise ValueError("mindrove_time_warp_prob must be in [0,1]")
            if cfg.mindrove_time_warp_sigma < 0:
                raise ValueError("mindrove_time_warp_sigma must be >= 0")
            if not isinstance(cfg.mindrove_time_warp_num_knots, int) or cfg.mindrove_time_warp_num_knots < 0:
                raise ValueError("mindrove_time_warp_num_knots must be int >= 0")
            if not isinstance(cfg.mindrove_time_warp_num_splines, int) or cfg.mindrove_time_warp_num_splines <= 0:
                raise ValueError("mindrove_time_warp_num_splines must be int > 0")

            if not (0.0 <= cfg.mindrove_emg_scaling_prob <= 1.0):
                raise ValueError("mindrove_emg_scaling_prob must be in [0,1]")
            if cfg.mindrove_emg_scaling_sigma < 0:
                raise ValueError("mindrove_emg_scaling_sigma must be >= 0")
            if not (0.0 <= cfg.mindrove_emg_noise_prob <= 1.0):
                raise ValueError("mindrove_emg_noise_prob must be in [0,1]")
            if cfg.mindrove_emg_noise_sigma < 0:
                raise ValueError("mindrove_emg_noise_sigma must be >= 0")
            if not (0.0 <= cfg.mindrove_emg_drift_prob <= 1.0):
                raise ValueError("mindrove_emg_drift_prob must be in [0,1]")
            if isinstance(cfg.mindrove_emg_drift_max, (int, float)):
                if float(cfg.mindrove_emg_drift_max) < 0.0:
                    raise ValueError("mindrove_emg_drift_max must be >= 0")
            elif isinstance(cfg.mindrove_emg_drift_max, tuple):
                if len(cfg.mindrove_emg_drift_max) != 2:
                    raise ValueError("mindrove_emg_drift_max tuple must have length 2")
                lo, hi = cfg.mindrove_emg_drift_max
                if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                    raise TypeError("mindrove_emg_drift_max tuple must contain numbers")
                if float(lo) < 0.0 or float(hi) < 0.0 or float(lo) > float(hi):
                    raise ValueError("mindrove_emg_drift_max must satisfy 0 <= low <= high")
            else:
                raise TypeError("mindrove_emg_drift_max must be float or tuple(low, high)")
            if isinstance(cfg.mindrove_emg_drift_n_points, int):
                if cfg.mindrove_emg_drift_n_points <= 0:
                    raise ValueError("mindrove_emg_drift_n_points must be > 0")
            elif isinstance(cfg.mindrove_emg_drift_n_points, list):
                if len(cfg.mindrove_emg_drift_n_points) == 0:
                    raise ValueError("mindrove_emg_drift_n_points list cannot be empty")
                if not all(isinstance(v, int) for v in cfg.mindrove_emg_drift_n_points):
                    raise TypeError("mindrove_emg_drift_n_points list must contain ints")
                if not all(v > 0 for v in cfg.mindrove_emg_drift_n_points):
                    raise ValueError("mindrove_emg_drift_n_points list must contain positive ints")
            else:
                raise TypeError("mindrove_emg_drift_n_points must be int or list[int]")
            if cfg.mindrove_emg_drift_kind not in ("additive", "multiplicative"):
                raise ValueError("mindrove_emg_drift_kind must be 'additive' or 'multiplicative'")
            if not isinstance(cfg.mindrove_emg_drift_per_channel, bool):
                raise TypeError("mindrove_emg_drift_per_channel must be bool")
            if not isinstance(cfg.mindrove_emg_drift_normalize, bool):
                raise TypeError("mindrove_emg_drift_normalize must be bool")

            if not (0.0 <= cfg.mindrove_imu_scaling_prob <= 1.0):
                raise ValueError("mindrove_imu_scaling_prob must be in [0,1]")
            if cfg.mindrove_imu_scaling_sigma < 0:
                raise ValueError("mindrove_imu_scaling_sigma must be >= 0")
            if not (0.0 <= cfg.mindrove_imu_noise_prob <= 1.0):
                raise ValueError("mindrove_imu_noise_prob must be in [0,1]")
            if cfg.mindrove_imu_noise_sigma < 0:
                raise ValueError("mindrove_imu_noise_sigma must be >= 0")
            if not (0.0 <= cfg.mindrove_imu_drift_prob <= 1.0):
                raise ValueError("mindrove_imu_drift_prob must be in [0,1]")
            if isinstance(cfg.mindrove_imu_drift_max, (int, float)):
                if float(cfg.mindrove_imu_drift_max) < 0.0:
                    raise ValueError("mindrove_imu_drift_max must be >= 0")
            elif isinstance(cfg.mindrove_imu_drift_max, tuple):
                if len(cfg.mindrove_imu_drift_max) != 2:
                    raise ValueError("mindrove_imu_drift_max tuple must have length 2")
                lo, hi = cfg.mindrove_imu_drift_max
                if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                    raise TypeError("mindrove_imu_drift_max tuple must contain numbers")
                if float(lo) < 0.0 or float(hi) < 0.0 or float(lo) > float(hi):
                    raise ValueError("mindrove_imu_drift_max must satisfy 0 <= low <= high")
            else:
                raise TypeError("mindrove_imu_drift_max must be float or tuple(low, high)")
            if isinstance(cfg.mindrove_imu_drift_n_points, int):
                if cfg.mindrove_imu_drift_n_points <= 0:
                    raise ValueError("mindrove_imu_drift_n_points must be > 0")
            elif isinstance(cfg.mindrove_imu_drift_n_points, list):
                if len(cfg.mindrove_imu_drift_n_points) == 0:
                    raise ValueError("mindrove_imu_drift_n_points list cannot be empty")
                if not all(isinstance(v, int) for v in cfg.mindrove_imu_drift_n_points):
                    raise TypeError("mindrove_imu_drift_n_points list must contain ints")
                if not all(v > 0 for v in cfg.mindrove_imu_drift_n_points):
                    raise ValueError("mindrove_imu_drift_n_points list must contain positive ints")
            else:
                raise TypeError("mindrove_imu_drift_n_points must be int or list[int]")
            if cfg.mindrove_imu_drift_kind not in ("additive", "multiplicative"):
                raise ValueError("mindrove_imu_drift_kind must be 'additive' or 'multiplicative'")
            if not isinstance(cfg.mindrove_imu_drift_per_channel, bool):
                raise TypeError("mindrove_imu_drift_per_channel must be bool")
            if not isinstance(cfg.mindrove_imu_drift_normalize, bool):
                raise TypeError("mindrove_imu_drift_normalize must be bool")

            if not (0.0 <= cfg.mindrove_emg_negate_prob <= 1.0):
                raise ValueError("mindrove_emg_negate_prob must be in [0,1]")
            if not (0.0 <= cfg.mindrove_imu_negate_prob <= 1.0):
                raise ValueError("mindrove_imu_negate_prob must be in [0,1]")
            if not (0.0 <= cfg.mindrove_emg_channel_dropout_prob <= 1.0):
                raise ValueError("mindrove_emg_channel_dropout_prob must be in [0,1]")
            if not isinstance(cfg.mindrove_emg_channel_dropout_max_channels, int) or cfg.mindrove_emg_channel_dropout_max_channels <= 0:
                raise ValueError("mindrove_emg_channel_dropout_max_channels must be int > 0")
            if not (0.0 <= cfg.mindrove_imu_channel_dropout_prob <= 1.0):
                raise ValueError("mindrove_imu_channel_dropout_prob must be in [0,1]")
            if not isinstance(cfg.mindrove_imu_channel_dropout_max_channels, int) or cfg.mindrove_imu_channel_dropout_max_channels <= 0:
                raise ValueError("mindrove_imu_channel_dropout_max_channels must be int > 0")

def _load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resize_depth_video_keep_dtype(video_tchw: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Depth resize：
    - 输入：[T,1,H,W]，dtype 可能 int32/uint16/uint8
    - 输出：[T,1,outH,outW]，dtype 尽量保持

    做法：
      1) 转 float32
      2) NEAREST resize
      3) round
      4) cast 回原 dtype
    """
    assert video_tchw.ndim == 4 and video_tchw.shape[1] >= 1
    orig_dtype = video_tchw.dtype

    out_frames = []
    for t in range(video_tchw.shape[0]):
        fr = video_tchw[t].to(torch.float32)
        fr = Fv2.resize(
            fr,
            size=list(out_hw),
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
        )
        if orig_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            fr = torch.round(fr)
        fr = fr.to(orig_dtype)
        out_frames.append(fr)

    return torch.stack(out_frames, dim=0).contiguous()


def _make_zero_rgb(cfg: PackedMultiModalConfig) -> torch.Tensor:
    """pad 模式下 RGB 零视频：uint8 [T,3,H,W]"""
    H, W = cfg.default_rgb_hw
    return torch.zeros((cfg.n_frames, 3, H, W), dtype=torch.uint8)


def _make_zero_depth(cfg: PackedMultiModalConfig) -> torch.Tensor:
    """pad 模式下 Depth 零视频：[T,1,H,W]"""
    H, W = cfg.default_depth_hw
    dtype_str = str(cfg.default_depth_dtype).lower()

    if dtype_str == "uint8":
        dtype = torch.uint8
    elif dtype_str == "int16":
        dtype = torch.int16
    elif dtype_str == "int64":
        dtype = torch.int64
    else:
        dtype = torch.int32

    return torch.zeros((cfg.n_frames, 1, H, W), dtype=dtype)


def _make_zero_mindrove_stream(signal: str, target_len: int) -> torch.Tensor:
    """
    pad 模式下的 MindRove 零序列，统一输出为 channel-first: [C,L]
    """
    if signal not in MINDROVE_SIGNAL_CHANNELS:
        raise KeyError(f"Unsupported signal: {signal}")
    C = MINDROVE_SIGNAL_CHANNELS[signal]
    return torch.zeros((C, target_len), dtype=torch.float32)


def _resample_mindrove_lc_to_cf(seq_lc: torch.Tensor, signal: str, target_len: int) -> torch.Tensor:
    """
    将 MindRove 单路序列从 [L,C] 重采样到固定长度，并转成 [C,L]。

    参数
    ----
    seq_lc : Tensor[L,C]
        原始序列
    signal : str
        "emg" 或 "imu"
    target_len : int
        重采样目标长度

    返回
    ----
    Tensor[C,target_len]
    """
    expected_c = MINDROVE_SIGNAL_CHANNELS[signal]

    if not torch.is_tensor(seq_lc):
        raise TypeError(f"MindRove stream must be a tensor, got {type(seq_lc)}")
    if seq_lc.ndim != 2:
        raise ValueError(f"MindRove stream must be [L,C], got shape {tuple(seq_lc.shape)}")
    if seq_lc.shape[1] != expected_c:
        raise ValueError(
            f"MindRove {signal} expects channel dim {expected_c}, got shape {tuple(seq_lc.shape)}"
        )
    if seq_lc.shape[0] <= 0:
        raise ValueError("MindRove sequence length must be > 0.")

    x = seq_lc.to(torch.float32).transpose(0, 1).unsqueeze(0)  # [1,C,L]
    y = Fnn.interpolate(
        x,
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return y.squeeze(0).contiguous()  # [C,L]


# ============================================================
# 4) MindRove 标准化与增强接口
# ============================================================

def _get_mindrove_norm_stats(
    cfg: PackedMultiModalConfig,
    hand: str,
    signal: str,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    读取某一路 MindRove 流对应的 mean/std 配置。
    """
    mean_attr = f"mindrove_{hand}_{signal}_mean"
    std_attr = f"mindrove_{hand}_{signal}_std"

    if not hasattr(cfg, mean_attr):
        raise AttributeError(f"Config missing normalization attribute: {mean_attr}")
    if not hasattr(cfg, std_attr):
        raise AttributeError(f"Config missing normalization attribute: {std_attr}")

    mean = getattr(cfg, mean_attr)
    std = getattr(cfg, std_attr)

    expected_c = MINDROVE_SIGNAL_CHANNELS[signal]
    mean = _validate_mindrove_channel_stats(mean_attr, mean, expected_c)
    std = _validate_mindrove_channel_stats(std_attr, std, expected_c)

    if any(v <= 0.0 for v in std):
        raise ValueError(f"{std_attr} must contain only positive values, got {std}")

    return mean, std


def _apply_mindrove_standardization_cf(
    x_cf: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    key: str,
) -> torch.Tensor:
    """
    对单条 [C,L] 的 MindRove 流做 per-channel mean/std 标准化。
    """
    if not torch.is_tensor(x_cf):
        raise TypeError(f"MindRove stream '{key}' must be tensor, got {type(x_cf)}")
    if x_cf.ndim != 2:
        raise ValueError(f"MindRove stream '{key}' must be [C,L], got {tuple(x_cf.shape)}")

    C = int(x_cf.shape[0])
    if len(mean) != C:
        raise ValueError(f"{key}: mean length {len(mean)} does not match channel count {C}")
    if len(std) != C:
        raise ValueError(f"{key}: std length {len(std)} does not match channel count {C}")

    mean_t = torch.tensor(mean, dtype=x_cf.dtype, device=x_cf.device).view(C, 1)
    std_t = torch.tensor(std, dtype=x_cf.dtype, device=x_cf.device).view(C, 1)

    if torch.any(std_t <= 0):
        raise ValueError(f"{key}: std must be strictly positive")

    y = (x_cf - mean_t) / std_t
    return y.contiguous()


def apply_mindrove_normalization(
    streams: Dict[str, torch.Tensor],
    cfg: PackedMultiModalConfig,
    stream_is_real: Optional[Dict[str, bool]] = None,
) -> Dict[str, torch.Tensor]:
    """
    对 Dict[str, Tensor[C,L]] 中的每一路 MindRove 流做标准化。

    规则
    ----
    1) 仅当 cfg.mindrove_apply_normalization=True 时执行
    2) 标准化发生在“重采样之后、增强之前”
    3) 若 stream_is_real 提供且某一路为 False，则该路保持原值不做标准化
       （典型场景：missing_policy='pad' 时的零填充分支）
    """
    if not isinstance(streams, dict):
        raise TypeError(f"streams must be dict[str, Tensor], got {type(streams)}")

    out: Dict[str, torch.Tensor] = {}
    for key, x in streams.items():
        if not torch.is_tensor(x):
            raise TypeError(f"MindRove stream '{key}' must be tensor, got {type(x)}")
        if x.ndim != 2:
            raise ValueError(f"MindRove stream '{key}' must be [C,L], got {tuple(x.shape)}")

        if stream_is_real is not None:
            if key not in stream_is_real:
                raise KeyError(f"stream_is_real is missing key '{key}'")
            if not stream_is_real[key]:
                out[key] = x.contiguous()
                continue

        parts = key.split("_")
        if len(parts) != 2:
            raise KeyError(
                f"Unsupported MindRove stream key '{key}'. Expected keys like 'left_emg' / 'right_imu'."
            )
        hand, signal = parts
        if hand not in ("left", "right"):
            raise KeyError(f"Unsupported hand in MindRove key '{key}'")
        if signal not in MINDROVE_SIGNAL_CHANNELS:
            raise KeyError(f"Unsupported signal in MindRove key '{key}'")

        mean, std = _get_mindrove_norm_stats(cfg, hand=hand, signal=signal)
        out[key] = _apply_mindrove_standardization_cf(x, mean=mean, std=std, key=key)

    if set(out.keys()) != set(streams.keys()):
        raise RuntimeError("Output keys changed after MindRove normalization")

    return out


def apply_mindrove_augmentation(
    streams: Dict[str, torch.Tensor],
    cfg: PackedMultiModalConfig,
) -> Dict[str, torch.Tensor]:
    """
    调用独立的 MindRove 增强模块。

    输入输出都保持 Dict[str, Tensor[C,L]]。
    要求输入已经是重采样后的 channel-first 格式。
    """
    return MindRoveAugmentation(streams, cfg)

def _merge_mindrove_hands(
    streams: Dict[str, torch.Tensor],
    cfg: PackedMultiModalConfig,
) -> Dict[str, torch.Tensor]:
    """
    将左右手同类信号在通道维拼接：
      left_emg + right_emg -> emg
      left_imu + right_imu -> imu

    约束
    ----
    cfg.mindrove_merge_hands=True 时：
    - 必须同时请求左右手
    - 对每个被请求的 signal，都必须同时拥有 left_* 和 right_*
    """
    if not cfg.mindrove_merge_hands:
        return streams

    out: Dict[str, torch.Tensor] = {}
    for signal in cfg.mindrove_signals:
        lk = f"left_{signal}"
        rk = f"right_{signal}"

        if lk not in streams or rk not in streams:
            raise KeyError(
                f"mindrove_merge_hands=True requires both '{lk}' and '{rk}' in streams, "
                f"but got keys: {sorted(streams.keys())}"
            )

        lv = streams[lk]
        rv = streams[rk]

        if lv.ndim != 2 or rv.ndim != 2:
            raise ValueError(f"Expected [C,L] tensors before merging, got {lv.shape} and {rv.shape}")
        if lv.shape[1] != rv.shape[1]:
            raise ValueError(f"Left/right {signal} lengths differ after resampling: {lv.shape} vs {rv.shape}")

        out[signal] = torch.cat([lv, rv], dim=0).contiguous()

    return out


# ============================================================
# 5) Dataset
# ============================================================

class PackedRGBDepthMindRoveMapDataset(Dataset):
    """
    基于 manifest 的 map-style Dataset。

    输出结构（在原 RGB / Depth 风格基础上新增 MindRove）：
    {
      "key": str,
      "sample_name": str,
      "tier_actions": {...},
      "tier_ids": {...},
      "lighting": str,
      "pos": str,
      "rgb": Tensor[T,3,H,W] 或 (view1, view2),
      "depth": Tensor[T,1,H,W],
      "mindrove": Dict[str, Tensor[C,L]] 或 (Dict, Dict)
    }
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        manifest_name: str,
        cfg: PackedMultiModalConfig,
        label_map: Optional[Dict[str, Dict[str, int]]] = None,
        verify_paths_on_init: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.manifest_path = self.dataset_root / manifest_name
        self.cfg = cfg
        
        # 非训练模式下，MindRove 只做读取与重采样，不做增强
        if "mindrove" in self.cfg.use_modalities and not self.cfg.is_train:
            self.cfg.mindrove_apply_augmentation = False

        _validate_config(self.cfg)

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.records = _load_manifest(self.manifest_path)

        if label_map is None:
            # 先尝试 cfg.label_map_path
            if self.cfg.label_map_path is not None:
                label_map_path = Path(self.cfg.label_map_path)
                if not label_map_path.is_absolute():
                    label_map_path = self.dataset_root / label_map_path
                if label_map_path.is_file():
                    self.label_map = load_label_map_json(label_map_path)
                else:
                    self.label_map = build_label_map_from_manifest(self.records)
            else:
                self.label_map = build_label_map_from_manifest(self.records)
        else:
            self.label_map = label_map

        if verify_paths_on_init:
            self.records = self._filter_valid_records(self.records)

        if len(self.records) == 0:
            raise RuntimeError("No valid records found after manifest/path filtering.")

    def _filter_valid_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        初始化阶段过滤掉路径缺失的样本，尽量贴近旧 loader 的 skip 行为。
        对 MindRove：
        - 路径不存在 -> 过滤
        - 若 missing_policy == "skip"，且请求的手在 manifest 中标记缺失 -> 过滤
        """
        valid = []
        for rec in records:
            ok = True

            if "rgb" in self.cfg.use_modalities:
                # 临时兼容 stage2 的脚本
                rgb_rel = rec.get("rgb", None)
                if rgb_rel is None:
                    rgb_rel = rec.get("001484412812_rgb", None)
                if rgb_rel is None or not (self.dataset_root / rgb_rel).is_file():
                    ok = False

            if "depth" in self.cfg.use_modalities:
                depth_rel = rec.get("depth", None)
                if depth_rel is None or not (self.dataset_root / depth_rel).is_file():
                    ok = False

            if "mindrove" in self.cfg.use_modalities:
                mr_rel = rec.get("mindrove", None)
                if mr_rel is None or not (self.dataset_root / mr_rel).is_file():
                    ok = False
                elif self.cfg.missing_policy == "skip":
                    if "left" in self.cfg.mindrove_hands and not bool(rec.get("mindrove_has_left", False)):
                        ok = False
                    if "right" in self.cfg.mindrove_hands and not bool(rec.get("mindrove_has_right", False)):
                        ok = False

            if ok:
                valid.append(rec)
        return valid

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, rec: Dict[str, Any]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        rgb_rel = rec.get("rgb", None)
        # 临时兼容 stage2的脚本
        if rgb_rel is None:
            rgb_rel = rec.get("001484412812_rgb", None)
        if rgb_rel is None:
            if self.cfg.missing_policy == "skip":
                raise FileNotFoundError(f"Record missing rgb path: {rec.get('sample_name', 'unknown')}")
            z = _make_zero_rgb(self.cfg)
            z = torch.as_tensor(self.cfg.rgb_transform(z)).contiguous()
            if self.cfg.rgb_two_views:
                return (z.clone(), z.clone())
            return z

        obj = torch.load(self.dataset_root / rgb_rel, map_location="cpu")
        video = obj["frames"] if isinstance(obj, dict) else obj

        # 期望 [T,3,H,W]
        if not torch.is_tensor(video) or video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(f"Invalid rgb tensor shape: {type(video)} / {getattr(video, 'shape', None)}")

        T = int(video.shape[0])

        if self.cfg.rgb_transform is None:
            raise RuntimeError(
                "cfg.rgb_transform is None. "
                "Please build dataset via build_packed_mapstyle_dataset(...) "
                "or loader via build_packed_mapstyle_loader(...)."
            )

        if self.cfg.rgb_two_views:
            idxs1, idxs2 = sample_two_views_indices(T=T, n=self.cfg.n_frames)
            v1 = video[idxs1]
            v2 = video[idxs2]
            v1 = torch.as_tensor(self.cfg.rgb_transform(v1)).contiguous()
            v2 = torch.as_tensor(self.cfg.rgb_transform(v2)).contiguous()
            return (v1, v2)

        idxs = sample_indices_strict(T, self.cfg.n_frames)
        v = video[idxs]
        v = torch.as_tensor(self.cfg.rgb_transform(v)).contiguous()
        return v

    def _load_depth(self, rec: Dict[str, Any]) -> torch.Tensor:
        depth_rel = rec.get("depth", None)
        if depth_rel is None:
            if self.cfg.missing_policy == "skip":
                raise FileNotFoundError(f"Record missing depth path: {rec.get('sample_name', 'unknown')}")
            z = _make_zero_depth(self.cfg)
            z = _resize_depth_video_keep_dtype(z, out_hw=self.cfg.depth_out_hw)
            return z

        obj = torch.load(self.dataset_root / depth_rel, map_location="cpu")
        video = obj["frames"] if isinstance(obj, dict) else obj

        if not torch.is_tensor(video) or video.ndim != 4:
            raise ValueError(f"Invalid depth tensor shape: {type(video)} / {getattr(video, 'shape', None)}")

        T = int(video.shape[0])
        idxs = sample_indices_strict(T, self.cfg.n_frames)
        v = video[idxs]

        if v.shape[1] == 1:
            v = _resize_depth_video_keep_dtype(v, out_hw=self.cfg.depth_out_hw)
        else:
            outs = []
            for c in range(v.shape[1]):
                vc = _resize_depth_video_keep_dtype(v[:, c:c + 1], out_hw=self.cfg.depth_out_hw)
                outs.append(vc)
            v = torch.cat(outs, dim=1).contiguous()

        return v

    def _load_single_mindrove_stream(
        self,
        mr_obj: Dict[str, Any],
        hand: str,
        signal: str,
        rec: Dict[str, Any],
    ) -> torch.Tensor:
        """
        加载单路 MindRove 信号，并按该 signal 的目标长度重采样，返回 [C,L_signal]
        """
        has_key = f"has_{hand}"
        data_key = f"{hand}_{signal}"

        hand_exists = bool(mr_obj.get(has_key, False))
        target_len = _get_mindrove_target_len(self.cfg, signal)

        if not hand_exists:
            if self.cfg.missing_policy == "skip":
                raise FileNotFoundError(
                    f"MindRove {hand} hand is missing for sample {rec.get('sample_name', 'unknown')}"
                )
            return _make_zero_mindrove_stream(signal=signal, target_len=target_len)

        if data_key not in mr_obj:
            if self.cfg.missing_policy == "skip":
                raise KeyError(f"MindRove key '{data_key}' not found in sample {rec.get('sample_name', 'unknown')}")
            return _make_zero_mindrove_stream(signal=signal, target_len=target_len)

        seq_lc = mr_obj[data_key]
        seq_cf = _resample_mindrove_lc_to_cf(
            seq_lc=seq_lc,
            signal=signal,
            target_len=target_len,
        )
        return seq_cf

    def _load_mindrove(
        self,
        rec: Dict[str, Any],
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        读取一个样本的 MindRove 数据。

        流程：
        1) 从文件加载 left/right + emg/imu
        2) 对每一条被请求的流按 signal 各自的目标长度重采样
        3) 若启用标准化：对重采样后的每一条流按左右手+模态各自的 mean/std 做标准化
        4) 若 single-view：输出 1 份（可选增强）
        5) 若 two-view：对同一个“已重采样且已标准化”的片段独立增强两次，输出 2 份
        6) 若启用 merge_hands：在标准化 / 增强后按 signal 合并左右手
        """
        mr_rel = rec.get("mindrove", None)
        if mr_rel is None:
            raise FileNotFoundError(f"Record missing mindrove path: {rec.get('sample_name', 'unknown')}")

        mr_path = self.dataset_root / mr_rel
        if not mr_path.is_file():
            raise FileNotFoundError(f"MindRove file not found: {mr_path}")

        mr_obj = torch.load(mr_path, map_location="cpu")
        if not isinstance(mr_obj, dict):
            raise TypeError(f"MindRove file must contain a dict, got {type(mr_obj)}")

        base_streams: Dict[str, torch.Tensor] = {}
        base_stream_is_real: Dict[str, bool] = {}

        for hand in self.cfg.mindrove_hands:
            for signal in self.cfg.mindrove_signals:
                key = f"{hand}_{signal}"

                has_key = f"has_{hand}"
                data_key = f"{hand}_{signal}"
                hand_exists = bool(mr_obj.get(has_key, False))
                stream_exists = hand_exists and (data_key in mr_obj)

                base_streams[key] = self._load_single_mindrove_stream(
                    mr_obj=mr_obj,
                    hand=hand,
                    signal=signal,
                    rec=rec,
                )
                base_stream_is_real[key] = bool(stream_exists)

        if self.cfg.mindrove_apply_normalization:
            base_streams = apply_mindrove_normalization(
                streams=base_streams,
                cfg=self.cfg,
                stream_is_real=base_stream_is_real,
            )

        if self.cfg.mindrove_two_views:
            view1 = apply_mindrove_augmentation(base_streams, self.cfg)
            view2 = apply_mindrove_augmentation(base_streams, self.cfg)

            if self.cfg.mindrove_merge_hands:
                view1 = _merge_mindrove_hands(view1, self.cfg)
                view2 = _merge_mindrove_hands(view2, self.cfg)

            return view1, view2

        out = apply_mindrove_augmentation(base_streams, self.cfg)
        if self.cfg.mindrove_merge_hands:
            out = _merge_mindrove_hands(out, self.cfg)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        out: Dict[str, Any] = {
            "key": rec.get("original_key", ""),
            "sample_name": rec.get("sample_name", ""),
            "lighting": rec.get("lighting", ""),
            "pos": rec.get("pos", ""),
            "global_index": int(idx),
            "idx": int(idx),
        }

        if self.cfg.load_labels:
            labels = {
                "tier1": rec.get("tier1", None),
                "tier2": rec.get("tier2", None),
                "tier3": rec.get("tier3", None),
            }
            tiers = get_required_tiers(self.cfg.tier_mode)
            tier_actions, tier_ids = map_tier_actions_to_ids(labels, self.label_map, tiers)
            out["tier_actions"] = tier_actions
            out["tier_ids"] = tier_ids

        if "rgb" in self.cfg.use_modalities:
            out["rgb"] = self._load_rgb(rec)

        if "depth" in self.cfg.use_modalities:
            out["depth"] = self._load_depth(rec)

        if "mindrove" in self.cfg.use_modalities:
            out["mindrove"] = self._load_mindrove(rec)

        return out


# ============================================================
# 6) collate_fn
# ============================================================

def _collate_nested(vals: List[Any]) -> Any:
    """
    递归 collate，支持：
    - Tensor
    - int / float / bool
    - str
    - dict
    - tuple / list
    """
    if len(vals) == 0:
        raise RuntimeError("Cannot collate an empty list.")

    first = vals[0]

    if torch.is_tensor(first) or isinstance(first, (int, float, bool)):
        return default_collate(vals)

    if isinstance(first, str):
        return list(vals)

    if isinstance(first, dict):
        keys = first.keys()
        return {k: _collate_nested([v[k] for v in vals]) for k in keys}

    if isinstance(first, tuple):
        return tuple(_collate_nested([v[i] for v in vals]) for i in range(len(first)))

    if isinstance(first, list):
        return [_collate_nested([v[i] for v in vals]) for i in range(len(first))]

    return list(vals)


def packed_multimodal_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 collate：
    - 保留 rgb two-view 为 (view1_batch, view2_batch)
    - 支持 mindrove single-view dict
    - 支持 mindrove two-view (dict1_batch, dict2_batch)
    """
    if len(batch) == 0:
        raise RuntimeError("Empty batch in collate.")

    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = _collate_nested([b[k] for b in batch])
    return out


# ============================================================
# 7) 构建 Dataset / DataLoader
# ============================================================

def build_packed_mapstyle_dataset(
    dataset_root: Union[str, Path],
    manifest_name: str,
    cfg: PackedMultiModalConfig,
    label_map: Optional[Dict[str, Dict[str, int]]] = None,
    verify_paths_on_init: bool = True,
) -> PackedRGBDepthMindRoveMapDataset:
    """
    构建 map-style Dataset。

    职责：
    1) 规范化配置
    2) 根据 train/val 挂载 RGB transform
    3) 返回 PackedRGBDepthMindRoveMapDataset
    """

    if "rgb" in cfg.use_modalities:
        if cfg.is_train:
            if cfg.rgb_apply_spatial_aug:
                hflip_p = cfg.rgb_hflip_p
                vflip_p = cfg.rgb_vflip_p
                jitter_p = cfg.rgb_jitter_p
                gray_p = cfg.rgb_gray_p
                blur_p = cfg.rgb_blur_p
            else:
                # 不新增无随机增强 transform。
                # 仍然使用 TemporallyConsistentSpatialAugmentation，
                # 但把 flip / jitter / gray / blur 的概率全部置 0。
                #
                # 注意：
                # 这里不会关闭 RandomResizedCrop。
                # 若要关闭 RandomResizedCrop，需要改 spatial_augmentation.py 的内部结构，
                # 或额外使用 ValidationAugmentation / resize-only transform。
                hflip_p = 0.0
                vflip_p = 0.0
                jitter_p = 0.0
                gray_p = 0.0
                blur_p = 0.0

            cfg.rgb_transform = TemporallyConsistentSpatialAugmentation(
                size=cfg.rgb_out_hw,
                crop_scale=cfg.rrc_scale,
                crop_ratio=cfg.rrc_ratio,

                flip_p=hflip_p,
                vflip_p=vflip_p,

                jitter_p=jitter_p,
                jitter_brightness=cfg.rgb_jitter_brightness,
                jitter_contrast=cfg.rgb_jitter_contrast,
                jitter_saturation=cfg.rgb_jitter_saturation,
                jitter_hue=cfg.rgb_jitter_hue,

                gray_p=gray_p,

                blur_p=blur_p,
                blur_kernel=cfg.rgb_blur_kernel,
                blur_sigma=cfg.rgb_blur_sigma,

                mean=cfg.rgb_mean,
                std=cfg.rgb_std,
            )
        else:
            cfg.rgb_transform = ValidationAugmentation(
                size=cfg.rgb_out_hw,
                mean=cfg.rgb_mean,
                std=cfg.rgb_std,
            )
        
    _validate_config(cfg)

    ds = PackedRGBDepthMindRoveMapDataset(
        dataset_root=dataset_root,
        manifest_name=manifest_name,
        cfg=cfg,
        label_map=label_map,
        verify_paths_on_init=verify_paths_on_init,
    )
    return ds


def build_packed_mapstyle_loader_from_dataset(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = None,
    sampler=None,
    pin_memory: bool = False,
) -> DataLoader:
    """
    把已构建好的 dataset 包装成 DataLoader。

    说明：
    - 若 sampler 不为 None（例如 DistributedSampler），则必须关闭 shuffle
    - collate_fn 固定使用 packed_multimodal_collate
    - worker_init_fn 会把 torch worker seed 同步给 numpy / random，
      以保证 MindRove drift 等使用 np.random 的增强在多 worker 下随机性正确
    """
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
        collate_fn=packed_multimodal_collate,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**loader_kwargs)
    return loader


def build_packed_mapstyle_loader(
    dataset_root: Union[str, Path],
    manifest_name: str,
    cfg: PackedMultiModalConfig,
    label_map: Optional[Dict[str, Dict[str, int]]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = None,
    sampler=None,
    verify_paths_on_init: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """
    兼容旧接口的薄封装。
    """
    ds = build_packed_mapstyle_dataset(
        dataset_root=dataset_root,
        manifest_name=manifest_name,
        cfg=cfg,
        label_map=label_map,
        verify_paths_on_init=verify_paths_on_init,
    )

    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        sampler=sampler,
        pin_memory=pin_memory,
    )
    return loader


# ============================================================
# 8) Weighted sampler
# ============================================================

def build_weighted_sampler_for_packed_dataset(
    dataset: PackedRGBDepthMindRoveMapDataset,
    tier_for_sampling: Optional[str] = None,
    mode: str = "sqrt_inv",
    replacement: bool = True,
    num_samples: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[WeightedRandomSampler, Dict[str, Any]]:
    """
    为 PackedRGBDepthMindRoveMapDataset 构建 WeightedRandomSampler。
    这里只使用 dataset.records 中的标签信息，不会真的加载视频或 MindRove 文件。
    """
    if not hasattr(dataset, "records"):
        raise TypeError("dataset must have attribute 'records'.")

    if not hasattr(dataset, "label_map"):
        raise TypeError("dataset must have attribute 'label_map'.")

    if not hasattr(dataset, "cfg"):
        raise TypeError("dataset must have attribute 'cfg'.")

    if tier_for_sampling is None:
        if dataset.cfg.tier_mode in ("tier1", "tier2", "tier3"):
            tier_for_sampling = dataset.cfg.tier_mode
        else:
            tier_for_sampling = "tier3"

    if tier_for_sampling not in ("tier1", "tier2", "tier3"):
        raise ValueError(
            f"tier_for_sampling must be one of ('tier1','tier2','tier3'), got {tier_for_sampling}"
        )

    if tier_for_sampling not in dataset.label_map:
        raise KeyError(f"dataset.label_map does not contain key '{tier_for_sampling}'")

    tier_label_map = dataset.label_map[tier_for_sampling]

    labels = []
    bad_indices = []

    for i, rec in enumerate(dataset.records):
        action_name = rec.get(tier_for_sampling, None)

        if action_name is None:
            bad_indices.append(i)
            labels.append(-1)
            continue

        action_name = str(action_name)
        class_id = tier_label_map.get(action_name, -1)

        if class_id < 0:
            bad_indices.append(i)

        labels.append(int(class_id))

    if len(bad_indices) > 0:
        raise ValueError(
            f"Found {len(bad_indices)} samples with invalid label for {tier_for_sampling}. "
            f"Example bad indices: {bad_indices[:10]}"
        )

    labels = torch.as_tensor(labels, dtype=torch.long)
    counter = Counter(labels.tolist())

    if len(counter) == 0:
        raise RuntimeError("No valid labels found for weighted sampling.")

    max_class_id = max(counter.keys())
    class_counts_tensor = torch.zeros(max_class_id + 1, dtype=torch.long)

    for cls_id, cnt in counter.items():
        class_counts_tensor[cls_id] = int(cnt)

    class_weights = torch.zeros_like(class_counts_tensor, dtype=torch.double)

    for cls_id, cnt in counter.items():
        if cnt <= 0:
            raise ValueError(f"Invalid class count for class {cls_id}: {cnt}")

        if mode == "inv":
            w = 1.0 / float(cnt)
        elif mode == "sqrt_inv":
            w = 1.0 / (float(cnt) ** 0.5)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'inv' or 'sqrt_inv'.")

        class_weights[cls_id] = w

    sample_weights = class_weights[labels].to(torch.double)

    if num_samples is None:
        num_samples = len(dataset)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement,
    )

    info = {
        "tier_for_sampling": tier_for_sampling,
        "labels": labels,
        "class_counts": counter,
        "class_weights": class_weights,
        "sample_weights": sample_weights,
        "num_samples": num_samples,
    }

    if verbose:
        print(f"[WeightedSampler] tier_for_sampling = {tier_for_sampling}")
        print(f"[WeightedSampler] mode = {mode}")
        print(f"[WeightedSampler] replacement = {replacement}")
        print(f"[WeightedSampler] num_samples = {num_samples}")
        print(f"[WeightedSampler] num_classes_in_split = {len(counter)}")
        print(f"[WeightedSampler] class_counts = {dict(sorted(counter.items()))}")

    return sampler, info


# ============================================================
# 9) 简单自测入口
# ============================================================

def _print_mindrove_shape_info(mr: Any) -> None:
    if isinstance(mr, tuple) and len(mr) == 2:
        print("  mindrove view1:")
        for k, v in mr[0].items():
            print(f"    {k}: {tuple(v.shape)} {v.dtype}")
        print("  mindrove view2:")
        for k, v in mr[1].items():
            print(f"    {k}: {tuple(v.shape)} {v.dtype}")
        return

    if isinstance(mr, dict):
        print("  mindrove:")
        for k, v in mr.items():
            print(f"    {k}: {tuple(v.shape)} {v.dtype}")
        return

    raise TypeError(f"Unexpected mindrove batch structure: {type(mr)}")


##########################################
# 可视化原始信号和标准化后信号
##########################################
def visualize_mindrove_original_vs_normalized(
    mindrove_pt_path: Union[str, Path],
    target_len: int,
    hands: Tuple[str, ...],
    signals: Tuple[str, ...],
    left_emg_mean: Optional[Tuple[float, ...]] = None,
    left_emg_std: Optional[Tuple[float, ...]] = None,
    right_emg_mean: Optional[Tuple[float, ...]] = None,
    right_emg_std: Optional[Tuple[float, ...]] = None,
    left_imu_mean: Optional[Tuple[float, ...]] = None,
    left_imu_std: Optional[Tuple[float, ...]] = None,
    right_imu_mean: Optional[Tuple[float, ...]] = None,
    right_imu_std: Optional[Tuple[float, ...]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    读取单个 mindrove.pt，按当前 dataloader 的逻辑先重采样，再做标准化，
    然后可视化 original 与 normalized 的对比。

    参数
    ----
    mindrove_pt_path:
        单个样本的 mindrove.pt 路径
    target_len:
        重采样目标长度
    hands:
        例如 ("left", "right")
    signals:
        例如 ("emg",) 或 ("imu",) 或 ("emg", "imu")

    其余 mean/std:
        与 dataloader 中的 PackedMultiModalConfig 完全同名同语义。
        若某个 hand+signal 被请求，则必须提供对应 mean/std。

    save_path:
        若不为 None，则保存图片
    show:
        是否弹窗显示
    """
    import matplotlib.pyplot as plt

    mindrove_pt_path = Path(mindrove_pt_path)
    if not mindrove_pt_path.is_file():
        raise FileNotFoundError(f"mindrove.pt not found: {mindrove_pt_path}")

    # --------------------------------------------------------
    # 1) 构造一个仅用于测试 normalization 的 cfg
    #    不做增强，只做 normalization
    # --------------------------------------------------------
    cfg = PackedMultiModalConfig(
        use_modalities=("mindrove",),
        missing_policy="skip",
        is_train=False,
        mindrove_two_views=False,
        mindrove_target_len=target_len,
        mindrove_hands=hands,
        mindrove_signals=signals,
        mindrove_merge_hands=False,
        mindrove_apply_augmentation=False,
        mindrove_apply_normalization=True,

        mindrove_left_emg_mean=left_emg_mean,
        mindrove_left_emg_std=left_emg_std,
        mindrove_right_emg_mean=right_emg_mean,
        mindrove_right_emg_std=right_emg_std,

        mindrove_left_imu_mean=left_imu_mean,
        mindrove_left_imu_std=left_imu_std,
        mindrove_right_imu_mean=right_imu_mean,
        mindrove_right_imu_std=right_imu_std,
    )

    # 触发与训练时一致的严格校验
    _validate_config(cfg)

    # --------------------------------------------------------
    # 2) 读取 mindrove.pt，并按 dataloader 的逻辑重采样
    # --------------------------------------------------------
    mr_obj = torch.load(mindrove_pt_path, map_location="cpu")
    if not isinstance(mr_obj, dict):
        raise TypeError(f"mindrove.pt must contain a dict, got {type(mr_obj)}")

    original_streams: Dict[str, torch.Tensor] = {}
    stream_is_real: Dict[str, bool] = {}

    sample_name = mindrove_pt_path.stem

    for hand in hands:
        for signal in signals:
            key = f"{hand}_{signal}"
            has_key = f"has_{hand}"
            data_key = f"{hand}_{signal}"

            hand_exists = bool(mr_obj.get(has_key, False))
            stream_exists = hand_exists and (data_key in mr_obj)

            if not stream_exists:
                raise KeyError(
                    f"Requested stream '{key}' is missing in {mindrove_pt_path}. "
                    f"has_{hand}={hand_exists}, key_exists={data_key in mr_obj}"
                )

            seq_lc = mr_obj[data_key]
            seq_cf = _resample_mindrove_lc_to_cf(
                seq_lc=seq_lc,
                signal=signal,
                target_len=target_len,
            )

            original_streams[key] = seq_cf
            stream_is_real[key] = True

    # --------------------------------------------------------
    # 3) 调用脚本现有的 normalization 逻辑
    # --------------------------------------------------------
    normalized_streams = apply_mindrove_normalization(
        streams=original_streams,
        cfg=cfg,
        stream_is_real=stream_is_real,
    )

    # --------------------------------------------------------
    # 4) 画图：每一路 3 列
    #    原始信号 / 标准化后信号 / 标准化后统计量
    # --------------------------------------------------------
    keys = list(original_streams.keys())
    n_rows = len(keys)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(18, 4.5 * n_rows),
        squeeze=False
    )

    def _plot_stacked_channels(ax, x_cf: torch.Tensor, title: str):
        """
        将 [C,L] 多通道信号按通道错开绘制，便于观察每个通道形状。
        """
        x_cf = x_cf.detach().cpu().float()
        C, L = x_cf.shape
        t = torch.arange(L).numpy()

        # gap 用于将各通道上下错开，避免重叠
        amp = max(float(x_cf.abs().max().item()), 1e-6)
        gap = amp * 2.5

        yticks = []
        yticklabels = []

        for c in range(C):
            y = x_cf[c].numpy() + c * gap
            ax.plot(t, y, linewidth=1.0)
            yticks.append(c * gap)
            yticklabels.append(f"ch{c}")

        ax.set_title(title)
        ax.set_xlabel("time index")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.grid(True, alpha=0.25)

    for row, key in enumerate(keys):
        x0 = original_streams[key]
        x1 = normalized_streams[key]

        # 左：原始信号
        _plot_stacked_channels(
            axes[row, 0],
            x0,
            f"{key} | original (resampled)"
        )

        # 中：标准化后信号
        _plot_stacked_channels(
            axes[row, 1],
            x1,
            f"{key} | normalized"
        )

        # 右：打印标准化前后每通道 mean/std
        orig_mean = x0.mean(dim=1).detach().cpu().numpy()
        orig_std = x0.std(dim=1).detach().cpu().numpy()
        norm_mean = x1.mean(dim=1).detach().cpu().numpy()
        norm_std = x1.std(dim=1).detach().cpu().numpy()

        ax_txt = axes[row, 2]
        ax_txt.axis("off")

        lines = [f"{key} channel statistics", ""]
        lines.append("Original:")
        for c in range(x0.shape[0]):
            lines.append(
                f"  ch{c}: mean={orig_mean[c]:.4f}, std={orig_std[c]:.4f}"
            )

        lines.append("")
        lines.append("Normalized:")
        for c in range(x1.shape[0]):
            lines.append(
                f"  ch{c}: mean={norm_mean[c]:.4f}, std={norm_std[c]:.4f}"
            )

        ax_txt.text(
            0.0, 1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

    fig.suptitle(
        f"MindRove original vs normalized | sample={sample_name}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Saved figure] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main_visualize_normalization():
    """
    单独测试：
    读取一个 mindrove.pt，
    可视化 original(resampled) 和 normalized 之后的信号。
    """
    mindrove_pt_path = r"C:\Junxi_data_for_training_speedup\mapstyle_dataset\train\sample_0565\mindrove.pt"

    visualize_mindrove_original_vs_normalized(
        mindrove_pt_path=mindrove_pt_path,
        target_len=512,
        hands=("left", "right"),
        signals=("imu",),

        # ---------------- 这里直接传你想测试的 normalization 参数 ----------------
        left_emg_mean=(0.0006, -0.0027, 0.0051, 0.0063, 0.0055, -0.0019, 0.0009, 0.0029),
        left_emg_std=(20.0286, 26.6126, 29.4076, 32.6272, 29.6518, 26.9408, 16.0612, 13.9617),

        right_emg_mean=(0.0056, -0.0043, -0.0147, -0.0193, -0.0166, 0.0023, 0.0032, 0.0055),
        right_emg_std=(21.9245, 30.5936, 34.8250, 41.6747, 38.2362, 29.3167, 17.7020, 15.3456),

        # 如果这次只看 EMG，IMU 可以不传
        left_imu_mean=(-0.0301, -0.0033, -0.0366, -0.2910, -0.0086, 2.9009),
        left_imu_std=(0.2028, 0.1693, 0.1153, 42.9318, 42.8149, 48.6560),
        right_imu_mean=(0.0275, -0.0127, -0.0457, 1.7059, -0.7678, -6.7844),
        right_imu_std=(0.2379, 0.2609, 0.1471, 63.1509, 66.9620, 69.8072),

        save_path=r"./debug_vis/original_vs_normalized_emg.png",
        show=True,
    )

def main():
    """
    你可以自行修改路径进行简单自测。
    """
    dataset_root = r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset"

    cfg = PackedMultiModalConfig(
        n_frames=16,
        use_modalities=("rgb",),
        missing_policy="skip",
        tier_mode="all",
        is_train=True,
        rgb_out_hw=(224, 224),
        depth_out_hw=(224, 224),
        rgb_two_views=True,
        label_map_path=r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\label_map.json",

        # ---------------- MindRove 示例 ----------------
        mindrove_two_views=True,
        mindrove_target_len=256,
        mindrove_hands=("left", "right"),
        mindrove_signals=("emg",),
        mindrove_merge_hands=True,
        mindrove_apply_augmentation=False,   # 当前占位实现为 identity
    )

    dataset = build_packed_mapstyle_dataset(
        dataset_root=dataset_root,
        manifest_name=r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\D_test\test_manifest.jsonl",
        cfg=cfg,
    )

    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )

    for i, batch in enumerate(loader):
        print(f"\n[batch {i}] keys = {list(batch.keys())}")
        print("  sample_name:", batch["sample_name"][:])

        if "tier_ids" in batch:
            print("  tier_ids:", batch["tier_ids"])

        if "rgb" in batch:
            rgb = batch["rgb"]
            if isinstance(rgb, tuple) and len(rgb) == 2:
                print("  rgb.view1:", tuple(rgb[0].shape), rgb[0].dtype)
                print("  rgb.view2:", tuple(rgb[1].shape), rgb[1].dtype)
            else:
                print("  rgb:", tuple(rgb.shape), rgb.dtype)

        if "depth" in batch:
            print("  depth:", tuple(batch["depth"].shape), batch["depth"].dtype)

        if "mindrove" in batch:
            _print_mindrove_shape_info(batch["mindrove"])

        break


if __name__ == "__main__":
    main()
