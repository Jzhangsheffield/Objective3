#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mindrove_augmentation_tensor.py

针对 MindRove EMG / IMU 数据的张量原生增强模块。

特点
====
1) 直接在 torch.Tensor 上做增强，不依赖 transformations.py 的 numpy/scipy 接口。
2) 主增强入口仍然保持与 dataloader 兼容：
      apply_mindrove_augmentation(streams, cfg) -> Dict[str, Tensor[C,L]]
3) 支持直接从单个 mindrove.pt 文件加载、重采样、增强、可视化。
4) 当前支持的增强包括：
   - per-hand shared time warp
   - per-stream drift（尽量对齐 tsaug.Drift，使用 scipy.interpolate.CubicSpline）
   - per-stream scaling / noise
   - per-stream negate
   - per-stream channel dropout
5) 同一只手内，EMG 和 IMU 尽量共享同一个 time-warp profile；
   若 EMG / IMU 长度不同，则共享同一组 knot_speed，并分别插值到各自长度。左右手各自独立采样。

说明
====
1) time warp 继续保留当前 torch-native 实现：
   - 先在 knot 上采样一条正值“速度曲线”
   - 再用 torch 的线性插值扩展到整段长度
   - 用累计和构造单调递增的 warped timestamps
   - 最后用 torch 实现的一维线性采样对每个通道重采样
2) drift 为了尽量与 tsaug.Drift 一致，改为直接使用 scipy.interpolate.CubicSpline
   生成平滑漂移曲线；对单条 [C,L] 流做增强时，会严格按照配置决定：
   - max_drift
   - n_drift_points
   - kind(additive / multiplicative)
   - per_channel
   - normalize
3) 本文件对配置不做兜底：类型或取值不合法会直接报错。

命令行示例
==========
1) 可视化一个样本：
   python mindrove_augmentation_tensor.py \
       --mindrove_pt /path/to/sample_mindrove.pt \
       --target_len 256 \
       --seed 42

2) 关闭 time warp，只看 scaling + noise：
   python mindrove_augmentation_tensor.py \
       --mindrove_pt /path/to/sample_mindrove.pt \
       --target_len 256 \
       --mindrove_time_warp_prob 0.0

3) 保存可视化结果而不弹窗：
   python mindrove_augmentation_tensor.py \
       --mindrove_pt /path/to/sample_mindrove.pt \
       --save_path /path/to/aug_vis.png \
       --no_show
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline


# ============================================================
# 0) 常量与配置
# ============================================================

EXPECTED_CHANNELS: Dict[str, int] = {
    "emg": 8,
    "imu": 6,
}


@dataclass
class MindRoveAugConfig:
    """
    独立增强脚本使用的最小配置。
    字段名尽量与 dataloader 中准备新增的字段保持一致，便于直接替换调用。
    """

    mindrove_apply_augmentation: bool = True

    mindrove_time_warp_prob: float = 0.8
    mindrove_time_warp_sigma: float = 0.2
    mindrove_time_warp_num_knots: int = 4
    mindrove_time_warp_num_splines: int = 150  # 保留同名字段，当前 tensor 实现不显式用 bank

    mindrove_emg_scaling_prob: float = 0.8
    mindrove_emg_scaling_sigma: float = 0.1
    mindrove_emg_noise_prob: float = 0.8
    mindrove_emg_noise_sigma: float = 0.05

    mindrove_imu_scaling_prob: float = 0.8
    mindrove_imu_scaling_sigma: float = 0.1
    mindrove_imu_noise_prob: float = 0.8
    mindrove_imu_noise_sigma: float = 0.05

    # Drift：尽量对齐 tsaug.Drift 的参数语义
    mindrove_emg_drift_prob: float = 0.0
    mindrove_emg_drift_max: Union[float, Tuple[float, float]] = 0.0
    mindrove_emg_drift_n_points: Union[int, List[int]] = 3
    mindrove_emg_drift_kind: str = "additive"
    mindrove_emg_drift_per_channel: bool = False
    mindrove_emg_drift_normalize: bool = True

    mindrove_imu_drift_prob: float = 0.0
    mindrove_imu_drift_max: Union[float, Tuple[float, float]] = 0.0
    mindrove_imu_drift_n_points: Union[int, List[int]] = 3
    mindrove_imu_drift_kind: str = "additive"
    mindrove_imu_drift_per_channel: bool = False
    mindrove_imu_drift_normalize: bool = False

    # 随机符号翻转（negated）：对整条流乘以 -1
    mindrove_emg_negate_prob: float = 0.0
    mindrove_imu_negate_prob: float = 0.0

    # 随机通道置零（channel dropout）：每次随机选择若干通道整段置 0
    # 说明：
    # - prob 决定是否触发该增强
    # - max_channels 决定一次最多丢弃多少个通道
    mindrove_emg_channel_dropout_prob: float = 0.0
    mindrove_emg_channel_dropout_max_channels: int = 1
    mindrove_imu_channel_dropout_prob: float = 0.0
    mindrove_imu_channel_dropout_max_channels: int = 1


# ============================================================
# 1) 严格工具函数
# ============================================================


def _require_cfg_attr(cfg: Any, name: str) -> Any:
    if not hasattr(cfg, name):
        raise AttributeError(f"Config is missing required attribute: {name}")
    return getattr(cfg, name)



def _ensure_tensor_cf(x: torch.Tensor, key: str) -> torch.Tensor:
    """
    严格要求单条流为 [C,L]。
    """
    if not torch.is_tensor(x):
        raise TypeError(f"MindRove stream '{key}' must be a torch.Tensor, got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"MindRove stream '{key}' must be [C,L], got shape {tuple(x.shape)}")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError(f"MindRove stream '{key}' must have positive shape, got {tuple(x.shape)}")
    return x.to(torch.float32).contiguous()



def _ensure_prob(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be int or float, got {type(value)}")
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {value}")
    return value



def _ensure_nonneg(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be int or float, got {type(value)}")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value



def _ensure_pos_int(name: str, value: int) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value)}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _ensure_drift_max(name: str, value: Union[float, Tuple[float, float]]) -> Union[float, Tuple[float, float]]:
    if isinstance(value, (int, float)):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")
        return value
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} tuple must have length 2, got {value}")
        lo, hi = value
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            raise TypeError(f"{name} tuple must contain numbers, got {value}")
        lo = float(lo)
        hi = float(hi)
        if lo < 0.0 or hi < 0.0 or lo > hi:
            raise ValueError(f"{name} must satisfy 0 <= low <= high, got {value}")
        return (lo, hi)
    raise TypeError(f"{name} must be a non-negative float or a 2-tuple of non-negative floats, got {type(value)}")


def _ensure_drift_points(name: str, value: Union[int, List[int]]) -> Union[int, List[int]]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(f"{name} list cannot be empty")
        if not all(isinstance(v, int) for v in value):
            raise TypeError(f"{name} list must contain ints, got {value}")
        if not all(v > 0 for v in value):
            raise ValueError(f"{name} list must contain positive ints, got {value}")
        return value
    raise TypeError(f"{name} must be a positive int or a list of positive ints, got {type(value)}")


def _ensure_drift_kind(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value)}")
    value = value.strip().lower()
    if value not in ("additive", "multiplicative"):
        raise ValueError(f"{name} must be 'additive' or 'multiplicative', got {value}")
    return value


def _ensure_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value)}")
    return value


def _parse_literal_arg(name: str, raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception as e:
        raise ValueError(f"Failed to parse argument {name}={raw!r} with ast.literal_eval") from e


def _should_apply(prob: float) -> bool:
    prob = _ensure_prob("prob", prob)
    return bool(torch.rand((), dtype=torch.float32).item() < prob)


# ============================================================
# 2) 张量原生 time warp
# ============================================================


def sample_shared_time_warp_timestamps_torch(
    length: int,
    sigma: float = 0.2,
    num_knots: int = 4,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    knot_speed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    生成一条 warped timestamps，shape=[L]。

    修正点
    ----
    1) 保证时间轴从 0 开始，而不是从一个正值开始
    2) 保证最后一个点落在 L-1
    3) 当 sigma=0 时，结果应尽量接近 identity: [0, 1, 2, ..., L-1]
    """
    length = _ensure_pos_int("length", length)
    sigma = _ensure_nonneg("sigma", sigma)
    if not isinstance(num_knots, int) or num_knots < 0:
        raise ValueError(f"num_knots must be >= 0 int, got {num_knots}")

    if device is None:
        device = torch.device("cpu")

    num_points = num_knots + 2

    # ------------------------------------------------------------
    # 可选共享 knot_speed
    # ------------------------------------------------------------
    # 原始版本在每个 length 上独立采样 knot_speed。
    # 为了支持 EMG / IMU 使用不同长度，同时尽量保留“同一只手共享同一个
    # time-warp 形状”的语义，可以在外部为同一只手采样一次 knot_speed，
    # 然后分别插值到 EMG 长度和 IMU 长度。
    if knot_speed is None:
        if sigma == 0.0:
            knot_speed = torch.ones((1, 1, num_points), device=device, dtype=dtype)
        else:
            knot_speed = torch.exp(
                torch.randn((1, 1, num_points), device=device, dtype=dtype) * sigma
            )
    else:
        if not torch.is_tensor(knot_speed):
            raise TypeError(f"knot_speed must be tensor or None, got {type(knot_speed)}")
        if knot_speed.ndim != 3 or knot_speed.shape[0] != 1 or knot_speed.shape[1] != 1:
            raise ValueError(
                f"knot_speed must have shape [1,1,num_points], got {tuple(knot_speed.shape)}"
            )
        if knot_speed.shape[2] != num_points:
            raise ValueError(
                f"knot_speed length {knot_speed.shape[2]} does not match num_knots+2={num_points}"
            )
        knot_speed = knot_speed.to(device=device, dtype=dtype)

    speed = F.interpolate(
        knot_speed,
        size=length,
        mode="linear",
        align_corners=True,
    ).view(length)

    cumulative = torch.cumsum(speed, dim=0)

    # 关键修正：把累计曲线平移到从 0 开始
    cumulative = cumulative - cumulative[0]

    # 防止极端情况下分母为 0
    denom = cumulative[-1].clamp_min(torch.finfo(cumulative.dtype).eps)

    timestamps = cumulative / denom * float(length - 1)

    # 数值上再保证首尾严格落到边界
    timestamps[0] = 0.0
    timestamps[-1] = float(length - 1)

    return timestamps.contiguous()



def apply_time_warp_cf_with_shared_timestamps(
    x_cf: torch.Tensor,
    distorted_timestamps: torch.Tensor,
) -> torch.Tensor:
    """
    对单条 [C,L] 流应用给定 warped timestamps。
    所有通道共享同一条 distorted_timestamps。
    """
    x_cf = _ensure_tensor_cf(x_cf, key="<time_warp_input>")
    if not torch.is_tensor(distorted_timestamps):
        raise TypeError(f"distorted_timestamps must be tensor, got {type(distorted_timestamps)}")
    if distorted_timestamps.ndim != 1:
        raise ValueError(f"distorted_timestamps must be 1D, got shape {tuple(distorted_timestamps.shape)}")
    if distorted_timestamps.shape[0] != x_cf.shape[1]:
        raise ValueError(
            f"distorted_timestamps length {distorted_timestamps.shape[0]} does not match signal length {x_cf.shape[1]}"
        )

    x = x_cf
    ts = distorted_timestamps.to(device=x.device, dtype=x.dtype).clamp(0, x.shape[1] - 1)

    idx0 = torch.floor(ts).to(torch.long)
    idx1 = torch.clamp(idx0 + 1, max=x.shape[1] - 1)
    w = (ts - idx0.to(ts.dtype)).unsqueeze(0)  # [1,L]

    gather0 = idx0.unsqueeze(0).expand(x.shape[0], -1)
    gather1 = idx1.unsqueeze(0).expand(x.shape[0], -1)

    x0 = torch.gather(x, dim=1, index=gather0)
    x1 = torch.gather(x, dim=1, index=gather1)
    out = x0 * (1.0 - w) + x1 * w
    return out.contiguous()


# ============================================================
# 3) 单条流增强：scaling / noise（纯 torch）
# ============================================================


def apply_scaling_cf(
    x_cf: torch.Tensor,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对单条 [C,L] 流做 scaling。
    语义保持为：每个通道一个缩放因子，沿时间维共享。

    返回
    ----
    y_cf : Tensor[C,L]
    factors : Tensor[C]
    """
    x_cf = _ensure_tensor_cf(x_cf, key="<scaling_input>")
    sigma = _ensure_nonneg("scaling sigma", sigma)

    if sigma == 0.0:
        factors = torch.ones((x_cf.shape[0],), device=x_cf.device, dtype=x_cf.dtype)
    else:
        factors = 1.0 + torch.randn((x_cf.shape[0],), device=x_cf.device, dtype=x_cf.dtype) * sigma
    y = x_cf * factors.unsqueeze(1)
    return y.contiguous(), factors.contiguous()



def apply_noise_cf(
    x_cf: torch.Tensor,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对单条 [C,L] 流加逐元素高斯噪声。

    返回
    ----
    y_cf : Tensor[C,L]
    noise : Tensor[C,L]
    """
    x_cf = _ensure_tensor_cf(x_cf, key="<noise_input>")
    sigma = _ensure_nonneg("noise sigma", sigma)

    if sigma == 0.0:
        noise = torch.zeros_like(x_cf)
    else:
        noise = torch.randn_like(x_cf) * sigma
    y = x_cf + noise
    return y.contiguous(), noise.contiguous()


def apply_drift_cf(
    x_cf: torch.Tensor,
    max_drift: Union[float, Tuple[float, float]],
    n_drift_points: Union[int, List[int]],
    kind: str = "additive",
    per_channel: bool = False,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对单条 [C,L] 流做 drift 增强。

    设计目标：尽量贴近 tsaug.Drift 的语义，并显式使用 scipy.interpolate.CubicSpline
    生成平滑漂移曲线。

    参数
    ----
    x_cf : Tensor[C,L]
    max_drift : float 或 (low, high)
    n_drift_points : int 或 List[int]
    kind : 'additive' 或 'multiplicative'
    per_channel : 是否每个通道独立采样一条 drift 曲线
    normalize : additive 模式下是否按每个通道的动态范围缩放 drift

    返回
    ----
    y_cf : Tensor[C,L]
    drift_cf : Tensor[C,L]
    """
    x_cf = _ensure_tensor_cf(x_cf, key="<drift_input>")
    max_drift = _ensure_drift_max("drift max_drift", max_drift)
    n_drift_points = _ensure_drift_points("drift n_drift_points", n_drift_points)
    kind = _ensure_drift_kind("drift kind", kind)
    per_channel = _ensure_bool("drift per_channel", per_channel)
    normalize = _ensure_bool("drift normalize", normalize)

    C, L = x_cf.shape
    n_series = C if per_channel else 1

    if isinstance(n_drift_points, int):
        n_choices = [n_drift_points]
    else:
        n_choices = sorted(set(n_drift_points))

    drift_tc = np.zeros((n_series, L), dtype=np.float32)
    choice_idx = np.random.choice(len(n_choices), size=n_series, replace=True)

    x_eval = np.arange(L, dtype=np.float64)
    for i, n in enumerate(n_choices):
        mask = (choice_idx == i)
        count = int(mask.sum())
        if count == 0:
            continue
        anchors = np.cumsum(np.random.normal(size=(count, n + 2)).astype(np.float64), axis=1)
        knot_x = np.linspace(0, L, n + 2, dtype=np.float64)
        spline = CubicSpline(knot_x, anchors, axis=1)
        drift_tc[mask, :] = spline(x_eval).astype(np.float32)

    drift_tc = drift_tc - drift_tc[:, :1]
    denom = np.abs(drift_tc).max(axis=1, keepdims=True)
    if np.any(denom <= 0.0):
        raise RuntimeError("Encountered non-positive drift normalization denominator.")
    drift_tc = drift_tc / denom

    if isinstance(max_drift, tuple):
        lo, hi = max_drift
        scale = np.random.uniform(low=lo, high=hi, size=(n_series, 1)).astype(np.float32)
        drift_tc = drift_tc * scale
    else:
        drift_tc = drift_tc * float(max_drift)

    if not per_channel:
        drift_tc = np.repeat(drift_tc, C, axis=0)

    drift_cf = torch.from_numpy(drift_tc).to(device=x_cf.device, dtype=x_cf.dtype)

    if kind == "additive":
        if normalize:
            value_range = x_cf.max(dim=1, keepdim=True).values - x_cf.min(dim=1, keepdim=True).values
            y = x_cf + drift_cf * value_range
        else:
            y = x_cf + drift_cf
    else:
        y = x_cf * (1.0 + drift_cf)

    return y.contiguous(), drift_cf.contiguous()


def apply_negate_cf(
    x_cf: torch.Tensor,
) -> torch.Tensor:
    """
    对单条 [C,L] 流做整段符号翻转。
    """
    x_cf = _ensure_tensor_cf(x_cf, key="<negate_input>")
    return (-x_cf).contiguous()


def apply_channel_dropout_cf(
    x_cf: torch.Tensor,
    max_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    随机选择若干个通道，并将这些通道整段时间序列置 0。

    参数
    ----
    x_cf : Tensor[C,L]
    max_channels : int
        一次最多丢弃多少个通道，实际会从 [1, min(max_channels, C)] 中
        均匀随机采样一个整数作为本次置零通道数。

    返回
    ----
    y_cf : Tensor[C,L]
    dropped_mask : Tensor[C] (bool)
        True 表示该通道本次被整段置 0。
    """
    x_cf = _ensure_tensor_cf(x_cf, key="<channel_dropout_input>")
    if not isinstance(max_channels, int):
        raise TypeError(f"channel dropout max_channels must be int, got {type(max_channels)}")
    if max_channels <= 0:
        raise ValueError(f"channel dropout max_channels must be > 0, got {max_channels}")

    C = x_cf.shape[0]
    n_drop_upper = min(max_channels, C)
    n_drop = int(torch.randint(1, n_drop_upper + 1, (1,), device=x_cf.device).item())

    perm = torch.randperm(C, device=x_cf.device)
    drop_idx = perm[:n_drop]

    y = x_cf.clone()
    y[drop_idx, :] = 0.0

    dropped_mask = torch.zeros((C,), dtype=torch.bool, device=x_cf.device)
    dropped_mask[drop_idx] = True
    return y.contiguous(), dropped_mask.contiguous()


# ============================================================
# 4) 主入口：样本级 MindRove 增强
# ============================================================


def apply_mindrove_augmentation_with_info(
    streams: Dict[str, torch.Tensor],
    cfg: Any,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    与 apply_mindrove_augmentation 类似，但额外返回调试信息，
    便于 main 可视化和检查增强是否正确。
    """
    if not isinstance(streams, dict):
        raise TypeError(f"streams must be dict[str, Tensor], got {type(streams)}")
    if len(streams) == 0:
        raise ValueError("streams cannot be empty")

    enabled = bool(_require_cfg_attr(cfg, "mindrove_apply_augmentation"))

    validated: Dict[str, torch.Tensor] = {
        k: _ensure_tensor_cf(v, key=k)
        for k, v in streams.items()
    }

    info: Dict[str, Any] = {
        "enabled": enabled,
        "time_warp": {
            "left": {"applied": False, "timestamps": None},
            "right": {"applied": False, "timestamps": None},
        },
        "streams": {
            k: {
                "drift_applied": False,
                "drift": None,
                "scaling_applied": False,
                "scaling_factors": None,
                "noise_applied": False,
                "noise": None,
                "negate_applied": False,
                "channel_dropout_applied": False,
                "channel_dropout_mask": None,
            }
            for k in validated.keys()
        },
    }

    if not enabled:
        return ({k: v.clone().contiguous() for k, v in validated.items()}, info)

    # -------- 读取配置 --------
    time_warp_prob = float(_require_cfg_attr(cfg, "mindrove_time_warp_prob"))
    time_warp_sigma = float(_require_cfg_attr(cfg, "mindrove_time_warp_sigma"))
    time_warp_num_knots = int(_require_cfg_attr(cfg, "mindrove_time_warp_num_knots"))
    _ = int(_require_cfg_attr(cfg, "mindrove_time_warp_num_splines"))  # 字段保留，便于兼容 dataloader 配置

    emg_scaling_prob = float(_require_cfg_attr(cfg, "mindrove_emg_scaling_prob"))
    emg_scaling_sigma = float(_require_cfg_attr(cfg, "mindrove_emg_scaling_sigma"))
    emg_noise_prob = float(_require_cfg_attr(cfg, "mindrove_emg_noise_prob"))
    emg_noise_sigma = float(_require_cfg_attr(cfg, "mindrove_emg_noise_sigma"))

    imu_scaling_prob = float(_require_cfg_attr(cfg, "mindrove_imu_scaling_prob"))
    imu_scaling_sigma = float(_require_cfg_attr(cfg, "mindrove_imu_scaling_sigma"))
    imu_noise_prob = float(_require_cfg_attr(cfg, "mindrove_imu_noise_prob"))
    imu_noise_sigma = float(_require_cfg_attr(cfg, "mindrove_imu_noise_sigma"))

    emg_drift_prob = float(_require_cfg_attr(cfg, "mindrove_emg_drift_prob"))
    emg_drift_max = _require_cfg_attr(cfg, "mindrove_emg_drift_max")
    emg_drift_n_points = _require_cfg_attr(cfg, "mindrove_emg_drift_n_points")
    emg_drift_kind = _require_cfg_attr(cfg, "mindrove_emg_drift_kind")
    emg_drift_per_channel = _require_cfg_attr(cfg, "mindrove_emg_drift_per_channel")
    emg_drift_normalize = _require_cfg_attr(cfg, "mindrove_emg_drift_normalize")

    imu_drift_prob = float(_require_cfg_attr(cfg, "mindrove_imu_drift_prob"))
    imu_drift_max = _require_cfg_attr(cfg, "mindrove_imu_drift_max")
    imu_drift_n_points = _require_cfg_attr(cfg, "mindrove_imu_drift_n_points")
    imu_drift_kind = _require_cfg_attr(cfg, "mindrove_imu_drift_kind")
    imu_drift_per_channel = _require_cfg_attr(cfg, "mindrove_imu_drift_per_channel")
    imu_drift_normalize = _require_cfg_attr(cfg, "mindrove_imu_drift_normalize")

    emg_negate_prob = float(_require_cfg_attr(cfg, "mindrove_emg_negate_prob"))
    imu_negate_prob = float(_require_cfg_attr(cfg, "mindrove_imu_negate_prob"))

    emg_channel_dropout_prob = float(_require_cfg_attr(cfg, "mindrove_emg_channel_dropout_prob"))
    emg_channel_dropout_max_channels = int(_require_cfg_attr(cfg, "mindrove_emg_channel_dropout_max_channels"))
    imu_channel_dropout_prob = float(_require_cfg_attr(cfg, "mindrove_imu_channel_dropout_prob"))
    imu_channel_dropout_max_channels = int(_require_cfg_attr(cfg, "mindrove_imu_channel_dropout_max_channels"))

    out: Dict[str, torch.Tensor] = {k: v.clone().contiguous() for k, v in validated.items()}

    # ========================================================
    # A) per-hand shared time warp
    # ========================================================
    hand_to_keys = {
        "left": [k for k in out.keys() if k.startswith("left_")],
        "right": [k for k in out.keys() if k.startswith("right_")],
    }

    for hand, keys in hand_to_keys.items():
        if len(keys) == 0:
            continue

        if _should_apply(time_warp_prob):
            # ------------------------------------------------------------
            # 支持 EMG / IMU 不同长度
            # ------------------------------------------------------------
            # 旧版本要求同一只手内所有流长度完全相同，原因是它只生成一条
            # timestamps 并直接应用到所有流。现在 EMG 和 IMU 可以有不同 L，
            # 因此改成：同一只手只采样一次 knot_speed，代表同一个时间扭曲
            # profile；然后针对每条流自己的 length 插值生成对应 timestamps。
            # 这样既不会因为长度不同报错，也尽量保持“同一只手共享同一类
            # time warp 形状”的语义。
            ref_tensor = next(iter(out.values()))
            num_points = int(time_warp_num_knots) + 2
            if float(time_warp_sigma) == 0.0:
                shared_knot_speed = torch.ones(
                    (1, 1, num_points),
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                )
            else:
                shared_knot_speed = torch.exp(
                    torch.randn(
                        (1, 1, num_points),
                        device=ref_tensor.device,
                        dtype=ref_tensor.dtype,
                    ) * float(time_warp_sigma)
                )

            timestamps_by_key: Dict[str, torch.Tensor] = {}
            for k in keys:
                length = int(out[k].shape[1])
                ts = sample_shared_time_warp_timestamps_torch(
                    length=length,
                    sigma=time_warp_sigma,
                    num_knots=time_warp_num_knots,
                    device=out[k].device,
                    dtype=out[k].dtype,
                    knot_speed=shared_knot_speed,
                )
                out[k] = apply_time_warp_cf_with_shared_timestamps(out[k], ts)
                timestamps_by_key[k] = ts.detach().cpu()

            info["time_warp"][hand]["applied"] = True
            # 若该手内所有流长度相同，保留旧字段语义：timestamps 是单个 Tensor；
            # 若长度不同，则 timestamps 改为 dict[key -> Tensor]，便于调试。
            unique_lengths = {int(ts.numel()) for ts in timestamps_by_key.values()}
            if len(unique_lengths) == 1:
                info["time_warp"][hand]["timestamps"] = next(iter(timestamps_by_key.values()))
            else:
                info["time_warp"][hand]["timestamps"] = timestamps_by_key

    # ========================================================
    # B) per-stream drift + scaling + noise + negate + channel dropout
    # ========================================================
    for key in list(out.keys()):
        x = out[key]
        if key.endswith("_emg"):
            if _should_apply(emg_drift_prob):
                x, drift = apply_drift_cf(
                    x,
                    max_drift=emg_drift_max,
                    n_drift_points=emg_drift_n_points,
                    kind=emg_drift_kind,
                    per_channel=emg_drift_per_channel,
                    normalize=emg_drift_normalize,
                )
                info["streams"][key]["drift_applied"] = True
                info["streams"][key]["drift"] = drift.detach().cpu()
            if _should_apply(emg_scaling_prob):
                x, factors = apply_scaling_cf(x, sigma=emg_scaling_sigma)
                info["streams"][key]["scaling_applied"] = True
                info["streams"][key]["scaling_factors"] = factors.detach().cpu()
            if _should_apply(emg_noise_prob):
                x, noise = apply_noise_cf(x, sigma=emg_noise_sigma)
                info["streams"][key]["noise_applied"] = True
                info["streams"][key]["noise"] = noise.detach().cpu()
            if _should_apply(emg_negate_prob):
                x = apply_negate_cf(x)
                info["streams"][key]["negate_applied"] = True
            if _should_apply(emg_channel_dropout_prob):
                x, dropped_mask = apply_channel_dropout_cf(x, max_channels=emg_channel_dropout_max_channels)
                info["streams"][key]["channel_dropout_applied"] = True
                info["streams"][key]["channel_dropout_mask"] = dropped_mask.detach().cpu()
        elif key.endswith("_imu"):
            if _should_apply(imu_drift_prob):
                x, drift = apply_drift_cf(
                    x,
                    max_drift=imu_drift_max,
                    n_drift_points=imu_drift_n_points,
                    kind=imu_drift_kind,
                    per_channel=imu_drift_per_channel,
                    normalize=imu_drift_normalize,
                )
                info["streams"][key]["drift_applied"] = True
                info["streams"][key]["drift"] = drift.detach().cpu()
            if _should_apply(imu_scaling_prob):
                x, factors = apply_scaling_cf(x, sigma=imu_scaling_sigma)
                info["streams"][key]["scaling_applied"] = True
                info["streams"][key]["scaling_factors"] = factors.detach().cpu()
            if _should_apply(imu_noise_prob):
                x, noise = apply_noise_cf(x, sigma=imu_noise_sigma)
                info["streams"][key]["noise_applied"] = True
                info["streams"][key]["noise"] = noise.detach().cpu()
            if _should_apply(imu_negate_prob):
                x = apply_negate_cf(x)
                info["streams"][key]["negate_applied"] = True
            if _should_apply(imu_channel_dropout_prob):
                x, dropped_mask = apply_channel_dropout_cf(x, max_channels=imu_channel_dropout_max_channels)
                info["streams"][key]["channel_dropout_applied"] = True
                info["streams"][key]["channel_dropout_mask"] = dropped_mask.detach().cpu()
        else:
            raise KeyError(
                f"Unsupported MindRove stream key '{key}'. Expected keys ending with '_emg' or '_imu'."
            )
        out[key] = x.contiguous()

    if set(out.keys()) != set(validated.keys()):
        raise RuntimeError("Output keys changed after MindRove augmentation")

    return out, info



def apply_mindrove_augmentation(
    streams: Dict[str, torch.Tensor],
    cfg: Any,
) -> Dict[str, torch.Tensor]:
    """
    dataloader 直接调用的主入口。
    接口保持为 Dict[str, Tensor[C,L]] -> Dict[str, Tensor[C,L]]。
    """
    out, _ = apply_mindrove_augmentation_with_info(streams, cfg)
    return out


# ============================================================
# 5) 供 main 使用的 mindrove.pt 读取 / 重采样 / 可视化
# ============================================================


def _resample_cf(x_cf: torch.Tensor, target_len: int) -> torch.Tensor:
    x_cf = _ensure_tensor_cf(x_cf, key="<resample_input>")
    target_len = _ensure_pos_int("target_len", target_len)
    y = F.interpolate(
        x_cf.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)
    return y.contiguous()



def _to_cf_from_mindrove_pt_tensor(x: torch.Tensor, signal: str) -> torch.Tensor:
    """
    将 mindrove.pt 中的单路信号统一成 [C,L]。

    允许两种输入：
    - [L,C] : 原始 MindRove 常见存法
    - [C,L] : 已处理后的格式
    """
    if signal not in EXPECTED_CHANNELS:
        raise KeyError(f"Unsupported signal: {signal}")
    exp_c = EXPECTED_CHANNELS[signal]

    if not torch.is_tensor(x):
        raise TypeError(f"MindRove raw stream must be tensor, got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"MindRove raw stream must be 2D, got shape {tuple(x.shape)}")

    x = x.to(torch.float32).contiguous()

    if x.shape[1] == exp_c and x.shape[0] != exp_c:
        # [L,C] -> [C,L]
        return x.transpose(0, 1).contiguous()
    if x.shape[0] == exp_c and x.shape[1] != exp_c:
        # 已经是 [C,L]
        return x.contiguous()

    if x.shape[0] == exp_c and x.shape[1] == exp_c:
        raise ValueError(
            f"Ambiguous MindRove stream shape {tuple(x.shape)} for signal '{signal}': both dims equal expected channels {exp_c}."
        )

    raise ValueError(
        f"MindRove stream shape {tuple(x.shape)} is incompatible with signal '{signal}' (expected channels={exp_c})."
    )



def load_streams_from_mindrove_pt(
    mindrove_pt: str | Path,
    target_len: int,
    hands: Tuple[str, ...] = ("left", "right"),
    signals: Tuple[str, ...] = ("emg", "imu"),
) -> Dict[str, torch.Tensor]:
    """
    从单个 mindrove.pt 文件读取并重采样，返回 Dict[str, Tensor[C,L]]。
    """
    pt_path = Path(mindrove_pt)
    if not pt_path.is_file():
        raise FileNotFoundError(f"mindrove.pt not found: {pt_path}")

    obj = torch.load(pt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"mindrove.pt must contain a dict, got {type(obj)}")

    out: Dict[str, torch.Tensor] = {}
    for hand in hands:
        if hand not in ("left", "right"):
            raise ValueError(f"Unsupported hand: {hand}")
        has_key = f"has_{hand}"
        hand_exists = bool(obj.get(has_key, False))
        if not hand_exists:
            continue

        for signal in signals:
            if signal not in EXPECTED_CHANNELS:
                raise ValueError(f"Unsupported signal: {signal}")
            key = f"{hand}_{signal}"
            if key not in obj:
                continue
            x_cf = _to_cf_from_mindrove_pt_tensor(obj[key], signal=signal)
            x_cf = _resample_cf(x_cf, target_len=target_len)
            out[key] = x_cf

    if len(out) == 0:
        raise RuntimeError(f"No valid MindRove streams were loaded from: {pt_path}")
    return out



def _stacked_gap(x1: torch.Tensor, x2: torch.Tensor) -> float:
    amp = max(float(x1.abs().max().item()), float(x2.abs().max().item()), 1e-6)
    return amp * 2.5



def _plot_stacked_multichannel(ax, x_cf: torch.Tensor, title: str, gap: float) -> None:
    x = _ensure_tensor_cf(x_cf, key=title).detach().cpu()
    C, L = x.shape
    t = torch.arange(L).cpu().numpy()

    yticks = []
    ylabels = []
    for c in range(C):
        y = x[c].numpy() + c * gap
        ax.plot(t, y, linewidth=0.9)
        yticks.append(c * gap)
        ylabels.append(f"ch{c}")

    ax.set_title(title)
    ax.set_xlabel("time index")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.grid(True, alpha=0.25)



def visualize_original_vs_augmented(
    original: Dict[str, torch.Tensor],
    augmented: Dict[str, torch.Tensor],
    info: Optional[Dict[str, Any]] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """
    按流绘制 original / augmented / delta 三列对比图。
    """
    keys = sorted(original.keys())
    if set(keys) != set(augmented.keys()):
        raise ValueError("original and augmented keys do not match")

    nrows = len(keys)
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 3.6 * nrows), squeeze=False)

    for r, key in enumerate(keys):
        x0 = _ensure_tensor_cf(original[key], key=f"original:{key}")
        x1 = _ensure_tensor_cf(augmented[key], key=f"augmented:{key}")
        delta = x1 - x0
        gap = _stacked_gap(x0, x1)

        _plot_stacked_multichannel(axes[r, 0], x0, f"{key} | original", gap)
        _plot_stacked_multichannel(axes[r, 1], x1, f"{key} | augmented", gap)
        _plot_stacked_multichannel(axes[r, 2], delta, f"{key} | delta", gap)

    fig.suptitle("MindRove augmentation inspection", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if info is not None:
        print("\n[Augmentation summary]")
        for hand in ("left", "right"):
            hand_info = info.get("time_warp", {}).get(hand, {})
            applied = bool(hand_info.get("applied", False))
            print(f"  time_warp {hand}: {applied}")
        for key, s in info.get("streams", {}).items():
            print(
                f"  {key}: drift={bool(s.get('drift_applied', False))}, "
                f"scaling={bool(s.get('scaling_applied', False))}, "
                f"noise={bool(s.get('noise_applied', False))}, "
                f"negate={bool(s.get('negate_applied', False))}, "
                f"channel_dropout={bool(s.get('channel_dropout_applied', False))}"
            )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Saved figure] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 6) main
# ============================================================


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MindRove tensor-native augmentation + visualization")
    p.add_argument("--mindrove_pt", type=str, required=True, help="Path to a single mindrove.pt")
    p.add_argument("--target_len", type=int, default=256, help="Resample target length before augmentation")
    p.add_argument("--hands", nargs="+", default=["left", "right"], choices=["left", "right"])
    p.add_argument("--signals", nargs="+", default=["emg", "imu"], choices=["emg", "imu"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--no_show", action="store_true")

    p.add_argument("--mindrove_apply_augmentation", type=int, default=1)
    p.add_argument("--mindrove_time_warp_prob", type=float, default=0.0)
    p.add_argument("--mindrove_time_warp_sigma", type=float, default=0.2)
    p.add_argument("--mindrove_time_warp_num_knots", type=int, default=4)
    p.add_argument("--mindrove_time_warp_num_splines", type=int, default=150)

    p.add_argument("--mindrove_emg_scaling_prob", type=float, default=0.0)
    p.add_argument("--mindrove_emg_scaling_sigma", type=float, default=0.1)
    p.add_argument("--mindrove_emg_noise_prob", type=float, default=0.8)
    p.add_argument("--mindrove_emg_noise_sigma", type=float, default=0.05)

    p.add_argument("--mindrove_imu_scaling_prob", type=float, default=0.0)
    p.add_argument("--mindrove_imu_scaling_sigma", type=float, default=0.1)
    p.add_argument("--mindrove_imu_noise_prob", type=float, default=0.8)
    p.add_argument("--mindrove_imu_noise_sigma", type=float, default=0.05)

    p.add_argument("--mindrove_emg_drift_prob", type=float, default=1.0)
    p.add_argument("--mindrove_emg_drift_max", type=str, default="30.0")
    p.add_argument("--mindrove_emg_drift_n_points", type=str, default="3")
    p.add_argument("--mindrove_emg_drift_kind", type=str, default="additive")
    p.add_argument("--mindrove_emg_drift_per_channel", type=int, default=0)
    p.add_argument("--mindrove_emg_drift_normalize", type=int, default=1)

    p.add_argument("--mindrove_imu_drift_prob", type=float, default=1.0)
    p.add_argument("--mindrove_imu_drift_max", type=str, default="30.0")
    p.add_argument("--mindrove_imu_drift_n_points", type=str, default="3")
    p.add_argument("--mindrove_imu_drift_kind", type=str, default="additive")
    p.add_argument("--mindrove_imu_drift_per_channel", type=int, default=1)
    p.add_argument("--mindrove_imu_drift_normalize", type=int, default=0)

    p.add_argument("--mindrove_emg_negate_prob", type=float, default=0.0)
    p.add_argument("--mindrove_imu_negate_prob", type=float, default=0.0)
    p.add_argument("--mindrove_emg_channel_dropout_prob", type=float, default=0.0)
    p.add_argument("--mindrove_emg_channel_dropout_max_channels", type=int, default=1)
    p.add_argument("--mindrove_imu_channel_dropout_prob", type=float, default=0.0)
    p.add_argument("--mindrove_imu_channel_dropout_max_channels", type=int, default=1)
    return p



def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(args.seed)

    cfg = MindRoveAugConfig(
        mindrove_apply_augmentation=bool(args.mindrove_apply_augmentation),
        mindrove_time_warp_prob=args.mindrove_time_warp_prob,
        mindrove_time_warp_sigma=args.mindrove_time_warp_sigma,
        mindrove_time_warp_num_knots=args.mindrove_time_warp_num_knots,
        mindrove_time_warp_num_splines=args.mindrove_time_warp_num_splines,
        mindrove_emg_scaling_prob=args.mindrove_emg_scaling_prob,
        mindrove_emg_scaling_sigma=args.mindrove_emg_scaling_sigma,
        mindrove_emg_noise_prob=args.mindrove_emg_noise_prob,
        mindrove_emg_noise_sigma=args.mindrove_emg_noise_sigma,
        mindrove_imu_scaling_prob=args.mindrove_imu_scaling_prob,
        mindrove_imu_scaling_sigma=args.mindrove_imu_scaling_sigma,
        mindrove_imu_noise_prob=args.mindrove_imu_noise_prob,
        mindrove_imu_noise_sigma=args.mindrove_imu_noise_sigma,
        mindrove_emg_drift_prob=args.mindrove_emg_drift_prob,
        mindrove_emg_drift_max=_parse_literal_arg("mindrove_emg_drift_max", args.mindrove_emg_drift_max),
        mindrove_emg_drift_n_points=_parse_literal_arg("mindrove_emg_drift_n_points", args.mindrove_emg_drift_n_points),
        mindrove_emg_drift_kind=args.mindrove_emg_drift_kind,
        mindrove_emg_drift_per_channel=bool(args.mindrove_emg_drift_per_channel),
        mindrove_emg_drift_normalize=bool(args.mindrove_emg_drift_normalize),
        mindrove_imu_drift_prob=args.mindrove_imu_drift_prob,
        mindrove_imu_drift_max=_parse_literal_arg("mindrove_imu_drift_max", args.mindrove_imu_drift_max),
        mindrove_imu_drift_n_points=_parse_literal_arg("mindrove_imu_drift_n_points", args.mindrove_imu_drift_n_points),
        mindrove_imu_drift_kind=args.mindrove_imu_drift_kind,
        mindrove_imu_drift_per_channel=bool(args.mindrove_imu_drift_per_channel),
        mindrove_imu_drift_normalize=bool(args.mindrove_imu_drift_normalize),
        mindrove_emg_negate_prob=args.mindrove_emg_negate_prob,
        mindrove_imu_negate_prob=args.mindrove_imu_negate_prob,
        mindrove_emg_channel_dropout_prob=args.mindrove_emg_channel_dropout_prob,
        mindrove_emg_channel_dropout_max_channels=args.mindrove_emg_channel_dropout_max_channels,
        mindrove_imu_channel_dropout_prob=args.mindrove_imu_channel_dropout_prob,
        mindrove_imu_channel_dropout_max_channels=args.mindrove_imu_channel_dropout_max_channels,
    )

    original = load_streams_from_mindrove_pt(
        mindrove_pt=args.mindrove_pt,
        target_len=args.target_len,
        hands=tuple(args.hands),
        signals=tuple(args.signals),
    )

    augmented, info = apply_mindrove_augmentation_with_info(original, cfg)

    print("[Loaded streams]")
    for k, v in original.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    visualize_original_vs_augmented(
        original=original,
        augmented=augmented,
        info=info,
        save_path=args.save_path,
        show=(not args.no_show),
    )


if __name__ == "__main__":
    main()
