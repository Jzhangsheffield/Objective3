#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_rgb_signal_fusion_finetune_and_test.py

RGB + MindRove Signal 多模态融合分类训练 / 微调 / 测试脚本。

相较于原单模态 train_mapstyle_finetune_and_test.py，本脚本实现：
1) RGB 使用 ResNet3D backbone
2) Signal 使用 ResNet1D backbone；EMG 与 IMU 可以分别使用独立 backbone
3) RGB / EMG / IMU backbone 可分别加载对比学习预训练权重（默认自动丢掉对比头 / 分类头）
4) 多路 pooled feature 送入可选融合模块：
   - concat_mlp
   - gated
   - weighted_sum
5) 融合后接统一分类头做分类
6) 支持：
   - train / test
   - full / fusion_head / head_only
   - weighted sampler / weighted CE / focal / AMP
   - optimizer 可选 SGD / Adam
   - 多组 manifest 与权重按顺序匹配或广播
   - 保存 best_val / last / per-sample csv / summary csv

本版已按你当前的 mapstype_dataloader_with_index_mindrove.py 对齐，并补齐了 RGB / MindRove 标准化与完整增强参数：
- PackedMultiModalConfig 使用字段：
    n_frames, use_modalities, rgb_out_hw, depth_out_hw, mindrove_*
- build_packed_mapstyle_dataset(..., manifest_name=..., cfg=...)
- build_weighted_sampler_for_packed_dataset(...) 返回 (sampler, info)
- batch["mindrove"] 为单视图 dict[str, Tensor[B,C,L]]
- label_map_json 格式为：
    {
      "tier1": {"xxx": 0, ...},
      "tier2": {...},
      "tier3": {...}
    }

重要约束
========
1) 本脚本支持 RGB + EMG、RGB + IMU、RGB + EMG + IMU 三种融合输入
2) 分类训练不使用 two-view：
   - rgb_two_views = False
   - mindrove_two_views = False
3) 需要你的：
   - ResNet3D backbone 提供 forward_features(x) -> [B,D]
   - ResNet1D backbone 提供 forward_features(x) -> [B,D]
4) 测试模式下，输入的 test_weight_paths 应该是“完整融合分类模型权重”。测试时的 signal_types、fusion_type、backbone 配置必须与训练保存权重时一致。
"""

from __future__ import annotations

import os
import re
import csv
import json
import math
import random
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple, Any

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

# ---------------- backbone ----------------
import backbone.resnet as resnet3d
from backbone.renet1d_my import build_resnet1d

# ---------------- fusion ----------------
from fusion_module.concat_mlp import ConcatMLPFusion
from fusion_module.gated_fusion import GatedFusion
from fusion_module.weighted_sum import WeightedSumFusion

# ---------------- helper ----------------
from utils_.log_training_dynamics import TrainingDynamicsLogger
from loss.focal_loss import FocalLoss

from utils_.mapstype_dataloader_with_index_mindrove import (
    PackedMultiModalConfig,
    load_label_map_json,
    build_packed_mapstyle_dataset,
    build_packed_mapstyle_loader_from_dataset,
    build_weighted_sampler_for_packed_dataset,
)

# ============================================================
# 0) CLI
# ============================================================

parser = argparse.ArgumentParser(
    description="Train / finetune / batch-test RGB + EMG/IMU fusion classifier on map-style dataset"
)

# ---------------- run mode ----------------
parser.add_argument("--run_mode", type=str, default="train", choices=["train", "test"])

# ---------------- paths ----------------
parser.add_argument("--save_path", type=str, default="./weights")
parser.add_argument("--datamap_csv_path", type=str, default="./datamaps")
parser.add_argument("--dataset_root", type=str, required=True)
parser.add_argument("--label_map_json", type=str, required=True)

# ---------------- manifests ----------------
parser.add_argument("--train_manifest", type=str, default=None)
parser.add_argument("--train_manifests", nargs="*", default=[])
parser.add_argument("--val_manifest", type=str, default=None)
parser.add_argument("--val_manifests", nargs="*", default=[])

parser.add_argument(
    "--test_manifest",
    nargs="+",
    default=[],
    help="一个或多个测试 manifest；若只给 1 个，则广播给所有 test_weight_paths"
)

# ---------------- naming ----------------
parser.add_argument("--pretrained_tag_mode", type=str, default="last_k_dirs",
                    choices=["legacy", "last_k_dirs", "relative_to_anchor"])
parser.add_argument("--pretrained_tag_last_k", type=int, default=4)
parser.add_argument("--pretrained_tag_anchor", type=str, default=None)

# ---------------- labels / data ----------------
parser.add_argument("--tier_mode", type=str, default="tier1",
                    choices=["tier1", "tier2", "tier3"])
parser.add_argument("--num_classes", type=int, default=17)
parser.add_argument("--n_frames", type=int, default=16)

parser.add_argument("--signal_type", type=str, default="emg", choices=["emg", "imu"],
                    help=(
                        "向后兼容旧脚本的单信号参数。"
                        "当没有显式提供 --signal_types 时，本参数决定使用 emg 或 imu。"
                    ))
parser.add_argument(
    "--signal_types",
    nargs="*",
    default=None,
    choices=["emg", "imu"],
    help=(
        "显式指定参与融合的 MindRove 信号列表。"
        "例如：--signal_types emg imu 表示 RGB+EMG+IMU 三模态融合；"
        "--signal_types emg 表示 RGB+EMG；--signal_types imu 表示 RGB+IMU。"
        "如果不提供，则自动回退到旧参数 --signal_type。"
    ),
)

parser.add_argument("--num_workers_train", type=int, default=8)
parser.add_argument("--num_workers_val", type=int, default=6)
parser.add_argument("--num_workers_test", type=int, default=8)

parser.add_argument("--prefetch_factor_train", type=int, default=2)
parser.add_argument("--prefetch_factor_val", type=int, default=2)
parser.add_argument("--prefetch_factor_test", type=int, default=2)

parser.add_argument("--disable_val", action="store_true")

# ---------------- RGB spatial / depth size ----------------
parser.add_argument("--rgb_size", type=int, default=224)
parser.add_argument("--depth_size", type=int, default=224)

parser.add_argument("--rrc_scale_min", type=float, default=0.6)
parser.add_argument("--rrc_scale_max", type=float, default=1.0)
parser.add_argument("--rrc_ratio_min", type=float, default=0.75)
parser.add_argument("--rrc_ratio_max", type=float, default=1.3333333333)

# ---------------- RGB normalization / augmentation ----------------
# 这些参数与单模态脚本 train_mapstyle_finetune_and_test.py 保持一致，
# 由 dataloader 内部的 RGB transform 使用；训练脚本本身不再重复 Normalize。
parser.add_argument(
    "--rgb_mean",
    nargs=3,
    type=float,
    default=[0.356, 0.363, 0.367],
    metavar=("R_MEAN", "G_MEAN", "B_MEAN"),
    help="RGB Normalize 使用的 mean，顺序为 R G B。",
)
parser.add_argument(
    "--rgb_std",
    nargs=3,
    type=float,
    default=[0.288, 0.271, 0.270],
    metavar=("R_STD", "G_STD", "B_STD"),
    help="RGB Normalize 使用的 std，顺序为 R G B，三个值必须为正数。",
)
parser.add_argument(
    "--rgb_apply_spatial_aug",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "训练集是否启用 RGB 随机空间增强中的 flip/jitter/gray/blur。"
        "设为 False 时，RandomResizedCrop 仍由 rrc_scale/rrc_ratio 控制。"
    ),
)
parser.add_argument("--rgb_hflip_p", type=float, default=0.5,
                    help="训练集 RGB RandomHorizontalFlip 概率。")
parser.add_argument("--rgb_vflip_p", type=float, default=0.5,
                    help="训练集 RGB RandomVerticalFlip 概率；机械操作视频通常建议设为 0。")
parser.add_argument("--rgb_jitter_p", type=float, default=0.5,
                    help="训练集 RGB ColorJitter 被应用的概率。")
parser.add_argument("--rgb_jitter_brightness", type=float, default=0.24,
                    help="ColorJitter brightness 强度。")
parser.add_argument("--rgb_jitter_contrast", type=float, default=0.24,
                    help="ColorJitter contrast 强度。")
parser.add_argument("--rgb_jitter_saturation", type=float, default=0.24,
                    help="ColorJitter saturation 强度。")
parser.add_argument("--rgb_jitter_hue", type=float, default=0.16,
                    help="ColorJitter hue 强度；torchvision 通常要求不超过 0.5。")
parser.add_argument("--rgb_gray_p", type=float, default=0.2,
                    help="训练集 RGB RandomGrayscale 概率。")
parser.add_argument("--rgb_blur_p", type=float, default=0.5,
                    help="训练集 RGB GaussianBlur 被应用的概率。")
parser.add_argument("--rgb_blur_kernel", type=int, default=7,
                    help="GaussianBlur kernel size，必须是 >=3 的奇数。")
parser.add_argument("--rgb_blur_sigma_min", type=float, default=0.1,
                    help="GaussianBlur sigma 下界。")
parser.add_argument("--rgb_blur_sigma_max", type=float, default=1.0,
                    help="GaussianBlur sigma 上界。")
parser.add_argument(
    "--disable_train_augmentation",
    action="store_true",
    help=(
        "统一关闭训练集增强。启用后：RGB 的 RandomResizedCrop 退化为不裁剪，"
        "flip/jitter/gray/blur 概率全部置 0；MindRove 样本级增强也会关闭。"
        "验证/测试集本来就不启用训练增强。"
    ),
)

# ---------------- MindRove ----------------
parser.add_argument("--mindrove_target_len", type=int, default=256,
                    help="MindRove 默认重采样长度；当 EMG/IMU 单独长度未设置时使用它。")
parser.add_argument("--mindrove_emg_target_len", type=int, default=None,
                    help="EMG 单独重采样长度；若不设置，则回退到 --mindrove_target_len。")
parser.add_argument("--mindrove_imu_target_len", type=int, default=None,
                    help="IMU 单独重采样长度；若不设置，则回退到 --mindrove_target_len。")
parser.add_argument("--mindrove_hands", nargs="+", default=["left", "right"],
                    choices=["left", "right"])
parser.add_argument("--mindrove_merge_hands", action="store_true")
parser.add_argument(
    "--mindrove_apply_augmentation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="训练集是否启用 MindRove 样本级增强；验证/测试集会由 dataloader 自动关闭"
)
parser.add_argument(
    "--mindrove_apply_normalization",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="是否在重采样后、增强前，对 MindRove 做 per-channel mean/std 标准化"
)

# ---------------- MindRove normalization stats ----------------
parser.add_argument("--mindrove_left_emg_mean", nargs="+", type=float, default=None,
                    help="左手 EMG 的 per-channel mean，长度必须为 8")
parser.add_argument("--mindrove_left_emg_std", nargs="+", type=float, default=None,
                    help="左手 EMG 的 per-channel std，长度必须为 8")
parser.add_argument("--mindrove_right_emg_mean", nargs="+", type=float, default=None,
                    help="右手 EMG 的 per-channel mean，长度必须为 8")
parser.add_argument("--mindrove_right_emg_std", nargs="+", type=float, default=None,
                    help="右手 EMG 的 per-channel std，长度必须为 8")
parser.add_argument("--mindrove_left_imu_mean", nargs="+", type=float, default=None,
                    help="左手 IMU 的 per-channel mean，长度必须为 6")
parser.add_argument("--mindrove_left_imu_std", nargs="+", type=float, default=None,
                    help="左手 IMU 的 per-channel std，长度必须为 6")
parser.add_argument("--mindrove_right_imu_mean", nargs="+", type=float, default=None,
                    help="右手 IMU 的 per-channel mean，长度必须为 6")
parser.add_argument("--mindrove_right_imu_std", nargs="+", type=float, default=None,
                    help="右手 IMU 的 per-channel std，长度必须为 6")

# ---------------- MindRove augmentation params ----------------
parser.add_argument("--mindrove_time_warp_prob", type=float, default=0.5)
parser.add_argument("--mindrove_time_warp_sigma", type=float, default=0.2)
parser.add_argument("--mindrove_time_warp_num_knots", type=int, default=3)
parser.add_argument("--mindrove_time_warp_num_splines", type=int, default=150)

parser.add_argument("--mindrove_emg_scaling_prob", type=float, default=0.8)
parser.add_argument("--mindrove_emg_scaling_sigma", type=float, default=0.1)
parser.add_argument("--mindrove_emg_noise_prob", type=float, default=0.8)
parser.add_argument("--mindrove_emg_noise_sigma", type=float, default=0.05)
parser.add_argument("--mindrove_emg_drift_prob", type=float, default=0.0)
parser.add_argument("--mindrove_emg_drift_max", nargs="+", type=float, default=[0.0],
                    help="EMG drift 的最大幅值；传 1 个值表示固定幅值，传 2 个值表示 [low, high]")
parser.add_argument("--mindrove_emg_drift_n_points", nargs="+", type=int, default=[3],
                    help="EMG drift 控制点数；传 1 个值表示固定值，传多个值表示候选列表")
parser.add_argument("--mindrove_emg_drift_kind", type=str, default="additive",
                    choices=["additive", "multiplicative"])
parser.add_argument("--mindrove_emg_drift_per_channel", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--mindrove_emg_drift_normalize", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--mindrove_emg_negate_prob", type=float, default=0.0)
parser.add_argument("--mindrove_emg_channel_dropout_prob", type=float, default=0.0)
parser.add_argument("--mindrove_emg_channel_dropout_max_channels", type=int, default=1)

parser.add_argument("--mindrove_imu_scaling_prob", type=float, default=0.8)
parser.add_argument("--mindrove_imu_scaling_sigma", type=float, default=0.1)
parser.add_argument("--mindrove_imu_noise_prob", type=float, default=0.8)
parser.add_argument("--mindrove_imu_noise_sigma", type=float, default=0.05)
parser.add_argument("--mindrove_imu_drift_prob", type=float, default=0.0)
parser.add_argument("--mindrove_imu_drift_max", nargs="+", type=float, default=[0.0],
                    help="IMU drift 的最大幅值；传 1 个值表示固定幅值，传 2 个值表示 [low, high]")
parser.add_argument("--mindrove_imu_drift_n_points", nargs="+", type=int, default=[3],
                    help="IMU drift 控制点数；传 1 个值表示固定值，传多个值表示候选列表")
parser.add_argument("--mindrove_imu_drift_kind", type=str, default="additive",
                    choices=["additive", "multiplicative"])
parser.add_argument("--mindrove_imu_drift_per_channel", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--mindrove_imu_drift_normalize", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--mindrove_imu_negate_prob", type=float, default=0.0)
parser.add_argument("--mindrove_imu_channel_dropout_prob", type=float, default=0.0)
parser.add_argument("--mindrove_imu_channel_dropout_max_channels", type=int, default=1)

# ---------------- RGB backbone ----------------
parser.add_argument("--model_depth", type=int, default=18)
parser.add_argument("--rgb_n_input_channels", type=int, default=3)
parser.add_argument("--rgb_conv1_t_size", type=int, default=7)
parser.add_argument("--rgb_conv1_t_stride", type=int, default=1)
parser.add_argument("--rgb_no_max_pool", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rgb_shortcut_type", type=str, default="B", choices=["A", "B"])
parser.add_argument("--rgb_widen_factor", type=float, default=1.0)

# ---------------- Signal backbone ----------------
parser.add_argument("--mindrove_arch", default="resnet10_1d",
                    choices=["resnet10_1d", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
parser.add_argument("--mindrove_base_channels", default=64, type=int)
parser.add_argument("--mindrove_stem_kernel_size", default=7, type=int)
parser.add_argument("--mindrove_stem_stride", default=2, type=int)
parser.add_argument("--mindrove_use_stem_pool", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--mindrove_zero_init_residual", action=argparse.BooleanOptionalAction, default=False)

# ---------------- Fusion ----------------
parser.add_argument("--fusion_type", type=str, default="gated",
                    choices=["concat_mlp", "gated", "weighted_sum"])

parser.add_argument("--fusion_hidden_dim", type=int, default=512)
parser.add_argument("--fusion_use_projection", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--fusion_use_pre_bn", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--fusion_activation", type=str, default="relu",
                    choices=["identity", "relu", "gelu", "tanh"])
parser.add_argument("--fusion_dropout", type=float, default=0.0)
parser.add_argument("--classifier_dropout", type=float, default=0.0)

parser.add_argument("--concat_mlp_hidden_dim", type=int, default=512)
parser.add_argument("--concat_fusion_out_dim", type=int, default=128)
parser.add_argument("--concat_projection_dropout", type=float, default=0.0)

parser.add_argument("--gated_gate_type", type=str, default="vector", choices=["scalar", "vector"])

parser.add_argument("--weighted_sum_method", type=str, default="scalar",
                    choices=["scalar", "feature"])
parser.add_argument("--weighted_sum_normalize", action=argparse.BooleanOptionalAction, default=True)

parser.add_argument("--head_hidden_dim", type=int, default=128)

# ---------------- training ----------------
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["sgd", "adam"],
    help=(
        "选择优化器。sgd 使用 torch.optim.SGD 并使用 momentum；"
        "adam 使用 torch.optim.Adam，不使用 momentum。"
    ),
)
parser.add_argument("--cos", action="store_true")
parser.add_argument("--schedules", default=[25, 50, 75], nargs="*", type=int)
parser.add_argument("--seed", type=int, default=None)

# ---------------- sampler / losses ----------------
parser.add_argument("--use_weighted_sampler", action="store_true")
parser.add_argument("--sampler_tier", type=str, default=None,
                    choices=["tier1", "tier2", "tier3"])
parser.add_argument("--sampler_mode", type=str, default="sqrt_inv",
                    choices=["inv", "sqrt_inv"])

parser.add_argument("--use_weighted_ce", action="store_true")
parser.add_argument("--weight_method", type=str, default="class_balanced",
                    choices=["class_balanced", "inv_freq"])
parser.add_argument("--cb_beta", type=float, default=0.999)
parser.add_argument("--weight_normalize_mean", action="store_true")

parser.add_argument("--use_focal", action="store_true")
parser.add_argument("--focal_gamma", type=float, default=2.0)
parser.add_argument("--focal_use_alpha", action="store_true")

parser.add_argument("--enable_amp", action="store_true")

# ---------------- pretrained ----------------
parser.add_argument("--rgb_pretrained_weight_paths", nargs="*", default=[])
parser.add_argument(
    "--signal_pretrained_weight_paths",
    nargs="*",
    default=[],
    help=(
        "向后兼容旧脚本的单 signal backbone 预训练权重列表。"
        "仅当最终只选择一个 signal_type 时使用；"
        "如果使用 RGB+EMG+IMU，请分别使用 --emg_pretrained_weight_paths 和 --imu_pretrained_weight_paths。"
    ),
)
parser.add_argument(
    "--emg_pretrained_weight_paths",
    nargs="*",
    default=[],
    help="EMG backbone 的预训练权重列表；可为 0/1/N 个，0 表示该分支 scratch，1 表示广播。",
)
parser.add_argument(
    "--imu_pretrained_weight_paths",
    nargs="*",
    default=[],
    help="IMU backbone 的预训练权重列表；可为 0/1/N 个，0 表示该分支 scratch，1 表示广播。",
)
parser.add_argument("--include_scratch_baseline", action="store_true")

parser.add_argument("--finetune_mode", type=str, default="full",
                    choices=["full", "fusion_head", "head_only"])

parser.add_argument("--keep_pretrained_head", action="store_true")
parser.add_argument("--pretrained_strict", action="store_true")

# ---------------- LR split ----------------
parser.add_argument("--use_discriminative_lr", action="store_true")
parser.add_argument("--backbone_learning_rate", type=float, default=None)
parser.add_argument("--fusion_learning_rate", type=float, default=None)
parser.add_argument("--head_learning_rate", type=float, default=None)

# ---------------- test ----------------
parser.add_argument("--test_weight_paths", nargs="*", default=[])
parser.add_argument("--test_results_csv", type=str, default=None)

# ---------------- misc ----------------
parser.add_argument("--save_last_checkpoint", action="store_true", default=True)


# ============================================================
# 1) utilities
# ============================================================

MINDROVE_SIGNAL_CHANNELS = {"emg": 8, "imu": 6}

HEAD_KEYWORDS = (
    "fc.",
    "classifier.",
    "head.",
    "projector.",
    "predictor.",
    "fusion.",
)

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(path: str | Path, obj: dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sanitize_tag(s: str) -> str:
    s = s.replace("\\", "/")
    s = re.sub(r"[^0-9a-zA-Z._/-]+", "_", s)
    s = s.strip("._/-")
    return s or "unknown"

def save_csv_rows(csv_path: str | Path, rows: Sequence[dict], append: bool = True) -> None:
    rows = list(rows)
    if not rows:
        return

    ensure_dir(Path(csv_path).parent)
    file_exists = Path(csv_path).is_file()
    mode = "a" if (append and file_exists) else "w"

    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

def use_bf16_available() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

def collect_single_and_multi_paths(single_path: Optional[str], multi_paths: Sequence[str]) -> List[str]:
    out = []
    if single_path:
        out.append(single_path)
    out.extend([x for x in multi_paths if x])
    return out

def broadcast_or_match(values: List[Any], target_len: int, name: str, allow_empty: bool = False) -> List[Any]:
    if len(values) == 0:
        if allow_empty:
            return [None] * target_len
        raise ValueError(f"{name} is empty, cannot broadcast to length={target_len}")
    if len(values) == 1:
        return values * target_len
    if len(values) == target_len:
        return values
    raise ValueError(f"{name} length mismatch: got {len(values)}, expected 1 or {target_len}")


def get_selected_signal_types(args) -> List[str]:
    """
    统一解析当前实验要使用哪些 MindRove signal 分支。

    兼容策略
    --------
    1) 新接口：--signal_types emg imu
       - 用于 RGB+EMG+IMU 三模态融合，也可只传 emg 或 imu。
    2) 旧接口：--signal_type emg / --signal_type imu
       - 当 --signal_types 没有显式提供时自动使用，保证旧命令不需要修改。

    返回值始终是去重且保持用户输入顺序的 list，例如 ["emg", "imu"]。
    """
    raw = args.signal_types if args.signal_types is not None and len(args.signal_types) > 0 else [args.signal_type]
    out: List[str] = []
    for sig in raw:
        sig = str(sig).strip().lower()
        if sig not in MINDROVE_SIGNAL_CHANNELS:
            raise ValueError(f"Unsupported signal type: {sig}. Supported: {sorted(MINDROVE_SIGNAL_CHANNELS)}")
        if sig not in out:
            out.append(sig)
    if len(out) == 0:
        raise ValueError("At least one signal type must be selected.")
    return out


def get_fusion_modality_names(args) -> List[str]:
    """
    返回送入 fusion module 的模态顺序。

    这个顺序必须在以下位置保持一致：
    - build_fusion_module(input_dims)
    - RGBSignalFusionNet.forward()
    - per-sample CSV 中解析 gated / weighted_sum 权重
    """
    return ["rgb"] + get_selected_signal_types(args)


def get_signal_pretrained_lists(args) -> Dict[str, List[str]]:
    """
    返回每个 signal 分支的预训练权重列表。

    兼容规则
    --------
    - 如果只选择一个 signal，例如 --signal_types emg 或旧的 --signal_type emg，
      则可以继续使用旧参数 --signal_pretrained_weight_paths。
    - 如果选择两个 signal，即 RGB+EMG+IMU，则必须用：
        --emg_pretrained_weight_paths
        --imu_pretrained_weight_paths
      分别指定两条 1D backbone 的权重，避免把一个列表误配到两个分支。
    - 某个列表为空表示该分支从 scratch 初始化。
    """
    selected = get_selected_signal_types(args)
    generic = list(args.signal_pretrained_weight_paths or [])
    specific = {
        "emg": list(args.emg_pretrained_weight_paths or []),
        "imu": list(args.imu_pretrained_weight_paths or []),
    }

    if len(selected) > 1 and len(generic) > 0:
        raise ValueError(
            "When using multiple signal branches, e.g. --signal_types emg imu, "
            "do not use --signal_pretrained_weight_paths. "
            "Please use --emg_pretrained_weight_paths and --imu_pretrained_weight_paths explicitly."
        )

    out: Dict[str, List[str]] = {}
    for sig in selected:
        if len(selected) == 1:
            if len(generic) > 0 and len(specific[sig]) > 0:
                raise ValueError(
                    f"Both --signal_pretrained_weight_paths and --{sig}_pretrained_weight_paths were provided. "
                    "Please keep only one to avoid ambiguous loading."
                )
            out[sig] = generic if len(generic) > 0 else specific[sig]
        else:
            out[sig] = specific[sig]
    return out


def infer_pretrained_run_count(args) -> int:
    """
    根据所有 active branch 的预训练权重列表推断实验数量 N。

    规则
    ----
    - 所有权重列表都为空：N=0，表示只跑 scratch。
    - 非空列表长度可以是 1 或 N：长度 1 会广播，长度 N 会逐项匹配。
    - 多个非空列表若长度不一致，且都不是 1，则报错。

    这样可以支持：
    - 只给 RGB 权重，EMG/IMU scratch；
    - 给 RGB/EMG/IMU 各 N 个权重并一一匹配；
    - 给某个分支 1 个权重并广播到 N 个实验。
    """
    path_lists: List[List[str]] = [list(args.rgb_pretrained_weight_paths or [])]
    path_lists.extend(get_signal_pretrained_lists(args).values())
    nonzero_lens = [len(xs) for xs in path_lists if len(xs) > 0]
    if len(nonzero_lens) == 0:
        return 0
    target = max(nonzero_lens)
    for n in nonzero_lens:
        if n not in (1, target):
            raise ValueError(
                f"Pretrained path list length mismatch. Non-empty lengths are {nonzero_lens}; "
                f"each must be 1 or the common target length {target}."
            )
    return target


def broadcast_optional_weight_list(values: Sequence[str], target_len: int, name: str) -> List[Optional[str]]:
    """
    把某个分支的预训练权重列表对齐到 target_len。

    - len=0 -> [None] * target_len，表示该分支 scratch。
    - len=1 -> 广播。
    - len=target_len -> 一一匹配。
    """
    values = [x for x in values if x]
    if len(values) == 0:
        return [None] * target_len
    if len(values) == 1:
        return list(values) * target_len
    if len(values) == target_len:
        return list(values)
    raise ValueError(f"{name} length mismatch: got {len(values)}, expected 0, 1 or {target_len}")

def resolve_manifest_arg(dataset_root: str, manifest_arg: str) -> str:
    """
    兼容：
    1) 直接给绝对路径
    2) 给相对 dataset_root 的路径
    3) 只给文件名
    返回给 loader 的 manifest_name 字符串。
    """
    p = Path(manifest_arg)
    if p.is_absolute():
        return str(p)

    root = Path(dataset_root)
    cand = root / manifest_arg
    if cand.exists():
        return str(cand)

    return manifest_arg

def validate_args(args) -> None:
    # ------------------------------------------------------------
    # RGB transform 参数检查
    # ------------------------------------------------------------
    # 这些检查提前发现命令行错误，避免 dataloader 构建到一半才报错。
    if not (0.0 < args.rrc_scale_min <= args.rrc_scale_max <= 1.0):
        raise ValueError(
            f"Require 0 < rrc_scale_min <= rrc_scale_max <= 1, "
            f"got ({args.rrc_scale_min}, {args.rrc_scale_max})"
        )

    if not (0.0 < args.rrc_ratio_min <= args.rrc_ratio_max):
        raise ValueError(
            f"Require 0 < rrc_ratio_min <= rrc_ratio_max, "
            f"got ({args.rrc_ratio_min}, {args.rrc_ratio_max})"
        )

    for name in [
        "rgb_hflip_p",
        "rgb_vflip_p",
        "rgb_jitter_p",
        "rgb_gray_p",
        "rgb_blur_p",
    ]:
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    for name in [
        "rgb_jitter_brightness",
        "rgb_jitter_contrast",
        "rgb_jitter_saturation",
    ]:
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    if not (0.0 <= args.rgb_jitter_hue <= 0.5):
        raise ValueError(f"rgb_jitter_hue must be in [0, 0.5], got {args.rgb_jitter_hue}")

    if args.rgb_blur_kernel < 3 or args.rgb_blur_kernel % 2 == 0:
        raise ValueError(
            f"rgb_blur_kernel must be an odd integer >= 3, got {args.rgb_blur_kernel}"
        )

    if not (0.0 < args.rgb_blur_sigma_min <= args.rgb_blur_sigma_max):
        raise ValueError(
            f"Require 0 < rgb_blur_sigma_min <= rgb_blur_sigma_max, "
            f"got ({args.rgb_blur_sigma_min}, {args.rgb_blur_sigma_max})"
        )

    if any(float(x) <= 0.0 for x in args.rgb_std):
        raise ValueError(f"All rgb_std values must be positive, got {args.rgb_std}")

    # ------------------------------------------------------------
    # MindRove target length 参数检查
    # ------------------------------------------------------------
    if not isinstance(args.mindrove_target_len, int) or args.mindrove_target_len <= 0:
        raise ValueError(f"mindrove_target_len must be a positive int, got {args.mindrove_target_len}")

    for name in ("mindrove_emg_target_len", "mindrove_imu_target_len"):
        value = getattr(args, name)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be None or a positive int, got {value}")

    # ------------------------------------------------------------
    # Signal branch / pretrained 参数检查
    # ------------------------------------------------------------
    # 这里会同时检查：
    # - --signal_types 是否合法；
    # - 多 signal 分支时是否错误使用了旧的 --signal_pretrained_weight_paths；
    # - 各分支预训练权重列表是否能广播或逐项匹配。
    selected_signals = get_selected_signal_types(args)
    _ = get_signal_pretrained_lists(args)
    n_pretrained = infer_pretrained_run_count(args)

    if args.run_mode == "train":
        train_manifests = collect_single_and_multi_paths(args.train_manifest, args.train_manifests)
        if len(train_manifests) == 0:
            raise ValueError("In train mode, you must provide --train_manifest or --train_manifests")

        # 如果有预训练实验，train/val manifest 需要能够对齐到实验数；
        # 如果全 scratch，则只需要至少一个 train manifest。
        if n_pretrained > 0:
            _ = broadcast_or_match(train_manifests, n_pretrained, "train_manifests")
            val_manifests = collect_single_and_multi_paths(args.val_manifest, args.val_manifests)
            _ = broadcast_or_match(val_manifests, n_pretrained, "val_manifests", allow_empty=True)

    elif args.run_mode == "test":
        if len(args.test_manifest) == 0:
            raise ValueError("In test mode, you must provide --test_manifest")
        if len(args.test_weight_paths) == 0:
            raise ValueError("In test mode, you must provide --test_weight_paths")

    if len(selected_signals) > 1 and args.finetune_mode == "head_only":
        # 不是错误，只是提醒：head_only 会冻结 RGB/EMG/IMU backbone 和 fusion，只训练 classifier。
        print("[warning] RGB+EMG+IMU with finetune_mode=head_only freezes all backbones and the fusion module; only classifier is trainable.")

def build_pretrained_tag(weight_path: Optional[str], args, prefix: str) -> str:
    if weight_path is None:
        return f"{prefix}_scratch"

    p = Path(weight_path)

    if args.pretrained_tag_mode == "legacy":
        return sanitize_tag(f"{p.parent.name}_{p.stem}")

    if args.pretrained_tag_mode == "last_k_dirs":
        parts = list(p.parts)
        last_dirs = parts[-(args.pretrained_tag_last_k + 1):-1] if args.pretrained_tag_last_k > 0 else []
        return sanitize_tag("/".join(last_dirs + [p.stem]))

    if args.pretrained_tag_mode == "relative_to_anchor":
        if not args.pretrained_tag_anchor:
            raise ValueError("pretrained_tag_mode=relative_to_anchor requires --pretrained_tag_anchor")
        parts = list(p.parts)
        if args.pretrained_tag_anchor in parts:
            idx = parts.index(args.pretrained_tag_anchor)
            tag_parts = parts[idx:]
            tag_parts[-1] = p.stem
            return sanitize_tag("/".join(tag_parts))
        return sanitize_tag(f"{prefix}_{p.stem}")

    raise ValueError(f"Unknown pretrained_tag_mode: {args.pretrained_tag_mode}")

def build_training_sources(args) -> List[dict]:
    """
    构造训练实验源。

    与旧版不同，本函数不再假设只有一个 signal backbone。
    每个 source 会携带：
      - rgb_pretrained_path: Optional[str]
      - signal_pretrained_paths: dict，例如 {"emg": path_or_None, "imu": path_or_None}

    预训练权重对齐规则由 infer_pretrained_run_count() 和
    broadcast_optional_weight_list() 统一处理。
    """
    train_manifests = collect_single_and_multi_paths(args.train_manifest, args.train_manifests)
    val_manifests = collect_single_and_multi_paths(args.val_manifest, args.val_manifests)
    selected_signals = get_selected_signal_types(args)
    signal_pretrained_lists = get_signal_pretrained_lists(args)

    n_pretrained = infer_pretrained_run_count(args)

    if n_pretrained == 0:
        train_list = broadcast_or_match(train_manifests, 1, "train_manifests")
        val_list = broadcast_or_match(val_manifests, 1, "val_manifests", allow_empty=True)
        return [{
            "train_manifest": train_list[0],
            "val_manifest": val_list[0],
            "rgb_pretrained_path": None,
            "signal_pretrained_paths": {sig: None for sig in selected_signals},
            "source_tag": "scratch",
        }]

    train_list = broadcast_or_match(train_manifests, n_pretrained, "train_manifests")
    val_list = broadcast_or_match(val_manifests, n_pretrained, "val_manifests", allow_empty=True)

    rgb_list = broadcast_optional_weight_list(args.rgb_pretrained_weight_paths, n_pretrained, "rgb_pretrained_weight_paths")
    signal_lists = {
        sig: broadcast_optional_weight_list(signal_pretrained_lists[sig], n_pretrained, f"{sig}_pretrained_weight_paths")
        for sig in selected_signals
    }

    sources = []
    for i in range(n_pretrained):
        rgb_p = rgb_list[i]
        sig_paths = {sig: signal_lists[sig][i] for sig in selected_signals}

        tags = [build_pretrained_tag(rgb_p, args, "rgb")]
        for sig in selected_signals:
            tags.append(build_pretrained_tag(sig_paths[sig], args, sig))

        sources.append({
            "train_manifest": train_list[i],
            "val_manifest": val_list[i],
            "rgb_pretrained_path": rgb_p,
            "signal_pretrained_paths": sig_paths,
            "source_tag": sanitize_tag("__".join(tags)),
        })

    if args.include_scratch_baseline:
        sources.append({
            "train_manifest": train_list[0],
            "val_manifest": val_list[0],
            "rgb_pretrained_path": None,
            "signal_pretrained_paths": {sig: None for sig in selected_signals},
            "source_tag": "scratch",
        })

    return sources

def build_test_pairs(args) -> List[dict]:
    test_manifests = broadcast_or_match(list(args.test_manifest), len(args.test_weight_paths), "test_manifest")
    return [{"test_manifest": m, "weight_path": w} for m, w in zip(test_manifests, args.test_weight_paths)]



# ============================================================
# 2) dataset / loader
# ============================================================

def _pack_cli_float_scalar_or_pair(values, arg_name: str):
    """
    将 argparse 读入的 float 列表整理为：
    - 1 个值 -> float
    - 2 个值 -> (float, float)

    这里用于 drift_max，严格限制只能传 1 个或 2 个值。
    """
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{arg_name} must be list/tuple, got {type(values)}")
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return (float(values[0]), float(values[1]))
    raise ValueError(f"{arg_name} must contain exactly 1 or 2 values, got {values}")


def _pack_cli_int_scalar_or_list(values, arg_name: str):
    """
    将 argparse 读入的 int 列表整理为：
    - 1 个值 -> int
    - 多个值 -> list[int]

    这里用于 drift_n_points。
    """
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{arg_name} must be list/tuple, got {type(values)}")
    if len(values) == 0:
        raise ValueError(f"{arg_name} cannot be empty")
    out = [int(v) for v in values]
    if len(out) == 1:
        return out[0]
    return out


def build_mapstyle_cfg(args, is_train: bool) -> PackedMultiModalConfig:
    """
    按当前 loader 的真实字段名构建配置。

    关键点
    ------
    1) 融合分类固定使用：RGB + MindRove。
    2) 分类脚本不做 two-view：
       - rgb_two_views = False
       - mindrove_two_views = False
    3) RGB 的 normalization / spatial augmentation 交给 dataloader：
       - 本函数只把命令行参数封装进 PackedMultiModalConfig；
       - 训练循环中不会再次 Normalize，避免重复标准化。
    4) MindRove 的标准化和增强也交给 dataloader：
       - 先重采样；
       - 再做可选标准化；
       - 再做训练期增强。
    """
    rgb_hw = (args.rgb_size, args.rgb_size)
    depth_hw = (args.depth_size, args.depth_size)

    # ------------------------------------------------------------
    # 训练增强总开关
    # ------------------------------------------------------------
    # 与单模态脚本保持一致：
    # - 默认不改变原始增强行为；
    # - 只有训练集 is_train=True 且用户显式传入 --disable_train_augmentation 时，
    #   才把 RGB 随机裁剪退化为不裁剪，并关闭 RGB/MindRove 随机增强。
    # - 验证/测试集仍然由 dataloader 的 is_train=False 路径控制。
    if is_train and bool(args.disable_train_augmentation):
        rrc_scale = (1.0, 1.0)
        rrc_ratio = (1.0, 1.0)
        rgb_apply_spatial_aug = False
        rgb_hflip_p = 0.0
        rgb_vflip_p = 0.0
        rgb_jitter_p = 0.0
        rgb_gray_p = 0.0
        rgb_blur_p = 0.0
        mindrove_apply_augmentation = False
    else:
        rrc_scale = (args.rrc_scale_min, args.rrc_scale_max)
        rrc_ratio = (args.rrc_ratio_min, args.rrc_ratio_max)
        rgb_apply_spatial_aug = bool(args.rgb_apply_spatial_aug)
        rgb_hflip_p = args.rgb_hflip_p
        rgb_vflip_p = args.rgb_vflip_p
        rgb_jitter_p = args.rgb_jitter_p
        rgb_gray_p = args.rgb_gray_p
        rgb_blur_p = args.rgb_blur_p
        mindrove_apply_augmentation = bool(args.mindrove_apply_augmentation)

    cfg = PackedMultiModalConfig(
        # -------- common --------
        n_frames=args.n_frames,
        rgb_two_views=False,
        use_modalities=("rgb", "mindrove"),
        missing_policy="skip",
        load_labels=True,
        label_map_path=args.label_map_json,
        tier_mode=args.tier_mode,
        is_train=is_train,

        # -------- rgb / depth --------
        rgb_out_hw=rgb_hw,

        # RGB Normalize 参数。这里使用 tuple，避免后续 config 内部误改原 args list。
        rgb_mean=tuple(float(x) for x in args.rgb_mean),
        rgb_std=tuple(float(x) for x in args.rgb_std),

        # RGB RandomResizedCrop 参数。
        rrc_scale=rrc_scale,
        rrc_ratio=rrc_ratio,

        # RGB 随机空间增强参数。
        rgb_apply_spatial_aug=rgb_apply_spatial_aug,
        rgb_hflip_p=rgb_hflip_p,
        rgb_vflip_p=rgb_vflip_p,
        rgb_jitter_p=rgb_jitter_p,
        rgb_jitter_brightness=args.rgb_jitter_brightness,
        rgb_jitter_contrast=args.rgb_jitter_contrast,
        rgb_jitter_saturation=args.rgb_jitter_saturation,
        rgb_jitter_hue=args.rgb_jitter_hue,
        rgb_gray_p=rgb_gray_p,
        rgb_blur_p=rgb_blur_p,
        rgb_blur_kernel=args.rgb_blur_kernel,
        rgb_blur_sigma=(args.rgb_blur_sigma_min, args.rgb_blur_sigma_max),

        depth_out_hw=depth_hw,
        default_rgb_hw=(256, 256),
        default_depth_hw=depth_hw,

        # -------- mindrove --------
        mindrove_two_views=False,
        mindrove_target_len=args.mindrove_target_len,
        # 允许 EMG / IMU 使用不同的输入长度。
        # 这两个字段需要配合支持 mindrove_emg_target_len / mindrove_imu_target_len
        # 的新版 dataloader 使用；若为 None，则 dataloader 会回退到 mindrove_target_len。
        mindrove_emg_target_len=args.mindrove_emg_target_len,
        mindrove_imu_target_len=args.mindrove_imu_target_len,
        mindrove_hands=tuple(args.mindrove_hands),
        mindrove_signals=tuple(get_selected_signal_types(args)),
        mindrove_merge_hands=bool(args.mindrove_merge_hands),
        mindrove_apply_augmentation=mindrove_apply_augmentation,
        mindrove_apply_normalization=bool(args.mindrove_apply_normalization),

        mindrove_left_emg_mean=args.mindrove_left_emg_mean,
        mindrove_left_emg_std=args.mindrove_left_emg_std,
        mindrove_right_emg_mean=args.mindrove_right_emg_mean,
        mindrove_right_emg_std=args.mindrove_right_emg_std,
        mindrove_left_imu_mean=args.mindrove_left_imu_mean,
        mindrove_left_imu_std=args.mindrove_left_imu_std,
        mindrove_right_imu_mean=args.mindrove_right_imu_mean,
        mindrove_right_imu_std=args.mindrove_right_imu_std,

        mindrove_time_warp_prob=args.mindrove_time_warp_prob,
        mindrove_time_warp_sigma=args.mindrove_time_warp_sigma,
        mindrove_time_warp_num_knots=args.mindrove_time_warp_num_knots,
        mindrove_time_warp_num_splines=args.mindrove_time_warp_num_splines,

        mindrove_emg_scaling_prob=args.mindrove_emg_scaling_prob,
        mindrove_emg_scaling_sigma=args.mindrove_emg_scaling_sigma,
        mindrove_emg_noise_prob=args.mindrove_emg_noise_prob,
        mindrove_emg_noise_sigma=args.mindrove_emg_noise_sigma,
        mindrove_emg_drift_prob=args.mindrove_emg_drift_prob,
        mindrove_emg_drift_max=_pack_cli_float_scalar_or_pair(args.mindrove_emg_drift_max, "mindrove_emg_drift_max"),
        mindrove_emg_drift_n_points=_pack_cli_int_scalar_or_list(args.mindrove_emg_drift_n_points, "mindrove_emg_drift_n_points"),
        mindrove_emg_drift_kind=args.mindrove_emg_drift_kind,
        mindrove_emg_drift_per_channel=bool(args.mindrove_emg_drift_per_channel),
        mindrove_emg_drift_normalize=bool(args.mindrove_emg_drift_normalize),
        mindrove_emg_negate_prob=args.mindrove_emg_negate_prob,
        mindrove_emg_channel_dropout_prob=args.mindrove_emg_channel_dropout_prob,
        mindrove_emg_channel_dropout_max_channels=args.mindrove_emg_channel_dropout_max_channels,

        mindrove_imu_scaling_prob=args.mindrove_imu_scaling_prob,
        mindrove_imu_scaling_sigma=args.mindrove_imu_scaling_sigma,
        mindrove_imu_noise_prob=args.mindrove_imu_noise_prob,
        mindrove_imu_noise_sigma=args.mindrove_imu_noise_sigma,
        mindrove_imu_drift_prob=args.mindrove_imu_drift_prob,
        mindrove_imu_drift_max=_pack_cli_float_scalar_or_pair(args.mindrove_imu_drift_max, "mindrove_imu_drift_max"),
        mindrove_imu_drift_n_points=_pack_cli_int_scalar_or_list(args.mindrove_imu_drift_n_points, "mindrove_imu_drift_n_points"),
        mindrove_imu_drift_kind=args.mindrove_imu_drift_kind,
        mindrove_imu_drift_per_channel=bool(args.mindrove_imu_drift_per_channel),
        mindrove_imu_drift_normalize=bool(args.mindrove_imu_drift_normalize),
        mindrove_imu_negate_prob=args.mindrove_imu_negate_prob,
        mindrove_imu_channel_dropout_prob=args.mindrove_imu_channel_dropout_prob,
        mindrove_imu_channel_dropout_max_channels=args.mindrove_imu_channel_dropout_max_channels,
    )
    return cfg

def build_one_mapstyle_dataset_and_loader(
    args,
    manifest_arg: str,
    is_train: bool,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    drop_last: bool,
    sampler=None,
):
    """
    直接沿用你当前单模态脚本的构建方式，只是 use_modalities 改为 RGB + mindrove。
    """
    label_map = load_label_map_json(args.label_map_json)
    manifest_name = resolve_manifest_arg(args.dataset_root, manifest_arg)
    cfg = build_mapstyle_cfg(args, is_train=is_train)

    dataset = build_packed_mapstyle_dataset(
        dataset_root=args.dataset_root,
        manifest_name=manifest_name,
        cfg=cfg,
        label_map=label_map,
        verify_paths_on_init=True,
    )

    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        sampler=sampler,
        pin_memory=False,
    )
    return dataset, loader

def prepare_train_val_loaders_for_manifests(args, train_manifest: str, val_manifest: Optional[str]):
    train_sampler = None
    train_shuffle = True

    train_dataset, _ = build_one_mapstyle_dataset_and_loader(
        args=args,
        manifest_arg=train_manifest,
        is_train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers_train,
        prefetch_factor=args.prefetch_factor_train,
        shuffle=True,
        drop_last=False,
        sampler=None,
    )

    if args.use_weighted_sampler:
        train_sampler, _sampler_info = build_weighted_sampler_for_packed_dataset(
            dataset=train_dataset,
            tier_for_sampling=args.sampler_tier,
            mode=args.sampler_mode,
            replacement=True,
            num_samples=len(train_dataset),
            verbose=True,
        )
        train_shuffle = False

    trainloader = build_packed_mapstyle_loader_from_dataset(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers_train,
        shuffle=train_shuffle,
        drop_last=True,
        prefetch_factor=args.prefetch_factor_train,
        sampler=train_sampler,
        pin_memory=False,
    )

    valloader = None
    if (not args.disable_val) and (val_manifest is not None):
        _val_dataset, valloader = build_one_mapstyle_dataset_and_loader(
            args=args,
            manifest_arg=val_manifest,
            is_train=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers_val,
            prefetch_factor=args.prefetch_factor_val,
            shuffle=False,
            drop_last=False,
            sampler=None,
        )

    return trainloader, valloader, train_dataset

def prepare_test_loader_for_manifest(args, test_manifest: str):
    _test_dataset, testloader = build_one_mapstyle_dataset_and_loader(
        args=args,
        manifest_arg=test_manifest,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers_test,
        prefetch_factor=args.prefetch_factor_test,
        shuffle=False,
        drop_last=False,
        sampler=None,
    )
    return testloader


# ============================================================
# 3) batch extraction
# ============================================================

def build_mindrove_input_keys(args, signal_type: str) -> List[str]:
    """
    返回某一个 signal 分支需要从 batch["mindrove"] 中读取的 key 顺序。

    当 mindrove_merge_hands=False：
      EMG -> ["left_emg", "right_emg"]
      IMU -> ["left_imu", "right_imu"]

    当 mindrove_merge_hands=True：
      EMG -> ["emg"]
      IMU -> ["imu"]
    """
    if args.mindrove_merge_hands:
        return [signal_type]
    return [f"{hand}_{signal_type}" for hand in args.mindrove_hands]


def get_signal_in_channels(args, signal_type: str) -> int:
    """
    自动计算某一个 1D backbone 的输入通道数。

    注意：EMG 和 IMU 使用独立 backbone，因此这里返回的是单个 signal 分支的通道数，
    而不是把 EMG+IMU 拼成一个大通道输入。
    """
    base = MINDROVE_SIGNAL_CHANNELS[signal_type]
    return base * len(args.mindrove_hands)


def concat_one_signal_batch_dict(batch_mindrove: dict, args, signal_type: str) -> torch.Tensor:
    """
    将 batch["mindrove"] 中属于某个 signal 的左右手数据拼接成一个 1D backbone 输入。

    返回
    ----
    Tensor[B, C_total, L]
        例如：
        - EMG 左右手：8+8=16 通道
        - IMU 左右手：6+6=12 通道
    """
    if not isinstance(batch_mindrove, dict):
        raise TypeError(f"Expect batch['mindrove'] to be dict, got {type(batch_mindrove)}")

    keys = build_mindrove_input_keys(args, signal_type=signal_type)
    missing = [k for k in keys if k not in batch_mindrove]
    if missing:
        raise KeyError(
            f"Missing MindRove keys for signal_type={signal_type}: {missing}; "
            f"available={sorted(batch_mindrove.keys())}"
        )

    tensors = []
    ref_bl = None
    for k in keys:
        x = batch_mindrove[k]
        if not torch.is_tensor(x):
            raise TypeError(f"batch['mindrove']['{k}'] must be Tensor, got {type(x)}")
        if x.ndim != 3:
            raise ValueError(f"batch['mindrove']['{k}'] must be [B,C,L], got {tuple(x.shape)}")

        if ref_bl is None:
            ref_bl = (x.shape[0], x.shape[2])
        else:
            if (x.shape[0], x.shape[2]) != ref_bl:
                raise ValueError(
                    f"Inconsistent MindRove shapes, key={k}, got {tuple(x.shape)}, expected B,L={ref_bl}"
                )
        tensors.append(x)

    return torch.cat(tensors, dim=1).contiguous()


def extract_multimodal_inputs_and_labels(batch: dict, tier_mode: str, args):
    """
    从 dataloader batch 中取出：
      - rgb_inputs: Tensor[B,T,C,H,W]
      - signal_inputs: dict[str, Tensor[B,C,L]]，例如 {"emg": ..., "imu": ...}
      - labels
      - clip_ids

    这里是 RGB+EMG+IMU 支持的关键入口：
    dataloader 负责一次性读出所有 requested mindrove_signals，
    本函数再把 EMG 和 IMU 拆给各自独立的 1D backbone。
    """
    clip_ids = batch.get("key", None)
    if clip_ids is None:
        clip_ids = batch.get("sample_name", None)
    if clip_ids is None:
        raise KeyError("Batch has neither 'key' nor 'sample_name'.")

    tier_ids = batch["tier_ids"]
    labels = tier_ids[tier_mode]

    if "rgb" not in batch:
        raise KeyError("Fusion script requires batch['rgb']")
    if "mindrove" not in batch:
        raise KeyError("Fusion script requires batch['mindrove']")

    rgb_inputs = batch["rgb"]
    if isinstance(rgb_inputs, tuple):
        raise ValueError("This classification script expects rgb_two_views=False, but batch['rgb'] is tuple")

    mr = batch["mindrove"]
    if isinstance(mr, tuple):
        raise ValueError("This classification script expects mindrove_two_views=False, but batch['mindrove'] is tuple")

    signal_inputs = {
        sig: concat_one_signal_batch_dict(mr, args, signal_type=sig)
        for sig in get_selected_signal_types(args)
    }
    return rgb_inputs, signal_inputs, labels, clip_ids


def ensure_bcthw(x_btchw: torch.Tensor) -> torch.Tensor:
    if x_btchw.ndim != 5:
        raise ValueError(f"Expect RGB input [B,T,C,H,W], got {tuple(x_btchw.shape)}")
    return x_btchw.permute(0, 2, 1, 3, 4).contiguous()


def preprocess_rgb(x_btchw: torch.Tensor) -> torch.Tensor:
    if x_btchw.dtype != torch.float32:
        x_btchw = x_btchw.to(torch.float32)
    return x_btchw


def preprocess_signal(x_bcl: torch.Tensor) -> torch.Tensor:
    if x_bcl.ndim != 3:
        raise ValueError(f"Expect signal input [B,C,L], got {tuple(x_bcl.shape)}")
    if x_bcl.dtype != torch.float32:
        x_bcl = x_bcl.to(torch.float32)
    return x_bcl.contiguous()


def move_and_prepare_multimodal_inputs(rgb_inputs, signal_inputs: Dict[str, torch.Tensor], device, non_blocking: bool = True):
    """
    将 RGB 和每个 signal 分支移动到 device，并整理成模型输入格式。

    RGB:    [B,T,C,H,W] -> [B,C,T,H,W]
    Signal: [B,C,L] 保持不变
    """
    rgb_inputs = ensure_bcthw(preprocess_rgb(rgb_inputs)).to(device, non_blocking=non_blocking)
    signal_inputs = {
        sig: preprocess_signal(x).to(device, non_blocking=non_blocking)
        for sig, x in signal_inputs.items()
    }
    return rgb_inputs, signal_inputs

# ============================================================
# 4) model
# ============================================================

class RGBSignalFusionNet(nn.Module):
    """
    顶层统一模型，支持 RGB + 一个或多个 MindRove signal 分支。

    forward 输入
    ------------
    rgb_x:
        Tensor[B,C,T,H,W]

    signal_x:
        dict[str, Tensor[B,C,L]]
        例如：
        - RGB+EMG:      {"emg": emg_tensor}
        - RGB+IMU:      {"imu": imu_tensor}
        - RGB+EMG+IMU:  {"emg": emg_tensor, "imu": imu_tensor}

    模型结构
    --------
    rgb_x      -> rgb_backbone.forward_features()      -> rgb_feat
    emg_x      -> signal_backbones["emg"].forward_features() -> emg_feat
    imu_x      -> signal_backbones["imu"].forward_features() -> imu_feat
    {rgb, emg, imu} -> fusion module -> classifier
    """
    def __init__(
        self,
        rgb_backbone: nn.Module,
        signal_backbones: Dict[str, nn.Module],
        fusion_module: nn.Module,
        fusion_output_dim: int,
        num_classes: int,
        head_hidden_dim: int,
        classifier_dropout: float = 0.0,
    ):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.signal_backbones = nn.ModuleDict(signal_backbones)

        # 向后兼容：如果只有一个 signal 分支，仍然提供 model.signal_backbone 属性。
        # 旧脚本外部若读取该属性不会立即失败；但新代码统一使用 signal_backbones。
        if len(self.signal_backbones) == 1:
            only_key = next(iter(self.signal_backbones.keys()))
            self.signal_backbone = self.signal_backbones[only_key]

        self.fusion = fusion_module
        self.fusion_output_dim = int(fusion_output_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_output_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )
        # 统一暴露 fc，方便 head_only / 外部分析脚本继续按 fc 查找分类头。
        self.fc = self.classifier

    def forward_features(self, rgb_x: torch.Tensor, signal_x: Dict[str, torch.Tensor]):
        rgb_feat = self.rgb_backbone.forward_features(rgb_x)
        signal_feats = {}
        for sig_name, backbone in self.signal_backbones.items():
            if sig_name not in signal_x:
                raise KeyError(
                    f"Missing signal input '{sig_name}' for model forward. "
                    f"Available signal inputs: {sorted(signal_x.keys())}"
                )
            signal_feats[sig_name] = backbone.forward_features(signal_x[sig_name])
        return rgb_feat, signal_feats

    def forward(self, rgb_x: torch.Tensor, signal_x: Dict[str, torch.Tensor], return_details: bool = False):
        rgb_feat, signal_feats = self.forward_features(rgb_x, signal_x)

        # insertion order 与 build_fusion_module(input_dims) 一致：rgb -> emg -> imu。
        modalities = {"rgb": rgb_feat}
        modalities.update(signal_feats)

        fused, fusion_info, projected = self.fusion(modalities, return_projected=True)
        logits = self.classifier(fused)

        if return_details:
            return {
                "logits": logits,
                "fused": fused,
                "fusion_info": fusion_info,
                "projected": projected,
                "rgb_feat": rgb_feat,
                "signal_feats": signal_feats,
                **{f"{sig}_feat": feat for sig, feat in signal_feats.items()},
            }
        return logits


def build_rgb_backbone(args) -> nn.Module:
    model = resnet3d.generate_model(
        args.model_depth,
        n_input_channels=args.rgb_n_input_channels,
        conv1_t_size=args.rgb_conv1_t_size,
        conv1_t_stride=args.rgb_conv1_t_stride,
        no_max_pool=args.rgb_no_max_pool,
        shortcut_type=args.rgb_shortcut_type,
        widen_factor=args.rgb_widen_factor,
        num_classes=args.num_classes,
    )
    if not hasattr(model, "forward_features"):
        raise AttributeError("RGB backbone must implement forward_features(x)")
    if not hasattr(model, "feature_dim"):
        if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
            model.feature_dim = int(model.fc.in_features)
        else:
            raise AttributeError("RGB backbone must provide feature_dim or fc.in_features")
    return model


def build_signal_backbone(args, signal_type: str) -> nn.Module:
    in_channels = get_signal_in_channels(args, signal_type=signal_type)
    model = build_resnet1d(
        arch=args.mindrove_arch,
        in_channels=in_channels,
        num_classes=args.num_classes,
        base_channels=args.mindrove_base_channels,
        stem_kernel_size=args.mindrove_stem_kernel_size,
        stem_stride=args.mindrove_stem_stride,
        use_stem_pool=args.mindrove_use_stem_pool,
        zero_init_residual=args.mindrove_zero_init_residual,
    )
    if not hasattr(model, "forward_features"):
        raise AttributeError(f"{signal_type.upper()} backbone must implement forward_features(x)")
    if not hasattr(model, "feature_dim"):
        if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
            model.feature_dim = int(model.fc.in_features)
        else:
            raise AttributeError(f"{signal_type.upper()} backbone must provide feature_dim or fc.in_features")
    return model


def build_fusion_module(args, input_dims: Dict[str, int]) -> Tuple[nn.Module, int]:
    """
    根据 input_dims 构建 fusion module。

    input_dims 示例：
      RGB+EMG:     {"rgb": 512, "emg": 512}
      RGB+IMU:     {"rgb": 512, "imu": 512}
      RGB+EMG+IMU: {"rgb": 512, "emg": 512, "imu": 512}

    你上传的 concat_mlp / gated / weighted_sum 模块本身都支持 dict[str, dim]
    形式的任意多模态输入，因此这里不需要改 fusion module 文件。
    """
    if args.fusion_type == "concat_mlp":
        fusion = ConcatMLPFusion(
            input_dims=input_dims,
            hidden_dim=args.fusion_hidden_dim,
            mlp_hidden_dim=args.concat_mlp_hidden_dim,
            fusion_out_dim=args.concat_fusion_out_dim,
            use_projection=args.fusion_use_projection,
            activation=args.fusion_activation,
            use_pre_bn=args.fusion_use_pre_bn,
            projection_dropout=args.concat_projection_dropout,
            mlp_dropout=args.fusion_dropout,
        )
        out_dim = int(args.concat_fusion_out_dim)

    elif args.fusion_type == "gated":
        fusion = GatedFusion(
            input_dims=input_dims,
            hidden_dim=args.fusion_hidden_dim,
            gate_type=args.gated_gate_type,
            activation=args.fusion_activation,
            use_projection=args.fusion_use_projection,
            use_pre_bn=args.fusion_use_pre_bn,
            use_post_fusion_proj=False,
            dropout=args.fusion_dropout,
        )
        out_dim = int(fusion.fusion_dim)

    elif args.fusion_type == "weighted_sum":
        fusion = WeightedSumFusion(
            input_dims=input_dims,
            hidden_dim=args.fusion_hidden_dim,
            sum_method=args.weighted_sum_method,
            normalize=args.weighted_sum_normalize,
            use_projection=args.fusion_use_projection,
            activation=args.fusion_activation,
            use_pre_bn=args.fusion_use_pre_bn,
            dropout=args.fusion_dropout,
        )
        out_dim = int(fusion.fusion_dim)

    else:
        raise ValueError(f"Unsupported fusion_type: {args.fusion_type}")

    return fusion, out_dim


def prepare_model(args) -> RGBSignalFusionNet:
    selected_signals = get_selected_signal_types(args)

    rgb_backbone = build_rgb_backbone(args)
    signal_backbones = {
        sig: build_signal_backbone(args, signal_type=sig)
        for sig in selected_signals
    }

    input_dims: Dict[str, int] = {"rgb": int(rgb_backbone.feature_dim)}
    for sig in selected_signals:
        input_dims[sig] = int(signal_backbones[sig].feature_dim)

    fusion_module, fusion_out_dim = build_fusion_module(
        args=args,
        input_dims=input_dims,
    )
    model = RGBSignalFusionNet(
        rgb_backbone=rgb_backbone,
        signal_backbones=signal_backbones,
        fusion_module=fusion_module,
        fusion_output_dim=fusion_out_dim,
        num_classes=args.num_classes,
        head_hidden_dim=args.head_hidden_dim,
        classifier_dropout=args.classifier_dropout,
    )
    return model


# ============================================================
# 5) checkpoint loading
# ============================================================

def extract_state_dict_from_checkpoint(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise TypeError("Unable to extract state_dict from checkpoint")

def strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "encoder_q.", "encoder.", "backbone.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out

def should_drop_pretrained_key(key: str) -> bool:
    return any(token in key for token in HEAD_KEYWORDS)

def normalize_and_filter_state_dict(
    raw_state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
    drop_pretrained_head: bool,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    stripped = strip_common_prefixes(raw_state_dict)

    filtered = {}
    dropped_by_head = []
    shape_mismatch = []
    missing_in_model = []

    for k, v in stripped.items():
        if drop_pretrained_head and should_drop_pretrained_key(k):
            dropped_by_head.append(k)
            continue

        if k not in model_state_dict:
            missing_in_model.append(k)
            continue

        if tuple(v.shape) != tuple(model_state_dict[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(model_state_dict[k].shape)))
            continue

        filtered[k] = v

    report = {
        "num_raw_keys": len(raw_state_dict),
        "num_after_strip": len(stripped),
        "num_loaded": len(filtered),
        "dropped_by_head": dropped_by_head,
        "shape_mismatch": shape_mismatch,
        "missing_in_model": missing_in_model,
    }
    return filtered, report

def load_backbone_pretrained_weights(
    backbone: nn.Module,
    ckpt_path: Optional[str],
    map_location: str = "cpu",
    drop_pretrained_head: bool = True,
    strict: bool = False,
):
    if ckpt_path is None:
        return {
            "checkpoint_path": None,
            "num_model_keys": len(backbone.state_dict()),
            "num_loaded": 0,
            "dropped_by_head": [],
            "shape_mismatch": [],
            "missing_in_model": [],
            "missing_keys_after_load": [],
            "unexpected_keys_after_load": [],
        }

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location)
    raw_state_dict = extract_state_dict_from_checkpoint(ckpt)
    model_state_dict = backbone.state_dict()

    filtered_state_dict, report = normalize_and_filter_state_dict(
        raw_state_dict=raw_state_dict,
        model_state_dict=model_state_dict,
        drop_pretrained_head=drop_pretrained_head,
    )

    load_msg = backbone.load_state_dict(filtered_state_dict, strict=strict)

    return {
        "checkpoint_path": ckpt_path,
        "num_model_keys": len(model_state_dict),
        **report,
        "missing_keys_after_load": list(load_msg.missing_keys),
        "unexpected_keys_after_load": list(load_msg.unexpected_keys),
    }

def load_full_model_for_eval(model: nn.Module, ckpt_path: str, map_location: str = "cpu"):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for evaluation: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location)
    raw_state_dict = extract_state_dict_from_checkpoint(ckpt)
    model_state_dict = model.state_dict()

    filtered_state_dict, report = normalize_and_filter_state_dict(
        raw_state_dict=raw_state_dict,
        model_state_dict=model_state_dict,
        drop_pretrained_head=False,
    )

    load_msg = model.load_state_dict(filtered_state_dict, strict=False)

    return {
        "checkpoint_path": ckpt_path,
        "num_model_keys": len(model_state_dict),
        **report,
        "missing_keys_after_load": list(load_msg.missing_keys),
        "unexpected_keys_after_load": list(load_msg.unexpected_keys),
    }


# ============================================================
# 6) finetune modes / optimizer / scheduler
# ============================================================

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag

def configure_finetune_mode(model: RGBSignalFusionNet, finetune_mode: str):
    if finetune_mode == "full":
        set_requires_grad(model, True)
        return
    if finetune_mode == "fusion_head":
        set_requires_grad(model.rgb_backbone, False)
        set_requires_grad(model.signal_backbones, False)
        set_requires_grad(model.fusion, True)
        set_requires_grad(model.classifier, True)
        return
    if finetune_mode == "head_only":
        set_requires_grad(model.rgb_backbone, False)
        set_requires_grad(model.signal_backbones, False)
        set_requires_grad(model.fusion, False)
        set_requires_grad(model.classifier, True)
        return
    raise ValueError(f"Unknown finetune_mode: {finetune_mode}")


def build_optimizer(model: RGBSignalFusionNet, args):
    configure_finetune_mode(model, args.finetune_mode)

    backbone_lr = args.backbone_learning_rate if args.backbone_learning_rate is not None else args.learning_rate
    fusion_lr = args.fusion_learning_rate if args.fusion_learning_rate is not None else args.learning_rate
    head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate

    if args.finetune_mode == "head_only":
        param_groups = [{
            "params": [p for p in model.classifier.parameters() if p.requires_grad],
            "lr": head_lr,
            "initial_lr": head_lr,
            "group_name": "classifier",
        }]
        mode = "head_only"

    elif args.finetune_mode == "fusion_head":
        param_groups = [
            {
                "params": [p for p in model.fusion.parameters() if p.requires_grad],
                "lr": fusion_lr,
                "initial_lr": fusion_lr,
                "group_name": "fusion",
            },
            {
                "params": [p for p in model.classifier.parameters() if p.requires_grad],
                "lr": head_lr,
                "initial_lr": head_lr,
                "group_name": "classifier",
            }
        ]
        mode = "fusion_head"

    else:
        if args.use_discriminative_lr:
            # backbone 组包含 RGB backbone + 所有 active signal backbone。
            backbone_params = [p for p in model.rgb_backbone.parameters() if p.requires_grad]
            for _sig_name, sig_backbone in model.signal_backbones.items():
                backbone_params.extend([p for p in sig_backbone.parameters() if p.requires_grad])

            param_groups = [
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "initial_lr": backbone_lr,
                    "group_name": "backbone",
                },
                {
                    "params": [p for p in model.fusion.parameters() if p.requires_grad],
                    "lr": fusion_lr,
                    "initial_lr": fusion_lr,
                    "group_name": "fusion",
                },
                {
                    "params": [p for p in model.classifier.parameters() if p.requires_grad],
                    "lr": head_lr,
                    "initial_lr": head_lr,
                    "group_name": "classifier",
                },
            ]
            mode = "full_triple_lr"
        else:
            param_groups = [{
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": args.learning_rate,
                "initial_lr": args.learning_rate,
                "group_name": "all",
            }]
            mode = "full_single_lr"

    # ------------------------------------------------------------
    # Optimizer selection
    # ------------------------------------------------------------
    # 参数分组和优化器类型解耦：
    # - 上面先根据 finetune_mode / use_discriminative_lr 决定哪些参数可训练、每组初始 lr；
    # - 这里再根据 --optimizer 选择 SGD 或 Adam。
    # 每个 param_group 已经显式包含 lr，因此 optimizer 构造时不再依赖单一全局 lr。
    optimizer_name = str(args.optimizer).lower().strip()

    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        optimizer_momentum = float(args.momentum)

    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            param_groups,
            weight_decay=args.weight_decay,
        )
        # Adam 不使用 SGD momentum 参数；这里记录 None，避免 summary 误读。
        optimizer_momentum = None

    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    meta = {
        "mode": mode,
        "optimizer": optimizer_name,
        "weight_decay": float(args.weight_decay),
        "momentum": optimizer_momentum,
        "backbone_lr": backbone_lr if args.finetune_mode == "full" else None,
        "fusion_lr": fusion_lr if args.finetune_mode in ("full", "fusion_head") else None,
        "head_lr": head_lr,
        "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return optimizer, meta

def adjust_learning_rate(optimizer, epoch: int, args):
    if args.cos:
        for param_group in optimizer.param_groups:
            base_lr = param_group["initial_lr"]
            param_group["lr"] = base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        for param_group in optimizer.param_groups:
            base_lr = param_group["initial_lr"]
            lr = base_lr
            for milestone in args.schedules:
                if epoch >= milestone:
                    lr *= 0.1
            param_group["lr"] = lr


# ============================================================
# 7) criterion / metrics
# ============================================================

def build_class_counts_from_mapstyle_dataset(dataset, tier_mode: str, num_classes: int):
    """
    直接沿用原单模态脚本的做法：从 dataset.records + dataset.label_map 统计，不走 __getitem__。
    """
    if not hasattr(dataset, "records"):
        raise TypeError("dataset must have attribute 'records'.")
    if not hasattr(dataset, "label_map"):
        raise TypeError("dataset must have attribute 'label_map'.")
    if tier_mode not in ("tier1", "tier2", "tier3"):
        raise ValueError(f"tier_mode must be one of ('tier1','tier2','tier3'), got {tier_mode}")
    if tier_mode not in dataset.label_map:
        raise KeyError(f"dataset.label_map does not contain key '{tier_mode}'")

    tier_label_map = dataset.label_map[tier_mode]
    counts = [0] * num_classes
    bad_indices = []

    for i, rec in enumerate(dataset.records):
        action_name = rec.get(tier_mode, None)
        if action_name is None:
            bad_indices.append(i)
            continue

        class_id = int(tier_label_map.get(str(action_name), -1))
        if class_id < 0 or class_id >= num_classes:
            bad_indices.append(i)
            continue

        counts[class_id] += 1

    if bad_indices:
        raise ValueError(
            f"Found {len(bad_indices)} samples with invalid label for {tier_mode}. "
            f"Example bad indices: {bad_indices[:10]}"
        )

    return counts

def compute_class_weights_from_counts(
    counts,
    method: str,
    beta: float = 0.999,
    normalize_mean: bool = False,
    eps: float = 1e-12,
):
    if isinstance(counts, torch.Tensor):
        counts_t = counts.to(torch.float32)
    else:
        counts_t = torch.tensor(counts, dtype=torch.float32)

    counts_t = torch.clamp(counts_t, min=1.0)

    if method == "inv_freq":
        w = 1.0 / (counts_t + eps)
    elif method == "class_balanced":
        w = (1.0 - beta) / (1.0 - torch.pow(beta, counts_t) + eps)
    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize_mean:
        w = w / (w.mean() + eps)
    return w

def build_criterion(args, train_dataset=None, device=None):
    class_weights = None
    if args.use_weighted_ce or (args.use_focal and args.focal_use_alpha):
        if train_dataset is None:
            raise ValueError("train_dataset is required when using weighted CE or focal alpha")

        counts = build_class_counts_from_mapstyle_dataset(
            dataset=train_dataset,
            tier_mode=args.tier_mode,
            num_classes=args.num_classes,
        )
        class_weights = compute_class_weights_from_counts(
            counts=counts,
            method=args.weight_method,
            beta=args.cb_beta,
            normalize_mean=args.weight_normalize_mean,
        )
        if device is not None:
            class_weights = class_weights.to(device)

    if args.use_focal:
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            alpha=class_weights if args.focal_use_alpha else None,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights if args.use_weighted_ce else None
        )

    return criterion, class_weights

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item())


# ============================================================
# 8) train / val / test
# ============================================================

def amp_context(device, enable_amp: bool, amp_dtype):
    if enable_amp and device.type == "cuda":
        return autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()

def run_one_epoch_train(
    model: RGBSignalFusionNet,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch: int,
    args,
    amp_dtype,
    td_logger: Optional[TrainingDynamicsLogger] = None,
):
    model.train()

    loss_meter = 0.0
    acc_meter = 0.0
    num_samples = 0

    pbar = tqdm.tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for batch in pbar:
        rgb_inputs, signal_inputs, labels, clip_ids = extract_multimodal_inputs_and_labels(
            batch=batch, tier_mode=args.tier_mode, args=args
        )

        rgb_inputs, signal_inputs = move_and_prepare_multimodal_inputs(
            rgb_inputs=rgb_inputs,
            signal_inputs=signal_inputs,
            device=device,
            non_blocking=True,
        )
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with amp_context(device, args.enable_amp, amp_dtype):
            logits = model(rgb_inputs, signal_inputs)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            bs = labels.size(0)
            acc = accuracy_from_logits(logits, labels)

            loss_meter += float(loss.item()) * bs
            acc_meter += acc * bs
            num_samples += bs

            if td_logger is not None:
                outputs = logits.detach().tolist()
                probs = torch.softmax(logits, dim=1)
                prob_true = probs.gather(1, labels.view(-1, 1)).squeeze(1).detach().cpu()
                td_preds = torch.argmax(logits, dim=-1).detach().cpu()
                td_labels = labels.detach().cpu()

                if not isinstance(clip_ids, list):
                    clip_ids = list(clip_ids)

                td_logger.log_minibatch(
                    uids=clip_ids,
                    golds=td_labels.tolist(),
                    logits=outputs,
                    probs_true=prob_true.tolist(),
                    preds=td_preds.tolist(),
                    epoch=epoch,
                )

            pbar.set_postfix({
                "loss": f"{loss_meter / max(num_samples, 1):.4f}",
                "acc": f"{acc_meter / max(num_samples, 1):.4f}",
            })

    return loss_meter / max(num_samples, 1), acc_meter / max(num_samples, 1), num_samples

@torch.no_grad()
def evaluate_one_epoch(
    model: RGBSignalFusionNet,
    loader,
    criterion,
    device,
    epoch: int,
    split_name: str,
    args,
    amp_dtype,
):
    model.eval()

    loss_meter = 0.0
    acc_meter = 0.0
    num_samples = 0

    pbar = tqdm.tqdm(loader, desc=f"{split_name} epoch {epoch}", leave=False)

    for batch in pbar:
        rgb_inputs, signal_inputs, labels, clip_ids = extract_multimodal_inputs_and_labels(
            batch=batch, tier_mode=args.tier_mode, args=args
        )

        rgb_inputs, signal_inputs = move_and_prepare_multimodal_inputs(
            rgb_inputs=rgb_inputs,
            signal_inputs=signal_inputs,
            device=device,
            non_blocking=True,
        )
        labels = labels.to(device, non_blocking=True).long()

        with amp_context(device, args.enable_amp, amp_dtype):
            logits = model(rgb_inputs, signal_inputs)
            loss = criterion(logits, labels)

        bs = labels.size(0)
        acc = accuracy_from_logits(logits, labels)

        loss_meter += float(loss.item()) * bs
        acc_meter += acc * bs
        num_samples += bs

        pbar.set_postfix({
            "loss": f"{loss_meter / max(num_samples, 1):.4f}",
            "acc": f"{acc_meter / max(num_samples, 1):.4f}",
        })

    return loss_meter / max(num_samples, 1), acc_meter / max(num_samples, 1), num_samples

def build_reverse_label_map(args) -> dict:
    """
    你的 label_map_json 是按 tier 分层的，所以这里只取当前 tier_mode 的反向映射。
    """
    label_map = load_label_map_json(args.label_map_json)
    if args.tier_mode not in label_map:
        raise KeyError(f"tier_mode={args.tier_mode} not found in label_map_json")
    return {int(v): str(k) for k, v in label_map[args.tier_mode].items()}


def add_fusion_info_to_row(row: dict, fusion_info, model: RGBSignalFusionNet, sample_index: int, args) -> None:
    """
    将 gated / weighted_sum 的融合权重写入 per-sample CSV 行。

    旧脚本写死了两个模态：rgb 和 signal。
    新脚本改成根据 model.fusion.modality_names 动态展开，因此可以自然支持：
      - rgb, emg
      - rgb, imu
      - rgb, emg, imu
    """
    if fusion_info is None:
        return

    modality_names = list(getattr(model.fusion, "modality_names", get_fusion_modality_names(args)))

    if args.fusion_type == "weighted_sum":
        # WeightedSumFusion 返回的是全局可学习权重，不随样本变化。
        if fusion_info.ndim == 1:
            for m_idx, name in enumerate(modality_names):
                row[f"fusion_weight_{name}"] = float(fusion_info[m_idx].item())
        elif fusion_info.ndim == 2:
            for m_idx, name in enumerate(modality_names):
                row[f"fusion_weight_{name}_mean"] = float(fusion_info[m_idx].mean().item())
        return

    if args.fusion_type == "gated":
        # GatedFusion 返回的是样本相关权重。
        if fusion_info.ndim == 2:
            # scalar gate: [B, M]
            for m_idx, name in enumerate(modality_names):
                row[f"gate_{name}"] = float(fusion_info[sample_index, m_idx].item())
        elif fusion_info.ndim == 3:
            # vector gate: [B, M, D]
            for m_idx, name in enumerate(modality_names):
                row[f"gate_{name}_mean"] = float(fusion_info[sample_index, m_idx].mean().item())
        return

@torch.no_grad()
def evaluate_test_with_per_sample_csv(
    model: RGBSignalFusionNet,
    loader,
    device,
    tier_mode: str,
    amp_dtype,
    reverse_label_map: dict,
    per_sample_csv_path: str,
    enable_amp: bool,
    args,
):
    model.eval()
    ensure_dir(Path(per_sample_csv_path).parent)

    rows = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm.tqdm(loader, desc="test", leave=False):
        rgb_inputs, signal_inputs, labels, clip_ids = extract_multimodal_inputs_and_labels(
            batch=batch, tier_mode=tier_mode, args=args
        )

        rgb_inputs, signal_inputs = move_and_prepare_multimodal_inputs(
            rgb_inputs=rgb_inputs,
            signal_inputs=signal_inputs,
            device=device,
            non_blocking=True,
        )
        labels = labels.to(device, non_blocking=True).long()

        with amp_context(device, enable_amp, amp_dtype):
            outputs = model(rgb_inputs, signal_inputs, return_details=True)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.size(0))

        fusion_info = outputs["fusion_info"]
        for i in range(labels.size(0)):
            true_id = int(labels[i].item())
            pred_id = int(preds[i].item())
            row = {
                "clip_id": clip_ids[i] if isinstance(clip_ids, (list, tuple)) else str(clip_ids),
                "true_label_id": true_id,
                "true_label_name": reverse_label_map.get(true_id, str(true_id)),
                "pred_label_id": pred_id,
                "pred_label_name": reverse_label_map.get(pred_id, str(pred_id)),
                "correct": int(pred_id == true_id),
                "max_prob": float(probs[i, pred_id].item()),
            }

            add_fusion_info_to_row(
                row=row,
                fusion_info=fusion_info,
                model=model,
                sample_index=i,
                args=args,
            )

            rows.append(row)

    with open(per_sample_csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_acc, avg_loss, total_samples


# ============================================================
# 9) experiment runner
# ============================================================

def build_run_name(args, run_index: int, source_tag: str, finetune_mode: str) -> str:
    """
    使用短目录名，避免 Windows 路径过长：
        run_01_a1b2c3d4
    """
    rand_suffix = f"{random.randint(0, 16**8 - 1):08x}"
    return f"run_{run_index:02d}_{rand_suffix}"

def save_checkpoint(path: str | Path, model: nn.Module, optimizer=None, epoch=None,
                    best_val_acc=None, extra: Optional[dict] = None):
    ensure_dir(Path(path).parent)
    payload = {"state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if best_val_acc is not None:
        payload["best_val_acc"] = best_val_acc
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)

def run_one_training_experiment(
    args,
    device,
    trainloader,
    valloader,
    train_dataset,
    run_index: int,
    rgb_pretrained_path: Optional[str],
    signal_pretrained_paths: Dict[str, Optional[str]],
    train_manifest_used: str,
    val_manifest_used: Optional[str],
    source_tag: str,
):
    """
    执行一次训练实验。

    这里最重要的变化是：
    - 旧脚本只加载一个 signal_pretrained_path；
    - 新脚本会按 signal_pretrained_paths 分别给 EMG / IMU backbone 加载权重。
    """
    model = prepare_model(args).to(device)

    rgb_load_report = load_backbone_pretrained_weights(
        model.rgb_backbone,
        rgb_pretrained_path,
        map_location="cpu",
        drop_pretrained_head=(not args.keep_pretrained_head),
        strict=args.pretrained_strict,
    )

    signal_load_reports = {}
    for sig_name, backbone in model.signal_backbones.items():
        sig_path = signal_pretrained_paths.get(sig_name, None)
        signal_load_reports[sig_name] = load_backbone_pretrained_weights(
            backbone,
            sig_path,
            map_location="cpu",
            drop_pretrained_head=(not args.keep_pretrained_head),
            strict=args.pretrained_strict,
        )

    optimizer, optimizer_meta = build_optimizer(model, args)
    criterion, class_weights = build_criterion(args, train_dataset=train_dataset, device=device)

    run_name = build_run_name(args, run_index, source_tag, args.finetune_mode)
    weight_dir = os.path.join(args.save_path, run_name)
    datamap_dir = os.path.join(args.datamap_csv_path, run_name)
    ensure_dir(weight_dir)
    ensure_dir(datamap_dir)

    save_json(os.path.join(weight_dir, "args.json"), vars(args))

    selected_signals = get_selected_signal_types(args)
    fusion_modalities = get_fusion_modality_names(args)

    meta = {
        "run_name": run_name,
        "source_tag": source_tag,
        "fusion_type": args.fusion_type,
        "fusion_modalities": fusion_modalities,
        "signal_type": args.signal_type,          # backward-compatible record
        "signal_types": selected_signals,         # new canonical record
        "mindrove_target_len": args.mindrove_target_len,
        "mindrove_emg_target_len": args.mindrove_emg_target_len,
        "mindrove_imu_target_len": args.mindrove_imu_target_len,
        "finetune_mode": args.finetune_mode,
        "train_manifest_used": train_manifest_used,
        "val_manifest_used": val_manifest_used,
        "rgb_pretrained_path": rgb_pretrained_path,
        "signal_pretrained_paths": signal_pretrained_paths,
        "meta_json_path": os.path.join(weight_dir, "meta.json"),
        "pretrained_tag_mode": args.pretrained_tag_mode,
        "pretrained_tag_last_k": args.pretrained_tag_last_k,
        "pretrained_tag_anchor": args.pretrained_tag_anchor,
        "save_path": args.save_path,
        "datamap_csv_path": args.datamap_csv_path,
        "optimizer": optimizer_meta.get("optimizer", args.optimizer),
        "optimizer_mode": optimizer_meta.get("mode"),
        "optimizer_weight_decay": optimizer_meta.get("weight_decay", args.weight_decay),
        "optimizer_momentum": optimizer_meta.get("momentum", None),
    }
    save_json(os.path.join(weight_dir, "meta.json"), meta)

    training_dynamics_logger = TrainingDynamicsLogger(datamap_dir)

    scaler = GradScaler("cuda") if (args.enable_amp and device.type == "cuda") else None
    amp_dtype = torch.bfloat16 if use_bf16_available() else torch.float16

    best_val_acc = -1.0
    best_val_epoch = -1
    best_val_path = os.path.join(weight_dir, "best_val.pth")
    last_ckpt_path = os.path.join(weight_dir, "last.pth")

    train_history = []

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch - 1, args)

        train_loss, train_acc, train_n = run_one_epoch_train(
            model=model,
            loader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            amp_dtype=amp_dtype,
            td_logger=training_dynamics_logger,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "num_train_samples": train_n,
        }

        if valloader is not None:
            val_loss, val_acc, val_n = evaluate_one_epoch(
                model=model,
                loader=valloader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                split_name="val",
                args=args,
                amp_dtype=amp_dtype,
            )
            row.update({
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_val_samples": val_n,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                save_checkpoint(
                    best_val_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_acc=best_val_acc,
                )
        else:
            val_loss = None
            val_acc = None

        if args.save_last_checkpoint:
            save_checkpoint(
                last_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc if best_val_epoch > 0 else None,
            )

        train_history.append(row)
        print(
            f"[{run_name}] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            + (f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}" if valloader is not None else "no_val")
        )

    save_csv_rows(os.path.join(weight_dir, "epoch_history.csv"), train_history, append=False)

    summary = {
        "run_name": run_name,
        "fusion_type": args.fusion_type,
        "fusion_modalities": fusion_modalities,
        "signal_type": args.signal_type,          # backward-compatible record
        "signal_types": selected_signals,         # new canonical record
        "mindrove_target_len": args.mindrove_target_len,
        "mindrove_emg_target_len": args.mindrove_emg_target_len,
        "mindrove_imu_target_len": args.mindrove_imu_target_len,
        "finetune_mode": args.finetune_mode,
        "train_manifest_used": train_manifest_used,
        "val_manifest_used": val_manifest_used,
        "rgb_pretrained_path": rgb_pretrained_path,
        "signal_pretrained_paths": signal_pretrained_paths,
        "best_val_acc": best_val_acc if best_val_epoch > 0 else None,
        "best_val_epoch": best_val_epoch if best_val_epoch > 0 else None,
        "best_val_path": best_val_path if best_val_epoch > 0 else None,
        "last_ckpt_path": last_ckpt_path if args.save_last_checkpoint else None,
        "optimizer_mode": optimizer_meta["mode"],
        "optimizer": optimizer_meta.get("optimizer", args.optimizer),
        "optimizer_weight_decay": optimizer_meta.get("weight_decay", args.weight_decay),
        "optimizer_momentum": optimizer_meta.get("momentum", None),
        "backbone_learning_rate": optimizer_meta["backbone_lr"],
        "fusion_learning_rate": optimizer_meta["fusion_lr"],
        "head_learning_rate": optimizer_meta["head_lr"],
        "num_trainable_params": optimizer_meta["num_trainable_params"],
        "rgb_num_loaded_keys": rgb_load_report["num_loaded"],
        "signal_num_loaded_keys": {
            sig: report["num_loaded"] for sig, report in signal_load_reports.items()
        },
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
    }

    save_json(os.path.join(weight_dir, "train_summary.json"), summary)
    return summary


# ============================================================
# 10) test mode
# ============================================================

def run_test_mode(args, device):
    pairs = build_test_pairs(args)
    reverse_label_map = build_reverse_label_map(args)
    amp_dtype = torch.bfloat16 if use_bf16_available() else torch.float16

    rows = []
    for pair in pairs:
        test_manifest_used = pair["test_manifest"]
        weight_path = pair["weight_path"]

        model = prepare_model(args).to(device)
        load_report = load_full_model_for_eval(model, weight_path, map_location="cpu")
        testloader = prepare_test_loader_for_manifest(args, test_manifest_used)

        save_dir = os.path.join(args.save_path, sanitize_tag(Path(weight_path).stem))
        ensure_dir(save_dir)
        per_sample_csv_path = os.path.join(
            save_dir,
            f"per_sample_{sanitize_tag(Path(test_manifest_used).stem)}.csv"
        )

        test_acc, test_loss, num_samples = evaluate_test_with_per_sample_csv(
            model=model,
            loader=testloader,
            device=device,
            tier_mode=args.tier_mode,
            amp_dtype=amp_dtype,
            reverse_label_map=reverse_label_map,
            per_sample_csv_path=per_sample_csv_path,
            enable_amp=bool(args.enable_amp),
            args=args,
        )

        rows.append({
            "weight_path": weight_path,
            "weight_name": Path(weight_path).name,
            "test_manifest_used": test_manifest_used,
            "tier_mode": args.tier_mode,
            "fusion_type": args.fusion_type,
            "fusion_modalities": json.dumps(get_fusion_modality_names(args), ensure_ascii=False),
            "signal_type": args.signal_type,
            "signal_types": json.dumps(get_selected_signal_types(args), ensure_ascii=False),
            "mindrove_target_len": args.mindrove_target_len,
            "mindrove_emg_target_len": args.mindrove_emg_target_len,
            "mindrove_imu_target_len": args.mindrove_imu_target_len,
            "num_samples": num_samples,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "num_loaded_keys": load_report["num_loaded"],
            "num_missing_after_load": len(load_report["missing_keys_after_load"]),
            "num_unexpected_after_load": len(load_report["unexpected_keys_after_load"]),
            "per_sample_csv": per_sample_csv_path,
        })

    csv_path = args.test_results_csv or os.path.join(args.save_path, "test_results.csv")
    save_csv_rows(csv_path, rows, append=True)
    print(f"[test results saved] {csv_path}")


# ============================================================
# 11) main
# ============================================================

def main(args):
    validate_args(args)

    if args.seed is not None:
        seed_everything(args.seed)
        print(f"using random seed: {args.seed}")
    else:
        print("training without explicit random seed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"use_bf16_supported: {use_bf16_available()}")

    if args.run_mode == "train":
        all_summaries = []
        sources = build_training_sources(args)
        loader_cache = {}

        for run_index, source in enumerate(sources, start=1):
            train_manifest_used = source["train_manifest"]
            val_manifest_used = source["val_manifest"]
            rgb_pretrained_path = source["rgb_pretrained_path"]
            signal_pretrained_paths = source["signal_pretrained_paths"]
            source_tag = source["source_tag"]

            cache_key = (train_manifest_used, val_manifest_used)
            if cache_key not in loader_cache:
                loader_cache[cache_key] = prepare_train_val_loaders_for_manifests(
                    args=args,
                    train_manifest=train_manifest_used,
                    val_manifest=val_manifest_used,
                )

            trainloader, valloader, train_dataset = loader_cache[cache_key]

            summary = run_one_training_experiment(
                args=args,
                device=device,
                trainloader=trainloader,
                valloader=valloader,
                train_dataset=train_dataset,
                run_index=run_index,
                rgb_pretrained_path=rgb_pretrained_path,
                signal_pretrained_paths=signal_pretrained_paths,
                train_manifest_used=train_manifest_used,
                val_manifest_used=val_manifest_used,
                source_tag=source_tag,
            )
            all_summaries.append(summary)

        summary_csv = os.path.join(args.save_path, "train_experiment_summary.csv")
        save_csv_rows(summary_csv, all_summaries, append=True)
        print(f"[train summary csv saved] {summary_csv}")

    elif args.run_mode == "test":
        run_test_mode(args, device)

    else:
        raise ValueError(f"Unknown run_mode: {args.run_mode}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
