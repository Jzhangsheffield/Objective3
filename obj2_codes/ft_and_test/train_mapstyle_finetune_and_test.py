#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_mapstyle_finetune_and_test.py

本版本在原始 map-style 分类训练 / 测试脚本基础上，支持以下功能：

1) 训练模式（run_mode=train）
   - 支持从头训练（scratch）
   - 支持提供多个预训练权重路径，逐个加载后分别训练
   - 预训练加载时可选择自动丢弃对比学习头 / 分类头等不需要的权重
   - 支持两种微调方式：
       a) full      : 全部参数都训练
       b) head_only : 只训练分类头，冻结 backbone
   - 支持 discriminative learning rate（区分学习率）
       a) backbone 一个学习率
       b) 分类头一个学习率
   - 每个实验（每个预训练源）都会单独保存：
       - 日志
       - checkpoint
       - datamap / training dynamics
       - 训练摘要

2) 测试模式（run_mode=test）
   - 支持提供一个或多个 test manifest
   - 支持提供多个已训练权重文件
   - 按顺序或广播方式逐个加载并在测试集上评估
   - 将测试结果汇总保存为 CSV

============================================================
本版本最重要的修改点：
============================================================
- 不再按文件名中的 imbalance_XXX 等内容自动匹配
- 改为按命令行输入顺序匹配

训练模式顺序匹配规则：
---------------------
假设你输入：
    --train_manifests A.jsonl B.jsonl C.jsonl
    --val_manifests   VA.jsonl VB.jsonl VC.jsonl
    --pretrained_weight_paths P1.pth P2.pth P3.pth

则会构造三个实验：
    A + VA + P1
    B + VB + P2
    C + VC + P3

广播规则：
---------
1) train manifest：
   - 可以提供 1 个，然后广播给所有 pretrained weights
   - 或提供与 pretrained weights 数量相同的多个，按顺序一一对应

2) val manifest：
   - 可以不提供（全部不做验证）
   - 可以提供 1 个，然后广播给所有实验
   - 或提供与实验数相同的多个，按顺序一一对应

测试模式顺序匹配规则：
---------------------
假设你输入：
    --test_manifest T1.jsonl T2.jsonl
    --test_weight_paths W1.pth W2.pth

则会测试：
    T1 + W1
    T2 + W2

广播规则：
---------
- 若只提供 1 个 test manifest，则广播给所有 test_weight_paths
"""

import os
import re
import csv
import json
import math
import time
import random
import argparse
from pathlib import Path
from collections import deque
from contextlib import nullcontext

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

import backbone.resnet as resnet3d
from backbone.renet1d_my import build_resnet1d

from utils_.log_training_dynamics import TrainingDynamicsLogger
from loss.focal_loss import FocalLoss

# ============================================================
# map-style loader 相关导入
# ============================================================
from utils_.mapstype_dataloader_with_index_mindrove_modified_varlen import (
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
        description="Train / finetune / batch-test a classifier on map-style dataset (rgb/depth/mindrove)")

# ---------------- 运行模式 ----------------
parser.add_argument(
    "--run_mode",
    type=str,
    default="train",
    choices=["train", "test"],
    help="train: 训练/微调；test: 仅批量测试多个权重",
)

# ---------------- 基本路径 ----------------
parser.add_argument(
    "--save_path",
    type=str,
    default=r"./weights",
    help="所有实验输出的根目录"
)
parser.add_argument(
    "--datamap_csv_path",
    type=str,
    default=r"./datamaps",
    help="datamap / training dynamics 根目录"
)
parser.add_argument(
    "--dataset_root",
    type=str,
    required=True,
    help="map-style 数据集根目录"
)
parser.add_argument(
    "--label_map_json",
    type=str,
    required=True,
    help="统一 label_map.json 路径"
)

# ---------------- manifest 路径 ----------------
# 为了兼容旧调用方式，仍然保留单个参数和多个参数两种接口。
# 其中：
#   - 单个参数 train_manifest / val_manifest 会被视为列表中的第一个元素
#   - 多个参数 train_manifests / val_manifests 会按输入顺序追加在后面
parser.add_argument(
    "--train_manifest",
    type=str,
    default=None,
    help="单个训练 manifest 文件路径或文件名；保留用于向后兼容"
)
parser.add_argument(
    "--train_manifests",
    nargs="*",
    default=[],
    help="多个训练 manifest；本版本按输入顺序匹配，而不是按文件名自动匹配"
)
parser.add_argument(
    "--val_manifest",
    type=str,
    default=None,
    help="单个验证 manifest 文件路径或文件名；保留用于向后兼容"
)
parser.add_argument(
    "--val_manifests",
    nargs="*",
    default=[],
    help="多个验证 manifest；本版本按输入顺序匹配，而不是按文件名自动匹配"
)

# 注意：这里已经改成支持多个 test manifest
parser.add_argument(
    "--test_manifest",
    nargs="+",
    default=[],
    help=(
        "一个或多个测试 manifest 文件路径或文件名。"
        "若只给 1 个，会广播给所有 test_weight_paths；"
        "若给多个，则按输入顺序与 test_weight_paths 一一对应。"
    )
)

# ---------------- 文件命名 -------------------
parser.add_argument(
    "--pretrained_tag_mode",
    type=str,
    default="legacy",
    choices=["legacy", "last_k_dirs", "relative_to_anchor"],
    help=(
        "如何从 pretrained_weight_paths 构造实验目录名。"
        "legacy: 旧逻辑，只用 parent.name + stem；"
        "last_k_dirs: 使用权重路径最后 k 层目录 + stem；"
        "relative_to_anchor: 取某个锚点目录之后的相对路径作为标签。"
    ),
)

parser.add_argument(
    "--pretrained_tag_last_k",
    type=int,
    default=4,
    help="当 pretrained_tag_mode=last_k_dirs 时，取权重路径末尾多少层目录参与命名"
)

parser.add_argument(
    "--pretrained_tag_anchor",
    type=str,
    default=None,
    help=(
        "当 pretrained_tag_mode=relative_to_anchor 时使用。"
        "例如设为 'J_test'，则从 J_test 之后的相对路径开始构造标签。"
    ),
)

# ---------------- 标签 / 模态 ----------------
parser.add_argument(
    "--tier_mode",
    type=str,
    default="tier1",
    choices=["tier1", "tier2", "tier3"],
    help="使用哪个 tier 的标签训练/测试"
)
parser.add_argument(
    "--n_frames",
    type=int,
    default=16,
    help="每个样本采样帧数"
)
parser.add_argument(
    "--use_modality",
    type=str,
    default="rgb",
    choices=["rgb", "depth", "mindrove"],
    help="当前分类只使用单模态输入：rgb / depth / mindrove"
)

# ---------------- DataLoader ----------------
parser.add_argument(
    "--num_workers_train",
    type=int,
    default=8,
    help="训练集 DataLoader worker 数量"
)
parser.add_argument(
    "--num_workers_val",
    type=int,
    default=6,
    help="验证集 DataLoader worker 数量"
)
parser.add_argument(
    "--num_workers_test",
    type=int,
    default=8,
    help="测试集 DataLoader worker 数量"
)
parser.add_argument(
    "--prefetch_factor_train",
    type=int,
    default=2,
    help="训练集 prefetch_factor；num_workers=0 时忽略"
)
parser.add_argument(
    "--prefetch_factor_val",
    type=int,
    default=2,
    help="验证集 prefetch_factor；num_workers=0 时忽略"
)
parser.add_argument(
    "--prefetch_factor_test",
    type=int,
    default=2,
    help="测试集 prefetch_factor；num_workers=0 时忽略"
)

# ---------------- 验证开关 ----------------
parser.add_argument(
    "--disable_val",
    action="store_true",
    help="训练时禁用验证"
)

# ---------------- 输出空间尺寸（由 loader 内部完成） ----------------
parser.add_argument(
    "--rgb_size",
    type=int,
    default=224,
    help="RGB 输出尺寸（H=W）"
)
parser.add_argument(
    "--depth_size",
    type=int,
    default=224,
    help="Depth 输出尺寸（H=W）"
)
# ---------------- RGB normalization 参数（由 loader 内部 Normalize 使用） ----------------
parser.add_argument(
    "--rgb_mean",
    nargs=3,
    type=float,
    default=[0.356, 0.363, 0.367],
    metavar=("R_MEAN", "G_MEAN", "B_MEAN"),
    help=(
        "RGB Normalize 使用的 mean，必须给 3 个 float，顺序为 R G B。"
        "例如：--rgb_mean 0.356 0.363 0.367"
    ),
)
parser.add_argument(
    "--rgb_std",
    nargs=3,
    type=float,
    default=[0.288, 0.271, 0.270],
    metavar=("R_STD", "G_STD", "B_STD"),
    help=(
        "RGB Normalize 使用的 std，必须给 3 个正数，顺序为 R G B。"
        "例如：--rgb_std 0.288 0.271 0.270"
    ),
)

# ---------------- RGB train augment 参数（由 loader 使用） ----------------
parser.add_argument(
    "--rrc_scale_min",
    type=float,
    default=0.6,
    help="RandomResizedCrop scale 最小值"
)
parser.add_argument(
    "--rrc_scale_max",
    type=float,
    default=1.0,
    help="RandomResizedCrop scale 最大值"
)
parser.add_argument(
    "--rrc_ratio_min",
    type=float,
    default=0.75,
    help="RandomResizedCrop ratio 最小值"
)
parser.add_argument(
    "--rrc_ratio_max",
    type=float,
    default=1.3333333333,
    help="RandomResizedCrop ratio 最大值"
)
parser.add_argument(
    "--rgb_apply_spatial_aug",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "训练集是否启用 RGB 随机空间增强中的 flip/jitter/gray/blur。"
        "注意：设为 False 时仍使用 TemporallyConsistentSpatialAugmentation，"
        "因此 RandomResizedCrop 仍然保留；只会把 flip/jitter/gray/blur 的概率置 0。"
        "验证/测试集不受该参数影响。"
    ),
)

parser.add_argument(
    "--rgb_hflip_p",
    type=float,
    default=0.5,
    help="训练集 RGB RandomHorizontalFlip 概率"
)
parser.add_argument(
    "--rgb_vflip_p",
    type=float,
    default=0.5,
    help="训练集 RGB RandomVerticalFlip 概率；机械操作视频通常建议为 0"
)

parser.add_argument(
    "--rgb_jitter_p",
    type=float,
    default=0.5,
    help="训练集 RGB ColorJitter 被应用的概率"
)
parser.add_argument(
    "--rgb_jitter_brightness",
    type=float,
    default=0.24,
    help="ColorJitter brightness 强度"
)
parser.add_argument(
    "--rgb_jitter_contrast",
    type=float,
    default=0.24,
    help="ColorJitter contrast 强度"
)
parser.add_argument(
    "--rgb_jitter_saturation",
    type=float,
    default=0.24,
    help="ColorJitter saturation 强度"
)
parser.add_argument(
    "--rgb_jitter_hue",
    type=float,
    default=0.16,
    help="ColorJitter hue 强度；torchvision 要求通常不超过 0.5"
)

parser.add_argument(
    "--rgb_gray_p",
    type=float,
    default=0.2,
    help="训练集 RGB RandomGrayscale 概率"
)

parser.add_argument(
    "--rgb_blur_p",
    type=float,
    default=0.5,
    help="训练集 RGB GaussianBlur 被应用的概率"
)
parser.add_argument(
    "--rgb_blur_kernel",
    type=int,
    default=7,
    help="GaussianBlur kernel size，必须是 >=3 的奇数"
)
parser.add_argument(
    "--rgb_blur_sigma_min",
    type=float,
    default=0.1,
    help="GaussianBlur sigma 下界"
)
parser.add_argument(
    "--rgb_blur_sigma_max",
    type=float,
    default=1.0,
    help="GaussianBlur sigma 上界"
)

# ---------------- MindRove data config ----------------
parser.add_argument(
    "--mindrove_target_len",
    type=int,
    default=256,
    help="MindRove 序列重采样后的统一长度"
)
parser.add_argument(
    "--mindrove_hands",
    nargs="+",
    default=["left", "right"],
    choices=["left", "right"],
    help="MindRove 使用哪些手的数据"
)
parser.add_argument(
    "--mindrove_signals",
    nargs="+",
    default=["emg", "imu"],
    choices=["emg", "imu"],
    help="MindRove 使用哪些信号"
)
parser.add_argument(
    "--mindrove_merge_hands",
    action="store_true",
    help="是否将左右手同类信号在通道维拼接后输出"
)
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
parser.add_argument(
    "--disable_train_augmentation",
    action="store_true",
    help=(
        "统一关闭训练集增强。启用后："
        "RGB 的 RandomResizedCrop 会退化为 scale=(1,1)、ratio=(1,1)，"
        "flip/jitter/gray/blur 概率全部置 0；"
        "MindRove 样本级增强也会关闭。验证/测试本来就不启用训练增强。"
    ),
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

# ---------------- 模型与训练 ----------------
parser.add_argument(
    "--model_depth",
    type=int,
    default=18,
    help="3D ResNet 深度"
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=17,
    help="分类类别数量"
)
parser.add_argument(
    "--l2_normalize_before_fc",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "是否在模型最终 fc 分类头之前对 backbone feature 做 L2 normalize。"
        "默认关闭，保持原始行为。注意：训练和测试必须使用相同设置。"
    ),
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="训练轮数"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="batch size"
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.05,
    help="默认基础学习率；单学习率模式下直接使用"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="SGD momentum"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="优化器的 weight decay；SGD 和 AdamW 都会使用该值"
)

parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["sgd", "adamw"],
    help=(
        "选择微调优化器。"
        "sgd: 使用 torch.optim.SGD，保留 momentum；"
        "adamw: 使用 torch.optim.AdamW，不使用 momentum，采用 decoupled weight decay。"
    ),
)

parser.add_argument(
    "--adamw_beta1",
    type=float,
    default=0.9,
    help="AdamW beta1；仅在 --optimizer adamw 时使用"
)

parser.add_argument(
    "--adamw_beta2",
    type=float,
    default=0.999,
    help="AdamW beta2；仅在 --optimizer adamw 时使用"
)

parser.add_argument(
    "--adamw_eps",
    type=float,
    default=1e-8,
    help="AdamW epsilon；仅在 --optimizer adamw 时使用"
)
parser.add_argument(
    "--cos",
    action="store_true",
    help="使用 cosine 学习率衰减"
)
parser.add_argument(
    "--schedules",
    default=[25, 50, 75],
    nargs="*",
    type=int,
    help="若不用 cosine，则使用 multi-step milestones"
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="随机种子"
)

# ---------------- MindRove 1D backbone ----------------
parser.add_argument("--mindrove_arch",default="resnet10_1d",choices=["resnet10_1d", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
                    help="which torchvision-style ResNet1D architecture to use")
parser.add_argument("--mindrove_base_channels",default=64,type=int,
                    help="base channel width of ResNet1D stem and stage1")
parser.add_argument("--mindrove_stem_kernel_size",default=7,type=int,
                    help="stem Conv1d kernel size of ResNet1D")
parser.add_argument("--mindrove_stem_stride", default=2, type=int,
                    help="stem Conv1d stride of ResNet1D")
parser.add_argument("--mindrove_use_stem_pool", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to use MaxPool1d after the stem")
parser.add_argument("--mindrove_zero_init_residual", action=argparse.BooleanOptionalAction, default=False,
                    help="whether to zero-initialize the last BN in each residual branch")

# ---------------- Weighted Sampler（train only） ----------------
parser.add_argument(
    "--use_weighted_sampler",
    action="store_true",
    help="是否对训练集启用 WeightedRandomSampler"
)
parser.add_argument(
    "--sampler_tier",
    type=str,
    default=None,
    choices=["tier1", "tier2", "tier3"],
    help="weighted sampler 按哪个 tier 重采样；默认跟随 tier_mode"
)
parser.add_argument(
    "--sampler_mode",
    type=str,
    default="sqrt_inv",
    choices=["inv", "sqrt_inv"],
    help="weighted sampler 权重方式"
)

# ---------------- Weighted CE ----------------
parser.add_argument(
    "--use_weighted_ce",
    action="store_true",
    help="启用 Weighted Cross-Entropy"
)
parser.add_argument(
    "--weight_method",
    type=str,
    default="class_balanced",
    choices=["class_balanced", "inv_freq"],
    help="类别权重计算方法"
)
parser.add_argument(
    "--cb_beta",
    type=float,
    default=0.999,
    help="class_balanced 权重中的 beta"
)
parser.add_argument(
    "--weight_normalize_mean",
    action="store_true",
    help="是否将类别权重归一化到均值=1"
)

# ---------------- Focal Loss ----------------
parser.add_argument(
    "--use_focal",
    action="store_true",
    help="启用 Focal Loss"
)
parser.add_argument(
    "--focal_gamma",
    type=float,
    default=2.0,
    help="Focal Loss gamma"
)
parser.add_argument(
    "--focal_use_alpha",
    action="store_true",
    help="Focal Loss 是否使用 alpha 类权重"
)

# ---------------- AMP ----------------
parser.add_argument(
    "--enable_amp",
    action="store_true",
    help="是否启用 AMP 混合精度训练"
)

# ---------------- 预训练 / 微调相关 ----------------
parser.add_argument(
    "--pretrained_weight_paths",
    nargs="*",
    default=[],
    help=(
        "训练模式下可传入多个预训练权重路径。"
        "本版本按输入顺序与 train_manifest(s) / val_manifest(s) 对齐。"
    ),
)
parser.add_argument(
    "--include_scratch_baseline",
    action="store_true",
    help="当提供多个预训练权重时，是否额外再跑 scratch baseline"
)
parser.add_argument(
    "--finetune_mode",
    type=str,
    default="full",
    choices=["full", "head_only"],
    help="full: 全部微调；head_only: 只训练分类头"
)

# 默认丢掉对比学习头 / projector / predictor 等
parser.add_argument(
    "--keep_pretrained_head",
    action="store_true",
    help="默认会丢掉预训练中的 fc/head/projector/predictor 等头部参数；传该开关可保留",
)
parser.add_argument(
    "--pretrained_strict",
    action="store_true",
    help="预训练加载时是否 strict=True；默认 strict=False，更适合微调",
)

# ---------------- 区分学习率（discriminative LR） ----------------
parser.add_argument(
    "--use_discriminative_lr",
    action="store_true",
    help="是否为 backbone 和分类头使用不同学习率（仅对 full finetune 有意义）",
)
parser.add_argument(
    "--backbone_learning_rate",
    type=float,
    default=None,
    help="backbone 学习率；为空则回退到 learning_rate",
)
parser.add_argument(
    "--head_learning_rate",
    type=float,
    default=None,
    help="分类头学习率；为空则回退到 learning_rate",
)

# ---------------- checkpoint 保存策略 ----------------
parser.add_argument(
    "--save_period",
    type=int,
    default=20,
    help="每隔多少个 epoch 保存一个周期性 checkpoint"
)
parser.add_argument(
    "--best_after_epoch",
    type=int,
    default=0,
    help="只在 epoch >= 该值之后保存 best checkpoint"
)

# ---------------- 批量测试 ----------------
parser.add_argument(
    "--test_weight_paths",
    nargs="*",
    default=[],
    help=(
        "测试模式下可传入多个已训练权重路径。"
        "本版本按输入顺序与 test_manifest 对齐。"
    ),
)
parser.add_argument(
    "--test_results_csv",
    type=str,
    default=None,
    help="测试结果 CSV 保存路径；为空则默认存到 save_path/test_results.csv",
)

args = parser.parse_args()


# ============================================================
# 全局 AMP / device 配置
# ============================================================
os.makedirs(args.save_path, exist_ok=True)
has_cuda = torch.cuda.is_available()
use_bf16 = torch.cuda.is_bf16_supported() if has_cuda else False
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
torch.backends.cudnn.benchmark = True


# ============================================================
# 1) 工具函数
# ============================================================
def seed_everything(s: int = 42):
    """
    尽量提高可复现性。

    注意：
    - 这会让 cudnn.deterministic=True
    - 可能会比 benchmark 模式更慢
    - 若 DataLoader 内部还有复杂随机增强，严格逐位复现仍可能受 worker 随机性影响
    """
    np.random.seed(s)
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_name(name: str) -> str:
    """
    将路径 stem / 任意字符串转换成更适合作为目录名或文件名片段的形式。
    """
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name))
    name = name.strip("._-")
    return name or "run"



def build_pretrained_src_tag(pretrained_path: str | None, args) -> str:
    """
    根据开关，从预训练权重路径构造更稳定、更不易冲突的 src tag。

    mode:
    - legacy:
        parent.name + stem
        例如: proto_1_checkpoint_0200

    - last_k_dirs:
        取路径最后 k 层目录 + stem
        例如:
        signal_emg/ablation_contrastive_rel/prem_0.5/proto_1/checkpoint_0200.pth
        -> signal_emg_ablation_contrastive_rel_prem_0.5_proto_1_checkpoint_0200

    - relative_to_anchor:
        从某个锚点目录之后开始取相对路径 + stem
        例如 anchor='J_test' 时:
        J_test/signal_emg/ablation_contrastive_rel/prem_0.5/proto_1/checkpoint_0200.pth
        -> signal_emg_ablation_contrastive_rel_prem_0.5_proto_1_checkpoint_0200
    """
    if pretrained_path is None:
        return "scratch"

    p = Path(pretrained_path)
    mode = args.pretrained_tag_mode

    if mode == "legacy":
        parent_tag = sanitize_name(p.parent.name)
        stem_tag = sanitize_name(p.stem)
        return f"{parent_tag}_{stem_tag}"

    if mode == "last_k_dirs":
        k = max(1, int(args.pretrained_tag_last_k))
        parent_parts = [sanitize_name(x) for x in p.parent.parts[-k:]]
        stem_tag = sanitize_name(p.stem)
        return "_".join(parent_parts + [stem_tag])

    if mode == "relative_to_anchor":
        anchor = args.pretrained_tag_anchor
        if not anchor:
            raise ValueError(
                "pretrained_tag_mode='relative_to_anchor' 时，必须提供 --pretrained_tag_anchor"
            )

        parts = list(p.parts)
        try:
            anchor_idx = parts.index(anchor)
        except ValueError:
            raise ValueError(
                f"anchor '{anchor}' 不在 pretrained path 中：{pretrained_path}"
            )

        rel_parts = [sanitize_name(x) for x in parts[anchor_idx + 1:-1]]
        if len(rel_parts) == 0:
            return sanitize_name(p.parent.name)
        return "_".join(rel_parts)

    raise ValueError(f"Unknown pretrained_tag_mode: {mode}")



def compact_manifest_stem(path_or_name: str | None) -> str:
    """
    将 manifest 文件名压缩成更短的实验标签。

    例如：
        train_manifest_M_MR_J.jsonl -> M_MR_J
        val_manifest_N_left.jsonl   -> N_left
        train_manifest.jsonl        -> data
    """
    if path_or_name is None:
        return "data"

    stem = sanitize_name(Path(str(path_or_name)).stem)

    removable_prefixes = [
        "train_manifest_",
        "train_manifest",
        "val_manifest_",
        "val_manifest",
        "test_manifest_",
        "test_manifest",
    ]

    for prefix in removable_prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break

    stem = stem.strip("._-")
    return stem or "data"


def resolve_manifest_arg(dataset_root: str, manifest_arg: str | None) -> str | None:
    """
    兼容两种传法：
    1) 直接传文件名，例如 train_manifest.jsonl
    2) 传绝对路径或相对路径

    对于绝对路径，如果它位于 dataset_root 内部，则转换成相对路径，
    这样更兼容很多 map-style dataset builder 的实现方式。
    """
    if manifest_arg is None:
        return None

    manifest_path = Path(manifest_arg)
    dataset_root_path = Path(dataset_root).resolve()

    if manifest_path.is_absolute():
        try:
            rel = manifest_path.resolve().relative_to(dataset_root_path)
            return str(rel)
        except Exception:
            # 若绝对路径不在 dataset_root 下，则直接原样返回
            return str(manifest_path)

    return str(manifest_path)


def combine_manifest_args(single_manifest: str | None, multi_manifests: list[str] | None) -> list[str]:
    """
    将单个 manifest 参数与多个 manifest 参数合并成一个去重后的有序列表。

    设计目的：
    - 保留 --train_manifest / --val_manifest 的旧接口
    - 同时支持新增的 --train_manifests / --val_manifests
    - 保证顺序稳定：single_manifest 会排在最前面，multi_manifests 依次追加
    """
    merged = []
    seen = set()

    def _add_one(x):
        if x is None:
            return
        x = str(x).strip()
        if not x:
            return
        if x not in seen:
            merged.append(x)
            seen.add(x)

    _add_one(single_manifest)

    if multi_manifests is not None:
        for item in multi_manifests:
            _add_one(item)

    return merged


def normalize_string_list(values) -> list[str]:
    """
    将 argparse 传入的列表参数清理成有序字符串列表。
    - 去掉空字符串
    - 去掉纯空白
    """
    out = []
    for x in values:
        if x is None:
            continue
        sx = str(x).strip()
        if not sx:
            continue
        out.append(sx)
    return out


def build_reverse_label_map(label_map_json_path: str, tier_mode: str) -> dict[int, str]:
    """
    从 label_map.json 构建:
        class_id -> class_name
    的反向映射。

    例如原始 label_map[tier_mode] 可能是:
        {
            "adjust": 0,
            "take": 1,
            ...
        }

    这里转换成:
        {
            0: "adjust",
            1: "take",
            ...
        }
    """
    label_map = load_label_map_json(label_map_json_path)

    if tier_mode not in label_map:
        raise KeyError(f"tier_mode '{tier_mode}' not found in label_map.json")

    forward_map = label_map[tier_mode]
    reverse_map = {int(v): str(k) for k, v in forward_map.items()}
    return reverse_map


def _safe_float(x):
    """
    将 numpy / torch 标量转换成 Python float，便于 json 保存。
    """
    if x is None:
        return None
    return float(x)


def round_metric_dict(d: dict, ndigits: int = 4) -> dict:
    """
    将 per-class metric dict 中的 float 保留固定小数位。
    None 保持为 None。
    """
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        else:
            out[k] = round(float(v), ndigits)
    return out


def compute_classification_metrics(
    y_true,
    y_pred,
    num_classes: int,
    reverse_label_map: dict[int, str] | None = None,
) -> dict:
    """
    计算多分类指标。

    返回内容包括：
    1) overall accuracy
    2) balanced accuracy
       - 等于所有 present classes 的 per-class recall 平均
       - present classes 指在当前 split 中真实样本数 > 0 的类别
    3) macro-F1
       - 先计算每个 present class 的 F1，再取平均
    4) per-class accuracy
       - 对每个类别 c:
         per_class_acc[c] = TP_c / (TP_c + FN_c)
       - 也就是该类别的 recall
    5) per-class support
       - 每个类别在当前 split 中的真实样本数

    注意：
    - 如果某个类别在当前 split 中 support=0，则它不参与 balanced_acc 和 macro_f1 平均。
    - 这比把缺失类别强行记为 0 更合理，因为验证集或测试集可能不包含所有类别。
    """

    y_true_t = torch.as_tensor(y_true, dtype=torch.long).view(-1)
    y_pred_t = torch.as_tensor(y_pred, dtype=torch.long).view(-1)

    if y_true_t.numel() != y_pred_t.numel():
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {y_true_t.numel()} and {y_pred_t.numel()}"
        )

    if y_true_t.numel() == 0:
        raise ValueError("Cannot compute metrics on empty y_true / y_pred.")

    if torch.any(y_true_t < 0) or torch.any(y_true_t >= num_classes):
        bad = y_true_t[(y_true_t < 0) | (y_true_t >= num_classes)]
        raise ValueError(
            f"Found labels outside [0, {num_classes - 1}]. "
            f"Bad labels preview: {bad[:10].tolist()}"
        )

    if torch.any(y_pred_t < 0) or torch.any(y_pred_t >= num_classes):
        bad = y_pred_t[(y_pred_t < 0) | (y_pred_t >= num_classes)]
        raise ValueError(
            f"Found predictions outside [0, {num_classes - 1}]. "
            f"Bad predictions preview: {bad[:10].tolist()}"
        )

    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for t, p in zip(y_true_t, y_pred_t):
        cm[int(t.item()), int(p.item())] += 1

    cm_f = cm.to(torch.float32)

    tp = torch.diag(cm_f)
    support = cm_f.sum(dim=1)      # TP + FN
    pred_count = cm_f.sum(dim=0)   # TP + FP

    present = support > 0

    recall = torch.zeros(num_classes, dtype=torch.float32)
    precision = torch.zeros(num_classes, dtype=torch.float32)
    f1 = torch.zeros(num_classes, dtype=torch.float32)

    recall[present] = tp[present] / support[present]

    pred_nonzero = pred_count > 0
    precision[pred_nonzero] = tp[pred_nonzero] / pred_count[pred_nonzero]

    denom = precision + recall
    valid_f1 = denom > 0
    f1[valid_f1] = 2.0 * precision[valid_f1] * recall[valid_f1] / denom[valid_f1]

    total = int(y_true_t.numel())
    correct = int((y_true_t == y_pred_t).sum().item())

    overall_acc = correct / max(1, total)

    if int(present.sum().item()) > 0:
        balanced_acc = float(recall[present].mean().item())
        macro_f1 = float(f1[present].mean().item())
    else:
        balanced_acc = 0.0
        macro_f1 = 0.0

    per_class_acc = {}
    per_class_precision = {}
    per_class_f1 = {}
    per_class_support = {}

    for class_id in range(num_classes):
        class_name = (
            reverse_label_map.get(class_id, str(class_id))
            if reverse_label_map is not None
            else str(class_id)
        )

        sup = int(support[class_id].item())
        per_class_support[class_name] = sup

        if sup > 0:
            per_class_acc[class_name] = float(recall[class_id].item())
            per_class_f1[class_name] = float(f1[class_id].item())
        else:
            per_class_acc[class_name] = None
            per_class_f1[class_name] = None

        if int(pred_count[class_id].item()) > 0:
            per_class_precision[class_name] = float(precision[class_id].item())
        else:
            per_class_precision[class_name] = None

    return {
        "acc": float(overall_acc),
        "balanced_acc": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "num_samples": total,
        "num_correct": correct,
        "num_present_classes": int(present.sum().item()),
        "per_class_acc": per_class_acc,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "confusion_matrix": cm.tolist(),
    }


def format_metrics_for_log(metrics: dict, prefix: str) -> str:
    """
    将 acc / balanced_acc / macro_f1 格式化成一段日志文本。
    """
    return (
        f"{prefix}_acc: {metrics['acc']:.4f}, "
        f"{prefix}_balanced_acc: {metrics['balanced_acc']:.4f}, "
        f"{prefix}_macro_f1: {metrics['macro_f1']:.4f}"
    )


def align_required_sequence(values: list[str], target_len: int, field_name: str) -> list[str]:
    """
    将“必须存在”的字符串列表对齐到 target_len。

    允许两种长度：
    1) len(values) == 1
       -> 广播给所有实验
    2) len(values) == target_len
       -> 按顺序一一对应

    其他长度直接报错。

    例如：
        values = ["train_a.jsonl"], target_len = 3
        -> ["train_a.jsonl", "train_a.jsonl", "train_a.jsonl"]

        values = ["a", "b", "c"], target_len = 3
        -> ["a", "b", "c"]
    """
    if len(values) == 0:
        raise ValueError(f"{field_name} 不能为空。")

    if len(values) == 1:
        return values * target_len

    if len(values) == target_len:
        return list(values)

    raise ValueError(
        f"{field_name} 的数量不合法：len={len(values)}，目标实验数={target_len}。\n"
        f"允许的情况只有：1（广播）或 {target_len}（逐项顺序匹配）。"
    )


def align_optional_sequence(values: list[str], target_len: int, field_name: str) -> list[str | None]:
    """
    将“可为空”的字符串列表对齐到 target_len。

    允许三种长度：
    1) len(values) == 0
       -> 全部置为 None
    2) len(values) == 1
       -> 广播给所有实验
    3) len(values) == target_len
       -> 按顺序一一对应
    """
    if len(values) == 0:
        return [None] * target_len

    if len(values) == 1:
        return values * target_len

    if len(values) == target_len:
        return list(values)

    raise ValueError(
        f"{field_name} 的数量不合法：len={len(values)}，目标实验数={target_len}。\n"
        f"允许的情况只有：0（全部 None）、1（广播）或 {target_len}（逐项顺序匹配）。"
    )


def build_train_manifest_list(args) -> list[str]:
    """
    统一生成训练 manifest 的有序列表。
    """
    return combine_manifest_args(args.train_manifest, args.train_manifests)


def build_val_manifest_list(args) -> list[str]:
    """
    统一生成验证 manifest 的有序列表。
    """
    return combine_manifest_args(args.val_manifest, args.val_manifests)


def build_test_manifest_list(args) -> list[str]:
    """
    统一生成测试 manifest 的有序列表。

    注意：
    - 本版本中 --test_manifest 已经可以接收多个值
    - 因此这里不再区分 single / multi 两套接口
    """
    return normalize_string_list(args.test_manifest)


def validate_args(args):
    """
    根据 run_mode 做条件检查。

    本版本的重点：
    - 不再检查 imbalance_XXX 等文件名标签
    - 改为检查“顺序对齐 / 广播”是否合法
    """
        # ---------------- RGB spatial augmentation 参数检查 ----------------
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
        
    if args.run_mode == "train":
        train_manifest_list = build_train_manifest_list(args)
        val_manifest_list = build_val_manifest_list(args)
        pretrained_list = normalize_string_list(args.pretrained_weight_paths)

        if len(train_manifest_list) == 0:
            raise ValueError("run_mode=train 时，必须提供 --train_manifest 或 --train_manifests")

        # 若没有预训练权重，则实验数由 train manifests 决定
        if len(pretrained_list) == 0:
            num_main_runs = len(train_manifest_list)

            # 训练集在这种情况下不需要广播，直接一条 train manifest 对应一个实验
            # 这里主要检查 val 是否能与 num_main_runs 对齐
            _ = align_optional_sequence(val_manifest_list, num_main_runs, "val manifest(s)")

        # 若有预训练权重，则 train / val 都要能对齐到 pretrained 数量
        else:
            num_main_runs = len(pretrained_list)
            _ = align_required_sequence(train_manifest_list, num_main_runs, "train manifest(s)")
            _ = align_optional_sequence(val_manifest_list, num_main_runs, "val manifest(s)")

        if args.use_discriminative_lr and args.finetune_mode == "head_only":
            print("[warning] finetune_mode=head_only 时，use_discriminative_lr 没有实际意义，将只使用分类头学习率。")

    elif args.run_mode == "test":
        test_manifest_list = build_test_manifest_list(args)
        test_weight_list = normalize_string_list(args.test_weight_paths)

        if len(test_manifest_list) == 0:
            raise ValueError("run_mode=test 时，必须提供至少一个 --test_manifest")
        if len(test_weight_list) == 0:
            raise ValueError("run_mode=test 时，必须至少提供一个 --test_weight_paths")

        # 测试 manifest 支持：
        # - 1 个：广播
        # - 与 test weights 数量相同：逐项匹配
        _ = align_required_sequence(test_manifest_list, len(test_weight_list), "test manifest(s)")


# ============================================================
# 2) 学习率调度（支持多 param group）
# ============================================================
def compute_lr_factor(epoch: int, args) -> float:
    """
    计算相对于“初始学习率”的缩放比例。

    这样设计的原因：
    - 单学习率时：当前 lr = learning_rate * factor
    - 双学习率时：
        backbone lr = backbone_initial_lr * factor
        head lr     = head_initial_lr * factor

    这样就不会在第一个 epoch 把双学习率覆盖成同一个值。
    """
    if args.cos:
        return 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))

    factor = 1.0
    for milestone in args.schedules:
        if epoch >= milestone:
            factor *= 0.1
    return factor


def adjust_learning_rate(optimizer, epoch, args) -> dict:
    """
    对所有 param group 按各自 initial_lr 成比例衰减。

    返回：
        一个字典，方便写日志，比如：
        {
            "backbone": 0.001,
            "head": 0.01,
        }
    """
    factor = compute_lr_factor(epoch, args)
    current_lrs = {}

    for idx, param_group in enumerate(optimizer.param_groups):
        initial_lr = float(param_group.get("initial_lr", args.learning_rate))
        current_lr = initial_lr * factor
        param_group["lr"] = current_lr
        group_name = param_group.get("group_name", f"group_{idx}")
        current_lrs[group_name] = current_lr

    return current_lrs


def format_lr_dict(lr_dict: dict) -> str:
    """
    将多个 param group 的学习率格式化成日志字符串。
    """
    parts = [f"{k}: {v:.6f}" for k, v in lr_dict.items()]
    return ", ".join(parts)


# ============================================================
# 3) 类别权重相关
# ============================================================
def build_class_weights_from_counts(
    counts,
    num_classes: int,
    method: str = "class_balanced",
    beta: float = 0.999,
    normalize_mean: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    根据给定类别计数表构造 loss 的类别权重。

    counts:
        可以是 list[int] 或 dict[int, int]
    method:
        - inv_freq
        - class_balanced
    """
    if isinstance(counts, dict):
        cnt = [0] * num_classes
        for k, v in counts.items():
            kk = int(k)
            if kk < 0 or kk >= num_classes:
                raise ValueError(f"class id {kk} out of range [0, {num_classes - 1}]")
            cnt[kk] = int(v)
        counts = cnt

    if len(counts) != num_classes:
        raise ValueError(f"counts length {len(counts)} != num_classes {num_classes}")

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


def build_class_counts_from_mapstyle_dataset(dataset, tier_mode: str, num_classes: int):
    """
    根据 map-style 训练数据集对象自动统计类别样本数。

    这里直接读取 dataset.records 中 manifest 标签字段，
    不走 __getitem__，因此不会真的加载视频帧文件。
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


# ============================================================
# 4) map-style dataset / loader 构建
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



def build_mapstyle_cfg(args, is_train: bool):
    """
    构建 map-style dataset config。

    这是分类脚本，不做 two-view 对比学习：
    - rgb_two_views = False
    - mindrove_two_views = False
    """
    rgb_hw = (args.rgb_size, args.rgb_size)
    depth_hw = (args.depth_size, args.depth_size)
    use_modalities = (args.use_modality,)

    # ------------------------------------------------------------
    # 训练增强总开关
    # ------------------------------------------------------------
    # 原脚本中 RGB 的 --no-rgb_apply_spatial_aug 只会关闭
    # flip / jitter / gray / blur，但 RandomResizedCrop 仍然存在。
    # 为了真正“关闭训练增强”，这里新增 --disable_train_augmentation：
    #   1) 只在 is_train=True 时生效；
    #   2) RGB: RRC 退化为不裁剪，所有随机概率置 0；
    #   3) MindRove: 样本级增强整体关闭。
    # 验证/测试集仍然依赖 dataloader 的 is_train=False 路径，不受影响。
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
        use_modalities=use_modalities,
        missing_policy="skip",
        load_labels=True,
        label_map_path=args.label_map_json,
        tier_mode=args.tier_mode,
        is_train=is_train,

        # -------- rgb / depth --------
        rgb_out_hw=rgb_hw,

        # RGB normalization
        rgb_mean=tuple(float(x) for x in args.rgb_mean),
        rgb_std=tuple(float(x) for x in args.rgb_std),

        # RandomResizedCrop
        rrc_scale=rrc_scale,
        rrc_ratio=rrc_ratio,

        # RGB spatial augmentation control
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
        mindrove_hands=tuple(args.mindrove_hands),
        mindrove_signals=tuple(args.mindrove_signals),
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
    构建单个 dataset + loader。

    说明：
    - train / val / test 都走这个统一入口
    - train 和 val/test 的区别主要由 is_train 控制
    - 验证/测试默认不使用 weighted sampler，不打乱，不 drop_last
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


def prepare_train_val_loaders_for_manifests(args, train_manifest: str, val_manifest: str | None):
    """
    根据显式给定的 train_manifest / val_manifest 构建 loaders。

    由于本版本的训练源现在是“按顺序分配”出来的，所以这里保持简单：
    给定一个明确的 train_manifest 和可选 val_manifest，直接构建对应 DataLoader。
    """
    train_sampler = None
    train_shuffle = True

    train_dataset, _dummy_loader = build_one_mapstyle_dataset_and_loader(
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


def prepare_train_val_loaders(args):
    """
    向后兼容接口：仍然允许旧代码只依赖 args.train_manifest / args.val_manifest。
    """
    return prepare_train_val_loaders_for_manifests(
        args=args,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
    )


def prepare_test_loader_for_manifest(args, test_manifest: str):
    """
    构建某一个 test manifest 对应的测试集 loader。

    本版本测试模式支持多个 test manifest，因此不能再只依赖 args.test_manifest 全局唯一值。
    """
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
# 5) 模型构建 / 预训练权重加载 / 冻结策略
# ============================================================
def prepare_model(args):
    """
    根据 use_modality 构建分类模型：
    - rgb / depth  -> 3D ResNet
    - mindrove     -> ResNet1D
    """
    if args.use_modality in ("rgb", "depth"):
        model = resnet3d.generate_model(
            args.model_depth,
            num_classes=args.num_classes,
            l2_normalize_before_fc=bool(args.l2_normalize_before_fc),
        )
        return model

    if args.use_modality == "mindrove":
        in_channels = compute_mindrove_in_channels(args)

        model = build_resnet1d(
            arch=args.mindrove_arch,
            in_channels=in_channels,
            num_classes=args.num_classes,
            base_channels=args.mindrove_base_channels,
            stem_kernel_size=args.mindrove_stem_kernel_size,
            stem_stride=args.mindrove_stem_stride,
            use_stem_pool=args.mindrove_use_stem_pool,
            zero_init_residual=args.mindrove_zero_init_residual,
            l2_normalize_before_fc=bool(args.l2_normalize_before_fc),
        )
        return model

    raise ValueError(f"Unsupported modality: {args.use_modality}")


def strip_prefixes_from_key(key: str) -> str:
    """
    去掉常见封装前缀，以提高预训练兼容性。

    典型前缀包括：
    - module.
    - model.
    - backbone.
    - encoder.
    - encoder_q.
    - base_encoder.
    - online_encoder.
    等等

    这里使用“循环剥离”的方式，避免出现多层包裹时只去掉一层前缀的问题。
    """
    prefixes = [
        "module.",
        "model.",
        "backbone.",
        "encoder_q.",
        "encoder_k.",
        "encoder.",
        "base_encoder.",
        "online_encoder.",
        "network.",
        "student.",
        "teacher.",
    ]

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def extract_state_dict_from_checkpoint(ckpt_obj):
    """
    从不同格式的 checkpoint 中取出真正的 state_dict。

    常见情况：
    - 直接就是 state_dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"model": ...}
    - 其他自定义保存形式
    """
    if not isinstance(ckpt_obj, dict):
        raise TypeError("Checkpoint object must be a dict-like object.")

    preferred_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "network",
    ]

    for k in preferred_keys:
        if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
            return ckpt_obj[k]

    # 如果本身已经像 state_dict：key -> Tensor
    tensor_like = 0
    for k, v in ckpt_obj.items():
        if isinstance(k, str) and torch.is_tensor(v):
            tensor_like += 1
    if tensor_like > 0:
        return ckpt_obj

    raise ValueError("Unable to find a valid state_dict inside checkpoint.")


def should_drop_pretrained_key(key: str) -> bool:
    """
    判断某个 key 是否属于应该在“微调加载预训练”时丢掉的头部参数。

    对比学习预训练通常会带有：
    - projector
    - predictor
    - mlp head
    - 旧的 fc / classifier

    这些层通常与当前下游分类任务不兼容，因此默认建议丢弃。
    """
    first_token = key.split(".")[0]
    drop_roots = {
        "fc",
        "classifier",
        "head",
        "heads",
        "mlp",
        "projector",
        "predictor",
        "projection_head",
        "contrastive_head",
        "pre_logits",
    }
    return first_token in drop_roots


def normalize_and_filter_state_dict(
    raw_state_dict: dict,
    model_state_dict: dict,
    drop_pretrained_head: bool,
):
    """
    对加载到的 state_dict 做三类清洗：

    1) 统一 key 前缀
    2) 可选丢弃预训练头 / 对比头
    3) 只保留“当前模型中存在且 shape 一致”的参数

    返回：
        filtered_state_dict, report
    """
    cleaned = {}
    dropped_head_keys = []
    dropped_missing_keys = []
    dropped_shape_keys = []

    for k, v in raw_state_dict.items():
        new_k = strip_prefixes_from_key(k)

        if drop_pretrained_head and should_drop_pretrained_key(new_k):
            dropped_head_keys.append(new_k)
            continue

        if new_k not in model_state_dict:
            dropped_missing_keys.append(new_k)
            continue

        if tuple(v.shape) != tuple(model_state_dict[new_k].shape):
            dropped_shape_keys.append((new_k, tuple(v.shape), tuple(model_state_dict[new_k].shape)))
            continue

        cleaned[new_k] = v

    report = {
        "num_loaded": len(cleaned),
        "num_dropped_head": len(dropped_head_keys),
        "num_dropped_missing": len(dropped_missing_keys),
        "num_dropped_shape": len(dropped_shape_keys),
        "dropped_head_keys_preview": dropped_head_keys[:20],
        "dropped_missing_keys_preview": dropped_missing_keys[:20],
        "dropped_shape_keys_preview": dropped_shape_keys[:20],
    }
    return cleaned, report


def load_pretrained_weights(
    model,
    ckpt_path: str,
    drop_pretrained_head: bool = True,
    strict: bool = False,
    map_location: str = "cpu",
):
    """
    用于“训练前加载预训练 backbone 权重”。

    与测试加载不同点：
    - 这里默认 drop_pretrained_head=True
    - 更符合对比学习预训练 -> 下游分类微调的场景
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location)
    raw_state_dict = extract_state_dict_from_checkpoint(ckpt)
    model_state_dict = model.state_dict()

    filtered_state_dict, report = normalize_and_filter_state_dict(
        raw_state_dict=raw_state_dict,
        model_state_dict=model_state_dict,
        drop_pretrained_head=drop_pretrained_head,
    )

    load_msg = model.load_state_dict(filtered_state_dict, strict=strict)

    final_report = {
        "checkpoint_path": ckpt_path,
        "strict": strict,
        "drop_pretrained_head": drop_pretrained_head,
        "num_model_keys": len(model_state_dict),
        **report,
        "missing_keys_after_load": list(load_msg.missing_keys),
        "unexpected_keys_after_load": list(load_msg.unexpected_keys),
    }
    return final_report


def load_model_weights_for_eval(model, ckpt_path: str, map_location: str = "cpu"):
    """
    用于“测试 / 评估时加载已训练好的分类模型权重”。

    这里不主动丢弃头部，因为测试时需要完整的分类模型。
    但仍然会：
    - 去常见前缀
    - 只加载存在且 shape 一致的参数
    """
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

    final_report = {
        "checkpoint_path": ckpt_path,
        "num_model_keys": len(model_state_dict),
        **report,
        "missing_keys_after_load": list(load_msg.missing_keys),
        "unexpected_keys_after_load": list(load_msg.unexpected_keys),
    }
    return final_report


def configure_finetune_mode(model, finetune_mode: str):
    """
    配置参数冻结策略。

    full:
        全部参数都可训练

    head_only:
        只训练 model.fc，其他全部冻结
    """
    if finetune_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    if finetune_mode == "head_only":
        if not hasattr(model, "fc"):
            raise AttributeError(
                "Current model does not have attribute 'fc'. "
                "Please adapt configure_finetune_mode() to your classifier head name."
            )
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"Unknown finetune_mode: {finetune_mode}")


def build_optimizer(model, args):
    """
    根据微调模式和双学习率设置，构建优化器。

    支持三种典型情况：
    1) full + 单学习率
    2) head_only
    3) full + discriminative lr（backbone / head 两组 lr）

    新增：
    - args.optimizer='sgd'   : 使用 SGD(momentum, weight_decay)
    - args.optimizer='adamw' : 使用 AdamW(betas, eps, decoupled weight_decay)，不使用 momentum

    注意：优化器选择和参数分组是解耦的。也就是说，
    head_only / full / discriminative lr 先决定训练哪些参数以及每组 lr，
    然后再由 args.optimizer 决定用 SGD 还是 AdamW 更新这些参数。

    返回：
        optimizer, optimizer_meta
    """
    configure_finetune_mode(model, args.finetune_mode)

    if args.finetune_mode == "head_only":
        head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
        param_groups = [
            {
                "params": [p for p in model.fc.parameters() if p.requires_grad],
                "lr": head_lr,
                "initial_lr": head_lr,
                "group_name": "head",
            }
        ]
        optimizer_meta = {
            "mode": "head_only",
            "backbone_lr": None,
            "head_lr": head_lr,
            "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    else:
        # full finetune
        if args.use_discriminative_lr:
            if not hasattr(model, "fc"):
                raise AttributeError(
                    "Current model does not have attribute 'fc'. "
                    "Please adapt build_optimizer() to your classifier head name."
                )

            backbone_lr = args.backbone_learning_rate if args.backbone_learning_rate is not None else args.learning_rate
            head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate

            backbone_params = []
            head_params = []

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("fc."):
                    head_params.append(p)
                else:
                    backbone_params.append(p)

            if len(backbone_params) == 0:
                raise RuntimeError("No trainable backbone parameters found.")
            if len(head_params) == 0:
                raise RuntimeError("No trainable head parameters found.")

            param_groups = [
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "initial_lr": backbone_lr,
                    "group_name": "backbone",
                },
                {
                    "params": head_params,
                    "lr": head_lr,
                    "initial_lr": head_lr,
                    "group_name": "head",
                },
            ]

            optimizer_meta = {
                "mode": "full_dual_lr",
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
        else:
            single_lr = args.learning_rate
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise RuntimeError("No trainable parameters found when building optimizer.")

            param_groups = [
                {
                    "params": trainable_params,
                    "lr": single_lr,
                    "initial_lr": single_lr,
                    "group_name": "all",
                }
            ]

            optimizer_meta = {
                "mode": "full_single_lr",
                "backbone_lr": single_lr,
                "head_lr": None,
                "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }

    # 记录优化器名称，方便后续 summary.csv / config.json 中区分实验。
    optimizer_name = str(args.optimizer).lower().strip()
    optimizer_meta["optimizer"] = optimizer_name
    optimizer_meta["weight_decay"] = float(args.weight_decay)

    if optimizer_name == "sgd":
        optimizer_meta["momentum"] = float(args.momentum)
        optimizer_meta["adamw_beta1"] = None
        optimizer_meta["adamw_beta2"] = None
        optimizer_meta["adamw_eps"] = None

        optimizer = optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    elif optimizer_name == "adamw":
        optimizer_meta["momentum"] = None
        optimizer_meta["adamw_beta1"] = float(args.adamw_beta1)
        optimizer_meta["adamw_beta2"] = float(args.adamw_beta2)
        optimizer_meta["adamw_eps"] = float(args.adamw_eps)

        optimizer = optim.AdamW(
            param_groups,
            betas=(args.adamw_beta1, args.adamw_beta2),
            eps=args.adamw_eps,
            weight_decay=args.weight_decay,
        )

    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    return optimizer, optimizer_meta


# ============================================================
# 6) 从 batch(dict) 中抽取 inputs / labels / ids
# ============================================================
MINDROVE_SIGNAL_CHANNELS = {
    "emg": 8,
    "imu": 6,
}


def build_mindrove_input_keys(args) -> list[str]:
    """
    根据命令行配置，确定 MindRove 输入在 batch["mindrove"] 中应该按什么顺序取出并拼接。
    这个顺序一旦定下，训练 / 测试 / 微调必须保持一致。
    """
    if args.mindrove_merge_hands:
        # merge 后的 key 只有 "emg" / "imu"
        return [sig for sig in args.mindrove_signals]

    keys = []
    for hand in args.mindrove_hands:
        for sig in args.mindrove_signals:
            keys.append(f"{hand}_{sig}")
    return keys


def compute_mindrove_in_channels(args) -> int:
    """
    自动计算 MindRove 输入通道数。
    """
    total = 0

    if args.mindrove_merge_hands:
        num_hands = 2
        for sig in args.mindrove_signals:
            total += MINDROVE_SIGNAL_CHANNELS[sig] * num_hands
        return total

    for _hand in args.mindrove_hands:
        for sig in args.mindrove_signals:
            total += MINDROVE_SIGNAL_CHANNELS[sig]
    return total


def concat_mindrove_batch_dict(batch_mindrove: dict, args) -> torch.Tensor:
    """
    将 batch["mindrove"] 的 dict[str, Tensor[B,C,L]] 按固定顺序在通道维拼接成 [B,C,L]。
    """
    if not isinstance(batch_mindrove, dict):
        raise TypeError(
            f"Expect batch['mindrove'] to be dict in classification mode, got {type(batch_mindrove)}"
        )

    keys = build_mindrove_input_keys(args)
    missing = [k for k in keys if k not in batch_mindrove]
    if missing:
        raise KeyError(
            f"Missing MindRove keys in batch['mindrove']: {missing}. "
            f"Available keys: {sorted(batch_mindrove.keys())}"
        )

    tensors = []
    ref_shape = None

    for k in keys:
        x = batch_mindrove[k]
        if not torch.is_tensor(x):
            raise TypeError(f"MindRove batch entry '{k}' must be Tensor, got {type(x)}")
        if x.ndim != 3:
            raise ValueError(f"MindRove batch entry '{k}' must be [B,C,L], got {tuple(x.shape)}")

        if ref_shape is None:
            ref_shape = (x.shape[0], x.shape[2])  # B, L
        else:
            if (x.shape[0], x.shape[2]) != ref_shape:
                raise ValueError(
                    f"Inconsistent MindRove batch shape for '{k}': got {tuple(x.shape)}, "
                    f"expected B,L = {ref_shape}"
                )

        tensors.append(x)

    return torch.cat(tensors, dim=1).contiguous()  # [B,C,L]


def _extract_inputs_and_labels(batch: dict, tier_mode: str, use_modality: str, args):
    """
    返回：
      inputs:
        - rgb/depth   -> Tensor[B,T,C,H,W]
        - mindrove    -> Tensor[B,C,L]
      labels  : [B]
      clip_ids: list[str]
    """
    clip_ids = batch.get("key", None)
    if clip_ids is None:
        clip_ids = batch.get("sample_name", None)
    if clip_ids is None:
        raise KeyError("Batch has neither 'key' nor 'sample_name' field.")

    tier_ids = batch["tier_ids"]
    labels = tier_ids[tier_mode]

    if use_modality == "rgb":
        inputs = batch["rgb"]

    elif use_modality == "depth":
        inputs = batch["depth"]

    elif use_modality == "mindrove":
        inputs = concat_mindrove_batch_dict(batch["mindrove"], args)

    else:
        raise ValueError(f"Unsupported modality: {use_modality}")

    return inputs, labels, clip_ids


def _ensure_bcthw(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    将 [B,T,C,H,W] 转成 3D CNN 常用的 [B,C,T,H,W]。
    """
    if x_btchw.ndim != 5:
        raise ValueError(f"Expect 5D tensor [B,T,C,H,W], got shape={tuple(x_btchw.shape)}")
    return x_btchw.permute(0, 2, 1, 3, 4).contiguous()


def preprocess_rgb_already_normed(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    map-style loader 内部已经完成 RGB 的空间增强、ToDtype 和 Normalize。
    因此这里不要重复 Normalize，只确保 dtype=float32。
    """
    if x_btchw.dtype != torch.float32:
        x_btchw = x_btchw.to(torch.float32)
    return x_btchw


def preprocess_depth_to_float(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    Depth 不额外做归一化，只转成 float32。
    """
    if x_btchw.dtype != torch.float32:
        x_btchw = x_btchw.to(torch.float32)
    return x_btchw

def _ensure_bcl(x_bcl: torch.Tensor) -> torch.Tensor:
    """
    确保 MindRove 输入为 [B,C,L]。
    """
    if x_bcl.ndim != 3:
        raise ValueError(f"Expect 3D tensor [B,C,L], got shape={tuple(x_bcl.shape)}")
    return x_bcl.contiguous()


def preprocess_mindrove_to_float(x_bcl: torch.Tensor) -> torch.Tensor:
    """
    MindRove 在这里不再做额外标准化。

    说明：
    - 若启用了标准化，已在 dataloader 内部完成
    - 这里仅保证 dtype=float32，避免训练脚本再次重复处理
    """
    if x_bcl.dtype != torch.float32:
        x_bcl = x_bcl.to(torch.float32)
    return x_bcl


def move_and_prepare_inputs(inputs, use_modality: str, device, args):
    """
    将不同模态输入统一变成模型可直接接受的格式。
    返回：
      rgb/depth  -> [B,C,T,H,W]
      mindrove   -> [B,C,L]
    """
    inputs = inputs.to(device, non_blocking=True)

    if use_modality == "rgb":
        inputs = preprocess_rgb_already_normed(inputs)
        inputs = _ensure_bcthw(inputs)
        return inputs

    if use_modality == "depth":
        inputs = preprocess_depth_to_float(inputs)
        inputs = _ensure_bcthw(inputs)
        return inputs

    if use_modality == "mindrove":
        inputs = preprocess_mindrove_to_float(inputs)
        inputs = _ensure_bcl(inputs)
        return inputs

    raise ValueError(f"Unsupported modality: {use_modality}")


# ============================================================
# 7) loss 构建
# ============================================================
def build_training_criterion(args, train_dataset, device):
    """
    构建训练损失函数。

    若启用：
    - Weighted CE
    - Focal(alpha)

    则先从 train_dataset.records 自动统计 class counts。
    """
    weights = None

    if args.use_weighted_ce or (args.use_focal and args.focal_use_alpha):
        class_counts = build_class_counts_from_mapstyle_dataset(
            dataset=train_dataset,
            tier_mode=args.tier_mode,
            num_classes=args.num_classes,
        )

        weights = build_class_weights_from_counts(
            counts=class_counts,
            num_classes=args.num_classes,
            method=args.weight_method,
            beta=args.cb_beta,
            normalize_mean=args.weight_normalize_mean,
        ).to(device)

        print("[class weights] source: auto-counted from train manifest via train_dataset.records")
        print("[class weights] tier_mode:", args.tier_mode)
        print("[class weights] method:", args.weight_method, "beta:", args.cb_beta,
              "normalize_mean:", args.weight_normalize_mean)
        print("[class weights] counts:", class_counts)
        print("[class weights] weights:", weights.detach().cpu().tolist())

    if args.use_focal:
        alpha = weights if args.focal_use_alpha else None
        print(f"[loss] use FocalLoss | gamma={args.focal_gamma} | use_alpha={args.focal_use_alpha}")
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha, reduction="mean").to(device)
    else:
        if args.use_weighted_ce:
            print("[loss] use Weighted CrossEntropyLoss.")
            criterion = nn.CrossEntropyLoss(weight=weights).to(device)
        else:
            print("[loss] use standard CrossEntropyLoss.")
            criterion = nn.CrossEntropyLoss().to(device)

    return criterion


# ============================================================
# 8) train / eval
# ============================================================
def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    criterion,
    training_dynamic_logger,
    device,
    tier_mode: str,
    use_modality: str,
    scaler_obj,
    amp_dtype,
    args,
    reverse_label_map: dict[int, str],
    save_datamap: bool = True,
    enable_amp: bool = False,
):
    model.train()

    total_seen = 0
    num_corrects = 0.0
    total_loss = 0.0

    all_labels = []
    all_preds = []

    use_amp = bool(enable_amp and device.type == "cuda")
    print(f"use amp: {use_amp}")

    # -------- 滑动窗口计时：便于观察瓶颈 --------
    data_times = deque(maxlen=50)
    prep_times = deque(maxlen=50)
    gpu_times = deque(maxlen=50)
    log_times = deque(maxlen=50)

    end = time.perf_counter()
    pbar = tqdm.tqdm(loader, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        # 1) data 时间
        t_data = time.perf_counter() - end
        data_times.append(t_data)

        # 2) 取数据
        inputs_raw, labels, clip_ids = _extract_inputs_and_labels(
            batch=batch,
            tier_mode=tier_mode,
            use_modality=use_modality,
            args=args,
        )

        labels = labels.to(device, non_blocking=True)

        t0_prep = time.perf_counter()
        inputs_model = move_and_prepare_inputs(
            inputs=inputs_raw,
            use_modality=use_modality,
            device=device,
            args=args,
        )
        prep_times.append(time.perf_counter() - t0_prep)

        if device.type == "cuda":
            torch.cuda.synchronize()
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            logits = model(inputs_model)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=-1)

        if use_amp and scaler_obj is not None and scaler_obj.is_enabled():
            scaler_obj.scale(loss).backward()
            scaler_obj.step(optimizer)
            scaler_obj.update()
        else:
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            ev_end.record()
            ev_end.synchronize()
            t_gpu = ev_start.elapsed_time(ev_end) / 1000.0
            gpu_times.append(t_gpu)

        # 6) 统计
        bs = inputs_model.size(0)
        total_seen += bs
        num_corrects += (preds == labels).sum().item()
        total_loss += loss.item() * bs

        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        # 7) Training Dynamics
        if save_datamap:
            t0_log = time.perf_counter()

            outputs = logits.detach().tolist()
            prob_true = probs.gather(1, labels.view(-1, 1)).squeeze(1).detach().cpu()
            td_preds = preds.detach().cpu()
            td_labels = labels.detach().cpu()

            if not isinstance(clip_ids, list):
                clip_ids = list(clip_ids)

            training_dynamic_logger.log_minibatch(
                uids=clip_ids,
                golds=td_labels.tolist(),
                logits=outputs,
                probs_true=prob_true.tolist(),
                preds=td_preds.tolist(),
                epoch=epoch,
            )

            log_times.append(time.perf_counter() - t0_log)

        # 8) 进度条显示
        md = sum(data_times) / len(data_times) if data_times else 0.0
        mp = sum(prep_times) / len(prep_times) if prep_times else 0.0
        mg = sum(gpu_times) / len(gpu_times) if gpu_times else 0.0
        ml = sum(log_times) / len(log_times) if log_times else 0.0

        pbar.set_postfix({
            "data(s)": f"{md:.3f}",
            "prep(s)": f"{mp:.3f}",
            "gpu(s)":  f"{mg:.3f}",
            "log(s)":  f"{ml:.3f}",
        })

        end = time.perf_counter()

    if save_datamap:
        training_dynamic_logger.flush_epoch(epoch)

    epoch_acc = num_corrects / max(1, total_seen)
    epoch_loss = total_loss / max(1, total_seen)

    train_metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        num_classes=args.num_classes,
        reverse_label_map=reverse_label_map,
    )

    print(
        f"[{epoch}] Epoch train acc: {epoch_acc:.4f}, "
        f"train loss: {epoch_loss:.4f}, "
        f"train balanced acc: {train_metrics['balanced_acc']:.4f}, "
        f"train macro-F1: {train_metrics['macro_f1']:.4f}"
    )

    return epoch_acc, epoch_loss, train_metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    tier_mode: str,
    use_modality: str,
    amp_dtype,
    args,
    reverse_label_map: dict[int, str],
    enable_amp: bool = False,
    split_name: str = "val",
):
    """
    统一验证 / 测试函数。

    split_name 仅用于日志显示，例如：
    - val
    - test
    """
    model.eval()

    total_seen = 0
    num_corrects = 0.0
    total_loss = 0.0

    all_labels = []
    all_preds = []

    use_amp = bool(enable_amp and device.type == "cuda")
    print(f"use amp: {use_amp}")

    for batch in tqdm.tqdm(loader, dynamic_ncols=True):
        inputs_raw, labels, _clip_ids = _extract_inputs_and_labels(
            batch=batch,
            tier_mode=tier_mode,
            use_modality=use_modality,
            args=args,
        )

        labels = labels.to(device, non_blocking=True)
        inputs_model = move_and_prepare_inputs(
            inputs=inputs_raw,
            use_modality=use_modality,
            device=device,
            args=args,
        )

        amp_ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            logits = model(inputs_model)
            loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=-1)

        bs = inputs_model.size(0)
        total_seen += bs
        num_corrects += (preds == labels).sum().item()
        total_loss += loss.item() * bs

        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    epoch_acc = num_corrects / max(1, total_seen)
    epoch_loss = total_loss / max(1, total_seen)

    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        num_classes=args.num_classes,
        reverse_label_map=reverse_label_map,
    )

    print(
        f"Epoch {split_name} acc: {epoch_acc:.4f}, "
        f"{split_name} loss: {epoch_loss:.4f}, "
        f"{split_name} balanced acc: {metrics['balanced_acc']:.4f}, "
        f"{split_name} macro-F1: {metrics['macro_f1']:.4f}"
    )

    return epoch_acc, epoch_loss, total_seen, metrics


@torch.no_grad()
def evaluate_test_with_per_sample_csv(
    model,
    loader,
    device,
    tier_mode: str,
    use_modality: str,
    amp_dtype,
    reverse_label_map: dict[int, str],
    per_sample_csv_path: str,
    args,
    enable_amp: bool = False,
    metrics_json_path: str | None = None,
):
    """
    专门用于 test 的详细评估函数。

    功能：
    1) 计算整体 test acc / test loss
    2) 为当前权重单独保存一个逐样本 CSV

    CSV 每行对应一个测试样本，包含：
    - sample_name
    - original_key
    - true_label_id
    - true_label_name
    - pred_label_id
    - pred_label_name
    - pred_confidence
    - true_class_probability
    - sample_loss
    - correct
    """
    model.eval()

    total_seen = 0
    num_corrects = 0.0
    total_loss = 0.0

    use_amp = bool(enable_amp and device.type == "cuda")
    print(f"use amp: {use_amp}")

    rows = []
    all_labels = []
    all_preds = []

    for batch in tqdm.tqdm(loader, dynamic_ncols=True):
        inputs_raw, labels, clip_ids = _extract_inputs_and_labels(
            batch=batch,
            tier_mode=tier_mode,
            use_modality=use_modality,
            args=args,
        )

        # 尽量保留 sample_name 与 original_key
        sample_names = batch.get("sample_name", None)
        original_keys = batch.get("key", None)

        if sample_names is None:
            sample_names = clip_ids
        if original_keys is None:
            original_keys = clip_ids

        if not isinstance(sample_names, list):
            sample_names = list(sample_names)
        if not isinstance(original_keys, list):
            original_keys = list(original_keys)

        labels = labels.to(device, non_blocking=True)

        inputs_model = move_and_prepare_inputs(
            inputs=inputs_raw,
            use_modality=use_modality,
            device=device,
            args=args,
        )

        amp_ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            logits = model(inputs_model)

        probs = torch.softmax(logits, dim=1)
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        preds = torch.argmax(logits, dim=-1)

        bs = inputs_model.size(0)
        total_seen += bs
        num_corrects += (preds == labels).sum().item()
        total_loss += per_sample_loss.sum().item()

        labels_cpu = labels.detach().cpu()
        preds_cpu = preds.detach().cpu()
        probs_cpu = probs.detach().cpu()
        loss_cpu = per_sample_loss.detach().cpu()

        all_labels.extend(labels_cpu.tolist())
        all_preds.extend(preds_cpu.tolist())

        for i in range(bs):
            true_id = int(labels_cpu[i].item())
            pred_id = int(preds_cpu[i].item())

            pred_confidence = float(probs_cpu[i, pred_id].item())
            true_class_probability = float(probs_cpu[i, true_id].item())

            is_correct = (true_id == pred_id)

            rows.append({
                "sample_name": sample_names[i],
                "original_key": original_keys[i],
                "true_label_id": true_id,
                "true_label_name": reverse_label_map.get(true_id, str(true_id)),
                "pred_label_id": pred_id,
                "pred_label_name": reverse_label_map.get(pred_id, str(pred_id)),
                "pred_confidence": pred_confidence,
                "true_class_probability": true_class_probability,
                "sample_loss": float(loss_cpu[i].item()),
                "correct": int(is_correct),
            })

    ensure_dir(os.path.dirname(per_sample_csv_path) or ".")

    with open(per_sample_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_name",
            "original_key",
            "true_label_id",
            "true_label_name",
            "pred_label_id",
            "pred_label_name",
            "pred_confidence",
            "true_class_probability",
            "sample_loss",
            "correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    epoch_acc = num_corrects / max(1, total_seen)
    epoch_loss = total_loss / max(1, total_seen)

    test_metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        num_classes=args.num_classes,
        reverse_label_map=reverse_label_map,
    )

    if metrics_json_path is not None:
        ensure_dir(os.path.dirname(metrics_json_path) or ".")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print(
        f"Epoch test acc: {epoch_acc:.4f}, "
        f"test loss: {epoch_loss:.4f}, "
        f"test balanced acc: {test_metrics['balanced_acc']:.4f}, "
        f"test macro-F1: {test_metrics['macro_f1']:.4f}"
    )
    print(f"[per-sample csv saved] {per_sample_csv_path}")
    if metrics_json_path is not None:
        print(f"[test metrics json saved] {metrics_json_path}")

    return epoch_acc, epoch_loss, total_seen, test_metrics


# ============================================================
# 9) checkpoint / 配置 / 结果保存
# ============================================================
def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    save_dir: str,
    model,
    optimizer,
    scaler_obj,
    epoch: int,
    args,
    is_best_val: bool = False,
    is_last: bool = False,
    extra_info: dict | None = None,
    ckpt_name: str | None = None,
):
    """
    保存 checkpoint。

    命名规则：
    - best_val.pth
    - last.pth
    - epoch_020.pth
    """
    if ckpt_name is not None:
        ckpt_path = os.path.join(save_dir, ckpt_name)
    elif is_best_val:
        ckpt_path = os.path.join(save_dir, "best_val.pth")
    elif is_last:
        ckpt_path = os.path.join(save_dir, "last.pth")
    else:
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pth")

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler_obj.state_dict() if (scaler_obj is not None and scaler_obj.is_enabled()) else None,
        "epoch": epoch,
        "args": vars(args),
    }
    if extra_info is not None:
        payload["extra_info"] = extra_info

    torch.save(payload, ckpt_path)


def save_train_summary_csv(csv_path: str, rows: list[dict], append: bool = False):
    """
    保存或追加汇总 CSV。

    参数
    ----
    csv_path : str
        目标 csv 路径
    rows : list[dict]
        要写入的多行数据
    append : bool
        False: 覆盖写入
        True : 追加写入；若文件不存在则自动写表头，若已存在则只追加内容
    """
    if len(rows) == 0:
        return

    ensure_dir(os.path.dirname(csv_path) or ".")
    fieldnames = list(rows[0].keys())

    file_exists = os.path.isfile(csv_path)
    mode = "a" if append else "w"

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # 只有以下两种情况才写表头：
        # 1) 覆盖写入
        # 2) 追加写入但文件原本不存在
        if (not append) or (not file_exists):
            writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 10) 训练实验源构造（按输入顺序对齐）
# ============================================================
def build_training_sources(args):
    """
    按输入顺序构造训练实验源。

    本函数是本版本最关键的逻辑之一。
    它彻底替代了旧版本基于文件名内容自动匹配的方式。

    返回格式：
        [
            {
                "pretrained_path": ... 或 None,
                "train_manifest": ...,
                "val_manifest": ... 或 None,
            },
            ...
        ]

    规则总结：
    ----------------------------------------
    A) 没有预训练权重
       - 实验数 = train manifests 数量
       - 每个 train manifest 对应一个实验
       - val manifests 可 0/1/N 个（N=实验数）

    B) 有预训练权重
       - 实验数 = pretrained_weight_paths 数量
       - train manifests 可 1/N 个（N=实验数）
       - val manifests 可 0/1/N 个（N=实验数）
       - 按顺序或广播对齐

    C) include_scratch_baseline=True
       - 会额外追加 scratch 实验
       - scratch 实验按“主实验解析后的 (train_manifest, val_manifest) 组合”去重后追加
       - 这样：
            * 若单个 train/val 被广播，只会追加 1 个 scratch baseline
            * 若不同 train/val 对应不同实验，则会为每个不同组合追加一个 scratch baseline
    """
    train_manifest_list = build_train_manifest_list(args)
    val_manifest_list = build_val_manifest_list(args)
    pretrained_list = normalize_string_list(args.pretrained_weight_paths)

    sources = []

    # ---------------- 情况 A：没有预训练权重，只跑 scratch ----------------
    if len(pretrained_list) == 0:
        num_runs = len(train_manifest_list)
        aligned_train = align_required_sequence(train_manifest_list, num_runs, "train manifest(s)")
        aligned_val = align_optional_sequence(val_manifest_list, num_runs, "val manifest(s)")

        for i in range(num_runs):
            sources.append({
                "pretrained_path": None,
                "train_manifest": aligned_train[i],
                "val_manifest": aligned_val[i],
            })
        return sources

    # ---------------- 情况 B：有预训练权重，按顺序对齐 ----------------
    num_runs = len(pretrained_list)
    aligned_train = align_required_sequence(train_manifest_list, num_runs, "train manifest(s)")
    aligned_val = align_optional_sequence(val_manifest_list, num_runs, "val manifest(s)")

    for i in range(num_runs):
        sources.append({
            "pretrained_path": pretrained_list[i],
            "train_manifest": aligned_train[i],
            "val_manifest": aligned_val[i],
        })

    # ---------------- 情况 C：额外追加 scratch baseline ----------------
    if args.include_scratch_baseline:
        seen_pairs = set()
        for i in range(num_runs):
            pair = (aligned_train[i], aligned_val[i])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                sources.append({
                    "pretrained_path": None,
                    "train_manifest": aligned_train[i],
                    "val_manifest": aligned_val[i],
                })

    return sources


def build_run_name(pretrained_path: str | None, train_manifest: str, val_manifest: str | None, run_index: int, args) -> str:
    train_short = compact_manifest_stem(train_manifest)

    if val_manifest is None:
        val_short = "noval"
    else:
        val_short = compact_manifest_stem(val_manifest)

    src = build_pretrained_src_tag(pretrained_path, args)

    mode = args.finetune_mode

    if args.use_discriminative_lr and args.finetune_mode == "full":
        blr = args.backbone_learning_rate if args.backbone_learning_rate is not None else args.learning_rate
        hlr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
        lr_tag = f"blr{blr:g}_hlr{hlr:g}"
    elif args.finetune_mode == "head_only":
        head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
        lr_tag = f"headlr{head_lr:g}"
    else:
        lr_tag = f"lr{args.learning_rate:g}"

    return f"run_{run_index:02d}_{src}"
    # return f"w{run_index:02d}"


# ============================================================
# 11) 单次训练实验
# ============================================================
def run_one_training_experiment(
    args,
    device,
    trainloader,
    valloader,
    train_dataset,
    run_index: int,
    pretrained_path: str | None,
    train_manifest_used: str,
    val_manifest_used: str | None,
):
    """
    执行一次完整训练实验。

    这里的一次实验定义为：
    - scratch 训练，或
    - 从某一个预训练权重初始化后进行微调

    每个实验都会写到独立子目录，避免多个实验互相覆盖。
    """
    run_name = build_run_name(
        pretrained_path=pretrained_path,
        train_manifest=train_manifest_used,
        val_manifest=val_manifest_used,
        run_index=run_index,
        args=args,
    )
    run_dir = os.path.join(args.save_path, run_name)
    datamap_dir = os.path.join(args.datamap_csv_path, run_name)
    ensure_dir(run_dir)
    ensure_dir(datamap_dir)

    print(f"\n==================== [TRAIN EXPERIMENT] {run_name} ====================")
    print(f"train_manifest_used: {train_manifest_used}")
    print(f"val_manifest_used:   {val_manifest_used}")
    if pretrained_path is None:
        print("Source: scratch")
    else:
        print(f"Source pretrained: {pretrained_path}")

    reverse_label_map = build_reverse_label_map(args.label_map_json, args.tier_mode)

    training_dynamics_logger = TrainingDynamicsLogger(datamap_dir)
    model = prepare_model(args).to(device)

    pretrained_report = None
    if pretrained_path is not None:
        pretrained_report = load_pretrained_weights(
            model=model,
            ckpt_path=pretrained_path,
            drop_pretrained_head=(not args.keep_pretrained_head),
            strict=args.pretrained_strict,
            map_location="cpu",
        )
        print("[pretrained load report]")
        print(json.dumps(pretrained_report, indent=2, ensure_ascii=False))

    criterion = build_training_criterion(args, train_dataset, device)
    optimizer, optimizer_meta = build_optimizer(model, args)

    scaler_obj = GradScaler(enabled=(device.type == "cuda" and args.enable_amp and (not use_bf16)))

    config_payload = {
        "run_name": run_name,
        "pretrained_path": pretrained_path,
        "train_manifest_used": train_manifest_used,
        "val_manifest_used": val_manifest_used,
        "optimizer_meta": optimizer_meta,
        "keep_pretrained_head": args.keep_pretrained_head,
        "pretrained_strict": args.pretrained_strict,
        "args": vars(args),
        "pretrained_report": pretrained_report,
    }
    save_json(os.path.join(run_dir, "config.json"), config_payload)

    best_val_acc = -1.0
    best_val_acc_loss = None
    best_val_acc_epoch = -1

    best_val_macro_f1 = -1.0
    best_val_macro_f1_loss = None
    best_val_macro_f1_epoch = -1

    best_val_balanced_acc = -1.0
    best_val_balanced_loss = None
    best_val_balanced_epoch = -1

    final_train_acc = None
    final_train_loss = None
    final_train_macro_f1 = None
    final_train_balanced_acc = None

    final_val_acc = None
    final_val_loss = None
    final_val_macro_f1 = None
    final_val_balanced_acc = None

    log_file = os.path.join(run_dir, "train_logs.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"run_name: {run_name}\n")
        f.write(f"pretrained_path: {pretrained_path}\n")
        f.write(f"train_manifest_used: {train_manifest_used}\n")
        f.write(f"val_manifest_used: {val_manifest_used}\n")
        f.write(f"optimizer_meta: {json.dumps(optimizer_meta, ensure_ascii=False)}\n")
        if pretrained_report is not None:
            f.write("pretrained_report:\n")
            f.write(json.dumps(pretrained_report, indent=2, ensure_ascii=False) + "\n")
        f.write("\n")

        for epoch in range(args.epochs):
            lr_dict = adjust_learning_rate(optimizer, epoch, args)

            train_acc, train_loss, train_metrics = train_one_epoch(
                epoch=epoch,
                model=model,
                loader=trainloader,
                optimizer=optimizer,
                criterion=criterion,
                training_dynamic_logger=training_dynamics_logger,
                device=device,
                tier_mode=args.tier_mode,
                use_modality=args.use_modality,
                scaler_obj=scaler_obj,
                amp_dtype=amp_dtype,
                args=args,
                reverse_label_map=reverse_label_map,
                save_datamap=True,
                enable_amp=bool(args.enable_amp),
            )

            final_train_acc = train_acc
            final_train_loss = train_loss
            final_train_macro_f1 = train_metrics["macro_f1"]
            final_train_balanced_acc = train_metrics["balanced_acc"]


            extra_info = {
                "run_name": run_name,
                "pretrained_path": pretrained_path,
                "optimizer_meta": optimizer_meta,
                "epoch": epoch,
                "lr_dict": lr_dict,
            }

            if valloader is not None:
                val_acc, val_loss, _, val_metrics = evaluate(
                    model=model,
                    loader=valloader,
                    criterion=criterion,
                    device=device,
                    tier_mode=args.tier_mode,
                    use_modality=args.use_modality,
                    amp_dtype=amp_dtype,
                    args=args,
                    reverse_label_map=reverse_label_map,
                    enable_amp=bool(args.enable_amp),
                    split_name="val",
                )

                final_val_acc = val_acc
                final_val_loss = val_loss
                final_val_macro_f1 = val_metrics["macro_f1"]
                final_val_balanced_acc = val_metrics["balanced_acc"]

                eligible_for_best = epoch >= args.best_after_epoch

                val_macro_f1 = val_metrics["macro_f1"]
                val_balanced_acc = val_metrics["balanced_acc"]

                extra_info_with_metrics = {
                    **extra_info,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "train_balanced_acc": train_metrics["balanced_acc"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_balanced_acc": val_balanced_acc,
                    "val_macro_f1": val_macro_f1,
                }

                if eligible_for_best and (val_acc > best_val_acc):
                    best_val_acc = val_acc
                    best_val_acc_loss = val_loss
                    best_val_acc_epoch = epoch+1

                    save_checkpoint(
                        save_dir=run_dir,
                        model=model,
                        optimizer=optimizer,
                        scaler_obj=scaler_obj,
                        epoch=epoch+1,
                        args=args,
                        ckpt_name="best_val.pth",
                        extra_info={
                            **extra_info_with_metrics,
                            "selection_metric": "val_acc",
                            "selection_metric_value": val_acc,
                        },
                    )

                if eligible_for_best and (val_macro_f1 > best_val_macro_f1):
                    best_val_macro_f1 = val_macro_f1
                    best_val_macro_f1_loss = val_loss
                    best_val_macro_f1_epoch = epoch+1

                    save_checkpoint(
                        save_dir=run_dir,
                        model=model,
                        optimizer=optimizer,
                        scaler_obj=scaler_obj,
                        epoch=epoch+1,
                        args=args,
                        ckpt_name="best_val_macro_f1.pth",
                        extra_info={
                            **extra_info_with_metrics,
                            "selection_metric": "val_macro_f1",
                            "selection_metric_value": val_macro_f1,
                        },
                    )

                if eligible_for_best and (val_balanced_acc > best_val_balanced_acc):
                    best_val_balanced_acc = val_balanced_acc
                    best_val_balanced_loss = val_loss
                    best_val_balanced_epoch = epoch+1

                    save_checkpoint(
                        save_dir=run_dir,
                        model=model,
                        optimizer=optimizer,
                        scaler_obj=scaler_obj,
                        epoch=epoch+1,
                        args=args,
                        ckpt_name="best_val_balanced.pth",
                        extra_info={
                            **extra_info_with_metrics,
                            "selection_metric": "val_balanced_acc",
                            "selection_metric_value": val_balanced_acc,
                        },
                    )

                if args.save_period > 0 and ((epoch + 1) % args.save_period == 0):
                    save_checkpoint(
                        save_dir=run_dir,
                        model=model,
                        optimizer=optimizer,
                        scaler_obj=scaler_obj,
                        epoch=epoch+1,
                        args=args,
                        is_best_val=False,
                        is_last=False,
                        extra_info=extra_info_with_metrics,
                    )

                f.write(
                    f"[{epoch}] | {format_lr_dict(lr_dict)} | "
                    f"train loss: {train_loss:.4f}, "
                    f"{format_metrics_for_log(train_metrics, 'train')} | "
                    f"val loss: {val_loss:.4f}, "
                    f"{format_metrics_for_log(val_metrics, 'val')}\n"
                )

                f.write(
                    "    train_per_class_acc: "
                    + json.dumps(round_metric_dict(train_metrics["per_class_acc"], 4), ensure_ascii=False)
                    + "\n"
                )

                f.write(
                    "    val_per_class_acc: "
                    + json.dumps(round_metric_dict(val_metrics["per_class_acc"], 4), ensure_ascii=False)
                    + "\n"
                )

                f.write(
                    "    val_per_class_support: "
                    + json.dumps(val_metrics["per_class_support"], ensure_ascii=False)
                    + "\n"
                )
            else:
                f.write(
                f"[{epoch}] | {format_lr_dict(lr_dict)} | "
                f"train loss: {train_loss:.4f}, "
                f"{format_metrics_for_log(train_metrics, 'train')} | "
                f"no validation\n"
            )

                f.write(
                    "    train_per_class_acc: "
                    + json.dumps(round_metric_dict(train_metrics["per_class_acc"], 4), ensure_ascii=False)
                    + "\n"
                )

            f.flush()

        save_checkpoint(
        save_dir=run_dir,
        model=model,
        optimizer=optimizer,
        scaler_obj=scaler_obj,
        epoch=args.epochs,
        args=args,
        is_last=True,
        extra_info={
            "run_name": run_name,
            "pretrained_path": pretrained_path,
            "optimizer_meta": optimizer_meta,

            "best_val_acc": best_val_acc,
            "best_val_acc_loss": best_val_acc_loss,
            "best_val_acc_epoch": best_val_acc_epoch,

            "best_val_macro_f1": best_val_macro_f1,
            "best_val_macro_f1_loss": best_val_macro_f1_loss,
            "best_val_macro_f1_epoch": best_val_macro_f1_epoch,

            "best_val_balanced_acc": best_val_balanced_acc,
            "best_val_balanced_loss": best_val_balanced_loss,
            "best_val_balanced_epoch": best_val_balanced_epoch,
        },
    )

    summary = {
        "run_name": run_name,
        "pretrained_path": pretrained_path if pretrained_path is not None else "scratch",
        "finetune_mode": args.finetune_mode,
        "train_manifest_used": train_manifest_used,
        "val_manifest_used": val_manifest_used,
        "use_discriminative_lr": bool(args.use_discriminative_lr),
        "backbone_learning_rate": args.backbone_learning_rate if args.backbone_learning_rate is not None else args.learning_rate,
        "head_learning_rate": args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate,
        "optimizer_mode": optimizer_meta["mode"],
        "optimizer": optimizer_meta.get("optimizer", args.optimizer),
        "optimizer_weight_decay": optimizer_meta.get("weight_decay", args.weight_decay),
        "optimizer_momentum": optimizer_meta.get("momentum", None),
        "optimizer_adamw_beta1": optimizer_meta.get("adamw_beta1", None),
        "optimizer_adamw_beta2": optimizer_meta.get("adamw_beta2", None),
        "optimizer_adamw_eps": optimizer_meta.get("adamw_eps", None),
        "l2_normalize_before_fc": bool(args.l2_normalize_before_fc),
        "disable_train_augmentation": bool(args.disable_train_augmentation),
        "num_trainable_params": optimizer_meta["num_trainable_params"],
        "final_train_acc": final_train_acc,
        "final_train_loss": final_train_loss,
        "final_train_macro_f1": final_train_macro_f1,
        "final_train_balanced_acc": final_train_balanced_acc,

        "best_val_acc": best_val_acc if valloader is not None else None,
        "best_val_acc_loss": best_val_acc_loss if valloader is not None else None,
        "best_val_acc_epoch": best_val_acc_epoch if valloader is not None else None,

        "best_val_macro_f1": best_val_macro_f1 if valloader is not None else None,
        "best_val_macro_f1_loss": best_val_macro_f1_loss if valloader is not None else None,
        "best_val_macro_f1_epoch": best_val_macro_f1_epoch if valloader is not None else None,

        "best_val_balanced_acc": best_val_balanced_acc if valloader is not None else None,
        "best_val_balanced_loss": best_val_balanced_loss if valloader is not None else None,
        "best_val_balanced_epoch": best_val_balanced_epoch if valloader is not None else None,

        "final_val_acc": final_val_acc if valloader is not None else None,
        "final_val_loss": final_val_loss if valloader is not None else None,
        "final_val_macro_f1": final_val_macro_f1 if valloader is not None else None,
        "final_val_balanced_acc": final_val_balanced_acc if valloader is not None else None,
        "run_dir": run_dir,
    }

    save_json(os.path.join(run_dir, "summary.json"), summary)
    print("[train summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================
# 12) 批量测试（按输入顺序对齐）
# ============================================================
def run_batch_test(args, device):
    """
    批量加载多个已训练好的分类模型权重，并在一个或多个 test manifest 上逐个测试。

    顺序匹配规则：
    ----------------------------------------
    - 若 test_manifest 只给 1 个，则广播给所有 test_weight_paths
    - 若给多个，则 test_manifest[i] 对应 test_weight_paths[i]

    输出两类结果：
    1) 总汇总 CSV：保存到 args.test_results_csv 或 args.save_path/test_results.csv
    2) 每个权重单独的逐样本 CSV：保存到该权重所在目录，避免重名冲突
    """
    ensure_dir(args.save_path)

    test_manifest_list = build_test_manifest_list(args)
    test_weight_list = normalize_string_list(args.test_weight_paths)

    aligned_test_manifests = align_required_sequence(
        test_manifest_list,
        len(test_weight_list),
        "test manifest(s)"
    )

    reverse_label_map = build_reverse_label_map(args.label_map_json, args.tier_mode)

    rows = []
    test_loader_cache = {}

    for idx, (test_manifest_used, weight_path) in enumerate(zip(aligned_test_manifests, test_weight_list), start=1):
        print(f"\n==================== [TEST {idx}/{len(test_weight_list)}] ====================")
        print(f"test_manifest_used: {test_manifest_used}")
        print(f"weight_path:        {weight_path}")

        # 若多个权重共用同一个 test manifest，则复用 loader，减少重复开销
        if test_manifest_used not in test_loader_cache:
            test_loader_cache[test_manifest_used] = prepare_test_loader_for_manifest(args, test_manifest_used)

        testloader = test_loader_cache[test_manifest_used]

        model = prepare_model(args).to(device)
        load_report = load_model_weights_for_eval(model, weight_path, map_location="cpu")
        print("[eval load report]")
        print(json.dumps(load_report, indent=2, ensure_ascii=False))

        # per_sample_test 详细命名
        # weight_path_obj = Path(weight_path)
        # weight_dir = str(weight_path_obj.parent)
        # weight_stem = sanitize_name(weight_path_obj.stem)
        # test_manifest_stem = sanitize_name(Path(test_manifest_used).stem)

        # per_sample_csv_name = (
        #     f"{weight_stem}_{test_manifest_stem}_{args.tier_mode}_{args.use_modality}_per_sample.csv"
        # )
        # per_sample_csv_path = os.path.join(weight_dir, per_sample_csv_name)

        # per_sample_test 简略命名，放置超过 260 字符限制
        weight_path_obj = Path(weight_path)
        weight_dir = str(weight_path_obj.parent)

        # 固定文件名，避免路径过长
        weight_stem = sanitize_name(Path(weight_path).stem)
        per_sample_csv_name = "per_sample_test.csv"
        per_sample_csv_path = os.path.join(
            weight_dir,
            f"{weight_stem}_per_sample_test.csv"
        )
        test_metrics_json_path = os.path.join(
            weight_dir,
            f"{weight_stem}_test_metrics.json"
        )

        test_acc, test_loss, num_samples, test_metrics = evaluate_test_with_per_sample_csv(
            model=model,
            loader=testloader,
            device=device,
            tier_mode=args.tier_mode,
            use_modality=args.use_modality,
            amp_dtype=amp_dtype,
            reverse_label_map=reverse_label_map,
            per_sample_csv_path=per_sample_csv_path,
            enable_amp=bool(args.enable_amp),
            args=args,
            metrics_json_path=test_metrics_json_path,
        )

        row = {
            "weight_path": weight_path,
            "weight_name": Path(weight_path).name,
            "test_manifest_used": test_manifest_used,
            "tier_mode": args.tier_mode,
            "use_modality": args.use_modality,
            "num_samples": num_samples,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_balanced_acc": test_metrics["balanced_acc"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_num_present_classes": test_metrics["num_present_classes"],
            "test_metrics_json": test_metrics_json_path,
            "test_per_class_acc_json": json.dumps(
                round_metric_dict(test_metrics["per_class_acc"], 4),
                ensure_ascii=False,
            ),
            "num_loaded_keys": load_report["num_loaded"],
            "num_missing_after_load": len(load_report["missing_keys_after_load"]),
            "num_unexpected_after_load": len(load_report["unexpected_keys_after_load"]),
            "per_sample_csv": per_sample_csv_path,
        }
        rows.append(row)

    csv_path = args.test_results_csv
    if not csv_path:
        csv_path = os.path.join(args.save_path, "test_results.csv")

    save_train_summary_csv(csv_path, rows, append=True)
    print(f"\n[test results saved] {csv_path}")


# ============================================================
# 13) main
# ============================================================
def main(args):
    validate_args(args)

    if args.seed is not None:
        seed_everything(args.seed)
        print(f"using random seed: {args.seed}")
    else:
        print("training without using random seed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"use_bf16_supported: {use_bf16}")

    if args.run_mode == "train":
        all_summaries = []
        sources = build_training_sources(args)

        # 为了避免同一份 manifest 被重复构建，使用 cache
        # cache key 由 (train_manifest, val_manifest) 组成
        loader_cache = {}

        for run_index, source in enumerate(sources, start=1):
            train_manifest_used = source["train_manifest"]
            val_manifest_used = source["val_manifest"]
            pretrained_path = source["pretrained_path"]

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
                pretrained_path=pretrained_path,
                train_manifest_used=train_manifest_used,
                val_manifest_used=val_manifest_used,
            )
            all_summaries.append(summary)

        summary_csv = os.path.join(args.save_path, "train_experiment_summary.csv")
        save_train_summary_csv(summary_csv, all_summaries, append=True)
        print(f"\n[train summary csv saved] {summary_csv}")

    elif args.run_mode == "test":
        run_batch_test(args, device)

    else:
        raise ValueError(f"Unknown run_mode: {args.run_mode}")


if __name__ == "__main__":
    main(args)
