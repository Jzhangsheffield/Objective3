#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import math
import random
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from backbone.renet1d_my import build_resnet1d
from backbone.MoCo_VAR_supcon_wds_1D import MoCo1D
from utils_.mapstype_dataloader_with_index_mindrove_modified_varlen import (
    PackedMultiModalConfig,
    load_label_map_json,
    build_packed_mapstyle_dataset,
    build_packed_mapstyle_loader_from_dataset,
)
from utils_.build_update_prototype_mapstyle_varproto_mindrove_modified_varlen import (
    PrototypeRefreshConfig,
    broadcast_proto_state,
    refresh_prototypes,
    summarize_proto_state,
)
from loss.prorotype_contrastive_loss_mapstyle_varproto import prototype_contrastive_loss_all_positive
from loss.prototype_directional_loss_mapstyle_varproto_topk import (
    differentiable_ema_directional_loss,
    ema_update_prototype_bank_,
)
from loss.proto_supcon_losses import SupLoss

# ============================================================
# 命令行参数
# ============================================================

parser = argparse.ArgumentParser(
        description="1D MindRove MoCo + supervised contrastive + prototype contrastive + prototype directional loss on map-style dataset (variable prototypes per class)"
        )

# ---------------- 数据与 I/O ----------------
parser.add_argument("--dataset_root", required=True, type=str,
                    help="map-style dataset root directory")
parser.add_argument("--train_manifest_name", default="train_manifest.jsonl", type=str,
                    help="training manifest file name under dataset_root")
parser.add_argument("--label_map_json", default="label_map.json", type=str,
                    help="label_map.json path; can be absolute or relative to dataset_root")
parser.add_argument("--weight_save_path", default=r"./weight", type=str,
                    help="checkpoint output directory")

# ---------------- 标签与 clip 设置 ----------------
parser.add_argument("--tier_mode", default="tier1", choices=["tier1", "tier2", "tier3"],
                    help="which tier id to use as the training label")
# parser.add_argument("--n_frames", default=16, type=int,
#                     help="number of frames per clip")

# ---------------- 训练 DataLoader ----------------
parser.add_argument("--batch_size", default=32, type=int,
                    help="training batch size per GPU/process")
parser.add_argument("--num_workers", default=8, type=int,
                    help="number of workers for the training DataLoader")
parser.add_argument("--prefetch_factor", default=None, type=int,
                    help="optional prefetch_factor for the training DataLoader")
parser.add_argument("--pin_memory", action="store_true",
                    help="enable pin_memory for the training DataLoader")
parser.add_argument("--verify_paths_on_init", action="store_true",
                    help="verify sample paths when constructing the training dataset")

# ---------------- 模型参数 ----------------
parser.add_argument("--ts_arch",default="resnet10_1d",choices=["resnet10_1d", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
                    help="which torchvision-style ResNet1D architecture to use")
parser.add_argument("--ts_base_channels",default=64,type=int,
                    help="base channel width of ResNet1D stem and stage1")
parser.add_argument("--ts_stem_kernel_size",default=7,type=int,
                    help="stem Conv1d kernel size of ResNet1D")
parser.add_argument("--ts_stem_stride", default=2, type=int,
                    help="stem Conv1d stride of ResNet1D")
parser.add_argument("--ts_use_stem_pool", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to use MaxPool1d after the stem")
parser.add_argument("--ts_zero_init_residual", action=argparse.BooleanOptionalAction, default=False,
                    help="whether to zero-initialize the last BN in each residual branch")
parser.add_argument("--proj_dim", default=128, type=int,
                    help="projection head output dimension")
parser.add_argument("--K_queue", default=3200, type=int,
                    help="MoCo queue size")
parser.add_argument("--temperature", default=0.07, type=float,
                    help="global temperature used in the KCL branch")
parser.add_argument("--mlp", action="store_true",
                    help="use MLP projection head")

# ---------------- mindrove 参数 ---------------
parser.add_argument("--mindrove_target_len", default=256, type=int,
                    help="fallback target sequence length after resampling MindRove streams")
parser.add_argument("--mindrove_emg_target_len", default=None, type=int,
                    help="optional EMG-specific target length; if None, falls back to --mindrove_target_len")
parser.add_argument("--mindrove_imu_target_len", default=None, type=int,
                    help="optional IMU-specific target length; if None, falls back to --mindrove_target_len")
parser.add_argument("--mindrove_pack_length_policy", default="max",
                    choices=["max", "min", "emg", "imu", "fixed", "error"],
                    help=(
                        "policy used before concatenating multiple MindRove keys into one [B,C,L] "
                        "tensor for the single-branch ResNet1D. If EMG/IMU lengths differ, "
                        "streams are resampled to this common length. 'fixed' requires "
                        "--mindrove_pack_target_len; 'error' preserves strict old behavior."
                    ))
parser.add_argument("--mindrove_pack_target_len", default=None, type=int,
                    help="explicit common length used when --mindrove_pack_length_policy=fixed, or as an override for any policy")
parser.add_argument("--mindrove_hands", nargs="+", default=["left", "right"],
                    choices=["left", "right"],
                    help="which hands to load from MindRove")
parser.add_argument("--mindrove_signals", nargs="+", default=["emg"],
                    choices=["emg", "imu"],
                    help="which MindRove signals to load")
parser.add_argument("--mindrove_merge_hands", action="store_true",
                    help="merge left/right streams of the same signal inside the dataloader")
parser.add_argument("--mindrove_apply_augmentation", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to apply MindRove augmentation in training loader")

# ---------------- MindRove normalization params ----------------
# 说明：
# 1) 这些参数用于把训练集统计得到的逐通道 mean / std 直接下传给 dataloader
# 2) 训练脚本不自己做标准化，只负责解析并传入 PackedMultiModalConfig
# 3) 这里仍然沿用与 drift 相同的风格：
#    - CLI 用字符串接收 list/tuple
#    - 在构造 PackedMultiModalConfig 时用 _parse_python_literal_arg 严格解析
# 4) 若启用标准化但某组 mean/std 缺失、长度不对、std<=0，
#    将由 dataloader 直接报错，不做兜底

parser.add_argument("--mindrove_apply_normalization", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to apply per-channel mean/std normalization for MindRove streams in the dataloader")

parser.add_argument("--mindrove_left_emg_mean", default=None, type=str,
                    help="left EMG per-channel mean, e.g. '[m1, ..., m8]'")
parser.add_argument("--mindrove_left_emg_std", default=None, type=str,
                    help="left EMG per-channel std, e.g. '[s1, ..., s8]'")
parser.add_argument("--mindrove_right_emg_mean", default=None, type=str,
                    help="right EMG per-channel mean, e.g. '[m1, ..., m8]'")
parser.add_argument("--mindrove_right_emg_std", default=None, type=str,
                    help="right EMG per-channel std, e.g. '[s1, ..., s8]'")

parser.add_argument("--mindrove_left_imu_mean", default=None, type=str,
                    help="left IMU per-channel mean, e.g. '[m1, ..., m6]' where channels are [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]'")
parser.add_argument("--mindrove_left_imu_std", default=None, type=str,
                    help="left IMU per-channel std, e.g. '[s1, ..., s6]' where channels are [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]'")
parser.add_argument("--mindrove_right_imu_mean", default=None, type=str,
                    help="right IMU per-channel mean, e.g. '[m1, ..., m6]' where channels are [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]'")
parser.add_argument("--mindrove_right_imu_std", default=None, type=str,
                    help="right IMU per-channel std, e.g. '[s1, ..., s6]' where channels are [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]'")

# ---------------- MindRove augmentation params ----------------
# 说明：
# 1) 这些参数会原样传入 PackedMultiModalConfig，再由 dataloader 传给
#    aug.mindrove_augmentation_tensor.apply_mindrove_augmentation
# 2) 训练脚本本身不实现增强逻辑，只负责把 CLI 配置正确下传
# 3) drift_max / drift_n_points 使用字符串接收，再用 ast.literal_eval 严格解析
#    例如：
#       --mindrove_emg_drift_max 0.1
#       --mindrove_emg_drift_max "(0.05, 0.15)"
#       --mindrove_emg_drift_n_points 3
#       --mindrove_emg_drift_n_points "[2,3,4]"

# time warp
parser.add_argument("--mindrove_time_warp_prob", default=0.5, type=float,
                    help="probability of applying shared time warp within each hand")
parser.add_argument("--mindrove_time_warp_sigma", default=0.2, type=float,
                    help="sigma used when sampling the shared time warp speed curve")
parser.add_argument("--mindrove_time_warp_num_knots", default=3, type=int,
                    help="number of interior knots used by time warp")
parser.add_argument("--mindrove_time_warp_num_splines", default=150, type=int,
                    help="kept for config compatibility with the augmentation module")

# EMG scaling / noise
parser.add_argument("--mindrove_emg_scaling_prob", default=0.8, type=float,
                    help="probability of applying EMG scaling")
parser.add_argument("--mindrove_emg_scaling_sigma", default=0.1, type=float,
                    help="sigma of EMG channel-wise scaling factors")
parser.add_argument("--mindrove_emg_noise_prob", default=0.8, type=float,
                    help="probability of applying EMG additive Gaussian noise")
parser.add_argument("--mindrove_emg_noise_sigma", default=0.05, type=float,
                    help="sigma of EMG additive Gaussian noise")

# EMG drift
parser.add_argument("--mindrove_emg_drift_prob", default=0.3, type=float,
                    help="probability of applying EMG drift augmentation")
parser.add_argument("--mindrove_emg_drift_max", default="0.5", type=str,
                    help="EMG drift max_drift; float or tuple string, e.g. 0.1 or '(0.05, 0.15)'")
parser.add_argument("--mindrove_emg_drift_n_points", default="3", type=str,
                    help="EMG drift n_drift_points; int or list string, e.g. 3 or '[2,3,4]'")
parser.add_argument("--mindrove_emg_drift_kind", default="additive", choices=["additive", "multiplicative"],
                    help="EMG drift mode")
parser.add_argument("--mindrove_emg_drift_per_channel", action=argparse.BooleanOptionalAction, default=False,
                    help="whether EMG drift samples an independent smooth trend for each channel")
parser.add_argument("--mindrove_emg_drift_normalize", action=argparse.BooleanOptionalAction, default=True,
                    help="whether EMG additive drift is scaled by per-channel value range")

# IMU scaling / noise
parser.add_argument("--mindrove_imu_scaling_prob", default=0.8, type=float,
                    help="probability of applying IMU scaling")
parser.add_argument("--mindrove_imu_scaling_sigma", default=0.1, type=float,
                    help="sigma of IMU channel-wise scaling factors")
parser.add_argument("--mindrove_imu_noise_prob", default=0.8, type=float,
                    help="probability of applying IMU additive Gaussian noise")
parser.add_argument("--mindrove_imu_noise_sigma", default=0.05, type=float,
                    help="sigma of IMU additive Gaussian noise")

# IMU drift
parser.add_argument("--mindrove_imu_drift_prob", default=0.3, type=float,
                    help="probability of applying IMU drift augmentation")
parser.add_argument("--mindrove_imu_drift_max", default="0.5", type=str,
                    help="IMU drift max_drift; float or tuple string, e.g. 0.03 or '(0.01, 0.05)'")
parser.add_argument("--mindrove_imu_drift_n_points", default="3", type=str,
                    help="IMU drift n_drift_points; int or list string, e.g. 3 or '[2,3,4]'")
parser.add_argument("--mindrove_imu_drift_kind", default="additive", choices=["additive", "multiplicative"],
                    help="IMU drift mode")
parser.add_argument("--mindrove_imu_drift_per_channel", action=argparse.BooleanOptionalAction, default=False,
                    help="whether IMU drift samples an independent smooth trend for each channel")
parser.add_argument("--mindrove_imu_drift_normalize", action=argparse.BooleanOptionalAction, default=False,
                    help="whether IMU additive drift is scaled by per-channel value range")

# negate
parser.add_argument("--mindrove_emg_negate_prob", default=0.3, type=float,
                    help="probability of applying EMG negation")
parser.add_argument("--mindrove_imu_negate_prob", default=0.3, type=float,
                    help="probability of applying IMU negation")

# channel dropout
parser.add_argument("--mindrove_emg_channel_dropout_prob", default=0.5, type=float,
                    help="probability of applying EMG channel dropout")
parser.add_argument("--mindrove_emg_channel_dropout_max_channels", default=3, type=int,
                    help="maximum number of EMG channels dropped in one augmentation call")
parser.add_argument("--mindrove_imu_channel_dropout_prob", default=0.5, type=float,
                    help="probability of applying IMU channel dropout")
parser.add_argument("--mindrove_imu_channel_dropout_max_channels", default=2, type=int,
                    help="maximum number of IMU channels dropped in one augmentation call")

# ---------------- 对比损失 参数 ----------------
parser.add_argument("--contrastive_loss", default="suploss", choices=["kcl", "suploss"],
                    help="which supervised contrastive branch to use: 'kcl' or 'suploss'")
parser.add_argument("--num_positive", default=6, type=int,
                    help="number of same-class positives sampled from the queue; 0 means use all")
parser.add_argument("--exclude_invalid_queue", action="store_true",
                    help="exclude queue entries whose labels are invalid")

# ---------------- Ablation 模式 ----------------
parser.add_argument(
    "--ablation_mode",
    default="contrastive_proto_rel",
    choices=[
        "contrastive_only",
        "contrastive_proto",
        "contrastive_rel",
        "contrastive_proto_rel",
    ],
    help=(
        "controls which auxiliary prototype losses are enabled: "
        "contrastive_only = only the main contrastive loss; "
        "contrastive_proto = main contrastive + prototype contrastive; "
        "contrastive_rel = main contrastive + prototype directional loss; "
        "contrastive_proto_rel = main contrastive + prototype contrastive + prototype directional loss"
    ),
)

# ---------------- Prototype 刷新参数 ----------------
parser.add_argument("--warmup_epochs", default=50, type=int,
                    help="number of epochs trained before the first prototype refresh")
parser.add_argument("--recluster_interval", default=5, type=int,
                    help="refresh prototypes every N epochs after warmup")
parser.add_argument(
    "--num_prototypes_per_class",
    default=None,
    type=str,
    help="comma-separated prototype counts for each class id, e.g. '2,2,3,1'; if None, all classes use --default_num_prototypes"
)
parser.add_argument(
    "--default_num_prototypes",
    default=1,
    type=int,
    help="default prototype count used for every class when --num_prototypes_per_class is None"
)
parser.add_argument("--lambda_proto", default=1.0, type=float,
                    help="weight of sample-to-prototype contrastive loss")
parser.add_argument("--proto_temperature", default=0.07, type=float,
                    help="global temperature used in prototype contrastive loss")
parser.add_argument("--enable_prototype_temperature_scaling", action="store_true",
                    help="enable prototype-specific relative temperature scaling")
parser.add_argument("--proto_temperature_eps", default=1e-6, type=float,
                    help="small epsilon used when estimating prototype temperature scaling")

# KMeans 超参数
parser.add_argument("--proto_kmeans_random_state", default=42, type=int,
                    help="random_state for per-class KMeans")
parser.add_argument("--proto_kmeans_n_init", default=10, type=int,
                    help="n_init for per-class KMeans")
parser.add_argument("--proto_kmeans_max_iter", default=300, type=int,
                    help="max_iter for per-class KMeans")

# refresh 专用 DataLoader 参数
parser.add_argument("--proto_refresh_batch_size", default=16, type=int,
                    help="batch size used during prototype refresh")
parser.add_argument("--proto_refresh_num_workers", default=6, type=int,
                    help="num_workers used during prototype refresh")
parser.add_argument("--proto_refresh_prefetch_factor", default=None, type=int,
                    help="prefetch_factor used during prototype refresh")
parser.add_argument("--proto_refresh_pin_memory", action=argparse.BooleanOptionalAction, default=None,
                    help="pin_memory used during prototype refresh; default: inherit training pin_memory")
parser.add_argument("--proto_refresh_verify_paths_on_init", action=argparse.BooleanOptionalAction, default=None,
                    help="verify_paths_on_init used during prototype refresh; default: inherit training verify_paths_on_init")

# ---------------- Prototype directional loss 参数 ----------------
parser.add_argument("--lambda_rel", default=0.5, type=float,
                    help="weight of prototype directional loss")
parser.add_argument("--proto_ema_momentum", default=0.99, type=float,
                    help="EMA momentum used to update the real prototype bank after optimizer.step()")
parser.add_argument("--preview_ema_momentum", default=0.5, type=float,
                    help="EMA momentum used to build the differentiable preview bank")
parser.add_argument("--rel_same_margin", default=0.0, type=float,
                    help="margin for the penalty on same-class prototypes becoming farther apart")
parser.add_argument("--rel_diff_margin", default=0.0, type=float,
                    help="margin for the penalty on different-class prototypes becoming closer")
parser.add_argument("--rel_same_weight", default=1.0, type=float,
                    help="weight of the same-class directional term")
parser.add_argument("--rel_diff_weight", default=1.0, type=float,
                    help="weight of the different-class directional term")
parser.add_argument("--rel_topk_diff_classes", default=0, type=int,
                    help=(
                        "when > 0, prototype directional loss only constrains the nearest K "
                        "different classes for each updated prototype; 0 means all different classes"
                    ))

# ---------------- 分阶段损失调度参数 ----------------
parser.add_argument("--enable_loss_stage_schedule", action="store_true",
                    help=(
                        "enable staged training: contrastive-only before proto_loss_start_epoch, "
                        "contrastive+proto before rel_loss_start_epoch, then add rel loss with "
                        "a scheduled lambda_rel"
                    ))
parser.add_argument("--proto_loss_start_epoch", default=50, type=int,
                    help="0-based epoch index at which prototype state/loss becomes active when staged scheduling is enabled")
parser.add_argument("--rel_loss_start_epoch", default=150, type=int,
                    help="0-based epoch index at which relative loss becomes active when staged scheduling is enabled")
parser.add_argument("--rel_loss_end_epoch", default=250, type=int,
                    help="0-based epoch index at which relative loss schedule reaches its final value; after this epoch lambda_rel stays at max")
parser.add_argument("--rel_lambda_schedule", default="cosine", choices=["constant", "cosine"],
                    help="lambda_rel schedule used during the relative-loss stage when staged scheduling is enabled")

# ---------------- 训练超参数 ----------------
parser.add_argument("--epochs", default=200, type=int,
                    help="number of training epochs")
parser.add_argument("--start_epoch", default=0, type=int,
                    help="starting epoch index")
parser.add_argument("--learning_rate", default=0.05, type=float,
                    help="initial learning rate")
parser.add_argument("--cos", action="store_true",
                    help="use cosine learning rate schedule")
parser.add_argument("--schedule", default=[50, 100, 150], nargs="*", type=int,
                    help="milestones for step LR schedule")
parser.add_argument("--weight_decay", default=1e-4, type=float,
                    help="weight decay. For AdamW this is decoupled weight decay.")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="SGD momentum; ignored when --optimizer adamw")
parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"], type=str,
                    help="optimizer to use. 'sgd' keeps the original behavior; 'adamw' uses torch.optim.AdamW")
parser.add_argument("--adamw_betas", nargs=2, type=float, default=[0.9, 0.999],
                    metavar=("BETA1", "BETA2"),
                    help="AdamW beta coefficients; used only when --optimizer adamw")
parser.add_argument("--adamw_eps", default=1e-8, type=float,
                    help="AdamW epsilon; used only when --optimizer adamw")
parser.add_argument("--adamw_amsgrad", action=argparse.BooleanOptionalAction, default=False,
                    help="whether to enable AMSGrad in AdamW; used only when --optimizer adamw")

# ---------------- 运行控制 ----------------
parser.add_argument("--no_ddp", action="store_true",
                    help="disable DDP and run in single-process mode")
parser.add_argument('--seed', type=int, default=None,
                    help='random seed; set to an integer for reproducibility, or leave unset for non-deterministic training')
parser.add_argument("--print_freq", default=20, type=int,
                    help="log logs every N iterations")
parser.add_argument("--save_interval", default=10, type=int,
                    help="save a checkpoint every N epochs")
parser.add_argument("--use_syncbn", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to convert BatchNorm to SyncBatchNorm when running DDP")
parser.add_argument("--find_unused_parameters", action=argparse.BooleanOptionalAction, default=False,
                    help="argument passed into DistributedDataParallel")

# ---------------- Debug 开关 ----------------
parser.add_argument("--debug_mode", action="store_true",
                    help="enable debug logging during training")
parser.add_argument("--debug_log_interval", default=20, type=int,
                    help="log debug information every N iterations")
parser.add_argument("--debug_grad_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="log gradient statistics when debug_mode is enabled")
parser.add_argument("--debug_param_update_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="log parameter update magnitude when debug_mode is enabled")
parser.add_argument("--debug_batch_label_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="log batch label distribution and positive-pair related stats")
parser.add_argument("--debug_proto_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="log prototype assignment / bank related stats")
parser.add_argument("--debug_feature_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="log q feature statistics")
parser.add_argument("--debug_nonfinite_check", action=argparse.BooleanOptionalAction, default=True,
                    help="check NaN / Inf in loss / q / prototype tensors")
parser.add_argument("--debug_abort_on_nonfinite", action=argparse.BooleanOptionalAction, default=False,
                    help="raise error immediately if NaN / Inf is detected")
parser.add_argument("--debug_grad_topk", default=8, type=int,
                    help="show top-k largest gradient norms")
parser.add_argument("--debug_param_patterns",
                    default="module.encoder_q.fc,encoder_q.fc,module.encoder_q.layer4,encoder_q.layer4,module.encoder_q.conv1,encoder_q.conv1",
                    type=str,
                    help="comma-separated substrings used to select parameters for update tracking")
parser.add_argument("--debug_param_fallback_last_n", default=4, type=int,
                    help="if no parameter matches debug_param_patterns, track the last N trainable parameters")
parser.add_argument("--debug_write_jsonl", action=argparse.BooleanOptionalAction, default=False,
                    help="optionally write debug payloads into a JSONL file")
parser.add_argument("--debug_jsonl_name", default="debug_train_log.jsonl", type=str,
                    help="JSONL debug log filename under weight_save_path")



def _parse_python_literal_arg(name: str, raw: str):
    """
    将命令行传入的 Python 字面量字符串解析为真实对象。

    典型用法
    --------
    1) --mindrove_emg_drift_max 0.1
       -> 0.1
    2) --mindrove_emg_drift_max "(0.05, 0.15)"
       -> (0.05, 0.15)
    3) --mindrove_emg_drift_n_points 3
       -> 3
    4) --mindrove_emg_drift_n_points "[2, 3, 4]"
       -> [2, 3, 4]

    说明
    ----
    这里不做兜底。若字符串不是合法 Python literal，则直接报错。
    后续 dataloader / augmentation 模块还会继续做严格类型与取值检查。
    """
    try:
        return ast.literal_eval(raw)
    except Exception as e:
        raise ValueError(f"Failed to parse argument {name}={raw!r} with ast.literal_eval") from e


def _parse_mindrove_aug_cli_args(args) -> dict:
    """
    将命令行中的 MindRove 增强参数整理成 dict，便于统一传入 PackedMultiModalConfig。

    这样做的好处：
    1) prepare_trainloader(...) 里构造 cfg 时更清晰
    2) 新增强项只需要在这里集中维护一次
    3) 训练脚本与 dataloader 的配置字段一一对应，避免漏传
    """
    return {
        # ---------- time warp ----------
        "mindrove_time_warp_prob": args.mindrove_time_warp_prob,
        "mindrove_time_warp_sigma": args.mindrove_time_warp_sigma,
        "mindrove_time_warp_num_knots": args.mindrove_time_warp_num_knots,
        "mindrove_time_warp_num_splines": args.mindrove_time_warp_num_splines,

        # ---------- EMG scaling / noise ----------
        "mindrove_emg_scaling_prob": args.mindrove_emg_scaling_prob,
        "mindrove_emg_scaling_sigma": args.mindrove_emg_scaling_sigma,
        "mindrove_emg_noise_prob": args.mindrove_emg_noise_prob,
        "mindrove_emg_noise_sigma": args.mindrove_emg_noise_sigma,

        # ---------- EMG drift ----------
        "mindrove_emg_drift_prob": args.mindrove_emg_drift_prob,
        "mindrove_emg_drift_max": _parse_python_literal_arg(
            "mindrove_emg_drift_max", args.mindrove_emg_drift_max
        ),
        "mindrove_emg_drift_n_points": _parse_python_literal_arg(
            "mindrove_emg_drift_n_points", args.mindrove_emg_drift_n_points
        ),
        "mindrove_emg_drift_kind": args.mindrove_emg_drift_kind,
        "mindrove_emg_drift_per_channel": args.mindrove_emg_drift_per_channel,
        "mindrove_emg_drift_normalize": args.mindrove_emg_drift_normalize,

        # ---------- IMU scaling / noise ----------
        "mindrove_imu_scaling_prob": args.mindrove_imu_scaling_prob,
        "mindrove_imu_scaling_sigma": args.mindrove_imu_scaling_sigma,
        "mindrove_imu_noise_prob": args.mindrove_imu_noise_prob,
        "mindrove_imu_noise_sigma": args.mindrove_imu_noise_sigma,

        # ---------- IMU drift ----------
        "mindrove_imu_drift_prob": args.mindrove_imu_drift_prob,
        "mindrove_imu_drift_max": _parse_python_literal_arg(
            "mindrove_imu_drift_max", args.mindrove_imu_drift_max
        ),
        "mindrove_imu_drift_n_points": _parse_python_literal_arg(
            "mindrove_imu_drift_n_points", args.mindrove_imu_drift_n_points
        ),
        "mindrove_imu_drift_kind": args.mindrove_imu_drift_kind,
        "mindrove_imu_drift_per_channel": args.mindrove_imu_drift_per_channel,
        "mindrove_imu_drift_normalize": args.mindrove_imu_drift_normalize,

        # ---------- negate ----------
        "mindrove_emg_negate_prob": args.mindrove_emg_negate_prob,
        "mindrove_imu_negate_prob": args.mindrove_imu_negate_prob,

        # ---------- channel dropout ----------
        "mindrove_emg_channel_dropout_prob": args.mindrove_emg_channel_dropout_prob,
        "mindrove_emg_channel_dropout_max_channels": args.mindrove_emg_channel_dropout_max_channels,
        "mindrove_imu_channel_dropout_prob": args.mindrove_imu_channel_dropout_prob,
        "mindrove_imu_channel_dropout_max_channels": args.mindrove_imu_channel_dropout_max_channels,
    }



def set_random_seed(seed: int | None, deterministic: bool = True):
    """
    设置随机种子。

    参数
    ----
    seed : int | None
        - int: 固定随机种子
        - None: 不固定随机种子
    deterministic : bool
        仅当 seed 不是 None 时生效：
        True  -> 使用确定性模式
        False -> 使用非确定性模式
    """
    if seed is None:
        log("[Info] No random seed is set. Training will be non-deterministic.")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return

    log(f"[Info] Setting random seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# ============================================================
# 基础工具
# ============================================================
GLOBAL_LOG_PATH: Optional[str] = None
GLOBAL_LOG_TO_FILE: bool = False



def log(msg: str) -> None:
    """
    统一日志函数：
    1) 始终打印到控制台
    2) 若已启用文件日志，则同时追加写入 train_log.txt
    """
    global GLOBAL_LOG_PATH, GLOBAL_LOG_TO_FILE

    print(msg, flush=True)

    if GLOBAL_LOG_TO_FILE and GLOBAL_LOG_PATH is not None:
        os.makedirs(os.path.dirname(GLOBAL_LOG_PATH), exist_ok=True)
        with open(GLOBAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")



def save_args(args, save_dir: str) -> None:
    """将命令行参数保存为 JSON，便于复现实验。"""
    os.makedirs(save_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["_timestamp"] = datetime.now().isoformat()
    tmp = os.path.join(save_dir, "args.json.tmp")
    final = os.path.join(save_dir, "args.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)
    os.replace(tmp, final)



def init_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """初始化 DDP，并返回 (rank, world_size, local_rank)。"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank



def init_single_process() -> Tuple[int, int, int]:
    """单进程模式初始化。"""
    rank, world_size, local_rank = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank



def is_main_process(rank: int) -> bool:
    return rank == 0



def _cleanup_ddp() -> None:
    """清理 DDP 进程组。"""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass



def _install_signal_handlers() -> None:
    """注册信号处理函数，便于作业中断时尽量优雅退出。"""
    def _handler(signum, frame):
        _cleanup_ddp()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handler)



def adjust_learning_rate(optimizer, epoch: int, args) -> None:
    """
    调整学习率。

    支持两种模式：
    1) cosine schedule
    2) step schedule
    """
    lr = args.learning_rate
    if args.cos:
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_optimizer(model: nn.Module, args) -> optim.Optimizer:
    """
    Build the optimizer used by the training loop.

    The default is the original SGD setting.  AdamW can be enabled with
    --optimizer adamw.  This deliberately uses torch.optim.AdamW, not Adam,
    so weight decay is decoupled from the gradient update.
    """
    opt_name = str(args.optimizer).strip().lower()

    if opt_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=float(args.learning_rate),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )

    if opt_name == "adamw":
        betas_raw = getattr(args, "adamw_betas", None)
        if betas_raw is None or len(betas_raw) != 2:
            raise ValueError(f"--adamw_betas must contain exactly 2 values, got {betas_raw}")

        beta1, beta2 = (float(betas_raw[0]), float(betas_raw[1]))
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"AdamW beta1 must satisfy 0 <= beta1 < 1, got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"AdamW beta2 must satisfy 0 <= beta2 < 1, got {beta2}")
        if float(args.adamw_eps) <= 0.0:
            raise ValueError(f"--adamw_eps must be > 0, got {args.adamw_eps}")

        return optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            betas=(beta1, beta2),
            eps=float(args.adamw_eps),
            weight_decay=float(args.weight_decay),
            amsgrad=bool(args.adamw_amsgrad),
        )

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def format_optimizer_config(args) -> str:
    """Return a compact one-line optimizer config for logging."""
    opt_name = str(args.optimizer).strip().lower()
    if opt_name == "sgd":
        return (
            f"[Optimizer] SGD | lr={float(args.learning_rate):.8g} | "
            f"momentum={float(args.momentum):.8g} | "
            f"weight_decay={float(args.weight_decay):.8g}"
        )
    if opt_name == "adamw":
        beta1, beta2 = (float(args.adamw_betas[0]), float(args.adamw_betas[1]))
        return (
            f"[Optimizer] AdamW | lr={float(args.learning_rate):.8g} | "
            f"betas=({beta1:.8g}, {beta2:.8g}) | "
            f"eps={float(args.adamw_eps):.8g} | "
            f"weight_decay={float(args.weight_decay):.8g} | "
            f"amsgrad={bool(args.adamw_amsgrad)}"
        )
    return f"[Optimizer] {args.optimizer}"



def _resolve_label_map_path(args) -> str:
    """解析 label_map.json 的绝对路径。"""
    path = Path(args.label_map_json)
    if path.is_absolute():
        return str(path)
    return str(Path(args.dataset_root) / path)



def parse_num_prototypes_per_class(
    spec: Optional[str],
    num_classes: int,
    default_num: int,
) -> List[int]:
    """
    将命令行传入的 per-class prototype 配置解析成长度为 num_classes 的 list[int]。

    例子：
        spec = "2,2,3,1"
        -> [2, 2, 3, 1]

    若 spec is None，则所有类别都使用 default_num。
    """
    if spec is None:
        return [int(default_num)] * int(num_classes)

    vals = [int(x.strip()) for x in spec.split(",") if x.strip() != ""]
    if len(vals) != num_classes:
        raise ValueError(
            f"--num_prototypes_per_class expects {num_classes} integers, "
            f"but got {len(vals)}: {vals}"
        )
    if any(v <= 0 for v in vals):
        raise ValueError(f"All prototype counts must be > 0, got {vals}")
    return vals



def resolve_ablation_flags(ablation_mode: str) -> Dict[str, bool]:
    """
    将高层 ablation_mode 解析为底层控制开关。

    返回三个布尔量：
    1) use_proto_state:
       是否需要整套 prototype 基础设施，包括：
       - prototype refresh
       - proto_state broadcast
       - batch 内构造 proto_ids
       - step 后的 prototype EMA update

    2) use_proto_loss:
       是否计算并加入 prototype contrastive loss

    3) use_rel_loss:
       是否计算并加入 prototype directional / relative loss

    说明：
    - contrastive_only:
        只训练主对比损失，不启用任何 prototype 相关状态和损失
    - contrastive_proto:
        主对比损失 + prototype contrastive loss
    - contrastive_rel:
        主对比损失 + prototype directional loss
        注意：虽然不使用 loss_proto，但仍然需要 prototype bank，
        因为 directional loss 本身依赖 prototype state
    - contrastive_proto_rel:
        主对比损失 + prototype contrastive loss + prototype directional loss
    """
    mapping = {
        "contrastive_only": {
            "use_proto_state": False,
            "use_proto_loss": False,
            "use_rel_loss": False,
        },
        "contrastive_proto": {
            "use_proto_state": True,
            "use_proto_loss": True,
            "use_rel_loss": False,
        },
        "contrastive_rel": {
            "use_proto_state": True,
            "use_proto_loss": False,
            "use_rel_loss": True,
        },
        "contrastive_proto_rel": {
            "use_proto_state": True,
            "use_proto_loss": True,
            "use_rel_loss": True,
        },
    }

    if ablation_mode not in mapping:
        raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")

    return mapping[ablation_mode]




def _cosine_ramp_value(
    start_epoch: int,
    end_epoch: int,
    current_epoch: int,
    final_value: float,
) -> float:
    """
    Cosine ramp from 0 to final_value over [start_epoch, end_epoch).

    Epochs before start_epoch return 0. Epochs at/after end_epoch return final_value.
    The training loop uses 0-based epoch indices.
    """
    if current_epoch < start_epoch:
        return 0.0
    if end_epoch <= start_epoch:
        return float(final_value)
    if current_epoch >= end_epoch:
        return float(final_value)

    progress = float(current_epoch - start_epoch) / float(max(1, end_epoch - start_epoch))
    progress = min(1.0, max(0.0, progress))
    return float(final_value) * 0.5 * (1.0 - math.cos(math.pi * progress))


def resolve_epoch_loss_flags(
    args,
    epoch: int,
    base_use_proto_state: bool,
    base_use_proto_loss: bool,
    base_use_rel_loss: bool,
) -> Dict[str, Any]:
    """
    Resolve effective loss switches and weights for the current epoch.

    Without --enable_loss_stage_schedule, this preserves the original behavior:
    ablation_mode controls active losses for all epochs, and lambda values are constant.

    With --enable_loss_stage_schedule:
      - epoch < proto_loss_start_epoch:
            contrastive / SupLoss only
      - proto_loss_start_epoch <= epoch < rel_loss_start_epoch:
            contrastive / SupLoss + prototype contrastive loss
      - epoch >= rel_loss_start_epoch:
            contrastive / SupLoss + prototype contrastive loss + relative loss
            lambda_rel is constant or cosine-ramped to args.lambda_rel by rel_loss_end_epoch,
            then kept at args.lambda_rel until training ends.

    ablation_mode still acts as an upper-level permission. For example,
    ablation_mode=contrastive_proto will not activate rel loss even in the rel stage.
    """
    if not args.enable_loss_stage_schedule:
        return {
            "use_proto_state": bool(base_use_proto_state),
            "use_proto_loss": bool(base_use_proto_loss),
            "use_rel_loss": bool(base_use_rel_loss),
            "lambda_proto": float(args.lambda_proto),
            "lambda_rel": float(args.lambda_rel),
            "refresh_anchor_epoch": int(args.warmup_epochs),
        }

    in_proto_stage = epoch >= int(args.proto_loss_start_epoch)
    in_rel_stage = epoch >= int(args.rel_loss_start_epoch)

    use_proto_state_eff = bool(base_use_proto_state and in_proto_stage)
    use_proto_loss_eff = bool(base_use_proto_loss and in_proto_stage)
    use_rel_loss_eff = bool(base_use_rel_loss and in_proto_stage and in_rel_stage)

    if use_rel_loss_eff:
        if args.rel_lambda_schedule == "cosine":
            lambda_rel_eff = _cosine_ramp_value(
                start_epoch=int(args.rel_loss_start_epoch),
                end_epoch=int(args.rel_loss_end_epoch),
                current_epoch=epoch,
                final_value=float(args.lambda_rel),
            )
        else:
            lambda_rel_eff = float(args.lambda_rel)
    else:
        lambda_rel_eff = 0.0

    return {
        "use_proto_state": use_proto_state_eff,
        "use_proto_loss": use_proto_loss_eff,
        "use_rel_loss": use_rel_loss_eff,
        "lambda_proto": float(args.lambda_proto) if use_proto_loss_eff else 0.0,
        "lambda_rel": lambda_rel_eff,
        "refresh_anchor_epoch": int(args.proto_loss_start_epoch),
    }


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


def infer_mindrove_in_channels(args) -> int:
    """
    根据当前 hand / signal / merge_hands 配置，推断最终输入到 ResNet1D 的通道数。
    """
    signal_channels = {"emg": 8, "imu": 6}
    hands = [str(x).lower() for x in args.mindrove_hands]
    signals = [str(x).lower() for x in args.mindrove_signals]

    if args.mindrove_merge_hands:
        # dataloader 已把左右手同模态合并:
        # emg -> 16, imu -> 12
        total = 0
        for s in signals:
            total += signal_channels[s] * 2
        return total

    total = 0
    for _h in hands:
        for s in signals:
            total += signal_channels[s]
    return total



def prepare_model(args) -> nn.Module:
    """
    构建 MoCo 模型，backbone 为新的 torchvision-style ResNet1D。

    注意：
    - 这里构造的是 base_encoder 的 partial
    - MoCo 内部会调用 base_encoder(num_classes=proj_dim)
    - 若 mlp=True，MoCo 会进一步把 backbone.fc 替换为 MLP projection head
    """
    in_channels = infer_mindrove_in_channels(args)

    base_encoder = partial(
        build_resnet1d,
        arch=args.ts_arch,
        in_channels=in_channels,
        base_channels=args.ts_base_channels,
        stem_kernel_size=args.ts_stem_kernel_size,
        stem_stride=args.ts_stem_stride,
        use_stem_pool=args.ts_use_stem_pool,
        zero_init_residual=args.ts_zero_init_residual,
    )

    model = MoCo1D(
        base_encoder=base_encoder,
        dim=args.proj_dim,
        K=args.K_queue,
        mlp=args.mlp,
        T=args.temperature,
        enable_kcl_loss=(args.contrastive_loss == "kcl"),
        num_positive=args.num_positive,
        exclude_invalid_queue=args.exclude_invalid_queue,
    )
    return model


def prepare_trainloader(args, use_ddp: bool, rank: int, world_size: int):
    """
    构建训练阶段使用的 MindRove map-style DataLoader。

    约定：
    1) 这里训练的是 MindRove，而不是 RGB / Depth
    2) mindrove_two_views=True
       因为 MoCo 需要 query / key 两个视角
    3) two-view 不是从长序列裁两个片段，
       而是对同一个重采样后序列独立增强两次
    4) 最新的 drift / negate / channel dropout 等增强也都在这里通过 cfg 传入，
       因此 MoCo 训练脚本无需改动主训练循环，只需保证配置完整下传
    """
    label_map_path = _resolve_label_map_path(args)
    label_map = load_label_map_json(label_map_path)

    # 新增的 drift / negate / channel dropout 等增强，
    # 都通过这个统一的 dict 下传给 PackedMultiModalConfig。
    # 训练脚本本身不直接操作样本，而是完全复用 dataloader + augmentation 模块的实现。
    mindrove_aug_cfg = _parse_mindrove_aug_cli_args(args)

    cfg = PackedMultiModalConfig(
        rgb_two_views=False,
        use_modalities=("mindrove",),
        missing_policy="skip",
        load_labels=True,
        tier_mode=args.tier_mode,
        is_train=True,
        label_map_path=label_map_path,

        mindrove_two_views=True,
        mindrove_target_len=args.mindrove_target_len,
        mindrove_emg_target_len=args.mindrove_emg_target_len,
        mindrove_imu_target_len=args.mindrove_imu_target_len,
        mindrove_hands=tuple(args.mindrove_hands),
        mindrove_signals=tuple(args.mindrove_signals),
        mindrove_merge_hands=args.mindrove_merge_hands,
        mindrove_apply_augmentation=args.mindrove_apply_augmentation,
        mindrove_apply_normalization=args.mindrove_apply_normalization,
        mindrove_left_emg_mean=(
            _parse_python_literal_arg("mindrove_left_emg_mean", args.mindrove_left_emg_mean)
            if args.mindrove_left_emg_mean is not None else None
        ),
        mindrove_left_emg_std=(
            _parse_python_literal_arg("mindrove_left_emg_std", args.mindrove_left_emg_std)
            if args.mindrove_left_emg_std is not None else None
        ),
        mindrove_right_emg_mean=(
            _parse_python_literal_arg("mindrove_right_emg_mean", args.mindrove_right_emg_mean)
            if args.mindrove_right_emg_mean is not None else None
        ),
        mindrove_right_emg_std=(
            _parse_python_literal_arg("mindrove_right_emg_std", args.mindrove_right_emg_std)
            if args.mindrove_right_emg_std is not None else None
        ),
        mindrove_left_imu_mean=(
            _parse_python_literal_arg("mindrove_left_imu_mean", args.mindrove_left_imu_mean)
            if args.mindrove_left_imu_mean is not None else None
        ),
        mindrove_left_imu_std=(
            _parse_python_literal_arg("mindrove_left_imu_std", args.mindrove_left_imu_std)
            if args.mindrove_left_imu_std is not None else None
        ),
        mindrove_right_imu_mean=(
            _parse_python_literal_arg("mindrove_right_imu_mean", args.mindrove_right_imu_mean)
            if args.mindrove_right_imu_mean is not None else None
        ),
        mindrove_right_imu_std=(
            _parse_python_literal_arg("mindrove_right_imu_std", args.mindrove_right_imu_std)
            if args.mindrove_right_imu_std is not None else None
        ),

        **mindrove_aug_cfg,
    )

    dataset = build_packed_mapstyle_dataset(
        dataset_root=args.dataset_root,
        manifest_name=args.train_manifest_name,
        cfg=cfg,
        label_map=label_map,
        verify_paths_on_init=args.verify_paths_on_init,
    )

    sampler = None
    shuffle = True
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        shuffle = False

    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        drop_last=True,
        sampler=sampler,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    return loader, sampler


def extract_two_views_and_labels(
    batch: dict,
    tier_mode: str,
    expected_keys: List[str],
    args=None,
):
    """
    从训练 batch 中提取：
        - 两个 MindRove 视图
        - labels
        - global indices

    预期输入：
        batch["mindrove"] -> (view1, view2)

    其中：
        - view1 / view2 可以已经是 Tensor[B,C,L]
        - 也可以是 dict[str, Tensor[B,C,L]]

    本函数会调用 pack_mindrove_batched_view()，
    将最终输入统一整理成 Conv1d 可直接使用的 Tensor[B,C,L]。
    """
    mr = batch["mindrove"]
    if not (isinstance(mr, (tuple, list)) and len(mr) == 2):
        raise RuntimeError(
            "Training loader is expected to output two MindRove views as (view1, view2), "
            f"but got type={type(mr)}"
        )

    view1 = pack_mindrove_batched_view(mr[0], expected_keys, args=args)
    view2 = pack_mindrove_batched_view(mr[1], expected_keys, args=args)

    tier_ids = batch["tier_ids"]
    labels = tier_ids[tier_mode] if isinstance(tier_ids, dict) else tier_ids

    if "global_index" in batch:
        global_index = batch["global_index"]
    elif "idx" in batch:
        global_index = batch["idx"]
    elif "sample_id" in batch:
        global_index = batch["sample_id"]
    else:
        raise KeyError("Batch does not contain global_index / idx / sample_id")

    return view1, view2, labels, global_index


def build_proto_refresh_config(
    args,
    device: torch.device,
    num_classes: int,
) -> PrototypeRefreshConfig:
    """
    根据命令行参数构建 PrototypeRefreshConfig。

    说明
    ----
    refresh 阶段需要与训练阶段使用相同的 MindRove normalization，
    否则 prototype 构建时看到的输入分布会和训练时不一致。
    """
    refresh_batch_size = args.proto_refresh_batch_size
    if refresh_batch_size is None:
        refresh_batch_size = args.batch_size

    refresh_num_workers = args.proto_refresh_num_workers
    if refresh_num_workers is None:
        refresh_num_workers = args.num_workers

    refresh_prefetch_factor = args.proto_refresh_prefetch_factor
    if refresh_prefetch_factor is None:
        refresh_prefetch_factor = args.prefetch_factor

    refresh_pin_memory = args.proto_refresh_pin_memory
    if refresh_pin_memory is None:
        refresh_pin_memory = args.pin_memory

    refresh_verify_paths = args.proto_refresh_verify_paths_on_init
    if refresh_verify_paths is None:
        refresh_verify_paths = args.verify_paths_on_init

    proto_counts = parse_num_prototypes_per_class(
        spec=args.num_prototypes_per_class,
        num_classes=num_classes,
        default_num=args.default_num_prototypes,
    )

    return PrototypeRefreshConfig(
        tier_mode=args.tier_mode,
        num_prototypes_per_class=proto_counts,
        default_num_prototypes=args.default_num_prototypes,
        random_state=(args.seed if args.seed is not None else args.proto_kmeans_random_state),
        n_init=args.proto_kmeans_n_init,
        max_iter=args.proto_kmeans_max_iter,
        batch_size=refresh_batch_size,
        num_workers=refresh_num_workers,
        pin_memory=refresh_pin_memory,
        prefetch_factor=refresh_prefetch_factor,
        verify_paths_on_init=refresh_verify_paths,
        device=device,
        require_main_process_only=True,
        enable_prototype_temperature_scaling=args.enable_prototype_temperature_scaling,
        proto_base_temperature=args.proto_temperature,
        proto_temperature_eps=args.proto_temperature_eps,

        # ---------------- 与训练阶段保持一致的 MindRove variable-length / packing ----------------
        mindrove_emg_target_len=args.mindrove_emg_target_len,
        mindrove_imu_target_len=args.mindrove_imu_target_len,
        mindrove_pack_length_policy=args.mindrove_pack_length_policy,
        mindrove_pack_target_len=args.mindrove_pack_target_len,

        # ---------------- 与训练阶段保持一致的 MindRove normalization ----------------
        mindrove_apply_normalization=args.mindrove_apply_normalization,

        mindrove_left_emg_mean=(
            _parse_python_literal_arg("mindrove_left_emg_mean", args.mindrove_left_emg_mean)
            if args.mindrove_left_emg_mean is not None else None
        ),
        mindrove_left_emg_std=(
            _parse_python_literal_arg("mindrove_left_emg_std", args.mindrove_left_emg_std)
            if args.mindrove_left_emg_std is not None else None
        ),
        mindrove_right_emg_mean=(
            _parse_python_literal_arg("mindrove_right_emg_mean", args.mindrove_right_emg_mean)
            if args.mindrove_right_emg_mean is not None else None
        ),
        mindrove_right_emg_std=(
            _parse_python_literal_arg("mindrove_right_emg_std", args.mindrove_right_emg_std)
            if args.mindrove_right_emg_std is not None else None
        ),
        mindrove_left_imu_mean=(
            _parse_python_literal_arg("mindrove_left_imu_mean", args.mindrove_left_imu_mean)
            if args.mindrove_left_imu_mean is not None else None
        ),
        mindrove_left_imu_std=(
            _parse_python_literal_arg("mindrove_left_imu_std", args.mindrove_left_imu_std)
            if args.mindrove_left_imu_std is not None else None
        ),
        mindrove_right_imu_mean=(
            _parse_python_literal_arg("mindrove_right_imu_mean", args.mindrove_right_imu_mean)
            if args.mindrove_right_imu_mean is not None else None
        ),
        mindrove_right_imu_std=(
            _parse_python_literal_arg("mindrove_right_imu_std", args.mindrove_right_imu_std)
            if args.mindrove_right_imu_std is not None else None
        ),
    )


# ============================================================
# Debug 配置与工具
# ============================================================

@dataclass
class DebugConfig:
    enabled: bool
    log_interval: int
    grad_stats: bool
    param_update_stats: bool
    batch_label_stats: bool
    proto_stats: bool
    feature_stats: bool
    nonfinite_check: bool
    abort_on_nonfinite: bool
    grad_topk: int
    param_patterns: List[str]
    param_fallback_last_n: int
    write_jsonl: bool
    jsonl_path: Optional[str]


def build_debug_config(args) -> DebugConfig:
    patterns = [x.strip() for x in args.debug_param_patterns.split(",") if x.strip()]
    jsonl_path = None
    if args.debug_write_jsonl:
        jsonl_path = os.path.join(args.weight_save_path, args.debug_jsonl_name)

    return DebugConfig(
        enabled=args.debug_mode,
        log_interval=max(1, int(args.debug_log_interval)),
        grad_stats=args.debug_grad_stats,
        param_update_stats=args.debug_param_update_stats,
        batch_label_stats=args.debug_batch_label_stats,
        proto_stats=args.debug_proto_stats,
        feature_stats=args.debug_feature_stats,
        nonfinite_check=args.debug_nonfinite_check,
        abort_on_nonfinite=args.debug_abort_on_nonfinite,
        grad_topk=max(1, int(args.debug_grad_topk)),
        param_patterns=patterns,
        param_fallback_last_n=max(1, int(args.debug_param_fallback_last_n)),
        write_jsonl=args.debug_write_jsonl,
        jsonl_path=jsonl_path,
    )


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    """将调试信息按 JSONL 形式追加写入，方便后续离线分析。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _get_current_lrs(optimizer) -> List[float]:
    """返回所有 param_group 当前学习率。"""
    return [float(pg["lr"]) for pg in optimizer.param_groups]


def _unwrap_model(model: nn.Module) -> nn.Module:
    """去掉 DDP 外壳，便于读取模块名和参数名。"""
    return model.module if hasattr(model, "module") else model


def _resolve_tracked_param_names(
    model: nn.Module,
    patterns: List[str],
    fallback_last_n: int,
) -> List[str]:
    """
    根据名字子串筛选想重点观察的参数。

    典型用途：
    - 看分类头是否在更新
    - 看 backbone 最后几层是否在更新
    """
    all_trainable = [name for name, p in model.named_parameters() if p.requires_grad]

    selected: List[str] = []
    seen = set()

    for pat in patterns:
        for name in all_trainable:
            if pat in name and name not in seen:
                selected.append(name)
                seen.add(name)

    if len(selected) == 0:
        selected = all_trainable[-fallback_last_n:]

    return selected


def _snapshot_selected_params(model: nn.Module, selected_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    保存若干关键参数的当前值，用于 optimizer.step() 后计算更新幅度。
    为降低显存占用，这里保存到 CPU。
    """
    name_set = set(selected_names)
    snap: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name in name_set:
            snap[name] = p.detach().float().cpu().clone()
    return snap


def _compute_param_update_stats(model: nn.Module, before_snapshot: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
    """
    计算被跟踪参数在一次 step 前后的更新幅度：
    - absolute_update_norm
    - relative_update_norm = ||delta|| / (||param_before|| + eps)

    这样可以直接判断：
    - 参数到底有没有更新
    - 更新是否小到几乎可以忽略
    """
    out = []
    if not before_snapshot:
        return out

    for name, p in model.named_parameters():
        if name not in before_snapshot:
            continue
        before = before_snapshot[name]
        after = p.detach().float().cpu()
        delta = after - before
        delta_norm = float(delta.norm().item())
        before_norm = float(before.norm().item())
        rel = delta_norm / (before_norm + 1e-12)
        out.append({
            "name": name,
            "absolute_update_norm": delta_norm,
            "relative_update_norm": rel,
            "param_norm_before": before_norm,
        })
    return out


def _compute_grad_stats(
    model: nn.Module,
    topk: int,
    tracked_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    统计梯度信息。

    输出包括：
    - total_grad_norm
    - none_grad 参数个数
    - near_zero_grad 参数个数
    - 最大梯度范数的 top-k 参数
    - 被重点跟踪参数的梯度范数
    """
    total_sq = 0.0
    none_grad_count = 0
    near_zero_grad_count = 0
    grad_entries: List[Tuple[str, float]] = []
    tracked = []

    tracked_set = set(tracked_names or [])

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            none_grad_count += 1
            if name in tracked_set:
                tracked.append({"name": name, "grad_norm": None})
            continue

        gnorm = float(p.grad.detach().data.norm(2).item())
        total_sq += gnorm * gnorm
        if gnorm < 1e-12:
            near_zero_grad_count += 1
        grad_entries.append((name, gnorm))

        if name in tracked_set:
            tracked.append({"name": name, "grad_norm": gnorm})

    grad_entries.sort(key=lambda x: x[1], reverse=True)
    total_grad_norm = math.sqrt(total_sq)

    return {
        "total_grad_norm": total_grad_norm,
        "none_grad_count": none_grad_count,
        "near_zero_grad_count": near_zero_grad_count,
        "topk_grad_norms": grad_entries[:topk],
        "tracked_grad_norms": tracked,
    }


def _compute_batch_label_stats(labels: torch.Tensor) -> Dict[str, Any]:
    """
    统计当前 batch 的标签信息。

    输出包括：
    1) batch 中有哪些类别
    2) 每个类别各有多少样本
    3) 仍然保留一些对 SupCon / KCL 有帮助的统计量
    """
    labels_cpu = labels.detach().cpu()
    uniq, counts = torch.unique(labels_cpu, return_counts=True)

    batch_size = int(labels_cpu.numel())
    counts_float = counts.float()

    # 若某类在 batch 中出现 n 次，
    # 则该类中的每个样本都有 n-1 个“同类其他样本”
    avg_same_class_others = float(
        (counts_float * (counts_float - 1)).sum().item() / max(1, batch_size)
    )
    anchors_without_same_class = int(counts[counts == 1].sum().item())
    num_unique_classes = int(uniq.numel())

    # 明确记录：batch 中有哪些类别
    present_labels = [int(x.item()) for x in uniq]

    # 明确记录：每个类别有多少样本
    label_count_pairs = [
        {"label": int(u.item()), "count": int(c.item())}
        for u, c in zip(uniq, counts)
    ]

    # 也保留一个按数量降序排列的版本，便于快速查看
    label_count_pairs_sorted = sorted(
        label_count_pairs,
        key=lambda x: x["count"],
        reverse=True
    )

    return {
        "batch_size": batch_size,
        "num_unique_classes": num_unique_classes,
        "present_labels": present_labels,
        "label_count_pairs": label_count_pairs,
        "label_count_pairs_sorted": label_count_pairs_sorted,
        "avg_same_class_others_per_anchor": avg_same_class_others,
        "anchors_without_same_class_in_batch": anchors_without_same_class,
    }


def _compute_feature_stats(q: torch.Tensor) -> Dict[str, Any]:
    """
    统计 query 特征 q 的基本分布。

    可用于观察：
    - 是否数值范围异常
    - 是否可能出现特征塌缩
    """
    q_det = q.detach().float()
    row_norm = q_det.norm(dim=1)

    return {
        "q_shape": list(q_det.shape),
        "q_mean": float(q_det.mean().item()),
        "q_std": float(q_det.std().item()),
        "q_min": float(q_det.min().item()),
        "q_max": float(q_det.max().item()),
        "q_row_l2_mean": float(row_norm.mean().item()),
        "q_row_l2_std": float(row_norm.std().item()),
        "q_row_l2_min": float(row_norm.min().item()),
        "q_row_l2_max": float(row_norm.max().item()),
        "q_feature_std_mean": float(q_det.std(dim=0).mean().item()),
    }


def _count_nonfinite(x: torch.Tensor) -> Dict[str, int]:
    """统计一个张量中 NaN / +Inf / -Inf 的数量。"""
    x_det = x.detach()
    return {
        "num_nan": int(torch.isnan(x_det).sum().item()),
        "num_posinf": int(torch.isposinf(x_det).sum().item()),
        "num_neginf": int(torch.isneginf(x_det).sum().item()),
    }


def _check_nonfinite_payload(
    loss: torch.Tensor,
    loss_supcon: torch.Tensor,
    loss_proto: torch.Tensor,
    loss_rel: torch.Tensor,
    q: torch.Tensor,
    proto_state: Optional[dict],
) -> Dict[str, Any]:
    """
    检查关键张量里是否有 NaN / Inf。
    """
    payload = {
        "loss": _count_nonfinite(loss),
        "loss_supcon": _count_nonfinite(loss_supcon),
        "loss_proto": _count_nonfinite(loss_proto),
        "loss_rel": _count_nonfinite(loss_rel),
        "q": _count_nonfinite(q),
    }

    if proto_state is not None and "prototype_bank" in proto_state and proto_state["prototype_bank"] is not None:
        payload["prototype_bank"] = _count_nonfinite(proto_state["prototype_bank"])

    has_bad = False
    for stats in payload.values():
        if any(v > 0 for v in stats.values()):
            has_bad = True
            break

    payload["has_nonfinite"] = has_bad
    return payload


def _compute_proto_batch_stats(
    global_index: torch.Tensor,
    proto_ids: Optional[torch.Tensor],
    sample_to_proto: Optional[torch.Tensor],
    valid_sample_mask_bank: Optional[torch.Tensor],
    proto_state: Optional[dict],
) -> Dict[str, Any]:
    """
    统计当前 batch 的 prototype 分配情况。
    """
    out: Dict[str, Any] = {}

    global_index = global_index.detach()
    out["batch_size"] = int(global_index.numel())

    if sample_to_proto is not None:
        in_range_mask = (global_index >= 0) & (global_index < sample_to_proto.numel())
        out["global_index_in_range_ratio"] = float(in_range_mask.float().mean().item())
        out["global_index_out_of_range_count"] = int((~in_range_mask).sum().item())
    else:
        out["global_index_in_range_ratio"] = None
        out["global_index_out_of_range_count"] = None

    if proto_ids is None:
        out["proto_ids_available"] = False
        return out

    proto_ids_det = proto_ids.detach()
    valid_assign_mask = proto_ids_det >= 0

    out["proto_ids_available"] = True
    out["valid_proto_assign_ratio"] = float(valid_assign_mask.float().mean().item())
    out["invalid_proto_assign_count"] = int((~valid_assign_mask).sum().item())
    out["num_unique_proto_ids_in_batch"] = int(torch.unique(proto_ids_det[valid_assign_mask]).numel()) if valid_assign_mask.any() else 0

    if valid_sample_mask_bank is not None and sample_to_proto is not None:
        in_range_mask = (global_index >= 0) & (global_index < sample_to_proto.numel())
        if in_range_mask.any():
            selected_valid = valid_sample_mask_bank[global_index[in_range_mask]]
            out["selected_valid_sample_ratio_among_in_range"] = float(selected_valid.float().mean().item())
        else:
            out["selected_valid_sample_ratio_among_in_range"] = None

    if proto_state is not None:
        class_num_prototypes = proto_state.get("class_num_prototypes", None)
        proto_rel_temperature_bank = proto_state.get("proto_rel_temperature_bank", None)
        valid_sample_mask = proto_state.get("valid_sample_mask", None)

        if class_num_prototypes is not None:
            out["class_num_prototypes"] = class_num_prototypes.detach().cpu().tolist()
            out["total_active_prototypes"] = int(class_num_prototypes.sum().item())

        if proto_rel_temperature_bank is not None:
            t = proto_rel_temperature_bank.detach().float()
            out["proto_rel_temperature_mean"] = float(t.mean().item())
            out["proto_rel_temperature_std"] = float(t.std().item())
            out["proto_rel_temperature_min"] = float(t.min().item())
            out["proto_rel_temperature_max"] = float(t.max().item())

        if valid_sample_mask is not None:
            out["valid_sample_mask_ratio_global"] = float(valid_sample_mask.detach().float().mean().item())

    return out


def _summarize_proto_refresh_debug(proto_state: dict) -> str:
    """
    在 prototype refresh 完成后输出更细一点的统计信息。
    """
    parts = []

    class_num_prototypes = proto_state.get("class_num_prototypes", None)
    if class_num_prototypes is not None:
        vals = class_num_prototypes.detach().cpu().tolist()
        parts.append(f"class_num_prototypes={vals}")
        parts.append(f"total_active_prototypes={int(class_num_prototypes.sum().item())}")

    valid_sample_mask = proto_state.get("valid_sample_mask", None)
    if valid_sample_mask is not None:
        ratio = float(valid_sample_mask.detach().float().mean().item())
        parts.append(f"valid_sample_ratio={ratio:.4f}")

    sample_to_proto = proto_state.get("sample_to_proto", None)
    if sample_to_proto is not None:
        stp = sample_to_proto.detach()
        valid_assign = (stp >= 0)
        ratio = float(valid_assign.float().mean().item())
        parts.append(f"sample_to_proto_valid_ratio={ratio:.4f}")
        if valid_assign.any():
            parts.append(f"num_unique_proto_ids={int(torch.unique(stp[valid_assign]).numel())}")

    proto_rel_temperature_bank = proto_state.get("proto_rel_temperature_bank", None)
    if proto_rel_temperature_bank is not None:
        t = proto_rel_temperature_bank.detach().float()
        parts.append(
            f"proto_rel_temp(mean={t.mean().item():.4f}, std={t.std().item():.4f}, "
            f"min={t.min().item():.4f}, max={t.max().item():.4f})"
        )

    return " | ".join(parts)


# ============================================================
# 统计器与 checkpoint
# ============================================================

class AverageMeter:
    """维护一个标量的当前值、累计和以及平均值。"""
    def __init__(self, name: str, fmt: str = ":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.average = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n: int = 1) -> None:
        value = float(value)
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / max(1, self.count)


def save_checkpoint(state: dict, filename: str) -> None:
    """保存 checkpoint。"""
    torch.save(state, filename)


# ============================================================
# 单个 epoch 的训练
# ============================================================

def train_one_epoch(
    trainloader,
    model,
    optimizer,
    scaler,
    device,
    rank: int,
    epoch: int,
    total_epochs: int,
    print_freq: int,
    proto_state: Optional[dict],
    use_proto_state: bool,
    use_proto_loss: bool,
    use_rel_loss: bool,
    lambda_proto: float,
    lambda_rel: float,
    proto_temperature: float,
    proto_ema_momentum: float,
    tier_mode: str,
    preview_ema_momentum: float,
    rel_same_margin: float,
    rel_diff_margin: float,
    rel_same_weight: float,
    rel_diff_weight: float,
    rel_topk_diff_classes: int,
    proto_temperature_eps: float,
    amp_dtype: torch.dtype,
    use_amp: bool,
    debug_cfg: DebugConfig,
    contrastive_loss_mode: str,
    sup_criterion: Optional[nn.Module],
    expected_mindrove_keys: List[str],
    args=None,
) -> None:
    """
    训练时包含一个主对比损失，以及最多两个 prototype 辅助损失：
    ----------------------------------------------------------
    1) 主对比损失：
    - kcl，或
    - suploss
    由 contrastive_loss_mode 控制

    2) prototype contrastive loss
    由 use_proto_loss 控制

    3) prototype directional / relative loss
    由 use_rel_loss 控制

    另外，prototype 相关状态（refresh / proto_ids / EMA update）
    由 use_proto_state 控制。

    注意：
    - contrastive_only 模式下，use_proto_state=False，此时整个 prototype 分支都关闭
    - contrastive_rel 模式下，虽然不使用 loss_proto，但仍然需要 prototype state，
    因为 relative loss 依赖 prototype bank 和 proto_ids

    这一版额外加入了一套 debug 能力，用来诊断：
    - 梯度是否存在
    - 参数是否真的更新
    - 当前 batch 的监督对比信号强不强
    - prototype 分支是否有效
    - 是否出现 NaN / Inf
    """
    losses = AverageMeter("loss")
    losses_supcon = AverageMeter("loss_supcon")
    losses_proto = AverageMeter("loss_proto")
    losses_rel = AverageMeter("loss_rel")

    log(f"use_amp: {use_amp}")
    model.train()

    tracked_param_names: List[str] = []
    if debug_cfg.enabled and (debug_cfg.grad_stats or debug_cfg.param_update_stats):
        tracked_param_names = _resolve_tracked_param_names(
            model=model,
            patterns=debug_cfg.param_patterns,
            fallback_last_n=debug_cfg.param_fallback_last_n,
        )
        if is_main_process(rank):
            log(f"[Debug] tracking params: {tracked_param_names}")

    for i, batch in enumerate(trainloader):
        step_idx = i + 1
        debug_this_iter = debug_cfg.enabled and (
            step_idx == 1 or (step_idx % debug_cfg.log_interval == 0)
        )

        view1, view2, labels, global_index = extract_two_views_and_labels(
            batch=batch,
            tier_mode=tier_mode,
            expected_keys=expected_mindrove_keys,
            args=args,
        )

        view1 = view1.to(device, non_blocking=True).float()
        view2 = view2.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        global_index = global_index.to(device, non_blocking=True).long()

        # 当前 1D MindRove 训练路径约定：
        # view1 / view2 在进入模型前已经是 [B, C, L]
        if view1.ndim != 3 or view2.ndim != 3:
            raise ValueError(
                f"MindRove views must be [B,C,L], got view1={tuple(view1.shape)}, view2={tuple(view2.shape)}"
            )

        view1 = view1.contiguous()
        view2 = view2.contiguous()

        optimizer.zero_grad(set_to_none=True)

        # 如果要观察参数是否真的更新，这里先对被跟踪参数做快照
        param_snapshot_before = None
        if debug_this_iter and debug_cfg.param_update_stats:
            param_snapshot_before = _snapshot_selected_params(model, tracked_param_names)

        with autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), dtype=amp_dtype, enabled=use_amp):
            features, target, loss_kcl, q, _ = model(im_q=view1, im_k=view2, labels=labels)

            if contrastive_loss_mode == "kcl":
                if loss_kcl is None:
                    raise RuntimeError("Model returned loss_kcl=None while contrastive_loss_mode='kcl'")
                loss_supcon = loss_kcl
            elif contrastive_loss_mode == "suploss":
                if sup_criterion is None:
                    raise RuntimeError("sup_criterion is None while contrastive_loss_mode='suploss'")
                loss_supcon = sup_criterion(features, target)
            else:
                raise ValueError(f"Unsupported contrastive_loss_mode: {contrastive_loss_mode}")
            
            # ------------------------------------------------------------
            # 根据 ablation 配置决定是否启用 prototype 相关分支
            #
            # use_proto_state:
            #   是否需要 prototype bank / proto_ids / EMA update 这一整套状态
            #
            # use_proto_loss:
            #   是否计算 prototype contrastive loss
            #
            # use_rel_loss:
            #   是否计算 prototype directional / relative loss
            #
            # 注意：
            #   contrastive_rel 模式下 use_proto_loss=False，但 use_proto_state=True，
            #   因为 relative loss 仍然依赖 prototype bank
            # ------------------------------------------------------------
            if (not use_proto_state) or (proto_state is None):
                loss_proto = torch.zeros((), device=device, dtype=q.dtype)
                loss_rel = torch.zeros((), device=device, dtype=q.dtype)
                proto_ids = None
                loss = loss_supcon
            else:
                sample_to_proto = proto_state["sample_to_proto"]
                prototype_bank = proto_state["prototype_bank"]
                class_num_prototypes = proto_state["class_num_prototypes"]
                proto_rel_temperature_bank = proto_state["proto_rel_temperature_bank"]
                valid_sample_mask_bank = proto_state.get("valid_sample_mask", None)

                # ------------------------------------------------------------
                # 安全构造 proto_ids：
                # 1) 若 global_index 超出 sample_to_proto 范围，则设为 -1
                # 2) 若 valid_sample_mask 显示该样本当前无效，则设为 -1
                # ------------------------------------------------------------
                proto_ids = torch.full_like(global_index, fill_value=-1)
                in_range_mask = (global_index >= 0) & (global_index < sample_to_proto.numel())

                if in_range_mask.any():
                    selected_index = global_index[in_range_mask]
                    selected_proto = sample_to_proto[selected_index]

                    if valid_sample_mask_bank is not None:
                        selected_valid = valid_sample_mask_bank[selected_index]
                        selected_proto = torch.where(
                            selected_valid,
                            selected_proto,
                            torch.full_like(selected_proto, -1),
                        )

                    proto_ids[in_range_mask] = selected_proto

                # 默认先置零，再按 ablation 开关决定是否真正计算
                loss_proto = torch.zeros((), device=device, dtype=q.dtype)
                loss_rel = torch.zeros((), device=device, dtype=q.dtype)

                # ------------------------------------------------------------
                # 1) prototype contrastive loss
                # ------------------------------------------------------------
                if use_proto_loss:
                    loss_proto = prototype_contrastive_loss_all_positive(
                        q=q,
                        labels=labels,
                        proto_ids=proto_ids,
                        prototype_bank=prototype_bank,
                        class_num_prototypes=class_num_prototypes,
                        temperature=proto_temperature,
                        use_prototype_temperature_scaling=proto_state.get(
                            "enable_prototype_temperature_scaling", False
                        ),
                        proto_rel_temperature_bank=proto_rel_temperature_bank,
                        temperature_eps=proto_temperature_eps,
                    )

                # ------------------------------------------------------------
                # 2) prototype directional / relative loss
                # ------------------------------------------------------------
                if use_rel_loss:
                    loss_rel, _, _, _ = differentiable_ema_directional_loss(
                        old_prototype_bank=prototype_bank,
                        q=q,
                        labels=labels,
                        proto_ids=proto_ids,
                        preview_ema_momentum=preview_ema_momentum,
                        same_margin=rel_same_margin,
                        diff_margin=rel_diff_margin,
                        same_weight=rel_same_weight,
                        diff_weight=rel_diff_weight,
                        topk_diff_classes=(rel_topk_diff_classes if rel_topk_diff_classes > 0 else None),
                        class_num_prototypes=class_num_prototypes,
                    )

                # ------------------------------------------------------------
                # 最终总损失：
                # 主损失始终存在；
                # proto / rel 是否加入，由 ablation 开关决定
                # ------------------------------------------------------------
                loss = loss_supcon
                if use_proto_loss:
                    loss = loss + lambda_proto * loss_proto
                if use_rel_loss:
                    loss = loss + lambda_rel * loss_rel

        # -----------------------------
        # 非有限值检查：在 backward 前做
        # -----------------------------
        nonfinite_payload = None
        if debug_cfg.enabled and debug_cfg.nonfinite_check:
            nonfinite_payload = _check_nonfinite_payload(
                loss=loss,
                loss_supcon=loss_supcon,
                loss_proto=loss_proto,
                loss_rel=loss_rel,
                q=q,
                proto_state=proto_state,
            )
            if nonfinite_payload["has_nonfinite"] and is_main_process(rank):
                log(f"[Debug][NonFinite] epoch={epoch+1} iter={step_idx} payload={nonfinite_payload}")
            if nonfinite_payload["has_nonfinite"] and debug_cfg.abort_on_nonfinite:
                raise FloatingPointError(f"NaN / Inf detected at epoch={epoch+1}, iter={step_idx}")

        scaler.scale(loss).backward()

        # AMP 下若要看真实梯度范数，先 unscale 再统计
        if use_amp:
            scaler.unscale_(optimizer)

        grad_payload = None
        if debug_this_iter and debug_cfg.grad_stats:
            grad_payload = _compute_grad_stats(
                model=model,
                topk=debug_cfg.grad_topk,
                tracked_names=tracked_param_names,
            )

        scaler.step(optimizer)
        scaler.update()

        param_update_payload = None
        if debug_this_iter and debug_cfg.param_update_stats:
            param_update_payload = _compute_param_update_stats(
                model=model,
                before_snapshot=param_snapshot_before or {},
            )

        # ------------------------------------------------------------
        # 只有在启用 prototype state 时，才进行真实 prototype bank 的 EMA 更新
        # contrastive_only 模式下，这一步必须彻底关闭
        # contrastive_rel 模式下，这一步仍然需要保留，因为 rel loss 依赖动态 prototype bank
        # ------------------------------------------------------------
        if use_proto_state and (proto_state is not None) and (proto_ids is not None):
            ema_update_prototype_bank_(
                prototype_bank=proto_state["prototype_bank"],
                q=q.detach(),
                labels=labels.detach(),
                proto_ids=proto_ids.detach(),
                bank_ema_momentum=proto_ema_momentum,
                class_num_prototypes=proto_state["class_num_prototypes"],
            )

        bs = view1.size(0)
        losses.update(loss.item(), n=bs)
        losses_supcon.update(loss_supcon.item(), n=bs)
        losses_proto.update(loss_proto.item(), n=bs)
        losses_rel.update(loss_rel.item(), n=bs)

        # -----------------------------
        # 常规训练日志
        # -----------------------------
        if is_main_process(rank) and ((step_idx % print_freq) == 0):
            msg = (
                f"[Epoch {epoch + 1:03d}/{total_epochs}] [Iter {step_idx}] "
                f"loss={loss.item():.4f} "
                f"supcon_loss={loss_supcon.item():.4f} "
                f"proto_loss={loss_proto.item():.4f} "
                f"rel_loss={loss_rel.item():.4f} "
                f"(avg_loss={losses.average:.4f}) "
                f"(avg_supcon={losses_supcon.average:.4f}) "
                f"(avg_proto={losses_proto.average:.4f}) "
                f"(avg_rel={losses_rel.average:.4f})"
            )
            log(msg)

        # -----------------------------
        # Debug 日志
        # -----------------------------
        if debug_this_iter and is_main_process(rank):
            debug_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "epoch": int(epoch + 1),
                "iter": int(step_idx),
                "lr_list": _get_current_lrs(optimizer),
                "loss": float(loss.detach().item()),
                "loss_supcon": float(loss_supcon.detach().item()),
                "loss_proto": float(loss_proto.detach().item()),
                "loss_rel": float(loss_rel.detach().item()),
                "weighted_proto_contrib": float(lambda_proto * loss_proto.detach().item()),
                "weighted_rel_contrib": float(lambda_rel * loss_rel.detach().item()),
                "running_avg_loss": float(losses.average),
                "running_avg_supcon": float(losses_supcon.average),
                "running_avg_proto": float(losses_proto.average),
                "running_avg_rel": float(losses_rel.average),
            }

            # 1) batch 标签统计
            if debug_cfg.batch_label_stats:
                debug_payload["batch_label_stats"] = _compute_batch_label_stats(labels)

            # 2) q 特征统计
            if debug_cfg.feature_stats:
                debug_payload["feature_stats"] = _compute_feature_stats(q)

            # 3) prototype 分配统计
            if debug_cfg.proto_stats:
                sample_to_proto = proto_state["sample_to_proto"] if proto_state is not None else None
                valid_sample_mask_bank = proto_state.get("valid_sample_mask", None) if proto_state is not None else None
                debug_payload["proto_batch_stats"] = _compute_proto_batch_stats(
                    global_index=global_index,
                    proto_ids=proto_ids,
                    sample_to_proto=sample_to_proto,
                    valid_sample_mask_bank=valid_sample_mask_bank,
                    proto_state=proto_state,
                )

            # 4) 梯度统计
            if grad_payload is not None:
                debug_payload["grad_stats"] = grad_payload

            # 5) 参数更新统计
            if param_update_payload is not None:
                debug_payload["param_update_stats"] = param_update_payload

            # 6) 非有限值统计
            if nonfinite_payload is not None:
                debug_payload["nonfinite_check"] = nonfinite_payload

            # 简明打印版
            log(f"[Debug][Epoch {epoch + 1:03d} Iter {step_idx}] lr={debug_payload['lr_list']}")
            log(
                f"[Debug][LossScale] total={debug_payload['loss']:.4f}, "
                f"loss_supcon={debug_payload['loss_supcon']:.4f}, "
                f"lambda_proto*proto={debug_payload['weighted_proto_contrib']:.4f}, "
                f"lambda_rel*rel={debug_payload['weighted_rel_contrib']:.4f}"
            )

            if debug_cfg.batch_label_stats:
                bstats = debug_payload["batch_label_stats"]
                log(
                    f"[Debug][BatchLabel] batch_size={bstats['batch_size']} "
                    f"unique_classes={bstats['num_unique_classes']} "
                    f"present_labels={bstats['present_labels']} "
                    f"label_counts={bstats['label_count_pairs_sorted']} "
                    f"avg_same_class_others={bstats['avg_same_class_others_per_anchor']:.4f} "
                    f"anchors_without_same_class={bstats['anchors_without_same_class_in_batch']}"
                )

            if debug_cfg.feature_stats:
                fstats = debug_payload["feature_stats"]
                log(
                    f"[Debug][Feature] q_shape={fstats['q_shape']} "
                    f"q_mean={fstats['q_mean']:.6f} q_std={fstats['q_std']:.6f} "
                    f"q_row_l2_mean={fstats['q_row_l2_mean']:.6f} "
                    f"q_row_l2_std={fstats['q_row_l2_std']:.6f} "
                    f"q_feature_std_mean={fstats['q_feature_std_mean']:.6f}"
                )

            if debug_cfg.proto_stats:
                pstats = debug_payload["proto_batch_stats"]
                log(
                    f"[Debug][ProtoBatch] "
                    f"in_range_ratio={pstats.get('global_index_in_range_ratio', None)} "
                    f"valid_assign_ratio={pstats.get('valid_proto_assign_ratio', None)} "
                    f"invalid_assign_count={pstats.get('invalid_proto_assign_count', None)} "
                    f"unique_proto_ids_in_batch={pstats.get('num_unique_proto_ids_in_batch', None)}"
                )

            if grad_payload is not None:
                log(
                    f"[Debug][Grad] total_grad_norm={grad_payload['total_grad_norm']:.6f} "
                    f"none_grad_count={grad_payload['none_grad_count']} "
                    f"near_zero_grad_count={grad_payload['near_zero_grad_count']} "
                    f"topk={grad_payload['topk_grad_norms']}"
                )
                if len(grad_payload["tracked_grad_norms"]) > 0:
                    log(f"[Debug][TrackedGrad] {grad_payload['tracked_grad_norms']}")

            if param_update_payload is not None and len(param_update_payload) > 0:
                log(f"[Debug][ParamUpdate] {param_update_payload}")

            if nonfinite_payload is not None and nonfinite_payload["has_nonfinite"]:
                log(f"[Debug][NonFiniteSummary] {nonfinite_payload}")

            if debug_cfg.write_jsonl and debug_cfg.jsonl_path is not None:
                _append_jsonl(debug_cfg.jsonl_path, debug_payload)


# ============================================================
# 主 worker
# ============================================================

def worker(args) -> None:
    """
    主训练入口。

    整体流程：
    ----------
    1) 初始化单进程或 DDP
    2) 构建模型、优化器、训练 loader
    3) 解析类别数，并构造 per-class prototype 配置
    4) 按 epoch 决定是否刷新 prototypes
    5) 执行训练
    6) 按设定间隔保存 checkpoint
    """

    global GLOBAL_LOG_PATH, GLOBAL_LOG_TO_FILE

    if args.no_ddp:
        rank, world_size, local_rank = init_single_process()
    else:
        rank, world_size, local_rank = init_distributed("nccl")

    train_log_path = os.path.join(args.weight_save_path, "train_log.txt")

    if is_main_process(rank):
        os.makedirs(args.weight_save_path, exist_ok=True)
        save_args(args, args.weight_save_path)

        GLOBAL_LOG_PATH = train_log_path
        GLOBAL_LOG_TO_FILE = True
    else:
        GLOBAL_LOG_PATH = None
        GLOBAL_LOG_TO_FILE = False
        
    log(f"\n===== Training started at {datetime.now().isoformat()} =====")

    if args.no_ddp:
        log("training without DDP")
    else:
        log("training with DDP")

    if args.contrastive_loss == "kcl":
        log("using kcl loss")
    else:
        log("using supcon loss")

    set_random_seed(args.seed, deterministic=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _install_signal_handlers()

    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=(use_amp and (not use_bf16)))

    debug_cfg = build_debug_config(args)

    ablation_flags = resolve_ablation_flags(args.ablation_mode)
    use_proto_state = ablation_flags["use_proto_state"]
    use_proto_loss = ablation_flags["use_proto_loss"]
    use_rel_loss = ablation_flags["use_rel_loss"]


    if is_main_process(rank):
        log(
            f"[Ablation] mode={args.ablation_mode} | "
            f"contrastive_loss={args.contrastive_loss} | "
            f"use_proto_state={use_proto_state} | "
            f"use_proto_loss={use_proto_loss} | "
            f"use_rel_loss={use_rel_loss}"
        )

    try:
        if debug_cfg.enabled:
            log("[Debug] debug_mode is enabled.")
            log(f"[Debug] debug_log_interval={debug_cfg.log_interval}")
            log(f"[Debug] debug_jsonl_path={debug_cfg.jsonl_path}")


        label_map = load_label_map_json(_resolve_label_map_path(args))
        if args.tier_mode not in label_map:
            raise KeyError(f"tier_mode={args.tier_mode} not found in label_map.json")
        num_classes = len(label_map[args.tier_mode])
        if num_classes <= 0:
            raise ValueError(f"No classes found for tier={args.tier_mode}")

        proto_refresh_cfg = build_proto_refresh_config(args, device=device, num_classes=num_classes)

        expected_mindrove_keys = build_mindrove_expected_keys(args)
        parsed_mindrove_aug_cfg = _parse_mindrove_aug_cli_args(args)

        if is_main_process(rank):
            log(f"[MindRove][AugConfig] {parsed_mindrove_aug_cfg}")
            log(
                f"[MindRove][NormConfig] "
                f"apply_normalization={args.mindrove_apply_normalization}, "
                f"left_emg_mean_provided={args.mindrove_left_emg_mean is not None}, "
                f"left_emg_std_provided={args.mindrove_left_emg_std is not None}, "
                f"right_emg_mean_provided={args.mindrove_right_emg_mean is not None}, "
                f"right_emg_std_provided={args.mindrove_right_emg_std is not None}, "
                f"left_imu_mean_provided={args.mindrove_left_imu_mean is not None}, "
                f"left_imu_std_provided={args.mindrove_left_imu_std is not None}, "
                f"right_imu_mean_provided={args.mindrove_right_imu_mean is not None}, "
                f"right_imu_std_provided={args.mindrove_right_imu_std is not None}"
            )
            log(f"[MindRove] expected_keys = {expected_mindrove_keys}")
            log(
                f"[MindRove] target_len={args.mindrove_target_len}, "
                f"emg_target_len={args.mindrove_emg_target_len}, "
                f"imu_target_len={args.mindrove_imu_target_len}, "
                f"pack_policy={args.mindrove_pack_length_policy}, "
                f"pack_target_len={args.mindrove_pack_target_len}"
            )
            log(f"[MindRove] inferred_in_channels = {infer_mindrove_in_channels(args)}")
            log(f"[DirectionalLoss] rel_topk_diff_classes={args.rel_topk_diff_classes}")

        model = prepare_model(args).to(device)

        if (
            args.use_syncbn
            and (not args.no_ddp)
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not args.no_ddp:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=args.find_unused_parameters,
            )

        sup_criterion = SupLoss(
            temperature=args.temperature,
            base_temperature=args.temperature,
            K=args.K_queue,
        ).to(device)
        
        optimizer = build_optimizer(model, args)
        if is_main_process(rank):
            log(format_optimizer_config(args))

        trainloader, train_sampler = prepare_trainloader(
            args=args,
            use_ddp=(not args.no_ddp),
            rank=rank,
            world_size=world_size,
        )

        proto_state = None

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            epoch_msg = f"\nEPOCH {epoch + 1}/{args.epochs} -----------------------------"
            if is_main_process(rank):
                log(epoch_msg)

            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            # ------------------------------------------------------------
            # Resolve the effective loss switches for this epoch.
            # - Original mode: same as ablation flags and original warmup refresh logic.
            # - Staged mode: proto / rel losses are activated by epoch schedule, and
            #   prototype refresh starts when prototype state first becomes active.
            # ------------------------------------------------------------
            epoch_loss_cfg = resolve_epoch_loss_flags(
                args=args,
                epoch=epoch,
                base_use_proto_state=use_proto_state,
                base_use_proto_loss=use_proto_loss,
                base_use_rel_loss=use_rel_loss,
            )
            use_proto_state_epoch = bool(epoch_loss_cfg["use_proto_state"])
            use_proto_loss_epoch = bool(epoch_loss_cfg["use_proto_loss"])
            use_rel_loss_epoch = bool(epoch_loss_cfg["use_rel_loss"])
            lambda_proto_epoch = float(epoch_loss_cfg["lambda_proto"])
            lambda_rel_epoch = float(epoch_loss_cfg["lambda_rel"])
            refresh_anchor_epoch = int(epoch_loss_cfg["refresh_anchor_epoch"])

            if is_main_process(rank):
                log(
                    f"[LossSchedule] epoch={epoch} | "
                    f"use_proto_state={use_proto_state_epoch} | "
                    f"use_proto_loss={use_proto_loss_epoch} | "
                    f"use_rel_loss={use_rel_loss_epoch} | "
                    f"lambda_proto={lambda_proto_epoch:.6f} | "
                    f"lambda_rel={lambda_rel_epoch:.6f}"
                )

            need_refresh = False
            if args.enable_loss_stage_schedule:
                if use_proto_state_epoch:
                    if proto_state is None:
                        need_refresh = True
                    elif epoch > refresh_anchor_epoch and ((epoch - refresh_anchor_epoch) % args.recluster_interval == 0):
                        need_refresh = True
            else:
                if use_proto_state_epoch:
                    if epoch == args.warmup_epochs:
                        need_refresh = True
                    elif epoch > args.warmup_epochs and ((epoch - args.warmup_epochs) % args.recluster_interval == 0):
                        need_refresh = True

            if need_refresh:
                if is_main_process(rank):
                    model_for_refresh = _unwrap_model(model)
                    proto_state = refresh_prototypes(
                        model=model_for_refresh,
                        args=args,
                        cfg=proto_refresh_cfg,
                    )
                    log("[Prototype Refresh] done.")
                    log(summarize_proto_state(proto_state))
                    if debug_cfg.enabled and debug_cfg.proto_stats:
                        log(f"[Debug][ProtoRefresh] {_summarize_proto_refresh_debug(proto_state)}")

                proto_state = broadcast_proto_state(proto_state, device=device, rank=rank)
            elif not use_proto_state_epoch:
                # 明确保证当前 epoch 不启用 prototype state 时不保留任何 prototype 状态。
                # 在 staged schedule 的 contrastive-only 阶段尤其重要。
                proto_state = None

            train_one_epoch(
                trainloader=trainloader,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                rank=rank,
                epoch=epoch,
                total_epochs=args.epochs,
                print_freq=args.print_freq,
                proto_state=proto_state,
                use_proto_state=use_proto_state_epoch,
                use_proto_loss=use_proto_loss_epoch,
                use_rel_loss=use_rel_loss_epoch,
                lambda_proto=lambda_proto_epoch,
                lambda_rel=lambda_rel_epoch,
                proto_temperature=args.proto_temperature,
                proto_ema_momentum=args.proto_ema_momentum,
                tier_mode=args.tier_mode,
                preview_ema_momentum=args.preview_ema_momentum,
                rel_same_margin=args.rel_same_margin,
                rel_diff_margin=args.rel_diff_margin,
                rel_same_weight=args.rel_same_weight,
                rel_diff_weight=args.rel_diff_weight,
                rel_topk_diff_classes=args.rel_topk_diff_classes,
                proto_temperature_eps=args.proto_temperature_eps,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                debug_cfg=debug_cfg,
                contrastive_loss_mode=args.contrastive_loss,
                sup_criterion=sup_criterion,
                expected_mindrove_keys=expected_mindrove_keys,
                args=args,
            )

            if is_main_process(rank) and ((epoch + 1) % args.save_interval == 0):
                pt_path = os.path.join(args.weight_save_path, f"checkpoint_{epoch + 1:04d}.pth")
                state = {
                    "epoch": epoch + 1,
                    "state_dict": (_unwrap_model(model).state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "contrastive_loss_mode": args.contrastive_loss,
                    "ablation_mode": args.ablation_mode,
                    "prototype_bank": (proto_state["prototype_bank"].cpu() if proto_state is not None else None),
                    "class_num_prototypes": (
                        proto_state["class_num_prototypes"].cpu() if proto_state is not None else None
                    ),
                    "proto_rel_temperature_bank": (
                        proto_state["proto_rel_temperature_bank"].cpu() if proto_state is not None else None
                    ),
                    "sample_to_proto": (proto_state["sample_to_proto"].cpu() if proto_state is not None else None),
                    "sample_to_class": (proto_state["sample_to_class"].cpu() if proto_state is not None else None),
                    "valid_sample_mask": (
                        proto_state["valid_sample_mask"].cpu() if proto_state is not None else None
                    ),
                    "enable_prototype_temperature_scaling": (
                        proto_state.get("enable_prototype_temperature_scaling", False) if proto_state is not None else False
                    ),
                }
                save_checkpoint(state, pt_path)

    finally:
        _cleanup_ddp()


if __name__ == "__main__":
    args = parser.parse_args()
    worker(args)