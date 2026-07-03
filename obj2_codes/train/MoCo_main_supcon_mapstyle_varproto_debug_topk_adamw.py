#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import backbone.resnet as ResNet3D
from backbone.MoCo_VAR_supcon_wds import MoCo3D
from utils_.mapstype_dataloader_with_index import (
    PackedMultiModalConfig,
    load_label_map_json,
    build_packed_mapstyle_dataset,
    build_packed_mapstyle_loader_from_dataset,
)
from utils_.build_update_prototype_mapstyle_varproto import (
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
# å‘½ä»¤è¡Œå‚æ•°
# ============================================================

parser = argparse.ArgumentParser(
    description="3D MoCo + SupCon + prototype contrastive + prototype directional loss on map-style dataset (variable prototypes per class)"
)

# ---------------- æ•°æ®ä¸Ž I/O ----------------
parser.add_argument("--dataset_root", required=True, type=str,
                    help="map-style dataset root directory")
parser.add_argument("--train_manifest_name", default="train_manifest.jsonl", type=str,
                    help="training manifest file name under dataset_root")
parser.add_argument("--label_map_json", default="label_map.json", type=str,
                    help="label_map.json path; can be absolute or relative to dataset_root")
parser.add_argument("--weight_save_path", default=r"./weight", type=str,
                    help="checkpoint output directory")

# ---------------- æ ‡ç­¾ä¸Ž clip è®¾ç½® ----------------
parser.add_argument("--tier_mode", default="tier1", choices=["tier1", "tier2", "tier3"],
                    help="which tier id to use as the training label")
parser.add_argument("--n_frames", default=16, type=int,
                    help="number of frames per clip")
parser.add_argument("--rgb_camera_id", default="001484412812", type=str,
                    help="RGB camera id to use when manifest has camera-specific fields, e.g. 001484412812")

parser.add_argument(
    "--rgb_mean",
    nargs=3,
    type=float,
    default=[0.356, 0.386, 0.395],
    metavar=("R_MEAN", "G_MEAN", "B_MEAN"),
    help=(
        "RGB channel mean used for Normalize after ToDtype(scale=True). "
        "Values must correspond to RGB scaled into [0,1]."
    ),
)

parser.add_argument(
    "--rgb_std",
    nargs=3,
    type=float,
    default=[0.292, 0.271, 0.263],
    metavar=("R_STD", "G_STD", "B_STD"),
    help=(
        "RGB channel std used for Normalize after ToDtype(scale=True). "
        "Values must correspond to RGB scaled into [0,1]."
    ),
)

# ---------------- RGB augmentation params ----------------
# è¿™äº›å‚æ•°ä¼šä¼ ç»™ utils_.mapstype_dataloader_with_index.PackedMultiModalConfigã€‚
# dataloader å†…éƒ¨çš„ TemporallyConsistentSpatialAugmentation ä¼šå¯¹åŒä¸€ä¸ª clip çš„æ‰€æœ‰å¸§
# ä½¿ç”¨æ—¶é—´ä¸€è‡´çš„ç©ºé—´å¢žå¼ºã€‚è‹¥ --no-rgb_apply_spatial_augï¼Œåˆ™ä¼šå…³é—­ flip / jitter /
# grayscale / blur ç­‰éšæœºå¢žå¼ºï¼Œä½†ä»ä¼šä¿ç•™ RandomResizedCropã€‚
parser.add_argument("--rgb_out_hw", nargs=2, type=int, default=[224, 224],
                    metavar=("H", "W"),
                    help="RGB output spatial size after augmentation / validation transform")
parser.add_argument("--rrc_scale", nargs=2, type=float, default=[0.6, 1.0],
                    metavar=("MIN", "MAX"),
                    help="RandomResizedCrop scale range for RGB training augmentation")
parser.add_argument("--rrc_ratio", nargs=2, type=float, default=[0.75, 1.3333333333],
                    metavar=("MIN", "MAX"),
                    help="RandomResizedCrop aspect-ratio range for RGB training augmentation")
parser.add_argument("--rgb_apply_spatial_aug", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to enable RGB random spatial augmentation probabilities in training")
parser.add_argument("--rgb_hflip_p", default=0.5, type=float,
                    help="RGB horizontal flip probability")
parser.add_argument("--rgb_vflip_p", default=0.5, type=float,
                    help="RGB vertical flip probability")
parser.add_argument("--rgb_jitter_p", default=0.5, type=float,
                    help="RGB ColorJitter probability")
parser.add_argument("--rgb_jitter_brightness", default=0.24, type=float,
                    help="RGB ColorJitter brightness strength")
parser.add_argument("--rgb_jitter_contrast", default=0.24, type=float,
                    help="RGB ColorJitter contrast strength")
parser.add_argument("--rgb_jitter_saturation", default=0.24, type=float,
                    help="RGB ColorJitter saturation strength")
parser.add_argument("--rgb_jitter_hue", default=0.16, type=float,
                    help="RGB ColorJitter hue strength; should be in [0, 0.5]")
parser.add_argument("--rgb_gray_p", default=0.2, type=float,
                    help="RGB random grayscale probability")
parser.add_argument("--rgb_blur_p", default=0.5, type=float,
                    help="RGB GaussianBlur probability")
parser.add_argument("--rgb_blur_kernel", default=7, type=int,
                    help="RGB GaussianBlur kernel size; must be odd and >= 3")
parser.add_argument("--rgb_blur_sigma", nargs=2, type=float, default=[0.1, 1.0],
                    metavar=("MIN", "MAX"),
                    help="RGB GaussianBlur sigma range")

# ---------------- è®­ç»ƒ DataLoader ----------------
parser.add_argument("--batch_size", default=16, type=int,
                    help="training batch size per GPU/process")
parser.add_argument("--num_workers", default=6, type=int,
                    help="number of workers for the training DataLoader")
parser.add_argument("--prefetch_factor", default=None, type=int,
                    help="optional prefetch_factor for the training DataLoader")
parser.add_argument("--pin_memory", action="store_true",
                    help="enable pin_memory for the training DataLoader")
parser.add_argument("--verify_paths_on_init", action="store_true",
                    help="verify sample paths when constructing the training dataset")

# ---------------- æ¨¡åž‹å‚æ•° ----------------
parser.add_argument("--model_depth", default=18, type=int,
                    help="3D ResNet depth")
parser.add_argument("--proj_dim", default=128, type=int,
                    help="projection head output dimension")
parser.add_argument("--K_queue", default=3200, type=int,
                    help="MoCo queue size")
parser.add_argument("--temperature", default=0.07, type=float,
                    help="global temperature used in the KCL branch")
parser.add_argument("--mlp", action="store_true",
                    help="use MLP projection head")

# ---------------- å¯¹æ¯”æŸå¤± å‚æ•° ----------------
parser.add_argument("--contrastive_loss", default="suploss", choices=["kcl", "suploss"],
                    help="which supervised contrastive branch to use: 'kcl' or 'suploss'")
parser.add_argument("--num_positive", default=6, type=int,
                    help="number of same-class positives sampled from the queue; 0 means use all")
parser.add_argument("--exclude_invalid_queue", action="store_true",
                    help="exclude queue entries whose labels are invalid")

# ---------------- Ablation æ¨¡å¼ ----------------
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

# ---------------- Prototype åˆ·æ–°å‚æ•° ----------------
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

# KMeans è¶…å‚æ•°
parser.add_argument("--proto_kmeans_random_state", default=42, type=int,
                    help="random_state for per-class KMeans")
parser.add_argument("--proto_kmeans_n_init", default=10, type=int,
                    help="n_init for per-class KMeans")
parser.add_argument("--proto_kmeans_max_iter", default=300, type=int,
                    help="max_iter for per-class KMeans")

# refresh ä¸“ç”¨ DataLoader å‚æ•°
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

# ---------------- Prototype directional loss å‚æ•° ----------------
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
                        "when > 0, the relative/directional loss only constrains the nearest K "
                        "different classes for each updated prototype, measured by old prototype "
                        "cosine distance; 0 means use all different classes"
                    ))

# ---------------- åˆ†é˜¶æ®µæŸå¤±è°ƒåº¦å‚æ•° ----------------
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
                    help="0-based exclusive epoch index at which relative loss schedule reaches its final value")
parser.add_argument("--rel_lambda_schedule", default="cosine", choices=["constant", "cosine"],
                    help="lambda_rel schedule used during the relative-loss stage when staged scheduling is enabled")

# ---------------- è®­ç»ƒè¶…å‚æ•° ----------------
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

# ---------------- è¿è¡ŒæŽ§åˆ¶ ----------------
parser.add_argument("--no_ddp", action="store_true",
                    help="disable DDP and run in single-process mode")
parser.add_argument('--seed', type=int, default=None,
                    help='random seed; set to an integer for reproducibility, or leave unset for non-deterministic training')
parser.add_argument("--print_freq", default=20, type=int,
                    help="print logs every N iterations")
parser.add_argument("--save_interval", default=10, type=int,
                    help="save a checkpoint every N epochs")
parser.add_argument("--use_syncbn", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to convert BatchNorm to SyncBatchNorm when running DDP")
parser.add_argument("--find_unused_parameters", action=argparse.BooleanOptionalAction, default=False,
                    help="argument passed into DistributedDataParallel")

# ---------------- Debug å¼€å…³ ----------------
parser.add_argument("--debug_mode", action="store_true",
                    help="enable debug logging during training")
parser.add_argument("--debug_log_interval", default=20, type=int,
                    help="print debug information every N iterations")
parser.add_argument("--debug_grad_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="print gradient statistics when debug_mode is enabled")
parser.add_argument("--debug_param_update_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="print parameter update magnitude when debug_mode is enabled")
parser.add_argument("--debug_batch_label_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="print batch label distribution and positive-pair related stats")
parser.add_argument("--debug_proto_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="print prototype assignment / bank related stats")
parser.add_argument("--debug_feature_stats", action=argparse.BooleanOptionalAction, default=True,
                    help="print q feature statistics")
parser.add_argument("--debug_nonfinite_check", action=argparse.BooleanOptionalAction, default=True,
                    help="check NaN / Inf in loss / q / prototype tensors")
parser.add_argument("--debug_abort_on_nonfinite", action=argparse.BooleanOptionalAction, default=False,
                    help="raise error immediately if NaN / Inf is detected")
parser.add_argument("--debug_grad_topk", default=8, type=int,
                    help="show top-k largest gradient norms")
parser.add_argument("--debug_param_patterns",
                    default="module.encoder_q.fc,encoder_q.fc,module.encoder_q.layer4,encoder_q.layer4,module.encoder_q.layer3,encoder_q.layer3",
                    type=str,
                    help="comma-separated substrings used to select parameters for update tracking")
parser.add_argument("--debug_param_fallback_last_n", default=4, type=int,
                    help="if no parameter matches debug_param_patterns, track the last N trainable parameters")
parser.add_argument("--debug_write_jsonl", action=argparse.BooleanOptionalAction, default=False,
                    help="optionally write debug payloads into a JSONL file")
parser.add_argument("--debug_jsonl_name", default="debug_train_log.jsonl", type=str,
                    help="JSONL debug log filename under weight_save_path")

args = parser.parse_args()

def set_random_seed(seed: int | None, deterministic: bool = True):
    """
    è®¾ç½®éšæœºç§å­ã€‚

    å‚æ•°
    ----
    seed : int | None
        - int: å›ºå®šéšæœºç§å­
        - None: ä¸å›ºå®šéšæœºç§å­
    deterministic : bool
        ä»…å½“ seed ä¸æ˜¯ None æ—¶ç”Ÿæ•ˆï¼š
        True  -> ä½¿ç”¨ç¡®å®šæ€§æ¨¡å¼
        False -> ä½¿ç”¨éžç¡®å®šæ€§æ¨¡å¼
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
# åŸºç¡€å·¥å…·
# ============================================================

GLOBAL_LOG_PATH: Optional[str] = None
GLOBAL_LOG_TO_FILE: bool = False


def log(msg: str) -> None:
    """
    ç»Ÿä¸€æ—¥å¿—å‡½æ•°ï¼š
    1) å§‹ç»ˆæ‰“å°åˆ°æŽ§åˆ¶å°
    2) è‹¥å·²å¯ç”¨æ–‡ä»¶æ—¥å¿—ï¼Œåˆ™åŒæ—¶è¿½åŠ å†™å…¥ train_log.txt
    """
    global GLOBAL_LOG_PATH, GLOBAL_LOG_TO_FILE

    print(msg, flush=True)

    if GLOBAL_LOG_TO_FILE and GLOBAL_LOG_PATH is not None:
        os.makedirs(os.path.dirname(GLOBAL_LOG_PATH), exist_ok=True)
        with open(GLOBAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")



def save_args(args, save_dir: str) -> None:
    """å°†å‘½ä»¤è¡Œå‚æ•°ä¿å­˜ä¸º JSONï¼Œä¾¿äºŽå¤çŽ°å®žéªŒã€‚"""
    os.makedirs(save_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["_timestamp"] = datetime.now().isoformat()
    tmp = os.path.join(save_dir, "args.json.tmp")
    final = os.path.join(save_dir, "args.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)
    os.replace(tmp, final)



def init_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """åˆå§‹åŒ– DDPï¼Œå¹¶è¿”å›ž (rank, world_size, local_rank)ã€‚"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank



def init_single_process() -> Tuple[int, int, int]:
    """å•è¿›ç¨‹æ¨¡å¼åˆå§‹åŒ–ã€‚"""
    rank, world_size, local_rank = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank



def is_main_process(rank: int) -> bool:
    return rank == 0



def _cleanup_ddp() -> None:
    """æ¸…ç† DDP è¿›ç¨‹ç»„ã€‚"""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass



def _install_signal_handlers() -> None:
    """æ³¨å†Œä¿¡å·å¤„ç†å‡½æ•°ï¼Œä¾¿äºŽä½œä¸šä¸­æ–­æ—¶å°½é‡ä¼˜é›…é€€å‡ºã€‚"""
    def _handler(signum, frame):
        _cleanup_ddp()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handler)



def adjust_learning_rate(optimizer, epoch: int, args) -> None:
    """
    è°ƒæ•´å­¦ä¹ çŽ‡ã€‚

    æ”¯æŒä¸¤ç§æ¨¡å¼ï¼š
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
    """è§£æž label_map.json çš„ç»å¯¹è·¯å¾„ã€‚"""
    path = Path(args.label_map_json)
    if path.is_absolute():
        return str(path)
    return str(Path(args.dataset_root) / path)



def _resolve_rgb_mean_std(args) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    è§£æž RGB normalization å‚æ•°ã€‚

    æ³¨æ„ï¼š
    - spatial_augmentation.py é‡Œä¼šå…ˆæ‰§è¡Œ ToDtype(torch.float32, scale=True)
    - å› æ­¤è¿™é‡Œçš„ mean/std å¿…é¡»å¯¹åº” [0,1] èŒƒå›´ï¼Œè€Œä¸æ˜¯ [0,255] èŒƒå›´
    """
    rgb_mean = tuple(float(x) for x in args.rgb_mean)
    rgb_std = tuple(float(x) for x in args.rgb_std)

    if len(rgb_mean) != 3:
        raise ValueError(f"--rgb_mean must contain exactly 3 values, got {rgb_mean}")

    if len(rgb_std) != 3:
        raise ValueError(f"--rgb_std must contain exactly 3 values, got {rgb_std}")

    if any(s <= 0 for s in rgb_std):
        raise ValueError(f"All values in --rgb_std must be > 0, got {rgb_std}")

    return rgb_mean, rgb_std



def _resolve_pair_arg(args, name: str, cast_type=float) -> Tuple[Any, Any]:
    """
    è§£æž argparse ä¸­ä½¿ç”¨ nargs=2 çš„ RGB å‚æ•°ï¼Œå¹¶è½¬ä¸º tupleã€‚

    ä¾‹å¦‚ï¼š
      --rgb_out_hw 224 224 -> (224, 224)
      --rrc_scale 0.6 1.0 -> (0.6, 1.0)
    """
    raw = getattr(args, name)
    if raw is None or len(raw) != 2:
        raise ValueError(f"--{name} must contain exactly 2 values, got {raw}")
    return (cast_type(raw[0]), cast_type(raw[1]))


def _resolve_rgb_aug_args(args) -> Dict[str, Any]:
    """
    å°† RGB dataloader ä¸­å·²æœ‰çš„ augmentation æŽ¥å£ä»Ž argparse æ•´ç†ä¸º dictã€‚
    è®­ç»ƒè„šæœ¬åªè´Ÿè´£è§£æžå’Œä¸‹ä¼ ï¼›å®žé™…å¢žå¼ºé€»è¾‘ä»ç”± dataloader ä¸­çš„
    TemporallyConsistentSpatialAugmentation å®žçŽ°ã€‚
    """
    rgb_out_hw = _resolve_pair_arg(args, "rgb_out_hw", int)
    rrc_scale = _resolve_pair_arg(args, "rrc_scale", float)
    rrc_ratio = _resolve_pair_arg(args, "rrc_ratio", float)
    rgb_blur_sigma = _resolve_pair_arg(args, "rgb_blur_sigma", float)

    if rgb_out_hw[0] <= 0 or rgb_out_hw[1] <= 0:
        raise ValueError(f"--rgb_out_hw must be positive, got {rgb_out_hw}")
    if not (0.0 < rrc_scale[0] <= rrc_scale[1] <= 1.0):
        raise ValueError(f"--rrc_scale must satisfy 0 < min <= max <= 1, got {rrc_scale}")
    if not (0.0 < rrc_ratio[0] <= rrc_ratio[1]):
        raise ValueError(f"--rrc_ratio must satisfy 0 < min <= max, got {rrc_ratio}")
    if not (0.0 < rgb_blur_sigma[0] <= rgb_blur_sigma[1]):
        raise ValueError(f"--rgb_blur_sigma must satisfy 0 < min <= max, got {rgb_blur_sigma}")

    prob_names = [
        "rgb_hflip_p", "rgb_vflip_p", "rgb_jitter_p",
        "rgb_gray_p", "rgb_blur_p",
    ]
    for name in prob_names:
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name} must be in [0, 1], got {value}")

    for name in ["rgb_jitter_brightness", "rgb_jitter_contrast", "rgb_jitter_saturation"]:
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name} must be >= 0, got {value}")

    if not (0.0 <= float(args.rgb_jitter_hue) <= 0.5):
        raise ValueError(f"--rgb_jitter_hue must be in [0, 0.5], got {args.rgb_jitter_hue}")
    if int(args.rgb_blur_kernel) < 3 or int(args.rgb_blur_kernel) % 2 == 0:
        raise ValueError(f"--rgb_blur_kernel must be an odd integer >= 3, got {args.rgb_blur_kernel}")

    return {
        "rgb_out_hw": rgb_out_hw,
        "rrc_scale": rrc_scale,
        "rrc_ratio": rrc_ratio,
        "rgb_apply_spatial_aug": bool(args.rgb_apply_spatial_aug),
        "rgb_hflip_p": float(args.rgb_hflip_p),
        "rgb_vflip_p": float(args.rgb_vflip_p),
        "rgb_jitter_p": float(args.rgb_jitter_p),
        "rgb_jitter_brightness": float(args.rgb_jitter_brightness),
        "rgb_jitter_contrast": float(args.rgb_jitter_contrast),
        "rgb_jitter_saturation": float(args.rgb_jitter_saturation),
        "rgb_jitter_hue": float(args.rgb_jitter_hue),
        "rgb_gray_p": float(args.rgb_gray_p),
        "rgb_blur_p": float(args.rgb_blur_p),
        "rgb_blur_kernel": int(args.rgb_blur_kernel),
        "rgb_blur_sigma": rgb_blur_sigma,
    }



def parse_num_prototypes_per_class(
    spec: Optional[str],
    num_classes: int,
    default_num: int,
) -> List[int]:
    """
    å°†å‘½ä»¤è¡Œä¼ å…¥çš„ per-class prototype é…ç½®è§£æžæˆé•¿åº¦ä¸º num_classes çš„ list[int]ã€‚

    ä¾‹å­ï¼š
        spec = "2,2,3,1"
        -> [2, 2, 3, 1]

    è‹¥ spec is Noneï¼Œåˆ™æ‰€æœ‰ç±»åˆ«éƒ½ä½¿ç”¨ default_numã€‚
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
    å°†é«˜å±‚ ablation_mode è§£æžä¸ºåº•å±‚æŽ§åˆ¶å¼€å…³ã€‚

    è¿”å›žä¸‰ä¸ªå¸ƒå°”é‡ï¼š
    1) use_proto_state:
       æ˜¯å¦éœ€è¦æ•´å¥— prototype åŸºç¡€è®¾æ–½ï¼ŒåŒ…æ‹¬ï¼š
       - prototype refresh
       - proto_state broadcast
       - batch å†…æž„é€  proto_ids
       - step åŽçš„ prototype EMA update

    2) use_proto_loss:
       æ˜¯å¦è®¡ç®—å¹¶åŠ å…¥ prototype contrastive loss

    3) use_rel_loss:
       æ˜¯å¦è®¡ç®—å¹¶åŠ å…¥ prototype directional / relative loss

    è¯´æ˜Žï¼š
    - contrastive_only:
        åªè®­ç»ƒä¸»å¯¹æ¯”æŸå¤±ï¼Œä¸å¯ç”¨ä»»ä½• prototype ç›¸å…³çŠ¶æ€å’ŒæŸå¤±
    - contrastive_proto:
        ä¸»å¯¹æ¯”æŸå¤± + prototype contrastive loss
    - contrastive_rel:
        ä¸»å¯¹æ¯”æŸå¤± + prototype directional loss
        æ³¨æ„ï¼šè™½ç„¶ä¸ä½¿ç”¨ loss_protoï¼Œä½†ä»ç„¶éœ€è¦ prototype bankï¼Œ
        å› ä¸º directional loss æœ¬èº«ä¾èµ– prototype state
    - contrastive_proto_rel:
        ä¸»å¯¹æ¯”æŸå¤± + prototype contrastive loss + prototype directional loss
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
    This helper uses the 0-based epoch index used by the training loop.
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
    the ablation mode controls the active losses for all epochs, and lambda values
    are constant.

    With --enable_loss_stage_schedule, the intended experimental schedule is:
      - epoch < proto_loss_start_epoch:
            contrastive / SupLoss only
      - proto_loss_start_epoch <= epoch < rel_loss_start_epoch:
            contrastive / SupLoss + prototype contrastive loss
      - epoch >= rel_loss_start_epoch:
            contrastive / SupLoss + prototype contrastive loss + relative loss
            with lambda_rel ramped from 0 to args.lambda_rel over
            [rel_loss_start_epoch, rel_loss_end_epoch), then kept at args.lambda_rel

    The ablation mode still acts as an upper-level permission. For example,
    ablation_mode=contrastive_proto will not activate rel loss even in the rel stage.
    """
    if not args.enable_loss_stage_schedule:
        return {
            "use_proto_state": bool(base_use_proto_state),
            "use_proto_loss": bool(base_use_proto_loss),
            "use_rel_loss": bool(base_use_rel_loss),
            "lambda_proto": float(args.lambda_proto),
            "lambda_rel": float(args.lambda_rel),
        }

    in_proto_stage = epoch >= int(args.proto_loss_start_epoch)

    # é‡è¦ä¿®æ­£ï¼šrel_loss_end_epoch åªæ˜¯ lambda_rel ramp åˆ°æœ€å¤§å€¼çš„ç»ˆç‚¹ï¼Œ
    # ä¸åº”è¯¥ä½œä¸ºå…³é—­ rel loss çš„ç»ˆç‚¹ã€‚
    # å› æ­¤ rel loss ä»Ž rel_loss_start_epoch å¼€å§‹åŽä¼šä¸€ç›´ä¿æŒå¯ç”¨ï¼›
    # _cosine_ramp_value(...) ä¼šåœ¨ current_epoch >= rel_loss_end_epoch æ—¶è¿”å›ž final_valueã€‚
    in_rel_stage = epoch >= int(args.rel_loss_start_epoch)

    use_proto_state_eff = bool(base_use_proto_state and in_proto_stage)
    use_proto_loss_eff = bool(base_use_proto_loss and in_proto_stage)
    use_rel_loss_eff = bool(base_use_rel_loss and in_rel_stage)

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
    }


def prepare_model(
    depth: int,
    K: int,
    mlp: bool,
    proj_dim: int,
    temp: float,
    enable_kcl_loss: bool, 
    num_positive: int,
    exclude_invalid_queue: bool,
) -> nn.Module:
    """æž„å»º MoCo3D æ¨¡åž‹ã€‚"""
    model = MoCo3D(
        partial(ResNet3D.generate_model, model_depth=depth),
        dim=proj_dim,
        K=K,
        mlp=mlp,
        T=temp,
        num_positive=num_positive,
        exclude_invalid_queue=exclude_invalid_queue,
        enable_kcl_loss = enable_kcl_loss
    )
    return model


def prepare_trainloader(args, use_ddp: bool, rank: int, world_size: int):
    """
    æž„å»ºè®­ç»ƒé˜¶æ®µä½¿ç”¨çš„ map-style DataLoaderã€‚

    è®­ç»ƒé˜¶æ®µä¸Ž refresh é˜¶æ®µçš„ä¸»è¦åŒºåˆ«ï¼š
    ------------------------------------
    1) rgb_two_views=True
       å› ä¸º MoCo è®­ç»ƒéœ€è¦ query / key ä¸¤ä¸ªè§†è§’ã€‚

    2) is_train=True
       è®­ç»ƒæ—¶ä½¿ç”¨éšæœºå¢žå¼ºã€‚

    3) DDP æ¨¡å¼ä¸‹ä½¿ç”¨ DistributedSampler
       å¹¶ä¸”æ¯ä¸ª epoch éœ€è¦ set_epoch(epoch)ã€‚
    """
    label_map_path = _resolve_label_map_path(args)
    label_map = load_label_map_json(label_map_path)

    rgb_mean, rgb_std = _resolve_rgb_mean_std(args)
    rgb_aug_cfg = _resolve_rgb_aug_args(args)

    cfg = PackedMultiModalConfig(
        n_frames=args.n_frames,
        rgb_two_views=True,
        rgb_camera_id=args.rgb_camera_id,
        use_modalities=("rgb",),
        missing_policy="skip",
        load_labels=True,
        tier_mode=args.tier_mode,
        is_train=True,
        label_map_path=label_map_path,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        **rgb_aug_cfg,
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


def extract_two_views_and_labels(batch: dict, tier_mode: str):
    """
    ä»Žè®­ç»ƒ loader çš„ batch ä¸­æå–ï¼š
        - ä¸¤ä¸ª RGB è§†è§’
        - labels
        - global indices

    é¢„æœŸè¾“å…¥ï¼š
        batch["rgb"]       -> (view1, view2)
        batch["tier_ids"]  -> dict æˆ– Tensor
        batch["global_index"] / batch["idx"] / batch["sample_id"]
    """
    rgb = batch["rgb"]
    if isinstance(rgb, dict):
        view1 = rgb["view1"]
        view2 = rgb["view2"]
    elif isinstance(rgb, (tuple, list)) and len(rgb) == 2:
        view1, view2 = rgb
    else:
        raise RuntimeError(
            "Training loader is expected to output two RGB views, "
            f"but got type={type(rgb)}"
        )

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
    æ ¹æ®å‘½ä»¤è¡Œå‚æ•°æž„å»º PrototypeRefreshConfigã€‚
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

    rgb_mean, rgb_std = _resolve_rgb_mean_std(args)
    _ = _resolve_rgb_aug_args(args)  # validate exposed RGB augmentation args early

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
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        rgb_camera_id=args.rgb_camera_id,
        device=device,
        require_main_process_only=True,
        enable_prototype_temperature_scaling=args.enable_prototype_temperature_scaling,
        proto_base_temperature=args.proto_temperature,
        proto_temperature_eps=args.proto_temperature_eps,
    )


# ============================================================
# Debug é…ç½®ä¸Žå·¥å…·
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
    """å°†è°ƒè¯•ä¿¡æ¯æŒ‰ JSONL å½¢å¼è¿½åŠ å†™å…¥ï¼Œæ–¹ä¾¿åŽç»­ç¦»çº¿åˆ†æžã€‚"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _get_current_lrs(optimizer) -> List[float]:
    """è¿”å›žæ‰€æœ‰ param_group å½“å‰å­¦ä¹ çŽ‡ã€‚"""
    return [float(pg["lr"]) for pg in optimizer.param_groups]


def _unwrap_model(model: nn.Module) -> nn.Module:
    """åŽ»æŽ‰ DDP å¤–å£³ï¼Œä¾¿äºŽè¯»å–æ¨¡å—åå’Œå‚æ•°åã€‚"""
    return model.module if hasattr(model, "module") else model


def _resolve_tracked_param_names(
    model: nn.Module,
    patterns: List[str],
    fallback_last_n: int,
) -> List[str]:
    """
    æ ¹æ®åå­—å­ä¸²ç­›é€‰æƒ³é‡ç‚¹è§‚å¯Ÿçš„å‚æ•°ã€‚

    å…¸åž‹ç”¨é€”ï¼š
    - çœ‹åˆ†ç±»å¤´æ˜¯å¦åœ¨æ›´æ–°
    - çœ‹ backbone æœ€åŽå‡ å±‚æ˜¯å¦åœ¨æ›´æ–°
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
    ä¿å­˜è‹¥å¹²å…³é”®å‚æ•°çš„å½“å‰å€¼ï¼Œç”¨äºŽ optimizer.step() åŽè®¡ç®—æ›´æ–°å¹…åº¦ã€‚
    ä¸ºé™ä½Žæ˜¾å­˜å ç”¨ï¼Œè¿™é‡Œä¿å­˜åˆ° CPUã€‚
    """
    name_set = set(selected_names)
    snap: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name in name_set:
            snap[name] = p.detach().float().cpu().clone()
    return snap


def _compute_param_update_stats(model: nn.Module, before_snapshot: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
    """
    è®¡ç®—è¢«è·Ÿè¸ªå‚æ•°åœ¨ä¸€æ¬¡ step å‰åŽçš„æ›´æ–°å¹…åº¦ï¼š
    - absolute_update_norm
    - relative_update_norm = ||delta|| / (||param_before|| + eps)

    è¿™æ ·å¯ä»¥ç›´æŽ¥åˆ¤æ–­ï¼š
    - å‚æ•°åˆ°åº•æœ‰æ²¡æœ‰æ›´æ–°
    - æ›´æ–°æ˜¯å¦å°åˆ°å‡ ä¹Žå¯ä»¥å¿½ç•¥
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
    ç»Ÿè®¡æ¢¯åº¦ä¿¡æ¯ã€‚

    è¾“å‡ºåŒ…æ‹¬ï¼š
    - total_grad_norm
    - none_grad å‚æ•°ä¸ªæ•°
    - near_zero_grad å‚æ•°ä¸ªæ•°
    - æœ€å¤§æ¢¯åº¦èŒƒæ•°çš„ top-k å‚æ•°
    - è¢«é‡ç‚¹è·Ÿè¸ªå‚æ•°çš„æ¢¯åº¦èŒƒæ•°
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
    ç»Ÿè®¡å½“å‰ batch çš„æ ‡ç­¾ä¿¡æ¯ã€‚

    è¾“å‡ºåŒ…æ‹¬ï¼š
    1) batch ä¸­æœ‰å“ªäº›ç±»åˆ«
    2) æ¯ä¸ªç±»åˆ«å„æœ‰å¤šå°‘æ ·æœ¬
    3) ä»ç„¶ä¿ç•™ä¸€äº›å¯¹ SupCon / KCL æœ‰å¸®åŠ©çš„ç»Ÿè®¡é‡
    """
    labels_cpu = labels.detach().cpu()
    uniq, counts = torch.unique(labels_cpu, return_counts=True)

    batch_size = int(labels_cpu.numel())
    counts_float = counts.float()

    # è‹¥æŸç±»åœ¨ batch ä¸­å‡ºçŽ° n æ¬¡ï¼Œ
    # åˆ™è¯¥ç±»ä¸­çš„æ¯ä¸ªæ ·æœ¬éƒ½æœ‰ n-1 ä¸ªâ€œåŒç±»å…¶ä»–æ ·æœ¬â€
    avg_same_class_others = float(
        (counts_float * (counts_float - 1)).sum().item() / max(1, batch_size)
    )
    anchors_without_same_class = int(counts[counts == 1].sum().item())
    num_unique_classes = int(uniq.numel())

    # æ˜Žç¡®è®°å½•ï¼šbatch ä¸­æœ‰å“ªäº›ç±»åˆ«
    present_labels = [int(x.item()) for x in uniq]

    # æ˜Žç¡®è®°å½•ï¼šæ¯ä¸ªç±»åˆ«æœ‰å¤šå°‘æ ·æœ¬
    label_count_pairs = [
        {"label": int(u.item()), "count": int(c.item())}
        for u, c in zip(uniq, counts)
    ]

    # ä¹Ÿä¿ç•™ä¸€ä¸ªæŒ‰æ•°é‡é™åºæŽ’åˆ—çš„ç‰ˆæœ¬ï¼Œä¾¿äºŽå¿«é€ŸæŸ¥çœ‹
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
    ç»Ÿè®¡ query ç‰¹å¾ q çš„åŸºæœ¬åˆ†å¸ƒã€‚

    å¯ç”¨äºŽè§‚å¯Ÿï¼š
    - æ˜¯å¦æ•°å€¼èŒƒå›´å¼‚å¸¸
    - æ˜¯å¦å¯èƒ½å‡ºçŽ°ç‰¹å¾å¡Œç¼©
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
    """ç»Ÿè®¡ä¸€ä¸ªå¼ é‡ä¸­ NaN / +Inf / -Inf çš„æ•°é‡ã€‚"""
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
    æ£€æŸ¥å…³é”®å¼ é‡é‡Œæ˜¯å¦æœ‰ NaN / Infã€‚
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
    ç»Ÿè®¡å½“å‰ batch çš„ prototype åˆ†é…æƒ…å†µã€‚
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
    åœ¨ prototype refresh å®ŒæˆåŽè¾“å‡ºæ›´ç»†ä¸€ç‚¹çš„ç»Ÿè®¡ä¿¡æ¯ã€‚
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
# ç»Ÿè®¡å™¨ä¸Ž checkpoint
# ============================================================

class AverageMeter:
    """ç»´æŠ¤ä¸€ä¸ªæ ‡é‡çš„å½“å‰å€¼ã€ç´¯è®¡å’Œä»¥åŠå¹³å‡å€¼ã€‚"""
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
    """ä¿å­˜ checkpointã€‚"""
    torch.save(state, filename)


# ============================================================
# å•ä¸ª epoch çš„è®­ç»ƒ
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
) -> None:
    """
    è®­ç»ƒæ—¶åŒ…å«ä¸€ä¸ªä¸»å¯¹æ¯”æŸå¤±ï¼Œä»¥åŠæœ€å¤šä¸¤ä¸ª prototype è¾…åŠ©æŸå¤±ï¼š
    ----------------------------------------------------------
    1) ä¸»å¯¹æ¯”æŸå¤±ï¼š
    - kclï¼Œæˆ–
    - suploss
    ç”± contrastive_loss_mode æŽ§åˆ¶

    2) prototype contrastive loss
    ç”± use_proto_loss æŽ§åˆ¶

    3) prototype directional / relative loss
    ç”± use_rel_loss æŽ§åˆ¶

    å¦å¤–ï¼Œprototype ç›¸å…³çŠ¶æ€ï¼ˆrefresh / proto_ids / EMA updateï¼‰
    ç”± use_proto_state æŽ§åˆ¶ã€‚

    æ³¨æ„ï¼š
    - contrastive_only æ¨¡å¼ä¸‹ï¼Œuse_proto_state=Falseï¼Œæ­¤æ—¶æ•´ä¸ª prototype åˆ†æ”¯éƒ½å…³é—­
    - contrastive_rel æ¨¡å¼ä¸‹ï¼Œè™½ç„¶ä¸ä½¿ç”¨ loss_protoï¼Œä½†ä»ç„¶éœ€è¦ prototype stateï¼Œ
    å› ä¸º relative loss ä¾èµ– prototype bank å’Œ proto_ids

    è¿™ä¸€ç‰ˆé¢å¤–åŠ å…¥äº†ä¸€å¥— debug èƒ½åŠ›ï¼Œç”¨æ¥è¯Šæ–­ï¼š
    - æ¢¯åº¦æ˜¯å¦å­˜åœ¨
    - å‚æ•°æ˜¯å¦çœŸçš„æ›´æ–°
    - å½“å‰ batch çš„ç›‘ç£å¯¹æ¯”ä¿¡å·å¼ºä¸å¼º
    - prototype åˆ†æ”¯æ˜¯å¦æœ‰æ•ˆ
    - æ˜¯å¦å‡ºçŽ° NaN / Inf
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

        view1, view2, labels, global_index = extract_two_views_and_labels(batch, tier_mode)

        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        global_index = global_index.to(device, non_blocking=True).long()

        # map-style loader è¾“å‡ºä¸º [B, T, 3, H, W]
        # 3D å·ç§¯ç½‘ç»œé€šå¸¸éœ€è¦ [B, 3, T, H, W]
        view1 = view1.permute(0, 2, 1, 3, 4).contiguous()
        view2 = view2.permute(0, 2, 1, 3, 4).contiguous()

        optimizer.zero_grad(set_to_none=True)

        # å¦‚æžœè¦è§‚å¯Ÿå‚æ•°æ˜¯å¦çœŸçš„æ›´æ–°ï¼Œè¿™é‡Œå…ˆå¯¹è¢«è·Ÿè¸ªå‚æ•°åšå¿«ç…§
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
            # æ ¹æ® ablation é…ç½®å†³å®šæ˜¯å¦å¯ç”¨ prototype ç›¸å…³åˆ†æ”¯
            #
            # use_proto_state:
            #   æ˜¯å¦éœ€è¦ prototype bank / proto_ids / EMA update è¿™ä¸€æ•´å¥—çŠ¶æ€
            #
            # use_proto_loss:
            #   æ˜¯å¦è®¡ç®— prototype contrastive loss
            #
            # use_rel_loss:
            #   æ˜¯å¦è®¡ç®— prototype directional / relative loss
            #
            # æ³¨æ„ï¼š
            #   contrastive_rel æ¨¡å¼ä¸‹ use_proto_loss=Falseï¼Œä½† use_proto_state=Trueï¼Œ
            #   å› ä¸º relative loss ä»ç„¶ä¾èµ– prototype bank
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
                # å®‰å…¨æž„é€  proto_idsï¼š
                # 1) è‹¥ global_index è¶…å‡º sample_to_proto èŒƒå›´ï¼Œåˆ™è®¾ä¸º -1
                # 2) è‹¥ valid_sample_mask æ˜¾ç¤ºè¯¥æ ·æœ¬å½“å‰æ— æ•ˆï¼Œåˆ™è®¾ä¸º -1
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

                # é»˜è®¤å…ˆç½®é›¶ï¼Œå†æŒ‰ ablation å¼€å…³å†³å®šæ˜¯å¦çœŸæ­£è®¡ç®—
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
                # æœ€ç»ˆæ€»æŸå¤±ï¼š
                # ä¸»æŸå¤±å§‹ç»ˆå­˜åœ¨ï¼›
                # proto / rel æ˜¯å¦åŠ å…¥ï¼Œç”± ablation å¼€å…³å†³å®š
                # ------------------------------------------------------------
                loss = loss_supcon
                if use_proto_loss:
                    loss = loss + lambda_proto * loss_proto
                if use_rel_loss:
                    loss = loss + lambda_rel * loss_rel

        # -----------------------------
        # éžæœ‰é™å€¼æ£€æŸ¥ï¼šåœ¨ backward å‰åš
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

        # AMP ä¸‹è‹¥è¦çœ‹çœŸå®žæ¢¯åº¦èŒƒæ•°ï¼Œå…ˆ unscale å†ç»Ÿè®¡
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
        # åªæœ‰åœ¨å¯ç”¨ prototype state æ—¶ï¼Œæ‰è¿›è¡ŒçœŸå®ž prototype bank çš„ EMA æ›´æ–°
        # contrastive_only æ¨¡å¼ä¸‹ï¼Œè¿™ä¸€æ­¥å¿…é¡»å½»åº•å…³é—­
        # contrastive_rel æ¨¡å¼ä¸‹ï¼Œè¿™ä¸€æ­¥ä»ç„¶éœ€è¦ä¿ç•™ï¼Œå› ä¸º rel loss ä¾èµ–åŠ¨æ€ prototype bank
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
        # å¸¸è§„è®­ç»ƒæ—¥å¿—
        # -----------------------------
        if is_main_process(rank) and ((step_idx % print_freq) == 0):
            log(
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

        # -----------------------------
        # Debug æ—¥å¿—
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

            # 1) batch æ ‡ç­¾ç»Ÿè®¡
            if debug_cfg.batch_label_stats:
                debug_payload["batch_label_stats"] = _compute_batch_label_stats(labels)

            # 2) q ç‰¹å¾ç»Ÿè®¡
            if debug_cfg.feature_stats:
                debug_payload["feature_stats"] = _compute_feature_stats(q)

            # 3) prototype åˆ†é…ç»Ÿè®¡
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

            # 4) æ¢¯åº¦ç»Ÿè®¡
            if grad_payload is not None:
                debug_payload["grad_stats"] = grad_payload

            # 5) å‚æ•°æ›´æ–°ç»Ÿè®¡
            if param_update_payload is not None:
                debug_payload["param_update_stats"] = param_update_payload

            # 6) éžæœ‰é™å€¼ç»Ÿè®¡
            if nonfinite_payload is not None:
                debug_payload["nonfinite_check"] = nonfinite_payload

            # ç®€æ˜Žæ‰“å°ç‰ˆ
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
# ä¸» worker
# ============================================================

def worker(args) -> None:
    """
    ä¸»è®­ç»ƒå…¥å£ã€‚

    æ•´ä½“æµç¨‹ï¼š
    ----------
    1) åˆå§‹åŒ–å•è¿›ç¨‹æˆ– DDP
    2) æž„å»ºæ¨¡åž‹ã€ä¼˜åŒ–å™¨ã€è®­ç»ƒ loader
    3) è§£æžç±»åˆ«æ•°ï¼Œå¹¶æž„é€  per-class prototype é…ç½®
    4) æŒ‰ epoch å†³å®šæ˜¯å¦åˆ·æ–° prototypes
    5) æ‰§è¡Œè®­ç»ƒ
    6) æŒ‰è®¾å®šé—´éš”ä¿å­˜ checkpoint
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
        if is_main_process(rank):
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

        model = prepare_model(
            args.model_depth,
            args.K_queue,
            args.mlp,
            args.proj_dim,
            args.temperature,
            num_positive=args.num_positive,
            exclude_invalid_queue=args.exclude_invalid_queue,
            enable_kcl_loss=(args.contrastive_loss == "kcl")
        ).to(device)

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
            #
            # In the original mode, these are identical to the ablation flags.
            # In staged mode, this implements:
            #   1) SupLoss only
            #   2) SupLoss + prototype loss
            #   3) SupLoss + prototype loss + relative loss with scheduled lambda_rel
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

            if is_main_process(rank) and args.enable_loss_stage_schedule:
                log(
                    f"[LossStage] epoch={epoch + 1} "
                    f"use_proto_state={use_proto_state_epoch} "
                    f"use_proto_loss={use_proto_loss_epoch} "
                    f"use_rel_loss={use_rel_loss_epoch} "
                    f"lambda_proto={lambda_proto_epoch:.6f} "
                    f"lambda_rel={lambda_rel_epoch:.6f}"
                )

            # ------------------------------------------------------------
            # æ˜¯å¦éœ€è¦è¿›è¡Œ prototype refresh
            #
            # åªæœ‰åœ¨å½“å‰ epoch éœ€è¦ prototype state æ—¶ï¼Œæ‰å…è®¸ refreshã€‚
            # staged schedule ä¸‹ï¼Œè¿™å¯ä»¥ä¿è¯å‰è‹¥å¹² epoch æ˜¯çœŸæ­£çš„ SupLoss-onlyã€‚
            # ------------------------------------------------------------
            need_refresh = False
            if use_proto_state_epoch:
                if proto_state is None:
                    need_refresh = True
                elif epoch == args.warmup_epochs:
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
                # æ˜Žç¡®ä¿è¯å½“å‰ epoch ä¸ä½¿ç”¨ prototype state æ—¶ï¼Œä¸ä¿ç•™ä»»ä½• prototype çŠ¶æ€ã€‚
                # å¯¹ staged schedule æ¥è¯´ï¼Œè¿™ä¿è¯å‰ 50 ä¸ª epoch ä¸åš prototype refresh / EMA updateã€‚
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
    worker(args)
