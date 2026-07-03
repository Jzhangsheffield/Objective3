#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_mapstyle_finetune_and_test.py

æœ¬ç‰ˆæœ¬åœ¨åŽŸå§‹ map-style åˆ†ç±»è®­ç»ƒ / æµ‹è¯•è„šæœ¬åŸºç¡€ä¸Šï¼Œæ”¯æŒä»¥ä¸‹åŠŸèƒ½ï¼š

1) è®­ç»ƒæ¨¡å¼ï¼ˆrun_mode=trainï¼‰
   - æ”¯æŒä»Žå¤´è®­ç»ƒï¼ˆscratchï¼‰
   - æ”¯æŒæä¾›å¤šä¸ªé¢„è®­ç»ƒæƒé‡è·¯å¾„ï¼Œé€ä¸ªåŠ è½½åŽåˆ†åˆ«è®­ç»ƒ
   - é¢„è®­ç»ƒåŠ è½½æ—¶å¯é€‰æ‹©è‡ªåŠ¨ä¸¢å¼ƒå¯¹æ¯”å­¦ä¹ å¤´ / åˆ†ç±»å¤´ç­‰ä¸éœ€è¦çš„æƒé‡
   - æ”¯æŒä¸¤ç§å¾®è°ƒæ–¹å¼ï¼š
       a) full      : å…¨éƒ¨å‚æ•°éƒ½è®­ç»ƒ
       b) head_only : åªè®­ç»ƒåˆ†ç±»å¤´ï¼Œå†»ç»“ backbone
   - æ”¯æŒ discriminative learning rateï¼ˆåŒºåˆ†å­¦ä¹ çŽ‡ï¼‰
       a) backbone ä¸€ä¸ªå­¦ä¹ çŽ‡
       b) åˆ†ç±»å¤´ä¸€ä¸ªå­¦ä¹ çŽ‡
   - æ¯ä¸ªå®žéªŒï¼ˆæ¯ä¸ªé¢„è®­ç»ƒæºï¼‰éƒ½ä¼šå•ç‹¬ä¿å­˜ï¼š
       - æ—¥å¿—
       - checkpoint
       - datamap / training dynamics
       - è®­ç»ƒæ‘˜è¦

2) æµ‹è¯•æ¨¡å¼ï¼ˆrun_mode=testï¼‰
   - æ”¯æŒæä¾›ä¸€ä¸ªæˆ–å¤šä¸ª test manifest
   - æ”¯æŒæä¾›å¤šä¸ªå·²è®­ç»ƒæƒé‡æ–‡ä»¶
   - æŒ‰é¡ºåºæˆ–å¹¿æ’­æ–¹å¼é€ä¸ªåŠ è½½å¹¶åœ¨æµ‹è¯•é›†ä¸Šè¯„ä¼°
   - å°†æµ‹è¯•ç»“æžœæ±‡æ€»ä¿å­˜ä¸º CSV

============================================================
æœ¬ç‰ˆæœ¬æœ€é‡è¦çš„ä¿®æ”¹ç‚¹ï¼š
============================================================
- ä¸å†æŒ‰æ–‡ä»¶åä¸­çš„ imbalance_XXX ç­‰å†…å®¹è‡ªåŠ¨åŒ¹é…
- æ”¹ä¸ºæŒ‰å‘½ä»¤è¡Œè¾“å…¥é¡ºåºåŒ¹é…

è®­ç»ƒæ¨¡å¼é¡ºåºåŒ¹é…è§„åˆ™ï¼š
---------------------
å‡è®¾ä½ è¾“å…¥ï¼š
    --train_manifests A.jsonl B.jsonl C.jsonl
    --val_manifests   VA.jsonl VB.jsonl VC.jsonl
    --pretrained_weight_paths P1.pth P2.pth P3.pth

åˆ™ä¼šæž„é€ ä¸‰ä¸ªå®žéªŒï¼š
    A + VA + P1
    B + VB + P2
    C + VC + P3

å¹¿æ’­è§„åˆ™ï¼š
---------
1) train manifestï¼š
   - å¯ä»¥æä¾› 1 ä¸ªï¼Œç„¶åŽå¹¿æ’­ç»™æ‰€æœ‰ pretrained weights
   - æˆ–æä¾›ä¸Ž pretrained weights æ•°é‡ç›¸åŒçš„å¤šä¸ªï¼ŒæŒ‰é¡ºåºä¸€ä¸€å¯¹åº”

2) val manifestï¼š
   - å¯ä»¥ä¸æä¾›ï¼ˆå…¨éƒ¨ä¸åšéªŒè¯ï¼‰
   - å¯ä»¥æä¾› 1 ä¸ªï¼Œç„¶åŽå¹¿æ’­ç»™æ‰€æœ‰å®žéªŒ
   - æˆ–æä¾›ä¸Žå®žéªŒæ•°ç›¸åŒçš„å¤šä¸ªï¼ŒæŒ‰é¡ºåºä¸€ä¸€å¯¹åº”

æµ‹è¯•æ¨¡å¼é¡ºåºåŒ¹é…è§„åˆ™ï¼š
---------------------
å‡è®¾ä½ è¾“å…¥ï¼š
    --test_manifest T1.jsonl T2.jsonl
    --test_weight_paths W1.pth W2.pth

åˆ™ä¼šæµ‹è¯•ï¼š
    T1 + W1
    T2 + W2

å¹¿æ’­è§„åˆ™ï¼š
---------
- è‹¥åªæä¾› 1 ä¸ª test manifestï¼Œåˆ™å¹¿æ’­ç»™æ‰€æœ‰ test_weight_paths
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
# map-style loader ç›¸å…³å¯¼å…¥
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

# ---------------- è¿è¡Œæ¨¡å¼ ----------------
parser.add_argument(
    "--run_mode",
    type=str,
    default="train",
    choices=["train", "test"],
    help="train: è®­ç»ƒ/å¾®è°ƒï¼›test: ä»…æ‰¹é‡æµ‹è¯•å¤šä¸ªæƒé‡",
)

# ---------------- åŸºæœ¬è·¯å¾„ ----------------
parser.add_argument(
    "--save_path",
    type=str,
    default=r"./weights",
    help="æ‰€æœ‰å®žéªŒè¾“å‡ºçš„æ ¹ç›®å½•"
)
parser.add_argument(
    "--datamap_csv_path",
    type=str,
    default=r"./datamaps",
    help="datamap / training dynamics æ ¹ç›®å½•"
)
parser.add_argument(
    "--dataset_root",
    type=str,
    required=True,
    help="map-style æ•°æ®é›†æ ¹ç›®å½•"
)
parser.add_argument(
    "--label_map_json",
    type=str,
    required=True,
    help="ç»Ÿä¸€ label_map.json è·¯å¾„"
)

# ---------------- manifest è·¯å¾„ ----------------
# ä¸ºäº†å…¼å®¹æ—§è°ƒç”¨æ–¹å¼ï¼Œä»ç„¶ä¿ç•™å•ä¸ªå‚æ•°å’Œå¤šä¸ªå‚æ•°ä¸¤ç§æŽ¥å£ã€‚
# å…¶ä¸­ï¼š
#   - å•ä¸ªå‚æ•° train_manifest / val_manifest ä¼šè¢«è§†ä¸ºåˆ—è¡¨ä¸­çš„ç¬¬ä¸€ä¸ªå…ƒç´ 
#   - å¤šä¸ªå‚æ•° train_manifests / val_manifests ä¼šæŒ‰è¾“å…¥é¡ºåºè¿½åŠ åœ¨åŽé¢
parser.add_argument(
    "--train_manifest",
    type=str,
    default=None,
    help="å•ä¸ªè®­ç»ƒ manifest æ–‡ä»¶è·¯å¾„æˆ–æ–‡ä»¶åï¼›ä¿ç•™ç”¨äºŽå‘åŽå…¼å®¹"
)
parser.add_argument(
    "--train_manifests",
    nargs="*",
    default=[],
    help="å¤šä¸ªè®­ç»ƒ manifestï¼›æœ¬ç‰ˆæœ¬æŒ‰è¾“å…¥é¡ºåºåŒ¹é…ï¼Œè€Œä¸æ˜¯æŒ‰æ–‡ä»¶åè‡ªåŠ¨åŒ¹é…"
)
parser.add_argument(
    "--val_manifest",
    type=str,
    default=None,
    help="å•ä¸ªéªŒè¯ manifest æ–‡ä»¶è·¯å¾„æˆ–æ–‡ä»¶åï¼›ä¿ç•™ç”¨äºŽå‘åŽå…¼å®¹"
)
parser.add_argument(
    "--val_manifests",
    nargs="*",
    default=[],
    help="å¤šä¸ªéªŒè¯ manifestï¼›æœ¬ç‰ˆæœ¬æŒ‰è¾“å…¥é¡ºåºåŒ¹é…ï¼Œè€Œä¸æ˜¯æŒ‰æ–‡ä»¶åè‡ªåŠ¨åŒ¹é…"
)

# æ³¨æ„ï¼šè¿™é‡Œå·²ç»æ”¹æˆæ”¯æŒå¤šä¸ª test manifest
parser.add_argument(
    "--test_manifest",
    nargs="+",
    default=[],
    help=(
        "ä¸€ä¸ªæˆ–å¤šä¸ªæµ‹è¯• manifest æ–‡ä»¶è·¯å¾„æˆ–æ–‡ä»¶åã€‚"
        "è‹¥åªç»™ 1 ä¸ªï¼Œä¼šå¹¿æ’­ç»™æ‰€æœ‰ test_weight_pathsï¼›"
        "è‹¥ç»™å¤šä¸ªï¼Œåˆ™æŒ‰è¾“å…¥é¡ºåºä¸Ž test_weight_paths ä¸€ä¸€å¯¹åº”ã€‚"
    )
)

# ---------------- æ–‡ä»¶å‘½å -------------------
parser.add_argument(
    "--pretrained_tag_mode",
    type=str,
    default="legacy",
    choices=["legacy", "last_k_dirs", "relative_to_anchor"],
    help=(
        "å¦‚ä½•ä»Ž pretrained_weight_paths æž„é€ å®žéªŒç›®å½•åã€‚"
        "legacy: æ—§é€»è¾‘ï¼Œåªç”¨ parent.name + stemï¼›"
        "last_k_dirs: ä½¿ç”¨æƒé‡è·¯å¾„æœ€åŽ k å±‚ç›®å½• + stemï¼›"
        "relative_to_anchor: å–æŸä¸ªé”šç‚¹ç›®å½•ä¹‹åŽçš„ç›¸å¯¹è·¯å¾„ä½œä¸ºæ ‡ç­¾ã€‚"
    ),
)

parser.add_argument(
    "--pretrained_tag_last_k",
    type=int,
    default=4,
    help="å½“ pretrained_tag_mode=last_k_dirs æ—¶ï¼Œå–æƒé‡è·¯å¾„æœ«å°¾å¤šå°‘å±‚ç›®å½•å‚ä¸Žå‘½å"
)

parser.add_argument(
    "--pretrained_tag_anchor",
    type=str,
    default=None,
    help=(
        "å½“ pretrained_tag_mode=relative_to_anchor æ—¶ä½¿ç”¨ã€‚"
        "ä¾‹å¦‚è®¾ä¸º 'J_test'ï¼Œåˆ™ä»Ž J_test ä¹‹åŽçš„ç›¸å¯¹è·¯å¾„å¼€å§‹æž„é€ æ ‡ç­¾ã€‚"
    ),
)

# ---------------- æ ‡ç­¾ / æ¨¡æ€ ----------------
parser.add_argument(
    "--tier_mode",
    type=str,
    default="tier1",
    choices=["tier1", "tier2", "tier3"],
    help="ä½¿ç”¨å“ªä¸ª tier çš„æ ‡ç­¾è®­ç»ƒ/æµ‹è¯•"
)
parser.add_argument(
    "--n_frames",
    type=int,
    default=16,
    help="æ¯ä¸ªæ ·æœ¬é‡‡æ ·å¸§æ•°"
)
parser.add_argument(
    "--rgb_camera_id",
    type=str,
    default="001484412812",
    help="RGB camera id to use when manifest has camera-specific fields, e.g. 001484412812"
)
parser.add_argument(
    "--use_modality",
    type=str,
    default="rgb",
    choices=["rgb", "depth", "mindrove"],
    help="å½“å‰åˆ†ç±»åªä½¿ç”¨å•æ¨¡æ€è¾“å…¥ï¼šrgb / depth / mindrove"
)

# ---------------- DataLoader ----------------
parser.add_argument(
    "--num_workers_train",
    type=int,
    default=8,
    help="è®­ç»ƒé›† DataLoader worker æ•°é‡"
)
parser.add_argument(
    "--num_workers_val",
    type=int,
    default=6,
    help="éªŒè¯é›† DataLoader worker æ•°é‡"
)
parser.add_argument(
    "--num_workers_test",
    type=int,
    default=8,
    help="æµ‹è¯•é›† DataLoader worker æ•°é‡"
)
parser.add_argument(
    "--prefetch_factor_train",
    type=int,
    default=2,
    help="è®­ç»ƒé›† prefetch_factorï¼›num_workers=0 æ—¶å¿½ç•¥"
)
parser.add_argument(
    "--prefetch_factor_val",
    type=int,
    default=2,
    help="éªŒè¯é›† prefetch_factorï¼›num_workers=0 æ—¶å¿½ç•¥"
)
parser.add_argument(
    "--prefetch_factor_test",
    type=int,
    default=2,
    help="æµ‹è¯•é›† prefetch_factorï¼›num_workers=0 æ—¶å¿½ç•¥"
)

# ---------------- éªŒè¯å¼€å…³ ----------------
parser.add_argument(
    "--disable_val",
    action="store_true",
    help="è®­ç»ƒæ—¶ç¦ç”¨éªŒè¯"
)

# ---------------- è¾“å‡ºç©ºé—´å°ºå¯¸ï¼ˆç”± loader å†…éƒ¨å®Œæˆï¼‰ ----------------
parser.add_argument(
    "--rgb_size",
    type=int,
    default=224,
    help="RGB è¾“å‡ºå°ºå¯¸ï¼ˆH=Wï¼‰"
)
parser.add_argument(
    "--depth_size",
    type=int,
    default=224,
    help="Depth è¾“å‡ºå°ºå¯¸ï¼ˆH=Wï¼‰"
)
# ---------------- RGB normalization å‚æ•°ï¼ˆç”± loader å†…éƒ¨ Normalize ä½¿ç”¨ï¼‰ ----------------
parser.add_argument(
    "--rgb_mean",
    nargs=3,
    type=float,
    default=[0.356, 0.363, 0.367],
    metavar=("R_MEAN", "G_MEAN", "B_MEAN"),
    help=(
        "RGB Normalize ä½¿ç”¨çš„ meanï¼Œå¿…é¡»ç»™ 3 ä¸ª floatï¼Œé¡ºåºä¸º R G Bã€‚"
        "ä¾‹å¦‚ï¼š--rgb_mean 0.356 0.363 0.367"
    ),
)
parser.add_argument(
    "--rgb_std",
    nargs=3,
    type=float,
    default=[0.288, 0.271, 0.270],
    metavar=("R_STD", "G_STD", "B_STD"),
    help=(
        "RGB Normalize ä½¿ç”¨çš„ stdï¼Œå¿…é¡»ç»™ 3 ä¸ªæ­£æ•°ï¼Œé¡ºåºä¸º R G Bã€‚"
        "ä¾‹å¦‚ï¼š--rgb_std 0.288 0.271 0.270"
    ),
)

# ---------------- RGB train augment å‚æ•°ï¼ˆç”± loader ä½¿ç”¨ï¼‰ ----------------
parser.add_argument(
    "--rrc_scale_min",
    type=float,
    default=0.6,
    help="RandomResizedCrop scale æœ€å°å€¼"
)
parser.add_argument(
    "--rrc_scale_max",
    type=float,
    default=1.0,
    help="RandomResizedCrop scale æœ€å¤§å€¼"
)
parser.add_argument(
    "--rrc_ratio_min",
    type=float,
    default=0.75,
    help="RandomResizedCrop ratio æœ€å°å€¼"
)
parser.add_argument(
    "--rrc_ratio_max",
    type=float,
    default=1.3333333333,
    help="RandomResizedCrop ratio æœ€å¤§å€¼"
)
parser.add_argument(
    "--rgb_apply_spatial_aug",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "è®­ç»ƒé›†æ˜¯å¦å¯ç”¨ RGB éšæœºç©ºé—´å¢žå¼ºä¸­çš„ flip/jitter/gray/blurã€‚"
        "æ³¨æ„ï¼šè®¾ä¸º False æ—¶ä»ä½¿ç”¨ TemporallyConsistentSpatialAugmentationï¼Œ"
        "å› æ­¤ RandomResizedCrop ä»ç„¶ä¿ç•™ï¼›åªä¼šæŠŠ flip/jitter/gray/blur çš„æ¦‚çŽ‡ç½® 0ã€‚"
        "éªŒè¯/æµ‹è¯•é›†ä¸å—è¯¥å‚æ•°å½±å“ã€‚"
    ),
)

parser.add_argument(
    "--rgb_hflip_p",
    type=float,
    default=0.5,
    help="è®­ç»ƒé›† RGB RandomHorizontalFlip æ¦‚çŽ‡"
)
parser.add_argument(
    "--rgb_vflip_p",
    type=float,
    default=0.5,
    help="è®­ç»ƒé›† RGB RandomVerticalFlip æ¦‚çŽ‡ï¼›æœºæ¢°æ“ä½œè§†é¢‘é€šå¸¸å»ºè®®ä¸º 0"
)

parser.add_argument(
    "--rgb_jitter_p",
    type=float,
    default=0.5,
    help="è®­ç»ƒé›† RGB ColorJitter è¢«åº”ç”¨çš„æ¦‚çŽ‡"
)
parser.add_argument(
    "--rgb_jitter_brightness",
    type=float,
    default=0.24,
    help="ColorJitter brightness å¼ºåº¦"
)
parser.add_argument(
    "--rgb_jitter_contrast",
    type=float,
    default=0.24,
    help="ColorJitter contrast å¼ºåº¦"
)
parser.add_argument(
    "--rgb_jitter_saturation",
    type=float,
    default=0.24,
    help="ColorJitter saturation å¼ºåº¦"
)
parser.add_argument(
    "--rgb_jitter_hue",
    type=float,
    default=0.16,
    help="ColorJitter hue å¼ºåº¦ï¼›torchvision è¦æ±‚é€šå¸¸ä¸è¶…è¿‡ 0.5"
)

parser.add_argument(
    "--rgb_gray_p",
    type=float,
    default=0.2,
    help="è®­ç»ƒé›† RGB RandomGrayscale æ¦‚çŽ‡"
)

parser.add_argument(
    "--rgb_blur_p",
    type=float,
    default=0.5,
    help="è®­ç»ƒé›† RGB GaussianBlur è¢«åº”ç”¨çš„æ¦‚çŽ‡"
)
parser.add_argument(
    "--rgb_blur_kernel",
    type=int,
    default=7,
    help="GaussianBlur kernel sizeï¼Œå¿…é¡»æ˜¯ >=3 çš„å¥‡æ•°"
)
parser.add_argument(
    "--rgb_blur_sigma_min",
    type=float,
    default=0.1,
    help="GaussianBlur sigma ä¸‹ç•Œ"
)
parser.add_argument(
    "--rgb_blur_sigma_max",
    type=float,
    default=1.0,
    help="GaussianBlur sigma ä¸Šç•Œ"
)

# ---------------- MindRove data config ----------------
parser.add_argument(
    "--mindrove_target_len",
    type=int,
    default=256,
    help="MindRove åºåˆ—é‡é‡‡æ ·åŽçš„ç»Ÿä¸€é•¿åº¦"
)
parser.add_argument(
    "--mindrove_hands",
    nargs="+",
    default=["left", "right"],
    choices=["left", "right"],
    help="MindRove ä½¿ç”¨å“ªäº›æ‰‹çš„æ•°æ®"
)
parser.add_argument(
    "--mindrove_signals",
    nargs="+",
    default=["emg", "imu"],
    choices=["emg", "imu"],
    help="MindRove ä½¿ç”¨å“ªäº›ä¿¡å·"
)
parser.add_argument(
    "--mindrove_merge_hands",
    action="store_true",
    help="æ˜¯å¦å°†å·¦å³æ‰‹åŒç±»ä¿¡å·åœ¨é€šé“ç»´æ‹¼æŽ¥åŽè¾“å‡º"
)
parser.add_argument(
    "--mindrove_apply_augmentation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="è®­ç»ƒé›†æ˜¯å¦å¯ç”¨ MindRove æ ·æœ¬çº§å¢žå¼ºï¼›éªŒè¯/æµ‹è¯•é›†ä¼šç”± dataloader è‡ªåŠ¨å…³é—­"
)
parser.add_argument(
    "--mindrove_apply_normalization",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="æ˜¯å¦åœ¨é‡é‡‡æ ·åŽã€å¢žå¼ºå‰ï¼Œå¯¹ MindRove åš per-channel mean/std æ ‡å‡†åŒ–"
)
parser.add_argument(
    "--disable_train_augmentation",
    action="store_true",
    help=(
        "ç»Ÿä¸€å…³é—­è®­ç»ƒé›†å¢žå¼ºã€‚å¯ç”¨åŽï¼š"
        "RGB çš„ RandomResizedCrop ä¼šé€€åŒ–ä¸º scale=(1,1)ã€ratio=(1,1)ï¼Œ"
        "flip/jitter/gray/blur æ¦‚çŽ‡å…¨éƒ¨ç½® 0ï¼›"
        "MindRove æ ·æœ¬çº§å¢žå¼ºä¹Ÿä¼šå…³é—­ã€‚éªŒè¯/æµ‹è¯•æœ¬æ¥å°±ä¸å¯ç”¨è®­ç»ƒå¢žå¼ºã€‚"
    ),
)

# ---------------- MindRove normalization stats ----------------
parser.add_argument("--mindrove_left_emg_mean", nargs="+", type=float, default=None,
                    help="å·¦æ‰‹ EMG çš„ per-channel meanï¼Œé•¿åº¦å¿…é¡»ä¸º 8")
parser.add_argument("--mindrove_left_emg_std", nargs="+", type=float, default=None,
                    help="å·¦æ‰‹ EMG çš„ per-channel stdï¼Œé•¿åº¦å¿…é¡»ä¸º 8")
parser.add_argument("--mindrove_right_emg_mean", nargs="+", type=float, default=None,
                    help="å³æ‰‹ EMG çš„ per-channel meanï¼Œé•¿åº¦å¿…é¡»ä¸º 8")
parser.add_argument("--mindrove_right_emg_std", nargs="+", type=float, default=None,
                    help="å³æ‰‹ EMG çš„ per-channel stdï¼Œé•¿åº¦å¿…é¡»ä¸º 8")
parser.add_argument("--mindrove_left_imu_mean", nargs="+", type=float, default=None,
                    help="å·¦æ‰‹ IMU çš„ per-channel meanï¼Œé•¿åº¦å¿…é¡»ä¸º 6")
parser.add_argument("--mindrove_left_imu_std", nargs="+", type=float, default=None,
                    help="å·¦æ‰‹ IMU çš„ per-channel stdï¼Œé•¿åº¦å¿…é¡»ä¸º 6")
parser.add_argument("--mindrove_right_imu_mean", nargs="+", type=float, default=None,
                    help="å³æ‰‹ IMU çš„ per-channel meanï¼Œé•¿åº¦å¿…é¡»ä¸º 6")
parser.add_argument("--mindrove_right_imu_std", nargs="+", type=float, default=None,
                    help="å³æ‰‹ IMU çš„ per-channel stdï¼Œé•¿åº¦å¿…é¡»ä¸º 6")

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
                    help="EMG drift çš„æœ€å¤§å¹…å€¼ï¼›ä¼  1 ä¸ªå€¼è¡¨ç¤ºå›ºå®šå¹…å€¼ï¼Œä¼  2 ä¸ªå€¼è¡¨ç¤º [low, high]")
parser.add_argument("--mindrove_emg_drift_n_points", nargs="+", type=int, default=[3],
                    help="EMG drift æŽ§åˆ¶ç‚¹æ•°ï¼›ä¼  1 ä¸ªå€¼è¡¨ç¤ºå›ºå®šå€¼ï¼Œä¼ å¤šä¸ªå€¼è¡¨ç¤ºå€™é€‰åˆ—è¡¨")
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
                    help="IMU drift çš„æœ€å¤§å¹…å€¼ï¼›ä¼  1 ä¸ªå€¼è¡¨ç¤ºå›ºå®šå¹…å€¼ï¼Œä¼  2 ä¸ªå€¼è¡¨ç¤º [low, high]")
parser.add_argument("--mindrove_imu_drift_n_points", nargs="+", type=int, default=[3],
                    help="IMU drift æŽ§åˆ¶ç‚¹æ•°ï¼›ä¼  1 ä¸ªå€¼è¡¨ç¤ºå›ºå®šå€¼ï¼Œä¼ å¤šä¸ªå€¼è¡¨ç¤ºå€™é€‰åˆ—è¡¨")
parser.add_argument("--mindrove_imu_drift_kind", type=str, default="additive",
                    choices=["additive", "multiplicative"])
parser.add_argument("--mindrove_imu_drift_per_channel", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--mindrove_imu_drift_normalize", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--mindrove_imu_negate_prob", type=float, default=0.0)
parser.add_argument("--mindrove_imu_channel_dropout_prob", type=float, default=0.0)
parser.add_argument("--mindrove_imu_channel_dropout_max_channels", type=int, default=1)

# ---------------- æ¨¡åž‹ä¸Žè®­ç»ƒ ----------------
parser.add_argument(
    "--model_depth",
    type=int,
    default=18,
    help="3D ResNet æ·±åº¦"
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=17,
    help="åˆ†ç±»ç±»åˆ«æ•°é‡"
)
parser.add_argument(
    "--l2_normalize_before_fc",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "æ˜¯å¦åœ¨æ¨¡åž‹æœ€ç»ˆ fc åˆ†ç±»å¤´ä¹‹å‰å¯¹ backbone feature åš L2 normalizeã€‚"
        "é»˜è®¤å…³é—­ï¼Œä¿æŒåŽŸå§‹è¡Œä¸ºã€‚æ³¨æ„ï¼šè®­ç»ƒå’Œæµ‹è¯•å¿…é¡»ä½¿ç”¨ç›¸åŒè®¾ç½®ã€‚"
    ),
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="è®­ç»ƒè½®æ•°"
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
    help="é»˜è®¤åŸºç¡€å­¦ä¹ çŽ‡ï¼›å•å­¦ä¹ çŽ‡æ¨¡å¼ä¸‹ç›´æŽ¥ä½¿ç”¨"
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
    help="ä¼˜åŒ–å™¨çš„ weight decayï¼›SGD å’Œ AdamW éƒ½ä¼šä½¿ç”¨è¯¥å€¼"
)

parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["sgd", "adamw"],
    help=(
        "é€‰æ‹©å¾®è°ƒä¼˜åŒ–å™¨ã€‚"
        "sgd: ä½¿ç”¨ torch.optim.SGDï¼Œä¿ç•™ momentumï¼›"
        "adamw: ä½¿ç”¨ torch.optim.AdamWï¼Œä¸ä½¿ç”¨ momentumï¼Œé‡‡ç”¨ decoupled weight decayã€‚"
    ),
)

parser.add_argument(
    "--adamw_beta1",
    type=float,
    default=0.9,
    help="AdamW beta1ï¼›ä»…åœ¨ --optimizer adamw æ—¶ä½¿ç”¨"
)

parser.add_argument(
    "--adamw_beta2",
    type=float,
    default=0.999,
    help="AdamW beta2ï¼›ä»…åœ¨ --optimizer adamw æ—¶ä½¿ç”¨"
)

parser.add_argument(
    "--adamw_eps",
    type=float,
    default=1e-8,
    help="AdamW epsilonï¼›ä»…åœ¨ --optimizer adamw æ—¶ä½¿ç”¨"
)
parser.add_argument(
    "--cos",
    action="store_true",
    help="ä½¿ç”¨ cosine å­¦ä¹ çŽ‡è¡°å‡"
)
parser.add_argument(
    "--schedules",
    default=[25, 50, 75],
    nargs="*",
    type=int,
    help="è‹¥ä¸ç”¨ cosineï¼Œåˆ™ä½¿ç”¨ multi-step milestones"
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="éšæœºç§å­"
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

# ---------------- Weighted Samplerï¼ˆtrain onlyï¼‰ ----------------
parser.add_argument(
    "--use_weighted_sampler",
    action="store_true",
    help="æ˜¯å¦å¯¹è®­ç»ƒé›†å¯ç”¨ WeightedRandomSampler"
)
parser.add_argument(
    "--sampler_tier",
    type=str,
    default=None,
    choices=["tier1", "tier2", "tier3"],
    help="weighted sampler æŒ‰å“ªä¸ª tier é‡é‡‡æ ·ï¼›é»˜è®¤è·Ÿéš tier_mode"
)
parser.add_argument(
    "--sampler_mode",
    type=str,
    default="sqrt_inv",
    choices=["inv", "sqrt_inv"],
    help="weighted sampler æƒé‡æ–¹å¼"
)

# ---------------- Weighted CE ----------------
parser.add_argument(
    "--use_weighted_ce",
    action="store_true",
    help="å¯ç”¨ Weighted Cross-Entropy"
)
parser.add_argument(
    "--weight_method",
    type=str,
    default="class_balanced",
    choices=["class_balanced", "inv_freq"],
    help="ç±»åˆ«æƒé‡è®¡ç®—æ–¹æ³•"
)
parser.add_argument(
    "--cb_beta",
    type=float,
    default=0.999,
    help="class_balanced æƒé‡ä¸­çš„ beta"
)
parser.add_argument(
    "--weight_normalize_mean",
    action="store_true",
    help="æ˜¯å¦å°†ç±»åˆ«æƒé‡å½’ä¸€åŒ–åˆ°å‡å€¼=1"
)

# ---------------- Focal Loss ----------------
parser.add_argument(
    "--use_focal",
    action="store_true",
    help="å¯ç”¨ Focal Loss"
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
    help="Focal Loss æ˜¯å¦ä½¿ç”¨ alpha ç±»æƒé‡"
)

# ---------------- AMP ----------------
parser.add_argument(
    "--enable_amp",
    action="store_true",
    help="æ˜¯å¦å¯ç”¨ AMP æ··åˆç²¾åº¦è®­ç»ƒ"
)

# ---------------- é¢„è®­ç»ƒ / å¾®è°ƒç›¸å…³ ----------------
parser.add_argument(
    "--pretrained_weight_paths",
    nargs="*",
    default=[],
    help=(
        "è®­ç»ƒæ¨¡å¼ä¸‹å¯ä¼ å…¥å¤šä¸ªé¢„è®­ç»ƒæƒé‡è·¯å¾„ã€‚"
        "æœ¬ç‰ˆæœ¬æŒ‰è¾“å…¥é¡ºåºä¸Ž train_manifest(s) / val_manifest(s) å¯¹é½ã€‚"
    ),
)
parser.add_argument(
    "--include_scratch_baseline",
    action="store_true",
    help="å½“æä¾›å¤šä¸ªé¢„è®­ç»ƒæƒé‡æ—¶ï¼Œæ˜¯å¦é¢å¤–å†è·‘ scratch baseline"
)
parser.add_argument(
    "--finetune_mode",
    type=str,
    default="full",
    choices=["full", "head_only"],
    help="full: å…¨éƒ¨å¾®è°ƒï¼›head_only: åªè®­ç»ƒåˆ†ç±»å¤´"
)

# é»˜è®¤ä¸¢æŽ‰å¯¹æ¯”å­¦ä¹ å¤´ / projector / predictor ç­‰
parser.add_argument(
    "--keep_pretrained_head",
    action="store_true",
    help="é»˜è®¤ä¼šä¸¢æŽ‰é¢„è®­ç»ƒä¸­çš„ fc/head/projector/predictor ç­‰å¤´éƒ¨å‚æ•°ï¼›ä¼ è¯¥å¼€å…³å¯ä¿ç•™",
)
parser.add_argument(
    "--pretrained_strict",
    action="store_true",
    help="é¢„è®­ç»ƒåŠ è½½æ—¶æ˜¯å¦ strict=Trueï¼›é»˜è®¤ strict=Falseï¼Œæ›´é€‚åˆå¾®è°ƒ",
)

# ---------------- åŒºåˆ†å­¦ä¹ çŽ‡ï¼ˆdiscriminative LRï¼‰ ----------------
parser.add_argument(
    "--use_discriminative_lr",
    action="store_true",
    help="æ˜¯å¦ä¸º backbone å’Œåˆ†ç±»å¤´ä½¿ç”¨ä¸åŒå­¦ä¹ çŽ‡ï¼ˆä»…å¯¹ full finetune æœ‰æ„ä¹‰ï¼‰",
)
parser.add_argument(
    "--backbone_learning_rate",
    type=float,
    default=None,
    help="backbone å­¦ä¹ çŽ‡ï¼›ä¸ºç©ºåˆ™å›žé€€åˆ° learning_rate",
)
parser.add_argument(
    "--head_learning_rate",
    type=float,
    default=None,
    help="åˆ†ç±»å¤´å­¦ä¹ çŽ‡ï¼›ä¸ºç©ºåˆ™å›žé€€åˆ° learning_rate",
)

# ---------------- checkpoint ä¿å­˜ç­–ç•¥ ----------------
parser.add_argument(
    "--save_period",
    type=int,
    default=20,
    help="æ¯éš”å¤šå°‘ä¸ª epoch ä¿å­˜ä¸€ä¸ªå‘¨æœŸæ€§ checkpoint"
)
parser.add_argument(
    "--best_after_epoch",
    type=int,
    default=0,
    help="åªåœ¨ epoch >= è¯¥å€¼ä¹‹åŽä¿å­˜ best checkpoint"
)

# ---------------- æ‰¹é‡æµ‹è¯• ----------------
parser.add_argument(
    "--test_weight_paths",
    nargs="*",
    default=[],
    help=(
        "æµ‹è¯•æ¨¡å¼ä¸‹å¯ä¼ å…¥å¤šä¸ªå·²è®­ç»ƒæƒé‡è·¯å¾„ã€‚"
        "æœ¬ç‰ˆæœ¬æŒ‰è¾“å…¥é¡ºåºä¸Ž test_manifest å¯¹é½ã€‚"
    ),
)
parser.add_argument(
    "--test_results_csv",
    type=str,
    default=None,
    help="æµ‹è¯•ç»“æžœ CSV ä¿å­˜è·¯å¾„ï¼›ä¸ºç©ºåˆ™é»˜è®¤å­˜åˆ° save_path/test_results.csv",
)

args = parser.parse_args()


# ============================================================
# å…¨å±€ AMP / device é…ç½®
# ============================================================
os.makedirs(args.save_path, exist_ok=True)
has_cuda = torch.cuda.is_available()
use_bf16 = torch.cuda.is_bf16_supported() if has_cuda else False
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
torch.backends.cudnn.benchmark = True


# ============================================================
# 1) å·¥å…·å‡½æ•°
# ============================================================
def seed_everything(s: int = 42):
    """
    å°½é‡æé«˜å¯å¤çŽ°æ€§ã€‚

    æ³¨æ„ï¼š
    - è¿™ä¼šè®© cudnn.deterministic=True
    - å¯èƒ½ä¼šæ¯” benchmark æ¨¡å¼æ›´æ…¢
    - è‹¥ DataLoader å†…éƒ¨è¿˜æœ‰å¤æ‚éšæœºå¢žå¼ºï¼Œä¸¥æ ¼é€ä½å¤çŽ°ä»å¯èƒ½å— worker éšæœºæ€§å½±å“
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
    å°†è·¯å¾„ stem / ä»»æ„å­—ç¬¦ä¸²è½¬æ¢æˆæ›´é€‚åˆä½œä¸ºç›®å½•åæˆ–æ–‡ä»¶åç‰‡æ®µçš„å½¢å¼ã€‚
    """
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name))
    name = name.strip("._-")
    return name or "run"



def build_pretrained_src_tag(pretrained_path: str | None, args) -> str:
    """
    æ ¹æ®å¼€å…³ï¼Œä»Žé¢„è®­ç»ƒæƒé‡è·¯å¾„æž„é€ æ›´ç¨³å®šã€æ›´ä¸æ˜“å†²çªçš„ src tagã€‚

    mode:
    - legacy:
        parent.name + stem
        ä¾‹å¦‚: proto_1_checkpoint_0200

    - last_k_dirs:
        å–è·¯å¾„æœ€åŽ k å±‚ç›®å½• + stem
        ä¾‹å¦‚:
        signal_emg/ablation_contrastive_rel/prem_0.5/proto_1/checkpoint_0200.pth
        -> signal_emg_ablation_contrastive_rel_prem_0.5_proto_1_checkpoint_0200

    - relative_to_anchor:
        ä»ŽæŸä¸ªé”šç‚¹ç›®å½•ä¹‹åŽå¼€å§‹å–ç›¸å¯¹è·¯å¾„ + stem
        ä¾‹å¦‚ anchor='J_test' æ—¶:
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
                "pretrained_tag_mode='relative_to_anchor' æ—¶ï¼Œå¿…é¡»æä¾› --pretrained_tag_anchor"
            )

        parts = list(p.parts)
        try:
            anchor_idx = parts.index(anchor)
        except ValueError:
            raise ValueError(
                f"anchor '{anchor}' ä¸åœ¨ pretrained path ä¸­ï¼š{pretrained_path}"
            )

        rel_parts = [sanitize_name(x) for x in parts[anchor_idx + 1:-1]]
        if len(rel_parts) == 0:
            return sanitize_name(p.parent.name)
        return "_".join(rel_parts)

    raise ValueError(f"Unknown pretrained_tag_mode: {mode}")



def compact_manifest_stem(path_or_name: str | None) -> str:
    """
    å°† manifest æ–‡ä»¶ååŽ‹ç¼©æˆæ›´çŸ­çš„å®žéªŒæ ‡ç­¾ã€‚

    ä¾‹å¦‚ï¼š
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
    å…¼å®¹ä¸¤ç§ä¼ æ³•ï¼š
    1) ç›´æŽ¥ä¼ æ–‡ä»¶åï¼Œä¾‹å¦‚ train_manifest.jsonl
    2) ä¼ ç»å¯¹è·¯å¾„æˆ–ç›¸å¯¹è·¯å¾„

    å¯¹äºŽç»å¯¹è·¯å¾„ï¼Œå¦‚æžœå®ƒä½äºŽ dataset_root å†…éƒ¨ï¼Œåˆ™è½¬æ¢æˆç›¸å¯¹è·¯å¾„ï¼Œ
    è¿™æ ·æ›´å…¼å®¹å¾ˆå¤š map-style dataset builder çš„å®žçŽ°æ–¹å¼ã€‚
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
            # è‹¥ç»å¯¹è·¯å¾„ä¸åœ¨ dataset_root ä¸‹ï¼Œåˆ™ç›´æŽ¥åŽŸæ ·è¿”å›ž
            return str(manifest_path)

    return str(manifest_path)


def combine_manifest_args(single_manifest: str | None, multi_manifests: list[str] | None) -> list[str]:
    """
    å°†å•ä¸ª manifest å‚æ•°ä¸Žå¤šä¸ª manifest å‚æ•°åˆå¹¶æˆä¸€ä¸ªåŽ»é‡åŽçš„æœ‰åºåˆ—è¡¨ã€‚

    è®¾è®¡ç›®çš„ï¼š
    - ä¿ç•™ --train_manifest / --val_manifest çš„æ—§æŽ¥å£
    - åŒæ—¶æ”¯æŒæ–°å¢žçš„ --train_manifests / --val_manifests
    - ä¿è¯é¡ºåºç¨³å®šï¼šsingle_manifest ä¼šæŽ’åœ¨æœ€å‰é¢ï¼Œmulti_manifests ä¾æ¬¡è¿½åŠ 
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
    å°† argparse ä¼ å…¥çš„åˆ—è¡¨å‚æ•°æ¸…ç†æˆæœ‰åºå­—ç¬¦ä¸²åˆ—è¡¨ã€‚
    - åŽ»æŽ‰ç©ºå­—ç¬¦ä¸²
    - åŽ»æŽ‰çº¯ç©ºç™½
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
    ä»Ž label_map.json æž„å»º:
        class_id -> class_name
    çš„åå‘æ˜ å°„ã€‚

    ä¾‹å¦‚åŽŸå§‹ label_map[tier_mode] å¯èƒ½æ˜¯:
        {
            "adjust": 0,
            "take": 1,
            ...
        }

    è¿™é‡Œè½¬æ¢æˆ:
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
    å°† numpy / torch æ ‡é‡è½¬æ¢æˆ Python floatï¼Œä¾¿äºŽ json ä¿å­˜ã€‚
    """
    if x is None:
        return None
    return float(x)


def round_metric_dict(d: dict, ndigits: int = 4) -> dict:
    """
    å°† per-class metric dict ä¸­çš„ float ä¿ç•™å›ºå®šå°æ•°ä½ã€‚
    None ä¿æŒä¸º Noneã€‚
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
    è®¡ç®—å¤šåˆ†ç±»æŒ‡æ ‡ã€‚

    è¿”å›žå†…å®¹åŒ…æ‹¬ï¼š
    1) overall accuracy
    2) balanced accuracy
       - ç­‰äºŽæ‰€æœ‰ present classes çš„ per-class recall å¹³å‡
       - present classes æŒ‡åœ¨å½“å‰ split ä¸­çœŸå®žæ ·æœ¬æ•° > 0 çš„ç±»åˆ«
    3) macro-F1
       - å…ˆè®¡ç®—æ¯ä¸ª present class çš„ F1ï¼Œå†å–å¹³å‡
    4) per-class accuracy
       - å¯¹æ¯ä¸ªç±»åˆ« c:
         per_class_acc[c] = TP_c / (TP_c + FN_c)
       - ä¹Ÿå°±æ˜¯è¯¥ç±»åˆ«çš„ recall
    5) per-class support
       - æ¯ä¸ªç±»åˆ«åœ¨å½“å‰ split ä¸­çš„çœŸå®žæ ·æœ¬æ•°

    æ³¨æ„ï¼š
    - å¦‚æžœæŸä¸ªç±»åˆ«åœ¨å½“å‰ split ä¸­ support=0ï¼Œåˆ™å®ƒä¸å‚ä¸Ž balanced_acc å’Œ macro_f1 å¹³å‡ã€‚
    - è¿™æ¯”æŠŠç¼ºå¤±ç±»åˆ«å¼ºè¡Œè®°ä¸º 0 æ›´åˆç†ï¼Œå› ä¸ºéªŒè¯é›†æˆ–æµ‹è¯•é›†å¯èƒ½ä¸åŒ…å«æ‰€æœ‰ç±»åˆ«ã€‚
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
    å°† acc / balanced_acc / macro_f1 æ ¼å¼åŒ–æˆä¸€æ®µæ—¥å¿—æ–‡æœ¬ã€‚
    """
    return (
        f"{prefix}_acc: {metrics['acc']:.4f}, "
        f"{prefix}_balanced_acc: {metrics['balanced_acc']:.4f}, "
        f"{prefix}_macro_f1: {metrics['macro_f1']:.4f}"
    )


def align_required_sequence(values: list[str], target_len: int, field_name: str) -> list[str]:
    """
    å°†â€œå¿…é¡»å­˜åœ¨â€çš„å­—ç¬¦ä¸²åˆ—è¡¨å¯¹é½åˆ° target_lenã€‚

    å…è®¸ä¸¤ç§é•¿åº¦ï¼š
    1) len(values) == 1
       -> å¹¿æ’­ç»™æ‰€æœ‰å®žéªŒ
    2) len(values) == target_len
       -> æŒ‰é¡ºåºä¸€ä¸€å¯¹åº”

    å…¶ä»–é•¿åº¦ç›´æŽ¥æŠ¥é”™ã€‚

    ä¾‹å¦‚ï¼š
        values = ["train_a.jsonl"], target_len = 3
        -> ["train_a.jsonl", "train_a.jsonl", "train_a.jsonl"]

        values = ["a", "b", "c"], target_len = 3
        -> ["a", "b", "c"]
    """
    if len(values) == 0:
        raise ValueError(f"{field_name} ä¸èƒ½ä¸ºç©ºã€‚")

    if len(values) == 1:
        return values * target_len

    if len(values) == target_len:
        return list(values)

    raise ValueError(
        f"{field_name} çš„æ•°é‡ä¸åˆæ³•ï¼šlen={len(values)}ï¼Œç›®æ ‡å®žéªŒæ•°={target_len}ã€‚\n"
        f"å…è®¸çš„æƒ…å†µåªæœ‰ï¼š1ï¼ˆå¹¿æ’­ï¼‰æˆ– {target_len}ï¼ˆé€é¡¹é¡ºåºåŒ¹é…ï¼‰ã€‚"
    )


def align_optional_sequence(values: list[str], target_len: int, field_name: str) -> list[str | None]:
    """
    å°†â€œå¯ä¸ºç©ºâ€çš„å­—ç¬¦ä¸²åˆ—è¡¨å¯¹é½åˆ° target_lenã€‚

    å…è®¸ä¸‰ç§é•¿åº¦ï¼š
    1) len(values) == 0
       -> å…¨éƒ¨ç½®ä¸º None
    2) len(values) == 1
       -> å¹¿æ’­ç»™æ‰€æœ‰å®žéªŒ
    3) len(values) == target_len
       -> æŒ‰é¡ºåºä¸€ä¸€å¯¹åº”
    """
    if len(values) == 0:
        return [None] * target_len

    if len(values) == 1:
        return values * target_len

    if len(values) == target_len:
        return list(values)

    raise ValueError(
        f"{field_name} çš„æ•°é‡ä¸åˆæ³•ï¼šlen={len(values)}ï¼Œç›®æ ‡å®žéªŒæ•°={target_len}ã€‚\n"
        f"å…è®¸çš„æƒ…å†µåªæœ‰ï¼š0ï¼ˆå…¨éƒ¨ Noneï¼‰ã€1ï¼ˆå¹¿æ’­ï¼‰æˆ– {target_len}ï¼ˆé€é¡¹é¡ºåºåŒ¹é…ï¼‰ã€‚"
    )


def build_train_manifest_list(args) -> list[str]:
    """
    ç»Ÿä¸€ç”Ÿæˆè®­ç»ƒ manifest çš„æœ‰åºåˆ—è¡¨ã€‚
    """
    return combine_manifest_args(args.train_manifest, args.train_manifests)


def build_val_manifest_list(args) -> list[str]:
    """
    ç»Ÿä¸€ç”ŸæˆéªŒè¯ manifest çš„æœ‰åºåˆ—è¡¨ã€‚
    """
    return combine_manifest_args(args.val_manifest, args.val_manifests)


def build_test_manifest_list(args) -> list[str]:
    """
    ç»Ÿä¸€ç”Ÿæˆæµ‹è¯• manifest çš„æœ‰åºåˆ—è¡¨ã€‚

    æ³¨æ„ï¼š
    - æœ¬ç‰ˆæœ¬ä¸­ --test_manifest å·²ç»å¯ä»¥æŽ¥æ”¶å¤šä¸ªå€¼
    - å› æ­¤è¿™é‡Œä¸å†åŒºåˆ† single / multi ä¸¤å¥—æŽ¥å£
    """
    return normalize_string_list(args.test_manifest)


def validate_args(args):
    """
    æ ¹æ® run_mode åšæ¡ä»¶æ£€æŸ¥ã€‚

    æœ¬ç‰ˆæœ¬çš„é‡ç‚¹ï¼š
    - ä¸å†æ£€æŸ¥ imbalance_XXX ç­‰æ–‡ä»¶åæ ‡ç­¾
    - æ”¹ä¸ºæ£€æŸ¥â€œé¡ºåºå¯¹é½ / å¹¿æ’­â€æ˜¯å¦åˆæ³•
    """
        # ---------------- RGB spatial augmentation å‚æ•°æ£€æŸ¥ ----------------
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
            raise ValueError("run_mode=train æ—¶ï¼Œå¿…é¡»æä¾› --train_manifest æˆ– --train_manifests")

        # è‹¥æ²¡æœ‰é¢„è®­ç»ƒæƒé‡ï¼Œåˆ™å®žéªŒæ•°ç”± train manifests å†³å®š
        if len(pretrained_list) == 0:
            num_main_runs = len(train_manifest_list)

            # è®­ç»ƒé›†åœ¨è¿™ç§æƒ…å†µä¸‹ä¸éœ€è¦å¹¿æ’­ï¼Œç›´æŽ¥ä¸€æ¡ train manifest å¯¹åº”ä¸€ä¸ªå®žéªŒ
            # è¿™é‡Œä¸»è¦æ£€æŸ¥ val æ˜¯å¦èƒ½ä¸Ž num_main_runs å¯¹é½
            _ = align_optional_sequence(val_manifest_list, num_main_runs, "val manifest(s)")

        # è‹¥æœ‰é¢„è®­ç»ƒæƒé‡ï¼Œåˆ™ train / val éƒ½è¦èƒ½å¯¹é½åˆ° pretrained æ•°é‡
        else:
            num_main_runs = len(pretrained_list)
            _ = align_required_sequence(train_manifest_list, num_main_runs, "train manifest(s)")
            _ = align_optional_sequence(val_manifest_list, num_main_runs, "val manifest(s)")

        if args.use_discriminative_lr and args.finetune_mode == "head_only":
            print("[warning] finetune_mode=head_only æ—¶ï¼Œuse_discriminative_lr æ²¡æœ‰å®žé™…æ„ä¹‰ï¼Œå°†åªä½¿ç”¨åˆ†ç±»å¤´å­¦ä¹ çŽ‡ã€‚")

    elif args.run_mode == "test":
        test_manifest_list = build_test_manifest_list(args)
        test_weight_list = normalize_string_list(args.test_weight_paths)

        if len(test_manifest_list) == 0:
            raise ValueError("run_mode=test æ—¶ï¼Œå¿…é¡»æä¾›è‡³å°‘ä¸€ä¸ª --test_manifest")
        if len(test_weight_list) == 0:
            raise ValueError("run_mode=test æ—¶ï¼Œå¿…é¡»è‡³å°‘æä¾›ä¸€ä¸ª --test_weight_paths")

        # æµ‹è¯• manifest æ”¯æŒï¼š
        # - 1 ä¸ªï¼šå¹¿æ’­
        # - ä¸Ž test weights æ•°é‡ç›¸åŒï¼šé€é¡¹åŒ¹é…
        _ = align_required_sequence(test_manifest_list, len(test_weight_list), "test manifest(s)")


# ============================================================
# 2) å­¦ä¹ çŽ‡è°ƒåº¦ï¼ˆæ”¯æŒå¤š param groupï¼‰
# ============================================================
def compute_lr_factor(epoch: int, args) -> float:
    """
    è®¡ç®—ç›¸å¯¹äºŽâ€œåˆå§‹å­¦ä¹ çŽ‡â€çš„ç¼©æ”¾æ¯”ä¾‹ã€‚

    è¿™æ ·è®¾è®¡çš„åŽŸå› ï¼š
    - å•å­¦ä¹ çŽ‡æ—¶ï¼šå½“å‰ lr = learning_rate * factor
    - åŒå­¦ä¹ çŽ‡æ—¶ï¼š
        backbone lr = backbone_initial_lr * factor
        head lr     = head_initial_lr * factor

    è¿™æ ·å°±ä¸ä¼šåœ¨ç¬¬ä¸€ä¸ª epoch æŠŠåŒå­¦ä¹ çŽ‡è¦†ç›–æˆåŒä¸€ä¸ªå€¼ã€‚
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
    å¯¹æ‰€æœ‰ param group æŒ‰å„è‡ª initial_lr æˆæ¯”ä¾‹è¡°å‡ã€‚

    è¿”å›žï¼š
        ä¸€ä¸ªå­—å…¸ï¼Œæ–¹ä¾¿å†™æ—¥å¿—ï¼Œæ¯”å¦‚ï¼š
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
    å°†å¤šä¸ª param group çš„å­¦ä¹ çŽ‡æ ¼å¼åŒ–æˆæ—¥å¿—å­—ç¬¦ä¸²ã€‚
    """
    parts = [f"{k}: {v:.6f}" for k, v in lr_dict.items()]
    return ", ".join(parts)


# ============================================================
# 3) ç±»åˆ«æƒé‡ç›¸å…³
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
    æ ¹æ®ç»™å®šç±»åˆ«è®¡æ•°è¡¨æž„é€  loss çš„ç±»åˆ«æƒé‡ã€‚

    counts:
        å¯ä»¥æ˜¯ list[int] æˆ– dict[int, int]
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
    æ ¹æ® map-style è®­ç»ƒæ•°æ®é›†å¯¹è±¡è‡ªåŠ¨ç»Ÿè®¡ç±»åˆ«æ ·æœ¬æ•°ã€‚

    è¿™é‡Œç›´æŽ¥è¯»å– dataset.records ä¸­ manifest æ ‡ç­¾å­—æ®µï¼Œ
    ä¸èµ° __getitem__ï¼Œå› æ­¤ä¸ä¼šçœŸçš„åŠ è½½è§†é¢‘å¸§æ–‡ä»¶ã€‚
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
# 4) map-style dataset / loader æž„å»º
# ============================================================
def _pack_cli_float_scalar_or_pair(values, arg_name: str):
    """
    å°† argparse è¯»å…¥çš„ float åˆ—è¡¨æ•´ç†ä¸ºï¼š
    - 1 ä¸ªå€¼ -> float
    - 2 ä¸ªå€¼ -> (float, float)

    è¿™é‡Œç”¨äºŽ drift_maxï¼Œä¸¥æ ¼é™åˆ¶åªèƒ½ä¼  1 ä¸ªæˆ– 2 ä¸ªå€¼ã€‚
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
    å°† argparse è¯»å…¥çš„ int åˆ—è¡¨æ•´ç†ä¸ºï¼š
    - 1 ä¸ªå€¼ -> int
    - å¤šä¸ªå€¼ -> list[int]

    è¿™é‡Œç”¨äºŽ drift_n_pointsã€‚
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
    æž„å»º map-style dataset configã€‚

    è¿™æ˜¯åˆ†ç±»è„šæœ¬ï¼Œä¸åš two-view å¯¹æ¯”å­¦ä¹ ï¼š
    - rgb_two_views = False
    - mindrove_two_views = False
    """
    rgb_hw = (args.rgb_size, args.rgb_size)
    depth_hw = (args.depth_size, args.depth_size)
    use_modalities = (args.use_modality,)

    # ------------------------------------------------------------
    # è®­ç»ƒå¢žå¼ºæ€»å¼€å…³
    # ------------------------------------------------------------
    # åŽŸè„šæœ¬ä¸­ RGB çš„ --no-rgb_apply_spatial_aug åªä¼šå…³é—­
    # flip / jitter / gray / blurï¼Œä½† RandomResizedCrop ä»ç„¶å­˜åœ¨ã€‚
    # ä¸ºäº†çœŸæ­£â€œå…³é—­è®­ç»ƒå¢žå¼ºâ€ï¼Œè¿™é‡Œæ–°å¢ž --disable_train_augmentationï¼š
    #   1) åªåœ¨ is_train=True æ—¶ç”Ÿæ•ˆï¼›
    #   2) RGB: RRC é€€åŒ–ä¸ºä¸è£å‰ªï¼Œæ‰€æœ‰éšæœºæ¦‚çŽ‡ç½® 0ï¼›
    #   3) MindRove: æ ·æœ¬çº§å¢žå¼ºæ•´ä½“å…³é—­ã€‚
    # éªŒè¯/æµ‹è¯•é›†ä»ç„¶ä¾èµ– dataloader çš„ is_train=False è·¯å¾„ï¼Œä¸å—å½±å“ã€‚
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
        rgb_camera_id=args.rgb_camera_id,
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
    æž„å»ºå•ä¸ª dataset + loaderã€‚

    è¯´æ˜Žï¼š
    - train / val / test éƒ½èµ°è¿™ä¸ªç»Ÿä¸€å…¥å£
    - train å’Œ val/test çš„åŒºåˆ«ä¸»è¦ç”± is_train æŽ§åˆ¶
    - éªŒè¯/æµ‹è¯•é»˜è®¤ä¸ä½¿ç”¨ weighted samplerï¼Œä¸æ‰“ä¹±ï¼Œä¸ drop_last
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
    æ ¹æ®æ˜¾å¼ç»™å®šçš„ train_manifest / val_manifest æž„å»º loadersã€‚

    ç”±äºŽæœ¬ç‰ˆæœ¬çš„è®­ç»ƒæºçŽ°åœ¨æ˜¯â€œæŒ‰é¡ºåºåˆ†é…â€å‡ºæ¥çš„ï¼Œæ‰€ä»¥è¿™é‡Œä¿æŒç®€å•ï¼š
    ç»™å®šä¸€ä¸ªæ˜Žç¡®çš„ train_manifest å’Œå¯é€‰ val_manifestï¼Œç›´æŽ¥æž„å»ºå¯¹åº” DataLoaderã€‚
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
    å‘åŽå…¼å®¹æŽ¥å£ï¼šä»ç„¶å…è®¸æ—§ä»£ç åªä¾èµ– args.train_manifest / args.val_manifestã€‚
    """
    return prepare_train_val_loaders_for_manifests(
        args=args,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
    )


def prepare_test_loader_for_manifest(args, test_manifest: str):
    """
    æž„å»ºæŸä¸€ä¸ª test manifest å¯¹åº”çš„æµ‹è¯•é›† loaderã€‚

    æœ¬ç‰ˆæœ¬æµ‹è¯•æ¨¡å¼æ”¯æŒå¤šä¸ª test manifestï¼Œå› æ­¤ä¸èƒ½å†åªä¾èµ– args.test_manifest å…¨å±€å”¯ä¸€å€¼ã€‚
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
# 5) æ¨¡åž‹æž„å»º / é¢„è®­ç»ƒæƒé‡åŠ è½½ / å†»ç»“ç­–ç•¥
# ============================================================
def prepare_model(args):
    """
    æ ¹æ® use_modality æž„å»ºåˆ†ç±»æ¨¡åž‹ï¼š
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
    åŽ»æŽ‰å¸¸è§å°è£…å‰ç¼€ï¼Œä»¥æé«˜é¢„è®­ç»ƒå…¼å®¹æ€§ã€‚

    å…¸åž‹å‰ç¼€åŒ…æ‹¬ï¼š
    - module.
    - model.
    - backbone.
    - encoder.
    - encoder_q.
    - base_encoder.
    - online_encoder.
    ç­‰ç­‰

    è¿™é‡Œä½¿ç”¨â€œå¾ªçŽ¯å‰¥ç¦»â€çš„æ–¹å¼ï¼Œé¿å…å‡ºçŽ°å¤šå±‚åŒ…è£¹æ—¶åªåŽ»æŽ‰ä¸€å±‚å‰ç¼€çš„é—®é¢˜ã€‚
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
    ä»Žä¸åŒæ ¼å¼çš„ checkpoint ä¸­å–å‡ºçœŸæ­£çš„ state_dictã€‚

    å¸¸è§æƒ…å†µï¼š
    - ç›´æŽ¥å°±æ˜¯ state_dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"model": ...}
    - å…¶ä»–è‡ªå®šä¹‰ä¿å­˜å½¢å¼
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

    # å¦‚æžœæœ¬èº«å·²ç»åƒ state_dictï¼škey -> Tensor
    tensor_like = 0
    for k, v in ckpt_obj.items():
        if isinstance(k, str) and torch.is_tensor(v):
            tensor_like += 1
    if tensor_like > 0:
        return ckpt_obj

    raise ValueError("Unable to find a valid state_dict inside checkpoint.")


def should_drop_pretrained_key(key: str) -> bool:
    """
    åˆ¤æ–­æŸä¸ª key æ˜¯å¦å±žäºŽåº”è¯¥åœ¨â€œå¾®è°ƒåŠ è½½é¢„è®­ç»ƒâ€æ—¶ä¸¢æŽ‰çš„å¤´éƒ¨å‚æ•°ã€‚

    å¯¹æ¯”å­¦ä¹ é¢„è®­ç»ƒé€šå¸¸ä¼šå¸¦æœ‰ï¼š
    - projector
    - predictor
    - mlp head
    - æ—§çš„ fc / classifier

    è¿™äº›å±‚é€šå¸¸ä¸Žå½“å‰ä¸‹æ¸¸åˆ†ç±»ä»»åŠ¡ä¸å…¼å®¹ï¼Œå› æ­¤é»˜è®¤å»ºè®®ä¸¢å¼ƒã€‚
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
    å¯¹åŠ è½½åˆ°çš„ state_dict åšä¸‰ç±»æ¸…æ´—ï¼š

    1) ç»Ÿä¸€ key å‰ç¼€
    2) å¯é€‰ä¸¢å¼ƒé¢„è®­ç»ƒå¤´ / å¯¹æ¯”å¤´
    3) åªä¿ç•™â€œå½“å‰æ¨¡åž‹ä¸­å­˜åœ¨ä¸” shape ä¸€è‡´â€çš„å‚æ•°

    è¿”å›žï¼š
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
    ç”¨äºŽâ€œè®­ç»ƒå‰åŠ è½½é¢„è®­ç»ƒ backbone æƒé‡â€ã€‚

    ä¸Žæµ‹è¯•åŠ è½½ä¸åŒç‚¹ï¼š
    - è¿™é‡Œé»˜è®¤ drop_pretrained_head=True
    - æ›´ç¬¦åˆå¯¹æ¯”å­¦ä¹ é¢„è®­ç»ƒ -> ä¸‹æ¸¸åˆ†ç±»å¾®è°ƒçš„åœºæ™¯
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
    ç”¨äºŽâ€œæµ‹è¯• / è¯„ä¼°æ—¶åŠ è½½å·²è®­ç»ƒå¥½çš„åˆ†ç±»æ¨¡åž‹æƒé‡â€ã€‚

    è¿™é‡Œä¸ä¸»åŠ¨ä¸¢å¼ƒå¤´éƒ¨ï¼Œå› ä¸ºæµ‹è¯•æ—¶éœ€è¦å®Œæ•´çš„åˆ†ç±»æ¨¡åž‹ã€‚
    ä½†ä»ç„¶ä¼šï¼š
    - åŽ»å¸¸è§å‰ç¼€
    - åªåŠ è½½å­˜åœ¨ä¸” shape ä¸€è‡´çš„å‚æ•°
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
    é…ç½®å‚æ•°å†»ç»“ç­–ç•¥ã€‚

    full:
        å…¨éƒ¨å‚æ•°éƒ½å¯è®­ç»ƒ

    head_only:
        åªè®­ç»ƒ model.fcï¼Œå…¶ä»–å…¨éƒ¨å†»ç»“
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
    æ ¹æ®å¾®è°ƒæ¨¡å¼å’ŒåŒå­¦ä¹ çŽ‡è®¾ç½®ï¼Œæž„å»ºä¼˜åŒ–å™¨ã€‚

    æ”¯æŒä¸‰ç§å…¸åž‹æƒ…å†µï¼š
    1) full + å•å­¦ä¹ çŽ‡
    2) head_only
    3) full + discriminative lrï¼ˆbackbone / head ä¸¤ç»„ lrï¼‰

    æ–°å¢žï¼š
    - args.optimizer='sgd'   : ä½¿ç”¨ SGD(momentum, weight_decay)
    - args.optimizer='adamw' : ä½¿ç”¨ AdamW(betas, eps, decoupled weight_decay)ï¼Œä¸ä½¿ç”¨ momentum

    æ³¨æ„ï¼šä¼˜åŒ–å™¨é€‰æ‹©å’Œå‚æ•°åˆ†ç»„æ˜¯è§£è€¦çš„ã€‚ä¹Ÿå°±æ˜¯è¯´ï¼Œ
    head_only / full / discriminative lr å…ˆå†³å®šè®­ç»ƒå“ªäº›å‚æ•°ä»¥åŠæ¯ç»„ lrï¼Œ
    ç„¶åŽå†ç”± args.optimizer å†³å®šç”¨ SGD è¿˜æ˜¯ AdamW æ›´æ–°è¿™äº›å‚æ•°ã€‚

    è¿”å›žï¼š
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

    # è®°å½•ä¼˜åŒ–å™¨åç§°ï¼Œæ–¹ä¾¿åŽç»­ summary.csv / config.json ä¸­åŒºåˆ†å®žéªŒã€‚
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
# 6) ä»Ž batch(dict) ä¸­æŠ½å– inputs / labels / ids
# ============================================================
MINDROVE_SIGNAL_CHANNELS = {
    "emg": 8,
    "imu": 6,
}


def build_mindrove_input_keys(args) -> list[str]:
    """
    æ ¹æ®å‘½ä»¤è¡Œé…ç½®ï¼Œç¡®å®š MindRove è¾“å…¥åœ¨ batch["mindrove"] ä¸­åº”è¯¥æŒ‰ä»€ä¹ˆé¡ºåºå–å‡ºå¹¶æ‹¼æŽ¥ã€‚
    è¿™ä¸ªé¡ºåºä¸€æ—¦å®šä¸‹ï¼Œè®­ç»ƒ / æµ‹è¯• / å¾®è°ƒå¿…é¡»ä¿æŒä¸€è‡´ã€‚
    """
    if args.mindrove_merge_hands:
        # merge åŽçš„ key åªæœ‰ "emg" / "imu"
        return [sig for sig in args.mindrove_signals]

    keys = []
    for hand in args.mindrove_hands:
        for sig in args.mindrove_signals:
            keys.append(f"{hand}_{sig}")
    return keys


def compute_mindrove_in_channels(args) -> int:
    """
    è‡ªåŠ¨è®¡ç®— MindRove è¾“å…¥é€šé“æ•°ã€‚
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
    å°† batch["mindrove"] çš„ dict[str, Tensor[B,C,L]] æŒ‰å›ºå®šé¡ºåºåœ¨é€šé“ç»´æ‹¼æŽ¥æˆ [B,C,L]ã€‚
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
    è¿”å›žï¼š
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
    å°† [B,T,C,H,W] è½¬æˆ 3D CNN å¸¸ç”¨çš„ [B,C,T,H,W]ã€‚
    """
    if x_btchw.ndim != 5:
        raise ValueError(f"Expect 5D tensor [B,T,C,H,W], got shape={tuple(x_btchw.shape)}")
    return x_btchw.permute(0, 2, 1, 3, 4).contiguous()


def preprocess_rgb_already_normed(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    map-style loader å†…éƒ¨å·²ç»å®Œæˆ RGB çš„ç©ºé—´å¢žå¼ºã€ToDtype å’Œ Normalizeã€‚
    å› æ­¤è¿™é‡Œä¸è¦é‡å¤ Normalizeï¼Œåªç¡®ä¿ dtype=float32ã€‚
    """
    if x_btchw.dtype != torch.float32:
        x_btchw = x_btchw.to(torch.float32)
    return x_btchw


def preprocess_depth_to_float(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    Depth ä¸é¢å¤–åšå½’ä¸€åŒ–ï¼Œåªè½¬æˆ float32ã€‚
    """
    if x_btchw.dtype != torch.float32:
        x_btchw = x_btchw.to(torch.float32)
    return x_btchw

def _ensure_bcl(x_bcl: torch.Tensor) -> torch.Tensor:
    """
    ç¡®ä¿ MindRove è¾“å…¥ä¸º [B,C,L]ã€‚
    """
    if x_bcl.ndim != 3:
        raise ValueError(f"Expect 3D tensor [B,C,L], got shape={tuple(x_bcl.shape)}")
    return x_bcl.contiguous()


def preprocess_mindrove_to_float(x_bcl: torch.Tensor) -> torch.Tensor:
    """
    MindRove åœ¨è¿™é‡Œä¸å†åšé¢å¤–æ ‡å‡†åŒ–ã€‚

    è¯´æ˜Žï¼š
    - è‹¥å¯ç”¨äº†æ ‡å‡†åŒ–ï¼Œå·²åœ¨ dataloader å†…éƒ¨å®Œæˆ
    - è¿™é‡Œä»…ä¿è¯ dtype=float32ï¼Œé¿å…è®­ç»ƒè„šæœ¬å†æ¬¡é‡å¤å¤„ç†
    """
    if x_bcl.dtype != torch.float32:
        x_bcl = x_bcl.to(torch.float32)
    return x_bcl


def move_and_prepare_inputs(inputs, use_modality: str, device, args):
    """
    å°†ä¸åŒæ¨¡æ€è¾“å…¥ç»Ÿä¸€å˜æˆæ¨¡åž‹å¯ç›´æŽ¥æŽ¥å—çš„æ ¼å¼ã€‚
    è¿”å›žï¼š
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
# 7) loss æž„å»º
# ============================================================
def build_training_criterion(args, train_dataset, device):
    """
    æž„å»ºè®­ç»ƒæŸå¤±å‡½æ•°ã€‚

    è‹¥å¯ç”¨ï¼š
    - Weighted CE
    - Focal(alpha)

    åˆ™å…ˆä»Ž train_dataset.records è‡ªåŠ¨ç»Ÿè®¡ class countsã€‚
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

    # -------- æ»‘åŠ¨çª—å£è®¡æ—¶ï¼šä¾¿äºŽè§‚å¯Ÿç“¶é¢ˆ --------
    data_times = deque(maxlen=50)
    prep_times = deque(maxlen=50)
    gpu_times = deque(maxlen=50)
    log_times = deque(maxlen=50)

    end = time.perf_counter()
    pbar = tqdm.tqdm(loader, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        # 1) data æ—¶é—´
        t_data = time.perf_counter() - end
        data_times.append(t_data)

        # 2) å–æ•°æ®
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

        # 6) ç»Ÿè®¡
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

        # 8) è¿›åº¦æ¡æ˜¾ç¤º
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
    ç»Ÿä¸€éªŒè¯ / æµ‹è¯•å‡½æ•°ã€‚

    split_name ä»…ç”¨äºŽæ—¥å¿—æ˜¾ç¤ºï¼Œä¾‹å¦‚ï¼š
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
    ä¸“é—¨ç”¨äºŽ test çš„è¯¦ç»†è¯„ä¼°å‡½æ•°ã€‚

    åŠŸèƒ½ï¼š
    1) è®¡ç®—æ•´ä½“ test acc / test loss
    2) ä¸ºå½“å‰æƒé‡å•ç‹¬ä¿å­˜ä¸€ä¸ªé€æ ·æœ¬ CSV

    CSV æ¯è¡Œå¯¹åº”ä¸€ä¸ªæµ‹è¯•æ ·æœ¬ï¼ŒåŒ…å«ï¼š
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

        # å°½é‡ä¿ç•™ sample_name ä¸Ž original_key
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
# 9) checkpoint / é…ç½® / ç»“æžœä¿å­˜
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
    ä¿å­˜ checkpointã€‚

    å‘½åè§„åˆ™ï¼š
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
    ä¿å­˜æˆ–è¿½åŠ æ±‡æ€» CSVã€‚

    å‚æ•°
    ----
    csv_path : str
        ç›®æ ‡ csv è·¯å¾„
    rows : list[dict]
        è¦å†™å…¥çš„å¤šè¡Œæ•°æ®
    append : bool
        False: è¦†ç›–å†™å…¥
        True : è¿½åŠ å†™å…¥ï¼›è‹¥æ–‡ä»¶ä¸å­˜åœ¨åˆ™è‡ªåŠ¨å†™è¡¨å¤´ï¼Œè‹¥å·²å­˜åœ¨åˆ™åªè¿½åŠ å†…å®¹
    """
    if len(rows) == 0:
        return

    ensure_dir(os.path.dirname(csv_path) or ".")
    fieldnames = list(rows[0].keys())

    file_exists = os.path.isfile(csv_path)
    mode = "a" if append else "w"

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # åªæœ‰ä»¥ä¸‹ä¸¤ç§æƒ…å†µæ‰å†™è¡¨å¤´ï¼š
        # 1) è¦†ç›–å†™å…¥
        # 2) è¿½åŠ å†™å…¥ä½†æ–‡ä»¶åŽŸæœ¬ä¸å­˜åœ¨
        if (not append) or (not file_exists):
            writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 10) è®­ç»ƒå®žéªŒæºæž„é€ ï¼ˆæŒ‰è¾“å…¥é¡ºåºå¯¹é½ï¼‰
# ============================================================
def build_training_sources(args):
    """
    æŒ‰è¾“å…¥é¡ºåºæž„é€ è®­ç»ƒå®žéªŒæºã€‚

    æœ¬å‡½æ•°æ˜¯æœ¬ç‰ˆæœ¬æœ€å…³é”®çš„é€»è¾‘ä¹‹ä¸€ã€‚
    å®ƒå½»åº•æ›¿ä»£äº†æ—§ç‰ˆæœ¬åŸºäºŽæ–‡ä»¶åå†…å®¹è‡ªåŠ¨åŒ¹é…çš„æ–¹å¼ã€‚

    è¿”å›žæ ¼å¼ï¼š
        [
            {
                "pretrained_path": ... æˆ– None,
                "train_manifest": ...,
                "val_manifest": ... æˆ– None,
            },
            ...
        ]

    è§„åˆ™æ€»ç»“ï¼š
    ----------------------------------------
    A) æ²¡æœ‰é¢„è®­ç»ƒæƒé‡
       - å®žéªŒæ•° = train manifests æ•°é‡
       - æ¯ä¸ª train manifest å¯¹åº”ä¸€ä¸ªå®žéªŒ
       - val manifests å¯ 0/1/N ä¸ªï¼ˆN=å®žéªŒæ•°ï¼‰

    B) æœ‰é¢„è®­ç»ƒæƒé‡
       - å®žéªŒæ•° = pretrained_weight_paths æ•°é‡
       - train manifests å¯ 1/N ä¸ªï¼ˆN=å®žéªŒæ•°ï¼‰
       - val manifests å¯ 0/1/N ä¸ªï¼ˆN=å®žéªŒæ•°ï¼‰
       - æŒ‰é¡ºåºæˆ–å¹¿æ’­å¯¹é½

    C) include_scratch_baseline=True
       - ä¼šé¢å¤–è¿½åŠ  scratch å®žéªŒ
       - scratch å®žéªŒæŒ‰â€œä¸»å®žéªŒè§£æžåŽçš„ (train_manifest, val_manifest) ç»„åˆâ€åŽ»é‡åŽè¿½åŠ 
       - è¿™æ ·ï¼š
            * è‹¥å•ä¸ª train/val è¢«å¹¿æ’­ï¼Œåªä¼šè¿½åŠ  1 ä¸ª scratch baseline
            * è‹¥ä¸åŒ train/val å¯¹åº”ä¸åŒå®žéªŒï¼Œåˆ™ä¼šä¸ºæ¯ä¸ªä¸åŒç»„åˆè¿½åŠ ä¸€ä¸ª scratch baseline
    """
    train_manifest_list = build_train_manifest_list(args)
    val_manifest_list = build_val_manifest_list(args)
    pretrained_list = normalize_string_list(args.pretrained_weight_paths)

    sources = []

    # ---------------- æƒ…å†µ Aï¼šæ²¡æœ‰é¢„è®­ç»ƒæƒé‡ï¼Œåªè·‘ scratch ----------------
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

    # ---------------- æƒ…å†µ Bï¼šæœ‰é¢„è®­ç»ƒæƒé‡ï¼ŒæŒ‰é¡ºåºå¯¹é½ ----------------
    num_runs = len(pretrained_list)
    aligned_train = align_required_sequence(train_manifest_list, num_runs, "train manifest(s)")
    aligned_val = align_optional_sequence(val_manifest_list, num_runs, "val manifest(s)")

    for i in range(num_runs):
        sources.append({
            "pretrained_path": pretrained_list[i],
            "train_manifest": aligned_train[i],
            "val_manifest": aligned_val[i],
        })

    # ---------------- æƒ…å†µ Cï¼šé¢å¤–è¿½åŠ  scratch baseline ----------------
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
# 11) å•æ¬¡è®­ç»ƒå®žéªŒ
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
    æ‰§è¡Œä¸€æ¬¡å®Œæ•´è®­ç»ƒå®žéªŒã€‚

    è¿™é‡Œçš„ä¸€æ¬¡å®žéªŒå®šä¹‰ä¸ºï¼š
    - scratch è®­ç»ƒï¼Œæˆ–
    - ä»ŽæŸä¸€ä¸ªé¢„è®­ç»ƒæƒé‡åˆå§‹åŒ–åŽè¿›è¡Œå¾®è°ƒ

    æ¯ä¸ªå®žéªŒéƒ½ä¼šå†™åˆ°ç‹¬ç«‹å­ç›®å½•ï¼Œé¿å…å¤šä¸ªå®žéªŒäº’ç›¸è¦†ç›–ã€‚
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
# 12) æ‰¹é‡æµ‹è¯•ï¼ˆæŒ‰è¾“å…¥é¡ºåºå¯¹é½ï¼‰
# ============================================================
def run_batch_test(args, device):
    """
    æ‰¹é‡åŠ è½½å¤šä¸ªå·²è®­ç»ƒå¥½çš„åˆ†ç±»æ¨¡åž‹æƒé‡ï¼Œå¹¶åœ¨ä¸€ä¸ªæˆ–å¤šä¸ª test manifest ä¸Šé€ä¸ªæµ‹è¯•ã€‚

    é¡ºåºåŒ¹é…è§„åˆ™ï¼š
    ----------------------------------------
    - è‹¥ test_manifest åªç»™ 1 ä¸ªï¼Œåˆ™å¹¿æ’­ç»™æ‰€æœ‰ test_weight_paths
    - è‹¥ç»™å¤šä¸ªï¼Œåˆ™ test_manifest[i] å¯¹åº” test_weight_paths[i]

    è¾“å‡ºä¸¤ç±»ç»“æžœï¼š
    1) æ€»æ±‡æ€» CSVï¼šä¿å­˜åˆ° args.test_results_csv æˆ– args.save_path/test_results.csv
    2) æ¯ä¸ªæƒé‡å•ç‹¬çš„é€æ ·æœ¬ CSVï¼šä¿å­˜åˆ°è¯¥æƒé‡æ‰€åœ¨ç›®å½•ï¼Œé¿å…é‡åå†²çª
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

        # è‹¥å¤šä¸ªæƒé‡å…±ç”¨åŒä¸€ä¸ª test manifestï¼Œåˆ™å¤ç”¨ loaderï¼Œå‡å°‘é‡å¤å¼€é”€
        if test_manifest_used not in test_loader_cache:
            test_loader_cache[test_manifest_used] = prepare_test_loader_for_manifest(args, test_manifest_used)

        testloader = test_loader_cache[test_manifest_used]

        model = prepare_model(args).to(device)
        load_report = load_model_weights_for_eval(model, weight_path, map_location="cpu")
        print("[eval load report]")
        print(json.dumps(load_report, indent=2, ensure_ascii=False))

        # per_sample_test è¯¦ç»†å‘½å
        # weight_path_obj = Path(weight_path)
        # weight_dir = str(weight_path_obj.parent)
        # weight_stem = sanitize_name(weight_path_obj.stem)
        # test_manifest_stem = sanitize_name(Path(test_manifest_used).stem)

        # per_sample_csv_name = (
        #     f"{weight_stem}_{test_manifest_stem}_{args.tier_mode}_{args.use_modality}_per_sample.csv"
        # )
        # per_sample_csv_path = os.path.join(weight_dir, per_sample_csv_name)

        # per_sample_test ç®€ç•¥å‘½åï¼Œæ”¾ç½®è¶…è¿‡ 260 å­—ç¬¦é™åˆ¶
        weight_path_obj = Path(weight_path)
        weight_dir = str(weight_path_obj.parent)

        # å›ºå®šæ–‡ä»¶åï¼Œé¿å…è·¯å¾„è¿‡é•¿
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

        # ä¸ºäº†é¿å…åŒä¸€ä»½ manifest è¢«é‡å¤æž„å»ºï¼Œä½¿ç”¨ cache
        # cache key ç”± (train_manifest, val_manifest) ç»„æˆ
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


