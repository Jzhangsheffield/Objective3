
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_mapstyle_captum_multi.py

============================================================
目的
============================================================
基于当前的 map-style 数据集、3D ResNet backbone、以及微调/测试时的
数据读取逻辑，生成单模态（RGB 或 Depth）分类模型的可视化结果。

本脚本支持 4 种方法：
1) Integrated Gradients      (Captum)
2) Guided Backpropagation    (Captum)
3) Occlusion                 (Captum)
4) Grad-CAM++                (内置 3D 视频版本)

============================================================
本版本新增功能：多权重 + 多保存路径一一对应
============================================================
你现在可以一次输入多个权重路径，以及多个保存根目录：

    --weight_paths w1.pth w2.pth w3.pth
    --save_roots   out1   out2   out3

它们会按顺序一一对应：

    w1.pth -> out1
    w2.pth -> out2
    w3.pth -> out3

脚本会依次执行每一组：
1) 加载当前权重
2) 扫描测试集，得到当前模型的预测结果
3) 按当前模型的预测结果进行样本筛选
4) 将可视化结果保存到对应的 save_root

============================================================
为什么必须“每个权重单独扫描”
============================================================
因为你的样本选择规则中，很多条件与“当前模型的预测结果”有关，例如：
- result_filter = correct / incorrect
- incorrect_target_mode = pred / true / both
- class_match_field = pred / true

因此：
------------------------------------------------------------
不同权重得到的 selected samples 可能并不相同
------------------------------------------------------------

所以不能只扫描一次测试集然后复用到所有权重；
必须对每个权重分别扫描，这样逻辑才是严格正确的。

============================================================
路径长度问题的处理方案
============================================================
Windows 下长路径很容易报错，因此本脚本采用短目录名策略：

save_root/
├─ run_info.json
├─ visualization_index.jsonl
├─ visualization_index.json
├─ ig/
│  ├─ vis_000496_a13f2c9d/
│  │  ├─ meta.json
│  │  ├─ attribution.npy
│  │  ├─ heatmap_thw.npy
│  │  ├─ original/
│  │  ├─ heatmap/
│  │  └─ overlay/
├─ gradcampp/
│  ├─ vis_000496_93be2d11/
...

说明：
- 目录名只保留短 hash，不再把长 key / sample_name 塞到路径里
- 完整语义信息写入：
  - run_info.json
  - visualization_index.jsonl
  - visualization_index.json
  - 各样本目录下的 meta.json

============================================================
典型用法示例
============================================================

# ----------------------------------------------------------
# 示例 1：单个权重，和旧版用法兼容
# ----------------------------------------------------------
python visualize_mapstyle_captum_multi.py ^
  --dataset_root L:\Dataset_thermal_crimper\mapstyle_dataset ^
  --label_map_json L:\Dataset_thermal_crimper\mapstyle_dataset\label_map.json ^
  --test_manifest test_manifest.jsonl ^
  --weight_path D:\weights\best_val.pth ^
  --save_root D:\viz_out ^
  --num_classes 17 ^
  --use_modality rgb ^
  --methods gradcampp ^
  --selection_scope global ^
  --result_filter correct

# ----------------------------------------------------------
# 示例 2：多个权重，对应多个保存路径
# ----------------------------------------------------------
python visualize_mapstyle_captum_multi.py ^
  --dataset_root L:\Dataset_thermal_crimper\mapstyle_dataset ^
  --label_map_json L:\Dataset_thermal_crimper\mapstyle_dataset\label_map.json ^
  --test_manifest test_manifest.jsonl ^
  --weight_paths ^
      D:\weights\run_01\best_val.pth ^
      D:\weights\run_02\best_val.pth ^
      D:\weights\run_03\best_val.pth ^
  --save_roots ^
      D:\viz_run01 ^
      D:\viz_run02 ^
      D:\viz_run03 ^
  --num_classes 17 ^
  --use_modality rgb ^
  --methods gradcampp ^
  --selection_scope per_class ^
  --result_filter correct ^
  --random_k 2

# ----------------------------------------------------------
# 示例 3：多个权重，每个都对错误样本同时解释 pred 和 true
# ----------------------------------------------------------
python visualize_mapstyle_captum_multi.py ^
  --dataset_root L:\Dataset_thermal_crimper\mapstyle_dataset ^
  --label_map_json L:\Dataset_thermal_crimper\mapstyle_dataset\label_map.json ^
  --test_manifest test_manifest.jsonl ^
  --weight_paths ^
      D:\weights\run_01\best_val.pth ^
      D:\weights\run_02\best_val.pth ^
  --save_roots ^
      D:\viz_run01 ^
      D:\viz_run02 ^
  --num_classes 17 ^
  --use_modality rgb ^
  --methods ig gradcampp ^
  --selection_scope class ^
  --class_names loosen ^
  --result_filter incorrect ^
  --random_k 3 ^
  --incorrect_target_mode both

# ----------------------------------------------------------
# 示例 4：Depth 模态，多权重
# ----------------------------------------------------------
python visualize_mapstyle_captum_multi.py ^
  --dataset_root L:\Dataset_thermal_crimper\mapstyle_dataset ^
  --label_map_json L:\Dataset_thermal_crimper\mapstyle_dataset\label_map.json ^
  --test_manifest test_manifest.jsonl ^
  --weight_paths D:\w1.pth D:\w2.pth ^
  --save_roots  D:\viz1   D:\viz2 ^
  --num_classes 17 ^
  --use_modality depth ^
  --methods gradcampp ^
  --selection_scope global ^
  --result_filter all
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm
from tqdm import tqdm


# ------------------------------------------------------------
# 让脚本所在目录优先进入 import 路径。
# 若脚本放在项目根目录（与 backbone/、utils_/ 同级），
# 则可直接导入现有模块。
# ------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import backbone.resnet as resnet3d
except Exception as e:
    raise ImportError(
        "Failed to import backbone.resnet. Please place this script in the project root, "
        "or ensure the project root is in PYTHONPATH."
    ) from e

try:
    from utils_.mapstype_dataloader_with_index import (
        PackedMultiModalConfig,
        load_label_map_json,
        build_packed_mapstyle_dataset,
        build_packed_mapstyle_loader_from_dataset,
    )
except Exception as e:
    raise ImportError(
        "Failed to import utils_.mapstype_dataloader_with_index. "
        "Please make sure this script is placed in the same project where your training code runs."
    ) from e

# Captum 为可选依赖：只有使用 ig / guided_bp / occlusion 时才必须安装
try:
    from captum.attr import IntegratedGradients, GuidedBackprop, Occlusion
    _HAS_CAPTUM = True
    _CAPTUM_IMPORT_ERROR = None
except Exception as e:
    _HAS_CAPTUM = False
    _CAPTUM_IMPORT_ERROR = e


# ============================================================
# 基础工具函数
# ============================================================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    """
    将字符串处理为相对安全的文件名片段。
    本脚本大多数目录名已经不依赖长字符串，这里主要用于辅助展示。
    """
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name))
    name = name.strip("._-")
    return name or "item"


def set_seed(seed: int) -> None:
    """
    设定随机种子，让随机抽样尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_manifest_arg(dataset_root: str, manifest_arg: str | None) -> str | None:
    """
    兼容两种传法：
    1) 直接传文件名，例如 test_manifest.jsonl
    2) 传绝对路径或相对路径

    若是绝对路径且位于 dataset_root 内部，则转成相对路径。
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
            return str(manifest_path)
    return str(manifest_path)


def parse_float_list(xs: Iterable[str], expected_len: int, name: str) -> list[float]:
    vals = [float(x) for x in xs]
    if len(vals) != expected_len:
        raise ValueError(f"{name} expects {expected_len} values, got {len(vals)}")
    return vals


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_run_pairs(args) -> list[tuple[str, str]]:
    """
    统一解析“单权重/单保存路径”和“多权重/多保存路径”两种模式。

    支持：
    ------------------------------------------------------------
    1) 单权重模式（兼容旧版）
       --weight_path xxx --save_root yyy

    2) 多权重模式（新增）
       --weight_paths w1 w2 w3 --save_roots s1 s2 s3

    返回：
    ------------------------------------------------------------
    [(weight_path_1, save_root_1), (weight_path_2, save_root_2), ...]
    """
    has_single = (args.weight_path is not None) or (args.save_root is not None)
    has_multi = (len(args.weight_paths) > 0) or (len(args.save_roots) > 0)

    if has_single and has_multi:
        raise ValueError(
            "Please use either single mode (--weight_path / --save_root) "
            "or multi mode (--weight_paths / --save_roots), not both."
        )

    if has_single:
        if args.weight_path is None or args.save_root is None:
            raise ValueError(
                "Single mode requires both --weight_path and --save_root."
            )
        return [(str(args.weight_path), str(args.save_root))]

    if has_multi:
        if len(args.weight_paths) == 0 or len(args.save_roots) == 0:
            raise ValueError(
                "Multi mode requires both --weight_paths and --save_roots."
            )
        if len(args.weight_paths) != len(args.save_roots):
            raise ValueError(
                f"The number of weight paths ({len(args.weight_paths)}) must equal "
                f"the number of save roots ({len(args.save_roots)})."
            )
        return list(zip([str(x) for x in args.weight_paths], [str(x) for x in args.save_roots]))

    raise ValueError(
        "You must provide either:\n"
        "  - single mode: --weight_path + --save_root\n"
        "  - multi mode : --weight_paths + --save_roots"
    )


# ============================================================
# 标签映射
# ============================================================
def build_reverse_label_map(label_map_json_path: str, tier_mode: str) -> dict[int, str]:
    label_map = load_label_map_json(label_map_json_path)
    if tier_mode not in label_map:
        raise KeyError(f"tier_mode '{tier_mode}' not found in label_map.json")
    forward_map = label_map[tier_mode]
    return {int(v): str(k) for k, v in forward_map.items()}


def build_forward_label_map(label_map_json_path: str, tier_mode: str) -> dict[str, int]:
    label_map = load_label_map_json(label_map_json_path)
    if tier_mode not in label_map:
        raise KeyError(f"tier_mode '{tier_mode}' not found in label_map.json")
    return {str(k): int(v) for k, v in label_map[tier_mode].items()}


# ============================================================
# 数据构建：尽量沿用训练脚本逻辑
# ============================================================
def build_mapstyle_cfg(args, is_train: bool):
    rgb_hw = (args.rgb_size, args.rgb_size)
    depth_hw = (args.depth_size, args.depth_size)
    use_modalities = (args.use_modality,)

    cfg = PackedMultiModalConfig(
        n_frames=args.n_frames,
        rgb_two_views=False,
        use_modalities=use_modalities,
        missing_policy="skip",
        load_labels=True,
        label_map_path=args.label_map_json,
        tier_mode=args.tier_mode,
        is_train=is_train,
        rgb_out_hw=rgb_hw,
        rrc_scale=(args.rrc_scale_min, args.rrc_scale_max),
        rrc_ratio=(args.rrc_ratio_min, args.rrc_ratio_max),
        depth_out_hw=depth_hw,
        default_rgb_hw=(256, 256),
        default_depth_hw=depth_hw,
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
):
    """
    构建单个 map-style dataset 和对应 DataLoader。
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

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=None,
        pin_memory=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = build_packed_mapstyle_loader_from_dataset(**loader_kwargs)
    return dataset, loader


def prepare_test_loader(args):
    """
    构建测试集 dataset / loader。
    """
    dataset, testloader = build_one_mapstyle_dataset_and_loader(
        args=args,
        manifest_arg=args.test_manifest,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers_test,
        prefetch_factor=args.prefetch_factor_test,
        shuffle=False,
        drop_last=False,
    )
    return dataset, testloader


# ============================================================
# 模型构建 / 权重加载
# ============================================================
def prepare_model(args):
    return resnet3d.generate_model(args.model_depth, num_classes=args.num_classes)


def strip_prefixes_from_key(key: str) -> str:
    """
    清理常见 checkpoint 前缀，方便从不同训练封装中加载。
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
    从 checkpoint 对象中提取真正的 state_dict。
    """
    if not isinstance(ckpt_obj, dict):
        raise TypeError("Checkpoint object must be a dict-like object.")

    preferred_keys = ["model_state_dict", "state_dict", "model", "net", "network"]
    for k in preferred_keys:
        if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
            return ckpt_obj[k]

    tensor_like = 0
    for k, v in ckpt_obj.items():
        if isinstance(k, str) and torch.is_tensor(v):
            tensor_like += 1
    if tensor_like > 0:
        return ckpt_obj

    raise ValueError("Unable to find a valid state_dict inside checkpoint.")


def normalize_and_filter_state_dict(raw_state_dict: dict, model_state_dict: dict):
    """
    清理 key 前缀并过滤：
    - 模型中不存在的 key
    - shape 不匹配的 key
    """
    cleaned = {}
    dropped_missing_keys = []
    dropped_shape_keys = []

    for k, v in raw_state_dict.items():
        new_k = strip_prefixes_from_key(k)
        if new_k not in model_state_dict:
            dropped_missing_keys.append(new_k)
            continue
        if tuple(v.shape) != tuple(model_state_dict[new_k].shape):
            dropped_shape_keys.append((new_k, tuple(v.shape), tuple(model_state_dict[new_k].shape)))
            continue
        cleaned[new_k] = v

    report = {
        "num_loaded": len(cleaned),
        "num_dropped_missing": len(dropped_missing_keys),
        "num_dropped_shape": len(dropped_shape_keys),
        "dropped_missing_keys_preview": dropped_missing_keys[:20],
        "dropped_shape_keys_preview": dropped_shape_keys[:20],
    }
    return cleaned, report


def load_model_weights_for_eval(model, ckpt_path: str, map_location: str = "cpu") -> dict[str, Any]:
    """
    加载用于推理/可视化的模型权重，并返回加载报告。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for evaluation: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location)
    raw_state_dict = extract_state_dict_from_checkpoint(ckpt)
    model_state_dict = model.state_dict()

    filtered_state_dict, report = normalize_and_filter_state_dict(
        raw_state_dict=raw_state_dict,
        model_state_dict=model_state_dict,
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


# ============================================================
# batch / item 解析与预处理
# ============================================================
def _extract_inputs_and_labels(batch: dict, tier_mode: str, use_modality: str):
    """
    从 batch 中取出：
    - 输入视频张量 [B,T,C,H,W]
    - 标签
    - clip_ids
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
    else:
        raise ValueError(f"Unsupported modality: {use_modality}")

    return inputs, labels, clip_ids


def _ensure_bcthw(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    [B,T,C,H,W] -> [B,C,T,H,W]
    """
    if x_btchw.ndim != 5:
        raise ValueError(f"Expect 5D tensor [B,T,C,H,W], got shape={tuple(x_btchw.shape)}")
    return x_btchw.permute(0, 2, 1, 3, 4).contiguous()


def preprocess_rgb_already_normed(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    RGB 已经在 dataloader 内完成过 Normalize，因此这里只保证 dtype。
    """
    if x_btchw.dtype != torch.float32:
        x_btchw = x_btchw.to(torch.float32)
    return x_btchw


def preprocess_depth_to_float(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    Depth 只转为 float32。
    """
    if x_btchw.dtype != torch.float32:
        x_btchw = x_btchw.to(torch.float32)
    return x_btchw


def preprocess_input_btchw(x_btchw: torch.Tensor, use_modality: str) -> torch.Tensor:
    """
    对输入进行模态相关预处理，并转成 [B,C,T,H,W]。
    """
    if use_modality == "rgb":
        x_btchw = preprocess_rgb_already_normed(x_btchw)
    elif use_modality == "depth":
        x_btchw = preprocess_depth_to_float(x_btchw)
    else:
        raise ValueError(f"Unsupported modality: {use_modality}")
    return _ensure_bcthw(x_btchw)


def extract_item_input_label_and_ids(item: dict, args) -> tuple[torch.Tensor, int, str, str | None]:
    """
    从 dataset[idx] 的单个 item 中抽取：
    - 输入 [1,T,C,H,W]
    - 真实标签 int
    - clip_id（优先 key）
    - sample_name（如果有）

    注意：
    这里假设 dataset.__getitem__ 返回的 item 结构与 batch 中单样本结构一致。
    """
    if not isinstance(item, dict):
        raise TypeError(
            "Expected dataset[idx] to return a dict-like object. "
            "If your dataset returns a tuple, please adapt extract_item_input_label_and_ids()."
        )

    clip_id = item.get("key", None)
    sample_name = item.get("sample_name", None)
    if clip_id is None:
        clip_id = sample_name
    if clip_id is None:
        clip_id = f"index_{item.get('global_index', 'unknown')}"

    if "tier_ids" not in item:
        raise KeyError("Single dataset item must contain 'tier_ids'.")

    labels_dict = item["tier_ids"]
    label_val = labels_dict[args.tier_mode]
    if torch.is_tensor(label_val):
        label_val = int(label_val.item())
    else:
        label_val = int(label_val)

    if args.use_modality not in item:
        raise KeyError(f"Single dataset item does not contain modality '{args.use_modality}'.")

    x = item[args.use_modality]
    if not torch.is_tensor(x):
        raise TypeError(f"Expected item['{args.use_modality}'] to be a torch.Tensor.")
    if x.ndim != 4:
        raise ValueError(
            f"Expected single item modality tensor to be [T,C,H,W], got shape={tuple(x.shape)}"
        )

    x = x.unsqueeze(0)  # -> [1,T,C,H,W]
    return x, label_val, str(clip_id), None if sample_name is None else str(sample_name)


# ============================================================
# 显示相关工具
# ============================================================
def tensor_rgb_for_display(x_cthw: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """
    输入：
        x_cthw: [C,T,H,W]，通常已经做过 Normalize
    输出：
        uint8 RGB 视频帧: [T,H,W,3]
    """
    if x_cthw.shape[0] != 3:
        raise ValueError(f"RGB display expects C=3, got shape={tuple(x_cthw.shape)}")

    x = x_cthw.detach().cpu().float().clone()
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1, 1)
    x = x * std_t + mean_t
    x = x.clamp(0.0, 1.0)
    x = x.permute(1, 2, 3, 0).contiguous().numpy()  # [T,H,W,3]
    x = (x * 255.0).round().astype(np.uint8)
    return x


def tensor_depth_for_display(x_cthw: torch.Tensor) -> np.ndarray:
    """
    输入：
        x_cthw: [1,T,H,W]
    输出：
        uint8 灰度转 3 通道: [T,H,W,3]
    """
    if x_cthw.shape[0] != 1:
        raise ValueError(f"Depth display expects C=1, got shape={tuple(x_cthw.shape)}")

    x = x_cthw.detach().cpu().float()[0]  # [T,H,W]
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = torch.zeros_like(x)
    x = (x * 255.0).round().byte().numpy()  # [T,H,W]
    x = np.repeat(x[..., None], repeats=3, axis=-1)  # [T,H,W,3]
    return x


def tensor_for_display(x_bcthw: torch.Tensor, args) -> np.ndarray:
    """
    [1,C,T,H,W] -> [T,H,W,3]
    """
    if x_bcthw.ndim != 5 or x_bcthw.shape[0] != 1:
        raise ValueError(f"Expect [1,C,T,H,W], got {tuple(x_bcthw.shape)}")
    x_cthw = x_bcthw[0]
    if args.use_modality == "rgb":
        return tensor_rgb_for_display(x_cthw, mean=args.rgb_mean, std=args.rgb_std)
    return tensor_depth_for_display(x_cthw)


def normalize_heatmap_3d(x_thw: np.ndarray, mode: str = "per_frame") -> np.ndarray:
    """
    将 [T,H,W] 的热图归一化到 [0,1]。

    mode:
      - per_frame: 每帧独立归一化，便于观察每帧细节
      - global   : 整个 clip 共用一个 min/max，便于跨帧比较强弱
    """
    x = x_thw.astype(np.float32)
    eps = 1e-8

    if mode == "global":
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min + eps)
        return np.zeros_like(x)

    if mode == "per_frame":
        out = np.zeros_like(x)
        for t in range(x.shape[0]):
            xx = x[t]
            mn = float(xx.min())
            mx = float(xx.max())
            if mx > mn:
                out[t] = (xx - mn) / (mx - mn + eps)
            else:
                out[t] = 0.0
        return out

    raise ValueError(f"Unknown normalize mode: {mode}")


def aggregate_attr_to_thw(attr: torch.Tensor, reduction: str = "abs_mean") -> np.ndarray:
    """
    将输入级归因 [1,C,T,H,W] 聚合成 [T,H,W]。
    """
    if attr.ndim != 5 or attr.shape[0] != 1:
        raise ValueError(f"Expect attr shape [1,C,T,H,W], got {tuple(attr.shape)}")

    x = attr.detach().cpu().float()[0]  # [C,T,H,W]

    if reduction == "abs_mean":
        x = x.abs().mean(dim=0)
    elif reduction == "abs_sum":
        x = x.abs().sum(dim=0)
    elif reduction == "mean":
        x = x.mean(dim=0)
    elif reduction == "sum":
        x = x.sum(dim=0)
    else:
        raise ValueError(f"Unknown attr reduction: {reduction}")

    return x.numpy()  # [T,H,W]


def heatmap_to_rgb(heat_01_hw: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    """
    将 [H,W] 的 [0,1] 热图转为伪彩色 RGB 图。
    """
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.clip(heat_01_hw, 0.0, 1.0))  # [H,W,4]
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return rgb


def overlay_heatmap_on_frame(frame_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """
    将热图叠加到原图上。
    """
    frame = frame_rgb.astype(np.float32)
    heat = heat_rgb.astype(np.float32)
    out = frame * (1.0 - alpha) + heat * alpha
    return np.clip(out, 0, 255).round().astype(np.uint8)


def save_rgb_frame(path: str | Path, img_hwc_uint8: np.ndarray) -> None:
    """
    保存单张 RGB 图像。
    """
    Image.fromarray(img_hwc_uint8).save(path)


# ============================================================
# Grad-CAM++：3D 视频版本
# ============================================================
class GradCAMPP3D:
    """
    3D 视频版 Grad-CAM++。

    核心思路：
    1) 对目标层注册 forward/backward hook
    2) 拿到该层 feature 和 gradient
    3) 统一为 [N,C,T,H,W]
    4) 按帧计算 Grad-CAM++ 权重，不跨 T 聚合
    5) 将得到的 [T_f,H_f,W_f] 一次性插值回输入 [T,H,W]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.feat = None
        self.grad = None
        self.h_fwd = self.target_layer.register_forward_hook(self._fwd_hook)
        self.h_bwd = self.target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inputs, output):
        self.feat = output.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0].detach()

    def remove(self):
        self.h_fwd.remove()
        self.h_bwd.remove()

    @staticmethod
    def _to_ncthw(x: torch.Tensor) -> torch.Tensor:
        """
        将常见 5D 排列统一成 [N,C,T,H,W]。
        """
        if x.ndim != 5:
            raise ValueError(f"Expect 5D tensor, got shape={tuple(x.shape)}")

        s = x.shape

        # case A: [N,T,H,W,C]
        if s[-1] not in (s[1], s[2], s[3]):
            return x.permute(0, 4, 1, 2, 3).contiguous()

        # case B: [N,T,C,H,W]
        if s[2] >= 32 and s[1] < 64:
            return x.permute(0, 2, 1, 3, 4).contiguous()

        # case C: 已经是 [N,C,T,H,W]
        return x.contiguous()

    @staticmethod
    def _gradcam_pp_per_frame_and_align(
        grad: torch.Tensor,
        feat: torch.Tensor,
        out_thw: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        输入：
            grad: [N,C,T_f,H_f,W_f]
            feat: [N,C,T_f,H_f,W_f]
            out_thw: (T_out, H_out, W_out)

        输出：
            cam_up: [N,T_out,H_out,W_out]
        """
        grad = grad.float()
        feat = feat.float()
        eps = 1e-6

        g2 = grad * grad
        g3 = g2 * grad
        a_sum = feat.sum(dim=(3, 4), keepdim=True)
        denom = 2.0 * g2 + a_sum * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        aij = g2 / (denom + eps)
        aij = torch.where(grad != 0, aij, torch.zeros_like(aij))
        relu_g = torch.clamp_min(grad, 0.0)

        weights = (aij * relu_g).sum(dim=(3, 4), keepdim=True)  # [N,C,T,1,1]
        cam = (weights * feat).sum(dim=1)                       # [N,T,H,W]
        cam = torch.clamp_min(cam, 0.0)

        cam = cam[:, None]  # [N,1,T,H,W]
        cam_up = F.interpolate(
            cam,
            size=out_thw,
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
        return cam_up

    def attribute(self, x_bcthw: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        对输入样本做 Grad-CAM++，输出 [B,T,H,W]。
        """
        if x_bcthw.ndim != 5:
            raise ValueError(f"Expect [B,C,T,H,W], got {tuple(x_bcthw.shape)}")

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x_bcthw)
        if logits.ndim != 2:
            raise ValueError(f"Model output is expected to be [B,num_classes], got {tuple(logits.shape)}")

        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)

        if self.feat is None or self.grad is None:
            raise RuntimeError("Grad-CAM++ hooks did not capture feature/gradient correctly.")

        feat = self._to_ncthw(self.feat)
        grad = self._to_ncthw(self.grad)

        _, _, t_in, h_in, w_in = x_bcthw.shape
        cam_up = self._gradcam_pp_per_frame_and_align(
            grad=grad,
            feat=feat,
            out_thw=(t_in, h_in, w_in),
        )
        return cam_up  # [B,T,H,W]


# ============================================================
# 模块路径解析：用于 Grad-CAM++ 目标层选择
# ============================================================
def resolve_module_by_path(root_module: nn.Module, path: str) -> nn.Module:
    """
    支持：
    - layer4
    - layer4.1
    - layer3.0.conv2

    数字 token 会被当作 Sequential / ModuleList 的索引。
    """
    module = root_module
    for token in path.split("."):
        if token == "":
            continue
        if token.isdigit():
            module = module[int(token)]
        else:
            module = getattr(module, token)
    return module


def get_default_gradcam_layer_path(model: nn.Module) -> str:
    """
    对当前 ResNet 结构，默认选 layer4 的最后一个 block。
    """
    if hasattr(model, "layer4") and isinstance(model.layer4, nn.Sequential) and len(model.layer4) > 0:
        return f"layer4.{len(model.layer4) - 1}"
    return "layer4"


# ============================================================
# 先扫描测试集，得到每个样本的预测信息
# ============================================================
@dataclass
class SampleInfo:
    global_index: int
    key: str
    sample_name: str | None
    true_label: int
    pred_label: int
    correct: bool
    confidence: float
    true_name: str
    pred_name: str


def scan_test_predictions(
    model: nn.Module,
    testloader,
    device: torch.device,
    args,
    reverse_label_map: dict[int, str],
) -> list[SampleInfo]:
    """
    在测试集上跑一遍前向，记录：
    - true / pred
    - 是否预测正确
    - 置信度
    - 对应的标签名
    """
    infos: list[SampleInfo] = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(testloader, desc="Scanning test predictions", dynamic_ncols=True):
            x_btchw, labels, clip_ids = _extract_inputs_and_labels(batch, args.tier_mode, args.use_modality)
            x_bcthw = preprocess_input_btchw(x_btchw, args.use_modality).to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            logits = model(x_bcthw)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            global_indices = batch.get("global_index", None)
            sample_names = batch.get("sample_name", None)
            keys = batch.get("key", None)

            bsz = x_bcthw.shape[0]
            for i in range(bsz):
                gi = int(global_indices[i].item()) if global_indices is not None else len(infos)

                if keys is not None:
                    key_i = str(keys[i])
                elif clip_ids is not None:
                    key_i = str(clip_ids[i])
                else:
                    key_i = f"index_{gi}"

                sample_name_i = None if sample_names is None else str(sample_names[i])
                true_i = int(labels[i].item())
                pred_i = int(preds[i].item())
                conf_i = float(confs[i].item())

                infos.append(
                    SampleInfo(
                        global_index=gi,
                        key=key_i,
                        sample_name=sample_name_i,
                        true_label=true_i,
                        pred_label=pred_i,
                        correct=(true_i == pred_i),
                        confidence=conf_i,
                        true_name=reverse_label_map.get(true_i, str(true_i)),
                        pred_name=reverse_label_map.get(pred_i, str(pred_i)),
                    )
                )
    return infos


# ============================================================
# 样本选择逻辑
# ============================================================
def parse_class_filter(args, forward_label_map: dict[str, int]) -> set[int] | None:
    class_ids: set[int] = set()

    if args.class_ids:
        for x in args.class_ids:
            class_ids.add(int(x))

    if args.class_names:
        for name in args.class_names:
            if name not in forward_label_map:
                raise KeyError(f"Class name '{name}' not found in label_map[{args.tier_mode}].")
            class_ids.add(int(forward_label_map[name]))

    return class_ids if len(class_ids) > 0 else None


def get_class_of_info(info: SampleInfo, class_match_field: str) -> int:
    if class_match_field == "true":
        return info.true_label
    if class_match_field == "pred":
        return info.pred_label
    raise ValueError(f"Unknown class_match_field: {class_match_field}")


def filter_by_result(infos: list[SampleInfo], result_filter: str) -> list[SampleInfo]:
    if result_filter == "all":
        return infos
    if result_filter == "correct":
        return [x for x in infos if x.correct]
    if result_filter == "incorrect":
        return [x for x in infos if not x.correct]
    raise ValueError(f"Unknown result_filter: {result_filter}")


def select_single_sample(infos: list[SampleInfo], args) -> list[SampleInfo]:
    """
    选择单一样本：
    - sample_index
    - sample_name
    - sample_key
    三者可组合，但最终必须只匹配到一个样本。
    """
    matches = infos

    if args.sample_index is not None:
        matches = [x for x in matches if x.global_index == int(args.sample_index)]
    if args.sample_name is not None:
        matches = [x for x in matches if x.sample_name == args.sample_name]
    if args.sample_key is not None:
        matches = [x for x in matches if x.key == args.sample_key]

    if len(matches) == 0:
        raise RuntimeError("No sample matched the specified single-sample condition.")
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple samples matched the specified single-sample condition. "
            "Please provide a more specific identifier."
        )
    return matches


def group_sample_infos_by_class(
    infos: list[SampleInfo],
    class_match_field: str,
) -> dict[int, list[SampleInfo]]:
    groups: dict[int, list[SampleInfo]] = {}
    for info in infos:
        cls_id = get_class_of_info(info, class_match_field)
        groups.setdefault(cls_id, []).append(info)
    return groups


def maybe_random_sample(infos: list[SampleInfo], random_k: int, rng: random.Random) -> list[SampleInfo]:
    if random_k is None or int(random_k) <= 0:
        return list(infos)
    if len(infos) <= int(random_k):
        return list(infos)
    return rng.sample(list(infos), k=int(random_k))


def select_samples(
    infos: list[SampleInfo],
    args,
    forward_label_map: dict[str, int],
) -> list[SampleInfo]:
    """
    统一样本选择入口。
    """
    rng = random.Random(args.seed)
    class_filter = parse_class_filter(args, forward_label_map)

    if args.selection_scope == "single":
        selected = select_single_sample(infos, args)
        return selected

    filtered = filter_by_result(infos, args.result_filter)

    if class_filter is not None:
        filtered = [
            x for x in filtered
            if get_class_of_info(x, args.class_match_field) in class_filter
        ]

    if args.selection_scope in {"global", "class"}:
        return maybe_random_sample(filtered, args.random_k, rng)

    if args.selection_scope == "per_class":
        groups = group_sample_infos_by_class(filtered, args.class_match_field)
        selected_all: list[SampleInfo] = []
        for cls_id in sorted(groups.keys()):
            group = groups[cls_id]
            picked = maybe_random_sample(group, args.random_k, rng)
            selected_all.extend(picked)
        return selected_all

    raise ValueError(f"Unknown selection_scope: {args.selection_scope}")


def select_samples_by_reference_indices(
    infos: list[SampleInfo],
    reference_selected_infos: list[SampleInfo],
) -> list[SampleInfo]:
    """
    多权重模式下，后续权重不再重新按自己的预测结果筛样本，
    而是复用“第一个权重”已经选中的样本集合。

    这里通过 global_index 进行对齐：
    - reference_selected_infos 决定“选哪些样本、顺序是什么”
    - 当前权重自己的 infos 决定“这些样本在当前权重下的 pred/confidence/correct 是什么”

    这样可以保证：
    1) 后续权重和第一个权重可视化的是同一批样本
    2) 每个权重仍然使用自己当前的预测结果来决定 target / meta 信息
    """
    info_map = {int(x.global_index): x for x in infos}
    selected: list[SampleInfo] = []
    missing_indices: list[int] = []

    for ref in reference_selected_infos:
        gi = int(ref.global_index)
        cur = info_map.get(gi, None)
        if cur is None:
            missing_indices.append(gi)
            continue
        selected.append(cur)

    if missing_indices:
        raise RuntimeError(
            "Some reference-selected samples were not found in the current weight run. "
            f"Missing global_index values: {missing_indices[:20]}"
        )

    return selected


# ============================================================
# target 选择：针对错误样本的 pred / true / both
# ============================================================
def build_target_specs(info: SampleInfo, args) -> list[tuple[str, int, str]]:
    """
    返回：
        (target_role, target_class_id, target_class_name)

    额外说明：
    ------------------------------------------------------------
    当启用多权重“参考首权重样本集合”模式时，
    若当前权重对该样本预测错误，则强制同时输出：
        - pred target
        - true target

    这样便于直接比较：同一个样本在不同权重下，若预测出错，
    能同时看到“模型实际认为重要的区域（pred）”和“真实类别应该关注的区域（true）”。
    """
    if info.correct:
        return [("pred", info.pred_label, info.pred_name)]

    if getattr(args, "force_both_targets_for_incorrect", False):
        return [
            ("pred", info.pred_label, info.pred_name),
            ("true", info.true_label, info.true_name),
        ]

    mode = args.incorrect_target_mode
    if mode == "pred":
        return [("pred", info.pred_label, info.pred_name)]
    if mode == "true":
        return [("true", info.true_label, info.true_name)]
    if mode == "both":
        return [
            ("pred", info.pred_label, info.pred_name),
            ("true", info.true_label, info.true_name),
        ]
    raise ValueError(f"Unknown incorrect_target_mode: {mode}")


# ============================================================
# Captum 方法封装
# ============================================================
def require_captum(method_name: str) -> None:
    if not _HAS_CAPTUM:
        raise ImportError(
            f"Method '{method_name}' requires Captum, but Captum import failed: {_CAPTUM_IMPORT_ERROR}\n"
            "Please install it with: pip install captum"
        )


def run_integrated_gradients(
    model: nn.Module,
    x_bcthw: torch.Tensor,
    target_class: int,
    args,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrated Gradients:
    返回：
    - 原始 attribution 数组
    - 聚合后的 [T,H,W] 热图
    """
    require_captum("ig")

    baseline = torch.zeros_like(x_bcthw)

    ig = IntegratedGradients(model)
    attr = ig.attribute(
        x_bcthw,
        baselines=baseline,
        target=target_class,
        n_steps=args.ig_steps,
        internal_batch_size=args.ig_internal_batch_size if args.ig_internal_batch_size > 0 else None,
    )
    heat_thw = aggregate_attr_to_thw(attr, reduction=args.attr_reduction)
    return attr.detach().cpu().numpy(), heat_thw


def run_guided_backprop(
    model: nn.Module,
    x_bcthw: torch.Tensor,
    target_class: int,
    args,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Guided Backprop:
    返回：
    - 原始 attribution 数组
    - 聚合后的 [T,H,W] 热图
    """
    require_captum("guided_bp")

    gbp = GuidedBackprop(model)
    attr = gbp.attribute(x_bcthw.requires_grad_(True), target=target_class)
    heat_thw = aggregate_attr_to_thw(attr, reduction=args.attr_reduction)
    return attr.detach().cpu().numpy(), heat_thw


def run_occlusion(
    model: nn.Module,
    x_bcthw: torch.Tensor,
    target_class: int,
    args,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Occlusion:
    返回：
    - 原始 attribution 数组
    - 聚合后的 [T,H,W] 热图
    """
    require_captum("occlusion")

    occ = Occlusion(model)

    c = int(x_bcthw.shape[1])
    sliding_window_shapes = (c, args.occlusion_window_t, args.occlusion_window_h, args.occlusion_window_w)
    strides = (c, args.occlusion_stride_t, args.occlusion_stride_h, args.occlusion_stride_w)

    attr = occ.attribute(
        x_bcthw,
        target=target_class,
        sliding_window_shapes=sliding_window_shapes,
        strides=strides,
        baselines=args.occlusion_baseline,
        perturbations_per_eval=args.occlusion_perturbations_per_eval,
        show_progress=args.occlusion_show_progress,
    )
    heat_thw = aggregate_attr_to_thw(attr, reduction=args.attr_reduction)
    return attr.detach().cpu().numpy(), heat_thw


def run_gradcampp(
    model: nn.Module,
    x_bcthw: torch.Tensor,
    target_class: int,
    target_layer: nn.Module,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM++:
    返回：
    - 原始 [1,T,H,W] CAM
    - [T,H,W] 热图
    """
    cam = GradCAMPP3D(model, target_layer)
    try:
        cam_up = cam.attribute(x_bcthw.requires_grad_(True), target_class=target_class)  # [1,T,H,W]
    finally:
        cam.remove()

    heat_thw = cam_up.detach().cpu().numpy()[0]
    return cam_up.detach().cpu().numpy(), heat_thw


# ============================================================
# 短目录命名 + 结果保存
# ============================================================
def build_short_vis_dir_name(info: SampleInfo, method: str, target_role: str, target_class: int) -> str:
    """
    生成短且稳定的目录名，避免 Windows 路径过长。
    """
    raw = f"{method}|{info.global_index}|{info.key}|{info.sample_name}|{target_role}|{target_class}"
    short_hash = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"vis_{info.global_index:06d}_{short_hash}"


def save_visualization_bundle(
    save_dir: Path,
    display_frames_thwc: np.ndarray,
    heat_thw_raw: np.ndarray,
    heat_normalize_mode: str,
    overlay_alpha: float,
    raw_attr: np.ndarray,
    meta: dict[str, Any],
    args,
) -> None:
    """
    保存单个可视化 bundle：
    - attribution.npy
    - heatmap_thw.npy
    - meta.json
    - original/*.png
    - heatmap/*.png
    - overlay/*.png
    """
    ensure_dir(save_dir)
    ensure_dir(save_dir / "original")
    ensure_dir(save_dir / "heatmap")
    ensure_dir(save_dir / "overlay")

    np.save(save_dir / "attribution.npy", raw_attr)
    np.save(save_dir / "heatmap_thw.npy", heat_thw_raw)

    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    heat_01 = normalize_heatmap_3d(heat_thw_raw, mode=heat_normalize_mode)
    t = display_frames_thwc.shape[0]

    for i in range(t):
        frame_rgb = display_frames_thwc[i]
        heat_rgb = heatmap_to_rgb(heat_01[i], cmap_name=args.cmap)
        overlay_rgb = overlay_heatmap_on_frame(frame_rgb, heat_rgb, alpha=overlay_alpha)

        save_rgb_frame(save_dir / "original" / f"frame_{i:04d}.png", frame_rgb)
        save_rgb_frame(save_dir / "heatmap" / f"frame_{i:04d}.png", heat_rgb)
        save_rgb_frame(save_dir / "overlay" / f"frame_{i:04d}.png", overlay_rgb)


# ============================================================
# 单样本执行入口
# ============================================================
def visualize_one_sample(
    model: nn.Module,
    dataset,
    info: SampleInfo,
    device: torch.device,
    args,
    target_layer: nn.Module,
):
    """
    对单个样本执行可视化。

    返回：
    ------------------------------------------------------------
    一个 index_record 列表。
    因为一个样本可能会输出多个：
    - method
    - target_role
    所以这里返回 list。
    """
    item = dataset[info.global_index]
    x_btchw, true_label_item, clip_id_item, sample_name_item = extract_item_input_label_and_ids(item, args)

    if int(true_label_item) != int(info.true_label):
        raise RuntimeError(
            f"Dataset item label mismatch at index={info.global_index}: "
            f"scan true={info.true_label}, item true={true_label_item}"
        )

    x_bcthw = preprocess_input_btchw(x_btchw, args.use_modality).to(device)
    display_frames = tensor_for_display(x_bcthw, args)  # [T,H,W,3]

    index_records: list[dict[str, Any]] = []

    for target_role, target_class, target_name in build_target_specs(info, args):
        for method in args.methods:
            method = str(method)
            model.zero_grad(set_to_none=True)

            if method == "ig":
                raw_attr, heat_thw = run_integrated_gradients(model, x_bcthw.clone(), target_class, args)
            elif method == "guided_bp":
                raw_attr, heat_thw = run_guided_backprop(model, x_bcthw.clone(), target_class, args)
            elif method == "occlusion":
                raw_attr, heat_thw = run_occlusion(model, x_bcthw.clone(), target_class, args)
            elif method == "gradcampp":
                raw_attr, heat_thw = run_gradcampp(model, x_bcthw.clone(), target_class, target_layer)
            else:
                raise ValueError(f"Unsupported method: {method}")

            short_dir_name = build_short_vis_dir_name(
                info=info,
                method=method,
                target_role=target_role,
                target_class=target_class,
            )

            save_dir = Path(args.current_save_root) / method / short_dir_name

            meta = {
                "save_dir_name": short_dir_name,
                "save_dir": str(save_dir),
                "global_index": info.global_index,
                "key": info.key,
                "sample_name": info.sample_name,
                "clip_id_item": clip_id_item,
                "sample_name_item": sample_name_item,
                "true_label": info.true_label,
                "true_name": info.true_name,
                "pred_label": info.pred_label,
                "pred_name": info.pred_name,
                "correct": info.correct,
                "confidence": info.confidence,
                "target_role": target_role,
                "target_class": target_class,
                "target_name": target_name,
                "method": method,
                "modality": args.use_modality,
                "tier_mode": args.tier_mode,
                "weight_path": args.current_weight_path,
                "weight_stem": args.current_weight_stem,
                "gradcam_target_layer": args.gradcam_target_layer_resolved,
                "selection_scope": args.selection_scope,
                "result_filter": args.result_filter,
                "class_match_field": args.class_match_field,
                "random_k": args.random_k,
                "incorrect_target_mode": args.incorrect_target_mode,
                "force_both_targets_for_incorrect": getattr(args, "force_both_targets_for_incorrect", False),
                "force_both_targets_for_incorrect": getattr(args, "force_both_targets_for_incorrect", False),
                "heat_normalize_mode": args.heat_normalize_mode,
                "overlay_alpha": args.overlay_alpha,
                "attr_reduction": args.attr_reduction,
            }

            save_visualization_bundle(
                save_dir=save_dir,
                display_frames_thwc=display_frames,
                heat_thw_raw=heat_thw,
                heat_normalize_mode=args.heat_normalize_mode,
                overlay_alpha=args.overlay_alpha,
                raw_attr=raw_attr,
                meta=meta,
                args=args,
            )

            index_record = {
                "save_dir_name": short_dir_name,
                "save_dir": str(save_dir),
                "method": method,
                "global_index": info.global_index,
                "key": info.key,
                "sample_name": info.sample_name,
                "clip_id_item": clip_id_item,
                "sample_name_item": sample_name_item,
                "true_label": info.true_label,
                "true_name": info.true_name,
                "pred_label": info.pred_label,
                "pred_name": info.pred_name,
                "correct": info.correct,
                "confidence": info.confidence,
                "target_role": target_role,
                "target_class": target_class,
                "target_name": target_name,
                "selection_scope": args.selection_scope,
                "result_filter": args.result_filter,
                "class_match_field": args.class_match_field,
                "random_k": args.random_k,
                "incorrect_target_mode": args.incorrect_target_mode,
                "force_both_targets_for_incorrect": getattr(args, "force_both_targets_for_incorrect", False),
                "weight_path": args.current_weight_path,
                "weight_stem": args.current_weight_stem,
                "save_root": args.current_save_root,
                "gradcam_target_layer": args.gradcam_target_layer_resolved,
                "short_dir_note": (
                    "Long sample/key/target information is intentionally stored in JSON "
                    "instead of folder names to avoid Windows path length issues."
                ),
            }
            index_records.append(index_record)

    return index_records


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a 3D ResNet classifier on a map-style dataset using Captum and Grad-CAM++."
    )

    # ---------------- 基本路径 ----------------
    parser.add_argument("--dataset_root", type=str, required=True, help="map-style 数据集根目录")
    parser.add_argument("--label_map_json", type=str, required=True, help="label_map.json 路径")
    parser.add_argument("--test_manifest", type=str, required=True, help="测试 manifest 路径或文件名")

    # ---------------- 单权重模式（兼容旧版） ----------------
    parser.add_argument("--weight_path", type=str, default=None, help="单个权重路径（单权重模式）")
    parser.add_argument("--save_root", type=str, default=None, help="单个结果保存根目录（单权重模式）")

    # ---------------- 多权重模式（新增） ----------------
    parser.add_argument("--weight_paths", nargs="+", default=[], help="多个权重路径（多权重模式）")
    parser.add_argument("--save_roots", nargs="+", default=[], help="多个保存根目录（多权重模式）")

    # ---------------- 模型/标签/模态 ----------------
    parser.add_argument("--model_depth", type=int, default=18, help="3D ResNet 深度")
    parser.add_argument("--num_classes", type=int, required=True, help="类别数量")
    parser.add_argument("--tier_mode", type=str, default="tier1", choices=["tier1", "tier2", "tier3"])
    parser.add_argument("--use_modality", type=str, default="rgb", choices=["rgb", "depth"])
    parser.add_argument("--n_frames", type=int, default=16, help="每个样本采样帧数")

    # ---------------- DataLoader ----------------
    parser.add_argument("--batch_size", type=int, default=32, help="测试扫描时的 batch size")
    parser.add_argument("--num_workers_test", type=int, default=8, help="测试 DataLoader worker 数")
    parser.add_argument("--prefetch_factor_test", type=int, default=2)

    # ---------------- 数据尺寸 ----------------
    parser.add_argument("--rgb_size", type=int, default=224)
    parser.add_argument("--depth_size", type=int, default=224)
    parser.add_argument("--rrc_scale_min", type=float, default=0.6)
    parser.add_argument("--rrc_scale_max", type=float, default=1.0)
    parser.add_argument("--rrc_ratio_min", type=float, default=0.75)
    parser.add_argument("--rrc_ratio_max", type=float, default=1.3333333333)

    # ---------------- 设备 / 随机性 ----------------
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    # ---------------- 方法选择 ----------------
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcampp"],
        choices=["ig", "guided_bp", "occlusion", "gradcampp"],
        help="可以一次指定一个或多个方法",
    )

    # ---------------- Grad-CAM++ ----------------
    parser.add_argument(
        "--gradcam_target_layer",
        type=str,
        default="auto",
        help="Grad-CAM++ 的目标层路径，例如 layer4.1、layer3.1.conv2；默认 auto=layer4 最后一个 block",
    )

    # ---------------- Integrated Gradients ----------------
    parser.add_argument("--ig_steps", type=int, default=32, help="IG 的积分步数")
    parser.add_argument(
        "--ig_internal_batch_size",
        type=int,
        default=0,
        help="IG 内部分块 batch size；<=0 表示不显式设置",
    )

    # ---------------- Occlusion ----------------
    parser.add_argument("--occlusion_window_t", type=int, default=4)
    parser.add_argument("--occlusion_window_h", type=int, default=32)
    parser.add_argument("--occlusion_window_w", type=int, default=32)
    parser.add_argument("--occlusion_stride_t", type=int, default=2)
    parser.add_argument("--occlusion_stride_h", type=int, default=16)
    parser.add_argument("--occlusion_stride_w", type=int, default=16)
    parser.add_argument("--occlusion_baseline", type=float, default=0.0)
    parser.add_argument("--occlusion_perturbations_per_eval", type=int, default=1)
    parser.add_argument("--occlusion_show_progress", action="store_true")

    # ---------------- 输入级归因聚合/可视化 ----------------
    parser.add_argument(
        "--attr_reduction",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_sum", "mean", "sum"],
        help="将 [C,T,H,W] 聚合成 [T,H,W] 的方式",
    )
    parser.add_argument(
        "--heat_normalize_mode",
        type=str,
        default="per_frame",
        choices=["per_frame", "global"],
        help="热图归一化方式",
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.4, help="热图叠加透明度")
    parser.add_argument("--cmap", type=str, default="jet", help="热图颜色映射")

    # ---------------- RGB 显示专用反归一化 ----------------
    parser.add_argument(
        "--rgb_mean",
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="RGB 显示用反归一化 mean（只影响显示，不影响模型输入）",
    )
    parser.add_argument(
        "--rgb_std",
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="RGB 显示用反归一化 std（只影响显示，不影响模型输入）",
    )

    # ---------------- 样本选择 ----------------
    parser.add_argument(
        "--selection_scope",
        type=str,
        default="global",
        choices=["single", "class", "global", "per_class"],
        help="single: 单一样本；class: 某类样本；global: 全集；per_class: 每个类别分别处理",
    )
    parser.add_argument(
        "--result_filter",
        type=str,
        default="all",
        choices=["all", "correct", "incorrect"],
        help="按预测结果筛选样本",
    )
    parser.add_argument(
        "--class_match_field",
        type=str,
        default="true",
        choices=["true", "pred"],
        help="类别筛选时按真实标签还是预测标签匹配；默认 true",
    )
    parser.add_argument("--class_ids", nargs="*", default=[], help="类别 id 列表")
    parser.add_argument("--class_names", nargs="*", default=[], help="类别名称列表")
    parser.add_argument(
        "--random_k",
        type=int,
        default=0,
        help="随机抽样数量；0 表示不随机抽样，保留全部满足条件样本",
    )

    # ---------------- 单一样本选择 ----------------
    parser.add_argument("--sample_index", type=int, default=None, help="按 global_index 选单样本")
    parser.add_argument("--sample_name", type=str, default=None, help="按 sample_name 选单样本")
    parser.add_argument("--sample_key", type=str, default=None, help="按 key/original_key 选单样本")

    # ---------------- 错误样本 target ----------------
    parser.add_argument(
        "--incorrect_target_mode",
        type=str,
        default="pred",
        choices=["pred", "true", "both"],
        help="对预测错误样本，按 pred / true / both 做解释",
    )

    return parser


# ============================================================
# 参数检查
# ============================================================
def validate_args(args) -> None:
    """
    对 CLI 参数做基础合法性检查。
    """
    normalize_run_pairs(args)

    if args.selection_scope == "single":
        num_id = int(args.sample_index is not None) + int(args.sample_name is not None) + int(args.sample_key is not None)
        if num_id == 0:
            raise ValueError(
                "selection_scope=single 时，必须提供 --sample_index / --sample_name / --sample_key 中至少一个。"
            )

    if args.selection_scope == "class" and len(args.class_ids) == 0 and len(args.class_names) == 0:
        raise ValueError(
            "selection_scope=class 时，必须提供 --class_ids 或 --class_names。"
        )

    if args.selection_scope == "per_class" and args.sample_index is not None:
        raise ValueError("selection_scope=per_class 时，不应再提供 --sample_index。")

    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("overlay_alpha must be in [0,1].")

    if args.occlusion_window_t <= 0 or args.occlusion_window_h <= 0 or args.occlusion_window_w <= 0:
        raise ValueError("Occlusion window sizes must be positive.")

    if args.occlusion_stride_t <= 0 or args.occlusion_stride_h <= 0 or args.occlusion_stride_w <= 0:
        raise ValueError("Occlusion strides must be positive.")


# ============================================================
# 单个权重的完整运行流程
# ============================================================
def run_one_weight(
    args,
    dataset,
    testloader,
    device: torch.device,
    reverse_label_map: dict[int, str],
    forward_label_map: dict[str, int],
    weight_path: str,
    save_root: str,
    reference_selected_infos: list[SampleInfo] | None = None,
    run_index: int = 0,
) -> list[SampleInfo]:
    """
    处理单个权重对应的一次完整运行：

    1) 更新当前 run 的上下文：
       - current_weight_path
       - current_save_root
       - current_weight_stem

    2) 构建模型并加载当前权重
    3) 解析 Grad-CAM++ 目标层
    4) 扫描测试集，得到当前权重下的预测结果
    5) 按当前权重下的预测结果选择样本
    6) 执行可视化
    7) 写出 run_info / visualization_index.jsonl / visualization_index.json
    """
    args_run = copy.deepcopy(args)
    args_run.current_weight_path = str(weight_path)
    args_run.current_save_root = str(save_root)
    args_run.current_weight_stem = Path(weight_path).stem
    args_run.run_index = int(run_index)
    args_run.is_reference_run = reference_selected_infos is None
    args_run.force_both_targets_for_incorrect = bool(reference_selected_infos is not None)

    ensure_dir(args_run.current_save_root)

    print("=" * 90)
    print(f"[Run] Weight : {args_run.current_weight_path}")
    print(f"[Run] Output : {args_run.current_save_root}")

    # 1) 构建模型并加载权重
    model = prepare_model(args_run).to(device)
    weight_load_report = load_model_weights_for_eval(model, args_run.current_weight_path, map_location=str(device))
    model.eval()

    # 2) 解析 Grad-CAM++ 目标层
    if args_run.gradcam_target_layer == "auto":
        args_run.gradcam_target_layer_resolved = get_default_gradcam_layer_path(model)
    else:
        args_run.gradcam_target_layer_resolved = args_run.gradcam_target_layer
    target_layer = resolve_module_by_path(model, args_run.gradcam_target_layer_resolved)

    # 3) 扫描测试集
    infos = scan_test_predictions(model, testloader, device, args_run, reverse_label_map)
    print(f"[Info] Scanned {len(infos)} samples for weight: {args_run.current_weight_stem}")

    # 4) 选择样本
    if reference_selected_infos is None:
        selected_infos = select_samples(infos, args_run, forward_label_map)
        print(f"[Info] Selected {len(selected_infos)} samples for visualization.")
    else:
        selected_infos = select_samples_by_reference_indices(infos, reference_selected_infos)
        print(
            f"[Info] Reused {len(selected_infos)} reference-selected samples from the first weight "
            f"for current weight: {args_run.current_weight_stem}"
        )

    # 5) 保存本次运行的整体信息
    run_info = {
        "save_root": str(Path(args_run.current_save_root).resolve()),
        "dataset_root": str(args_run.dataset_root),
        "label_map_json": str(args_run.label_map_json),
        "test_manifest": str(args_run.test_manifest),
        "weight_path": str(args_run.current_weight_path),
        "weight_stem": args_run.current_weight_stem,
        "device": str(device),
        "methods": list(args_run.methods),
        "model_depth": args_run.model_depth,
        "num_classes": args_run.num_classes,
        "tier_mode": args_run.tier_mode,
        "use_modality": args_run.use_modality,
        "n_frames": args_run.n_frames,
        "selection_scope": args_run.selection_scope,
        "result_filter": args_run.result_filter,
        "class_match_field": args_run.class_match_field,
        "class_ids": list(args_run.class_ids),
        "class_names": list(args_run.class_names),
        "random_k": args_run.random_k,
        "incorrect_target_mode": args_run.incorrect_target_mode,
        "force_both_targets_for_incorrect": args_run.force_both_targets_for_incorrect,
        "gradcam_target_layer": args_run.gradcam_target_layer_resolved,
        "heat_normalize_mode": args_run.heat_normalize_mode,
        "overlay_alpha": args_run.overlay_alpha,
        "attr_reduction": args_run.attr_reduction,
        "rgb_mean": args_run.rgb_mean,
        "rgb_std": args_run.rgb_std,
        "num_scanned_samples": len(infos),
        "num_selected_samples": len(selected_infos),
        "weight_load_report": weight_load_report,
        "short_path_strategy": {
            "enabled": True,
            "description": (
                "Visualization directories use short names like vis_000123_ab12cd34 "
                "to avoid Windows path length issues. Full sample/key/target info is "
                "stored in visualization_index.jsonl / visualization_index.json / meta.json."
            ),
        },
        "multi_weight_mode": {
            "enabled": True,
            "note": (
                "This save_root corresponds to exactly one weight in the user-provided "
                "ordered mapping of weight_paths <-> save_roots."
            ),
        },
    }
    write_json(Path(args_run.current_save_root) / "run_info.json", run_info)

    if len(selected_infos) == 0:
        print("[Warning] No samples matched the current selection rule. Index files will be empty.")
        write_jsonl(Path(args_run.current_save_root) / "visualization_index.jsonl", [])
        write_json(Path(args_run.current_save_root) / "visualization_index.json", [])
        return selected_infos

    # 6) 执行可视化
    all_index_records: list[dict[str, Any]] = []
    for info in tqdm(selected_infos, desc=f"Visualizing [{args_run.current_weight_stem}]", dynamic_ncols=True):
        records = visualize_one_sample(
            model=model,
            dataset=dataset,
            info=info,
            device=device,
            args=args_run,
            target_layer=target_layer,
        )
        all_index_records.extend(records)

    # 7) 写总索引
    write_jsonl(Path(args_run.current_save_root) / "visualization_index.jsonl", all_index_records)
    write_json(Path(args_run.current_save_root) / "visualization_index.json", all_index_records)

    print(f"[Done] Finished weight: {args_run.current_weight_stem}")
    print(f"[Done] Saved to    : {args_run.current_save_root}")
    return selected_infos


# ============================================================
# 主流程
# ============================================================
def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    args.rgb_mean = parse_float_list(args.rgb_mean, 3, "rgb_mean")
    args.rgb_std = parse_float_list(args.rgb_std, 3, "rgb_std")

    set_seed(args.seed)
    device = torch.device(args.device)

    # 1) 解析运行对
    run_pairs = normalize_run_pairs(args)
    print(f"[Info] Total runs to execute: {len(run_pairs)}")

    # 2) 构建标签映射
    reverse_label_map = build_reverse_label_map(args.label_map_json, args.tier_mode)
    forward_label_map = build_forward_label_map(args.label_map_json, args.tier_mode)

    # 3) 测试集和 loader 与权重无关，因此只构建一次即可复用
    dataset, testloader = prepare_test_loader(args)

    # 4) 依次处理每个权重 / 保存路径对
    #    - 第一个权重：按用户规则真正筛选样本
    #    - 后续权重：复用第一个权重已经选中的样本集合（按 global_index 对齐）
    #    - 对于后续权重，只要当前权重对某个复用样本预测错误，就强制同时输出 pred / true 两套可视化
    reference_selected_infos: list[SampleInfo] | None = None

    for idx, (weight_path, save_root) in enumerate(run_pairs, start=1):
        print("\n" + "#" * 100)
        print(f"[Progress] Run {idx}/{len(run_pairs)}")
        selected_infos = run_one_weight(
            args=args,
            dataset=dataset,
            testloader=testloader,
            device=device,
            reverse_label_map=reverse_label_map,
            forward_label_map=forward_label_map,
            weight_path=weight_path,
            save_root=save_root,
            reference_selected_infos=reference_selected_infos,
            run_index=idx - 1,
        )
        if idx == 1:
            reference_selected_infos = selected_infos

    print("\n[All Done] All weight/save_root pairs have been processed.")


if __name__ == "__main__":
    main()
