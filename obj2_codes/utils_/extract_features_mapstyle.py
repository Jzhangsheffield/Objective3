#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
extract_features_mapstyle_keys.py
------------------------------------------------------------
基于 map-style dataloader 提取模型特征。

本版本按你的最新要求实现：
1) 普通 R3D 分类模型
   - 正常构建并正常加载 checkpoint
   - 默认提取分类头 fc 的输入特征
   - 保存 preds.npy / probs_true.npy

2) MoCo 对比学习模型
   - 先完整构建一个 MoCo3D 模型
   - 从 checkpoint 的 state_dict 正确加载整个 MoCo3D
   - 然后只取其中的 encoder_q 作为真正的特征提取模型
   - 默认提取：
       a) 投影头输入特征（fc:input）
       b) 投影头输出特征（fc:output，未归一化）
   - 可额外提取 q_norm（encoder_q 输出后做 L2 normalize）
   - 也支持显式指定 encoder_q 内任意模块

重要设计说明：
- 对于 MoCo，这个脚本不会调用完整的 MoCo.forward(...)
- 只会调用 encoder_q(x)
- 因此不会触发 queue、momentum encoder、enqueue/dequeue 等训练逻辑
- 这正是离线特征提取时最干净的方式

输出文件：
- embeddings_<feature_name>.npy   : 每个目标特征一个文件，形状 [N, D]
- embeddings.npy                  : 当只提取一个目标特征时，额外保存兼容文件名
- labels.npy                      : [N]
- preds.npy                       : [N]，仅 R3D 保存
- probs_true.npy                  : [N]，仅 R3D 保存
- global_indices.npy              : [N]，若 batch 中提供 global_index / idx / sample_id
- keys.txt                        : [N] 行，样本主标识（严格使用 dataloader 输出的 key）
- sample_records.jsonl            : 每行一个样本的对齐信息，顺序与特征行严格一致
- meta.json                       : 参数、特征文件名、形状等元信息
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


# ============================================================
# 0) 让脚本在 utils_ 目录下单独运行时也能导入同级 backbone / utils_
# ============================================================
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import backbone.resnet as resnet3d
from backbone.MoCo_VAR_supcon_wds import MoCo3D
from utils_.mapstype_dataloader_with_index import (
    PackedMultiModalConfig,
    load_label_map_json,
    build_packed_mapstyle_dataset,
    build_packed_mapstyle_loader_from_dataset,
)


# ============================================================
# 1) CLI
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Extract features from R3D or MoCo(encoder_q) on map-style dataset"
    )

    # ------------------------------
    # 数据输入
    # ------------------------------
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="map-style 数据集根目录")
    parser.add_argument("--manifest_name", type=str, required=True,
                        help="manifest 文件名，通常相对于 dataset_root，例如 test_manifest.jsonl")
    parser.add_argument("--label_map_json", type=str, required=True,
                        help="label_map.json 路径；支持绝对路径，或相对于 dataset_root 的相对路径")
    parser.add_argument("--verify_paths_on_init", action="store_true",
                        help="构建数据集时是否检查样本路径")

    # ------------------------------
    # 模型输入
    # ------------------------------
    parser.add_argument("--model_type", type=str, required=True, choices=["r3d", "moco"],
                        help="模型类型：普通分类 R3D 或 MoCo")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint 路径")

    # ------------------------------
    # 标签与模态
    # ------------------------------
    parser.add_argument("--tier_mode", type=str, default="tier1", choices=["tier1", "tier2", "tier3"],
                        help="当前使用哪个 tier 的标签")
    parser.add_argument("--use_modality", type=str, default="rgb", choices=["rgb", "depth"],
                        help="输入模态")

    # ------------------------------
    # clip 设置
    # ------------------------------
    parser.add_argument("--n_frames", type=int, default=16,
                        help="每个样本采样帧数")

    # ------------------------------
    # R3D 结构参数
    # ------------------------------
    parser.add_argument("--model_depth", type=int, default=18,
                        help="ResNet3D depth")
    parser.add_argument("--num_classes", type=int, default=17,
                        help="普通 R3D 分类模型类别数")

    # ------------------------------
    # MoCo 结构参数：必须与训练时一致
    # ------------------------------
    parser.add_argument("--moco_dim", type=int, default=128,
                        help="MoCo projection dim")
    parser.add_argument("--moco_k", type=int, default=3392,
                        help="MoCo queue size，应与训练时一致")
    parser.add_argument("--moco_m", type=float, default=0.999,
                        help="MoCo momentum，应与训练时一致")
    parser.add_argument("--moco_t", type=float, default=0.1,
                        help="MoCo temperature，应与训练时一致")
    parser.add_argument("--moco_mlp", action="store_true",
                        help="训练时若使用了 MLP projection head，这里也必须加上")

    # ------------------------------
    # 特征提取目标
    # ------------------------------
    parser.add_argument(
        "--feature_targets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "要提取的特征目标。\n"
            "支持两类写法：\n"
            "1) 别名：r3d_fc_input, r3d_fc_output, moco_proj_input, moco_proj_output, q_norm\n"
            "2) 通用模块写法：<module_name>:input 或 <module_name>:output\n"
            "   R3D 例子：fc:input, layer4:output\n"
            "   MoCo 例子：fc:input, layer4:output, encoder_q.fc:input, encoder_q.layer4:output"
        ),
    )

    # ------------------------------
    # DataLoader
    # ------------------------------
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)

    # ------------------------------
    # 输出
    # ------------------------------
    parser.add_argument("--out_dir", type=str, default="./emb_outputs",
                        help="输出目录")

    # ------------------------------
    # 推理设置
    # ------------------------------
    parser.add_argument("--enable_amp", action="store_true",
                        help="是否开启 AMP 推理（仅 cuda 上有效）")
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="只跑前 N 个 batch；-1 表示全跑")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最多保存 N 个样本；-1 表示不限制")

    return parser.parse_args()


# ============================================================
# 2) 路径解析与 DataLoader 构建
# ============================================================
def _resolve_label_map_path(args) -> Path:
    """
    解析 label_map.json 路径。

    支持：
    1) 绝对路径
    2) 相对于 dataset_root 的相对路径
    """
    path = Path(args.label_map_json)
    if path.is_absolute():
        return path
    return Path(args.dataset_root) / path



def prepare_loader_det(args):
    """
    构建确定性的 map-style 特征提取 DataLoader。

    关键设置：
    1) rgb_two_views=False
       特征提取只需要单个 clip。

    2) is_train=False
       使用验证式、确定性的变换，避免随机增强影响特征稳定性。

    3) shuffle=False, drop_last=False
       必须完整扫描 manifest，并保持输出顺序与保存顺序一致。
    """
    label_map_path = _resolve_label_map_path(args)
    label_map = load_label_map_json(str(label_map_path))

    cfg_loader = PackedMultiModalConfig(
        n_frames=args.n_frames,
        rgb_two_views=False,
        use_modalities=(args.use_modality,),
        missing_policy="skip",
        load_labels=True,
        tier_mode=args.tier_mode,
        is_train=False,
        label_map_path=str(label_map_path),
    )

    dataset = build_packed_mapstyle_dataset(
        dataset_root=args.dataset_root,
        manifest_name=args.manifest_name,
        cfg=cfg_loader,
        label_map=label_map,
        verify_paths_on_init=args.verify_paths_on_init,
    )

    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        sampler=None,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    return loader


# ============================================================
# 3) 从 map-style batch 中提取输入、标签与样本信息
# ============================================================
def _get_single_clip_from_batch(batch: Dict[str, Any], use_modality: str) -> torch.Tensor:
    """
    从 batch 中取出单个 clip。

    期望：
    - rgb_two_views=False
    - 当前只使用一个模态

    支持：
    - batch[modality] 直接是 Tensor[B, T, C, H, W]
    - batch[modality] 是 dict，但其中只有一个 Tensor 分支
    """
    if use_modality not in batch:
        raise KeyError(f"Batch does not contain modality '{use_modality}'")

    x = batch[use_modality]

    if torch.is_tensor(x):
        return x

    if isinstance(x, dict):
        if len(x) != 1:
            raise RuntimeError(
                f"Expected a single '{use_modality}' tensor branch, but got keys={list(x.keys())}"
            )
        x = next(iter(x.values()))
        if not torch.is_tensor(x):
            raise TypeError(
                f"Batch modality '{use_modality}' dict value is not a tensor, got type={type(x)}"
            )
        return x

    raise TypeError(
        f"Batch modality '{use_modality}' must be Tensor or dict[str, Tensor], got type={type(x)}"
    )



def _get_labels_from_batch(batch: Dict[str, Any], tier_mode: str) -> torch.Tensor:
    """
    从 batch 中取出当前 tier 的标签张量。
    """
    if "tier_ids" not in batch:
        raise KeyError("Batch does not contain 'tier_ids'")

    tier_ids = batch["tier_ids"]
    labels = tier_ids[tier_mode] if isinstance(tier_ids, dict) else tier_ids

    if not torch.is_tensor(labels):
        raise TypeError(f"Labels must be a tensor, got type={type(labels)}")
    return labels



def _batch_field_to_list(value: Any, batch_size: int, field_name: str) -> List[Any]:
    """
    将 batch 中某个字段转换成长度为 batch_size 的 Python 列表。

    支持输入类型：
    - torch.Tensor
    - np.ndarray
    - list / tuple
    - 当 batch_size == 1 时，也允许单个标量值

    设计原则：
    - 不做兜底猜测
    - 只接受明确可按 batch 维展开的结构
    - 长度不匹配直接报错
    """
    if torch.is_tensor(value):
        if value.shape[0] != batch_size:
            raise ValueError(
                f"Field '{field_name}' tensor first dimension mismatch: "
                f"expect {batch_size}, got {value.shape[0]}"
            )
        value_cpu = value.detach().cpu()
        if value_cpu.ndim == 1:
            return [value_cpu[i].item() for i in range(batch_size)]
        return [value_cpu[i].tolist() for i in range(batch_size)]

    if isinstance(value, np.ndarray):
        if value.shape[0] != batch_size:
            raise ValueError(
                f"Field '{field_name}' ndarray first dimension mismatch: "
                f"expect {batch_size}, got {value.shape[0]}"
            )
        if value.ndim == 1:
            return [value[i].item() for i in range(batch_size)]
        return [value[i].tolist() for i in range(batch_size)]

    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(
                f"Field '{field_name}' list length mismatch: "
                f"expect {batch_size}, got {len(value)}"
            )
        return list(value)

    if batch_size == 1:
        return [value]

    raise TypeError(
        f"Field '{field_name}' cannot be converted to per-sample list, "
        f"got type={type(value)}"
    )
    


def _ensure_bcthw(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    将 [B, T, C, H, W] 转成 3D CNN 所需的 [B, C, T, H, W]。
    """
    if x_btchw.ndim != 5:
        raise ValueError(f"Expect [B,T,C,H,W], got shape={tuple(x_btchw.shape)}")
    return x_btchw.permute(0, 2, 1, 3, 4).contiguous()



def _preprocess_input(x_btchw: torch.Tensor) -> torch.Tensor:
    """
    只统一 dtype 到 float32。

    resize / normalize / sampling 已经由 dataloader 完成，这里不重复做。
    """
    if x_btchw.dtype != torch.float32:
        x_btchw = x_btchw.to(torch.float32)
    return x_btchw



def _require_key_list(batch: Dict[str, Any], batch_size: int) -> List[str]:
    """
    严格从 batch 中读取 key，并转换成长度为 B 的字符串列表。

    设计原则：
    - key 是当前 map-style dataloader 中最关键、最稳定的样本标识。
    - 本脚本不再使用 uid / sample_name 作为主标识。
    - 若 batch 中不存在 key，直接报错，不做回退。
    """
    if "key" not in batch:
        raise KeyError("Batch does not contain required field 'key'.")

    key_values = _batch_field_to_list(batch["key"], batch_size, "key")
    return [str(x) for x in key_values]


def _extract_current_tier_label_names(
    batch: Dict[str, Any],
    tier_mode: str,
    batch_size: int,
) -> Optional[List[Optional[str]]]:
    """
    从 batch['tier_actions'] 中提取当前 tier_mode 对应的字符串标签名。

    返回：
    - 长度为 B 的列表，例如 ["take", "push", ...]
    - 若 batch 中没有 tier_actions，则返回 None

    注意：
    - 这里只提取“当前分析 tier”的标签名，不保存整个多层级字典，
      这样 sample_records.jsonl 更简洁，也更适合后续检索。
    """
    if "tier_actions" not in batch:
        return None

    tier_actions = batch["tier_actions"]
    if not isinstance(tier_actions, dict):
        raise TypeError(f"Batch field 'tier_actions' must be a dict, got type={type(tier_actions)}")
    if tier_mode not in tier_actions:
        raise KeyError(f"Batch field 'tier_actions' does not contain tier '{tier_mode}'")

    values = _batch_field_to_list(tier_actions[tier_mode], batch_size, f"tier_actions[{tier_mode}]")
    out: List[Optional[str]] = []
    for v in values:
        out.append(None if v is None else str(v))
    return out


def _extract_sample_records(
    batch: Dict[str, Any],
    labels: torch.Tensor,
    tier_mode: str,
    batch_keep: int,
) -> Tuple[List[str], List[Dict[str, Any]], Optional[np.ndarray]]:
    """
    从 batch 中抽取与每一行特征严格对齐的样本级信息。

    新版设计：
    1) 不再保存 uid，也不再自动猜测主标识字段。
    2) 严格使用 dataloader 提供的 key 作为样本主标识。
    3) sample_records.jsonl 以 key + label 为核心，并保留 batch 中真实存在的辅助字段。

    返回：
    - key_list_this_batch: 与当前 batch 保存的特征逐行对齐
    - records_this_batch: 每个样本一条记录
    - global_indices: 若 batch 中提供 global_index / idx，则返回对应数组
    """
    batch_size = int(labels.shape[0])
    key_values = _require_key_list(batch, batch_size)
    labels_list = [int(x) for x in labels.detach().cpu().tolist()]
    label_names = _extract_current_tier_label_names(batch, tier_mode, batch_size)

    optional_fields = [
        "sample_name",
        "lighting",
        "pos",
        "global_index",
        "idx",
    ]

    extracted_optional: Dict[str, List[Any]] = {}
    for field_name in optional_fields:
        if field_name not in batch:
            continue
        extracted_optional[field_name] = _batch_field_to_list(batch[field_name], batch_size, field_name)

    global_indices = None
    for gi_key in ["global_index", "idx"]:
        if gi_key in extracted_optional:
            global_indices = np.asarray(extracted_optional[gi_key], dtype=np.int64)
            break

    key_list_this_batch: List[str] = []
    records_this_batch: List[Dict[str, Any]] = []

    for i in range(batch_keep):
        key_str = str(key_values[i])
        key_list_this_batch.append(key_str)

        record: Dict[str, Any] = {
            "key": key_str,
            "label_id": labels_list[i],
        }
        if label_names is not None:
            record["label_name"] = label_names[i]
        for field_name, values in extracted_optional.items():
            record[field_name] = values[i]
        records_this_batch.append(record)

    if global_indices is not None:
        global_indices = global_indices[:batch_keep]

    return key_list_this_batch, records_this_batch, global_indices


# ============================================================
# 4) 特征目标定义与 hook 管理
# ============================================================
@dataclass(frozen=True)
class FeatureRequest:
    """
    特征提取请求。

    name:
        保存时使用的名字。

    module_name + capture:
        常规模块 hook 的定义。

    special:
        不是靠 hook 抓取，而是脚本直接生成的特殊特征。
        当前只支持：q_norm
    """
    name: str
    module_name: Optional[str] = None
    capture: Optional[str] = None
    special: Optional[str] = None


ALIAS_TO_REQUEST_R3D = {
    "r3d_fc_input": FeatureRequest(name="r3d_fc_input", module_name="fc", capture="input"),
    "r3d_fc_output": FeatureRequest(name="r3d_fc_output", module_name="fc", capture="output"),
}

ALIAS_TO_REQUEST_MOCO = {
    "moco_proj_input": FeatureRequest(name="moco_proj_input", module_name="fc", capture="input"),
    "moco_proj_output": FeatureRequest(name="moco_proj_output", module_name="fc", capture="output"),
    "q_norm": FeatureRequest(name="q_norm", special="q_norm"),
}



def default_feature_requests(model_type: str) -> List[FeatureRequest]:
    if model_type == "r3d":
        return [ALIAS_TO_REQUEST_R3D["r3d_fc_input"]]
    if model_type == "moco":
        return [
            ALIAS_TO_REQUEST_MOCO["moco_proj_input"],
            ALIAS_TO_REQUEST_MOCO["moco_proj_output"],
        ]
    raise ValueError(f"Unsupported model_type: {model_type}")



def _sanitize_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ["_", "-", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).replace(".", "_")



def _normalize_custom_module_name_for_model_type(module_name: str, model_type: str) -> str:
    """
    规范化用户写的模块名。

    规则：
    - R3D：模块名按模型本身处理，不做特殊改写
    - MoCo：真正执行 forward 的模型是 encoder_q，因此：
        * 若用户写 encoder_q.xxx，则自动去掉 encoder_q.
        * 若用户写 encoder_k.xxx，则直接报错，因为本脚本不从 encoder_k 提特征
    """
    module_name = module_name.strip()

    if model_type == "moco":
        if module_name == "encoder_q":
            return ""
        if module_name.startswith("encoder_q."):
            return module_name[len("encoder_q."):]
        if module_name == "encoder_k" or module_name.startswith("encoder_k."):
            raise ValueError(
                "This script extracts MoCo features only from encoder_q. "
                "Please target encoder_q or omit the prefix."
            )

    return module_name



def parse_feature_targets(args) -> List[FeatureRequest]:
    """
    解析用户指定的特征目标。

    支持：
    - 别名：r3d_fc_input / moco_proj_input / q_norm 等
    - 显式模块：fc:input, layer4:output, encoder_q.fc:input ...
    """
    if args.feature_targets is None:
        return default_feature_requests(args.model_type)

    requests: List[FeatureRequest] = []
    seen_names = set()

    for spec in args.feature_targets:
        spec = spec.strip()
        if not spec:
            continue

        if args.model_type == "r3d" and spec in ALIAS_TO_REQUEST_R3D:
            req = ALIAS_TO_REQUEST_R3D[spec]
        elif args.model_type == "moco" and spec in ALIAS_TO_REQUEST_MOCO:
            req = ALIAS_TO_REQUEST_MOCO[spec]
        else:
            if ":" not in spec:
                raise ValueError(
                    f"Invalid feature target: '{spec}'. Use alias name or '<module_name>:input/output'."
                )
            module_name, capture = spec.rsplit(":", 1)
            module_name = _normalize_custom_module_name_for_model_type(module_name, args.model_type)
            capture = capture.strip()

            if capture not in {"input", "output"}:
                raise ValueError(f"Invalid capture type in '{spec}'. Expect input or output.")

            display_module_name = module_name if module_name != "" else "root"
            req = FeatureRequest(
                name=_sanitize_name(f"{display_module_name}_{capture}"),
                module_name=module_name,
                capture=capture,
            )

        if req.name in seen_names:
            raise ValueError(f"Duplicate feature target name: {req.name}")
        seen_names.add(req.name)
        requests.append(req)

    if not requests:
        raise ValueError("No valid feature target is provided.")
    return requests


class SingleHookBuffer:
    def __init__(self, request: FeatureRequest):
        self.request = request
        self.buffer: Optional[torch.Tensor] = None

    def _pick_tensor(self, obj, where: str) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
            return obj[0]
        raise TypeError(
            f"Feature target '{self.request.name}' captured non-tensor {where}. "
            f"Please change the target to a module whose {where} is a tensor."
        )

    def __call__(self, module, inputs, outputs):
        if self.request.capture == "input":
            if len(inputs) == 0:
                raise RuntimeError(f"Feature target '{self.request.name}' captured empty inputs.")
            x = self._pick_tensor(inputs[0], "input")
        elif self.request.capture == "output":
            x = self._pick_tensor(outputs, "output")
        else:
            raise ValueError(f"Unknown capture type: {self.request.capture}")
        self.buffer = x.detach()


class FeatureHookManager:
    """
    为所有需要通过 forward hook 抓取的特征统一管理 hook。
    """
    def __init__(self, model: nn.Module, requests: List[FeatureRequest]):
        self.model = model
        self.requests = requests
        self.handles: List[Any] = []
        self.hook_buffers: Dict[str, SingleHookBuffer] = {}

        named_modules = dict(model.named_modules())
        # 顶层模块名设为空字符串，便于需要时支持 root:output 这类目标
        named_modules[""] = model

        for req in requests:
            if req.special is not None:
                continue

            if req.module_name not in named_modules:
                available = sorted(named_modules.keys())
                preview = ", ".join([x if x != "" else "<root>" for x in available[:40]])
                raise KeyError(
                    f"Module '{req.module_name}' not found for feature target '{req.name}'.\n"
                    f"Some available module names: {preview}"
                )

            buf = SingleHookBuffer(req)
            handle = named_modules[req.module_name].register_forward_hook(buf)
            self.handles.append(handle)
            self.hook_buffers[req.name] = buf

            display_module = req.module_name if req.module_name != "" else "<root>"
            print(f"[hook] {req.name} <= {display_module}:{req.capture}")

    def clear(self):
        for buf in self.hook_buffers.values():
            buf.buffer = None

    def fetch_hook_features(self) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for name, buf in self.hook_buffers.items():
            if buf.buffer is None:
                raise RuntimeError(
                    f"Feature target '{name}' did not receive data in this forward pass. "
                    f"Please check whether the target module is actually executed."
                )
            result[name] = flatten_feature_tensor(buf.buffer)
        return result

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================
# 5) checkpoint 提取与模型构建
# ============================================================
def _strip_optional_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    若 state_dict 的 key 全部带有 module. 前缀，则统一去掉。
    """
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict



def _extract_r3d_state_dict_from_ckpt(ckpt_obj) -> Dict[str, torch.Tensor]:
    """
    提取普通 R3D checkpoint 中真正的 state_dict。

    你的分类训练脚本保存形式是：
    - {"model_state_dict": ...}
    """
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
    else:
        state = ckpt_obj

    if not isinstance(state, dict):
        raise TypeError("R3D checkpoint does not contain a valid state_dict.")

    return _strip_optional_module_prefix(state)



def _extract_moco_state_dict_from_ckpt(ckpt_obj) -> Dict[str, torch.Tensor]:
    """
    提取 MoCo checkpoint 中真正的模型参数字典。

    你的 MoCo 训练脚本保存形式是：
    - {"state_dict": _unwrap_model(model).state_dict(), ...}

    注意：
    - 这里只取 state_dict
    - prototype_bank / optimizer / epoch 等其它训练状态都不会送入 model.load_state_dict
    """
    if not (isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj):
        raise TypeError("MoCo checkpoint must contain key 'state_dict'.")

    state = ckpt_obj["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("MoCo checkpoint['state_dict'] is not a valid state_dict.")

    return _strip_optional_module_prefix(state)



def build_r3d_model(args) -> nn.Module:
    """
    构建普通 R3D 分类模型。
    """
    n_input_channels = 3 if args.use_modality == "rgb" else 1
    model = resnet3d.generate_model(
        args.model_depth,
        n_input_channels=n_input_channels,
        num_classes=args.num_classes,
    )
    return model



def build_moco_full_model(args) -> MoCo3D:
    """
    构建完整的 MoCo3D 模型。

    虽然最终提特征只用 encoder_q，
    但加载 checkpoint 时必须先构建完整 MoCo3D，
    因为 checkpoint 中保存的是完整 MoCo3D 的参数结构。
    """
    n_input_channels = 3 if args.use_modality == "rgb" else 1

    base_encoder = partial(
        resnet3d.generate_model,
        model_depth=args.model_depth,
        n_input_channels=n_input_channels,
    )

    model = MoCo3D(
        base_encoder=base_encoder,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        mlp=args.moco_mlp,
        enable_kcl_loss=False,
    )
    return model



def load_models_for_extraction(args, device: torch.device) -> Tuple[nn.Module, Optional[nn.Module], str]:
    """
    根据 model_type 构建并加载模型。

    返回：
    - feature_model:
        真正用于 forward 与安装 hook 的模型
        * R3D: 整个分类模型
        * MoCo: 完整 MoCo3D 中的 encoder_q

    - full_model:
        完整模型本体
        * R3D: None
        * MoCo: 完整 MoCo3D，用于持有完整结构与已加载参数

    - feature_model_name:
        写入 meta.json，说明当前提特征时真正使用的是谁
    """
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if args.model_type == "r3d":
        model = build_r3d_model(args)
        state = _extract_r3d_state_dict_from_ckpt(ckpt)
        model.load_state_dict(state, strict=True)
        model = model.to(device)
        model.eval()
        return model, None, "r3d"

    if args.model_type == "moco":
        moco_model = build_moco_full_model(args)
        state = _extract_moco_state_dict_from_ckpt(ckpt)
        moco_model.load_state_dict(state, strict=True)
        moco_model = moco_model.to(device)
        moco_model.eval()

        # 你的最新要求：MoCo 只从 encoder_q 提特征
        feature_model = moco_model.encoder_q
        feature_model.eval()
        return feature_model, moco_model, "encoder_q"

    raise ValueError(f"Unsupported model_type: {args.model_type}")


# ============================================================
# 6) 特征张量整理
# ============================================================
def flatten_feature_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    将任意形状的中间特征展平为 [B, D]。
    """
    if x.ndim < 2:
        raise ValueError(f"Expect feature tensor with batch dimension, got shape={tuple(x.shape)}")
    return x.reshape(x.shape[0], -1).detach()


# ============================================================
# 7) 单次 forward：执行特征提取
# ============================================================
def run_forward_and_collect_features(
    feature_model: nn.Module,
    model_type: str,
    inputs_bcthw: torch.Tensor,
    hook_manager: FeatureHookManager,
    feature_requests: List[FeatureRequest],
    amp_dtype: torch.dtype,
    use_amp: bool,
):
    """
    返回：
    - feature_dict: Dict[name, Tensor[B,D]]
    - logits: Tensor[B,C] 或 None

    说明：
    - R3D：feature_model 就是整个分类模型，forward 返回 logits
    - MoCo：feature_model 就是 encoder_q，forward 返回投影特征；若请求 q_norm，则在这里额外构造
    """
    device_type = "cuda" if inputs_bcthw.device.type == "cuda" else "cpu"
    hook_manager.clear()

    if model_type == "r3d":
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = feature_model(inputs_bcthw)
        feature_dict = hook_manager.fetch_hook_features()
        return feature_dict, logits

    if model_type == "moco":
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            q_raw = feature_model(inputs_bcthw)

        feature_dict = hook_manager.fetch_hook_features() if hook_manager.hook_buffers else {}

        for req in feature_requests:
            if req.special == "q_norm":
                feature_dict[req.name] = flatten_feature_tensor(F.normalize(q_raw, dim=1).detach())

        return feature_dict, None

    raise ValueError(f"Unsupported model_type: {model_type}")


# ============================================================
# 8) 主流程：提取并保存
# ============================================================
@torch.no_grad()
def run_extract(args):
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_cuda = (device.type == "cuda")
    use_bf16 = torch.cuda.is_bf16_supported() if has_cuda else False
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = bool(args.enable_amp and has_cuda)

    # 1) loader
    loader = prepare_loader_det(args)

    # 2) model(s)
    feature_model, full_model, feature_model_name = load_models_for_extraction(args, device)

    # 3) 解析特征目标并安装 hook
    feature_requests = parse_feature_targets(args)
    print("[features] targets:", [req.name for req in feature_requests])
    hook_manager = FeatureHookManager(feature_model, feature_requests)

    # 4) 收集容器
    feature_store: Dict[str, List[np.ndarray]] = {req.name: [] for req in feature_requests}
    label_chunks: List[np.ndarray] = []
    pred_chunks: List[np.ndarray] = []
    prob_true_chunks: List[np.ndarray] = []
    global_index_chunks: List[np.ndarray] = []
    key_list: List[str] = []
    sample_records: List[Dict[str, Any]] = []

    saved = 0

    for step, batch in enumerate(loader):
        if args.max_batches > 0 and step >= args.max_batches:
            break
        if args.max_samples > 0 and saved >= args.max_samples:
            break

        inputs_btchw = _get_single_clip_from_batch(batch, args.use_modality)
        labels = _get_labels_from_batch(batch, args.tier_mode)

        if not torch.is_tensor(inputs_btchw):
            raise TypeError(f"Input clip must be a tensor, got type={type(inputs_btchw)}")

        inputs_btchw = _preprocess_input(inputs_btchw.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True).long()
        inputs_bcthw = _ensure_bcthw(inputs_btchw)

        feature_dict, logits = run_forward_and_collect_features(
            feature_model=feature_model,
            model_type=args.model_type,
            inputs_bcthw=inputs_bcthw,
            hook_manager=hook_manager,
            feature_requests=feature_requests,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )

        labels_np = labels.detach().cpu().numpy().astype(np.int64)

        preds_np = None
        prob_true_np = None
        if logits is not None:
            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=1)
            prob_true = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            preds_np = preds.detach().cpu().numpy().astype(np.int64)
            prob_true_np = prob_true.detach().cpu().numpy().astype(np.float32)

        if args.max_samples > 0:
            remain = args.max_samples - saved
            if remain <= 0:
                break
            batch_keep = min(labels_np.shape[0], remain)
        else:
            batch_keep = labels_np.shape[0]

        # 先保存特征
        for name, feat in feature_dict.items():
            feat_np = feat.detach().cpu().float().numpy()
            feature_store[name].append(feat_np[:batch_keep])

        # 再保存样本级对应关系
        key_batch, records_batch, global_idx_batch = _extract_sample_records(
            batch=batch,
            labels=labels,
            tier_mode=args.tier_mode,
            batch_keep=batch_keep,
        )

        start_row = saved
        for offset, rec in enumerate(records_batch):
            rec["row_index"] = start_row + offset
        sample_records.extend(records_batch)
        key_list.extend(key_batch)
        label_chunks.append(labels_np[:batch_keep])

        if global_idx_batch is not None:
            global_index_chunks.append(global_idx_batch.astype(np.int64))

        if preds_np is not None:
            pred_chunks.append(preds_np[:batch_keep])
        if prob_true_np is not None:
            prob_true_chunks.append(prob_true_np[:batch_keep])

        saved += batch_keep

        if step % 20 == 0:
            feat_shapes = {k: tuple(v[-1].shape) for k, v in feature_store.items() if len(v) > 0}
            print(f"[extract] step={step} saved={saved} batch_feature_shapes={feat_shapes}")

    hook_manager.remove()

    # 5) 拼接
    final_features: Dict[str, np.ndarray] = {}
    for name, chunks in feature_store.items():
        if chunks:
            arr = np.concatenate(chunks, axis=0).astype(np.float32)
        else:
            arr = np.zeros((0, 0), dtype=np.float32)
        final_features[name] = arr

    labels_all = np.concatenate(label_chunks, axis=0) if label_chunks else np.zeros((0,), dtype=np.int64)
    preds_all = np.concatenate(pred_chunks, axis=0) if pred_chunks else None
    probs_true_all = np.concatenate(prob_true_chunks, axis=0) if prob_true_chunks else None
    global_indices_all = np.concatenate(global_index_chunks, axis=0) if global_index_chunks else None

    # 6) 保存
    np.save(os.path.join(args.out_dir, "labels.npy"), labels_all.astype(np.int64))

    with open(os.path.join(args.out_dir, "keys.txt"), "w", encoding="utf-8") as f:
        for k in key_list:
            f.write(str(k) + "\n")

    if global_indices_all is not None:
        np.save(os.path.join(args.out_dir, "global_indices.npy"), global_indices_all.astype(np.int64))

    feature_file_map = {}
    for name, arr in final_features.items():
        file_name = f"embeddings_{name}.npy"
        np.save(os.path.join(args.out_dir, file_name), arr)
        feature_file_map[name] = file_name

    if len(final_features) == 1:
        only_name = next(iter(final_features.keys()))
        np.save(os.path.join(args.out_dir, "embeddings.npy"), final_features[only_name])

    if preds_all is not None:
        np.save(os.path.join(args.out_dir, "preds.npy"), preds_all.astype(np.int64))
    if probs_true_all is not None:
        np.save(os.path.join(args.out_dir, "probs_true.npy"), probs_true_all.astype(np.float32))

    with open(os.path.join(args.out_dir, "sample_records.jsonl"), "w", encoding="utf-8") as f:
        for rec in sample_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "model_type": args.model_type,
        "feature_model_name": feature_model_name,
        "ckpt": args.ckpt,
        "dataset_root": args.dataset_root,
        "manifest_name": args.manifest_name,
        "label_map_json": str(_resolve_label_map_path(args)),
        "verify_paths_on_init": bool(args.verify_paths_on_init),
        "tier_mode": args.tier_mode,
        "use_modality": args.use_modality,
        "n_frames": args.n_frames,
        "model_depth": args.model_depth,
        "num_classes": args.num_classes,
        "moco_dim": args.moco_dim,
        "moco_k": args.moco_k,
        "moco_m": args.moco_m,
        "moco_t": args.moco_t,
        "moco_mlp": bool(args.moco_mlp),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "prefetch_factor": args.prefetch_factor,
        "enable_amp": bool(args.enable_amp),
        "feature_targets": [
            {
                "name": req.name,
                "module_name": req.module_name,
                "capture": req.capture,
                "special": req.special,
            }
            for req in feature_requests
        ],
        "feature_files": feature_file_map,
        "feature_shapes": {name: list(arr.shape) for name, arr in final_features.items()},
        "num_samples": int(labels_all.shape[0]),
        "saved_preds": preds_all is not None,
        "saved_probs_true": probs_true_all is not None,
        "saved_global_indices": global_indices_all is not None,
        "saved_keys": True,
        "saved_sample_records": True,
        "note": (
            "Feature extraction uses deterministic map-style loader (is_train=False). "
            "For MoCo, the script loads the full MoCo checkpoint but extracts features only from encoder_q."
        ),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("========== Extract Done ==========")
    print("out_dir:", os.path.abspath(args.out_dir))
    for name, arr in final_features.items():
        print(f"{name}: {arr.shape} {arr.dtype}")
    print("labels:", labels_all.shape)
    print("keys:", len(key_list))
    if global_indices_all is not None:
        print("global_indices:", global_indices_all.shape)
    if preds_all is not None:
        print("preds:", preds_all.shape)
    if probs_true_all is not None:
        print("probs_true:", probs_true_all.shape)

    # 防止静态检查认为 full_model 未使用；这里显式保留引用语义
    _ = full_model



def main():
    args = get_args()
    run_extract(args)


if __name__ == "__main__":
    main()
