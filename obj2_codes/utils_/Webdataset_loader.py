#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebDataset 多模态读取器（CPU 侧：选帧 + 解码 + 组装 Tensor）

本次改动目标（按你的最新要求）：
------------------------------------------------------------
1) ✅ 关闭 pin_memory
   - Windows 下 pin_memory 线程更容易触发 CUDA "resource already mapped"。

2) ✅ 不再在 loader 内手写 v2.functional 的 crop/resize
   - 改为：在 collate_fn 之前（也就是 WebDataset 的 .map 阶段）
     直接调用你 augmentation.py 中提供的：
       - TemporallyConsistentSpatialAugmentation（训练）
       - ValidationAugmentation（验证）
   - 两个类内部会把 Tensor 包装成 tv_tensors.Video，并对整段视频一次性采样/应用增强，
     从而保证“时间一致的空间增强”。

3) ✅ out 中移除 labels
   - 仍解析 labels.json，但只在内部字段 _labels 中暂存
   - 输出只包含 tier_actions / tier_ids（以及你启用的模态数据）

4) Depth 增强：暂时不做决策（保持最稳的“只保证能 stack”策略）
   - 如果你启用 depth：仍在 loader 内做确定性 resize（NEAREST）到固定尺寸
   - 不做 Normalize（避免把 RGB 的 mean/std 用到 depth 上）
   - 你后续确定 depth 的处理方式后，我们再把 depth augmentation 接入
   
5) 使用哈希将每个样本的 key 映射为 int64 方便后续的训练和样本匹配.

------------------------------------------------------------
输出结构（单 cam 示例）：
out = {
  "key": str,
  "tier_actions": {"tier1": "..."} 或 {"tier1": ..., "tier2": ...},
  "tier_ids":     {"tier1": int}   或 {...},
  "rgb":   Tensor[T,3,H,W]  (float32, 已 ToDtype+Normalize，H/W 由 augmentation 决定)
  "depth": Tensor[T,1,H,W]  (uint16/uint8 或 float32，H/W 由 cfg.depth_out_hw 保证一致)
}

注意：
- 如果你把 RGB 的 Normalize/ToDtype 放到了 augmentation.py 中，那么训练脚本里就不要再重复 Normalize。
"""

from __future__ import annotations

import io
import json
import glob
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image

# Depth resize 仍需要 v2.functional（RGB 的空间增强已经交给 augmentation.py）
from torchvision.transforms.v2 import functional as Fv2
from torchvision.transforms import InterpolationMode

# ✅ 关键：使用你提供的“时间一致增强”
from aug.spatial_augmentation import TemporallyConsistentSpatialAugmentation, ValidationAugmentation

# ✅ 新增：two-view 时间采样（按你的最新规则：前半/后半起点 + linspace 必含最后一帧；短视频 last-frame padding）
from aug.temporal_augmentation_adaptive import sample_indices_strict, sample_two_views_indices


# ============================================================
# 1) 配置
# ============================================================

@dataclass
class MultiModalConfig:
    # -------- 视频/图像帧采样 --------
    n_frames: int = 16

    # ✅ 是否输出 two-view（用于对比学习）
    # - False：与原来行为一致，只返回一个 rgb Tensor[T,3,H,W]
    # - True ：返回 (view1, view2) 两个 rgb Tensor，每个都是 [T,3,H,W]
    #
    # 注意：当原视频帧数 T <= n_frames 时，采样器会自动 last-frame padding，
    # 且返回两个完全相同的 view（相当于不做时间增强）。
    rgb_two_views: bool = False

    # 摄像头选择：
    # - None 或 ()：禁用该模态（输出中不出现对应 key）
    # - ("001431512812",) ：启用单 cam（输出 Tensor）
    # - 多 cam：输出 dict(camid->Tensor)
    rgb_cam_ids: Optional[Tuple[str, ...]] = None
    depth_cam_ids: Optional[Tuple[str, ...]] = None

    # -------- MindRove（先留空）--------
    use_mindrove: bool = False

    # 缺失策略：
    # - "skip"：启用的任何模态缺失 => 丢弃样本（推荐）
    # - "pad" ：启用模态缺失 => 用零 Tensor 填充（保证样本不丢）
    missing_policy: str = "skip"  # "skip" or "pad"

    # -------- labels / tier 映射 --------
    load_labels: bool = True
    label_map_path: str = "label_map.json"

    # tier 输出模式：
    # - "all"：tier1/tier2/tier3 全输出
    # - "tier1"/"tier2"/"tier3"：只输出一个 tier
    tier_mode: str = "all"

    # -------- Train/Val 标志 --------
    is_train: bool = True

    # RGB 输出尺寸（由 augmentation.py 的 RandomResizedCrop/Resize 使用）
    rgb_out_hw: Tuple[int, int] = (224, 224)

    # RandomResizedCrop 参数（传给 TemporallyConsistentSpatialAugmentation）
    rrc_scale: Tuple[float, float] = (0.6, 1.0)
    rrc_ratio: Tuple[float, float] = (0.75, 1.3333333333)

    # Depth 输出尺寸（暂时保持确定性 resize 以保证 batch 可 stack）
    depth_out_hw: Tuple[int, int] = (224, 224)

    # pad 默认尺寸
    default_rgb_hw: Tuple[int, int] = (224, 224)
    default_depth_hw: Tuple[int, int] = (224, 224)

    # Depth dtype 可能是 uint16（常见）或 uint8
    default_depth_dtype: str = "uint16"  # "uint16" or "uint8"

    # ✅ 做法 A：把 RGB transform 对象存到 cfg（在 build_loader 时构造）
    # 说明：该字段不参与 dataclass 自动比较；worker 里只读取并调用
    rgb_transform: Optional[Any] = None


# ============================================================
# 2) label_map 加载与 tier 映射
# ============================================================

def load_label_map_json(path: str) -> Dict[str, Dict[str, int]]:
    """读取离线 label_map.json：{tier1:{action->id}, tier2:{...}, tier3:{...}}
        输出的 out 是一个列表，里面的元素是字典，每一个字典的 key 都是动作字符串，值是整型标签"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, Dict[str, int]] = {}
    for tier in ("tier1", "tier2", "tier3"):
        mp = obj.get(tier, {}) or {}
        if not isinstance(mp, dict):
            raise ValueError(f"label_map.json: '{tier}' must be a dict, got {type(mp)}")
        out[tier] = {str(k): int(v) for k, v in mp.items()}
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
# 3) sample 内 key 列表工具
# ============================================================

def _enabled_camids(cam_ids: Optional[Tuple[str, ...]]) -> Tuple[bool, Tuple[str, ...]]:
    """将 cam_ids 规范化：None/空 => 禁用；否则启用并返回清洗后的 tuple"""
    if cam_ids is None:
        return False, ()
    cam_ids = tuple(str(x) for x in cam_ids if str(x).strip() != "")
    if len(cam_ids) == 0:
        return False, ()
    return True, cam_ids


def list_frame_keys(sample: Dict[str, Any], modality: str, camid: str, ext: str) -> List[str]:
    """列出某个 sample 内指定 modality+cam 的帧 key，并排序保证时间顺序稳定。"""
    prefix = f"{modality}_{camid}."
    keys = [k for k in sample.keys()
            if isinstance(k, str) and k.startswith(prefix) and k.endswith(ext)]
    keys.sort()
    return keys



# ============================================================
# 4) CPU 解码：返回单帧 Tensor
# ============================================================

def decode_rgb_uint8_original(b: bytes) -> torch.Tensor:
    """RGB 解码：jpg bytes -> uint8 [3,H,W]"""
    img = Image.open(io.BytesIO(b)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)              # [H,W,3]
    t = torch.from_numpy(arr).permute(2, 0, 1)       # [3,H,W]
    return t.contiguous()


def decode_depth_keep_dtype_original(b: bytes) -> torch.Tensor:
    """Depth 解码：png bytes -> [1,H,W]，保持原 dtype（常见 uint16/uint8）"""
    img = Image.open(io.BytesIO(b))
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    t = torch.from_numpy(arr).unsqueeze(0)           # [1,H,W]
    return t.contiguous()


# ============================================================
# 5) labels.json 解析（只保留 tier1/2/3，写到 sample["_labels"]）
# ============================================================

def parse_labels_if_needed(sample: Dict[str, Any], cfg: MultiModalConfig) -> Dict[str, Any]:
    """解析 labels.json，仅保留 tier1/2/3，暂存到 _labels。"""
    if not cfg.load_labels:
        sample["_labels"] = {}
        return sample

    b = sample.get("labels.json", None)
    if b is None:
        sample["_labels"] = {}
        return sample

    try:
        obj = json.loads(b.decode("utf-8"))
        sample["_labels"] = {k: obj.get(k, None) for k in ("tier1", "tier2", "tier3")}
    except Exception:
        sample["_labels"] = {}

    return sample


def attach_tier_ids(
    out: Dict[str, Any],
    cfg: MultiModalConfig,
    label_map: Dict[str, Dict[str, int]],
) -> Optional[Dict[str, Any]]:
    """将 _labels 中的 tier action 映射为整型 id，并输出 tier_actions/tier_ids。"""
    labels = out.get("_labels", {}) or {}
    tiers = get_required_tiers(cfg.tier_mode)
    tier_actions, tier_ids = map_tier_actions_to_ids(labels, label_map, tiers)

    # tier 映射失败（-1）时，skip 模式直接丢弃
    has_bad = any(tier_ids[t] < 0 for t in tiers)
    if has_bad and cfg.missing_policy == "skip":
        return None

    out["tier_actions"] = tier_actions
    out["tier_ids"] = tier_ids

    # ✅ 彻底移除 labels，避免进 batch
    out.pop("_labels", None)

    return out


# ============================================================
# 6) Depth：确定性 resize（NEAREST）保证 batch 可 stack（暂不做更复杂增强）
# ============================================================

def _resize_depth_video_keep_dtype(video_t1hw: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Depth resize：
    - 输入：[T,1,H,W]，dtype 可能 uint16/uint8
    - 输出：[T,1,outH,outW]，dtype 尽量保持

    说明：
    torchvision 的 resize 对整型在不同版本支持不完全一致；为稳妥：
      1) 转 float32
      2) NEAREST resize
      3) round
      4) cast 回原 dtype
    """
    assert video_t1hw.ndim == 4 and video_t1hw.shape[1] == 1
    orig_dtype = video_t1hw.dtype

    out_frames = []
    for t in range(video_t1hw.shape[0]):
        fr = video_t1hw[t].to(torch.float32)
        fr = Fv2.resize(fr, size=list(out_hw), interpolation=InterpolationMode.NEAREST, antialias=False)
        if orig_dtype in (torch.uint8, torch.uint16, torch.int16, torch.int32, torch.int64):
            fr = torch.round(fr)
        fr = fr.to(orig_dtype)
        out_frames.append(fr)

    return torch.stack(out_frames, dim=0).contiguous()


# ============================================================
# 7) RGB / Depth 读取 +（RGB 调 augmentation.py）
# ============================================================

def _make_zero_rgb(cfg: MultiModalConfig) -> torch.Tensor:
    """pad 模式下 RGB 零视频：uint8 [T,3,H,W]（注意：若启用 rgb_transform，会在后面再 transform）"""
    H, W = cfg.default_rgb_hw
    return torch.zeros((cfg.n_frames, 3, H, W), dtype=torch.uint8)


def _make_zero_depth(cfg: MultiModalConfig) -> torch.Tensor:
    """pad 模式下 Depth 零视频：[T,1,H,W]"""
    H, W = cfg.default_depth_hw
    dtype = torch.uint16 if str(cfg.default_depth_dtype).lower() == "uint16" else torch.uint8
    return torch.zeros((cfg.n_frames, 1, H, W), dtype=dtype)


def load_modalities(sample: Dict[str, Any], cfg: MultiModalConfig, key_to_index: Optional[Dict[str, int]]=None
                    ) -> Optional[Dict[str, Any]]:
    """根据 cfg 选择性输出 rgb/depth，并在 collate 前完成必要处理。"""

    if cfg.use_mindrove:
        raise NotImplementedError(
            "MindRove (EMG+IMU) 处理尚未确定，本版本暂不支持。请先设置 cfg.use_mindrove=False。"
        )

    # ✅ labels 不进 out，仅保留内部 _labels（attach_tier_ids 会 pop 掉）
    
    key = sample.get("__key__", "")
    # url = sample.get("__url__", "")  # WebDataset 通常会带这个；没有也没关系
    # uid = f"{url}::{key}" if url else key
    uid = key

    if not key:
        if cfg.missing_policy == "skip":
            return None
        raise RuntimeError("当前样本缺少 __key__，无法从 sample_index_map_json 中查找连续 idx。")

    if key_to_index is None:
        raise RuntimeError(
            "key_to_index is None. "
            "请在 build_multimodal_wds_loader 中加载 sample_index_map_json，"
            "并通过 partial 传给 load_modalities。"
        )

    if key not in key_to_index:
        raise KeyError(
            f"样本 key 不在 sample_index_map_json 中: {key}"
        )

    idx = int(key_to_index[key])

    out: Dict[str, Any] = {
        "key": key,
        "uid": uid,
        "idx": idx,                  # ✅ 使用连续 index，不再使用哈希 int64
        "_labels": sample.get("_labels", {}),
    }

    # ---------------- RGB ----------------
    rgb_enabled, rgb_camids = _enabled_camids(cfg.rgb_cam_ids)
    if rgb_enabled:
        rgb_out: Dict[str, torch.Tensor] = {}
        for camid in rgb_camids:
            frame_keys = list_frame_keys(sample, "rgb", camid, ext=".jpg")
            T = len(frame_keys)
            if T == 0:
                continue

            # ------------------------------------------------------------
            # ✅ 时间采样（支持 two-view）
            #
            # 旧的“全局均匀采样 n_frames”仍保留为备用（先注释掉）：
            #   idxs = sample_indices_strict(T, cfg.n_frames)
            #
            # 当前使用的新规则（见 temporal_two_view_sampling_halves_linspace_lastpad.py）：
            # - 若 T > n_frames：
            #     max_start = T - n_frames
            #     start1 ∈ [0, max_start//2], start2 ∈ [max_start//2, max_start]
            #     每个 view：idxs = round(linspace(start, T-1, n_frames))，并强制包含最后一帧 T-1
            # - 若 T <= n_frames：
            #     last-frame padding 到 n_frames，并返回两个相同的 view（不做时间增强）
            # ------------------------------------------------------------
            if cfg.rgb_two_views:
                idxs1, idxs2 = sample_two_views_indices(T=T, n=cfg.n_frames)
                idxs_list = [idxs1, idxs2]
            else:
                idxs_list = [sample_indices_strict(T, cfg.n_frames)]

            # ✅ 关键：在 collate 前使用你 augmentation.py 的“时间一致增强”
            #    - 每个 view 独立调用一次 transform（空间增强参数不同）
            #    - 但单个 view 内部所有帧共享同一组空间增强参数（时间一致）
            if cfg.rgb_transform is None:
                # 这通常意味着 build_multimodal_wds_loader 没有正确为 cfg 配置 transform
                raise RuntimeError(
                    "cfg.rgb_transform is None. "
                    "请在 build_multimodal_wds_loader 中为 train/val 分别设置 "
                    "TemporallyConsistentSpatialAugmentation / ValidationAugmentation。"
                )

            views: List[torch.Tensor] = []
            for idxs in idxs_list:
                # idxs: List[int]，长度为 cfg.n_frames
                b0 = sample.get(frame_keys[idxs[0]], None)
                if b0 is None:
                    # tar 中该帧缺失：跳过该 view
                    continue

                f0 = decode_rgb_uint8_original(b0)  # [3,H,W] uint8
                H, W = int(f0.shape[1]), int(f0.shape[2])

                # 先把 n_frames 帧解码成一个 uint8 video: [T,3,H,W]
                vid = torch.empty((cfg.n_frames, 3, H, W), dtype=f0.dtype)
                vid[0] = f0
                for j in range(1, cfg.n_frames):
                    bj = sample.get(frame_keys[idxs[j]], None)
                    if bj is None:
                        # tar 缺帧：用上一帧顶上，避免 shape 变化导致 stack 失败
                        vid[j] = vid[j - 1]
                    else:
                        vid[j] = decode_rgb_uint8_original(bj)

                # 对该 view 做“时间一致的空间增强”（内部会 ToDtype+Normalize）
                v = cfg.rgb_transform(vid)
                v = torch.as_tensor(v).contiguous()
                views.append(v)

            # 组装输出：single-view 或 two-view
            if cfg.rgb_two_views:
                if len(views) != 2:
                    # 采样/解码异常时按 missing_policy 处理
                    if cfg.missing_policy == "skip":
                        return None
                    # pad：用同一个 zero clip 复制两份
                    z = _make_zero_rgb(cfg)
                    z = torch.as_tensor(cfg.rgb_transform(z)).contiguous()
                    views = [z.clone(), z.clone()]
                rgb_out[camid] = (views[0], views[1])
            else:
                if len(views) != 1:
                    if cfg.missing_policy == "skip":
                        return None
                    z = _make_zero_rgb(cfg)
                    z = torch.as_tensor(cfg.rgb_transform(z)).contiguous()
                    views = [z]
                rgb_out[camid] = views[0]

        if len(rgb_out) == 0:
            if cfg.missing_policy == "skip":
                return None

            # pad：先造 uint8 零视频，再过同样的 rgb_transform
            z = _make_zero_rgb(cfg)
            z = torch.as_tensor(cfg.rgb_transform(z)).contiguous()
            for camid in rgb_camids:
                if cfg.rgb_two_views:
                    rgb_out[camid] = (z.clone(), z.clone())
                else:
                    rgb_out[camid] = z.clone()

        if len(rgb_camids) == 1:
            out["rgb"] = rgb_out[rgb_camids[0]]
        else:
            out["rgb"] = rgb_out

    # ---------------- Depth ----------------
    depth_enabled, depth_camids = _enabled_camids(cfg.depth_cam_ids)
    if depth_enabled:
        depth_out: Dict[str, torch.Tensor] = {}
        for camid in depth_camids:
            frame_keys = list_frame_keys(sample, "depth", camid, ext=".png")
            T = len(frame_keys)
            if T == 0:
                continue
            idxs = sample_indices_strict(T, cfg.n_frames)

            b0 = sample.get(frame_keys[idxs[0]], None)
            if b0 is None:
                continue

            f0 = decode_depth_keep_dtype_original(b0)  # [1,H,W]
            H, W = int(f0.shape[1]), int(f0.shape[2])

            vid = torch.empty((cfg.n_frames, 1, H, W), dtype=f0.dtype)
            vid[0] = f0
            for j in range(1, cfg.n_frames):
                bj = sample.get(frame_keys[idxs[j]], None)
                if bj is None:
                    vid[j] = vid[j - 1]
                else:
                    vid[j] = decode_depth_keep_dtype_original(bj)

            # 暂时只做确定性 resize，保证 batch 维度一致
            vid = _resize_depth_video_keep_dtype(vid, out_hw=cfg.depth_out_hw)
            depth_out[camid] = vid

        if len(depth_out) == 0:
            if cfg.missing_policy == "skip":
                return None
            z = _make_zero_depth(cfg)
            z = _resize_depth_video_keep_dtype(z, out_hw=cfg.depth_out_hw)
            for camid in depth_camids:
                depth_out[camid] = z.clone()

        if len(depth_camids) == 1:
            out["depth"] = depth_out[depth_camids[0]]
        else:
            out["depth"] = depth_out

    if (not rgb_enabled) and (not depth_enabled):
        raise ValueError("At least one modality must be enabled (rgb or depth).")

    return out


# ============================================================
# 8) Windows 多进程兼容：map/select 顶层函数包装
# ============================================================

def map_parse_labels(sample: Dict[str, Any], cfg: MultiModalConfig) -> Dict[str, Any]:
    return parse_labels_if_needed(sample, cfg)


def map_load_modalities(sample: Dict[str, Any], cfg: MultiModalConfig, key_to_index: Optional[Dict[str, int]] = None,
                        ) -> Optional[Dict[str, Any]]:
    return load_modalities(sample, cfg, key_to_index=key_to_index)


def map_attach_tiers(
    out: Dict[str, Any],
    cfg: MultiModalConfig,
    label_map: Dict[str, Dict[str, int]],
) -> Optional[Dict[str, Any]]:
    return attach_tier_ids(out, cfg, label_map)


def is_not_none(x: Any) -> bool:
    return x is not None


# ============================================================
# 9) shards 路径兼容（Windows / Linux）
# ============================================================

def _looks_like_uri(s: str) -> bool:
    s = s.strip().lower()
    return (
        s.startswith("file:")
        or s.startswith("pipe:")
        or s.startswith("http://")
        or s.startswith("https://")
        or s.startswith("s3:")
        or "://" in s
    )


def normalize_shards_to_uris(shards: Union[str, List[str]]) -> List[str]:
    """将 shards 输入规范化为 WebDataset 可用的 URI 列表（支持 glob / list / uri）。"""
    if isinstance(shards, str):
        s = shards.strip()
        if _looks_like_uri(s):
            return [s]

        has_glob = any(ch in s for ch in ["*", "?", "["])
        paths = sorted(glob.glob(s)) if has_glob else [s]

        uris: List[str] = []
        for p in paths:
            pp = Path(p).expanduser().resolve()
            uris.append("file:" + pp.as_posix())
        return uris

    uris: List[str] = []
    for item in shards:
        si = str(item).strip()
        if _looks_like_uri(si):
            uris.append(si)
        else:
            pp = Path(si).expanduser().resolve()
            uris.append("file:" + pp.as_posix())
    return uris


# ============================================================
# 10) 构建 DataLoader（pin_memory=False）
# ============================================================

def build_multimodal_wds_loader(
    shards: Union[str, List[str]],
    cfg: MultiModalConfig,
    label_map: Dict[str, Dict[str, int]],
    batch_size: int = 8,
    num_workers: int = 4,
    shardshuffle: Union[int, bool] = 0,
    shuffle_samples: int = 0,
    drop_last = False,
    sample_index_map_json = None
) -> DataLoader:
    """
    返回 batch(dict)。

    ✅ 做法 A：在这里为 cfg 构造并挂载 rgb_transform（train/val 不同）：
      - cfg.is_train=True  -> TemporallyConsistentSpatialAugmentation
      - cfg.is_train=False -> ValidationAugmentation

    注意：
    - 这里构造的 transform 会随 cfg 一起通过 partial 传入 worker。
    - augmentation.py 内部已经 ToDtype+Normalize，因此训练脚本不要再重复 Normalize。
    """
    key_to_index = None
    if sample_index_map_json is not None and str(sample_index_map_json).strip() != "":
        with open(sample_index_map_json, "r", encoding="utf-8") as f:
            key_to_index = json.load(f)

        # 可选：把 value 统一转成 int，避免 JSON 里类型不稳
        key_to_index = {str(k): int(v) for k, v in key_to_index.items()}
        
    shard_uris = normalize_shards_to_uris(shards)

    # --------- 配置 RGB transform（只在启用 RGB 时需要）---------
    rgb_enabled, _ = _enabled_camids(cfg.rgb_cam_ids)
    if rgb_enabled:
        if cfg.is_train:
            cfg.rgb_transform = TemporallyConsistentSpatialAugmentation(
                size=cfg.rgb_out_hw,
                crop_scale=cfg.rrc_scale,
                crop_ratio=cfg.rrc_ratio,
            )
        else:
            cfg.rgb_transform = ValidationAugmentation(
                size=cfg.rgb_out_hw,
            )

    ds = wds.WebDataset(shard_uris, shardshuffle=shardshuffle)
    if shuffle_samples and shuffle_samples > 0:
        ds = ds.shuffle(shuffle_samples)

    parse_fn = partial(map_parse_labels, cfg=cfg)
    load_fn = partial(map_load_modalities, cfg=cfg, key_to_index=key_to_index)
    tier_fn = partial(map_attach_tiers, cfg=cfg, label_map=label_map)

    ds = (
        ds.map(parse_fn)
          .map(load_fn)
          .select(is_not_none)
          .map(tier_fn)
          .select(is_not_none)
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,                 # ✅ 关键：关闭 pin_memory
        persistent_workers=(num_workers > 0),
        drop_last=drop_last
        # 你后续若想提高吞吐，可以在这里加 prefetch_factor=4/8
    )
    return loader


# ============================================================
# 11) 自测 main（可删）
# ============================================================

def main():
    """
    自测：验证 two-view temporal sampling 是否生效（不做可视化，只打印结构/shape）。

    你需要修改下面两个路径：
    1) label_map.json 路径
    2) shards_glob（*.tar）路径
    """
    # 1) 读取离线 label_map.json
    label_map = load_label_map_json(
        r"G:\thermal_crimp\webdataset_cam001431512812\label_map.json"
    )

    # 2) 配置：开启 two-view
    cfg = MultiModalConfig(
        n_frames=16,
        rgb_cam_ids=("001431512812",),
        depth_cam_ids=(),          # 这里只测 RGB two-view
        use_mindrove=False,
        missing_policy="skip",
        tier_mode="tier1",

        is_train=True,             # True -> TemporallyConsistentSpatialAugmentation；False -> ValidationAugmentation
        rgb_out_hw=(224, 224),

        rgb_two_views=True,        # ✅ 开启 two-view
    )

    # 3) shards（按你自己的路径修改）
    shards_glob = r"G:\thermal_crimp\webdataset_cam001431512812\train\shards\*.tar"

    # 4) 构建 DataLoader：为排错建议先 num_workers=0；确认 OK 后再开多进程
    loader = build_multimodal_wds_loader(
        shards=shards_glob,
        cfg=cfg,
        label_map=label_map,
        batch_size=2,
        num_workers=0,
        shardshuffle=False,
        shuffle_samples=0,
        sample_index_map_json=r"G:\thermal_crimp\webdataset_cam001431512812\train\train_sample_index_map\key_to_index.json"
    )

    # 5) 取一个 batch 验证输出结构
    for i, batch in enumerate(loader):
        print(f"\n[batch {i}] keys={list(batch.keys())}")
        print("  key:", batch.get("key"))
        print("idx", batch.get("idx"))

        if "tier_ids" in batch:
            print("  tier_ids:", batch["tier_ids"])

        rgb = batch.get("rgb", None)
        if rgb is None:
            print("  ERROR: batch 中没有 rgb，请检查 cfg.rgb_cam_ids 与 tar 内 key 命名是否匹配。")
            break

        # 单 cam：rgb 是 (view1, view2)
        if isinstance(rgb, (tuple, list)) and len(rgb) == 2:
            v1, v2 = rgb
            print("  rgb.view1:", tuple(v1.shape), v1.dtype, float(v1.min()), float(v1.max()))
            print("  rgb.view2:", tuple(v2.shape), v2.dtype, float(v2.min()), float(v2.max()))
            # 简单检查两 view 是否不同（不保证总不同，但通常会不同）
            diff = (v1 - v2).abs().mean().item()
            print(f"  mean(|view1-view2|) = {diff:.6f}")

        # 多 cam：rgb 是 dict(camid -> (view1, view2))
        elif isinstance(rgb, dict):
            for camid, vv in rgb.items():
                if isinstance(vv, (tuple, list)) and len(vv) == 2:
                    v1, v2 = vv
                    print(f"  rgb[{camid}].view1:", tuple(v1.shape), v1.dtype)
                    print(f"  rgb[{camid}].view2:", tuple(v2.shape), v2.dtype)
                else:
                    print(f"  ERROR: rgb[{camid}] 不是 (view1, view2)，实际类型={type(vv)}")
        else:
            print("  ERROR: cfg.rgb_two_views=True 时，期望 rgb 是 (view1, view2) 或 dict(cam->(v1,v2))，但实际为：", type(rgb))

        break


if __name__ == "__main__":
    main()
