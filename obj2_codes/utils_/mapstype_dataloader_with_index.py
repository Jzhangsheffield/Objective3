#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mapstyle_loader_rgb_depth.py

适用于你新打包的 map-style RGB/Depth 数据集。

数据集结构（示例）：
Dataset/
├── Train/
│   ├── sample_0001/
│   │   ├── rgb.pt
│   │   ├── depth.pt
│   │   └── label.txt
│   ├── sample_0002/
│   └── ...
├── Train_manifest.jsonl
├── sample_mapping.json
└── dataset_meta.json

manifest 每行示例：
{
  "sample_name": "sample_0001",
  "original_key": "N/cap_red_pen/run_11_clip_000030_left_elbow",
  "person": "N",
  "action": "cap_red_pen",
  "segment": "run_11_clip_000030_left_elbow",
  "tier1": "cap",
  "tier2": "cap_pen",
  "tier3": "cap_red_pen",
  "lighting": "left",
  "pos": "elbow",
  "rgb": "Train/sample_0001/rgb.pt",
  "depth": "Train/sample_0001/depth.pt",
  "label_txt": "Train/sample_0001/label.txt"
}

本 loader 目标：
1) 尽量保持你旧 WebDataset loader 的输出风格：
   - key
   - tier_actions
   - tier_ids
   - rgb / depth
2) 支持：
   - 只用 RGB
   - 只用 Depth
   - RGB + Depth
3) RGB:
   - 支持单视角或 two-view
   - 继续复用你现有的 spatial_augmentation.py
4) Depth:
   - 读取打包好的 depth.pt
   - 默认做确定性 NEAREST resize 到固定尺寸，保证 batch 可 stack
5) 使用 manifest 作为唯一索引来源，不再扫描目录

注意：
- RGB 打包后通常为 uint8 [T,3,256,256]
- Depth 打包后通常为 int32 [T,1,H,W]
- 如果你的 spatial_augmentation.py 内部已经 ToDtype+Normalize，
  那训练脚本里不要再对 RGB 重复 Normalize。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from torchvision.transforms.v2 import functional as Fv2
from torchvision.transforms import InterpolationMode

# 你现有的空间增强文件
from aug.spatial_augmentation import TemporallyConsistentSpatialAugmentation, ValidationAugmentation

# 新拆出来的时间增强文件
from aug.temporal_augmentation_adaptive import sample_indices_strict, sample_two_views_indices


# ============================================================
# 1) 配置
# ============================================================

@dataclass
class PackedMultiModalConfig:
    # -------- 时间采样 --------
    n_frames: int = 16

    # RGB 是否 two-view（对比学习）
    rgb_two_views: bool = False

    # 启用哪些模态
    use_modalities: Tuple[str, ...] = ("rgb", "depth")

    # 缺失策略
    missing_policy: str = "skip"  # or "pad"

    # -------- labels / tier --------
    load_labels: bool = True
    label_map_path: Optional[str] = None

    # "all" / "tier1" / "tier2" / "tier3"
    tier_mode: str = "all"

    # -------- train/val --------
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

    # -------- Depth 输出尺寸 --------
    depth_out_hw: Tuple[int, int] = (224, 224)

    # pad 默认尺寸
    default_rgb_hw: Tuple[int, int] = (256, 256)
    default_depth_hw: Tuple[int, int] = (224, 224)

    # Depth pad dtype
    default_depth_dtype: str = "int32"

    # 由 build_loader 在运行时挂进去
    rgb_transform: Optional[Any] = field(default=None, repr=False, compare=False)


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
# 3) 工具函数
# ============================================================

def _normalize_modalities(use_modalities: Tuple[str, ...]) -> Tuple[str, ...]:
    x = tuple(str(m).strip().lower() for m in use_modalities if str(m).strip() != "")
    valid = {"rgb", "depth"}
    bad = [m for m in x if m not in valid]
    if bad:
        raise ValueError(f"Unsupported modalities: {bad}. Currently supported: rgb, depth")
    if len(x) == 0:
        raise ValueError("At least one modality must be enabled.")
    return x


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
    assert video_tchw.ndim == 4 and video_tchw.shape[1] == 1
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
        if orig_dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            fr = torch.round(fr)
        fr = fr.to(orig_dtype)
        out_frames.append(fr)

    return torch.stack(out_frames, dim=0).contiguous()


def _make_zero_rgb(cfg: PackedMultiModalConfig) -> torch.Tensor:
    """
    pad 模式下 RGB 零视频：
    uint8 [T,3,H,W]
    """
    H, W = cfg.default_rgb_hw
    return torch.zeros((cfg.n_frames, 3, H, W), dtype=torch.uint8)


def _make_zero_depth(cfg: PackedMultiModalConfig) -> torch.Tensor:
    """
    pad 模式下 Depth 零视频：[T,1,H,W]
    """
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


def _load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================
# 4) Dataset
# ============================================================

class PackedRGBDepthMapDataset(Dataset):
    """
    基于 manifest 的 map-style Dataset。

    输出结构（尽量贴近旧 loader）：
    {
      "key": str,
      "sample_name": str,
      "tier_actions": {...},
      "tier_ids": {...},
      "lighting": str,
      "pos": str,
      "rgb": Tensor[T,3,H,W] 或 (view1, view2),
      "depth": Tensor[T,1,H,W]
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
        self.cfg.use_modalities = _normalize_modalities(self.cfg.use_modalities)

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
            if ok:
                valid.append(rec)
        return valid

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, rec: Dict[str, Any]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 临时兼容 stage2 的脚本
        rgb_rel = rec.get("rgb", None)
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
        if isinstance(obj, dict):
            video = obj["frames"]
        else:
            video = obj

        # 期望 [T,3,H,W]
        if not torch.is_tensor(video) or video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(f"Invalid rgb tensor shape: {type(video)} / {getattr(video, 'shape', None)}")

        T = int(video.shape[0])

        if self.cfg.rgb_transform is None:
            raise RuntimeError(
                "cfg.rgb_transform is None. "
                "Please build the dataset via build_packed_mapstyle_dataset(...) "
                "or the loader via build_packed_mapstyle_loader(...)."
            )

        if self.cfg.rgb_two_views:
            idxs1, idxs2 = sample_two_views_indices(T=T, n=self.cfg.n_frames)
            v1 = video[idxs1]   # [n,3,H,W] uint8
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
        if isinstance(obj, dict):
            video = obj["frames"]
        else:
            video = obj

        # 期望 [T,1,H,W]，极少数也可能是 [T,C,H,W]
        if not torch.is_tensor(video) or video.ndim != 4:
            raise ValueError(f"Invalid depth tensor shape: {type(video)} / {getattr(video, 'shape', None)}")

        T = int(video.shape[0])

        idxs = sample_indices_strict(T, self.cfg.n_frames)
        v = video[idxs]

        # 若意外出现多通道 depth，这里只要维度是 [T,C,H,W] 仍然支持
        if v.shape[1] == 1:
            v = _resize_depth_video_keep_dtype(v, out_hw=self.cfg.depth_out_hw)
        else:
            # 多通道情况：逐通道 NEAREST resize
            outs = []
            for c in range(v.shape[1]):
                vc = _resize_depth_video_keep_dtype(v[:, c:c+1], out_hw=self.cfg.depth_out_hw)
                outs.append(vc)
            v = torch.cat(outs, dim=1).contiguous()

        return v

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        # 这里显式返回 map-style dataset 的连续全局索引 idx。
        # 后续 prototype 构建、prototype contrastive loss、在线 EMA 更新
        # 都依赖这个稳定索引来执行 sample_to_proto[global_index]。
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

        return out


# ============================================================
# 5) collate_fn
# ============================================================

def packed_multimodal_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 collate：
    - 保留 rgb two-view 为 (view1_batch, view2_batch)
    - 其余字段尽量按默认规则 stack / list
    """
    if len(batch) == 0:
        raise RuntimeError("Empty batch in collate.")

    out: Dict[str, Any] = {}
    keys = batch[0].keys()

    for k in keys:
        vals = [b[k] for b in batch]

        # RGB two-view：vals 是 [(v1,v2), (v1,v2), ...]
        if k == "rgb" and isinstance(vals[0], (tuple, list)) and len(vals[0]) == 2:
            v1 = default_collate([x[0] for x in vals])
            v2 = default_collate([x[1] for x in vals])
            out[k] = (v1, v2)
            continue

        # 普通 dict（例如 tier_actions / tier_ids）
        if isinstance(vals[0], dict):
            out[k] = {}
            subkeys = vals[0].keys()
            for sk in subkeys:
                subv = [x[sk] for x in vals]
                # 数值可 stack，字符串保留 list
                if torch.is_tensor(subv[0]):
                    out[k][sk] = default_collate(subv)
                elif isinstance(subv[0], (int, float, bool)):
                    out[k][sk] = default_collate(subv)
                else:
                    out[k][sk] = list(subv)
            continue

        # Tensor / int / float / bool -> default_collate
        if torch.is_tensor(vals[0]) or isinstance(vals[0], (int, float, bool)):
            out[k] = default_collate(vals)
            continue

        # 字符串等 -> list
        out[k] = list(vals)

    return out


# ============================================================
# 6) 构建 DataLoader
# ============================================================

# ============================================================
# 6) 构建 Dataset / DataLoader
# ============================================================

def build_packed_mapstyle_dataset(
    dataset_root: Union[str, Path],
    manifest_name: str,
    cfg: PackedMultiModalConfig,
    label_map: Optional[Dict[str, Dict[str, int]]] = None,
    verify_paths_on_init: bool = True,
) -> PackedRGBDepthMapDataset:
    """
    只负责构建 map-style Dataset。

    职责：
    1) 规范化 use_modalities
    2) 根据 train/val 挂载 RGB transform
    3) 返回 PackedRGBDepthMapDataset

    这样拆开后，外部如果需要：
    - 先创建 dataset
    - 再创建 DistributedSampler
    - 再包装成 DataLoader
    会更自然，也不会重复构造 dataset。
    """
    use_modalities = _normalize_modalities(cfg.use_modalities)
    cfg.use_modalities = use_modalities

    # 只在启用 RGB 时配置 RGB transform
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
            # 但把除 RandomResizedCrop 外的随机增强概率全部设为 0。
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

    ds = PackedRGBDepthMapDataset(
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
    只负责把已经构建好的 dataset 包装成 DataLoader。

    说明：
    - 若 sampler 不为 None（例如 DistributedSampler），则必须关闭 shuffle
    - collate_fn 固定使用 packed_multimodal_collate
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

    内部流程：
      build_packed_mapstyle_dataset(...)
      -> build_packed_mapstyle_loader_from_dataset(...)

    这样旧代码基本不用改；
    新代码如果想更清晰，也可以直接分别调用上面两个函数。
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
# 8) 构建 weighted sampler
# ============================================================
from collections import Counter
from typing import Optional, Dict, Any, Tuple
import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler_for_packed_dataset(
    dataset: "PackedRGBDepthMapDataset",
    tier_for_sampling: Optional[str] = None,
    mode: str = "sqrt_inv",
    replacement: bool = True,
    num_samples: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[WeightedRandomSampler, Dict[str, Any]]:
    """
    为 PackedRGBDepthMapDataset 构建 WeightedRandomSampler。

    参数
    ----
    dataset : PackedRGBDepthMapDataset
        已经构建好的 map-style 数据集对象。
        本函数不会读取 rgb/depth 文件，只会利用 dataset.records 中的标签信息。

    tier_for_sampling : str or None
        指定按哪一层标签做重采样，可选:
            - "tier1"
            - "tier2"
            - "tier3"
        若为 None：
            - 如果 dataset.cfg.tier_mode 是 "tier1"/"tier2"/"tier3"，则自动使用它
            - 如果 dataset.cfg.tier_mode == "all"，默认使用 "tier3"

    mode : str
        类别权重构造方式：
            - "inv"      : weight = 1 / count
            - "sqrt_inv" : weight = 1 / sqrt(count)
        一般推荐先用 "sqrt_inv"，更稳，不容易过度重采样少数类。

    replacement : bool
        是否有放回采样。类别不平衡场景下通常应设为 True。

    num_samples : int or None
        一个 epoch 抽取多少个样本。
        若为 None，则默认等于 len(dataset)。

    verbose : bool
        是否打印一些统计信息。

    返回
    ----
    sampler : WeightedRandomSampler
        可直接传给 DataLoader(..., sampler=sampler, shuffle=False)

    info : dict
        调试信息，包括：
            - tier_for_sampling
            - labels
            - class_counts
            - class_weights
            - sample_weights
            - num_samples
    """
    if not hasattr(dataset, "records"):
        raise TypeError("dataset must have attribute 'records'.")

    if not hasattr(dataset, "label_map"):
        raise TypeError("dataset must have attribute 'label_map'.")

    if not hasattr(dataset, "cfg"):
        raise TypeError("dataset must have attribute 'cfg'.")

    # --------------------------------------------------------
    # 1) 自动决定按哪个 tier 做采样
    # --------------------------------------------------------
    if tier_for_sampling is None:
        if dataset.cfg.tier_mode in ("tier1", "tier2", "tier3"):
            tier_for_sampling = dataset.cfg.tier_mode
        else:
            # tier_mode == "all" 时，默认按最细粒度 tier3 采样
            tier_for_sampling = "tier3"

    if tier_for_sampling not in ("tier1", "tier2", "tier3"):
        raise ValueError(
            f"tier_for_sampling must be one of ('tier1','tier2','tier3'), "
            f"got {tier_for_sampling}"
        )

    if tier_for_sampling not in dataset.label_map:
        raise KeyError(f"dataset.label_map does not contain key '{tier_for_sampling}'")

    tier_label_map = dataset.label_map[tier_for_sampling]

    # --------------------------------------------------------
    # 2) 从 manifest records 中提取每个样本的整数标签
    #    注意：这里完全不走 __getitem__，避免真的加载视频文件
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 3) 统计当前 dataset 内各类别样本数
    #    这里按“当前 split 中实际存在的样本”统计，而不是外部写死统计表
    # --------------------------------------------------------
    counter = Counter(labels.tolist())

    if len(counter) == 0:
        raise RuntimeError("No valid labels found for weighted sampling.")

    max_class_id = max(counter.keys())
    class_counts_tensor = torch.zeros(max_class_id + 1, dtype=torch.long)

    for cls_id, cnt in counter.items():
        class_counts_tensor[cls_id] = int(cnt)

    # --------------------------------------------------------
    # 4) 构造类别权重
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 5) 映射为每个样本的权重
    # --------------------------------------------------------
    sample_weights = class_weights[labels]   # [N]
    sample_weights = sample_weights.to(torch.double)

    # --------------------------------------------------------
    # 6) num_samples 默认等于数据集长度
    # --------------------------------------------------------
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


#  ============================================================
# 9) 可视化工具（调试增强是否生效）
# ============================================================
def _tensor_to_rgb_vis(
    frame_chw: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Any:
    """
    把单帧 RGB Tensor[C,H,W] 转成可 imshow 的 numpy 数组 [H,W,C]。

    支持两种情况：
    1) uint8 [0,255]：直接显示
    2) float 且已 Normalize：按给定 mean/std 反归一化后显示
    """
    x = frame_chw.detach().cpu()

    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected RGB frame [3,H,W], got {tuple(x.shape)}")

    # [C,H,W] -> [H,W,C]
    if x.dtype == torch.uint8:
        return x.permute(1, 2, 0).contiguous().numpy()

    x = x.to(torch.float32)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    # 反归一化：x = x * std + mean
    x = x * std_t + mean_t

    # clip 到合法显示范围
    x = torch.clamp(x, 0.0, 1.0)

    # [C,H,W] -> [H,W,C]
    x = x.permute(1, 2, 0).contiguous()

    return x.numpy()


def _tensor_to_depth_vis(frame_chw: torch.Tensor) -> Any:
    """
    把单帧 Depth Tensor[1,H,W] 或 [C,H,W] 转成可 imshow 的 2D numpy。

    做法：
    - 若多通道，只取第一个通道
    - 用 min-max 拉伸到 0~1
    """
    x = frame_chw.detach().cpu()

    if x.ndim != 3:
        raise ValueError(f"Expected depth frame [C,H,W], got {tuple(x.shape)}")

    # 只显示第一个通道
    x = x[0].to(torch.float32)

    x_min = float(x.min())
    x_max = float(x.max())

    if x_max - x_min < 1e-8:
        x = torch.zeros_like(x)
    else:
        x = (x - x_min) / (x_max - x_min)

    return x.numpy()


def visualize_batch_sample(
    batch: Dict[str, Any],
    cfg: PackedMultiModalConfig,
    sample_idx: int = 0,
    max_frames: int = 4,
    save_path: Optional[str] = None,
) -> None:
    has_rgb = "rgb" in batch
    has_depth = "depth" in batch

    if not has_rgb and not has_depth:
        raise RuntimeError("Batch has neither rgb nor depth.")

    rgb_mode = None
    rgb_v1 = rgb_v2 = None

    if has_rgb:
        rgb = batch["rgb"]
        if isinstance(rgb, (tuple, list)) and len(rgb) == 2:
            rgb_mode = "two_view"
            rgb_v1 = rgb[0][sample_idx]
            rgb_v2 = rgb[1][sample_idx]
            T_rgb = rgb_v1.shape[0]
        else:
            rgb_mode = "single_view"
            rgb_v1 = rgb[sample_idx]
            T_rgb = rgb_v1.shape[0]
    else:
        T_rgb = None

    if has_depth:
        depth = batch["depth"][sample_idx]
        T_depth = depth.shape[0]
    else:
        T_depth = None

    if T_rgb is not None and T_depth is not None:
        valid_T = min(T_rgb, T_depth)
    elif T_rgb is not None:
        valid_T = T_rgb
    else:
        valid_T = T_depth

    n_show = min(max_frames, valid_T)
    frame_ids = torch.linspace(0, valid_T - 1, steps=n_show).round().long().tolist()

    row_titles = []
    if has_rgb:
        if rgb_mode == "single_view":
            row_titles.append("RGB")
        else:
            row_titles.append("RGB view1")
            row_titles.append("RGB view2")
    if has_depth:
        row_titles.append("Depth")

    n_rows = len(row_titles)
    n_cols = n_show

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        squeeze=False
    )

    row_ptr = 0

    if has_rgb:
        if rgb_mode == "single_view":
            for j, t in enumerate(frame_ids):
                img = _tensor_to_rgb_vis(
                    rgb_v1[t],
                    mean=cfg.rgb_mean,
                    std=cfg.rgb_std,
                )
                ax = axes[row_ptr][j]
                ax.imshow(img)
                ax.set_title(f"RGB t={t}")
                ax.axis("off")
            row_ptr += 1
        else:
            for j, t in enumerate(frame_ids):
                img1 = _tensor_to_rgb_vis(
                    rgb_v1[t],
                    mean=cfg.rgb_mean,
                    std=cfg.rgb_std,
                )
                ax = axes[row_ptr][j]
                ax.imshow(img1)
                ax.set_title(f"RGB v1 t={t}")
                ax.axis("off")
            row_ptr += 1

            for j, t in enumerate(frame_ids):
                img2 = _tensor_to_rgb_vis(
                    rgb_v2[t],
                    mean=cfg.rgb_mean,
                    std=cfg.rgb_std,
                )
                ax = axes[row_ptr][j]
                ax.imshow(img2)
                ax.set_title(f"RGB v2 t={t}")
                ax.axis("off")
            row_ptr += 1

    if has_depth:
        for j, t in enumerate(frame_ids):
            dep = _tensor_to_depth_vis(depth[t])
            ax = axes[row_ptr][j]
            ax.imshow(dep, cmap="gray")
            ax.set_title(f"Depth t={t}")
            ax.axis("off")
        row_ptr += 1

    for i, title in enumerate(row_titles):
        axes[i][0].set_ylabel(title, fontsize=12)

    sample_name = batch.get("sample_name", ["unknown"])[sample_idx]
    key = batch.get("key", [""])[sample_idx]

    tier3 = None
    if "tier_actions" in batch and "tier3" in batch["tier_actions"]:
        tier3 = batch["tier_actions"]["tier3"][sample_idx]

    fig.suptitle(
        f"sample_idx={sample_idx} | sample_name={sample_name}\n"
        f"key={key} | tier3={tier3}",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Saved visualization] {save_path}")
        plt.close(fig)
    else:
        plt.show()

def main():
    """
    你可以自行修改路径进行简单自测。
    """
    dataset_root = r"C:\Junxi_data_for_training_speedup\mapstyle_dataset"

    cfg = PackedMultiModalConfig(
        n_frames=16,
        use_modalities=("rgb", "depth"),   # 可以改成 ("rgb",) 或 ("depth",)
        missing_policy="skip",
        tier_mode="all",
        is_train=True,
        rgb_out_hw=(224, 224),
        depth_out_hw=(224, 224),
        rgb_two_views=True,   # two-view 时更容易观察增强是否不同
        label_map_path=r"C:\Junxi_data_for_training_speedup\mapstyle_dataset\label_map.json",
    )

    dataset = build_packed_mapstyle_dataset(
    dataset_root=dataset_root,
    manifest_name="test_manifest.jsonl",
    cfg=cfg,
    )

    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=8,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )

    for i, batch in enumerate(loader):
        print(f"\n[batch {i}] keys = {list(batch.keys())}")
        print("  key:", batch["key"])

        if "tier_ids" in batch:
            print("  tier_ids:", batch["tier_ids"])

        if "rgb" in batch:
            rgb = batch["rgb"]
            if isinstance(rgb, (tuple, list)) and len(rgb) == 2:
                print("  rgb.view1:", tuple(rgb[0].shape), rgb[0].dtype)
                print("  rgb.view2:", tuple(rgb[1].shape), rgb[1].dtype)
            else:
                print("  rgb:", tuple(rgb.shape), rgb.dtype)

        if "depth" in batch:
            print("  depth:", tuple(batch["depth"].shape), batch["depth"].dtype)

        # ====================================================
        # 可视化：看第 0 个样本的若干帧
        # ====================================================
        visualize_batch_sample(
            cfg=cfg,
            batch=batch,
            sample_idx=0,      # 查看 batch 中第一个样本
            max_frames=16,      # 最多显示 4 帧
            save_path=None,    # 改成路径字符串可直接保存
            # save_path="debug_aug_sample.png"
        )

        # 只看一个 batch 就退出，避免一直弹窗
        break



if __name__ == "__main__":
    main()