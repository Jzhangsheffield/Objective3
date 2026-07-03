#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mapstyle_loader_rgb_depth.py

é€‚ç”¨äºŽä½ æ–°æ‰“åŒ…çš„ map-style RGB/Depth æ•°æ®é›†ã€‚

æ•°æ®é›†ç»“æž„ï¼ˆç¤ºä¾‹ï¼‰ï¼š
Dataset/
â”œâ”€â”€ Train/
â”‚   â”œâ”€â”€ sample_0001/
â”‚   â”‚   â”œâ”€â”€ rgb.pt
â”‚   â”‚   â”œâ”€â”€ depth.pt
â”‚   â”‚   â””â”€â”€ label.txt
â”‚   â”œâ”€â”€ sample_0002/
â”‚   â””â”€â”€ ...
â”œâ”€â”€ Train_manifest.jsonl
â”œâ”€â”€ sample_mapping.json
â””â”€â”€ dataset_meta.json

manifest æ¯è¡Œç¤ºä¾‹ï¼š
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

æœ¬ loader ç›®æ ‡ï¼š
1) å°½é‡ä¿æŒä½ æ—§ WebDataset loader çš„è¾“å‡ºé£Žæ ¼ï¼š
   - key
   - tier_actions
   - tier_ids
   - rgb / depth
2) æ”¯æŒï¼š
   - åªç”¨ RGB
   - åªç”¨ Depth
   - RGB + Depth
3) RGB:
   - æ”¯æŒå•è§†è§’æˆ– two-view
   - ç»§ç»­å¤ç”¨ä½ çŽ°æœ‰çš„ spatial_augmentation.py
4) Depth:
   - è¯»å–æ‰“åŒ…å¥½çš„ depth.pt
   - é»˜è®¤åšç¡®å®šæ€§ NEAREST resize åˆ°å›ºå®šå°ºå¯¸ï¼Œä¿è¯ batch å¯ stack
5) ä½¿ç”¨ manifest ä½œä¸ºå”¯ä¸€ç´¢å¼•æ¥æºï¼Œä¸å†æ‰«æç›®å½•

æ³¨æ„ï¼š
- RGB æ‰“åŒ…åŽé€šå¸¸ä¸º uint8 [T,3,256,256]
- Depth æ‰“åŒ…åŽé€šå¸¸ä¸º int32 [T,1,H,W]
- å¦‚æžœä½ çš„ spatial_augmentation.py å†…éƒ¨å·²ç» ToDtype+Normalizeï¼Œ
  é‚£è®­ç»ƒè„šæœ¬é‡Œä¸è¦å†å¯¹ RGB é‡å¤ Normalizeã€‚
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

# ä½ çŽ°æœ‰çš„ç©ºé—´å¢žå¼ºæ–‡ä»¶
from aug.spatial_augmentation import TemporallyConsistentSpatialAugmentation, ValidationAugmentation

# æ–°æ‹†å‡ºæ¥çš„æ—¶é—´å¢žå¼ºæ–‡ä»¶
from aug.temporal_augmentation_adaptive import sample_indices_strict, sample_two_views_indices


# ============================================================
# 1) é…ç½®
# ============================================================

@dataclass
class PackedMultiModalConfig:
    # -------- æ—¶é—´é‡‡æ · --------
    n_frames: int = 16

    # RGB æ˜¯å¦ two-viewï¼ˆå¯¹æ¯”å­¦ä¹ ï¼‰
    rgb_two_views: bool = False

    # Camera-specific RGB key to use when manifest has fields such as "001484412812_rgb".
    # record["rgb"] is still preferred when present; this is the fallback camera field.
    rgb_camera_id: Optional[str] = "001484412812"

    # å¯ç”¨å“ªäº›æ¨¡æ€
    use_modalities: Tuple[str, ...] = ("rgb", "depth")

    # ç¼ºå¤±ç­–ç•¥
    missing_policy: str = "skip"  # or "pad"

    # -------- labels / tier --------
    load_labels: bool = True
    label_map_path: Optional[str] = None

    # "all" / "tier1" / "tier2" / "tier3"
    tier_mode: str = "all"

    # -------- train/val --------
    is_train: bool = True

    # -------- RGB ç©ºé—´å¢žå¼ºå‚æ•° --------
    rgb_out_hw: Tuple[int, int] = (224, 224)

    # RandomResizedCrop
    rrc_scale: Tuple[float, float] = (0.6, 1.0)
    rrc_ratio: Tuple[float, float] = (0.75, 1.3333333333)

    # æ˜¯å¦å¯ç”¨è®­ç»ƒé˜¶æ®µéšæœº spatial augmentation
    # æ³¨æ„ï¼šè¿™é‡Œä¸æ–°å¢žæ— éšæœºå¢žå¼º transformã€‚
    # å¦‚æžœè®¾ä¸º Falseï¼Œåˆ™åœ¨ build_packed_mapstyle_dataset ä¸­æŠŠå„å¢žå¼ºæ¦‚çŽ‡ç½® 0ï¼Œ
    # ä½†ä»ç„¶ä½¿ç”¨ TemporallyConsistentSpatialAugmentationã€‚
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

    # RGB normalize å‚æ•°ï¼ˆç”¨äºŽå¯è§†åŒ–åå½’ä¸€åŒ–ï¼‰
    rgb_mean: Tuple[float, float, float] = (0.356, 0.363, 0.367)
    rgb_std: Tuple[float, float, float] = (0.288, 0.271, 0.270)

    # -------- Depth è¾“å‡ºå°ºå¯¸ --------
    depth_out_hw: Tuple[int, int] = (224, 224)

    # pad é»˜è®¤å°ºå¯¸
    default_rgb_hw: Tuple[int, int] = (256, 256)
    default_depth_hw: Tuple[int, int] = (224, 224)

    # Depth pad dtype
    default_depth_dtype: str = "int32"

    # ç”± build_loader åœ¨è¿è¡Œæ—¶æŒ‚è¿›åŽ»
    rgb_transform: Optional[Any] = field(default=None, repr=False, compare=False)


# ============================================================
# 2) label_map åŠ è½½ä¸Ž tier æ˜ å°„
# ============================================================

def load_label_map_json(path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    è¯»å– label_map.jsonï¼š
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
    å¦‚æžœæ²¡æœ‰æä¾›å¤–éƒ¨ label_map.jsonï¼Œå°±ä»Ž manifest åŠ¨æ€æž„é€ ã€‚
    æ³¨æ„ï¼š
    - è¿™ç§åšæ³•é€‚åˆå• split è°ƒè¯•
    - æ­£å¼ train/val/test æœ€å¥½è¿˜æ˜¯ç”¨ç»Ÿä¸€ label_map.json
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
# 3) å·¥å…·å‡½æ•°
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
    Depth resizeï¼š
    - è¾“å…¥ï¼š[T,1,H,W]ï¼Œdtype å¯èƒ½ int32/uint16/uint8
    - è¾“å‡ºï¼š[T,1,outH,outW]ï¼Œdtype å°½é‡ä¿æŒ

    åšæ³•ï¼š
      1) è½¬ float32
      2) NEAREST resize
      3) round
      4) cast å›žåŽŸ dtype
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
    pad æ¨¡å¼ä¸‹ RGB é›¶è§†é¢‘ï¼š
    uint8 [T,3,H,W]
    """
    H, W = cfg.default_rgb_hw
    return torch.zeros((cfg.n_frames, 3, H, W), dtype=torch.uint8)


def _make_zero_depth(cfg: PackedMultiModalConfig) -> torch.Tensor:
    """
    pad æ¨¡å¼ä¸‹ Depth é›¶è§†é¢‘ï¼š[T,1,H,W]
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
    åŸºäºŽ manifest çš„ map-style Datasetã€‚

    è¾“å‡ºç»“æž„ï¼ˆå°½é‡è´´è¿‘æ—§ loaderï¼‰ï¼š
    {
      "key": str,
      "sample_name": str,
      "tier_actions": {...},
      "tier_ids": {...},
      "lighting": str,
      "pos": str,
      "rgb": Tensor[T,3,H,W] æˆ– (view1, view2),
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
            # å…ˆå°è¯• cfg.label_map_path
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

    def _resolve_rgb_rel_from_record(self, rec: Dict[str, Any]) -> Optional[str]:
        """
        Resolve the RGB tensor path for one manifest record.

        Priority:
        1) generic record["rgb"] for older/single-camera manifests;
        2) camera-specific record[f"{rgb_camera_id}_rgb"] for multi-camera Stage-2 manifests.
        """
        rgb_rel = rec.get("rgb", None)
        if rgb_rel is not None:
            return rgb_rel

        camera_id = getattr(self.cfg, "rgb_camera_id", None)
        if camera_id:
            return rec.get(f"{camera_id}_rgb", None)

        return None
    def _filter_valid_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        åˆå§‹åŒ–é˜¶æ®µè¿‡æ»¤æŽ‰è·¯å¾„ç¼ºå¤±çš„æ ·æœ¬ï¼Œå°½é‡è´´è¿‘æ—§ loader çš„ skip è¡Œä¸ºã€‚
        """
        valid = []
        for rec in records:
            ok = True
            if "rgb" in self.cfg.use_modalities:
                rgb_rel = self._resolve_rgb_rel_from_record(rec)
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
        rgb_rel = self._resolve_rgb_rel_from_record(rec)
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

        # æœŸæœ› [T,3,H,W]
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

        # æœŸæœ› [T,1,H,W]ï¼Œæžå°‘æ•°ä¹Ÿå¯èƒ½æ˜¯ [T,C,H,W]
        if not torch.is_tensor(video) or video.ndim != 4:
            raise ValueError(f"Invalid depth tensor shape: {type(video)} / {getattr(video, 'shape', None)}")

        T = int(video.shape[0])

        idxs = sample_indices_strict(T, self.cfg.n_frames)
        v = video[idxs]

        # è‹¥æ„å¤–å‡ºçŽ°å¤šé€šé“ depthï¼Œè¿™é‡Œåªè¦ç»´åº¦æ˜¯ [T,C,H,W] ä»ç„¶æ”¯æŒ
        if v.shape[1] == 1:
            v = _resize_depth_video_keep_dtype(v, out_hw=self.cfg.depth_out_hw)
        else:
            # å¤šé€šé“æƒ…å†µï¼šé€é€šé“ NEAREST resize
            outs = []
            for c in range(v.shape[1]):
                vc = _resize_depth_video_keep_dtype(v[:, c:c+1], out_hw=self.cfg.depth_out_hw)
                outs.append(vc)
            v = torch.cat(outs, dim=1).contiguous()

        return v

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        # è¿™é‡Œæ˜¾å¼è¿”å›ž map-style dataset çš„è¿žç»­å…¨å±€ç´¢å¼• idxã€‚
        # åŽç»­ prototype æž„å»ºã€prototype contrastive lossã€åœ¨çº¿ EMA æ›´æ–°
        # éƒ½ä¾èµ–è¿™ä¸ªç¨³å®šç´¢å¼•æ¥æ‰§è¡Œ sample_to_proto[global_index]ã€‚
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
    è‡ªå®šä¹‰ collateï¼š
    - ä¿ç•™ rgb two-view ä¸º (view1_batch, view2_batch)
    - å…¶ä½™å­—æ®µå°½é‡æŒ‰é»˜è®¤è§„åˆ™ stack / list
    """
    if len(batch) == 0:
        raise RuntimeError("Empty batch in collate.")

    out: Dict[str, Any] = {}
    keys = batch[0].keys()

    for k in keys:
        vals = [b[k] for b in batch]

        # RGB two-viewï¼švals æ˜¯ [(v1,v2), (v1,v2), ...]
        if k == "rgb" and isinstance(vals[0], (tuple, list)) and len(vals[0]) == 2:
            v1 = default_collate([x[0] for x in vals])
            v2 = default_collate([x[1] for x in vals])
            out[k] = (v1, v2)
            continue

        # æ™®é€š dictï¼ˆä¾‹å¦‚ tier_actions / tier_idsï¼‰
        if isinstance(vals[0], dict):
            out[k] = {}
            subkeys = vals[0].keys()
            for sk in subkeys:
                subv = [x[sk] for x in vals]
                # æ•°å€¼å¯ stackï¼Œå­—ç¬¦ä¸²ä¿ç•™ list
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

        # å­—ç¬¦ä¸²ç­‰ -> list
        out[k] = list(vals)

    return out


# ============================================================
# 6) æž„å»º DataLoader
# ============================================================

# ============================================================
# 6) æž„å»º Dataset / DataLoader
# ============================================================

def build_packed_mapstyle_dataset(
    dataset_root: Union[str, Path],
    manifest_name: str,
    cfg: PackedMultiModalConfig,
    label_map: Optional[Dict[str, Dict[str, int]]] = None,
    verify_paths_on_init: bool = True,
) -> PackedRGBDepthMapDataset:
    """
    åªè´Ÿè´£æž„å»º map-style Datasetã€‚

    èŒè´£ï¼š
    1) è§„èŒƒåŒ– use_modalities
    2) æ ¹æ® train/val æŒ‚è½½ RGB transform
    3) è¿”å›ž PackedRGBDepthMapDataset

    è¿™æ ·æ‹†å¼€åŽï¼Œå¤–éƒ¨å¦‚æžœéœ€è¦ï¼š
    - å…ˆåˆ›å»º dataset
    - å†åˆ›å»º DistributedSampler
    - å†åŒ…è£…æˆ DataLoader
    ä¼šæ›´è‡ªç„¶ï¼Œä¹Ÿä¸ä¼šé‡å¤æž„é€  datasetã€‚
    """
    use_modalities = _normalize_modalities(cfg.use_modalities)
    cfg.use_modalities = use_modalities

    # åªåœ¨å¯ç”¨ RGB æ—¶é…ç½® RGB transform
    if cfg.is_train:
        if cfg.rgb_apply_spatial_aug:
            hflip_p = cfg.rgb_hflip_p
            vflip_p = cfg.rgb_vflip_p
            jitter_p = cfg.rgb_jitter_p
            gray_p = cfg.rgb_gray_p
            blur_p = cfg.rgb_blur_p
        else:
            # ä¸æ–°å¢žæ— éšæœºå¢žå¼º transformã€‚
            # ä»ç„¶ä½¿ç”¨ TemporallyConsistentSpatialAugmentationï¼Œ
            # ä½†æŠŠé™¤ RandomResizedCrop å¤–çš„éšæœºå¢žå¼ºæ¦‚çŽ‡å…¨éƒ¨è®¾ä¸º 0ã€‚
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
    åªè´Ÿè´£æŠŠå·²ç»æž„å»ºå¥½çš„ dataset åŒ…è£…æˆ DataLoaderã€‚

    è¯´æ˜Žï¼š
    - è‹¥ sampler ä¸ä¸º Noneï¼ˆä¾‹å¦‚ DistributedSamplerï¼‰ï¼Œåˆ™å¿…é¡»å…³é—­ shuffle
    - collate_fn å›ºå®šä½¿ç”¨ packed_multimodal_collate
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
    å…¼å®¹æ—§æŽ¥å£çš„è–„å°è£…ã€‚

    å†…éƒ¨æµç¨‹ï¼š
      build_packed_mapstyle_dataset(...)
      -> build_packed_mapstyle_loader_from_dataset(...)

    è¿™æ ·æ—§ä»£ç åŸºæœ¬ä¸ç”¨æ”¹ï¼›
    æ–°ä»£ç å¦‚æžœæƒ³æ›´æ¸…æ™°ï¼Œä¹Ÿå¯ä»¥ç›´æŽ¥åˆ†åˆ«è°ƒç”¨ä¸Šé¢ä¸¤ä¸ªå‡½æ•°ã€‚
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
# 8) æž„å»º weighted sampler
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
    ä¸º PackedRGBDepthMapDataset æž„å»º WeightedRandomSamplerã€‚

    å‚æ•°
    ----
    dataset : PackedRGBDepthMapDataset
        å·²ç»æž„å»ºå¥½çš„ map-style æ•°æ®é›†å¯¹è±¡ã€‚
        æœ¬å‡½æ•°ä¸ä¼šè¯»å– rgb/depth æ–‡ä»¶ï¼Œåªä¼šåˆ©ç”¨ dataset.records ä¸­çš„æ ‡ç­¾ä¿¡æ¯ã€‚

    tier_for_sampling : str or None
        æŒ‡å®šæŒ‰å“ªä¸€å±‚æ ‡ç­¾åšé‡é‡‡æ ·ï¼Œå¯é€‰:
            - "tier1"
            - "tier2"
            - "tier3"
        è‹¥ä¸º Noneï¼š
            - å¦‚æžœ dataset.cfg.tier_mode æ˜¯ "tier1"/"tier2"/"tier3"ï¼Œåˆ™è‡ªåŠ¨ä½¿ç”¨å®ƒ
            - å¦‚æžœ dataset.cfg.tier_mode == "all"ï¼Œé»˜è®¤ä½¿ç”¨ "tier3"

    mode : str
        ç±»åˆ«æƒé‡æž„é€ æ–¹å¼ï¼š
            - "inv"      : weight = 1 / count
            - "sqrt_inv" : weight = 1 / sqrt(count)
        ä¸€èˆ¬æŽ¨èå…ˆç”¨ "sqrt_inv"ï¼Œæ›´ç¨³ï¼Œä¸å®¹æ˜“è¿‡åº¦é‡é‡‡æ ·å°‘æ•°ç±»ã€‚

    replacement : bool
        æ˜¯å¦æœ‰æ”¾å›žé‡‡æ ·ã€‚ç±»åˆ«ä¸å¹³è¡¡åœºæ™¯ä¸‹é€šå¸¸åº”è®¾ä¸º Trueã€‚

    num_samples : int or None
        ä¸€ä¸ª epoch æŠ½å–å¤šå°‘ä¸ªæ ·æœ¬ã€‚
        è‹¥ä¸º Noneï¼Œåˆ™é»˜è®¤ç­‰äºŽ len(dataset)ã€‚

    verbose : bool
        æ˜¯å¦æ‰“å°ä¸€äº›ç»Ÿè®¡ä¿¡æ¯ã€‚

    è¿”å›ž
    ----
    sampler : WeightedRandomSampler
        å¯ç›´æŽ¥ä¼ ç»™ DataLoader(..., sampler=sampler, shuffle=False)

    info : dict
        è°ƒè¯•ä¿¡æ¯ï¼ŒåŒ…æ‹¬ï¼š
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
    # 1) è‡ªåŠ¨å†³å®šæŒ‰å“ªä¸ª tier åšé‡‡æ ·
    # --------------------------------------------------------
    if tier_for_sampling is None:
        if dataset.cfg.tier_mode in ("tier1", "tier2", "tier3"):
            tier_for_sampling = dataset.cfg.tier_mode
        else:
            # tier_mode == "all" æ—¶ï¼Œé»˜è®¤æŒ‰æœ€ç»†ç²’åº¦ tier3 é‡‡æ ·
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
    # 2) ä»Ž manifest records ä¸­æå–æ¯ä¸ªæ ·æœ¬çš„æ•´æ•°æ ‡ç­¾
    #    æ³¨æ„ï¼šè¿™é‡Œå®Œå…¨ä¸èµ° __getitem__ï¼Œé¿å…çœŸçš„åŠ è½½è§†é¢‘æ–‡ä»¶
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
    # 3) ç»Ÿè®¡å½“å‰ dataset å†…å„ç±»åˆ«æ ·æœ¬æ•°
    #    è¿™é‡ŒæŒ‰â€œå½“å‰ split ä¸­å®žé™…å­˜åœ¨çš„æ ·æœ¬â€ç»Ÿè®¡ï¼Œè€Œä¸æ˜¯å¤–éƒ¨å†™æ­»ç»Ÿè®¡è¡¨
    # --------------------------------------------------------
    counter = Counter(labels.tolist())

    if len(counter) == 0:
        raise RuntimeError("No valid labels found for weighted sampling.")

    max_class_id = max(counter.keys())
    class_counts_tensor = torch.zeros(max_class_id + 1, dtype=torch.long)

    for cls_id, cnt in counter.items():
        class_counts_tensor[cls_id] = int(cnt)

    # --------------------------------------------------------
    # 4) æž„é€ ç±»åˆ«æƒé‡
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
    # 5) æ˜ å°„ä¸ºæ¯ä¸ªæ ·æœ¬çš„æƒé‡
    # --------------------------------------------------------
    sample_weights = class_weights[labels]   # [N]
    sample_weights = sample_weights.to(torch.double)

    # --------------------------------------------------------
    # 6) num_samples é»˜è®¤ç­‰äºŽæ•°æ®é›†é•¿åº¦
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
# 9) å¯è§†åŒ–å·¥å…·ï¼ˆè°ƒè¯•å¢žå¼ºæ˜¯å¦ç”Ÿæ•ˆï¼‰
# ============================================================
def _tensor_to_rgb_vis(
    frame_chw: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Any:
    """
    æŠŠå•å¸§ RGB Tensor[C,H,W] è½¬æˆå¯ imshow çš„ numpy æ•°ç»„ [H,W,C]ã€‚

    æ”¯æŒä¸¤ç§æƒ…å†µï¼š
    1) uint8 [0,255]ï¼šç›´æŽ¥æ˜¾ç¤º
    2) float ä¸”å·² Normalizeï¼šæŒ‰ç»™å®š mean/std åå½’ä¸€åŒ–åŽæ˜¾ç¤º
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

    # åå½’ä¸€åŒ–ï¼šx = x * std + mean
    x = x * std_t + mean_t

    # clip åˆ°åˆæ³•æ˜¾ç¤ºèŒƒå›´
    x = torch.clamp(x, 0.0, 1.0)

    # [C,H,W] -> [H,W,C]
    x = x.permute(1, 2, 0).contiguous()

    return x.numpy()


def _tensor_to_depth_vis(frame_chw: torch.Tensor) -> Any:
    """
    æŠŠå•å¸§ Depth Tensor[1,H,W] æˆ– [C,H,W] è½¬æˆå¯ imshow çš„ 2D numpyã€‚

    åšæ³•ï¼š
    - è‹¥å¤šé€šé“ï¼Œåªå–ç¬¬ä¸€ä¸ªé€šé“
    - ç”¨ min-max æ‹‰ä¼¸åˆ° 0~1
    """
    x = frame_chw.detach().cpu()

    if x.ndim != 3:
        raise ValueError(f"Expected depth frame [C,H,W], got {tuple(x.shape)}")

    # åªæ˜¾ç¤ºç¬¬ä¸€ä¸ªé€šé“
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
    ä½ å¯ä»¥è‡ªè¡Œä¿®æ”¹è·¯å¾„è¿›è¡Œç®€å•è‡ªæµ‹ã€‚
    """
    dataset_root = r"C:\Junxi_data_for_training_speedup\mapstyle_dataset"

    cfg = PackedMultiModalConfig(
        n_frames=16,
        use_modalities=("rgb", "depth"),   # å¯ä»¥æ”¹æˆ ("rgb",) æˆ– ("depth",)
        missing_policy="skip",
        tier_mode="all",
        is_train=True,
        rgb_out_hw=(224, 224),
        depth_out_hw=(224, 224),
        rgb_two_views=True,   # two-view æ—¶æ›´å®¹æ˜“è§‚å¯Ÿå¢žå¼ºæ˜¯å¦ä¸åŒ
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
        # å¯è§†åŒ–ï¼šçœ‹ç¬¬ 0 ä¸ªæ ·æœ¬çš„è‹¥å¹²å¸§
        # ====================================================
        visualize_batch_sample(
            cfg=cfg,
            batch=batch,
            sample_idx=0,      # æŸ¥çœ‹ batch ä¸­ç¬¬ä¸€ä¸ªæ ·æœ¬
            max_frames=16,      # æœ€å¤šæ˜¾ç¤º 4 å¸§
            save_path=None,    # æ”¹æˆè·¯å¾„å­—ç¬¦ä¸²å¯ç›´æŽ¥ä¿å­˜
            # save_path="debug_aug_sample.png"
        )

        # åªçœ‹ä¸€ä¸ª batch å°±é€€å‡ºï¼Œé¿å…ä¸€ç›´å¼¹çª—
        break



if __name__ == "__main__":
    main()

