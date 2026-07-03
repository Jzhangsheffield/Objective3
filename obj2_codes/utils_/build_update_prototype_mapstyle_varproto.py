#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .mapstype_dataloader_with_index import (
    PackedMultiModalConfig,
    load_label_map_json,
    build_packed_mapstyle_dataset,
    build_packed_mapstyle_loader_from_dataset,
)


@dataclass
class PrototypeRefreshConfig:
    """
    æŽ§åˆ¶ prototype åˆ·æ–°è¿‡ç¨‹çš„é…ç½®ã€‚

    è¿™ä¸ªç‰ˆæœ¬å¯¹åº”â€œæ¯ä¸ªç±»åˆ«çš„ prototype æ•°é‡å¯ä»¥ä¸åŒâ€çš„åœºæ™¯ã€‚
    ä¸Žå›ºå®š prototype ç‰ˆæœ¬ç›¸æ¯”ï¼Œæœ€å¤§çš„å·®å¼‚æ˜¯ï¼š

        - ä¸å†ä½¿ç”¨å•ä¸ª num_prototypes
        - è€Œæ˜¯ä½¿ç”¨ num_prototypes_per_class / default_num_prototypes

    å‚æ•°ï¼š
    ------
    tier_mode:
        ä½¿ç”¨å“ªä¸ª tier çš„æ ‡ç­¾åšç±»åˆ« idï¼Œä¾‹å¦‚ tier1 / tier2 / tier3ã€‚

    num_prototypes_per_class:
        æ¯ä¸ªç±»åˆ«å¸Œæœ›ç”Ÿæˆå¤šå°‘ä¸ª prototypeã€‚
        é•¿åº¦åº”ç­‰äºŽ num_classesã€‚
        è‹¥ä¸º Noneï¼Œåˆ™æ‰€æœ‰ç±»åˆ«éƒ½ä½¿ç”¨ default_num_prototypesã€‚

    default_num_prototypes:
        å½“ num_prototypes_per_class ä¸º None æ—¶ï¼Œæ‰€æœ‰ç±»åˆ«ç»Ÿä¸€ä½¿ç”¨çš„é»˜è®¤å€¼ã€‚

    random_state / n_init / max_iter:
        sklearn KMeans çš„è¶…å‚æ•°ã€‚

    batch_size / num_workers:
        refresh é˜¶æ®µä¸“ç”¨ DataLoader çš„ batch_size / num_workersã€‚

    pin_memory / prefetch_factor:
        refresh é˜¶æ®µä¸“ç”¨ DataLoader è®¾ç½®ã€‚

    verify_paths_on_init:
        æž„å»º refresh æ•°æ®é›†æ—¶æ˜¯å¦åœ¨åˆå§‹åŒ–é˜¶æ®µæ£€æŸ¥è·¯å¾„æœ‰æ•ˆæ€§ã€‚
        è¿™ä¸ªå€¼æœ€å¥½ä¸Žè®­ç»ƒé›†æž„å»ºä¿æŒä¸€è‡´ï¼Œé¿å…å› ä¸ºè¿‡æ»¤è§„åˆ™ä¸åŒè€Œå¯¼è‡´ global_index å¯¹ä¸ä¸Šã€‚

    device:
        refresh é˜¶æ®µæ‰€ä½¿ç”¨çš„è®¾å¤‡ã€‚

    require_main_process_only:
        è‹¥ä¸º Trueï¼Œåˆ™åªå…è®¸ä¸»è¿›ç¨‹æ‰§è¡Œ refreshã€‚
        è¿™æ˜¯å½“å‰æœ€ç¨³å¦¥ã€æœ€ç›´æŽ¥çš„ä½¿ç”¨æ–¹å¼ã€‚

    enable_prototype_temperature_scaling:
        æ˜¯å¦ä¸ºæ¯ä¸ª prototype ä¼°è®¡ä¸€ä¸ªç›¸å¯¹æ¸©åº¦ç¼©æ”¾ç³»æ•°ã€‚

    proto_base_temperature:
        prototype å¯¹æ¯”æŸå¤±çš„å…¨å±€æ¸©åº¦ã€‚è‹¥å¯ç”¨ prototype æ¸©åº¦ç¼©æ”¾ï¼Œ
        è¿™é‡Œä¼šä½œä¸º density å½’ä¸€åŒ–çš„ç›®æ ‡æ¸©åº¦ã€‚

    proto_temperature_eps:
        æ•°å€¼ç¨³å®šé¡¹ã€‚
    """
    tier_mode: str = "tier1"
    num_prototypes_per_class: Optional[List[int]] = None
    default_num_prototypes: int = 3

    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    verify_paths_on_init: bool = False

    rgb_mean: Tuple[float, float, float] = (0.356, 0.363, 0.367)
    rgb_std: Tuple[float, float, float] = (0.288, 0.271, 0.270)
    rgb_camera_id: Optional[str] = "001484412812"

    device: Optional[torch.device] = None
    require_main_process_only: bool = True

    enable_prototype_temperature_scaling: bool = False
    proto_base_temperature: float = 0.07
    proto_temperature_eps: float = 1e-6


# ============================================================
# åŸºç¡€å·¥å…·
# ============================================================

def is_main_process() -> bool:
    """åˆ¤æ–­å½“å‰è¿›ç¨‹æ˜¯å¦ä¸ºä¸»è¿›ç¨‹ã€‚å•è¿›ç¨‹æ¨¡å¼ä¸‹å§‹ç»ˆè¿”å›ž Trueã€‚"""
    if (not dist.is_available()) or (not dist.is_initialized()):
        return True
    return dist.get_rank() == 0



def _resolve_label_map_path(args) -> Path:
    """
    è§£æž label_map.json çš„ç»å¯¹è·¯å¾„ã€‚

    æ”¯æŒä¸¤ç§å†™æ³•ï¼š
    1) ç›´æŽ¥ä¼ ç»å¯¹è·¯å¾„
    2) ä¼ ç›¸å¯¹äºŽ dataset_root çš„ç›¸å¯¹è·¯å¾„
    """
    path = Path(args.label_map_json)
    if path.is_absolute():
        return path
    return Path(args.dataset_root) / path



def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """è‹¥æ¨¡åž‹è¢« DDP åŒ…è£¹ï¼Œåˆ™è¿”å›ž model.moduleï¼›å¦åˆ™åŽŸæ ·è¿”å›žã€‚"""
    return model.module if hasattr(model, "module") else model



def _extract_single_view_and_labels_and_indices(
    batch: Dict[str, Any],
    tier_mode: str,
):
    """
    ä»Ž refresh é˜¶æ®µçš„ batch ä¸­æå–ï¼š
        - å•ä¸ª RGB view
        - labels
        - global indices

    é¢„æœŸè¾“å…¥ï¼š
        batch["rgb"]       -> Tensor[B, T, 3, H, W]
        batch["tier_ids"]  -> dict æˆ– Tensor
        batch["global_index"] / batch["idx"] / batch["sample_id"] -> LongTensor[B]

    è¿”å›žï¼š
    ------
    rgb:
        å•è§†è§’ RGB clipï¼Œå½¢çŠ¶ [B, T, 3, H, W]ã€‚

    labels:
        å½“å‰ tier çš„æ ‡ç­¾ï¼Œå½¢çŠ¶ [B]ã€‚

    indices:
        æ ·æœ¬åœ¨æ•´ä¸ª map-style æ•°æ®é›†ä¸­çš„ global indexï¼Œå½¢çŠ¶ [B]ã€‚
    """
    rgb = batch["rgb"]
    if isinstance(rgb, dict):
        rgb = next(iter(rgb.values()))

    if not torch.is_tensor(rgb):
        raise RuntimeError(
            "Feature extraction loader is expected to output a single RGB tensor, "
            f"but got type={type(rgb)}"
        )

    tier_ids = batch["tier_ids"]
    labels = tier_ids[tier_mode] if isinstance(tier_ids, dict) else tier_ids

    if "global_index" in batch:
        indices = batch["global_index"]
    elif "idx" in batch:
        indices = batch["idx"]
    elif "sample_id" in batch:
        indices = batch["sample_id"]
    else:
        raise KeyError(
            "Batch does not contain a usable global id field. "
            "Expected one of: global_index / idx / sample_id"
        )

    return rgb, labels, indices


# ============================================================
# æž„å»º refresh é˜¶æ®µä¸“ç”¨ loader
# ============================================================

def build_feature_extraction_loader(
    args,
    cfg: PrototypeRefreshConfig,
) -> torch.utils.data.DataLoader:
    """
    æž„å»º refresh é˜¶æ®µä¸“ç”¨çš„ map-style DataLoaderã€‚

    ä¸Žè®­ç»ƒé˜¶æ®µ loader çš„å…³é”®åŒºåˆ«ï¼š
    --------------------------------
    1) rgb_two_views=False
       refresh åªéœ€è¦å•ä¸ª clipï¼Œä¸éœ€è¦ MoCo çš„åŒè§†è§’è¾“å‡ºã€‚

    2) is_train=False
       refresh ä½¿ç”¨ç¡®å®šæ€§çš„éªŒè¯å¼å˜æ¢ï¼Œé¿å…éšæœºå¢žå¼ºå¹²æ‰°èšç±»ç»“æžœã€‚

    3) shuffle=False, drop_last=False
       refresh å¿…é¡»æ‰«å®Œæ•´ä¸ªè®­ç»ƒé›†ï¼Œå¹¶ä¿æŒ global_index å¯¹é½ã€‚
    """
    label_map_path = _resolve_label_map_path(args)
    label_map = load_label_map_json(label_map_path)

    cfg_loader = PackedMultiModalConfig(
        n_frames=args.n_frames,
        rgb_two_views=False,
        rgb_camera_id=cfg.rgb_camera_id,
        use_modalities=("rgb",),
        missing_policy="skip",
        load_labels=True,
        tier_mode=cfg.tier_mode,
        is_train=False,
        label_map_path=str(label_map_path),
        rgb_mean=cfg.rgb_mean,
        rgb_std=cfg.rgb_std,
    )

    dataset = build_packed_mapstyle_dataset(
        dataset_root=args.dataset_root,
        manifest_name=args.train_manifest_name,
        cfg=cfg_loader,
        label_map=label_map,
        verify_paths_on_init=cfg.verify_paths_on_init,
    )

    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        sampler=None,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
    )
    return loader


# ============================================================
# å…¨é‡æç‰¹å¾
# ============================================================

@torch.no_grad()
def extract_features_momentum(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tier_mode: str,
    main_process_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    ä½¿ç”¨ MoCo çš„ momentum encoderï¼ˆencoder_kï¼‰å¯¹æ•´ä¸ª loader æå–ç‰¹å¾ã€‚

    è¿”å›žï¼š
    ------
    {
        "indices": LongTensor[N],
        "labels":  LongTensor[N],
        "feats":   FloatTensor[N, D],
    }

    è¯´æ˜Žï¼š
    ------
    1) è¿™é‡Œæå–çš„æ˜¯ encoder_k çš„è¾“å‡ºã€‚
    2) ç‰¹å¾ä¼šåœ¨è¿”å›žå‰åš L2 normalizeã€‚
    3) å½“å‰å®žçŽ°é»˜è®¤æŽ¨èåªåœ¨ä¸»è¿›ç¨‹æ‰§è¡Œè¿™ä¸ªè¿‡ç¨‹ã€‚
    """
    if main_process_only and (not is_main_process()):
        return {}

    model_u = _unwrap_model(model)
    if not hasattr(model_u, "encoder_k"):
        raise AttributeError("Model does not have encoder_k; cannot refresh prototypes")

    was_training = model_u.training
    model_u.eval()

    all_indices: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_feats: List[torch.Tensor] = []

    for batch in loader:
        rgb, labels, indices = _extract_single_view_and_labels_and_indices(batch, tier_mode)

        rgb = rgb.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        indices = indices.to(device, non_blocking=True).long()

        # map-style loader è¾“å‡ºä¸º [B, T, 3, H, W]
        # 3D å·ç§¯ç½‘ç»œé€šå¸¸éœ€è¦ [B, 3, T, H, W]
        rgb = rgb.permute(0, 2, 1, 3, 4).contiguous()

        feats = model_u.encoder_k(rgb)
        feats = F.normalize(feats.float(), dim=1)

        all_indices.append(indices.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_feats.append(feats.detach().cpu())

    if was_training:
        model_u.train()

    if len(all_feats) == 0:
        raise RuntimeError("No features were extracted during prototype refresh")

    indices = torch.cat(all_indices, dim=0)
    labels = torch.cat(all_labels, dim=0)
    feats = torch.cat(all_feats, dim=0)

    return {
        "indices": indices,
        "labels": labels,
        "feats": feats,
    }


# ============================================================
# æŒ‰ç±»åˆ«åˆ†ç»„
# ============================================================

def group_features_by_class(
    indices: torch.Tensor,
    labels: torch.Tensor,
    feats: torch.Tensor,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    å°†å…¨é‡æ ·æœ¬æŒ‰ class_id åˆ†ç»„ï¼Œä¾¿äºŽåŽç»­â€œæ¯ç±»å•ç‹¬åš KMeansâ€ã€‚

    è¿”å›žæ ¼å¼ï¼š
    ----------
    grouped[class_id] = {
        "indices": LongTensor[n_c],
        "labels":  LongTensor[n_c],
        "feats":   FloatTensor[n_c, D],
    }
    """
    if indices.ndim != 1 or labels.ndim != 1 or feats.ndim != 2:
        raise ValueError(
            f"Expect indices=[N], labels=[N], feats=[N,D], got shapes "
            f"{tuple(indices.shape)}, {tuple(labels.shape)}, {tuple(feats.shape)}"
        )
    if not (indices.shape[0] == labels.shape[0] == feats.shape[0]):
        raise ValueError("indices, labels, feats must have the same first dimension")

    grouped: Dict[int, Dict[str, torch.Tensor]] = {}
    unique_labels = torch.unique(labels)

    for cls in unique_labels.tolist():
        mask = labels == cls
        grouped[int(cls)] = {
            "indices": indices[mask].clone(),
            "labels": labels[mask].clone(),
            "feats": feats[mask].clone(),
        }

    return grouped


# ============================================================
# KMeans ä¸Ž prototype æ¸©åº¦ä¼°è®¡
# ============================================================

def _pad_1d_to_fixed_k(values: np.ndarray, target_k: int) -> np.ndarray:
    """
    å°†ä¸€ç»´æ•°ç»„è¡¥é½åˆ°å›ºå®šé•¿åº¦ target_kã€‚

    åœ¨ varproto ç‰ˆæœ¬ä¸­ï¼Œprototype_bank çš„ç©ºé—´å½¢çŠ¶ä»ç„¶é‡‡ç”¨ [C, M_max, D]ï¼Œ
    å…¶ä¸­ M_max = max(num_prototypes_per_class)ã€‚

    å› æ­¤ï¼Œå¯¹äºŽæŸä¸ªç±»åˆ«ï¼Œè‹¥å®ƒå®žé™…åªä½¿ç”¨ k_eff ä¸ª prototypeï¼Œ
    é‚£ä¹ˆå®ƒçš„ç›¸å¯¹æ¸©åº¦æ•°ç»„ä¹Ÿéœ€è¦è¡¥é½åˆ° M_maxï¼Œæ‰èƒ½ä¸Ž bank å¯¹é½ã€‚

    è¡¥é½ç­–ç•¥ï¼š
        - è‹¥å·²æœ‰å€¼ï¼Œåˆ™é‡å¤æœ€åŽä¸€ä¸ªå€¼
        - è‹¥æœ¬èº«ä¸ºç©ºï¼Œåˆ™è°ƒç”¨æ–¹ä¸åº”èµ°åˆ°è¿™é‡Œ
    """
    k_eff = values.shape[0]
    if k_eff == target_k:
        return values
    if k_eff <= 0:
        raise ValueError("values must contain at least one element")

    pad_num = target_k - k_eff
    pad = np.repeat(values[-1:], repeats=pad_num, axis=0)
    return np.concatenate([values, pad], axis=0)



def _compute_proto_relative_temperatures(
    z_c: torch.Tensor,
    centers_t: torch.Tensor,
    assign_t: torch.Tensor,
    target_k: int,
    enable_scaling: bool,
    eps: float,
    base_temperature: float = 0.07,
    clamp_percentile_low: float = 10.0,
    clamp_percentile_high: float = 90.0,
    log_count_bias: float = 10.0,
) -> torch.Tensor:
    """
    ä¸ºå½“å‰ç±»åˆ«ä¸­çš„æ¯ä¸ª prototype è®¡ç®—ç›¸å¯¹æ¸©åº¦ç¼©æ”¾ç³»æ•°ã€‚

    è¿™é‡Œé‡‡ç”¨ä¸Ž PCL density é€»è¾‘å…¼å®¹çš„æ–¹å¼ï¼š
        1) å…ˆä¼°è®¡æ¯ä¸ªç°‡çš„ density
        2) å¯¹ density åšåˆ†ä½è£å‰ª
        3) å°† density çš„å‡å€¼ç¼©æ”¾åˆ° base_temperature
        4) å†è½¬æ¢æˆâ€œç›¸å¯¹å€çŽ‡â€è¿”å›ž

    è¿”å›žå€¼å«ä¹‰ï¼š
    ------------
    è¿”å›žçš„æ˜¯ rel_tempï¼Œè€Œä¸æ˜¯æœ€ç»ˆæ¸©åº¦æœ¬èº«ã€‚

    è‹¥è®­ç»ƒæ—¶åœ¨ prototype loss ä¸­ä½¿ç”¨ï¼š
        tau_eff = temperature * rel_temp

    å¹¶ä¸”ä¼ å…¥çš„ temperature == base_temperatureï¼Œ
    é‚£ä¹ˆ tau_eff å°±ç­‰äºŽè¿™é‡Œä¼°è®¡å‡ºæ¥çš„ density æ¸©åº¦ã€‚

    è‹¥ä¸å¯ç”¨ scalingï¼Œåˆ™ç›´æŽ¥è¿”å›žå…¨ 1ã€‚

    æ³¨æ„ï¼š
    ------
    è¿™é‡Œçš„ target_k ä¸æ˜¯å…¨å±€çš„ M_maxï¼Œ
    è€Œæ˜¯â€œè¯¥ç±»åˆ«çœŸæ­£æœ‰æ•ˆçš„ prototype ä¸ªæ•° k_effâ€ã€‚
    ä¹‹åŽå†ç”±è°ƒç”¨æ–¹å°†ç»“æžœå†™å…¥å‰ k_eff ä¸ªæ§½ä½ã€‚
    """
    if not enable_scaling:
        return torch.ones((target_k,), dtype=torch.float32)

    if base_temperature <= 0:
        raise ValueError(f"base_temperature must be > 0, got {base_temperature}")

    if z_c.numel() == 0 or centers_t.numel() == 0:
        return torch.ones((target_k,), dtype=torch.float32)

    z_t = z_c.float()
    centers_t = centers_t.float()
    assign_t = assign_t.long()

    n_c = z_t.shape[0]
    k_eff = centers_t.shape[0]
    if n_c <= 0 or k_eff <= 0:
        return torch.ones((target_k,), dtype=torch.float32)

    dist_sq_all = torch.cdist(z_t, centers_t, p=2) ** 2
    assigned_dist_sq = dist_sq_all[torch.arange(n_c), assign_t]

    density = torch.zeros((k_eff,), dtype=torch.float32)

    for m in range(k_eff):
        mask_m = assign_t == m
        num_m = int(mask_m.sum().item())
        if num_m > 1:
            dist_m = assigned_dist_sq[mask_m]
            d_m = dist_m.sqrt().mean() / torch.log(
                torch.tensor(float(num_m + log_count_bias), dtype=torch.float32)
            )
            density[m] = d_m

    dmax = density.max()
    if dmax.item() <= 0:
        dmax = torch.tensor(1.0, dtype=torch.float32)

    for m in range(k_eff):
        mask_m = assign_t == m
        num_m = int(mask_m.sum().item())
        if num_m <= 1:
            density[m] = dmax

    density_np = density.cpu().numpy()
    low = np.percentile(density_np, clamp_percentile_low)
    high = np.percentile(density_np, clamp_percentile_high)
    density_np = np.clip(density_np, low, high)

    mean_density = float(density_np.mean())
    if mean_density <= eps:
        density_np = np.ones_like(density_np, dtype=np.float32) * float(base_temperature)
    else:
        density_np = float(base_temperature) * density_np / mean_density

    rel_temp_np = density_np / float(base_temperature)
    return torch.from_numpy(rel_temp_np).float()



def run_per_class_kmeans(
    grouped: Dict[int, Dict[str, torch.Tensor]],
    num_classes: int,
    feat_dim: int,
    num_prototypes_per_class: List[int],
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    enable_prototype_temperature_scaling: bool = False,
    base_temperature: float = 0.07,
    proto_temperature_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    å¯¹æ¯ä¸ªç±»åˆ«å•ç‹¬æ‰§è¡Œ KMeansï¼Œå¹¶æž„å»ºè®­ç»ƒé˜¶æ®µæ‰€éœ€çš„ prototype stateã€‚

    è¿™ä¸ª varproto ç‰ˆæœ¬çš„æ ¸å¿ƒè®¾è®¡æ˜¯ï¼š
    --------------------------------
    1) å¯¹å¤–ä»ç„¶è¿”å›žä¸€ä¸ªè‡´å¯†å¼ é‡ prototype_bankï¼Œå½¢çŠ¶å›ºå®šä¸º [C, M_max, D]
       å…¶ä¸­ M_max = max(num_prototypes_per_class)

    2) ä½†æ¯ä¸ªç±»åˆ«çœŸæ­£æœ‰æ•ˆçš„ prototype ä¸ªæ•°ç”± class_num_prototypes[cls] æŒ‡å®š
       åªæœ‰ bank[cls, :class_num_prototypes[cls]] æ‰æ˜¯æœ‰æ•ˆæ§½ä½

    3) æ‰€æœ‰ä¸‹æ¸¸æ¨¡å—ï¼ˆprototype loss / directional loss / EMA updateï¼‰
       éƒ½åº”è¯¥ä½¿ç”¨ class_num_prototypes æ¥è¿‡æ»¤ padded æ§½ä½

    è¿”å›žçš„æ ¸å¿ƒå­—æ®µï¼š
    ------------------
    prototype_bank:
        [C, M_max, D]ï¼Œè‡´å¯†å­˜å‚¨çš„ prototype bankã€‚

    proto_rel_temperature_bank:
        [C, M_max]ï¼Œæ¯ä¸ª prototype å¯¹åº”çš„ç›¸å¯¹æ¸©åº¦å€çŽ‡ã€‚
        å¯¹äºŽ padded æ§½ä½ï¼Œé»˜è®¤ä¿æŒä¸º 1ã€‚

    class_num_prototypes:
        [C]ï¼Œè®°å½•æ¯ä¸ªç±»åˆ«å½“å‰çœŸæ­£æœ‰æ•ˆçš„ prototype æ•°é‡ã€‚

    sample_to_proto:
        [N_slot]ï¼Œè®°å½•æ¯ä¸ª global_index å±žäºŽè¯¥ç±»åˆ«å†…éƒ¨çš„å“ªä¸ª prototypeã€‚
        æ³¨æ„è¿™é‡Œå­˜çš„æ˜¯â€œç±»å†… prototype idâ€ï¼ŒèŒƒå›´æ˜¯ [0, class_num_prototypes[cls)-1]ã€‚

    sample_to_class:
        [N_slot]ï¼Œè®°å½•æ¯ä¸ª global_index çš„ç±»åˆ« idã€‚

    valid_sample_mask:
        [N_slot]ï¼Œæ ‡è®°å“ªäº› global_index åœ¨ refresh æ•°æ®é›†ä¸­å®žé™…æœ‰æ•ˆã€‚
    """
    if len(num_prototypes_per_class) != num_classes:
        raise ValueError(
            f"num_prototypes_per_class length mismatch: expect {num_classes}, "
            f"got {len(num_prototypes_per_class)}"
        )
    if any(int(v) <= 0 for v in num_prototypes_per_class):
        raise ValueError(f"All prototype counts must be > 0, got {num_prototypes_per_class}")

    all_indices = [item["indices"] for item in grouped.values()]
    if len(all_indices) == 0:
        raise RuntimeError("grouped is empty; cannot run KMeans")

    max_index = int(torch.cat(all_indices, dim=0).max().item())
    num_slots = max_index + 1

    requested_num_prototypes_per_class = [int(x) for x in num_prototypes_per_class]
    max_num_prototypes = int(max(requested_num_prototypes_per_class))

    prototype_bank = torch.zeros((num_classes, max_num_prototypes, feat_dim), dtype=torch.float32)
    proto_rel_temperature_bank = torch.ones((num_classes, max_num_prototypes), dtype=torch.float32)
    class_num_prototypes = torch.zeros((num_classes,), dtype=torch.long)

    sample_to_proto = torch.full((num_slots,), -1, dtype=torch.long)
    sample_to_class = torch.full((num_slots,), -1, dtype=torch.long)
    valid_sample_mask = torch.zeros((num_slots,), dtype=torch.bool)

    counts_per_class: Dict[int, int] = {}
    cluster_meta: Dict[int, Dict[str, Any]] = {}

    for cls in range(num_classes):
        requested_k = requested_num_prototypes_per_class[cls]

        if cls not in grouped:
            counts_per_class[cls] = 0
            class_num_prototypes[cls] = 0
            cluster_meta[cls] = {
                "n_samples": 0,
                "requested_k": int(requested_k),
                "k_eff": 0,
                "note": "no samples for this class",
                "rel_temperatures": [],
            }
            continue

        idx_c = grouped[cls]["indices"].long()
        y_c = grouped[cls]["labels"].long()
        z_c = grouped[cls]["feats"].float()

        n_c = int(z_c.shape[0])
        counts_per_class[cls] = n_c

        sample_to_class[idx_c] = y_c
        valid_sample_mask[idx_c] = True

        if n_c <= 0:
            class_num_prototypes[cls] = 0
            cluster_meta[cls] = {
                "n_samples": 0,
                "requested_k": int(requested_k),
                "k_eff": 0,
                "note": "empty class",
                "rel_temperatures": [],
            }
            continue

        # æŸä¸€ç±»çš„æœ‰æ•ˆ prototype ä¸ªæ•°ä¸èƒ½è¶…è¿‡è¯¥ç±»æ ·æœ¬æ•°ã€‚
        k_eff = min(int(requested_k), n_c)
        class_num_prototypes[cls] = int(k_eff)

        z_np = z_c.numpy()

        if n_c == 1:
            centers = z_np.copy()
            assign = np.zeros((1,), dtype=np.int64)
            inertia = 0.0
        else:
            km = KMeans(
                n_clusters=k_eff,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
            )
            assign = km.fit_predict(z_np)
            centers = km.cluster_centers_
            inertia = float(km.inertia_)

        centers_t = torch.from_numpy(centers).float()
        centers_t = F.normalize(centers_t, dim=1)

        prototype_bank[cls, :k_eff] = centers_t

        assign_t = torch.from_numpy(assign).long()
        sample_to_proto[idx_c] = assign_t

        rel_temp_eff = _compute_proto_relative_temperatures(
            z_c=z_c,
            centers_t=centers_t,
            assign_t=assign_t,
            target_k=k_eff,
            enable_scaling=enable_prototype_temperature_scaling,
            eps=proto_temperature_eps,
            base_temperature=base_temperature,
            clamp_percentile_low=10.0,
            clamp_percentile_high=90.0,
            log_count_bias=10.0,
        )
        proto_rel_temperature_bank[cls, :k_eff] = rel_temp_eff

        cluster_meta[cls] = {
            "n_samples": int(n_c),
            "requested_k": int(requested_k),
            "k_eff": int(k_eff),
            "inertia": inertia,
            "note": "kmeans" if n_c > 1 else "single-sample class",
            "rel_temperatures": [float(x) for x in rel_temp_eff.tolist()],
        }

    # åªå¯¹çœŸæ­£æœ‰æ•ˆçš„æ§½ä½åš normalizeã€‚
    # padded æ§½ä½ä¿æŒä¸º 0ï¼Œä¸å‚ä¸Žè®­ç»ƒã€‚
    for cls in range(num_classes):
        k_eff = int(class_num_prototypes[cls].item())
        if k_eff > 0:
            prototype_bank[cls, :k_eff] = F.normalize(prototype_bank[cls, :k_eff], dim=1)

    return {
        "prototype_bank": prototype_bank,
        "proto_rel_temperature_bank": proto_rel_temperature_bank,
        "class_num_prototypes": class_num_prototypes,
        "max_num_prototypes": max_num_prototypes,
        "sample_to_proto": sample_to_proto,
        "sample_to_class": sample_to_class,
        "valid_sample_mask": valid_sample_mask,
        "counts_per_class": counts_per_class,
        "cluster_meta": cluster_meta,
    }


# ============================================================
# é«˜å±‚æŽ¥å£ï¼šåˆ·æ–° prototypes
# ============================================================

def refresh_prototypes(
    model: torch.nn.Module,
    args,
    cfg: PrototypeRefreshConfig,
) -> Dict[str, Any]:
    """
    åˆ·æ–° prototype çš„é«˜å±‚å…¥å£ã€‚

    æµç¨‹ï¼š
    ------
    1) æž„å»º refresh ä¸“ç”¨ loader
    2) ç”¨ momentum encoder å¯¹æ•´ä¸ªè®­ç»ƒé›†æç‰¹å¾
    3) æŒ‰ class_id åˆ†ç»„
    4) å¯¹æ¯ä¸ªç±»åˆ«å•ç‹¬åš KMeans
    5) è¿”å›žè®­ç»ƒé˜¶æ®µçœŸæ­£éœ€è¦çš„ prototype state

    è¿”å›žå­—æ®µï¼š
    ----------
    prototype_bank:
        [C, M_max, D]

    proto_rel_temperature_bank:
        [C, M_max]

    class_num_prototypes:
        [C]

    sample_to_proto:
        [N_slot]

    sample_to_class:
        [N_slot]
        è¿™æ˜¯æŒ‰ä½ çš„è¦æ±‚ä¿ç•™çš„è¾“å‡ºå­—æ®µã€‚
        å®ƒåœ¨å½“å‰è®­ç»ƒä¸»å¾ªçŽ¯ä¸­ä¸æ˜¯å¿…é¡»çš„ï¼Œä½†å¯¹è°ƒè¯•å’ŒåŽç»­æ£€æŸ¥å¾ˆæœ‰å¸®åŠ©ã€‚

    valid_sample_mask:
        [N_slot]

    counts_per_class / cluster_meta:
        ä¾¿äºŽæ‰“å°å’Œè°ƒè¯•
    """
    if cfg.require_main_process_only and (not is_main_process()):
        return {}

    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = load_label_map_json(_resolve_label_map_path(args))
    if cfg.tier_mode not in label_map:
        raise KeyError(f"tier_mode={cfg.tier_mode} not found in label_map.json")

    num_classes = len(label_map[cfg.tier_mode])
    if num_classes <= 0:
        raise ValueError(f"No classes found for tier={cfg.tier_mode}")

    if cfg.num_prototypes_per_class is None:
        num_prototypes_per_class = [int(cfg.default_num_prototypes)] * num_classes
    else:
        num_prototypes_per_class = [int(x) for x in cfg.num_prototypes_per_class]
        if len(num_prototypes_per_class) != num_classes:
            raise ValueError(
                f"num_prototypes_per_class length mismatch: expect {num_classes}, "
                f"got {len(num_prototypes_per_class)}"
            )
        if any(x <= 0 for x in num_prototypes_per_class):
            raise ValueError(f"All prototype counts must be > 0, got {num_prototypes_per_class}")

    feat_loader = build_feature_extraction_loader(args, cfg)

    extracted = extract_features_momentum(
        model=model,
        loader=feat_loader,
        device=device,
        tier_mode=cfg.tier_mode,
        main_process_only=cfg.require_main_process_only,
    )

    indices = extracted["indices"].long()
    labels = extracted["labels"].long()
    feats = extracted["feats"].float()

    if feats.ndim != 2 or feats.shape[0] <= 0:
        raise RuntimeError(f"Invalid extracted feature shape: {tuple(feats.shape)}")

    feat_dim = int(feats.shape[1])
    grouped = group_features_by_class(indices=indices, labels=labels, feats=feats)

    clustered = run_per_class_kmeans(
        grouped=grouped,
        num_classes=num_classes,
        feat_dim=feat_dim,
        num_prototypes_per_class=num_prototypes_per_class,
        random_state=cfg.random_state,
        n_init=cfg.n_init,
        max_iter=cfg.max_iter,
        enable_prototype_temperature_scaling=cfg.enable_prototype_temperature_scaling,
        base_temperature=cfg.proto_base_temperature,
        proto_temperature_eps=cfg.proto_temperature_eps,
    )

    return {
        "prototype_bank": clustered["prototype_bank"],
        "proto_rel_temperature_bank": clustered["proto_rel_temperature_bank"],
        "class_num_prototypes": clustered["class_num_prototypes"],
        "sample_to_proto": clustered["sample_to_proto"],
        "sample_to_class": clustered["sample_to_class"],
        "valid_sample_mask": clustered["valid_sample_mask"],
        "counts_per_class": clustered["counts_per_class"],
        "cluster_meta": clustered["cluster_meta"],
        "tier_mode": cfg.tier_mode,
        "num_classes": num_classes,
        "requested_num_prototypes_per_class": num_prototypes_per_class,
        "max_num_prototypes": clustered["max_num_prototypes"],
        "feat_dim": feat_dim,
        "enable_prototype_temperature_scaling": cfg.enable_prototype_temperature_scaling,
    }


# ============================================================
# è¾…åŠ©æ‰“å°ä¸Žå¹¿æ’­
# ============================================================

def summarize_proto_state(proto_state: Dict[str, Any]) -> str:
    """
    å°† prototype state æ ¼å¼åŒ–æˆä¸€è¡Œç®€æ´å­—ç¬¦ä¸²ï¼Œä¾¿äºŽæ—¥å¿—æ‰“å°ã€‚
    """
    if not proto_state:
        return "[prototype_refresh] empty state"

    num_classes = proto_state["num_classes"]
    max_num_prototypes = proto_state["max_num_prototypes"]
    feat_dim = proto_state["feat_dim"]
    counts_per_class = proto_state["counts_per_class"]
    requested_num_prototypes_per_class = proto_state["requested_num_prototypes_per_class"]
    use_temp_scaling = proto_state.get("enable_prototype_temperature_scaling", False)

    valid_classes = sum(1 for _, c in counts_per_class.items() if c > 0)
    total_samples = sum(int(c) for c in counts_per_class.values())

    return (
        f"[prototype_refresh] total_samples={total_samples}, "
        f"valid_classes={valid_classes}/{num_classes}, "
        f"max_num_prototypes={max_num_prototypes}, "
        f"requested_per_class={requested_num_prototypes_per_class}, "
        f"feat_dim={feat_dim}, proto_temp_scaling={use_temp_scaling}"
    )



def broadcast_proto_state(
    proto_state: Optional[dict],
    device: torch.device,
    rank: int,
) -> Optional[dict]:
    """
    å°†è®­ç»ƒé˜¶æ®µçœŸæ­£éœ€è¦çš„ prototype çŠ¶æ€å¹¿æ’­åˆ°æ‰€æœ‰ rankã€‚

    å¹¿æ’­å­—æ®µï¼š
    ----------
    - prototype_bank
    - proto_rel_temperature_bank
    - class_num_prototypes
    - sample_to_proto
    - valid_sample_mask
    - enable_prototype_temperature_scaling

    å¦å¤–ï¼Œä¸ºäº†ä¿ç•™è°ƒè¯•ä¾¿åˆ©æ€§ï¼Œæœ¬å®žçŽ°ä¹Ÿä¼šæŠŠ sample_to_class ä¸€å¹¶å¸¦ä¸Šã€‚
    è®­ç»ƒæœ¬èº«å¹¶ä¸ä¾èµ– sample_to_classï¼Œä½†ä¿ç•™å®ƒä¸ä¼šå½±å“ä¸»æµç¨‹ã€‚

    è¯´æ˜Žï¼š
    ------
    1) å•è¿›ç¨‹æ¨¡å¼ä¸‹ï¼Œåªåš device æ¬è¿ã€‚
    2) å¤šè¿›ç¨‹æ¨¡å¼ä¸‹ï¼Œè¦æ±‚ rank 0 çš„ proto_state éžç©ºï¼Œå…¶ä½™ rank å¯ä¼  Noneã€‚
    """
    if proto_state is None:
        return None

    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
        out = dict(proto_state)
        out["prototype_bank"] = out["prototype_bank"].to(device)
        out["proto_rel_temperature_bank"] = out["proto_rel_temperature_bank"].to(device)
        out["class_num_prototypes"] = out["class_num_prototypes"].to(device).long()
        out["sample_to_proto"] = out["sample_to_proto"].to(device).long()
        out["sample_to_class"] = out["sample_to_class"].to(device).long()
        out["valid_sample_mask"] = out["valid_sample_mask"].to(device).bool()
        return out

    obj_list = [None]
    if rank == 0:
        obj_list[0] = {
            "prototype_bank": proto_state["prototype_bank"].cpu(),
            "proto_rel_temperature_bank": proto_state["proto_rel_temperature_bank"].cpu(),
            "class_num_prototypes": proto_state["class_num_prototypes"].cpu(),
            "sample_to_proto": proto_state["sample_to_proto"].cpu(),
            "sample_to_class": proto_state["sample_to_class"].cpu(),
            "valid_sample_mask": proto_state["valid_sample_mask"].cpu(),
            "enable_prototype_temperature_scaling": proto_state.get(
                "enable_prototype_temperature_scaling", False
            ),
        }

    dist.broadcast_object_list(obj_list, src=0)
    out = obj_list[0]
    out["prototype_bank"] = out["prototype_bank"].to(device, non_blocking=True)
    out["proto_rel_temperature_bank"] = out["proto_rel_temperature_bank"].to(device, non_blocking=True)
    out["class_num_prototypes"] = out["class_num_prototypes"].to(device, non_blocking=True).long()
    out["sample_to_proto"] = out["sample_to_proto"].to(device, non_blocking=True).long()
    out["sample_to_class"] = out["sample_to_class"].to(device, non_blocking=True).long()
    out["valid_sample_mask"] = out["valid_sample_mask"].to(device, non_blocking=True).bool()
    return out


