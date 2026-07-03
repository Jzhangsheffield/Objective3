#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_feature_clusters.py

面向当前“特征提取 -> 聚类分析 -> 样本回溯”流程的可视化脚本。

本脚本的设计目标不是“靠肉眼重新找最优 K”，而是：
1) 使用同一份 2D 投影，解释全局聚类分析结果；
2) 对单个真实类别单独重新投影，解释类内子簇；
3) 将每张图背后的点坐标与样本信息一并保存，便于快速回查原始样本。

推荐输入：
- features: 提取好的高维特征 [N, D]，支持 .npy / .pt
- sample_lookup: analyze_feature_clusters*.py 输出的 sample_lookup.csv / sample_lookup.jsonl
  其中至少应包含：
    - key
    - label
    - label_name（若没有，可结合 label_map_json 生成）
  若 sample_lookup 中还包含：
    - global_kmeans_cluster
    - global_hdbscan_cluster
    - global_hdbscan_probability
    - local_kmeans_cluster
    - local_hdbscan_cluster
    - local_hdbscan_probability
    - lighting / pos / person / original_key / sample_name ...
  则脚本会利用这些信息做更有解释力的可视化。

核心可视化策略：
A) 全局可视化（global）
   - 对全体样本做一次 PCA / UMAP / t-SNE 投影
   - 在同一套 2D 坐标上，分别按以下字段上色：
       * 真实标签（label_name）
       * 全局 KMeans 簇（global_kmeans_cluster）
       * 全局 HDBSCAN 簇（global_hdbscan_cluster）
       * HDBSCAN 噪声标记（global_hdbscan_noise）
       * HDBSCAN 概率（global_hdbscan_probability）
       * 以及 metadata（lighting / pos / person 等）

B) 类内可视化（per-class rerun projection）
   - 对每个真实类别单独重新投影（推荐用 UMAP）
   - 用于解释：为什么某个类别的 local recommended K > 1，或者为什么该类别内有多个 HDBSCAN 子簇
   - 在每个类别自己的 2D 图中，分别按以下字段上色：
       * local_kmeans_cluster
       * local_hdbscan_cluster
       * local_hdbscan_probability
       * metadata（lighting / pos / person 等）

C) 点坐标表（point tables）
   - 每次投影后，脚本都会把 2D 坐标和 sample_lookup 中的字段一起保存为 CSV
   - 方便你直接筛选特定簇、特定 metadata、特定噪声点并回查 key / original_key

严格性说明：
- 不做静默兜底。
- 输入长度不一致、sample_lookup 缺关键字段、row_index 不一致、指定列不存在，都会直接报错。
- PCA / UMAP / t-SNE 仅按用户指定的方法执行；若依赖库不存在，则直接报错。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ============================================================
# 1) 基础加载与校验
# ============================================================

def load_array_file(path: str | Path, name: str) -> np.ndarray:
    """
    读取 .npy / .pt 数组。

    支持：
    - .npy -> numpy.ndarray
    - .pt  -> torch.Tensor 或 numpy.ndarray

    不接受 dict / list / 其他不明确对象。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{name} file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            raise TypeError(
                f"Unsupported object type in {name} .pt file: {type(obj)}. "
                "Expected torch.Tensor or numpy.ndarray."
            )
    else:
        raise ValueError(f"Unsupported file suffix for {name}: {suffix}. Only .npy and .pt are supported.")

    return np.asarray(arr)



def ensure_features_2d(features: np.ndarray) -> np.ndarray:
    """要求特征为 [N, D]，并且无 NaN/Inf。"""
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D [N, D], but got shape {features.shape}")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Features must be non-empty, but got shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or Inf.")
    return features.astype(np.float64, copy=False)



def load_sample_lookup(path: str | Path) -> pd.DataFrame:
    """
    读取 sample_lookup.csv 或 sample_lookup.jsonl。

    要求：
    - 必须有 key 和 label 两列
    - 若存在 row_index，则必须唯一且等于 0..N-1
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"sample_lookup file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".json"}:
        # 这里要求 jsonl 是 records+lines 格式
        df = pd.read_json(path, orient="records", lines=True)
    else:
        raise ValueError("sample_lookup must be .csv or .jsonl")

    if len(df) == 0:
        raise ValueError("sample_lookup is empty.")
    if "key" not in df.columns:
        raise KeyError("sample_lookup must contain column 'key'.")
    if "label" not in df.columns:
        raise KeyError("sample_lookup must contain column 'label'.")

    if df["key"].isna().any():
        raise ValueError("sample_lookup contains NaN in column 'key'.")
    if df["key"].duplicated().any():
        dup = df.loc[df["key"].duplicated(), "key"].iloc[0]
        raise ValueError(f"sample_lookup contains duplicate key: {dup}")

    if "row_index" in df.columns:
        if df["row_index"].isna().any():
            raise ValueError("sample_lookup contains NaN in column 'row_index'.")
        if df["row_index"].duplicated().any():
            raise ValueError("sample_lookup contains duplicate row_index values.")
        df = df.sort_values("row_index").reset_index(drop=True)
        expected = np.arange(len(df))
        actual = df["row_index"].to_numpy()
        if not np.array_equal(actual, expected):
            raise ValueError(
                "sample_lookup.row_index is not exactly 0..N-1 after sorting. "
                "This script requires row_index to match feature row order."
            )
    else:
        # 没有 row_index 时，默认当前文件顺序就是特征顺序
        df = df.reset_index(drop=True)
        df.insert(0, "row_index", np.arange(len(df), dtype=np.int64))

    return df



def load_label_map(path: Optional[str | Path], tier: Optional[str] = None) -> Dict[int, str]:
    """
    可选读取 label_map.json，并构造 int_label -> class_name 映射。

    支持三种格式：
    1) 扁平：{"class_name": 0, ...}
    2) 扁平：{"0": "class_name", ...}
    3) 嵌套：{"tier1": {...}, "tier2": {...}, ...}
    """
    if path is None:
        return {}

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label_map_json does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or len(obj) == 0:
        raise ValueError("label_map_json must be a non-empty JSON object.")

    keys = list(obj.keys())
    values = list(obj.values())

    # 1) class_name -> int_id
    if all(isinstance(k, str) for k in keys) and all(isinstance(v, int) for v in values):
        out: Dict[int, str] = {}
        for class_name, label_id in obj.items():
            if label_id in out:
                raise ValueError(f"Duplicate label id in label_map_json: {label_id}")
            out[int(label_id)] = str(class_name)
        return out

    # 2) "0" -> class_name
    if all(isinstance(k, str) for k in keys) and all(isinstance(v, str) for v in values):
        out2: Dict[int, str] = {}
        try:
            for k, v in obj.items():
                kk = int(k)
                if kk in out2:
                    raise ValueError(f"Duplicate label id in label_map_json: {kk}")
                out2[kk] = v
            return out2
        except ValueError:
            pass

    # 3) nested tier map
    if any(k in obj for k in ("tier1", "tier2", "tier3")):
        if tier is None:
            raise ValueError(
                "label_map_json is a nested tier map, but no tier was provided. "
                "Please set --label_map_tier."
            )
        if tier not in obj:
            raise KeyError(f"Requested tier '{tier}' not found in label_map_json. Available keys: {list(obj.keys())}")
        tier_map = obj[tier]
        if not isinstance(tier_map, dict):
            raise TypeError(f"label_map_json[{tier!r}] must be a JSON object.")
        out3: Dict[int, str] = {}
        for class_name, label_id in tier_map.items():
            if not isinstance(class_name, str) or not isinstance(label_id, int):
                raise TypeError(f"Invalid nested label map item in {tier}: {class_name!r} -> {label_id!r}")
            if label_id in out3:
                raise ValueError(f"Duplicate label id in label_map_json[{tier}]: {label_id}")
            out3[int(label_id)] = class_name
        return out3

    raise ValueError("Unsupported label_map_json format.")



def attach_label_names(df: pd.DataFrame, id_to_name: Dict[int, str]) -> pd.DataFrame:
    """
    确保 sample_lookup 中存在 label_name 列。

    规则：
    - 若已有 label_name，则保留；但若存在 NaN，则用 id_to_name/str(label) 填充
    - 若没有 label_name，则根据 label_map_json 或 str(label) 生成
    """
    out = df.copy()

    label_values = out["label"].to_numpy()
    if "label_name" not in out.columns:
        out["label_name"] = [id_to_name.get(int(x), str(int(x))) for x in label_values]
    else:
        mask = out["label_name"].isna()
        if mask.any():
            out.loc[mask, "label_name"] = [id_to_name.get(int(x), str(int(x))) for x in out.loc[mask, "label"]]

    return out



def validate_alignment(features: np.ndarray, sample_lookup: pd.DataFrame) -> None:
    """要求特征行数与 sample_lookup 行数一致。"""
    if features.shape[0] != len(sample_lookup):
        raise ValueError(
            f"Feature row count ({features.shape[0]}) != sample_lookup row count ({len(sample_lookup)})"
        )



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) 特征预处理与降维
# ============================================================

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)



def pca_2d(x: np.ndarray) -> np.ndarray:
    """
    通过 SVD 实现 PCA 2D。

    优点：
    - 确定性强
    - 适合作为全局参考图
    """
    x0 = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x0, full_matrices=False)
    z = u[:, :2] * s[:2]
    return z.astype(np.float32)



def umap_2d(x: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int) -> np.ndarray:
    """UMAP 2D；若 umap-learn 未安装则直接报错。"""
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "UMAP is requested, but package 'umap-learn' is not installed. "
            "Please install it first."
        ) from e

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    z = reducer.fit_transform(x)
    return np.asarray(z, dtype=np.float32)



def tsne_2d(x: np.ndarray, perplexity: float, n_iter: int, seed: int) -> np.ndarray:
    """
    t-SNE 2D。

    注意：
    - 要求 perplexity < N
    - 全局距离不如 PCA / UMAP 那么稳定，建议作为补充图而不是主图
    """
    from sklearn.manifold import TSNE

    n = x.shape[0]
    if perplexity >= n:
        raise ValueError(f"t-SNE requires perplexity < number of samples, but got perplexity={perplexity}, N={n}")

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        verbose=1,
    )
    z = model.fit_transform(x)
    return np.asarray(z, dtype=np.float32)



def project_embeddings(x: np.ndarray, method: str, args: argparse.Namespace) -> np.ndarray:
    """根据 method 选择一种 2D 投影方法。"""
    if method == "pca":
        return pca_2d(x)
    if method == "umap":
        return umap_2d(
            x,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            seed=args.random_state,
        )
    if method == "tsne":
        return tsne_2d(
            x,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_iter,
            seed=args.random_state,
        )
    raise ValueError(f"Unsupported projection method: {method}")


# ============================================================
# 3) 上色字段与绘图工具
# ============================================================

def sanitize_name(name: Any) -> str:
    s = str(name)
    bad = '<>:"/\\|?*\n\r\t'
    for ch in bad:
        s = s.replace(ch, "_")
    s = s.strip().replace(" ", "_")
    return s if s else "unnamed"



def resolve_requested_columns(
    df: pd.DataFrame,
    requested: Optional[Sequence[str]],
    defaults: Sequence[str],
    derived_supported: Sequence[str],
    context: str,
) -> List[str]:
    """
    解析用户请求的 color_by 列。

    规则：
    - 若 requested 为 None，则从 defaults 中选取“当前存在 or 可推导”的列
    - 若 requested 非空，则每一列都必须存在或可推导；否则直接报错
    """
    derived_set = set(derived_supported)

    def is_available(col: str) -> bool:
        return (col in df.columns) or (col in derived_set)

    if requested is None:
        cols = [c for c in defaults if is_available(c)]
        if len(cols) == 0:
            raise ValueError(f"No default color_by columns are available for {context} visualization.")
        return cols

    cols2 = list(requested)
    if len(cols2) == 0:
        raise ValueError(f"Requested {context} color_by list is empty.")
    missing = [c for c in cols2 if not is_available(c)]
    if missing:
        raise KeyError(f"Requested {context} color_by columns are missing: {missing}")
    return cols2



def make_derived_column(df: pd.DataFrame, col: str) -> pd.Series:
    """生成派生列。"""
    if col == "global_hdbscan_noise":
        if "global_hdbscan_cluster" not in df.columns:
            raise KeyError("Cannot derive global_hdbscan_noise: missing global_hdbscan_cluster")
        return (df["global_hdbscan_cluster"].astype("Int64") == -1).map({True: "noise", False: "non_noise"})

    if col == "local_hdbscan_noise":
        if "local_hdbscan_cluster" not in df.columns:
            raise KeyError("Cannot derive local_hdbscan_noise: missing local_hdbscan_cluster")
        return (df["local_hdbscan_cluster"].astype("Int64") == -1).map({True: "noise", False: "non_noise"})

    raise KeyError(f"Unsupported derived column: {col}")



def get_color_series(df: pd.DataFrame, col: str) -> pd.Series:
    """从真实列或派生列中取出用于上色的 Series。"""
    if col in df.columns:
        return df[col]
    return make_derived_column(df, col)



def infer_color_mode(series: pd.Series, col: str) -> str:
    """
    判断某个字段应该按“离散类别”还是“连续数值”上色。

    当前规则：
    - 概率列（*_probability / probs_true）视为连续
    - object / string / bool 视为离散
    - 数值型：若唯一值 <= 20，则按离散；否则按连续
    """
    if col.endswith("_probability") or col == "probs_true":
        return "continuous"

    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return "categorical"

    if pd.api.types.is_numeric_dtype(series):
        nunique = int(series.nunique(dropna=True))
        return "categorical" if nunique <= 20 else "continuous"

    return "categorical"



def build_categorical_legend_values(series: pd.Series) -> List[Any]:
    """生成分类图例顺序。"""
    s = series.copy()
    s = s.astype("object")
    s = s.where(~pd.isna(s), "<NA>")
    vals = list(pd.unique(s))

    # 尝试将“看起来像整数”的类别按整数排序
    def try_int(x: Any) -> Tuple[int, Any]:
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))

    vals.sort(key=try_int)
    return vals



def scatter_categorical(
    coords: np.ndarray,
    df: pd.DataFrame,
    color_col: str,
    out_png: Path,
    title: str,
    point_size: float,
    alpha: float,
    annotate_centroids: bool,
) -> None:
    """
    绘制按离散类别上色的 2D 散点图。
    """
    series = get_color_series(df, color_col).copy()
    series = series.astype("object")
    series = series.where(~pd.isna(series), "<NA>")
    categories = build_categorical_legend_values(series)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")

    for i, cat in enumerate(categories):
        mask = (series == cat).to_numpy()
        color = cmap(i % cmap.N)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=str(cat),
            edgecolors="none",
            c=[color],
        )

        if annotate_centroids and np.any(mask):
            centroid = coords[mask].mean(axis=0)
            ax.text(float(centroid[0]), float(centroid[1]), str(cat), fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)



def scatter_continuous(
    coords: np.ndarray,
    df: pd.DataFrame,
    color_col: str,
    out_png: Path,
    title: str,
    point_size: float,
    alpha: float,
) -> None:
    """
    绘制按连续数值上色的 2D 散点图。

    NaN 会单独以浅灰色绘制。
    """
    values = pd.to_numeric(get_color_series(df, color_col), errors="coerce")
    arr = values.to_numpy(dtype=np.float64)
    valid = np.isfinite(arr)

    fig, ax = plt.subplots(figsize=(10, 8))

    if np.any(~valid):
        ax.scatter(
            coords[~valid, 0],
            coords[~valid, 1],
            s=point_size,
            alpha=alpha,
            c="lightgray",
            edgecolors="none",
            label="<NA>",
        )

    sc = ax.scatter(
        coords[valid, 0],
        coords[valid, 1],
        s=point_size,
        alpha=alpha,
        c=arr[valid],
        cmap="viridis",
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_col)

    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    if np.any(~valid):
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)



def save_projection_table(coords: np.ndarray, df: pd.DataFrame, out_csv: Path) -> None:
    """
    保存当前投影对应的点坐标表。

    输出表包含：
    - x, y
    - sample_lookup 中的全部列
    """
    if coords.shape[0] != len(df):
        raise ValueError("Coordinate count does not match dataframe row count when saving projection table.")

    out_df = df.copy()
    out_df.insert(0, "y", coords[:, 1])
    out_df.insert(0, "x", coords[:, 0])
    out_df.to_csv(out_csv, index=False)


# ============================================================
# 4) 全局与类内可视化主逻辑
# ============================================================

def run_global_visualization(
    features: np.ndarray,
    sample_lookup: pd.DataFrame,
    methods: Sequence[str],
    color_bys: Sequence[str],
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    """
    全局可视化：对全部样本投影一次，然后在同一坐标上按多个字段上色。
    """
    global_dir = out_root / "global"
    ensure_dir(global_dir)

    for method in methods:
        print(f"[global] projecting with {method} ...")
        coords = project_embeddings(features, method, args)
        if coords.shape != (features.shape[0], 2):
            raise ValueError(f"Projection output for method={method} must be [N, 2], but got {coords.shape}")

        method_dir = global_dir / method
        figures_dir = method_dir / "figures"
        ensure_dir(figures_dir)

        save_projection_table(coords, sample_lookup, method_dir / "projection_points.csv")

        for color_col in color_bys:
            series = get_color_series(sample_lookup, color_col)
            mode = infer_color_mode(series, color_col)
            out_png = figures_dir / f"color_by_{sanitize_name(color_col)}.png"
            title = f"GLOBAL | {method.upper()} | color_by={color_col}"

            if mode == "categorical":
                scatter_categorical(
                    coords=coords,
                    df=sample_lookup,
                    color_col=color_col,
                    out_png=out_png,
                    title=title,
                    point_size=args.point_size,
                    alpha=args.alpha,
                    annotate_centroids=args.annotate_centroids,
                )
            else:
                scatter_continuous(
                    coords=coords,
                    df=sample_lookup,
                    color_col=color_col,
                    out_png=out_png,
                    title=title,
                    point_size=args.point_size,
                    alpha=args.alpha,
                )



def resolve_class_subset(sample_lookup: pd.DataFrame, requested: Optional[Sequence[str]]) -> List[Tuple[int, str]]:
    """
    解析需要做类内可视化的类别列表。

    返回：[(label_id, label_name), ...]

    支持：
    - 不指定 -> 全部类别
    - 指定整数标签，如 0 1 2
    - 指定类别名，如 take push
    """
    pairs = sample_lookup[["label", "label_name"]].drop_duplicates().copy()
    pairs["label"] = pairs["label"].astype(int)
    pairs = pairs.sort_values(["label", "label_name"]).reset_index(drop=True)

    available = [(int(r.label), str(r.label_name)) for r in pairs.itertuples(index=False)]
    id_to_name = {lab: name for lab, name in available}
    name_to_id: Dict[str, int] = {}
    for lab, name in available:
        if name in name_to_id and name_to_id[name] != lab:
            raise ValueError(f"label_name '{name}' is duplicated across different label ids.")
        name_to_id[name] = lab

    if requested is None:
        return available

    out: List[Tuple[int, str]] = []
    for item in requested:
        try:
            lab = int(item)
            if lab not in id_to_name:
                raise KeyError(f"Requested class id {lab} is not present in sample_lookup.")
            out.append((lab, id_to_name[lab]))
        except ValueError:
            if item not in name_to_id:
                raise KeyError(f"Requested class name '{item}' is not present in sample_lookup.")
            lab2 = name_to_id[item]
            out.append((lab2, id_to_name[lab2]))

    # 保持用户请求顺序，并去重
    seen = set()
    dedup: List[Tuple[int, str]] = []
    for x in out:
        if x[0] not in seen:
            dedup.append(x)
            seen.add(x[0])
    return dedup



def run_per_class_visualization(
    features: np.ndarray,
    sample_lookup: pd.DataFrame,
    methods: Sequence[str],
    color_bys: Sequence[str],
    selected_classes: Sequence[Tuple[int, str]],
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    """
    类内可视化：每个真实类别单独重新投影。

    这是解释“类内为什么会被分成多个子簇”的主分析入口。
    """
    per_class_root = out_root / "per_class_rerun"
    ensure_dir(per_class_root)

    for class_id, class_name in selected_classes:
        mask = (sample_lookup["label"].astype(int).to_numpy() == int(class_id))
        if not np.any(mask):
            raise ValueError(f"Requested class {class_id}/{class_name} has no samples.")

        cls_features = features[mask]
        cls_df = sample_lookup.loc[mask].copy().reset_index(drop=True)

        if cls_features.shape[0] < 2:
            raise ValueError(f"Class {class_id}/{class_name} has fewer than 2 samples; cannot visualize.")

        class_dir = per_class_root / f"label_{int(class_id):03d}_{sanitize_name(class_name)}"
        ensure_dir(class_dir)

        for method in methods:
            print(f"[per-class] class={class_name} ({class_id}) | projecting with {method} ...")
            coords = project_embeddings(cls_features, method, args)
            if coords.shape != (cls_features.shape[0], 2):
                raise ValueError(
                    f"Projection output for per-class method={method} must be [N, 2], but got {coords.shape}"
                )

            method_dir = class_dir / method
            figures_dir = method_dir / "figures"
            ensure_dir(figures_dir)

            save_projection_table(coords, cls_df, method_dir / "projection_points.csv")

            for color_col in color_bys:
                series = get_color_series(cls_df, color_col)
                mode = infer_color_mode(series, color_col)
                out_png = figures_dir / f"color_by_{sanitize_name(color_col)}.png"
                title = f"PER-CLASS | class={class_name} ({class_id}) | {method.upper()} | color_by={color_col}"

                if mode == "categorical":
                    scatter_categorical(
                        coords=coords,
                        df=cls_df,
                        color_col=color_col,
                        out_png=out_png,
                        title=title,
                        point_size=args.point_size,
                        alpha=args.alpha,
                        annotate_centroids=args.annotate_centroids,
                    )
                else:
                    scatter_continuous(
                        coords=coords,
                        df=cls_df,
                        color_col=color_col,
                        out_png=out_png,
                        title=title,
                        point_size=args.point_size,
                        alpha=args.alpha,
                    )


# ============================================================
# 5) 参数与主程序
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize extracted features together with global/local clustering results and sample metadata."
    )

    # 必需输入
    parser.add_argument("--features", required=True, help="Path to features (.npy or .pt), shape [N, D].")
    parser.add_argument(
        "--sample_lookup",
        required=True,
        help="Path to sample_lookup.csv or sample_lookup.jsonl from analyze_feature_clusters*.py.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save all visualization outputs.")

    # 可选：label_map，用于在 sample_lookup 缺 label_name 时补全
    parser.add_argument("--label_map_json", default=None, help="Optional label_map.json.")
    parser.add_argument(
        "--label_map_tier",
        default=None,
        choices=[None, "tier1", "tier2", "tier3"],
        help="Tier to use when label_map_json is nested."
    )

    # 特征预处理
    parser.add_argument(
        "--l2_normalize_features",
        action="store_true",
        help="Whether to L2-normalize features before projection. Useful for normalized contrastive features.",
    )

    # 投影方法
    parser.add_argument(
        "--global_methods",
        nargs="+",
        default=["pca", "umap"],
        choices=["pca", "umap", "tsne"],
        help="Projection methods for global visualization.",
    )
    parser.add_argument(
        "--per_class_methods",
        nargs="+",
        default=["umap"],
        choices=["pca", "umap", "tsne"],
        help="Projection methods for per-class rerun visualization.",
    )

    # UMAP 参数
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", type=str, default="euclidean")

    # t-SNE 参数
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)

    # 绘图外观
    parser.add_argument("--point_size", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument(
        "--annotate_centroids",
        action="store_true",
        help="Whether to annotate category centroids in categorical plots.",
    )

    # 可视化任务开关
    parser.add_argument("--run_global", action="store_true", help="Run global visualization.")
    parser.add_argument("--run_per_class", action="store_true", help="Run per-class rerun visualization.")

    # 若用户不显式指定，脚本会按推荐列自动挑选当前存在的列
    parser.add_argument(
        "--global_color_by",
        nargs="+",
        default=None,
        help=(
            "Columns to use for global color-by plots. If omitted, the script will use a recommended set from: "
            "label_name, global_kmeans_cluster, global_hdbscan_cluster, global_hdbscan_noise, "
            "global_hdbscan_probability, lighting, pos, person"
        ),
    )
    parser.add_argument(
        "--per_class_color_by",
        nargs="+",
        default=None,
        help=(
            "Columns to use for per-class color-by plots. If omitted, the script will use a recommended set from: "
            "local_kmeans_cluster, local_hdbscan_cluster, local_hdbscan_noise, local_hdbscan_probability, "
            "lighting, pos, person"
        ),
    )

    # 选择做哪些类别
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional subset of classes for per-class visualization. Each item can be label id or label_name.",
    )

    # 随机种子
    parser.add_argument("--random_state", type=int, default=0)

    args = parser.parse_args()

    # 若用户没有明确指定运行模式，则默认两者都开
    if not args.run_global and not args.run_per_class:
        args.run_global = True
        args.run_per_class = True

    return args



def main() -> None:
    args = parse_args()

    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    # 1) 保存配置
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 2) 加载输入
    features = ensure_features_2d(load_array_file(args.features, "features"))
    sample_lookup = load_sample_lookup(args.sample_lookup)
    id_to_name = load_label_map(args.label_map_json, tier=args.label_map_tier)
    sample_lookup = attach_label_names(sample_lookup, id_to_name)
    validate_alignment(features, sample_lookup)

    if args.l2_normalize_features:
        features = l2_normalize(features)

    # 3) 解析推荐默认字段
    global_defaults = [
        "label_name",
        "global_kmeans_cluster",
        "global_hdbscan_cluster",
        "global_hdbscan_noise",
        "global_hdbscan_probability",
        "lighting",
        "pos",
        "person",
    ]
    per_class_defaults = [
        "local_kmeans_cluster",
        "local_hdbscan_cluster",
        "local_hdbscan_noise",
        "local_hdbscan_probability",
        "lighting",
        "pos",
        "person",
    ]
    derived_cols = ["global_hdbscan_noise", "local_hdbscan_noise"]

    if args.run_global:
        global_color_bys = resolve_requested_columns(
            df=sample_lookup,
            requested=args.global_color_by,
            defaults=global_defaults,
            derived_supported=derived_cols,
            context="global",
        )
    else:
        global_color_bys = []

    if args.run_per_class:
        per_class_color_bys = resolve_requested_columns(
            df=sample_lookup,
            requested=args.per_class_color_by,
            defaults=per_class_defaults,
            derived_supported=derived_cols,
            context="per-class",
        )
    else:
        per_class_color_bys = []

    # 4) 保存一份总表，便于核对这次可视化实际使用了哪些字段
    viz_meta = {
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "global_methods": list(args.global_methods),
        "per_class_methods": list(args.per_class_methods),
        "run_global": bool(args.run_global),
        "run_per_class": bool(args.run_per_class),
        "global_color_by": global_color_bys,
        "per_class_color_by": per_class_color_bys,
        "classes_requested": list(args.classes) if args.classes is not None else None,
        "available_columns": list(sample_lookup.columns),
    }
    with open(out_root / "visualization_plan.json", "w", encoding="utf-8") as f:
        json.dump(viz_meta, f, ensure_ascii=False, indent=2)

    # 5) 全局可视化
    if args.run_global:
        run_global_visualization(
            features=features,
            sample_lookup=sample_lookup,
            methods=args.global_methods,
            color_bys=global_color_bys,
            out_root=out_root,
            args=args,
        )

    # 6) 类内可视化
    if args.run_per_class:
        selected_classes = resolve_class_subset(sample_lookup, args.classes)
        run_per_class_visualization(
            features=features,
            sample_lookup=sample_lookup,
            methods=args.per_class_methods,
            color_bys=per_class_color_bys,
            selected_classes=selected_classes,
            out_root=out_root,
            args=args,
        )

    print("[done] visualization outputs saved to:", str(out_root))


if __name__ == "__main__":
    main()
