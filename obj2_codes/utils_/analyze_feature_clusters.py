#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_feature_clusters.py

输入一份已经提取好的特征与标签，并自动完成以下三类分析：

1) 全局分析（global）
   - 对整个数据集的所有特征一起做聚类分析
   - 输出 HDBSCAN 结果
   - 输出固定 K 扫描结果（KMeans + GMM-BIC）
   - 输出 silhouette / CH / DB / BIC / inertia / rank_sum 曲线图
   - 给出推荐最优 K，并与真实类别数对比

2) 类内独立分析（per-class local analysis）
   - 对每一个真实类别单独取出其样本特征
   - 分别做 HDBSCAN + K 扫描 + 最优 K 推荐
   - 观察某个真实类别内部是否还存在多个子簇

3) 基于全局聚类结果的逐类别分布分析（per-class on global clusters）
   - 先对全数据做全局聚类
   - 再统计每个真实类别在“全局推荐 K 的 KMeans 簇”中的分布
   - 以及在“HDBSCAN 全局簇”中的分布
   - 这一步不重新聚类，只分析真实类别在全局簇中的分布情况

此外，脚本还会构建“样本回溯表”：
- 读取 keys.txt（逐行与特征顺序一一对应）
- 读取 manifest.jsonl，并使用其中的 original_key 作为样本唯一键
- 用 key == original_key 进行严格匹配
- 输出 sample_lookup.csv / sample_lookup.jsonl，方便你从聚类结果快速反查原文件

注意：
- 本脚本是严格模式，不做静默跳过、不做兜底。
- 如果输入长度不一致、字段缺失、key 在 manifest 中找不到、某个类别样本数不足以完成你指定的 K 扫描，脚本会直接报错。
- HDBSCAN 依赖第三方包 hdbscan，请确保环境中已安装。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

try:
    import hdbscan
except ImportError as e:
    raise ImportError(
        "The package 'hdbscan' is required but not installed. "
        "Please install it first, e.g. 'pip install hdbscan'."
    ) from e


# =========================
# 基础加载与校验
# =========================

def load_array_file(path: str | Path, name: str) -> np.ndarray:
    """
    从 .npy 或 .pt 读取数组。

    支持：
    - .npy -> numpy.ndarray
    - .pt  -> torch.Tensor 或 numpy.ndarray

    不支持字典、列表等不明确对象；遇到这类输入会直接报错。
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
                "Expected a torch.Tensor or numpy.ndarray."
            )
    else:
        raise ValueError(f"Unsupported {name} file suffix: {suffix}. Only .npy and .pt are supported.")

    return np.asarray(arr)



def load_keys_txt(path: str | Path) -> List[str]:
    """逐行读取 key，保留顺序，并要求 key 唯一。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"keys.txt does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]

    if len(keys) == 0:
        raise ValueError(f"keys.txt is empty: {path}")
    if len(set(keys)) != len(keys):
        raise ValueError("keys.txt contains duplicate keys, but each key must be unique.")
    return keys



def load_manifest_jsonl(path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    读取 manifest.jsonl，并构造 original_key -> record 的严格映射。

    要求：
    - 每行都是合法 JSON object
    - 必须有 original_key 字段
    - original_key 不允许重复
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest file does not exist: {path}")

    records: List[Dict[str, Any]] = []
    by_original_key: Dict[str, Dict[str, Any]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at manifest line {line_idx}: {e}") from e

            if not isinstance(rec, dict):
                raise TypeError(f"Manifest line {line_idx} is not a JSON object.")
            if "original_key" not in rec:
                raise KeyError(f"Manifest line {line_idx} is missing required field 'original_key'.")

            original_key = rec["original_key"]
            if not isinstance(original_key, str) or original_key == "":
                raise ValueError(f"Manifest line {line_idx} has invalid 'original_key': {original_key!r}")
            if original_key in by_original_key:
                raise ValueError(f"Duplicate original_key found in manifest: {original_key}")

            records.append(rec)
            by_original_key[original_key] = rec

    if len(records) == 0:
        raise ValueError(f"Manifest is empty: {path}")

    return records, by_original_key



def load_label_map(path: Optional[str | Path], tier: Optional[str] = None) -> Dict[int, str]:
    """
    读取 label_map.json，并构造 int_label -> class_name 的映射。

    支持三种明确格式：

    1) 扁平格式：class_name -> int_id
       例如：
           {"take": 0, "put": 1}

    2) 扁平格式：int_id(str) -> class_name
       例如：
           {"0": "take", "1": "put"}

    3) 分层嵌套格式：
       例如：
           {
             "tier1": {"take": 0, "put": 1},
             "tier2": {...},
             "tier3": {...},
             "__meta__": {...}
           }

       此时必须指定 tier，例如 tier="tier1"。
    """
    if path is None:
        return {}

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label_map_json does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise TypeError("label_map_json must contain a JSON object.")
    if len(obj) == 0:
        raise ValueError("label_map_json is empty.")

    # -------------------------
    # 情况 1 / 2：扁平格式
    # -------------------------
    keys = list(obj.keys())
    values = list(obj.values())

    # 形式 1: class_name -> int_id
    if all(isinstance(k, str) for k in keys) and all(isinstance(v, int) for v in values):
        inv: Dict[int, str] = {}
        for class_name, label_id in obj.items():
            if label_id in inv:
                raise ValueError(f"Duplicate label id found in label_map_json: {label_id}")
            inv[int(label_id)] = str(class_name)
        return inv

    # 形式 2: "0" -> class_name
    if all(isinstance(k, str) for k in keys) and all(isinstance(v, str) for v in values):
        out: Dict[int, str] = {}
        try:
            for k, v in obj.items():
                kk = int(k)
                if kk in out:
                    raise ValueError(f"Duplicate label id found in label_map_json: {kk}")
                out[kk] = v
            return out
        except ValueError:
            pass

    # -------------------------
    # 情况 3：分层嵌套格式
    # -------------------------
    is_nested_tier_map = (
        any(k in obj for k in ("tier1", "tier2", "tier3"))
        and any(isinstance(v, dict) for v in obj.values())
    )

    if is_nested_tier_map:
        if tier is None:
            raise ValueError(
                "label_map_json is a nested tiered format, but no tier was provided. "
                "Please set --label_map_tier to one of: tier1, tier2, tier3."
            )
        if tier not in obj:
            raise KeyError(
                f"Requested tier '{tier}' is not found in label_map_json. "
                f"Available keys: {list(obj.keys())}"
            )

        tier_map = obj[tier]
        if not isinstance(tier_map, dict):
            raise TypeError(f"label_map_json[{tier!r}] must be a JSON object.")

        inv: Dict[int, str] = {}
        for class_name, label_id in tier_map.items():
            if not isinstance(class_name, str):
                raise TypeError(f"Class name in {tier} must be str, but got {type(class_name)}")
            if not isinstance(label_id, int):
                raise TypeError(
                    f"Label id for class '{class_name}' in {tier} must be int, but got {type(label_id)}"
                )
            if label_id in inv:
                raise ValueError(f"Duplicate label id found in label_map_json[{tier}]: {label_id}")
            inv[int(label_id)] = class_name

        return inv

    raise ValueError(
        "Unsupported label_map_json format. Supported formats are: "
        "{'class_name': 0, ...}, {'0': 'class_name', ...}, "
        "or nested tier format like {'tier1': {...}, 'tier2': {...}, 'tier3': {...}}."
    )



def ensure_2d_features(features: np.ndarray) -> np.ndarray:
    """要求特征必须是 [N, D] 二维矩阵。"""
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D [N, D], but got shape {features.shape}")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Features must be non-empty, but got shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or Inf.")
    return features.astype(np.float64, copy=False)



def ensure_1d_labels(labels: np.ndarray) -> np.ndarray:
    """要求标签是一维长度为 N 的向量。"""
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D [N], but got shape {labels.shape}")
    if labels.shape[0] == 0:
        raise ValueError("Labels are empty.")
    return labels



def build_sample_lookup_dataframe(
    features: np.ndarray,
    labels: np.ndarray,
    keys: Sequence[str],
    manifest_by_original_key: Dict[str, Dict[str, Any]],
    id_to_name: Dict[int, str],
) -> pd.DataFrame:
    """
    构建样本回溯表。

    严格要求：
    - len(features) == len(labels) == len(keys)
    - 每个 key 都必须能在 manifest 中用 original_key 精确找到
    """
    n = features.shape[0]
    if labels.shape[0] != n:
        raise ValueError(f"Feature count ({n}) != label count ({labels.shape[0]})")
    if len(keys) != n:
        raise ValueError(f"Feature count ({n}) != key count ({len(keys)})")

    rows: List[Dict[str, Any]] = []
    for idx, (key, label) in enumerate(zip(keys, labels)):
        if key not in manifest_by_original_key:
            raise KeyError(f"key '{key}' is not found in manifest original_key field.")
        rec = manifest_by_original_key[key]

        row: Dict[str, Any] = {
            "row_index": idx,
            "key": key,
            "label": label.item() if isinstance(label, np.generic) else label,
            "label_name": id_to_name.get(int(label), str(label)) if np.issubdtype(np.asarray(label).dtype, np.integer) else str(label),
        }

        # 这里直接展开 manifest 中的所有字段，方便后续完整回溯。
        # 若与已有列名冲突，则以 manifest_ 前缀写入，避免覆盖 row_index / key / label 等核心字段。
        for k, v in rec.items():
            out_key = k if k not in row else f"manifest_{k}"
            row[out_key] = v

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# 聚类指标与推荐规则
# =========================

def compute_ranks(values: Sequence[float], larger_is_better: bool) -> np.ndarray:
    """
    将一组指标值转换成排名，最优排名为 1。

    说明：
    - larger_is_better=True  时，值越大排名越靠前
    - larger_is_better=False 时，值越小排名越靠前
    - 输入中的每个 K 都唯一，因此这里直接用排序结果给出唯一排名
    """
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(-arr if larger_is_better else arr)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks



def summarize_cluster_labels(cluster_labels: np.ndarray) -> pd.DataFrame:
    """将一组聚类标签汇总成 counts / proportions 表。"""
    labels_unique, counts = np.unique(cluster_labels, return_counts=True)
    proportions = counts / counts.sum()
    return pd.DataFrame({
        "cluster_label": labels_unique,
        "count": counts,
        "proportion": proportions,
    }).sort_values("cluster_label").reset_index(drop=True)



def run_k_scan(
    X: np.ndarray,
    ks: Sequence[int],
    random_state: int,
    gmm_covariance_type: str,
    gmm_reg_covar: float,
) -> pd.DataFrame:
    """
    对给定特征矩阵执行固定 K 扫描。

    对每个 K：
    - 运行 KMeans，计算 inertia / silhouette / CH / DB
    - 运行 GMM，计算 BIC

    返回一个 DataFrame，每行对应一个 K。
    """
    rows: List[Dict[str, Any]] = []

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, km_labels, metric="euclidean")
        ch = calinski_harabasz_score(X, km_labels)
        db = davies_bouldin_score(X, km_labels)
        inertia = float(kmeans.inertia_)

        gmm = GaussianMixture(
            n_components=k,
            covariance_type=gmm_covariance_type,
            reg_covar=gmm_reg_covar,
            random_state=random_state,
        )
        gmm.fit(X)
        bic = float(gmm.bic(X))

        rows.append({
            "k": int(k),
            "silhouette": float(sil),
            "calinski_harabasz": float(ch),
            "davies_bouldin": float(db),
            "bic": float(bic),
            "inertia": inertia,
        })

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # 各指标的独立最优 K
    df["silhouette_rank"] = compute_ranks(df["silhouette"].to_numpy(), larger_is_better=True)
    df["calinski_harabasz_rank"] = compute_ranks(df["calinski_harabasz"].to_numpy(), larger_is_better=True)
    df["davies_bouldin_rank"] = compute_ranks(df["davies_bouldin"].to_numpy(), larger_is_better=False)
    df["bic_rank"] = compute_ranks(df["bic"].to_numpy(), larger_is_better=False)

    # 透明的多指标聚合：用 rank sum 作为总推荐依据
    # 这不是静默 heuristics，而是一个明确、可解释的共识排序规则：
    # 每个指标先给出排名，再求和；总排名越小越好。
    df["rank_sum"] = (
        df["silhouette_rank"]
        + df["calinski_harabasz_rank"]
        + df["davies_bouldin_rank"]
        + df["bic_rank"]
    )

    return df



def choose_recommended_k(k_scan_df: pd.DataFrame) -> int:
    """
    根据 rank_sum 选择推荐最优 K。

    规则：
    1) 先取 rank_sum 最小
    2) 若有并列，则依次比较：silhouette_rank、calinski_harabasz_rank、bic_rank、davies_bouldin_rank

    这样可以避免把某一个指标静默设为绝对主导，同时保持规则明确可解释。
    """
    sort_cols = [
        "rank_sum",
        "silhouette_rank",
        "calinski_harabasz_rank",
        "bic_rank",
        "davies_bouldin_rank",
        "k",
    ]
    best_row = k_scan_df.sort_values(sort_cols, ascending=[True, True, True, True, True, True]).iloc[0]
    return int(best_row["k"])



def fit_recommended_kmeans(X: np.ndarray, k: int, random_state: int) -> Tuple[np.ndarray, KMeans]:
    """在推荐 K 下重新拟合 KMeans。"""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, model



def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int],
    metric: str,
    cluster_selection_method: str,
) -> Tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """执行 HDBSCAN，并返回 labels / probabilities / clusterer。"""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(X)
    probabilities = clusterer.probabilities_
    return labels, probabilities, clusterer


# =========================
# 绘图函数
# =========================

def save_line_plot(
    x: Sequence[int],
    y: Sequence[float],
    title: str,
    ylabel: str,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    plt.figure(figsize=(7, 4.5))
    plt.plot(list(x), list(y), marker="o")
    plt.xlabel("K")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(list(x))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def save_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    save_path: str | Path,
    value_format: str = ".2f",
) -> None:
    """保存简单热图，用于展示类别在全局簇中的分布比例。"""
    save_path = Path(save_path)
    fig_w = max(8, 0.7 * len(col_labels) + 3)
    fig_h = max(4, 0.5 * len(row_labels) + 2)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Class")
    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], value_format), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# 分析主逻辑
# =========================

def sanitize_name(name: str) -> str:
    """将类名或分区名转成适合文件夹名称的形式。"""
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)



def analyze_partition(
    X: np.ndarray,
    partition_name: str,
    out_dir: str | Path,
    k_min: int,
    k_max: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: Optional[int],
    hdbscan_metric: str,
    hdbscan_cluster_selection_method: str,
    random_state: int,
    gmm_covariance_type: str,
    gmm_reg_covar: float,
    true_num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    对一个特征子集做完整分析，并将结果保存到 out_dir。

    返回：
    - k_scan_df
    - recommended_k
    - recommended_kmeans_labels
    - hdbscan_labels
    - hdbscan_probabilities
    - summary_dict
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples, n_features = X.shape
    if n_samples < 3:
        raise ValueError(f"Partition '{partition_name}' has only {n_samples} samples. At least 3 are required.")
    if k_min < 2:
        raise ValueError(f"k_min must be >= 2, but got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max must be >= k_min, but got k_min={k_min}, k_max={k_max}")
    if k_max > n_samples - 1:
        raise ValueError(
            f"Partition '{partition_name}' has {n_samples} samples, so k_max must be <= {n_samples - 1}, "
            f"but got {k_max}."
        )
    if hdbscan_min_cluster_size < 2:
        raise ValueError(f"hdbscan_min_cluster_size must be >= 2, but got {hdbscan_min_cluster_size}")
    if hdbscan_min_cluster_size > n_samples:
        raise ValueError(
            f"Partition '{partition_name}' has {n_samples} samples, but "
            f"hdbscan_min_cluster_size={hdbscan_min_cluster_size}."
        )
    if hdbscan_min_samples is not None and hdbscan_min_samples < 1:
        raise ValueError(f"hdbscan_min_samples must be >= 1 when provided, but got {hdbscan_min_samples}")

    ks = list(range(k_min, k_max + 1))

    # 1) 固定 K 扫描
    k_scan_df = run_k_scan(
        X=X,
        ks=ks,
        random_state=random_state,
        gmm_covariance_type=gmm_covariance_type,
        gmm_reg_covar=gmm_reg_covar,
    )
    k_scan_csv = out_dir / "k_scan.csv"
    k_scan_df.to_csv(k_scan_csv, index=False)

    # 2) 推荐 K
    recommended_k = choose_recommended_k(k_scan_df)
    km_labels, km_model = fit_recommended_kmeans(X, recommended_k, random_state)
    np.save(out_dir / "recommended_kmeans_labels.npy", km_labels)
    np.save(out_dir / "recommended_kmeans_centers.npy", km_model.cluster_centers_)
    summarize_cluster_labels(km_labels).to_csv(out_dir / "recommended_kmeans_cluster_summary.csv", index=False)

    # 3) HDBSCAN
    hdb_labels, hdb_probabilities, _ = run_hdbscan(
        X=X,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
    )
    np.save(out_dir / "hdbscan_labels.npy", hdb_labels)
    np.save(out_dir / "hdbscan_probabilities.npy", hdb_probabilities)
    summarize_cluster_labels(hdb_labels).to_csv(out_dir / "hdbscan_cluster_summary.csv", index=False)

    # 4) 统计 summary
    hdb_unique = set(int(v) for v in np.unique(hdb_labels).tolist())
    hdb_num_clusters = len([v for v in hdb_unique if v != -1])
    hdb_noise_fraction = float(np.mean(hdb_labels == -1))

    metric_best = {
        "silhouette_best_k": int(k_scan_df.loc[k_scan_df["silhouette"].idxmax(), "k"]),
        "calinski_harabasz_best_k": int(k_scan_df.loc[k_scan_df["calinski_harabasz"].idxmax(), "k"]),
        "davies_bouldin_best_k": int(k_scan_df.loc[k_scan_df["davies_bouldin"].idxmin(), "k"]),
        "bic_best_k": int(k_scan_df.loc[k_scan_df["bic"].idxmin(), "k"]),
    }

    summary = {
        "partition_name": partition_name,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "recommended_k": int(recommended_k),
        **metric_best,
        "hdbscan_num_clusters_excluding_noise": int(hdb_num_clusters),
        "hdbscan_noise_fraction": float(hdb_noise_fraction),
    }
    if true_num_classes is not None:
        summary["true_num_classes"] = int(true_num_classes)
        summary["recommended_k_minus_true_num_classes"] = int(recommended_k - true_num_classes)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 5) 绘图
    save_line_plot(k_scan_df["k"], k_scan_df["silhouette"], f"{partition_name} - Silhouette vs K", "Silhouette", out_dir / "silhouette_vs_k.png")
    save_line_plot(k_scan_df["k"], k_scan_df["calinski_harabasz"], f"{partition_name} - CH vs K", "Calinski-Harabasz", out_dir / "ch_vs_k.png")
    save_line_plot(k_scan_df["k"], k_scan_df["davies_bouldin"], f"{partition_name} - DB vs K", "Davies-Bouldin", out_dir / "db_vs_k.png")
    save_line_plot(k_scan_df["k"], k_scan_df["bic"], f"{partition_name} - GMM BIC vs K", "BIC", out_dir / "bic_vs_k.png")
    save_line_plot(k_scan_df["k"], k_scan_df["inertia"], f"{partition_name} - KMeans Inertia vs K", "Inertia", out_dir / "inertia_vs_k.png")
    save_line_plot(k_scan_df["k"], k_scan_df["rank_sum"], f"{partition_name} - Rank Sum vs K", "Rank Sum (lower is better)", out_dir / "rank_sum_vs_k.png")

    return {
        "k_scan_df": k_scan_df,
        "recommended_k": recommended_k,
        "recommended_kmeans_labels": km_labels,
        "hdbscan_labels": hdb_labels,
        "hdbscan_probabilities": hdb_probabilities,
        "summary": summary,
    }



def build_distribution_tables(
    class_labels: np.ndarray,
    cluster_labels: np.ndarray,
    label_to_name: Dict[Any, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    计算“真实类别 × 聚类簇” 的 counts / proportions 表。

    返回：
    - counts_df
    - proportions_df
    - proportions_matrix（用于画热图）
    - row_labels（类名）
    - col_labels（簇名）
    """
    unique_classes = np.unique(class_labels)
    unique_clusters = np.unique(cluster_labels)

    counts = np.zeros((len(unique_classes), len(unique_clusters)), dtype=np.int64)
    for i, cls in enumerate(unique_classes):
        cls_mask = class_labels == cls
        cls_clusters = cluster_labels[cls_mask]
        for j, clu in enumerate(unique_clusters):
            counts[i, j] = int(np.sum(cls_clusters == clu))

    proportions = counts.astype(np.float64) / counts.sum(axis=1, keepdims=True)

    row_labels = [label_to_name.get(cls.item() if isinstance(cls, np.generic) else cls, str(cls)) for cls in unique_classes]
    col_labels = [str(c.item() if isinstance(c, np.generic) else c) for c in unique_clusters]

    counts_df = pd.DataFrame(counts, index=row_labels, columns=col_labels)
    counts_df.index.name = "class_name"

    proportions_df = pd.DataFrame(proportions, index=row_labels, columns=col_labels)
    proportions_df.index.name = "class_name"

    return counts_df, proportions_df, proportions, row_labels, col_labels


# =========================
# 主程序
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze extracted features with global clustering, per-class local clustering, and per-class distribution on global clusters."
    )

    # 必要输入
    parser.add_argument("--features", required=True, help="Path to feature file (.npy or .pt), shape [N, D].")
    parser.add_argument("--labels", required=True, help="Path to label file (.npy or .pt), shape [N].")
    parser.add_argument("--keys", required=True, help="Path to keys.txt; each line corresponds to one feature row.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl; original_key must equal key.")
    parser.add_argument("--output_dir", required=True, help="Directory to save all outputs.")
    parser.add_argument(
    "--label_map_tier",
    type=str,
    default="tier1",
    choices=["tier1", "tier2", "tier3"],
    help="Which tier to use from a nested label_map.json."
)

    # 可选：类名映射
    parser.add_argument("--label_map_json", default=None, help="Optional label map json for readable class names.")

    # 全局 K 扫描
    parser.add_argument("--global_k_min", type=int, default=2)
    parser.add_argument("--global_k_max", type=int, default=15)

    # 类内 K 扫描
    parser.add_argument("--local_k_min", type=int, default=2)
    parser.add_argument("--local_k_max", type=int, default=10)

    # HDBSCAN（全局）
    parser.add_argument("--hdbscan_min_cluster_size_global", type=int, default=5)
    parser.add_argument("--hdbscan_min_samples_global", type=int, default=None)

    # HDBSCAN（类内）
    parser.add_argument("--hdbscan_min_cluster_size_local", type=int, default=5)
    parser.add_argument("--hdbscan_min_samples_local", type=int, default=None)

    # HDBSCAN 共同设置
    parser.add_argument("--hdbscan_metric", type=str, default="euclidean")
    parser.add_argument("--hdbscan_cluster_selection_method", type=str, default="eom", choices=["eom", "leaf"])

    # GMM 设置
    parser.add_argument("--gmm_covariance_type", type=str, default="full", choices=["full", "tied", "diag", "spherical"])
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-6)

    # 随机种子
    parser.add_argument("--random_state", type=int, default=0)

    return parser.parse_args()



def main() -> None:
    args = parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 保存配置，便于完全复现
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 1) 加载输入
    features = ensure_2d_features(load_array_file(args.features, "features"))
    labels = ensure_1d_labels(load_array_file(args.labels, "labels"))
    keys = load_keys_txt(args.keys)
    _, manifest_by_original_key = load_manifest_jsonl(args.manifest)
    id_to_name = load_label_map(args.label_map_json, tier=args.label_map_tier)
    
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of feature rows ({features.shape[0]}) != number of labels ({labels.shape[0]}).")
    if features.shape[0] != len(keys):
        raise ValueError(f"Number of feature rows ({features.shape[0]}) != number of keys ({len(keys)}).")

    # 2) 构建样本回溯表（基础列）
    sample_lookup_df = build_sample_lookup_dataframe(
        features=features,
        labels=labels,
        keys=keys,
        manifest_by_original_key=manifest_by_original_key,
        id_to_name=id_to_name,
    )
    sample_lookup_df.to_csv(out_root / "sample_lookup_base.csv", index=False)
    sample_lookup_df.to_json(out_root / "sample_lookup_base.jsonl", orient="records", force_ascii=False, lines=True)

    unique_labels = np.unique(labels)
    true_num_classes = len(unique_labels)
    label_to_name_runtime: Dict[Any, str] = {}
    for lab in unique_labels:
        lab_key = lab.item() if isinstance(lab, np.generic) else lab
        if isinstance(lab_key, (int, np.integer)):
            label_to_name_runtime[lab_key] = id_to_name.get(int(lab_key), str(int(lab_key)))
        else:
            label_to_name_runtime[lab_key] = str(lab_key)

    # 3) 全局分析
    global_dir = out_root / "global"
    global_result = analyze_partition(
        X=features,
        partition_name="global",
        out_dir=global_dir,
        k_min=args.global_k_min,
        k_max=args.global_k_max,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size_global,
        hdbscan_min_samples=args.hdbscan_min_samples_global,
        hdbscan_metric=args.hdbscan_metric,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        random_state=args.random_state,
        gmm_covariance_type=args.gmm_covariance_type,
        gmm_reg_covar=args.gmm_reg_covar,
        true_num_classes=true_num_classes,
    )

    # 将全局结果写入样本回溯表
    sample_lookup_df["global_kmeans_recommended_k"] = int(global_result["recommended_k"])
    sample_lookup_df["global_kmeans_cluster"] = global_result["recommended_kmeans_labels"]
    sample_lookup_df["global_hdbscan_cluster"] = global_result["hdbscan_labels"]
    sample_lookup_df["global_hdbscan_probability"] = global_result["hdbscan_probabilities"]

    # 4) 类内独立分析
    per_class_local_dir = out_root / "per_class_local"
    per_class_local_dir.mkdir(parents=True, exist_ok=True)

    local_summary_rows: List[Dict[str, Any]] = []

    # 预先在样本回溯表中创建列，后续逐类填充
    sample_lookup_df["local_recommended_k"] = pd.Series([pd.NA] * len(sample_lookup_df), dtype="Int64")
    sample_lookup_df["local_kmeans_cluster"] = pd.Series([pd.NA] * len(sample_lookup_df), dtype="Int64")
    sample_lookup_df["local_hdbscan_cluster"] = pd.Series([pd.NA] * len(sample_lookup_df), dtype="Int64")
    sample_lookup_df["local_hdbscan_probability"] = np.nan

    for raw_label in unique_labels:
        label_value = raw_label.item() if isinstance(raw_label, np.generic) else raw_label
        class_mask = labels == raw_label
        X_cls = features[class_mask]
        cls_name = label_to_name_runtime.get(label_value, str(label_value))
        cls_dir_name = f"label_{label_value}_{sanitize_name(cls_name)}"
        cls_out_dir = per_class_local_dir / cls_dir_name

        cls_result = analyze_partition(
            X=X_cls,
            partition_name=f"class_{cls_name}",
            out_dir=cls_out_dir,
            k_min=args.local_k_min,
            k_max=args.local_k_max,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size_local,
            hdbscan_min_samples=args.hdbscan_min_samples_local,
            hdbscan_metric=args.hdbscan_metric,
            hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
            random_state=args.random_state,
            gmm_covariance_type=args.gmm_covariance_type,
            gmm_reg_covar=args.gmm_reg_covar,
            true_num_classes=None,
        )

        row = {
            "label": label_value,
            "class_name": cls_name,
            **cls_result["summary"],
        }
        local_summary_rows.append(row)

        row_indices = sample_lookup_df.index[class_mask]
        if len(row_indices) != X_cls.shape[0]:
            raise RuntimeError("Internal error: class_mask and row_indices length mismatch.")

        sample_lookup_df.loc[row_indices, "local_recommended_k"] = int(cls_result["recommended_k"])
        sample_lookup_df.loc[row_indices, "local_kmeans_cluster"] = cls_result["recommended_kmeans_labels"]
        sample_lookup_df.loc[row_indices, "local_hdbscan_cluster"] = cls_result["hdbscan_labels"]
        sample_lookup_df.loc[row_indices, "local_hdbscan_probability"] = cls_result["hdbscan_probabilities"]

    per_class_summary_df = pd.DataFrame(local_summary_rows).sort_values(["label", "class_name"]).reset_index(drop=True)
    per_class_summary_df.to_csv(per_class_local_dir / "per_class_local_summary.csv", index=False)

    # 5) 基于全局聚类结果的逐类别分布分析
    per_class_global_dir = out_root / "per_class_on_global_clusters"
    per_class_global_dir.mkdir(parents=True, exist_ok=True)

    # 5.1 对全局推荐 KMeans 簇做分布统计
    km_counts_df, km_props_df, km_props_matrix, km_row_labels, km_col_labels = build_distribution_tables(
        class_labels=labels,
        cluster_labels=global_result["recommended_kmeans_labels"],
        label_to_name=label_to_name_runtime,
    )
    km_counts_df.to_csv(per_class_global_dir / "global_kmeans_counts.csv")
    km_props_df.to_csv(per_class_global_dir / "global_kmeans_proportions.csv")
    save_heatmap(
        matrix=km_props_matrix,
        row_labels=km_row_labels,
        col_labels=km_col_labels,
        title="Per-class distribution on global recommended KMeans clusters",
        save_path=per_class_global_dir / "global_kmeans_proportions_heatmap.png",
        value_format=".2f",
    )

    # 5.2 对全局 HDBSCAN 簇做分布统计
    hdb_counts_df, hdb_props_df, hdb_props_matrix, hdb_row_labels, hdb_col_labels = build_distribution_tables(
        class_labels=labels,
        cluster_labels=global_result["hdbscan_labels"],
        label_to_name=label_to_name_runtime,
    )
    hdb_counts_df.to_csv(per_class_global_dir / "global_hdbscan_counts.csv")
    hdb_props_df.to_csv(per_class_global_dir / "global_hdbscan_proportions.csv")
    save_heatmap(
        matrix=hdb_props_matrix,
        row_labels=hdb_row_labels,
        col_labels=hdb_col_labels,
        title="Per-class distribution on global HDBSCAN clusters",
        save_path=per_class_global_dir / "global_hdbscan_proportions_heatmap.png",
        value_format=".2f",
    )

    # 6) 输出最终样本回溯表
    sample_lookup_df.to_csv(out_root / "sample_lookup.csv", index=False)
    sample_lookup_df.to_json(out_root / "sample_lookup.jsonl", orient="records", force_ascii=False, lines=True)

    # 7) 输出总览 summary
    overall_summary = {
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "true_num_classes": int(true_num_classes),
        "global_recommended_k": int(global_result["recommended_k"]),
        "global_hdbscan_num_clusters_excluding_noise": int(global_result["summary"]["hdbscan_num_clusters_excluding_noise"]),
        "global_hdbscan_noise_fraction": float(global_result["summary"]["hdbscan_noise_fraction"]),
        "global_silhouette_best_k": int(global_result["summary"]["silhouette_best_k"]),
        "global_calinski_harabasz_best_k": int(global_result["summary"]["calinski_harabasz_best_k"]),
        "global_davies_bouldin_best_k": int(global_result["summary"]["davies_bouldin_best_k"]),
        "global_bic_best_k": int(global_result["summary"]["bic_best_k"]),
    }
    with open(out_root / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print("Analysis finished successfully.")
    print(f"Output directory: {out_root}")
    print(f"Global recommended K: {global_result['recommended_k']}")
    print(f"True number of classes: {true_num_classes}")


if __name__ == "__main__":
    main()
