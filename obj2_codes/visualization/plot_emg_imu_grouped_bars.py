#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据汇总文件绘制 4 张分组柱状图：

1) EMG + head_only
2) EMG + full
3) IMU + head_only
4) IMU + full

每张图：
- 横坐标：训练配置组合（method + prem + proto）
- 默认每个配置下 4 个柱子：
    - train mean ± std
    - val mean ± std
    - test mean ± std
    - test+val mean ± std
- 支持通过命令行参数自由选择绘制哪些柱子，顺序也可自定义
- 这些 mean/std 都是基于 4 个受试者统计得到
- 在每个柱子上标注 “均值±标准差”

输入 CSV 需要包含列：
modality, finetune_mode, method, prem, proto,
mean_train_acc, std_train_acc,
mean_val_acc, std_val_acc,
mean_test_acc, std_test_acc,
mean_test_val_acc, std_test_val_acc

默认输入：
    /mnt/data/emg_imu_config_mean_std_summary.csv

默认输出目录：
    /mnt/data/emg_imu_mean_std_grouped_bars_v3
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "modality", "finetune_mode", "method", "prem", "proto",
    "mean_train_acc", "std_train_acc",
    "mean_val_acc", "std_val_acc",
    "mean_test_acc", "std_test_acc",
    "mean_test_val_acc", "std_test_val_acc",
]

METHOD_ORDER = {
    "contrastive_only": 0,
    "contrastive_proto": 1,
    "contrastive_rel": 2,
    "contrastive_proto_rel": 3,
}

METHOD_SHORT = {
    "contrastive_only": "only",
    "contrastive_proto": "proto",
    "contrastive_rel": "rel",
    "contrastive_proto_rel": "proto+rel",
}

# 可选指标规格
METRIC_REGISTRY = {
    "train": ("mean_train_acc", "std_train_acc", "Train", "#4C78A8"),
    "val": ("mean_val_acc", "std_val_acc", "Val", "#F58518"),
    "test": ("mean_test_acc", "std_test_acc", "Test", "#54A24B"),
    "test+val": ("mean_test_val_acc", "std_test_val_acc", "Test+Val", "#B279A2"),
}

VALID_METRICS = list(METRIC_REGISTRY.keys())


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"输入 CSV 缺少必要列: {missing}")


def validate_selected_metrics(selected_metrics):
    if not selected_metrics:
        raise ValueError("至少需要选择一个指标。可选值：train val test test+val")
    bad = [m for m in selected_metrics if m not in METRIC_REGISTRY]
    if bad:
        raise ValueError(
            f"存在非法指标 {bad}。可选值仅为：{VALID_METRICS}"
        )


def build_config_label(row: pd.Series) -> str:
    method = METHOD_SHORT.get(str(row["method"]), str(row["method"]))
    prem = row["prem"]
    proto = row["proto"]
    return f"{method}\np={prem}, pr={proto}"


def sort_configs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_method_order"] = out["method"].map(METHOD_ORDER).fillna(999)
    out = out.sort_values(
        by=["modality", "finetune_mode", "_method_order", "prem", "proto"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    out.drop(columns=["_method_order"], inplace=True)
    return out


def annotate_bars(ax, bars, means, stds):
    for rect, mean_v, std_v in zip(bars, means, stds):
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_height()
        txt = f"{mean_v:.3f}\n±{std_v:.3f}"
        ax.text(
            x,
            y + 0.008,
            txt,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def metric_specs_from_selection(selected_metrics):
    return [METRIC_REGISTRY[m] for m in selected_metrics]


def plot_one_subset(
    df_subset: pd.DataFrame,
    modality: str,
    finetune_mode: str,
    out_dir: Path,
    selected_metrics,
    dpi: int = 220
) -> Path:
    if df_subset.empty:
        raise ValueError(f"空子集: {modality}, {finetune_mode}")

    metric_specs = metric_specs_from_selection(selected_metrics)

    dfp = df_subset.copy()
    dfp["config_label"] = dfp.apply(build_config_label, axis=1)

    n = len(dfp)
    x = np.arange(n)

    k = len(metric_specs)
    total_group_width = 0.80
    width = total_group_width / k
    offsets = np.linspace(
        -total_group_width / 2 + width / 2,
        total_group_width / 2 - width / 2,
        k
    )

    fig_width = max(16, n * 0.75 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 8.8))

    all_means = []
    all_stds = []

    for (mean_col, std_col, label, color), offset in zip(metric_specs, offsets):
        means = dfp[mean_col].astype(float).to_numpy()
        stds = dfp[std_col].astype(float).to_numpy()
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )
        annotate_bars(ax, bars, means, stds)
        all_means.append(means)
        all_stds.append(stds)

    all_means = np.concatenate(all_means)
    all_stds = np.concatenate(all_stds)
    ymax = min(1.05, max(0.80, float(np.max(all_means + all_stds)) + 0.10))

    selected_title = ", ".join(selected_metrics)

    ax.set_title(
        f"{modality.upper()} | {finetune_mode} | mean ± std across 4 subjects\nMetrics: {selected_title}",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Training configurations", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(dfp["config_label"].tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0.0, ymax)
    ax.legend(
        ncol=min(len(metric_specs), 4),
        frameon=True,
        fontsize=10,
        loc="upper center",
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # 用浅灰背景分隔不同 method 段，便于横向看 only/proto/rel/proto+rel
    method_blocks = []
    start_idx = 0
    labels = dfp["method"].tolist()
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[i - 1]:
            method_blocks.append((start_idx, i - 1, labels[i - 1]))
            start_idx = i

    shade = True
    for start, end, method_name in method_blocks:
        if shade:
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.05, color="gray")
        shade = not shade
        ax.text(
            (start + end) / 2.0,
            ymax - 0.03,
            METHOD_SHORT.get(method_name, method_name),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()

    safe_metrics = "_".join(
        m.replace("+", "_plus_").replace("/", "_") for m in selected_metrics
    )
    out_path = out_dir / f"{modality}_{finetune_mode}_{safe_metrics}_grouped_bar.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/mnt/data/emg_imu_config_mean_std_summary.csv")
    parser.add_argument("--out_dir", type=str, default="/mnt/data/emg_imu_mean_std_grouped_bars_v3")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["train", "val", "test", "test+val"],
        help="要绘制的指标，可选：train val test test+val。示例：--metrics test test+val"
    )
    args = parser.parse_args()

    selected_metrics = args.metrics
    validate_selected_metrics(selected_metrics)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_columns(df)
    df = sort_configs(df)

    saved = []
    for modality in ["emg", "imu"]:
        for finetune_mode in ["head_only", "full"]:
            sub = df[
                (df["modality"] == modality) &
                (df["finetune_mode"] == finetune_mode)
            ].copy()
            if sub.empty:
                continue
            saved.append(
                plot_one_subset(
                    sub,
                    modality,
                    finetune_mode,
                    out_dir,
                    selected_metrics=selected_metrics
                )
            )

    print("Saved figures:")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
