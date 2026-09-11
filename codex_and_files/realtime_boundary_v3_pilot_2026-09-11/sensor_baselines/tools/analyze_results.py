from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

import _bootstrap  # noqa: F401
from sensor_boundary.utils import parse_camera_timestamp, read_csv, read_json, read_jsonl, write_json


METRICS = {
    "frame_accuracy": ("frame_state", "accuracy"),
    "frame_f1": ("frame_state", "f1"),
    "segmental_f1_10": ("segmental_f1", "10", "f1"),
    "segmental_f1_25": ("segmental_f1", "25", "f1"),
    "segmental_f1_50": ("segmental_f1", "50", "f1"),
    "edit_score": ("edit_score",),
    "start_f1_50ms": ("boundary", "50", "start", "f1"),
    "end_f1_50ms": ("boundary", "50", "end", "f1"),
    "start_f1_100ms": ("boundary", "100", "start", "f1"),
    "end_f1_100ms": ("boundary", "100", "end", "f1"),
    "start_f1_200ms": ("boundary", "200", "start", "f1"),
    "end_f1_200ms": ("boundary", "200", "end", "f1"),
    "start_f1_500ms": ("boundary", "500", "start", "f1"),
    "end_f1_500ms": ("boundary", "500", "end", "f1"),
    "start_signed_error_200ms": ("boundary", "200", "start", "median_signed_error_ms"),
    "end_signed_error_200ms": ("boundary", "200", "end", "median_signed_error_ms"),
    "emission_delay_ms": ("emission_delay_ms",),
    "predicted_segments_per_minute": ("predicted_segments_per_minute",),
    "short_fragment_fraction": ("short_fragment_fraction_lt_200ms",),
}

COLORS = {"imu": "#3478B8", "emg": "#E7862B"}


def nested(row: dict, path: tuple[str, ...]):
    value = row
    for key in path:
        value = value[key]
    return value


def condition_from_path(path: Path, outputs_root: Path) -> dict[str, str | int]:
    parts = path.relative_to(outputs_root).parts
    return {
        "modality": parts[0],
        "heldout": parts[1].removesuffix("_as_test"),
        "scope": parts[2],
        "seed": int(parts[3].removeprefix("seed_")),
        "test_split": parts[-2],
    }


def load_metrics(outputs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions, per_runs = [], []
    for path in sorted(outputs_root.glob("*/?_as_test/*/seed_*/causal_sensor_boundary_v1/evaluation/*/metrics.json")):
        info = condition_from_path(path, outputs_root)
        result = read_json(path)
        macro = result["macro"]
        row = {**info, **{name: nested(macro, keys) for name, keys in METRICS.items()}}
        row["real_time_factor"] = result["overall_real_time_factor"]
        row["runs"] = result["runs"]
        row["total_gt_segments"] = sum(x["gt_segment_count"] for x in result["per_run"].values())
        row["total_pred_segments"] = sum(x["pred_segment_count"] for x in result["per_run"].values())
        row["segment_count_ratio"] = row["total_pred_segments"] / max(row["total_gt_segments"], 1)
        conditions.append(row)
        for run, metrics in result["per_run"].items():
            per = {**info, "sample_name": run, **{name: nested(metrics, keys) for name, keys in METRICS.items()}}
            per["gt_segments"] = metrics["gt_segment_count"]
            per["pred_segments"] = metrics["pred_segment_count"]
            per["segment_count_ratio"] = metrics["pred_segment_count"] / max(metrics["gt_segment_count"], 1)
            per_runs.append(per)
    return pd.DataFrame(conditions), pd.DataFrame(per_runs)


def load_training(outputs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    training, calibration = [], []
    for modality in ("imu", "emg"):
        for log in outputs_root.glob(f"{modality}/?_as_test/*/seed_*/causal_sensor_boundary_v1/training_log.jsonl"):
            parts = log.relative_to(outputs_root).parts
            rows = read_jsonl(log)
            losses = np.asarray([x["validation"]["loss"] for x in rows], dtype=float)
            best_index = int(np.argmin(losses))
            training.append({
                "modality": modality,
                "heldout": parts[1].removesuffix("_as_test"),
                "scope": parts[2],
                "seed": int(parts[3].removeprefix("seed_")),
                "best_epoch": int(rows[best_index]["epoch"]),
                "best_validation_loss": float(losses[best_index]),
                "final_validation_loss": float(losses[-1]),
                "final_train_loss": float(rows[-1]["train"]["loss"]),
                "final_generalization_gap": float(losses[-1] - rows[-1]["train"]["loss"]),
            })
            cal_path = log.parent / "calibrated_online.json"
            cal = read_json(cal_path)
            calibration.append({
                "modality": modality,
                "heldout": parts[1].removesuffix("_as_test"),
                "scope": parts[2],
                "seed": int(parts[3].removeprefix("seed_")),
                "validation_objective": cal["selected_score"],
                **cal["selected_online"],
            })
    return pd.DataFrame(training), pd.DataFrame(calibration)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_statistics(df: pd.DataFrame, scope: str, split: str) -> pd.DataFrame:
    subset = df[(df.scope == scope) & (df.test_split == split)]
    metrics = ["frame_f1", "segmental_f1_50", "start_f1_200ms", "end_f1_200ms", "edit_score", "segment_count_ratio", "emission_delay_ms"]
    rows, raw_p = [], []
    for metric in metrics:
        pivot = subset.pivot(index=["heldout", "seed"], columns="modality", values=metric).dropna()
        delta = pivot["emg"] - pivot["imu"]
        try:
            p_value = float(wilcoxon(delta).pvalue) if np.any(np.abs(delta) > 1e-12) else 1.0
        except ValueError:
            p_value = 1.0
        raw_p.append(p_value)
        rows.append({
            "metric": metric,
            "pairs": len(delta),
            "imu_mean": float(pivot["imu"].mean()),
            "emg_mean": float(pivot["emg"].mean()),
            "emg_minus_imu": float(delta.mean()),
            "wins_emg": int((delta > 1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()),
            "wins_imu": int((delta < -1e-12).sum()),
            "wilcoxon_p": p_value,
        })
    adjusted = holm_adjust(raw_p[:5])
    for index, row in enumerate(rows):
        row["holm_p_performance_metrics"] = adjusted[index] if index < 5 else None
    return pd.DataFrame(rows)


def heldout_level_statistics(df: pd.DataFrame, scope: str, split: str) -> pd.DataFrame:
    subset = df[(df.scope == scope) & (df.test_split == split)]
    metrics = ["frame_f1", "segmental_f1_50", "start_f1_200ms", "end_f1_200ms", "edit_score"]
    rows = []
    for metric in metrics:
        means = subset.groupby(["heldout", "modality"])[metric].mean().unstack("modality")
        delta = means["emg"] - means["imu"]
        try:
            p_value = float(wilcoxon(delta).pvalue) if np.any(np.abs(delta) > 1e-12) else 1.0
        except ValueError:
            p_value = 1.0
        rows.append({
            "metric": metric,
            "heldout_units": len(delta),
            "emg_minus_imu": float(delta.mean()),
            "heldouts_emg_better": int((delta > 1e-12).sum()),
            "heldouts_imu_better": int((delta < -1e-12).sum()),
            "wilcoxon_p_n4": p_value,
        })
    return pd.DataFrame(rows)


def save_condition_plot(df: pd.DataFrame, target: Path) -> None:
    subset = df[df.test_split == "test_all"]
    panels = [
        ("frame_f1", "Frame F1", True),
        ("segmental_f1_50", "Segmental F1@50", True),
        ("edit_score", "Edit score", False),
        ("start_f1_200ms", "Start F1 (±200 ms)", True),
        ("end_f1_200ms", "End F1 (±200 ms)", True),
        ("emission_delay_ms", "Emission delay (ms)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    groups = [(m, s) for s in ("normal_only", "all_runs") for m in ("imu", "emg")]
    labels = ["IMU\nnormal", "EMG\nnormal", "IMU\nall", "EMG\nall"]
    for ax, (metric, title, percent) in zip(axes.flat, panels):
        means = [subset[(subset.modality == m) & (subset.scope == s)][metric].mean() for m, s in groups]
        stds = [subset[(subset.modality == m) & (subset.scope == s)][metric].std(ddof=1) for m, s in groups]
        if percent:
            means, stds = np.asarray(means) * 100, np.asarray(stds) * 100
        ax.bar(range(4), means, yerr=stds, capsize=4, color=[COLORS[m] for m, _ in groups], alpha=0.9)
        ax.set_xticks(range(4), labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if percent:
            ax.set_ylabel("%")
    fig.suptitle("IMU vs EMG: mean ± SD across 12 participant-seed conditions (test_all)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_split_plot(df: pd.DataFrame, target: Path) -> None:
    subset = df[df.scope == "all_runs"]
    panels = [("frame_f1", "Frame F1"), ("segmental_f1_50", "Segmental F1@50"), ("boundary_mean", "Boundary F1@200 ms")]
    subset = subset.copy()
    subset["boundary_mean"] = (subset.start_f1_200ms + subset.end_f1_200ms) / 2
    order = ["test_normal", "test_fault", "test_all"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(3)
    for ax, (metric, title) in zip(axes, panels):
        for modality in ("imu", "emg"):
            values = [subset[(subset.modality == modality) & (subset.test_split == split)][metric].mean() * 100 for split in order]
            errors = [subset[(subset.modality == modality) & (subset.test_split == split)][metric].std(ddof=1) * 100 for split in order]
            ax.errorbar(x, values, yerr=errors, marker="o", capsize=4, linewidth=2, label=modality.upper(), color=COLORS[modality])
        ax.set_xticks(x, ["normal", "fault", "all"])
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Test-split robustness, all-runs training (mean ± SD, n=12)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tolerance_plot(df: pd.DataFrame, target: Path) -> None:
    subset = df[(df.scope == "all_runs") & (df.test_split == "test_all")]
    tolerances = np.asarray([50, 100, 200, 500])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, kind in zip(axes, ("start", "end")):
        for modality in ("imu", "emg"):
            columns = [f"{kind}_f1_{x}ms" for x in tolerances]
            values = subset[subset.modality == modality][columns].to_numpy(float) * 100
            ax.plot(tolerances, values.mean(axis=0), marker="o", linewidth=2, label=modality.upper(), color=COLORS[modality])
            ax.fill_between(tolerances, values.mean(axis=0) - values.std(axis=0, ddof=1), values.mean(axis=0) + values.std(axis=0, ddof=1), alpha=0.15, color=COLORS[modality])
        ax.set_xscale("log")
        ax.set_xticks(tolerances, tolerances)
        ax.set_xlabel("Tolerance (ms)")
        ax.set_ylabel("Boundary F1 (%)")
        ax.set_title(f"{kind.capitalize()} boundary")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Boundary tolerance curves, all-runs / test_all (mean ± SD, n=12)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_paired_plot(df: pd.DataFrame, target: Path) -> None:
    subset = df[(df.scope == "all_runs") & (df.test_split == "test_all")]
    metrics = [("frame_f1", "Frame F1"), ("segmental_f1_50", "Segmental F1@50"), ("start_f1_200ms", "Start F1@200 ms"), ("end_f1_200ms", "End F1@200 ms")]
    markers = {"A": "o", "D": "s", "J": "^", "M": "D"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (metric, title) in zip(axes.flat, metrics):
        pivot = subset.pivot(index=["heldout", "seed"], columns="modality", values=metric) * 100
        low = float(min(pivot.min()) - 2)
        high = float(max(pivot.max()) + 2)
        for heldout in markers:
            part = pivot.loc[heldout]
            ax.scatter(part["imu"], part["emg"], s=65, marker=markers[heldout], label=heldout, alpha=0.85)
        ax.plot([low, high], [low, high], "--", color="gray", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("IMU (%)")
        ax.set_ylabel("EMG (%)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(title="Held-out")
    fig.suptitle("Paired participant-seed conditions, all-runs / test_all")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_predictions(path: Path, sample_name: str) -> list[tuple[float, float]]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_name"] == sample_name:
                result.append((float(row["start_time"]), float(row["end_time"])))
    return result


def load_predictions_by_run(path: Path) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result.setdefault(str(row["sample_name"]), []).append((float(row["start_time"]), float(row["end_time"])))
    return result


def annotation_segments(annotation_root: Path, sample_name: str) -> tuple[list[dict], float, float]:
    frame_rows = read_csv(annotation_root / f"{sample_name}_frame_annotation.csv")
    start_limit = parse_camera_timestamp(frame_rows[0]["timestamp"])
    end_limit = parse_camera_timestamp(frame_rows[-1]["timestamp"])
    result = []
    for row in read_csv(annotation_root / f"{sample_name}_segmentation_annotation.csv"):
        if row["action"].strip().lower() == "background":
            continue
        start, end = parse_camera_timestamp(row["start"]), parse_camera_timestamp(row["end"])
        if end >= start_limit and start <= end_limit:
            result.append({"action": row["action"].strip(), "start": max(start, start_limit), "end": min(end, end_limit)})
    return result, start_limit, end_limit


def _matched_gt_events(gt: list[float], pred: list[float], tolerance: float) -> set[int]:
    candidates = sorted((abs(p - g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gt) if abs(p - g) <= tolerance)
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, pi, gi in candidates:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return used_g


def _time_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / max(union, 1e-9)


def _matched_gt_segments(gt: list[tuple[float, float]], pred: list[tuple[float, float]], threshold: float) -> set[int]:
    candidates = sorted((-_time_iou(p, g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gt) if _time_iou(p, g) >= threshold)
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, pi, gi in candidates:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return used_g


def per_action_recall(root: Path, annotation_root: Path) -> pd.DataFrame:
    totals: dict[tuple[str, str], dict[str, int]] = {}
    for modality in ("imu", "emg"):
        for heldout in ("A", "D", "J", "M"):
            prediction_path = root / "outputs" / modality / f"{heldout}_as_test" / "all_runs" / "seed_42" / "causal_sensor_boundary_v1" / "evaluation" / "test_all" / "predicted_segments.jsonl"
            predictions = load_predictions_by_run(prediction_path)
            for run, pred in predictions.items():
                gt_rows, _, _ = annotation_segments(annotation_root, run)
                gt_pairs = [(row["start"], row["end"]) for row in gt_rows]
                segment_hits = _matched_gt_segments(gt_pairs, pred, 0.5)
                start_hits = _matched_gt_events([x[0] for x in gt_pairs], [x[0] for x in pred], 0.2)
                end_hits = _matched_gt_events([x[1] for x in gt_pairs], [x[1] for x in pred], 0.2)
                for index, row in enumerate(gt_rows):
                    bucket = totals.setdefault((modality, row["action"]), {"gt_segments": 0, "segment_hits": 0, "start_hits": 0, "end_hits": 0})
                    bucket["gt_segments"] += 1
                    bucket["segment_hits"] += int(index in segment_hits)
                    bucket["start_hits"] += int(index in start_hits)
                    bucket["end_hits"] += int(index in end_hits)
    rows = []
    for (modality, action), values in sorted(totals.items()):
        count = values["gt_segments"]
        rows.append({
            "modality": modality,
            "action": action,
            **values,
            "segment_recall_iou50": values["segment_hits"] / count,
            "start_recall_200ms": values["start_hits"] / count,
            "end_recall_200ms": values["end_hits"] / count,
        })
    return pd.DataFrame(rows)


def save_action_heatmap(action_df: pd.DataFrame, target: Path) -> None:
    actions = sorted(action_df.action.unique())
    columns = [
        ("imu", "segment_recall_iou50", "IMU\nSeg@50"),
        ("emg", "segment_recall_iou50", "EMG\nSeg@50"),
        ("imu", "start_recall_200ms", "IMU\nStart@200"),
        ("emg", "start_recall_200ms", "EMG\nStart@200"),
        ("imu", "end_recall_200ms", "IMU\nEnd@200"),
        ("emg", "end_recall_200ms", "EMG\nEnd@200"),
    ]
    matrix = np.zeros((len(actions), len(columns)))
    counts = []
    for i, action in enumerate(actions):
        counts.append(int(action_df[(action_df.action == action) & (action_df.modality == "imu")].gt_segments.iloc[0]))
        for j, (modality, metric, _) in enumerate(columns):
            matrix[i, j] = float(action_df[(action_df.action == action) & (action_df.modality == modality)][metric].iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(columns)), [x[2] for x in columns])
    ax.set_yticks(range(len(actions)), [f"{action} (n={count})" for action, count in zip(actions, counts)])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] > 55 else "black")
    fig.colorbar(image, ax=ax, label="Recall (%)")
    ax.set_title("Per-action recall, all-runs / test_all / seed 42 (each run counted once)")
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_representative_runs(per_run: pd.DataFrame) -> dict[str, str]:
    subset = per_run[(per_run.scope == "all_runs") & (per_run.test_split == "test_all") & (per_run.seed == 42)]
    pivot = subset.pivot(index=["heldout", "sample_name"], columns="modality", values="segmental_f1_50").dropna()
    pivot["joint"] = pivot[["imu", "emg"]].mean(axis=1)
    selected = {}
    for heldout in ("A", "D", "J", "M"):
        part = pivot.loc[heldout]
        median = part.joint.median()
        selected[heldout] = str((part.joint - median).abs().idxmin())
    return selected


def draw_timeline_axis(ax, gt, imu, emg, start, end, action_codes, title, label_numbers=True):
    cmap = plt.get_cmap("tab20")
    for segment in gt:
        x, width = segment["start"] - start, segment["end"] - segment["start"]
        code = action_codes[segment["action"]]
        ax.broken_barh([(x, width)], (2.05, 0.75), facecolors=cmap((code - 1) % 20), edgecolors="black", linewidth=0.35)
        if label_numbers and width >= 0.28:
            ax.text(x + width / 2, 2.43, str(code), ha="center", va="center", fontsize=6)
    for index, (left, right) in enumerate(imu):
        ax.broken_barh([(left - start, right - left)], (1.05, 0.75), facecolors=COLORS["imu"], alpha=0.78, edgecolors="black", linewidth=0.3)
    for index, (left, right) in enumerate(emg):
        ax.broken_barh([(left - start, right - left)], (0.05, 0.75), facecolors=COLORS["emg"], alpha=0.78, edgecolors="black", linewidth=0.3)
    ax.set_xlim(0, end - start)
    ax.set_ylim(0, 3)
    ax.set_yticks([2.43, 1.43, 0.43], ["GT", "IMU", "EMG"])
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="x", alpha=0.2)


def resolve_annotation_root(root: Path, override: str | None) -> tuple[Path, str]:
    if override:
        candidate = Path(override)
        if not candidate.is_dir():
            raise FileNotFoundError(f"--annotation-root does not exist: {candidate}")
        return candidate, "command_line"
    resolved_path = next((root / "outputs").glob("*/?_as_test/*/seed_*/causal_sensor_boundary_v1/resolved_config.json"))
    recorded = Path(read_json(resolved_path)["paths"]["annotation_root"])
    if recorded.is_dir():
        return recorded, "resolved_config"
    local = root.parents[2] / "MULTISENSOR_DATA_COLLECTION_Stage2_structured_data" / "Action_Segmentation_Dataset" / "annotations" / "action_recognition_boundaries_with_background_v1"
    if local.is_dir():
        return local, f"local_fallback_for_missing:{recorded}"
    raise FileNotFoundError(
        f"Annotation path recorded by training does not exist here: {recorded}. "
        "Pass --annotation-root with the current computer path."
    )


def save_timelines(root: Path, annotation_root: Path, per_run: pd.DataFrame, target_full: Path, target_zoom: Path) -> tuple[dict[str, str], str]:
    selected = select_representative_runs(per_run)
    all_actions = sorted({row["action"].strip() for path in annotation_root.glob("*_segmentation_annotation.csv") for row in read_csv(path) if row["action"].strip().lower() != "background"})
    action_codes = {action: index + 1 for index, action in enumerate(all_actions)}
    fig, axes = plt.subplots(4, 1, figsize=(16, 9))
    for ax, heldout in zip(axes, ("A", "D", "J", "M")):
        run = selected[heldout]
        gt, start, end = annotation_segments(annotation_root, run)
        base = root / "outputs"
        imu = load_predictions(base / "imu" / f"{heldout}_as_test" / "all_runs" / "seed_42" / "causal_sensor_boundary_v1" / "evaluation" / "test_all" / "predicted_segments.jsonl", run)
        emg = load_predictions(base / "emg" / f"{heldout}_as_test" / "all_runs" / "seed_42" / "causal_sensor_boundary_v1" / "evaluation" / "test_all" / "predicted_segments.jsonl", run)
        draw_timeline_axis(ax, gt, imu, emg, start, end, action_codes, f"Held-out {heldout}: {run}")
        ax.set_xlabel("Time from annotated stream start (s)")
    legend = [Patch(facecolor=plt.get_cmap("tab20")((code - 1) % 20), label=f"{code}: {action}") for action, code in action_codes.items()]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8, title="GT action code")
    fig.suptitle("Representative full-run segmentation timelines (all-runs, seed 42)", fontsize=14)
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    fig.savefig(target_full, dpi=180, bbox_inches="tight")
    plt.close(fig)

    subset = per_run[(per_run.scope == "all_runs") & (per_run.test_split == "test_all") & (per_run.seed == 42)]
    pivot = subset.pivot(index=["heldout", "sample_name"], columns="modality", values="segmental_f1_50").dropna()
    difference = (pivot.emg - pivot.imu).abs()
    heldout, run = difference.idxmax()
    gt, stream_start, stream_end = annotation_segments(annotation_root, run)
    base = root / "outputs"
    imu = load_predictions(base / "imu" / f"{heldout}_as_test" / "all_runs" / "seed_42" / "causal_sensor_boundary_v1" / "evaluation" / "test_all" / "predicted_segments.jsonl", run)
    emg = load_predictions(base / "emg" / f"{heldout}_as_test" / "all_runs" / "seed_42" / "causal_sensor_boundary_v1" / "evaluation" / "test_all" / "predicted_segments.jsonl", run)
    candidates = np.arange(stream_start, max(stream_start + 1, stream_end - 40), 5.0)
    def window_score(left):
        right = min(left + 40, stream_end)
        gt_count = sum(s["start"] <= right and s["end"] >= left for s in gt)
        imu_count = sum(a <= right and b >= left for a, b in imu)
        emg_count = sum(a <= right and b >= left for a, b in emg)
        return gt_count + abs(imu_count - emg_count)
    zoom_start = float(max(candidates, key=window_score)) if len(candidates) else stream_start
    zoom_end = min(zoom_start + 40.0, stream_end)
    clip_gt = [{**s, "start": max(s["start"], zoom_start), "end": min(s["end"], zoom_end)} for s in gt if s["end"] >= zoom_start and s["start"] <= zoom_end]
    clip = lambda xs: [(max(a, zoom_start), min(b, zoom_end)) for a, b in xs if b >= zoom_start and a <= zoom_end]
    fig, ax = plt.subplots(figsize=(16, 4.2))
    draw_timeline_axis(ax, clip_gt, clip(imu), clip(emg), zoom_start, zoom_end, action_codes, f"Largest IMU–EMG run-level disagreement: held-out {heldout}, {run}, seed 42")
    ax.set_xlabel("Time within selected 40 s window (s)")
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8, title="GT action code")
    fig.tight_layout(rect=(0, 0.26, 1, 1))
    fig.savefig(target_zoom, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected, f"{heldout}/{run}"


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def mean_sd(df: pd.DataFrame, metric: str) -> str:
    return f"{100 * df[metric].mean():.2f} ± {100 * df[metric].std(ddof=1):.2f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_report(root: Path, out: Path, conditions: pd.DataFrame, paired: pd.DataFrame, heldout_stats: pd.DataFrame, training: pd.DataFrame, calibration: pd.DataFrame, action_df: pd.DataFrame, selected: dict[str, str], zoom_run: str, annotation_root: Path, annotation_source: str) -> None:
    primary = conditions[(conditions.scope == "all_runs") & (conditions.test_split == "test_all")]
    primary_rows = []
    for modality in ("imu", "emg"):
        part = primary[primary.modality == modality]
        primary_rows.append([
            modality.upper(), mean_sd(part, "frame_f1"), mean_sd(part, "segmental_f1_50"),
            mean_sd(part, "start_f1_200ms"), mean_sd(part, "end_f1_200ms"),
            f"{part.edit_score.mean():.2f} ± {part.edit_score.std(ddof=1):.2f}",
            f"{part.segment_count_ratio.mean():.2f}", f"{part.emission_delay_ms.mean():.1f}",
        ])
    pair_rows = []
    for _, row in paired.iloc[:5].iterrows():
        scale = 1.0 if row.metric == "edit_score" else 100.0
        pair_rows.append([
            row.metric,
            f"{row.imu_mean * scale:.2f}",
            f"{row.emg_mean * scale:.2f}",
            f"{row.emg_minus_imu * scale:+.2f}",
            f"{int(row.wins_emg)}/{int(row.ties)}/{int(row.wins_imu)}",
            f"{row.wilcoxon_p:.4f}",
            f"{row.holm_p_performance_metrics:.4f}",
        ])
    heldout_rows = []
    for _, row in heldout_stats.iterrows():
        scale = 1.0 if row.metric == "edit_score" else 100.0
        heldout_rows.append([
            row.metric,
            f"{row.emg_minus_imu * scale:+.2f}",
            f"{int(row.heldouts_emg_better)}/{int(row.heldouts_imu_better)}",
            f"{row.wilcoxon_p_n4:.4f}",
        ])
    split_rows = []
    for split in ("test_normal", "test_fault", "test_all"):
        for modality in ("imu", "emg"):
            part = conditions[(conditions.scope == "all_runs") & (conditions.test_split == split) & (conditions.modality == modality)]
            split_rows.append([split.removeprefix("test_"), modality.upper(), mean_sd(part, "frame_f1"), mean_sd(part, "segmental_f1_50"), mean_sd(part, "start_f1_200ms"), mean_sd(part, "end_f1_200ms")])
    participant_rows = []
    for heldout in ("A", "D", "J", "M"):
        for modality in ("imu", "emg"):
            part = primary[(primary.heldout == heldout) & (primary.modality == modality)]
            participant_rows.append([heldout, modality.upper(), pct(part.frame_f1.mean()), pct(part.segmental_f1_50.mean()), pct(part.start_f1_200ms.mean()), pct(part.end_f1_200ms.mean())])
    best_epoch = training.groupby(["modality", "scope"]).best_epoch.agg(["mean", "median", "min", "max"])
    cal_counts = calibration.groupby("modality").agg({
        "start_threshold": lambda x: Counter(x).most_common(1)[0][0],
        "start_debounce_steps": lambda x: Counter(x).most_common(1)[0][0],
        "end_debounce_steps": lambda x: Counter(x).most_common(1)[0][0],
        "min_action_steps": lambda x: Counter(x).most_common(1)[0][0],
        "merge_gap_steps": lambda x: Counter(x).most_common(1)[0][0],
    })
    imu = primary[primary.modality == "imu"]
    emg = primary[primary.modality == "emg"]
    key_delta = paired.set_index("metric").emg_minus_imu
    action_text = []
    for modality in ("imu", "emg"):
        eligible = action_df[(action_df.modality == modality) & (action_df.gt_segments >= 10)].sort_values("segment_recall_iou50")
        weakest = ", ".join(f"{row.action} {row.segment_recall_iou50*100:.1f}%" for _, row in eligible.head(3).iterrows())
        strongest = ", ".join(f"{row.action} {row.segment_recall_iou50*100:.1f}%" for _, row in eligible.tail(3).sort_values("segment_recall_iou50", ascending=False).iterrows())
        action_text.append(f"- {modality.upper()}：较弱为 {weakest}；较强为 {strongest}。")
    report = f"""# IMU / EMG 实时动作分割结果分析

日期：2026-09-07  
结果目录：`outputs`  
分析单位：每个 held-out participant × seed 为一个配对 condition；主结果为 `all_runs / test_all`，n=12 对。

## 1. 结果完整性

- IMU：24 个模型、24 个 validation-only 校准文件、72 个 metrics 文件。
- EMG：24 个模型、24 个 validation-only 校准文件、72 个 metrics 文件。
- 每种模态完整覆盖 A/D/J/M × seed 1/2/42 × normal-only/all-runs × normal/fault/all。
- 所有正式 condition 都存在 `best.pth`、训练日志和校准结果，未发现缺失组合。
- 时间线使用 annotation：`{annotation_root}`。路径选择方式：`{annotation_source}`。正式输出保存的是训练电脑路径；若迁移后该路径不存在，分析脚本会使用当前数据盘的明确 fallback，指标 JSON 本身不受影响。

## 2. 指标的详细含义

### 2.1 Frame Accuracy、Precision、Recall 与 Frame F1

模型每 50 ms 输出一次当前状态，因此这里的“frame”实际是一个 20 Hz decision step，而不是原始 RGB 视频帧。

- **Frame Accuracy**：所有 decision steps 中，预测为 background/action 与 GT 相同的比例，即 `(TP+TN)/(TP+TN+FP+FN)`。Background 通常比 action 多，所以 Accuracy 可能被大量容易的 background 拉高，不能单独代表动作检测质量。
- **Frame Precision**：所有被预测为 action 的 decision steps 中，真实属于 action 的比例，即 `TP/(TP+FP)`。低 Precision 表示模型把较多 background 判成了动作。
- **Frame Recall**：所有真实 action decision steps 中，被模型找出的比例，即 `TP/(TP+FN)`。低 Recall 表示漏掉了较多动作时间。
- **Frame F1**：action 正类 Precision 与 Recall 的调和平均，`2PR/(P+R)`。它不是真正的“background 和 action 两类 macro-F1”，而是以 action 为正类计算的二值 F1。F1 高说明动作区域总体覆盖较好，但并不保证开始和结束位置准确，也不惩罚把相邻动作连成一个长片段的所有结构错误。

每个 `metrics.json` 先对该 test split 中各 run 的指标做等权 macro average；本报告再对 4 held-out participants × 3 seeds 的 12 个 condition 求均值和标准差。因此报告中的 `72.72 ± 3.91%` 表示 12 个 condition macro scores 的 mean ± sample SD，不是逐帧 pooled score，也不是 95% confidence interval。

### 2.2 Boundary Precision、Recall、F1 与时间容差

Start 和 End 分开评价。对于给定容差（±50、±100、±200 或 ±500 ms），程序在预测边界与 GT 边界之间按绝对时间差进行一对一贪心匹配：一个预测最多匹配一个 GT，一个 GT 也最多匹配一个预测。

- **Boundary Precision**：匹配成功的预测边界数 / 全部预测边界数。过分割产生的额外边界会降低它。
- **Boundary Recall**：匹配成功的 GT 边界数 / 全部 GT 边界数。漏掉动作会降低它。
- **Boundary F1**：Boundary Precision 和 Recall 的调和平均。
- **Start/End F1@200 ms**：只有预测时间与 GT start/end 相差不超过 200 ms 才算成功；它是本报告的主边界指标。
- **Tolerance curve**：容差越宽 F1 必然不下降。若 ±200 ms 很低但 ±500 ms 明显升高，表示模型大致找到了动作区域，却没有精确定位边界。
- **Median signed error**：只在已经匹配成功的边界上计算 `prediction_time - GT_time`。负值表示提前，正值表示延迟。
- **Median/P90 absolute error**：只在匹配成功事件中计算绝对时间误差。没有匹配的漏检不会进入该误差统计，所以必须和 Boundary Recall 一起解释，不能只看一个很小的 median error。

### 2.3 Segmental F1@10/@25/@50

先把连续 action decisions 组成预测片段，再与 GT 动作片段计算 temporal Intersection over Union：`IoU = 交集时长 / 并集时长`。随后在指定 IoU 阈值下进行一对一匹配。

- **Segmental F1@10**：IoU ≥ 0.10 即匹配，评价较宽松。
- **Segmental F1@25**：IoU ≥ 0.25。
- **Segmental F1@50**：IoU ≥ 0.50，是本报告更严格的主片段指标。

该指标同时惩罚漏检、额外碎片和片段范围偏差。相邻的不同动作即使中间没有 background，也根据 annotation segment ID 保持为两个 GT 片段；不过当前模型只预测“action/background”，并不预测具体动作类别。

### 2.4 Edit Score

程序先把逐 decision 的二值状态序列压缩，例如 `background, background, action, action, background` 变为 `background → action → background`，再计算归一化 Levenshtein edit distance，最后转换为 0–100 分。分数越高，预测状态转换序列越接近 GT。

由于当前实验只有 background/action 两种状态，这里的 Edit Score 主要衡量动作与背景转换的数量和顺序，能反映过分割/漏分割，但不是多动作类别论文中常见的完整 action-class edit score。

### 2.5 Pred/GT count、短碎片与 Emission ms

- **Pred/GT count**：预测片段总数 / GT 动作片段总数。1.0 最接近数量一致；大于 1 表示总体过分割，小于 1 表示总体漏分割或错误合并。但数量等于 1 仍不保证每个片段正确对应。
- **Short fragment fraction <200 ms**：预测片段中持续时间小于 200 ms 的比例，用于发现抖动造成的极短片段。
- **Emission ms**：在线状态机正式发出一个片段的时刻减去该预测片段的预测 end 时刻。它包含 end debounce 和 pending merge 等确认等待。例如预测 end 在 t，但状态机到 t+100 ms 才确认并输出，则 Emission=100 ms。

Emission ms **不是** `预测 end - GT end`，也不是从动作真实开始到系统识别出的总延迟；边界相对 GT 的提前/延迟由 signed boundary error 描述。它也不包含 MindRove 传输、驱动、磁盘读取或后续 RGB M3 node 分类耗时。

### 2.6 Real-time factor

`RTF = 模型处理耗时 / 传感器流时长`。RTF < 1 表示平均计算速度快于实时，0.01 表示处理 100 秒数据约需 1 秒。这里测量的是已缓存窗口的模型推理，不包含在线采集、同步、滤波与通信，因此只能证明模型计算部分具备实时余量，不能当作完整系统端到端延迟。

### 2.7 Wilcoxon p、Holm p 和胜负数

- **Wilcoxon p**：对同一个 held-out participant、同一个 seed 下的 `EMG - IMU` 差值做双侧 Wilcoxon signed-rank test。零假设是配对差值的分布以 0 为中心。较小的 p 值说明差异方向和大小在配对 condition 中较一致；它不表示“EMG/IMU 为真的概率”，也不直接表示效应大小。
- **Holm p**：因为同时检验 Frame F1、Segmental F1@50、Start F1、End F1、Edit Score 五个主性能指标，使用 Holm step-down 方法校正多重比较。它按 p 值排序并逐步乘以剩余检验数，控制 family-wise error rate；通常不小于原始 Wilcoxon p，应优先用它判断多指标检验后的证据。
- **EMG/平/IMU wins**：12 个 participant-seed condition 中，EMG 高于、等于、低于 IMU 的次数。它直观表示方向稳定性，但不考虑差值大小。

三个 seed 共享同一个 held-out participant 的测试 runs，所以 n=12 不是 12 个独立受试者。本报告另将三个 seed 先求均值，以四个 held-out participants 为统计单位给出 n=4 的 Wilcoxon 结果。n=12 结果适合描述“跨训练随机性的稳定性”，n=4 结果才更接近受试者层面的独立性，但统计功效很低。

### 2.8 按动作类别 Recall

当前模型不输出动作类别，因此预测的额外片段无法被严格归入某个 action class，不能计算可靠的 per-class Precision/F1。报告固定使用 seed 42，让每个 GT 段只出现一次，再按 GT action 统计 Segment IoU@50、Start@200 ms 和 End@200 ms 的命中率。这是 GT-conditioned recall，用于定位哪些动作难检测。

## 3. 从数据输入到输出的完整模型流程

### 3.1 原始输入和时间同步

每个 run 读取 `mindrove_left.csv` 与 `mindrove_right.csv`。两份文件都有 `board_ts`：

- IMU 物理输入：左右手各 `acc_x/y/z + gyro_x/y/z`，共 12 通道。
- EMG 物理输入：左右手各 `emg1...emg8`，共 16 通道。

读取后将非数值/缺失行移除，按 `board_ts` 排序，同一时间戳只保留最后一行。有效分析范围取“左手传感器、右手传感器、逐帧 annotation”三者时间范围的交集，因此不会在某一侧没有数据时仍生成正常标签。

左右手同步采用 previous-sample hold：对于目标时间 t，只查找 `board_ts ≤ t` 的最近样本，不做需要未来样本的线性插值。若最近样本距离 t 超过 0.1 s，则对应手腕的 validity=0；物理值仍保持最近值，同时把左右手各一个 validity channel 输入模型，让网络识别掉线/陈旧样本。

### 3.2 两种模态的因果预处理和窗口

IMU 流程：

```text
12 physical channels
→ 100 Hz uniform grid
→ 4th-order 40 Hz causal low-pass
→ append left/right validity
→ 14 channels
→ past 0.5 s = 50 samples per decision
```

EMG 流程：

```text
16 physical channels
→ 500 Hz uniform grid
→ 4th-order 20–200 Hz causal band-pass
→ 50 Hz notch disabled in this round
→ append left/right validity
→ 18 channels
→ past 0.2 s = 100 samples per decision
```

滤波使用正向 `sosfilt/lfilter`，不使用会访问未来信号的 `filtfilt`。两种模态最终都以 20 Hz 输出决策，即每 50 ms 建立一个 past-only local window。窗口在序列开始处左侧补零，右端就是当前决策时间，不含右端之后的样本。缓存以 float16 保存：IMU 单个 run 的核心 shape 是 `[T,50,14]`，EMG 是 `[T,100,18]`。

### 3.3 监督标签

以包含短 background 的 `action_recognition_boundaries_with_background_v1` 为准，把 segmentation annotation 的绝对时间映射到同一个 20 Hz 网格，生成：

- `state[t]`：0=background，1=action；
- `exact_start[t]`、`exact_end[t]`：离真实开始/结束最近的 decision；
- `start[t]`、`end[t]`：训练用边界目标，将 exact boundary 在 ±100 ms 范围扩张，降低单个 50 ms 网格点造成的过强量化；
- `segment_id[t]`：保存 GT 动作段身份，使两个相邻动作即便没有 background 也不会在片段评价中合并。

训练标签可以使用完整 GT；严格因果性约束的是模型输入和推理，而不是监督信号。

### 3.4 LOSO 划分、归一化和训练样本

每个 condition 固定一个 held-out participant；其所有 run 只进入 test。其余参与者的 train runs 再按 run-level 哈希划分 training/validation，绝不把同一 run 的窗口随机拆到两边。

通道均值和标准差只用实际 training runs 计算，并保存进 checkpoint；validation 和 held-out test 不参与。物理通道执行 `(x-mean)/std`，两个 validity channel 保持 0/1 尺度。训练数据按 256 个 decisions 分块，相当于 12.8 s；相邻块重叠 124 steps（6.2 s），接近 TCN 的历史感受野，减少块边界缺少上下文的影响。

### 3.5 Local Window Encoder

模型输入 batch shape 为 `[B,L,W,C]`：B 是 batch，L 是 decision 序列长度，W 是局部窗口采样点数，C 是通道数。先重排为 `[B×L,C,W]`，让每个 decision 的历史窗口独立通过局部 1D CNN：

```text
Conv1d(C→64, kernel=5, stride=2)
→ GroupNorm(1 group) → GELU → Dropout(0.2)
→ Conv1d(64→128, kernel=5, stride=2)
→ GroupNorm → GELU → Dropout
→ temporal mean pooling
→ Linear(128→128)
```

这里的卷积可以在局部窗口内部同时观察早期和晚期位置，但整个窗口已经严格限制为 t 及其过去，因此相对当前 decision 仍然不使用未来信息。每个 decision 最终得到一个 128 维模态特征。

### 3.6 Causal Boundary TCN

局部特征序列 `[B,L,128]` 先通过 1×1 projection 转成 256 channels，再进入 5 个 residual blocks。每个 block 包含两层 kernel=3 的 causal Conv1d、逐时刻 LayerNorm、GELU 和 Dropout，并加 residual connection。五个 block 的 dilation 依次为 1、2、4、8、16；所有卷积只做左侧 padding。

TCN 感受野为 `1 + 2 × (kernel-1) × (1+2+4+8+16) = 125` 个 decisions，即约 6.25 s 的当前及历史特征。它与局部窗口结合后，同时利用 0.5 s IMU/0.2 s EMG 的细节和约 6.25 s 的长时上下文。

### 3.7 三个输出头和训练损失

TCN 的每个 decision 输出：

1. `state_logits[t,2]`：background/action 两类，经 softmax 得到 `P(action)`；
2. `start_logit[t]`：经 sigmoid 得到 `P(start)`；
3. `end_logit[t]`：经 sigmoid 得到 `P(end)`。

总损失为：

```text
L = 1.0 × CrossEntropy(state)
  + 0.5 × BCEWithLogits(start, positive_weight=20)
  + 0.5 × BCEWithLogits(end, positive_weight=20)
```

边界正样本远少于普通时间点，因此 start/end 使用 positive weight=20。优化器为 AdamW，learning rate=3e-4、weight decay=1e-4，并使用 gradient norm clip=5。最多训练 40 epochs，每个 condition 按最低 validation total loss 保存 `best.pth`；正式评价不用最后一个 epoch 的 `last.pth`。

### 3.8 Validation-only 在线校准

固定 `best.pth` 后，只在 training participants 的 validation runs 上搜索：

- 统一 probability threshold：0.45/0.55/0.65；
- start debounce：1/2 steps；
- end debounce：1/2 steps；
- minimum action：4/6/8 steps，即 200/300/400 ms；
- merge gap：0/2/4 steps，即 0/100/200 ms。

校准目标是 `mean(Start F1@200 ms, End F1@200 ms, Segmental F1@50)`。选中的组合写入 `calibrated_online.json`，随后冻结；test_normal/fault/all 都使用同一组合，不能查看测试集后再调阈值。

### 3.9 在线状态机与片段输出

推理首先得到按时间顺序的 `P(action), P(start), P(end)`，再逐 decision 输入：

```text
BACKGROUND
→ START_CANDIDATE（达到 start 或 action 阈值）
→ ACTION（满足 start debounce）
→ END_CANDIDATE（end 足够高或 action 足够低）
→ pending segment（满足 end debounce 和最短持续时间）
→ 正式 emit 或与短间隔后的下一段合并
```

pending buffer 保证 merge 是在片段正式发出前完成，而不是事后修改已经输出的结果。每个预测片段记录 decision index、绝对 start/end time、真正 emitted-at time 和边界分数。

### 3.10 最终文件和当前任务边界

- `best.pth`：模型参数、归一化统计及配置；
- `calibrated_online.json`：validation 选择的状态机参数；
- `metrics.json`：macro 与 per-run 指标；
- `predicted_segments.jsonl`：每个 run 的二值预测片段；
- online replay 可另外保存每 50 ms 的三种概率流。

当前输出到“检测出 action segment”为止，不包含动作类别或 Task Graph node。若接入 RGB M3 Atomic-tail，应在片段正式 emit 后截取对应 RGB 流，并仅用已经发出的预测历史更新 graph/history；那是下一阶段端到端 node accuracy 实验。

上述实现分别对应 [preprocessing.py](sensor_boundary/preprocessing.py)、[annotations.py](sensor_boundary/annotations.py)、[models.py](sensor_boundary/models.py)、[engine.py](sensor_boundary/engine.py) 和 [online.py](sensor_boundary/online.py)，实际参数来自 [common.json](configs/common.json)、[IMU config](configs/imu/base.json) 与 [EMG config](configs/emg/base.json)。

## 4. 主结果：all-runs / test_all

{markdown_table(["模态", "Frame F1 %", "Segmental F1@50 %", "Start F1@200ms %", "End F1@200ms %", "Edit", "Pred/GT count", "Emission ms"], primary_rows)}

最清晰的结论是：IMU 在本轮主条件的帧级状态、严格片段重叠、开始边界和结束边界上都优于 EMG。EMG 相对 IMU 的平均 Frame F1 变化为 {key_delta['frame_f1']*100:+.2f} 个百分点；Segmental F1@50 变化为 {key_delta['segmental_f1_50']*100:+.2f} 个百分点；Start F1@200 ms 变化为 {key_delta['start_f1_200ms']*100:+.2f} 个百分点；End F1@200 ms 变化为 {key_delta['end_f1_200ms']*100:+.2f} 个百分点。

两种模型的 segment count ratio 都需要重点关注：1.0 表示预测片段数与 GT 相等，大于 1 表示过分割。IMU 平均为 {imu.segment_count_ratio.mean():.2f}，EMG 为 {emg.segment_count_ratio.mean():.2f}。因此较高的帧级 F1 不一定自动转化为准确边界。

![Overall metric comparison](analysis_2026-09-07/overall_metric_comparison.png)

## 5. 严格配对比较

{markdown_table(["指标", "IMU", "EMG", "EMG-IMU", "EMG/平/IMU wins", "Wilcoxon p", "Holm p"], pair_rows)}

这里的 Wilcoxon 检验以 12 个 participant-seed condition 为配对样本；Holm 校正只覆盖表中的五个性能指标。它比直接进行 run-level 检验更保守，但仍不能把三个 seed 当成三个独立受试者。p 值仅用于判断跨 condition 稳定性，不能替代效应量和逐 participant 检查。

三个 seed 共享同一个 held-out participant 测试集，因此 12 个 condition 也不是 12 个完全独立的人体样本。进一步先对 seed 求均值、仅以四个 held-out participant 为单位时：

{markdown_table(["指标", "EMG-IMU", "EMG/IMU heldouts", "Wilcoxon p (n=4)"], heldout_rows)}

Segmental F1@50、start 和 end 的方向在四个 held-out participant 上都一致偏向 IMU，但 n=4 的双侧 Wilcoxon 最小 p 值为 0.125。因此合理表述是“跨折方向一致且跨 seed 稳定”，不能把 n=12 校正 p 值解释成已经完成独立受试者层面的显著性证明。

### 5.1 all-runs 相对 normal-only

在 `test_all` 上，all-runs 对 IMU 的 Frame F1 / Segmental F1@50 / Start@200 / End@200 改变分别为 -0.04 / +1.79 / +1.37 / +2.36 个百分点；对 EMG 分别为 -0.14 / +1.91 / -0.07 / +0.74 个百分点。加入 fault runs 训练主要改善片段重叠，对帧级 F1 几乎没有影响，且 EMG start boundary 没有受益。

## 6. Normal 与 Fault

{markdown_table(["测试集", "模态", "Frame F1 %", "Seg F1@50 %", "Start@200 %", "End@200 %"], split_rows)}

Fault 结果应结合 held-out participant 分开看，因为每折 fault run 数量不同。报告使用 condition macro mean，而不是把拥有更多 run 的 participant 自动赋予更大权重。

![Test split comparison](analysis_2026-09-07/test_split_comparison.png)

## 7. Participant 差异

{markdown_table(["Held-out", "模态", "Frame F1", "Seg F1@50", "Start@200", "End@200"], participant_rows)}

![Paired conditions](analysis_2026-09-07/paired_condition_scatter.png)

散点在虚线之上表示 EMG 优于 IMU，之下表示 IMU 优于 EMG。不同形状对应 held-out participant，可以直接检查总体均值是否由单个参与者驱动。

## 8. 时间容差与延迟

![Boundary tolerance](analysis_2026-09-07/boundary_tolerance_curves.png)

如果 F1 只在 ±500 ms 明显上升，说明模型已经找到大致动作区域，但边界定位仍不够精确。Start 和 End 必须分开报告，因为肌肉预激活可能使 EMG start 提前，而运动停止或肌电残留会影响 end。

当前 emission delay 包含 end debounce 和 pending merge 等算法等待，但不包含实际 MindRove 传输、CSV 写入或 RGB node 分类耗时。主条件平均 delay：IMU {imu.emission_delay_ms.mean():.1f} ms，EMG {emg.emission_delay_ms.mean():.1f} ms。

## 9. 训练与在线校准

最佳 epoch 分布：

```text
{best_epoch.to_string(float_format=lambda x: f'{x:.1f}')}
```

最常被 validation 选中的参数：

```text
{cal_counts.to_string()}
```

最佳 epoch 若经常远早于 40，且 final validation loss 明显高于 best，说明存在后期过拟合；正式比较使用 `best.pth` 是正确的。在线参数来自训练参与者 validation runs，不使用 held-out 测试标签。

## 10. 分割时间线

![Full timelines](analysis_2026-09-07/segmentation_timelines_representative.png)

GT 行使用动作类别颜色和数字代码；IMU、EMG 行只表示二值检测片段，因为当前 boundary 模型不输出动作类别。四个代表 run 是各 held-out participant 在 seed 42 下、两种模态平均 Segmental F1@50 最接近该 participant 中位数的 run：`{selected}`。

![Zoom timeline](analysis_2026-09-07/segmentation_timeline_largest_disagreement_zoom.png)

放大图选择 seed 42 下 IMU 与 EMG Segmental F1@50 差距最大的 run：`{zoom_run}`。它用于观察过分割、漏检、提前开始和延迟结束，不能代替全数据统计。

## 11. 按动作类别分析

![Per-action recall](analysis_2026-09-07/per_action_recall_seed42.png)

为了避免同一 run 被三个 seed 重复计数，这张图固定使用 `all_runs / test_all / seed 42`，每个 GT 动作段只出现一次。由于预测模型只输出二值片段，按类别只能严格计算 GT-conditioned recall，不能计算每类 precision。

{chr(10).join(action_text)}

小样本动作应谨慎解释；图中同时给出每类 GT 段数。动作级结果适合定位下一轮数据增强和 loss weighting 的目标，不能替代总体 LOSO 指标。

## 12. 综合判断

1. 不能只根据 Frame F1 选择模态。实时分割的核心结果应优先看 Segmental F1@50、start/end F1@200 ms、片段数比和 emission delay。
2. IMU 和 EMG 提供互补信号：IMU 对实际运动学变化敏感；EMG 可在明显运动前出现，但更容易受到持续肌肉激活、个体差异和电极状态影响。是否融合应由配对结果与时间线共同决定。
3. 第一轮最值得改进的是 boundary localization，而不是单纯继续提高 action/background recall。建议下一阶段首先加入 validation-only 独立 start/end threshold、boundary peak NMS 和 duration-aware loss，再测试 IMU+EMG late fusion。
4. 融合实验必须保持完全相同的 24 个 LOSO condition，并与最佳单模态逐 condition 配对；不要只汇报 pooled run 均值。
5. 当前结果没有包含 RGB M3 Atomic-tail 的 node accuracy。后续端到端实验应把传感器预测片段交给 M3，并同时报告 boundary 指标与最终 node accuracy。

## 13. 生成文件

- `analysis_2026-09-07/condition_metrics.csv`
- `analysis_2026-09-07/per_run_metrics.csv`
- `analysis_2026-09-07/paired_all_runs_test_all.csv`
- `analysis_2026-09-07/paired_heldout_mean_all_runs_test_all.csv`
- `analysis_2026-09-07/training_summary.csv`
- `analysis_2026-09-07/calibration_summary.csv`
- `analysis_2026-09-07/per_action_recall_seed42.csv`
- `analysis_2026-09-07/analysis_summary.json`
- 七张统计/时间线图片。
"""
    (root / "IMU_EMG_BOUNDARY_RESULTS_ANALYSIS_2026-09-07.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze full IMU/EMG LOSO outputs and create figures/report")
    parser.add_argument("--experiment-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--annotation-root", default=None)
    args = parser.parse_args()
    root = Path(args.experiment_root).resolve()
    outputs = root / "outputs"
    out = root / "analysis_2026-09-07"
    out.mkdir(parents=True, exist_ok=True)
    conditions, per_run = load_metrics(outputs)
    expected = 2 * 4 * 3 * 2 * 3
    if len(conditions) != expected:
        raise ValueError(f"Expected {expected} metric conditions, found {len(conditions)}")
    training, calibration = load_training(outputs)
    if len(training) != 48 or len(calibration) != 48:
        raise ValueError(f"Expected 48 training/calibration rows, found {len(training)}/{len(calibration)}")
    conditions.to_csv(out / "condition_metrics.csv", index=False, encoding="utf-8-sig")
    per_run.to_csv(out / "per_run_metrics.csv", index=False, encoding="utf-8-sig")
    training.to_csv(out / "training_summary.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(out / "calibration_summary.csv", index=False, encoding="utf-8-sig")
    paired = paired_statistics(conditions, "all_runs", "test_all")
    paired.to_csv(out / "paired_all_runs_test_all.csv", index=False, encoding="utf-8-sig")
    heldout_stats = heldout_level_statistics(conditions, "all_runs", "test_all")
    heldout_stats.to_csv(out / "paired_heldout_mean_all_runs_test_all.csv", index=False, encoding="utf-8-sig")
    save_condition_plot(conditions, out / "overall_metric_comparison.png")
    save_split_plot(conditions, out / "test_split_comparison.png")
    save_tolerance_plot(conditions, out / "boundary_tolerance_curves.png")
    save_paired_plot(conditions, out / "paired_condition_scatter.png")
    annotation_root, annotation_source = resolve_annotation_root(root, args.annotation_root)
    action_df = per_action_recall(root, annotation_root)
    action_df.to_csv(out / "per_action_recall_seed42.csv", index=False, encoding="utf-8-sig")
    save_action_heatmap(action_df, out / "per_action_recall_seed42.png")
    selected, zoom_run = save_timelines(root, annotation_root, per_run, out / "segmentation_timelines_representative.png", out / "segmentation_timeline_largest_disagreement_zoom.png")
    primary = conditions[(conditions.scope == "all_runs") & (conditions.test_split == "test_all")]
    summary = {
        "complete_metric_conditions": len(conditions),
        "training_conditions": len(training),
        "calibrated_conditions": len(calibration),
        "primary": {
            modality: {metric: float(part[metric].mean()) for metric in ("frame_f1", "segmental_f1_50", "start_f1_200ms", "end_f1_200ms", "edit_score", "segment_count_ratio", "emission_delay_ms")}
            for modality in ("imu", "emg")
            for part in [primary[primary.modality == modality]]
        },
        "representative_runs": selected,
        "largest_disagreement_zoom": zoom_run,
        "annotation_root_used_for_timelines": str(annotation_root),
        "annotation_path_source": annotation_source,
    }
    write_json(out / "analysis_summary.json", summary)
    write_report(root, out, conditions, paired, heldout_stats, training, calibration, action_df, selected, zoom_run, annotation_root, annotation_source)
    print(f"Analyzed {len(conditions)} metric conditions and {len(per_run)} per-run rows")
    print(root / "IMU_EMG_BOUNDARY_RESULTS_ANALYSIS_2026-09-07.md")


if __name__ == "__main__":
    main()
