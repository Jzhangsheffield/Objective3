from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

import _bootstrap  # noqa: F401
from boundary_experiment.engine import infer_run, load_boundary_checkpoint
from boundary_experiment.features import load_feature_cache
from boundary_experiment.metrics import edit_score, segment_iou
from boundary_experiment.online import run_state_machine


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs" / "A_as_test" / "all_runs" / "seed_1" / "causal_boundary_tcn_v1"
CACHE = ROOT / "cache" / "features" / "A_as_test" / "all_runs" / "seed_1" / "stride_1"
FIGURE_ROOT = ROOT / "docs"

INITIAL_SETTINGS = {
    "start_threshold": 0.55,
    "end_threshold": 0.55,
    "action_threshold": 0.55,
    "start_debounce": 2,
    "end_debounce": 2,
    "min_action_steps": 3,
    "merge_gap_steps": 0,
}

# These values were selected exclusively on the 12 training-side validation
# runs. The final merge is implemented outside the current state machine because
# its existing merge path cannot retract a segment that has already been emitted.
CALIBRATED_SETTINGS = {
    "start_threshold": 0.85,
    "end_threshold": 0.85,
    "action_threshold": 0.65,
    "start_debounce": 2,
    "end_debounce": 2,
    "min_action_steps": 5,
    "merge_gap_steps": 0,
}
CALIBRATED_MERGE_GAP = 3

COLORS = {
    "ground_truth": "#4C78A8",
    "initial_match": "#F28E2B",
    "calibrated_match": "#59A14F",
    "unmatched": "#E15759",
    "background": "#E6E6E6",
    "state_probability": "#4C78A8",
    "start_probability": "#B279A2",
    "end_probability": "#F28E2B",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ground_truth_segment_details(cache: dict) -> list[dict]:
    """Use segment_no so adjacent actions without background remain separate."""
    state = cache["state"].numpy()
    segment_no = cache["segment_no"].numpy()
    result: list[dict] = []
    for value in dict.fromkeys(int(item) for item in segment_no):
        indices = np.flatnonzero((segment_no == value) & (state == 1))
        if len(indices):
            anchor = int(indices[0])
            action = str(cache["action"][anchor]).strip()
            object_name = str(cache["object"][anchor]).strip()
            label = action if not object_name or object_name == "none" else f"{action} · {object_name}"
            result.append(
                {
                    "segment": (int(indices[0]), int(indices[-1])),
                    "action": action,
                    "object": object_name,
                    "label": label,
                }
            )
    return sorted(result, key=lambda item: item["segment"])


def ground_truth_segments(cache: dict) -> list[tuple[int, int]]:
    return [item["segment"] for item in ground_truth_segment_details(cache)]


def tuple_segments(rows: list[dict]) -> list[tuple[int, int]]:
    return [(int(row["start_index"]), int(row["end_index"])) for row in rows]


def merge_short_gaps(segments: list[tuple[int, int]], maximum_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(segments):
        if merged and start - merged[-1][1] - 1 <= maximum_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def one_to_one_matches(
    ground_truth: list[tuple[int, int]], prediction: list[tuple[int, int]], threshold: float = 0.5
) -> set[int]:
    candidates = sorted(
        (-segment_iou(pred, target), pred_index, target_index)
        for pred_index, pred in enumerate(prediction)
        for target_index, target in enumerate(ground_truth)
        if segment_iou(pred, target) >= threshold
    )
    used_prediction: set[int] = set()
    used_target: set[int] = set()
    for _, pred_index, target_index in candidates:
        if pred_index not in used_prediction and target_index not in used_target:
            used_prediction.add(pred_index)
            used_target.add(target_index)
    return used_prediction


def segment_metrics(ground_truth: list[tuple[int, int]], prediction: list[tuple[int, int]]) -> dict[str, float | int]:
    matched = len(one_to_one_matches(ground_truth, prediction))
    precision = matched / len(prediction) if prediction else 0.0
    recall = matched / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gt": len(ground_truth),
        "pred": len(prediction),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def infer_calibrated(model, cache: dict, device: torch.device) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    probabilities = infer_run(model, cache, device)
    raw = tuple_segments(run_state_machine(**probabilities, settings=CALIBRATED_SETTINGS))
    return probabilities, merge_short_gaps(raw, CALIBRATED_MERGE_GAP)


def draw_lane(
    axis,
    segments: list[tuple[int, int]],
    y: float,
    color: str,
    matched: set[int] | None = None,
    height: float = 0.62,
) -> None:
    for index, (start, end) in enumerate(segments):
        segment_color = color if matched is None or index in matched else COLORS["unmatched"]
        axis.broken_barh([(start, end - start + 1)], (y - height / 2, height), facecolors=segment_color)


def build_action_label_map(cases: list[dict]) -> dict[str, int]:
    """Assign one stable number to each action-object category in first-seen order."""
    labels: dict[str, int] = {}
    for case in cases:
        for detail in case["ground_truth_details"]:
            if detail["label"] not in labels:
                labels[detail["label"]] = len(labels) + 1
    return labels


def annotate_ground_truth(
    axis, case: dict, label_map: dict[str, int], visible_range: tuple[int, int] | None = None
) -> None:
    """Put a compact numeric category label in every GT segment."""
    for detail in case["ground_truth_details"]:
        start, end = detail["segment"]
        visible_start, visible_end = start, end
        if visible_range is not None:
            visible_start = max(start, visible_range[0])
            visible_end = min(end, visible_range[1])
            if visible_start > visible_end:
                continue
        midpoint = (visible_start + visible_end) / 2
        axis.text(
            midpoint, 3, str(label_map[detail["label"]]),
            ha="center", va="center", color="white",
            fontsize=7.5, fontweight="bold", clip_on=True,
        )


def draw_action_key(axis, label_map: dict[str, int], labels: set[str] | None = None, columns: int = 5) -> None:
    """Draw a compact number-to-action mapping below a timeline."""
    items = sorted(((number, label) for label, number in label_map.items() if labels is None or label in labels))
    rows = max(int(np.ceil(len(items) / columns)), 1)
    axis.set_axis_off()
    axis.text(0, 1.0, "Ground-truth action key (number = action · object)", ha="left", va="top", fontsize=8.5, fontweight="bold")
    for position, (number, label) in enumerate(items):
        column = position // rows
        row = position % rows
        x = column / columns
        y = 0.78 - row * (0.70 / max(rows - 1, 1))
        axis.text(x, y, f"{number:>2}  {label}", ha="left", va="top", fontsize=7.1)


def plot_full_timelines(cases: list[dict], label_map: dict[str, int]) -> None:
    figure = plt.figure(figsize=(15, 11.5), constrained_layout=False)
    grid = figure.add_gridspec(
        len(cases) + 1, 1, height_ratios=[1] * len(cases) + [0.62],
        left=0.12, right=0.99, bottom=0.035, top=0.86, hspace=0.68,
    )
    axes = [figure.add_subplot(grid[index]) for index in range(len(cases))]
    key_axis = figure.add_subplot(grid[-1])
    for axis, case in zip(axes, cases):
        ground_truth = case["ground_truth"]
        initial = case["initial"]
        calibrated = case["calibrated"]
        total = case["length"]
        initial_matches = one_to_one_matches(ground_truth, initial)
        calibrated_matches = one_to_one_matches(ground_truth, calibrated)
        for center in (3, 2, 1):
            axis.broken_barh([(0, total)], (center - 0.32, 0.64), facecolors=COLORS["background"])
        draw_lane(axis, ground_truth, 3, COLORS["ground_truth"])
        draw_lane(axis, initial, 2, COLORS["initial_match"], initial_matches)
        draw_lane(axis, calibrated, 1, COLORS["calibrated_match"], calibrated_matches)
        annotate_ground_truth(axis, case, label_map)
        for start, end in ground_truth:
            axis.vlines((start, end), 0.58, 3.42, color=COLORS["ground_truth"], alpha=0.18, linewidth=0.55)
        initial_result = segment_metrics(ground_truth, initial)
        calibrated_result = segment_metrics(ground_truth, calibrated)
        axis.set_title(
            f"{case['run']} ({case['kind']}): GT {len(ground_truth)}, "
            f"initial {len(initial)} / F1 {initial_result['f1']:.3f}, "
            f"calibrated {len(calibrated)} / F1 {calibrated_result['f1']:.3f}",
            loc="left",
            fontsize=10,
        )
        axis.set_yticks([3, 2, 1], ["Ground truth", "Initial", "Calibrated"])
        axis.set_xlim(0, total)
        axis.set_ylim(0.5, 3.5)
        axis.set_xlabel("Elapsed annotated frame (stride-1 anchor)")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    legend = [
        Patch(facecolor=COLORS["ground_truth"], label="Ground-truth action"),
        Patch(facecolor=COLORS["initial_match"], label="Initial prediction matched at IoU >= 0.5"),
        Patch(facecolor=COLORS["calibrated_match"], label="Calibrated prediction matched at IoU >= 0.5"),
        Patch(facecolor=COLORS["unmatched"], label="Unmatched prediction at IoU 0.5"),
        Patch(facecolor=COLORS["background"], label="Background"),
    ]
    figure.suptitle("Qualitative boundary comparison on held-out participant A", fontsize=14, y=0.985)
    figure.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.952))
    draw_action_key(key_axis, label_map)
    figure.savefig(FIGURE_ROOT / "boundary_timeline_full_runs.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def comparison_window(case: dict, width: int = 650) -> tuple[int, int]:
    """Choose a dense window that also contains successful calibrated matches."""
    total = case["length"]
    ground_truth = case["ground_truth"]
    initial = case["initial"]
    calibrated = case["calibrated"]
    calibrated_matches = one_to_one_matches(ground_truth, calibrated)
    initial_starts = np.asarray([start for start, _ in initial])
    candidates = list(range(0, max(total - width + 1, 1), 25)) or [0]

    def overlaps(segment: tuple[int, int], left: int, right: int) -> bool:
        return segment[1] >= left and segment[0] < right

    def score(left: int) -> float:
        right = min(left + width, total)
        raw_count = int(np.sum((initial_starts >= left) & (initial_starts < right)))
        gt_count = sum(overlaps(segment, left, right) for segment in ground_truth)
        matched_count = sum(
            overlaps(segment, left, right) for index, segment in enumerate(calibrated)
            if index in calibrated_matches
        )
        unmatched_count = sum(
            overlaps(segment, left, right) for index, segment in enumerate(calibrated)
            if index not in calibrated_matches
        )
        return raw_count + 12 * matched_count + 3 * gt_count - 5 * unmatched_count

    start = max(candidates, key=score)
    return start, min(start + width, total)


def plot_zoom(case: dict, label_map: dict[str, int]) -> None:
    ground_truth = case["ground_truth"]
    initial = case["initial"]
    calibrated = case["calibrated"]
    probabilities = case["probabilities"]
    left, right = comparison_window(case)
    figure = plt.figure(figsize=(15, 7.35), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=[1.0, 1.45, 0.24])
    timeline = figure.add_subplot(grid[0])
    probability_axis = figure.add_subplot(grid[1], sharex=timeline)
    key_axis = figure.add_subplot(grid[2])
    for center in (3, 2, 1):
        timeline.broken_barh([(left, right - left)], (center - 0.32, 0.64), facecolors=COLORS["background"])
    draw_lane(timeline, ground_truth, 3, COLORS["ground_truth"])
    draw_lane(timeline, initial, 2, COLORS["initial_match"], one_to_one_matches(ground_truth, initial))
    draw_lane(timeline, calibrated, 1, COLORS["calibrated_match"], one_to_one_matches(ground_truth, calibrated))
    annotate_ground_truth(timeline, case, label_map, visible_range=(left, right))
    for start, end in ground_truth:
        if end >= left and start <= right:
            timeline.vlines((start, end), 0.58, 3.42, color=COLORS["ground_truth"], alpha=0.25, linewidth=0.8)
            probability_axis.vlines((start, end), 0, 1, color=COLORS["ground_truth"], alpha=0.16, linewidth=0.8)
    timeline.set_yticks([3, 2, 1], ["Ground truth", "Initial", "Calibrated"])
    timeline.set_ylim(0.5, 3.5)
    timeline.set_xlim(left, right)
    timeline.set_title(
        f"{case['run']} zoom: initial fragmentation and validation-calibrated consolidation",
        loc="left", fontsize=11,
    )
    timeline.spines[["top", "right", "left", "bottom"]].set_visible(False)
    timeline.tick_params(axis="x", length=0, labelbottom=False)
    timeline.tick_params(axis="y", length=0)

    frames = np.arange(case["length"])
    probability_axis.plot(frames, probabilities["state_probability"], color=COLORS["state_probability"], label="Action state probability", linewidth=1.1)
    probability_axis.plot(frames, probabilities["start_probability"], color=COLORS["start_probability"], label="Start probability", linewidth=0.85, alpha=0.9)
    probability_axis.plot(frames, probabilities["end_probability"], color=COLORS["end_probability"], label="End probability", linewidth=0.85, alpha=0.9)
    probability_axis.axhline(0.55, color="#666666", linestyle="--", linewidth=0.9, label="Initial threshold 0.55")
    probability_axis.axhline(0.65, color=COLORS["state_probability"], linestyle=":", linewidth=1.0, label="Calibrated state threshold 0.65")
    probability_axis.axhline(0.85, color="#333333", linestyle=":", linewidth=1.0, label="Calibrated boundary threshold 0.85")
    probability_axis.set_ylim(0, 1.02)
    probability_axis.set_ylabel("Probability")
    probability_axis.set_xlabel("Elapsed annotated frame (stride-1 anchor)")
    probability_axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
    probability_axis.spines[["top", "right"]].set_visible(False)
    probability_axis.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    visible_labels = {
        detail["label"] for detail in case["ground_truth_details"]
        if detail["segment"][1] >= left and detail["segment"][0] <= right
    }
    draw_action_key(key_axis, label_map, labels=visible_labels, columns=max(len(visible_labels), 1))
    figure.savefig(FIGURE_ROOT / "boundary_timeline_zoom_run_sample_000001.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def binary_state(length: int, segments: list[tuple[int, int]]) -> np.ndarray:
    result = np.zeros(length, dtype=np.int64)
    for start, end in segments:
        result[start : end + 1] = 1
    return result


def aggregate_split(cases: list[dict]) -> dict:
    totals = {"gt": 0, "initial": 0, "calibrated": 0, "initial_match": 0, "calibrated_match": 0}
    initial_edits: list[float] = []
    calibrated_edits: list[float] = []
    for case in cases:
        gt = case["ground_truth"]
        initial = case["initial"]
        calibrated = case["calibrated"]
        totals["gt"] += len(gt)
        totals["initial"] += len(initial)
        totals["calibrated"] += len(calibrated)
        totals["initial_match"] += len(one_to_one_matches(gt, initial))
        totals["calibrated_match"] += len(one_to_one_matches(gt, calibrated))
        target_state = case["cache"]["state"].numpy()
        initial_edits.append(edit_score(target_state, binary_state(len(target_state), initial)))
        calibrated_edits.append(edit_score(target_state, binary_state(len(target_state), calibrated)))
    for prefix in ("initial", "calibrated"):
        prediction = totals[prefix]
        match = totals[f"{prefix}_match"]
        precision = match / prediction if prediction else 0.0
        recall = match / totals["gt"] if totals["gt"] else 0.0
        totals[f"{prefix}_precision"] = precision
        totals[f"{prefix}_recall"] = recall
        totals[f"{prefix}_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    totals["initial_edit_macro"] = float(np.mean(initial_edits))
    totals["calibrated_edit_macro"] = float(np.mean(calibrated_edits))
    return totals


def plot_aggregate(summary: dict[str, dict]) -> None:
    splits = ["Normal", "Fault", "All"]
    keys = ["test_normal", "test_fault", "test_all"]
    x = np.arange(len(keys))
    width = 0.24
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    count_bars = [
        axes[0].bar(x - width, [summary[key]["gt"] for key in keys], width, color=COLORS["ground_truth"], label="Ground truth"),
        axes[0].bar(x, [summary[key]["initial"] for key in keys], width, color=COLORS["unmatched"], label="Initial"),
        axes[0].bar(x + width, [summary[key]["calibrated"] for key in keys], width, color=COLORS["calibrated_match"], label="Calibrated"),
    ]
    axes[0].set_title("Segment count")
    axes[0].set_ylabel("Number of segments")
    axes[0].set_xticks(x, splits)
    axes[0].legend(frameon=False, fontsize=8)

    f1_bars = [
        axes[1].bar(x - width / 2, [100 * summary[key]["initial_f1"] for key in keys], width, color=COLORS["unmatched"], label="Initial"),
        axes[1].bar(x + width / 2, [100 * summary[key]["calibrated_f1"] for key in keys], width, color=COLORS["calibrated_match"], label="Calibrated"),
    ]
    axes[1].set_title("Segment F1 at IoU >= 0.5")
    axes[1].set_ylabel("F1 (%)")
    axes[1].set_xticks(x, splits)
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False, fontsize=8)

    edit_bars = [
        axes[2].bar(x - width / 2, [summary[key]["initial_edit_macro"] for key in keys], width, color=COLORS["unmatched"], label="Initial"),
        axes[2].bar(x + width / 2, [summary[key]["calibrated_edit_macro"] for key in keys], width, color=COLORS["calibrated_match"], label="Calibrated"),
    ]
    axes[2].set_title("Edit score")
    axes[2].set_ylabel("Macro edit score")
    axes[2].set_xticks(x, splits)
    axes[2].set_ylim(0, 100)
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    for bars in count_bars:
        axes[0].bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    for bars in f1_bars:
        axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    for bars in edit_bars:
        axes[2].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    figure.suptitle("Initial decoder versus validation-calibrated decoder on held-out participant A", fontsize=13)
    figure.savefig(FIGURE_ROOT / "boundary_aggregate_before_after.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_boundary_checkpoint(EXPERIMENT / "best.pth", device)
    initial_rows = read_jsonl(EXPERIMENT / "evaluation" / "test_all" / "predicted_segments.jsonl")
    initial_by_run: dict[str, list[dict]] = defaultdict(list)
    for row in initial_rows:
        initial_by_run[str(row["sample_name"])].append(row)

    split_runs = {
        split: list(read_json(EXPERIMENT / "evaluation" / split / "metrics.json")["per_run"])
        for split in ("test_normal", "test_fault", "test_all")
    }
    all_cases: dict[str, dict] = {}
    for run_position, run in enumerate(split_runs["test_all"], 1):
        cache = load_feature_cache(CACHE / f"{run}.pt")
        probabilities, calibrated = infer_calibrated(model, cache, device)
        all_cases[run] = {
            "run": run,
            "kind": "fault" if run in split_runs["test_fault"] else "normal",
            "cache": cache,
            "length": len(cache["state"]),
            "ground_truth_details": ground_truth_segment_details(cache),
            "ground_truth": ground_truth_segments(cache),
            "initial": tuple_segments(initial_by_run[run]),
            "calibrated": calibrated,
            "probabilities": probabilities,
        }
        print(f"[{run_position}/{len(split_runs['test_all'])}] {run}")

    selected = [all_cases[name] for name in ("run_sample_000001", "run_sample_000007", "run_sample_000020")]
    label_map = build_action_label_map(selected)
    plot_full_timelines(selected, label_map)
    plot_zoom(all_cases["run_sample_000001"], label_map)
    summary = {
        split: aggregate_split([all_cases[name] for name in runs])
        for split, runs in split_runs.items()
    }
    plot_aggregate(summary)
    output = {
        "source_checkpoint": str(EXPERIMENT / "best.pth"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "initial_settings": INITIAL_SETTINGS,
        "calibrated_settings": CALIBRATED_SETTINGS,
        "calibrated_merge_gap": CALIBRATED_MERGE_GAP,
        "selection_policy": "decoder parameters selected on the 12 training-side validation runs only",
        "summary": summary,
        "qualitative_runs": [case["run"] for case in selected],
        "ground_truth_action_key": {str(number): label for label, number in label_map.items()},
    }
    (FIGURE_ROOT / "boundary_visualization_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote figures and summary to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
