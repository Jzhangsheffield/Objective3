from __future__ import annotations

from typing import Iterable

import numpy as np


def binary_classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    tp = int(np.sum(target & prediction))
    fp = int(np.sum(~target & prediction))
    fn = int(np.sum(target & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": float(np.mean(target == prediction))}


def match_events(gt: Iterable[int], pred: Iterable[int], tolerance: int) -> dict[str, float | int]:
    gt = sorted(int(x) for x in gt)
    pred = sorted(int(x) for x in pred)
    candidates = sorted(
        (abs(p - g), p_index, g_index)
        for p_index, p in enumerate(pred)
        for g_index, g in enumerate(gt)
        if abs(p - g) <= tolerance
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    errors: list[int] = []
    for _, p_index, g_index in candidates:
        if p_index not in used_pred and g_index not in used_gt:
            used_pred.add(p_index)
            used_gt.add(g_index)
            errors.append(pred[p_index] - gt[g_index])
    tp, fp, fn = len(errors), len(pred) - len(errors), len(gt) - len(errors)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
        "mean_signed_error_frames": float(np.mean(errors)) if errors else float("nan"),
        "mean_absolute_error_frames": float(np.mean(np.abs(errors))) if errors else float("nan"),
    }


def segments_from_binary(labels: Iterable[int]) -> list[tuple[int, int]]:
    labels = [int(value) for value in labels]
    result: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(labels + [0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index - 1))
            start = None
    return result


def segment_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return intersection / union


def segmental_f1(gt: list[tuple[int, int]], pred: list[tuple[int, int]], threshold: float) -> dict[str, float]:
    candidates = sorted(
        (-segment_iou(p, g), p_index, g_index)
        for p_index, p in enumerate(pred)
        for g_index, g in enumerate(gt)
        if segment_iou(p, g) >= threshold
    )
    used_p, used_g = set(), set()
    for _, p_index, g_index in candidates:
        if p_index not in used_p and g_index not in used_g:
            used_p.add(p_index); used_g.add(g_index)
    tp, fp, fn = len(used_p), len(pred) - len(used_p), len(gt) - len(used_g)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def levenshtein(first: list[int], second: list[int]) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def edit_score(gt_binary: Iterable[int], pred_binary: Iterable[int]) -> float:
    def collapse(values):
        output = []
        for value in values:
            value = int(value)
            if not output or output[-1] != value:
                output.append(value)
        return output
    gt, pred = collapse(gt_binary), collapse(pred_binary)
    denominator = max(len(gt), len(pred), 1)
    return 100.0 * (1.0 - levenshtein(gt, pred) / denominator)


def evaluate_run(
    target_state: np.ndarray,
    pred_state: np.ndarray,
    gt_starts: list[int],
    gt_ends: list[int],
    pred_segments: list[tuple[int, int]],
    tolerances: list[int],
    pred_starts_for_boundary: list[int] | None = None,
    pred_ends_for_boundary: list[int] | None = None,
) -> dict:
    gt_segments = segments_from_binary(target_state)
    pred_starts = [segment[0] for segment in pred_segments] if pred_starts_for_boundary is None else pred_starts_for_boundary
    pred_ends = [segment[1] for segment in pred_segments] if pred_ends_for_boundary is None else pred_ends_for_boundary
    return {
        "frame_state": binary_classification_metrics(target_state, pred_state),
        "boundary": {
            str(tolerance): {
                "start": match_events(gt_starts, pred_starts, tolerance),
                "end": match_events(gt_ends, pred_ends, tolerance),
            }
            for tolerance in tolerances
        },
        "segmental_f1": {str(int(t * 100)): segmental_f1(gt_segments, pred_segments, t) for t in (0.1, 0.25, 0.5)},
        "edit_score": edit_score(target_state, pred_state),
        "gt_segment_count": len(gt_segments),
        "pred_segment_count": len(pred_segments),
    }
