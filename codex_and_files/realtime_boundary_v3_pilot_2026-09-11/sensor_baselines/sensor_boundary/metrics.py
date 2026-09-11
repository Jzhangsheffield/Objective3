from __future__ import annotations

from typing import Iterable

import numpy as np


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target, prediction = np.asarray(target).astype(bool), np.asarray(prediction).astype(bool)
    tp, fp, fn = int(np.sum(target & prediction)), int(np.sum(~target & prediction)), int(np.sum(target & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": float(np.mean(target == prediction))}


def match_events(gt: Iterable[float], pred: Iterable[float], tolerance_seconds: float) -> dict:
    gt, pred = sorted(float(x) for x in gt), sorted(float(x) for x in pred)
    candidates = sorted(
        (abs(p - g), pi, gi)
        for pi, p in enumerate(pred)
        for gi, g in enumerate(gt)
        if abs(p - g) <= tolerance_seconds
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    errors: list[float] = []
    for _, pi, gi in candidates:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            errors.append(1000.0 * (pred[pi] - gt[gi]))
    tp, fp, fn = len(errors), len(pred) - len(errors), len(gt) - len(errors)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_signed_error_ms": float(np.median(errors)) if errors else None,
        "median_absolute_error_ms": float(np.median(np.abs(errors))) if errors else None,
        "p90_absolute_error_ms": float(np.percentile(np.abs(errors), 90)) if errors else None,
    }


def segments_from_ids(ids: Iterable[int]) -> list[tuple[int, int]]:
    values = [int(x) for x in ids]
    result: list[tuple[int, int]] = []
    start: int | None = None
    current = 0
    for index, value in enumerate(values + [0]):
        if value > 0 and (start is None or value != current):
            if start is not None:
                result.append((start, index - 1))
            start, current = index, value
        elif value == 0 and start is not None:
            result.append((start, index - 1))
            start, current = None, 0
    return result


def segment_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return intersection / max(union, 1)


def segmental_f1(gt: list[tuple[int, int]], pred: list[tuple[int, int]], threshold: float) -> dict[str, float]:
    candidates = sorted(
        (-segment_iou(p, g), pi, gi)
        for pi, p in enumerate(pred)
        for gi, g in enumerate(gt)
        if segment_iou(p, g) >= threshold
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, pi, gi in candidates:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    tp, fp, fn = len(used_p), len(pred) - len(used_p), len(gt) - len(used_g)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _collapse(values: Iterable[int]) -> list[int]:
    output: list[int] = []
    for value in values:
        value = int(value)
        if not output or output[-1] != value:
            output.append(value)
    return output


def _levenshtein(first: list[int], second: list[int]) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def edit_score(gt: Iterable[int], pred: Iterable[int]) -> float:
    first, second = _collapse(gt), _collapse(pred)
    return 100.0 * (1.0 - _levenshtein(first, second) / max(len(first), len(second), 1))


def evaluate_run(
    target_state: np.ndarray,
    pred_state: np.ndarray,
    gt_segment_ids: np.ndarray,
    gt_start_times: list[float],
    gt_end_times: list[float],
    pred_segments: list[tuple[int, int]],
    pred_start_times: list[float],
    pred_end_times: list[float],
    tolerances_ms: list[int],
    duration_seconds: float,
) -> dict:
    gt_segments = segments_from_ids(gt_segment_ids)
    fragment_count = sum((end - start + 1) * 1000.0 / max(len(target_state) / duration_seconds, 1e-9) < 200 for start, end in pred_segments)
    return {
        "frame_state": binary_metrics(target_state, pred_state),
        "boundary": {
            str(ms): {
                "start": match_events(gt_start_times, pred_start_times, ms / 1000.0),
                "end": match_events(gt_end_times, pred_end_times, ms / 1000.0),
            }
            for ms in tolerances_ms
        },
        "segmental_f1": {str(int(x * 100)): segmental_f1(gt_segments, pred_segments, x) for x in (0.1, 0.25, 0.5)},
        "edit_score": edit_score(target_state, pred_state),
        "gt_segment_count": len(gt_segments),
        "pred_segment_count": len(pred_segments),
        "predicted_segments_per_minute": len(pred_segments) * 60.0 / max(duration_seconds, 1e-9),
        "short_fragment_fraction_lt_200ms": fragment_count / max(len(pred_segments), 1),
    }
