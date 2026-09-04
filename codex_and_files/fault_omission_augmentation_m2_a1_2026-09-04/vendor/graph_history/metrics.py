from __future__ import annotations

from typing import Any

import numpy as np
import torch


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            matrix[truth, pred] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = np.diag(matrix).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    present = support > 0
    total = matrix.sum()
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "present_class_count": int(present.sum()),
        "total_class_count": int(matrix.shape[0]),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def classification_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, Any]:
    true_array = np.asarray(y_true, dtype=np.int64)
    pred_array = np.asarray(y_pred, dtype=np.int64)
    return metrics_from_confusion(confusion_matrix(true_array, pred_array, num_classes))


def aggregate_node_probabilities(
    node_probabilities: torch.Tensor,
    node_to_tier3: torch.Tensor,
    num_tier3: int,
) -> torch.Tensor:
    output_shape = (*node_probabilities.shape[:-1], num_tier3)
    result = torch.zeros(output_shape, device=node_probabilities.device, dtype=node_probabilities.dtype)
    index = node_to_tier3.view(*([1] * (node_probabilities.ndim - 1)), -1).expand_as(node_probabilities)
    result.scatter_add_(-1, index, node_probabilities)
    return result
