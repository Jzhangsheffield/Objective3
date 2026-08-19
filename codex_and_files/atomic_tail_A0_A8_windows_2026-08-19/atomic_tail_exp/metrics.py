from __future__ import annotations

from typing import Any


def classification_metrics(truth: list[int], prediction: list[int], num_classes: int) -> dict[str, Any]:
    total = len(truth)
    accuracy = sum(int(left == right) for left, right in zip(truth, prediction)) / max(1, total)
    f1_values = []
    recalls = []
    supports = []
    for class_id in range(num_classes):
        tp = sum(t == class_id and p == class_id for t, p in zip(truth, prediction))
        fp = sum(t != class_id and p == class_id for t, p in zip(truth, prediction))
        fn = sum(t == class_id and p != class_id for t, p in zip(truth, prediction))
        support = sum(t == class_id for t in truth)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        if support:
            f1_values.append(f1)
            recalls.append(recall)
            supports.append(support)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / max(1, len(f1_values)),
        "balanced_accuracy": sum(recalls) / max(1, len(recalls)),
        "evaluated_classes": len(f1_values),
        "support": total,
    }
