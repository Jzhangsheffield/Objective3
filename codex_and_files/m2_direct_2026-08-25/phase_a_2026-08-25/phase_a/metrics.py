from __future__ import annotations

from collections import Counter
from typing import Any


def classification_metrics(truth: list[int], prediction: list[int], classes: int) -> dict[str, Any]:
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    for true, pred in zip(truth, prediction):
        confusion[true][pred] += 1
    per_class = []
    for label in range(classes):
        tp = confusion[label][label]
        support = sum(confusion[label])
        predicted = sum(row[label] for row in confusion)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "class_id": label, "support": support, "precision": precision,
            "recall": recall, "f1": f1,
        })
    present = [row for row in per_class if row["support"]]
    return {
        "accuracy": sum(int(a == b) for a, b in zip(truth, prediction)) / max(1, len(truth)),
        "macro_f1": sum(row["f1"] for row in present) / max(1, len(present)),
        "macro_recall": sum(row["recall"] for row in present) / max(1, len(present)),
        "balanced_accuracy": sum(row["recall"] for row in present) / max(1, len(present)),
        "present_class_count": len(present), "total_class_count": classes,
        "weakest_class_recall": min((row["recall"] for row in present), default=0.0),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def top_confusions(truth: list[int], prediction: list[int], limit: int = 12) -> list[dict[str, int]]:
    counts = Counter((true, pred) for true, pred in zip(truth, prediction) if true != pred)
    return [
        {"true_id": true, "pred_id": pred, "count": count}
        for (true, pred), count in counts.most_common(limit)
    ]


def aggregate_node_probabilities(node_probabilities, node_to_tier3: list[int], classes: int = 31):
    import torch
    result = node_probabilities.new_zeros((node_probabilities.shape[0], classes))
    mapping = torch.tensor(node_to_tier3, dtype=torch.long, device=node_probabilities.device)
    result.scatter_add_(1, mapping.unsqueeze(0).expand(node_probabilities.shape[0], -1), node_probabilities)
    return result


def derive_node_to_tier3(rows: list[dict[str, Any]], nodes: int = 35) -> list[int]:
    mapping: dict[int, int] = {}
    for row in rows:
        node = int(row["node_idx"]) - 1
        tier3 = int(row["tier3_id"])
        if node in mapping and mapping[node] != tier3:
            raise ValueError(f"node {node + 1} maps to multiple Tier3 labels")
        mapping[node] = tier3
    missing = set(range(nodes)) - set(mapping)
    if missing:
        raise ValueError(f"Cannot derive node-to-Tier3 map; missing nodes {sorted(x + 1 for x in missing)}")
    return [mapping[index] for index in range(nodes)]
