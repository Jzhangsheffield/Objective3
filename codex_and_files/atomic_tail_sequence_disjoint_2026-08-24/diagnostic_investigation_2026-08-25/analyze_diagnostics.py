from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from torchzip_numpy import load_torch_zip


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "experiment_config.json"
MODELS = (
    "M2-Direct-RealOrder",
    "A1-Legacy-Once",
    "A3-DualPos-Once",
)
AUGMENTATION_MODELS = (
    "A1-Legacy-Once",
    "A1-Legacy-Every10-Replace",
    "A3-DualPos-Once",
    "A3-DualPos-Every10",
)
PARTICIPANTS = ("A", "D", "J", "M")
SEEDS = (1, 2, 42)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_runs(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["participant"]), str(row["run"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["annotation_row_index"]))
    return dict(grouped)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    result = np.exp(shifted)
    return result / np.sum(result, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


class NumpyDirectHistoryClassifier:
    def __init__(self, state: dict[str, np.ndarray], model_config: dict[str, Any]) -> None:
        self.s = {key: np.asarray(value, dtype=np.float32) for key, value in state.items()}
        self.num_heads = int(model_config["num_heads"])
        self.max_history = int(model_config["max_history"])
        self.zero_shift_index = self.max_history - 1
        self.d_model = int(model_config["d_model"])
        self.head_dim = self.d_model // self.num_heads

    def current_project(self, current: np.ndarray) -> np.ndarray:
        x = linear(current, self.s["current_projection.0.weight"], self.s["current_projection.0.bias"])
        return layer_norm(x, self.s["current_projection.1.weight"], self.s["current_projection.1.bias"])

    def history_project(self, history: np.ndarray) -> np.ndarray:
        x = linear(history, self.s["history_projection.0.weight"], self.s["history_projection.0.bias"])
        return layer_norm(x, self.s["history_projection.1.weight"], self.s["history_projection.1.bias"])

    def forward_projected(
        self,
        current_feature: np.ndarray,
        current: np.ndarray,
        history: np.ndarray,
        position_ids: np.ndarray,
        shift_ids: np.ndarray,
        padding_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch, length, _ = history.shape
        if length:
            clipped_positions = np.clip(position_ids, 0, self.max_history)
            history = history + self.s["position_embedding.weight"][clipped_positions]
            max_shift = self.max_history - 1
            shift_indices = np.clip(shift_ids, -max_shift, max_shift) + self.zero_shift_index
            history = history + self.s["shift_embedding.weight"][shift_indices]
        null = np.broadcast_to(self.s["null_history"], (batch, 1, self.d_model))
        keys = np.concatenate([null, history], axis=1)
        padding = np.concatenate([np.zeros((batch, 1), dtype=bool), padding_mask], axis=1)

        in_weight = self.s["attention.in_proj_weight"]
        in_bias = self.s["attention.in_proj_bias"]
        query = linear(current, in_weight[: self.d_model], in_bias[: self.d_model])
        key = linear(keys, in_weight[self.d_model : 2 * self.d_model], in_bias[self.d_model : 2 * self.d_model])
        value = linear(keys, in_weight[2 * self.d_model :], in_bias[2 * self.d_model :])
        query = query.reshape(batch, self.num_heads, self.head_dim)
        key = key.reshape(batch, keys.shape[1], self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch, keys.shape[1], self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = np.einsum("bhd,bhld->bhl", query, key) / math.sqrt(self.head_dim)
        scores = np.where(padding[:, None, :], -1e30, scores)
        attention = softmax(scores, axis=-1)
        context = np.einsum("bhl,bhld->bhd", attention, value).reshape(batch, self.d_model)
        context = linear(context, self.s["attention.out_proj.weight"], self.s["attention.out_proj.bias"])
        fused = linear(
            np.concatenate([current_feature, context], axis=-1),
            self.s["fusion.weight"],
            self.s["fusion.bias"],
        )
        normalized = layer_norm(fused, self.s["node_classifier.0.weight"], self.s["node_classifier.0.bias"])
        logits = linear(normalized, self.s["node_classifier.1.weight"], self.s["node_classifier.1.bias"])
        return logits, attention


def tier3_probabilities(node_probabilities: np.ndarray, node_to_tier3: list[int]) -> np.ndarray:
    result = np.zeros((len(node_probabilities), max(node_to_tier3) + 1), dtype=np.float32)
    for node_index, tier3_index in enumerate(node_to_tier3):
        result[:, tier3_index] += node_probabilities[:, node_index]
    return result


def macro_f1(true: np.ndarray, predicted: np.ndarray) -> float:
    classes = sorted(set(int(v) for v in true))
    if not classes:
        return float("nan")
    values = []
    for cls in classes:
        tp = int(np.sum((true == cls) & (predicted == cls)))
        fp = int(np.sum((true != cls) & (predicted == cls)))
        fn = int(np.sum((true == cls) & (predicted != cls)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    middle = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / middle), axis=-1) + 0.5 * np.sum(q * np.log(q / middle), axis=-1)


def protocol_root(participant: str) -> Path:
    return PACKAGE_ROOT / "inputs" / f"{participant}_as_test" / "cam_001484412812" / "protocols" / "all_runs"


def feature_cache(participant: str, seed: int, split: str = "test_all") -> Path:
    return (
        PACKAGE_ROOT
        / "outputs"
        / "upstream"
        / f"{participant}_as_test"
        / "cam_001484412812"
        / f"seed_{seed}"
        / "features"
        / "retrained_all_runs"
        / f"{split}.pt"
    )


def checkpoint(model: str, participant: str, seed: int) -> Path:
    return (
        PACKAGE_ROOT
        / "outputs"
        / "history_models"
        / model
        / "all_runs"
        / f"{participant}_as_test"
        / f"seed_{seed}"
        / "last.pth"
    )


def prediction_csv(model: str, participant: str, seed: int) -> Path:
    return checkpoint(model, participant, seed).parent / "test_results_actual_order" / "test_all_predictions.csv"


def ordered_examples(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    result = []
    for run_rows in group_runs(rows).values():
        for index, current in enumerate(run_rows):
            result.append((current, list(run_rows[:index])))
    return sorted(result, key=lambda item: (str(item[0]["participant"]), str(item[0]["run"]), int(item[0]["annotation_row_index"])))


def stable_random_seed(seed: int, sample_name: str, stream: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"diagnostic:{seed}:{sample_name}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def make_permutations(
    rows: list[dict[str, Any]], graph: Any, seed: int, active_tail_only: bool
) -> list[np.ndarray]:
    from atomic_tail_exp.augmentation import augment_history, stable_seed

    output = []
    for current, history in ordered_examples(rows):
        result = augment_history(
            history,
            graph,
            stable_seed(seed, 17, str(current["sample_name"]), stream="diagnostic_test"),
            active_tail_only,
            "uniform",
            None,
            16,
            0.75,
            0.35,
            2,
            1,
        )
        actual_lookup = {str(row["sample_name"]): i for i, row in enumerate(history)}
        output.append(np.asarray([actual_lookup[str(row["sample_name"])] for row in result.rows], dtype=np.int64))
    return output


def variant_order(length: int, variant: str, graph_order: np.ndarray, seed: int, sample_name: str) -> np.ndarray:
    if variant in {"actual", "no_history"}:
        return np.arange(length, dtype=np.int64)
    if variant in {"graph_valid_native", "graph_valid_truepos_control"}:
        return graph_order
    if variant == "reverse_presented":
        return np.arange(length - 1, -1, -1, dtype=np.int64)
    if variant == "random_presented":
        order = list(range(length))
        random.Random(stable_random_seed(seed, sample_name, variant)).shuffle(order)
        return np.asarray(order, dtype=np.int64)
    raise ValueError(variant)


def build_variant_batch(
    batch_examples: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    actual_projected: list[np.ndarray],
    graph_orders: list[np.ndarray],
    variant: str,
    model: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = [0 if variant == "no_history" else len(history) for _, history in batch_examples]
    max_length = max(lengths, default=0)
    d_model = actual_projected[0].shape[-1] if actual_projected else 256
    features = np.zeros((len(batch_examples), max_length, d_model), dtype=np.float32)
    positions = np.zeros((len(batch_examples), max_length), dtype=np.int64)
    shifts = np.zeros((len(batch_examples), max_length), dtype=np.int64)
    padding = np.ones((len(batch_examples), max_length), dtype=bool)
    for i, ((current, history), projected, graph_order) in enumerate(zip(batch_examples, actual_projected, graph_orders)):
        length = lengths[i]
        if not length:
            continue
        order = variant_order(length, variant, graph_order, seed, str(current["sample_name"]))
        features[i, :length] = projected[order]
        padding[i, :length] = False
        true_recency = length - order
        presented = np.arange(length, 0, -1, dtype=np.int64)
        if variant == "actual":
            positions[i, :length] = np.arange(length, 0, -1, dtype=np.int64)
        elif variant == "graph_valid_truepos_control":
            positions[i, :length] = true_recency
        elif variant == "graph_valid_native" and model == "A3-DualPos-Once":
            positions[i, :length] = true_recency
            shifts[i, :length] = presented - true_recency
        else:
            positions[i, :length] = presented
    return features, positions, shifts, padding


def run_order_sensitivity(graph: Any, node_to_tier3: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = (
        "actual",
        "graph_valid_truepos_control",
        "graph_valid_native",
        "random_presented",
        "reverse_presented",
        "no_history",
    )
    sample_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    for participant in PARTICIPANTS:
        test_rows = read_jsonl(protocol_root(participant) / "test_all.jsonl")
        examples = ordered_examples(test_rows)
        for seed in SEEDS:
            cache = load_torch_zip(feature_cache(participant, seed))
            lookup = {str(row["sample_name"]): index for index, row in enumerate(cache["records"])}
            current_features = np.stack([cache["features"][lookup[str(current["sample_name"])]] for current, _ in examples]).astype(np.float32)
            history_features = [
                np.stack([cache["features"][lookup[str(row["sample_name"])]] for row in history]).astype(np.float32)
                if history
                else np.zeros((0, current_features.shape[1]), dtype=np.float32)
                for _, history in examples
            ]
            broad_orders = make_permutations(test_rows, graph, seed, active_tail_only=False)
            tail_orders = make_permutations(test_rows, graph, seed, active_tail_only=True)
            for model_name in MODELS:
                saved = load_torch_zip(checkpoint(model_name, participant, seed))
                model = NumpyDirectHistoryClassifier(saved["model_state_dict"], saved["model_config"])
                all_probabilities: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
                all_attention_history_mass: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
                batch_size = 64
                for start in range(0, len(examples), batch_size):
                    stop = min(len(examples), start + batch_size)
                    batch_examples = examples[start:stop]
                    batch_current_features = current_features[start:stop]
                    batch_current = model.current_project(batch_current_features)
                    batch_projected = [model.history_project(values) for values in history_features[start:stop]]
                    graph_orders = (tail_orders if model_name == "A3-DualPos-Once" else broad_orders)[start:stop]
                    for variant in variants:
                        h, p, s, mask = build_variant_batch(
                            batch_examples, batch_projected, graph_orders, variant, model_name, seed
                        )
                        logits, attention = model.forward_projected(
                            batch_current_features, batch_current, h, p, s, mask
                        )
                        probs = softmax(logits, axis=-1).astype(np.float32)
                        all_probabilities[variant].append(probs)
                        all_attention_history_mass[variant].append(1.0 - attention[:, :, 0].mean(axis=1))
                probabilities = {key: np.concatenate(value, axis=0) for key, value in all_probabilities.items()}
                history_mass = {key: np.concatenate(value, axis=0) for key, value in all_attention_history_mass.items()}
                actual = probabilities["actual"]
                actual_tier3 = tier3_probabilities(actual, node_to_tier3)
                stored_predictions = pd.read_csv(prediction_csv(model_name, participant, seed))
                stored_map = dict(zip(stored_predictions.sample_name.astype(str), stored_predictions.pred_node_idx.astype(int)))
                reproduced = np.argmax(actual, axis=1) + 1
                expected = np.asarray([stored_map[str(current["sample_name"])] for current, _ in examples])
                validation_records.append({
                    "model": model_name,
                    "participant": participant,
                    "seed": seed,
                    "samples": len(examples),
                    "prediction_match_fraction": float(np.mean(reproduced == expected)),
                    "mismatches": int(np.sum(reproduced != expected)),
                })
                for index, (current, history) in enumerate(examples):
                    actual_node = int(current["node_idx"]) - 1
                    actual_tier = int(current["tier3_id"])
                    graph_order = (tail_orders if model_name == "A3-DualPos-Once" else broad_orders)[index]
                    for variant in variants:
                        variant_probs = probabilities[variant][index]
                        variant_tier3 = tier3_probabilities(variant_probs[None, :], node_to_tier3)[0]
                        sample_records.append({
                            "model": model_name,
                            "participant": participant,
                            "seed": seed,
                            "sample_name": str(current["sample_name"]),
                            "condition": "fault" if str(current["sample_name"]) in fault_samples(participant) else "normal",
                            "stage": int(current["stage_id"]),
                            "history_length": len(history),
                            "eligible_reorder": len(history) >= 2,
                            "graph_permutation_changed": bool(len(history) >= 2 and not np.array_equal(graph_order, np.arange(len(history)))),
                            "variant": variant,
                            "true_node_idx": actual_node + 1,
                            "true_tier3_id": actual_tier,
                            "pred_node_idx": int(np.argmax(variant_probs)) + 1,
                            "pred_tier3_id": int(np.argmax(variant_tier3)),
                            "node_correct": int(np.argmax(variant_probs)) == actual_node,
                            "tier3_correct": int(np.argmax(variant_tier3)) == actual_tier,
                            "node_confidence": float(np.max(variant_probs)),
                            "true_node_probability": float(variant_probs[actual_node]),
                            "attention_history_mass": float(history_mass[variant][index]),
                            "node_js_vs_actual": float(js_divergence(actual[index:index+1], variant_probs[None, :])[0]),
                            "node_total_variation_vs_actual": float(0.5 * np.sum(np.abs(actual[index] - variant_probs))),
                            "top1_changed_vs_actual": int(np.argmax(actual[index])) != int(np.argmax(variant_probs)),
                            "tier3_top1_changed_vs_actual": int(np.argmax(actual_tier3[index])) != int(np.argmax(variant_tier3)),
                            "abs_true_node_probability_change": float(abs(actual[index, actual_node] - variant_probs[actual_node])),
                        })
                print(f"order sensitivity: {model_name} {participant} seed={seed}", flush=True)
    samples = pd.DataFrame(sample_records)
    validation = pd.DataFrame(validation_records)
    samples.to_csv(HERE / "order_sensitivity_samples.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(HERE / "order_sensitivity_reproduction_check.csv", index=False, encoding="utf-8-sig")
    return samples, validation


_FAULT_CACHE: dict[str, set[str]] = {}


def fault_samples(participant: str) -> set[str]:
    if participant not in _FAULT_CACHE:
        rows = read_jsonl(protocol_root(participant) / "test_fault.jsonl")
        _FAULT_CACHE[participant] = {str(row["sample_name"]) for row in rows}
    return _FAULT_CACHE[participant]


def summarize_order_sensitivity(samples: pd.DataFrame) -> pd.DataFrame:
    eligible = samples[(samples.variant != "actual") & (samples.history_length >= 2)].copy()
    scopes = [("eligible_history_len_ge_2", eligible)]
    graph_changed = eligible[
        eligible.variant.isin(["graph_valid_truepos_control", "graph_valid_native"])
        & eligible.graph_permutation_changed
    ]
    scopes.append(("graph_permutation_changed_only", graph_changed))
    outputs = []
    for scope_name, scoped in scopes:
        grouped = scoped.groupby(["model", "variant"], sort=False)
        summary = grouped.agg(
            samples=("sample_name", "count"),
            node_top1_change_rate=("top1_changed_vs_actual", "mean"),
            tier3_top1_change_rate=("tier3_top1_changed_vs_actual", "mean"),
            mean_js=("node_js_vs_actual", "mean"),
            median_js=("node_js_vs_actual", "median"),
            mean_total_variation=("node_total_variation_vs_actual", "mean"),
            mean_abs_true_probability_change=("abs_true_node_probability_change", "mean"),
            tier3_accuracy=("tier3_correct", "mean"),
            history_attention_mass=("attention_history_mass", "mean"),
        ).reset_index()
        summary.insert(2, "analysis_scope", scope_name)
        actual_lookup = samples[
            (samples.variant == "actual")
            & samples.set_index(["model", "participant", "seed", "sample_name"]).index.isin(
                scoped.set_index(["model", "participant", "seed", "sample_name"]).index
            )
        ].groupby("model")["tier3_correct"].mean()
        summary["tier3_accuracy_delta_vs_actual"] = summary.apply(
            lambda row: float(row.tier3_accuracy - actual_lookup[row.model]), axis=1
        )
        outputs.append(summary)
    summary = pd.concat(outputs, ignore_index=True)
    summary.to_csv(HERE / "order_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def add_group_metadata(graph: Any) -> pd.DataFrame:
    records = []
    for participant in PARTICIPANTS:
        train_rows = read_jsonl(protocol_root(participant) / "train.jsonl")
        test_rows = read_jsonl(protocol_root(participant) / "test_all.jsonl")
        train_examples = ordered_examples(train_rows)
        exact_prefixes = {tuple(int(row["node_idx"]) for row in history) for _, history in train_examples}
        suffix_sets = {
            k: {
                tuple(int(row["node_idx"]) for row in history[-k:])
                for _, history in train_examples
                if len(history) >= k
            }
            for k in (1, 2, 3)
        }
        from atomic_tail_exp.augmentation import select_active_tail

        for current, history in ordered_examples(test_rows):
            nodes = tuple(int(row["node_idx"]) for row in history)
            decision = select_active_tail(history, graph)
            length = len(history)
            if length == 0:
                length_bin = "0"
            elif length <= 2:
                length_bin = "1-2"
            elif length <= 5:
                length_bin = "3-5"
            elif length <= 10:
                length_bin = "6-10"
            elif length <= 20:
                length_bin = "11-20"
            else:
                length_bin = "21+"
            row = {
                "participant": participant,
                "sample_name": str(current["sample_name"]),
                "condition": "fault" if str(current["sample_name"]) in fault_samples(participant) else "normal",
                "history_length": length,
                "history_length_bin": length_bin,
                "stage": int(current["stage_id"]),
                "active_tail": decision.reason == "active_incomplete_atomic_prefix",
                "tail_reason": decision.reason,
                "tail_length": len(decision.node_ids),
                "exact_full_prefix_seen": nodes in exact_prefixes,
            }
            for k in (1, 2, 3):
                row[f"suffix{k}_status"] = "insufficient" if length < k else ("seen" if nodes[-k:] in suffix_sets[k] else "unseen")
            row["local_prefix3_status"] = row["suffix3_status"]
            records.append(row)
    metadata = pd.DataFrame(records)
    metadata.to_csv(HERE / "test_group_metadata.csv", index=False, encoding="utf-8-sig")
    return metadata


def grouped_performance(metadata: pd.DataFrame) -> pd.DataFrame:
    result = []
    groupings = {
        "exact_full_prefix": "exact_full_prefix_seen",
        "local_suffix_1": "suffix1_status",
        "local_suffix_2": "suffix2_status",
        "local_prefix_3": "local_prefix3_status",
        "history_length": "history_length_bin",
        "stage": "stage",
        "active_tail": "active_tail",
        "tail_reason": "tail_reason",
    }
    for model in MODELS:
        for participant in PARTICIPANTS:
            base_meta = metadata[metadata.participant == participant]
            for seed in SEEDS:
                predictions = pd.read_csv(prediction_csv(model, participant, seed))
                merged = predictions.merge(base_meta, on=["participant", "sample_name"], how="left", validate="one_to_one")
                for condition in ("normal", "fault", "all"):
                    subset = merged if condition == "all" else merged[merged.condition == condition]
                    for grouping, column in groupings.items():
                        for value, values in subset.groupby(column, dropna=False):
                            true_node = values.true_node_idx.to_numpy(int)
                            pred_node = values.pred_node_idx.to_numpy(int)
                            true_tier = values.true_tier3_id.to_numpy(int)
                            pred_tier = values.pred_tier3_id.to_numpy(int)
                            result.append({
                                "model": model,
                                "participant": participant,
                                "seed": seed,
                                "condition": condition,
                                "grouping": grouping,
                                "group": str(value),
                                "n": len(values),
                                "node_accuracy": float(np.mean(true_node == pred_node)),
                                "node_macro_f1": macro_f1(true_node, pred_node),
                                "tier3_accuracy": float(np.mean(true_tier == pred_tier)),
                                "tier3_macro_f1": macro_f1(true_tier, pred_tier),
                            })
    detailed = pd.DataFrame(result)
    detailed.to_csv(HERE / "grouped_performance_fold_seed.csv", index=False, encoding="utf-8-sig")
    pivot = detailed.pivot_table(
        index=["participant", "seed", "condition", "grouping", "group"],
        columns="model",
        values=["tier3_accuracy", "tier3_macro_f1", "node_accuracy"],
        aggfunc="first",
    )
    delta_rows = []
    for comparison in ("A1-Legacy-Once", "A3-DualPos-Once"):
        for index, row in pivot.iterrows():
            if ("tier3_accuracy", comparison) not in row.index or ("tier3_accuracy", "M2-Direct-RealOrder") not in row.index:
                continue
            participant, seed, condition, grouping, group = index
            delta_rows.append({
                "comparison_model": comparison,
                "participant": participant,
                "seed": seed,
                "condition": condition,
                "grouping": grouping,
                "group": group,
                "tier3_accuracy_delta_vs_m2": row[("tier3_accuracy", comparison)] - row[("tier3_accuracy", "M2-Direct-RealOrder")],
                "tier3_macro_f1_delta_vs_m2": row[("tier3_macro_f1", comparison)] - row[("tier3_macro_f1", "M2-Direct-RealOrder")],
                "node_accuracy_delta_vs_m2": row[("node_accuracy", comparison)] - row[("node_accuracy", "M2-Direct-RealOrder")],
            })
    delta_detail = pd.DataFrame(delta_rows)
    delta_detail.to_csv(HERE / "grouped_performance_model_deltas_fold_seed.csv", index=False, encoding="utf-8-sig")
    delta_summary = (
        delta_detail.groupby(["comparison_model", "condition", "grouping", "group"], sort=False)
        .agg(
            contributing_jobs=("tier3_accuracy_delta_vs_m2", "count"),
            tier3_accuracy_delta_mean=("tier3_accuracy_delta_vs_m2", "mean"),
            tier3_accuracy_delta_sd=("tier3_accuracy_delta_vs_m2", "std"),
            tier3_macro_f1_delta_mean=("tier3_macro_f1_delta_vs_m2", "mean"),
            node_accuracy_delta_mean=("node_accuracy_delta_vs_m2", "mean"),
        )
        .reset_index()
    )
    delta_summary.to_csv(HERE / "grouped_performance_model_deltas_summary.csv", index=False, encoding="utf-8-sig")
    summary = (
        detailed.groupby(["model", "condition", "grouping", "group"], sort=False)
        .agg(
            contributing_jobs=("tier3_accuracy", "count"),
            total_samples=("n", "sum"),
            mean_n_per_job=("n", "mean"),
            tier3_accuracy_mean=("tier3_accuracy", "mean"),
            tier3_accuracy_sd=("tier3_accuracy", "std"),
            tier3_macro_f1_mean=("tier3_macro_f1", "mean"),
            tier3_macro_f1_sd=("tier3_macro_f1", "std"),
            node_accuracy_mean=("node_accuracy", "mean"),
            node_accuracy_sd=("node_accuracy", "std"),
        )
        .reset_index()
    )
    summary.to_csv(HERE / "grouped_performance_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def normalized_kendall(order: list[int]) -> float:
    length = len(order)
    if length <= 1:
        return 0.0
    inversions = sum(order[i] > order[j] for i in range(length) for j in range(i + 1, length))
    return inversions / (length * (length - 1) / 2)


def transition_counts(rows: list[dict[str, Any]]) -> tuple[dict[int, Counter[int]], set[tuple[int, int]]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    edges: set[tuple[int, int]] = set()
    for run_rows in group_runs(rows).values():
        nodes = [int(row["node_idx"]) for row in run_rows]
        for left, right in zip(nodes, nodes[1:]):
            counts[left][right] += 1
            edges.add((left, right))
    return dict(counts), edges


def transition_log_score(nodes: list[int], counts: dict[int, Counter[int]], num_nodes: int, laplace: float = 0.5) -> float:
    if len(nodes) <= 1:
        return 0.0
    score = 0.0
    for left, right in zip(nodes, nodes[1:]):
        outgoing = counts.get(left, Counter())
        denominator = sum(outgoing.values()) + laplace * num_nodes
        score += math.log((outgoing[right] + laplace) / denominator)
    return score / (len(nodes) - 1)


def immediate_constraint_violations(nodes: list[int], immediate: dict[int, int]) -> int:
    violations = 0
    positions = {node: index for index, node in enumerate(nodes)}
    for node, previous in immediate.items():
        if node in positions and previous in positions and positions[node] != positions[previous] + 1:
            violations += 1
    return violations


def refresh_rounds(model_name: str) -> list[int]:
    return list(range(5)) if "Every10" in model_name else [0]


def run_augmentation_audit(graph: Any, graph_json: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from atomic_tail_exp.augmentation import augment_history, stable_seed
    from atomic_tail_exp.graph import is_graph_valid

    immediate = {
        int(node["node_idx"]): int(node["execution_constraints"]["must_immediately_previous_node"])
        for node in graph_json["nodes"]
        if int(node.get("action_id_tier3", -1)) >= 0
        and node.get("execution_constraints", {}).get("must_immediately_previous_node") is not None
    }
    node_names = {
        int(node["node_idx"]): str(node.get("action_label_tier3", node.get("node_id", node["node_idx"])))
        for node in graph_json["nodes"]
        if int(node.get("action_id_tier3", -1)) >= 0
    }
    records = []
    diversity_records = []
    for participant in PARTICIPANTS:
        train_rows = read_jsonl(protocol_root(participant) / "train.jsonl")
        examples = ordered_examples(train_rows)
        counts, observed_edges = transition_counts(train_rows)
        actual_prefixes = {tuple(int(row["node_idx"]) for row in history) for _, history in examples}
        for model_name in AUGMENTATION_MODELS:
            active_tail_only = model_name.startswith("A3")
            per_sample_signatures: dict[str, list[tuple[int, ...]]] = defaultdict(list)
            per_sample_changed: dict[str, list[bool]] = defaultdict(list)
            for seed in SEEDS:
                for refresh_round in refresh_rounds(model_name):
                    for current, history in examples:
                        result = augment_history(
                            history,
                            graph,
                            stable_seed(seed, refresh_round, str(current["sample_name"])),
                            active_tail_only,
                            "uniform",
                            None,
                            16,
                            0.75,
                            0.35,
                            2,
                            1,
                        )
                        actual_nodes = [int(row["node_idx"]) for row in history]
                        augmented_nodes = [int(row["node_idx"]) for row in result.rows]
                        current_node = int(current["node_idx"])
                        lookup = {str(row["sample_name"]): index for index, row in enumerate(history)}
                        order = [lookup[str(row["sample_name"])] for row in result.rows]
                        changed_positions = sum(i != value for i, value in enumerate(order))
                        mean_abs_shift = float(np.mean(np.abs(np.arange(len(order)) - np.asarray(order)))) if order else 0.0
                        augmented_edges = list(zip(augmented_nodes, augmented_nodes[1:]))
                        novel_edges = sum(edge not in observed_edges for edge in augmented_edges)
                        actual_stages = [graph.node_to_stage[node - 1] for node in actual_nodes]
                        aug_stages = [graph.node_to_stage[node - 1] for node in augmented_nodes]
                        stage_back_actual = sum(right < left for left, right in zip(actual_stages, actual_stages[1:]))
                        stage_back_aug = sum(right < left for left, right in zip(aug_stages, aug_stages[1:]))
                        target_sequence_rows = list(result.rows) + [current]
                        actual_graph_valid = is_graph_valid(history, graph)
                        actual_target_graph_valid = is_graph_valid(history + [current], graph)
                        target_graph_valid = is_graph_valid(target_sequence_rows, graph)
                        actual_target_immediate_ok = current_node not in immediate or not actual_nodes or actual_nodes[-1] == immediate[current_node]
                        target_immediate_ok = current_node not in immediate or not augmented_nodes or augmented_nodes[-1] == immediate[current_node]
                        signature = tuple(augmented_nodes)
                        sample_name = str(current["sample_name"])
                        per_sample_signatures[sample_name].append(signature)
                        per_sample_changed[sample_name].append(bool(result.changed))
                        records.append({
                            "model": model_name,
                            "participant": participant,
                            "seed": seed,
                            "refresh_round": refresh_round,
                            "sample_name": sample_name,
                            "current_node": current_node,
                            "current_action": node_names[current_node],
                            "history_length": len(history),
                            "tail_reason": result.decision.reason,
                            "tail_length": len(result.decision.node_ids),
                            "changed": bool(result.changed),
                            "kendall_distance": float(result.normalized_kendall_distance),
                            "changed_positions": changed_positions,
                            "changed_position_fraction": changed_positions / max(1, len(history)),
                            "mean_absolute_index_shift": mean_abs_shift,
                            "latest_history_token_moved": bool(order and order[-1] != len(order) - 1),
                            "actual_graph_valid": bool(actual_graph_valid),
                            "graph_valid": bool(is_graph_valid(list(result.rows), graph)),
                            "graph_valid_regressed": bool(actual_graph_valid and not is_graph_valid(list(result.rows), graph)),
                            "actual_target_appended_graph_valid": bool(actual_target_graph_valid),
                            "target_appended_graph_valid": bool(target_graph_valid),
                            "target_graph_valid_regressed": bool(actual_target_graph_valid and not target_graph_valid),
                            "immediate_constraint_violations_actual": immediate_constraint_violations(actual_nodes, immediate),
                            "immediate_constraint_violations_augmented": immediate_constraint_violations(augmented_nodes, immediate),
                            "immediate_constraint_violation_increase": immediate_constraint_violations(augmented_nodes, immediate) - immediate_constraint_violations(actual_nodes, immediate),
                            "actual_target_immediate_predecessor_ok": bool(actual_target_immediate_ok),
                            "target_immediate_predecessor_ok": bool(target_immediate_ok),
                            "target_immediate_predecessor_regressed": bool(actual_target_immediate_ok and not target_immediate_ok),
                            "augmented_transition_count": len(augmented_edges),
                            "novel_transition_count": novel_edges,
                            "novel_transition_fraction": novel_edges / max(1, len(augmented_edges)),
                            "actual_mean_transition_log_probability": transition_log_score(actual_nodes, counts, graph.num_nodes),
                            "augmented_mean_transition_log_probability": transition_log_score(augmented_nodes, counts, graph.num_nodes),
                            "stage_backward_actual": stage_back_actual,
                            "stage_backward_augmented": stage_back_aug,
                            "stage_backward_increase": stage_back_aug - stage_back_actual,
                            "augmented_signature_seen_as_real_train_prefix": signature in actual_prefixes,
                            "actual_nodes": " ".join(map(str, actual_nodes)),
                            "augmented_nodes": " ".join(map(str, augmented_nodes)),
                            "actual_actions": " -> ".join(node_names[node] for node in actual_nodes),
                            "augmented_actions": " -> ".join(node_names[node] for node in augmented_nodes),
                        })
            for sample_name, signatures in per_sample_signatures.items():
                unique = len(set(signatures))
                views = len(signatures)
                diversity_records.append({
                    "model": model_name,
                    "participant": participant,
                    "sample_name": sample_name,
                    "views": views,
                    "unique_sequences": unique,
                    "unique_ratio": unique / max(1, views),
                    "duplicate_ratio": 1.0 - unique / max(1, views),
                    "all_views_identical": unique == 1,
                    "any_changed": any(per_sample_changed[sample_name]),
                    "all_unchanged": not any(per_sample_changed[sample_name]),
                })
        print(f"augmentation audit: {participant}", flush=True)
    detailed = pd.DataFrame(records)
    diversity = pd.DataFrame(diversity_records)
    detailed.to_csv(HERE / "augmentation_history_audit_detailed.csv", index=False, encoding="utf-8-sig")
    diversity.to_csv(HERE / "augmentation_diversity_by_sample.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for analysis_scope, values in (("all_generated_views", detailed), ("changed_views_only", detailed[detailed.changed])):
        scoped_summary = (
            values.groupby(["model", "participant"], sort=False)
            .agg(
                generated_views=("sample_name", "count"),
                changed_fraction=("changed", "mean"),
                mean_kendall=("kendall_distance", "mean"),
                mean_changed_position_fraction=("changed_position_fraction", "mean"),
                mean_absolute_index_shift=("mean_absolute_index_shift", "mean"),
                latest_token_moved_fraction=("latest_history_token_moved", "mean"),
                graph_valid_fraction=("graph_valid", "mean"),
                graph_valid_regression_rate=("graph_valid_regressed", "mean"),
                target_appended_graph_valid_fraction=("target_appended_graph_valid", "mean"),
                target_graph_valid_regression_rate=("target_graph_valid_regressed", "mean"),
            target_immediate_predecessor_ok_fraction=("target_immediate_predecessor_ok", "mean"),
            target_immediate_predecessor_regression_rate=("target_immediate_predecessor_regressed", "mean"),
                immediate_constraint_violation_increase_rate=("immediate_constraint_violation_increase", lambda x: float(np.mean(np.asarray(x) > 0))),
                novel_transition_fraction=("novel_transition_fraction", "mean"),
                real_prefix_fraction=("augmented_signature_seen_as_real_train_prefix", "mean"),
                mean_actual_transition_log_probability=("actual_mean_transition_log_probability", "mean"),
                mean_augmented_transition_log_probability=("augmented_mean_transition_log_probability", "mean"),
                stage_backward_increase_rate=("stage_backward_increase", lambda x: float(np.mean(np.asarray(x) > 0))),
            )
            .reset_index()
        )
        scoped_summary.insert(2, "analysis_scope", analysis_scope)
        summaries.append(scoped_summary)
    summary = pd.concat(summaries, ignore_index=True)
    diversity_summary = (
        diversity.groupby(["model", "participant"], sort=False)
        .agg(
            samples=("sample_name", "count"),
            mean_unique_ratio=("unique_ratio", "mean"),
            mean_duplicate_ratio=("duplicate_ratio", "mean"),
            all_views_identical_fraction=("all_views_identical", "mean"),
            all_unchanged_fraction=("all_unchanged", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(diversity_summary, on=["model", "participant"], how="left")
    summary.to_csv(HERE / "augmentation_history_audit_summary.csv", index=False, encoding="utf-8-sig")
    return detailed, diversity, summary


def select_manual_candidates(detailed: pd.DataFrame) -> pd.DataFrame:
    once = detailed[(detailed.seed == 1) & (detailed.refresh_round == 0) & detailed.model.isin(["A1-Legacy-Once", "A3-DualPos-Once"])].copy()
    candidates = []
    for (model, participant), values in once.groupby(["model", "participant"]):
        changed = values[values.changed].copy()
        selections: list[tuple[str, pd.Series]] = []
        if not changed.empty:
            selections.append(("largest_order_perturbation", changed.sort_values(["kendall_distance", "changed_position_fraction"], ascending=False).iloc[0]))
            selections.append(("most_empirically_atypical", changed.sort_values(["novel_transition_fraction", "augmented_mean_transition_log_probability"], ascending=[False, True]).iloc[0]))
            plausible = changed[(changed.novel_transition_count == 0) & (changed.stage_backward_increase <= 0) & changed.target_appended_graph_valid]
            if not plausible.empty:
                selections.append(("empirically_supported_example", plausible.sort_values("kendall_distance", ascending=False).iloc[0]))
        unchanged = values[~values.changed]
        if not unchanged.empty:
            selections.append(("unchanged_or_ineligible", unchanged.sort_values("history_length", ascending=False).iloc[0]))
        seen = set()
        for category, row in selections:
            key = str(row.sample_name)
            if key in seen:
                continue
            seen.add(key)
            record = row.to_dict()
            record["selection_category"] = category
            record["manual_review_status"] = "pending_human_semantic_review"
            candidates.append(record)
    output = pd.DataFrame(candidates)
    columns = [
        "model", "participant", "sample_name", "selection_category", "current_action",
        "history_length", "tail_reason", "changed", "kendall_distance", "novel_transition_fraction",
        "target_appended_graph_valid", "target_immediate_predecessor_ok", "stage_backward_increase",
        "actual_actions", "augmented_actions", "manual_review_status",
    ]
    output[columns].to_csv(HERE / "manual_audit_candidates.csv", index=False, encoding="utf-8-sig")
    return output[columns]


def build_compact_results(
    order_summary: pd.DataFrame,
    validation: pd.DataFrame,
    grouped_summary: pd.DataFrame,
    augmentation_summary: pd.DataFrame,
    diversity: pd.DataFrame,
) -> dict[str, Any]:
    def records(df: pd.DataFrame) -> list[dict[str, Any]]:
        clean = df.replace({np.nan: None})
        return json.loads(clean.to_json(orient="records", force_ascii=False))

    selected_groups = grouped_summary[
        (grouped_summary.condition == "all")
        & grouped_summary.grouping.isin(["local_prefix_3", "history_length", "stage", "active_tail", "exact_full_prefix"])
    ]
    delta_summary = pd.read_csv(HERE / "grouped_performance_model_deltas_summary.csv")
    selected_deltas = delta_summary[
        (delta_summary.condition == "all")
        & delta_summary.grouping.isin(["local_prefix_3", "history_length", "stage", "active_tail", "exact_full_prefix"])
    ]
    aggregate_aug = (
        augmentation_summary.groupby(["model", "analysis_scope"])
        .agg({
            "changed_fraction": "mean",
            "mean_kendall": "mean",
            "mean_changed_position_fraction": "mean",
            "mean_absolute_index_shift": "mean",
            "latest_token_moved_fraction": "mean",
            "graph_valid_fraction": "mean",
            "graph_valid_regression_rate": "mean",
            "target_appended_graph_valid_fraction": "mean",
            "target_graph_valid_regression_rate": "mean",
            "target_immediate_predecessor_ok_fraction": "mean",
            "target_immediate_predecessor_regression_rate": "mean",
            "immediate_constraint_violation_increase_rate": "mean",
            "novel_transition_fraction": "mean",
            "real_prefix_fraction": "mean",
            "stage_backward_increase_rate": "mean",
            "mean_duplicate_ratio": "mean",
            "all_views_identical_fraction": "mean",
        })
        .reset_index()
    )
    result = {
        "analysis_date": "2026-08-25",
        "scope": {
            "models": list(MODELS),
            "augmentation_models": list(AUGMENTATION_MODELS),
            "participants": list(PARTICIPANTS),
            "seeds": list(SEEDS),
        },
        "reproduction_checks": records(validation),
        "order_sensitivity": records(order_summary),
        "selected_grouped_performance": records(selected_groups),
        "selected_grouped_model_deltas_vs_m2": records(selected_deltas),
        "augmentation_audit_fold_aggregate": records(augmentation_summary),
        "augmentation_audit_overall": records(aggregate_aug),
        "diversity_overall": records(diversity.groupby("model").agg(
            samples=("sample_name", "count"),
            mean_unique_ratio=("unique_ratio", "mean"),
            mean_duplicate_ratio=("duplicate_ratio", "mean"),
            all_views_identical_fraction=("all_views_identical", "mean"),
            all_unchanged_fraction=("all_unchanged", "mean"),
        ).reset_index()),
    }
    write_json(HERE / "diagnostic_results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-order", action="store_true")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    legacy_root = Path(config["paths"]["legacy_atomic_package_root"])
    if not legacy_root.exists():
        legacy_root = PACKAGE_ROOT.parent / "atomic_tail_A0_A8_windows_2026-08-19"
    sys.path.insert(0, str(legacy_root))
    from atomic_tail_exp.graph import TaskGraph

    task_graph_path = legacy_root / "assets" / "integrated_task_graph_latest.json"
    relation_path = legacy_root / "assets" / "integrated_feature_history_matrix.json"
    graph_json = read_json(task_graph_path)
    graph = TaskGraph.load(task_graph_path, relation_path)
    node_to_tier3 = list(graph.node_to_tier3)

    if args.skip_order and (HERE / "order_sensitivity_samples.csv").exists():
        order_samples = pd.read_csv(HERE / "order_sensitivity_samples.csv")
        validation = pd.read_csv(HERE / "order_sensitivity_reproduction_check.csv")
    else:
        order_samples, validation = run_order_sensitivity(graph, node_to_tier3)
    order_summary = summarize_order_sensitivity(order_samples)
    metadata = add_group_metadata(graph)
    performance_summary = grouped_performance(metadata)
    aug_detailed, diversity, aug_summary = run_augmentation_audit(graph, graph_json)
    select_manual_candidates(aug_detailed)
    build_compact_results(order_summary, validation, performance_summary, aug_summary, diversity)
    print(f"Wrote diagnostic outputs to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
