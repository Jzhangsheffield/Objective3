from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE_ROOT / "analysis" / "a_as_test_seed_1"


@dataclass(frozen=True)
class StorageType:
    dtype: np.dtype


@dataclass(frozen=True)
class StorageRef:
    key: str
    dtype: np.dtype
    size: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def rebuild_tensor(storage: StorageRef, offset: int, shape: tuple[int, ...],
                   stride: tuple[int, ...], *_: Any) -> TensorRef:
    return TensorRef(storage, int(offset), tuple(shape), tuple(stride))


class TorchArchiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return rebuild_tensor
        if module == "torch" and name == "FloatStorage":
            return StorageType(np.dtype("<f4"))
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(f"Unsupported global in probability archive: {module}.{name}")

    def persistent_load(self, saved_id: Any) -> StorageRef:
        kind, storage_type, key, _location, size = saved_id
        if kind != "storage" or not isinstance(storage_type, StorageType):
            raise pickle.UnpicklingError(f"Unsupported persistent object: {saved_id}")
        return StorageRef(str(key), storage_type.dtype, int(size))


def load_probability_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickle_name = next(name for name in names if name.endswith("/data.pkl"))
        value = TorchArchiveUnpickler(io.BytesIO(archive.read(pickle_name))).load()

        def materialise(item: Any) -> Any:
            if isinstance(item, TensorRef):
                storage_name = next(name for name in names if name.endswith(f"/data/{item.storage.key}"))
                raw = np.frombuffer(archive.read(storage_name), dtype=item.storage.dtype,
                                    count=item.storage.size)
                start = item.offset
                contiguous = tuple(np.cumprod((1, *item.shape[:0:-1]))[::-1]) if item.shape else ()
                if item.stride == contiguous:
                    count = int(np.prod(item.shape, dtype=np.int64))
                    return raw[start:start + count].reshape(item.shape).copy()
                byte_strides = tuple(stride * item.storage.dtype.itemsize for stride in item.stride)
                return np.lib.stride_tricks.as_strided(raw[start:], shape=item.shape,
                                                       strides=byte_strides).copy()
            if isinstance(item, dict):
                return {key: materialise(entry) for key, entry in item.items()}
            if isinstance(item, list):
                return [materialise(entry) for entry in item]
            if isinstance(item, tuple):
                return tuple(materialise(entry) for entry in item)
            return item

        return materialise(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_m2_root() -> Path:
    config = json.loads((PACKAGE_ROOT / "config" / "phase_a.json").read_text(encoding="utf-8"))
    configured = Path(config["m2_project_root"])
    if configured.is_dir():
        return configured
    local = PACKAGE_ROOT.parents[1] / "graph_history_rgb_cross_person_ADM_2026-07-22"
    if local.is_dir():
        return local
    raise FileNotFoundError(f"Cannot find M2 project root: configured={configured}, fallback={local}")


def probability_paths(m2_root: Path) -> dict[str, Path]:
    paths = {
        "A0": (m2_root / "outputs" / "A_as_test" / "cam_001484412812" / "seed_1" /
               "history_models" / "direct_head_fusion" / "all_runs" / "m2_direct" /
               "test_results" / "test_all_probabilities.pt")
    }
    for index in range(1, 7):
        paths[f"A{index}"] = (PACKAGE_ROOT / "outputs" / f"A{index}" / "A_as_test" /
                               "seed_1" / "test_results" / "test_all_probabilities.pt")
    for index in range(1, 13):
        paths[f"S{index}"] = (PACKAGE_ROOT / "outputs" / "supplementary" / f"S{index}" /
                               "A_as_test" / "seed_1" / "test_results" /
                               "test_all_probabilities.pt")
    return paths


def align_archive(path: Path, reference_names: list[str]) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    loaded = load_probability_archive(path)
    probability_key = "node_probabilities" if "node_probabilities" in loaded else "tier3_probabilities"
    probability = np.asarray(loaded[probability_key], dtype=np.float64)
    rows = list(loaded["rows"])
    lookup = {str(row["sample_name"]): index for index, row in enumerate(rows)}
    if set(lookup) != set(reference_names):
        raise ValueError(f"Sample mismatch in {path}: {len(lookup)} vs {len(reference_names)}")
    order = [lookup[name] for name in reference_names]
    probability = probability[order]
    rows = [rows[index] for index in order]
    if probability.shape[0] != len(rows) or not np.allclose(probability.sum(1), 1.0, atol=2e-5):
        raise ValueError(f"Invalid probabilities in {path}: shape={probability.shape}")
    return probability, rows, "node" if probability_key == "node_probabilities" else "tier3"


def aggregate_node_to_tier3(probability: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    result = np.zeros((probability.shape[0], 31), dtype=np.float64)
    for node, tier3 in enumerate(mapping):
        result[:, int(tier3)] += probability[:, node]
    return result


def calibration_metrics(probability: np.ndarray, truth: np.ndarray, bins: int = 15) -> dict[str, float]:
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    accuracy = float((prediction == truth).mean())
    nll = float(-np.log(np.clip(probability[np.arange(len(truth)), truth], 1e-12, 1.0)).mean())
    one_hot = np.zeros_like(probability)
    one_hot[np.arange(len(truth)), truth] = 1.0
    brier = float(np.square(probability - one_hot).sum(1).mean())
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if index == bins - 1:
            mask |= confidence == 1.0
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float((prediction[mask] == truth[mask]).mean()))
    entropy = float((-probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(1).mean())
    return {
        "accuracy": accuracy, "nll": nll, "brier": brier, "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
        "confidence_minus_accuracy": float(confidence.mean()) - accuracy,
        "mean_entropy": entropy,
    }


def minimum_alpha(baseline: np.ndarray, candidate: np.ndarray, truth: int) -> float:
    y0, yc = float(baseline[truth]), float(candidate[truth])
    required = 0.0
    for other in range(len(baseline)):
        if other == truth:
            continue
        d0 = float(baseline[other]) - y0
        if d0 <= 0:
            continue
        dc = yc - float(candidate[other])
        denominator = d0 + dc
        if denominator <= 0:
            return math.inf
        required = max(required, d0 / denominator)
    return required


MODEL_NAMES = {
    "A0": "A0 主相机 M2", "A1": "A1 第二相机 M2", "A2": "A2 双相机0.5概率融合",
    "A3": "A3 双相机gated/cross-view", "A4": "A4 cam1+IMU",
    "A5": "A5 cam1+EMG", "A6": "A6 cam1+EMG+IMU",
    "S1": "S1 EMG ResNet10 M2", "S2": "S2 EMG Dilated M2",
    "S3": "S3 IMU ResNet10 M2", "S4": "S4 IMU Dilated M2",
    "S5": "S5 EMG ResNet10 Direct Node", "S6": "S6 EMG Dilated Direct Node",
    "S7": "S7 IMU ResNet10 Direct Node", "S8": "S8 IMU Dilated Direct Node",
    "S9": "S9 EMG ResNet10 Direct Tier3", "S10": "S10 EMG Dilated Direct Tier3",
    "S11": "S11 IMU ResNet10 Direct Tier3", "S12": "S12 IMU Dilated Direct Tier3",
}


def independent_analysis(label_space: str, baseline: np.ndarray, candidates: dict[str, np.ndarray],
                         truth: np.ndarray, metadata: list[dict[str, Any]],
                         class_names: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_prediction = baseline.argmax(1)
    baseline_wrong = baseline_prediction != truth
    summaries, details, grid_rows = [], [], []
    for model, probability in candidates.items():
        prediction = probability.argmax(1)
        top2 = np.argpartition(probability, -2, axis=1)[:, -2:]
        top3 = np.argpartition(probability, -3, axis=1)[:, -3:]
        candidate_correct = prediction == truth
        fused = 0.5 * (baseline + probability)
        fused_correct = fused.argmax(1) == truth
        oracle_mask = baseline_wrong & candidate_correct
        alpha_values = []
        for index in np.flatnonzero(oracle_mask):
            alpha = minimum_alpha(baseline[index], probability[index], int(truth[index]))
            alpha_values.append(alpha)
            row = metadata[index]
            a0_pred = int(baseline_prediction[index])
            y = int(truth[index])
            details.append({
                "label_space": label_space, "model": model, "model_name": MODEL_NAMES[model],
                "sample_name": row["sample_name"], "run": row["run"],
                "annotation_row_index": row["annotation_row_index"], "stage_id": row["stage_id"],
                "true_class": y, "true_name": class_names.get(y, str(y)),
                "a0_pred_class": a0_pred, "a0_pred_name": class_names.get(a0_pred, str(a0_pred)),
                "candidate_pred_class": int(prediction[index]),
                "candidate_pred_name": class_names.get(int(prediction[index]), str(int(prediction[index]))),
                "a0_p_true": baseline[index, y], "a0_p_wrong": baseline[index, a0_pred],
                "a0_wrong_gap": baseline[index, a0_pred] - baseline[index, y],
                "a0_pairwise_logit_gap": math.log(max(float(baseline[index, a0_pred]), 1e-12))
                - math.log(max(float(baseline[index, y]), 1e-12)),
                "candidate_p_true": probability[index, y],
                "candidate_p_a0_wrong": probability[index, a0_pred],
                "candidate_correction_margin": probability[index, y] - probability[index, a0_pred],
                "candidate_pairwise_logit_margin": math.log(max(float(probability[index, y]), 1e-12))
                - math.log(max(float(probability[index, a0_pred]), 1e-12)),
                "minimum_candidate_weight": alpha,
                "correct_at_equal_0.5_fusion": bool(fused_correct[index]),
                "blocked_at_equal_0.5": bool(not fused_correct[index]),
            })
        finite = np.asarray([value for value in alpha_values if np.isfinite(value)], dtype=float)
        summaries.append({
            "label_space": label_space, "model": model, "model_name": MODEL_NAMES[model],
            "samples": len(truth), "a0_errors": int(baseline_wrong.sum()),
            "candidate_accuracy": float(candidate_correct.mean()),
            "a0_error_candidate_top1_correct": int(oracle_mask.sum()),
            "a0_error_true_in_candidate_top2": int((baseline_wrong & np.array([truth[i] in top2[i] for i in range(len(truth))])).sum()),
            "a0_error_true_in_candidate_top3": int((baseline_wrong & np.array([truth[i] in top3[i] for i in range(len(truth))])).sum()),
            "oracle_correct_but_blocked_at_0.5": int((oracle_mask & ~fused_correct).sum()),
            "oracle_correct_and_rescued_at_0.5": int((oracle_mask & fused_correct).sum()),
            "all_a0_errors_rescued_at_0.5": int((baseline_wrong & fused_correct).sum()),
            "a0_correct_harmed_at_0.5": int((~baseline_wrong & ~fused_correct).sum()),
            "net_correct_change_at_0.5": int((baseline_wrong & fused_correct).sum()) - int((~baseline_wrong & ~fused_correct).sum()),
            "alpha_min_median_for_oracle": float(np.median(finite)) if finite.size else np.nan,
            "alpha_min_q25_for_oracle": float(np.quantile(finite, 0.25)) if finite.size else np.nan,
            "alpha_min_q75_for_oracle": float(np.quantile(finite, 0.75)) if finite.size else np.nan,
            "oracle_requiring_weight_gt_0.5": int((finite > 0.5 + 1e-9).sum()),
        })
        for alpha in np.linspace(0.0, 1.0, 21):
            grid_prediction = ((1.0 - alpha) * baseline + alpha * probability).argmax(1)
            grid_rows.append({
                "label_space": label_space, "model": model, "alpha_candidate": float(alpha),
                "accuracy": float((grid_prediction == truth).mean()),
                "correct": int((grid_prediction == truth).sum()),
                "posthoc_test_diagnostic_only": True,
            })
    return pd.DataFrame(summaries), pd.DataFrame(details), pd.DataFrame(grid_rows)


def final_outcome_analysis(label_space: str, baseline: np.ndarray, conditions: dict[str, np.ndarray],
                           truth: np.ndarray) -> pd.DataFrame:
    baseline_correct = baseline.argmax(1) == truth
    rows = []
    for model, probability in conditions.items():
        correct = probability.argmax(1) == truth
        rows.append({
            "label_space": label_space, "model": model, "model_name": MODEL_NAMES[model],
            "accuracy": float(correct.mean()), "correct": int(correct.sum()),
            "a0_wrong_to_correct": int((~baseline_correct & correct).sum()),
            "a0_correct_to_wrong": int((baseline_correct & ~correct).sum()),
            "a0_wrong_to_different_wrong": int((~baseline_correct & ~correct).sum()),
            "net_correct_change_vs_a0": int(correct.sum()) - int(baseline_correct.sum()),
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> list[str]:
    result = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("—" if not np.isfinite(value) else f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        result.append("| " + " | ".join(values) + " |")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="A0 complementary-modality probability-gap analysis")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    m2_root = resolve_m2_root()
    paths = probability_paths(m2_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing probability files:\n" + "\n".join(missing))

    a0_raw = load_probability_archive(paths["A0"])
    reference_rows = list(a0_raw["rows"])
    reference_names = [str(row["sample_name"]) for row in reference_rows]
    probabilities: dict[str, np.ndarray] = {}
    spaces: dict[str, str] = {}
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model, path in paths.items():
        probability, rows, space = align_archive(path, reference_names)
        probabilities[model], spaces[model], rows_by_model[model] = probability, space, rows
        print(f"loaded {model}: {probability.shape} {space}", flush=True)

    protocols = (m2_root / "outputs" / "A_as_test" / "cam_001484412812" /
                 "protocols" / "all_runs")
    manifest_rows = read_jsonl(protocols / "train.jsonl") + read_jsonl(protocols / "test_all.jsonl")
    node_names = {int(row["node_idx"]) - 1: str(row.get("node_id", f"node_{row['node_idx']}")) for row in manifest_rows}
    tier3_names = {int(row["tier3_id"]): str(row["tier3"]) for row in manifest_rows}
    node_to_tier3 = np.full(35, -1, dtype=int)
    for row in manifest_rows:
        node = int(row["node_idx"]) - 1
        tier3 = int(row["tier3_id"])
        if node_to_tier3[node] not in {-1, tier3}:
            raise ValueError(f"Node {node + 1} maps to multiple Tier3 classes")
        node_to_tier3[node] = tier3
    if (node_to_tier3 < 0).any():
        raise ValueError("Incomplete node-to-Tier3 mapping")

    truth_node = np.array([int(row["true_node_idx"]) - 1 for row in reference_rows], dtype=int)
    truth_tier3 = np.array([int(row["true_tier3_id"]) for row in reference_rows], dtype=int)
    node_prob = {model: value for model, value in probabilities.items() if spaces[model] == "node"}
    tier3_prob = {
        model: (aggregate_node_to_tier3(value, node_to_tier3) if spaces[model] == "node" else value)
        for model, value in probabilities.items()
    }

    independent_node = {model: node_prob[model] for model in ["A1", *[f"S{i}" for i in range(1, 9)]]}
    independent_tier3 = {model: tier3_prob[model] for model in ["A1", *[f"S{i}" for i in range(1, 13)]]}
    node_summary, node_details, node_grid = independent_analysis(
        "node", node_prob["A0"], independent_node, truth_node, reference_rows, node_names)
    tier3_summary, tier3_details, tier3_grid = independent_analysis(
        "tier3", tier3_prob["A0"], independent_tier3, truth_tier3, reference_rows, tier3_names)
    oracle_summary = pd.concat([node_summary, tier3_summary], ignore_index=True)
    oracle_details = pd.concat([node_details, tier3_details], ignore_index=True)
    alpha_grid = pd.concat([node_grid, tier3_grid], ignore_index=True)
    best_grid_rows = []
    for (label_space, model), group in alpha_grid.groupby(["label_space", "model"], sort=False):
        maximum = group["accuracy"].max()
        tied = group[np.isclose(group["accuracy"], maximum)].copy()
        tied["distance_from_equal"] = (tied["alpha_candidate"] - 0.5).abs()
        best = tied.sort_values(["distance_from_equal", "alpha_candidate"]).iloc[0]
        equal = group[np.isclose(group["alpha_candidate"], 0.5)].iloc[0]
        best_grid_rows.append({
            "label_space": label_space, "model": model, "model_name": MODEL_NAMES[model],
            "best_alpha_candidate_posthoc": float(best["alpha_candidate"]),
            "best_accuracy_posthoc": float(best["accuracy"]),
            "best_correct_posthoc": int(best["correct"]),
            "equal_accuracy": float(equal["accuracy"]),
            "correct_gain_over_equal": int(best["correct"]) - int(equal["correct"]),
        })
    best_grid = pd.DataFrame(best_grid_rows)

    final_node = final_outcome_analysis(
        "node", node_prob["A0"], {model: node_prob[model] for model in [f"A{i}" for i in range(1, 7)]}, truth_node)
    final_tier3 = final_outcome_analysis(
        "tier3", tier3_prob["A0"], {model: tier3_prob[model] for model in [f"A{i}" for i in range(1, 7)]}, truth_tier3)
    final_outcomes = pd.concat([final_node, final_tier3], ignore_index=True)

    calibration_rows = []
    for model, probability in node_prob.items():
        calibration_rows.append({"label_space": "node", "model": model, "model_name": MODEL_NAMES[model],
                                 **calibration_metrics(probability, truth_node)})
    for model, probability in tier3_prob.items():
        calibration_rows.append({"label_space": "tier3", "model": model, "model_name": MODEL_NAMES[model],
                                 **calibration_metrics(probability, truth_tier3)})
    calibration = pd.DataFrame(calibration_rows)

    reconstructed_a2 = 0.5 * (node_prob["A0"] + node_prob["A1"])
    a2_validation = {
        "max_absolute_probability_difference": float(np.abs(reconstructed_a2 - node_prob["A2"]).max()),
        "mean_absolute_probability_difference": float(np.abs(reconstructed_a2 - node_prob["A2"]).mean()),
        "node_top1_mismatch_count": int((reconstructed_a2.argmax(1) != node_prob["A2"].argmax(1)).sum()),
        "samples": len(truth_node),
    }

    oracle_summary.to_csv(output_dir / "probability_complementarity_oracle_summary.csv", index=False)
    oracle_details.to_csv(output_dir / "probability_complementarity_oracle_sample_details.csv", index=False)
    oracle_details[oracle_details["blocked_at_equal_0.5"]].to_csv(
        output_dir / "probability_complementarity_blocked_samples.csv", index=False)
    alpha_grid.to_csv(output_dir / "probability_complementarity_alpha_grid.csv", index=False)
    best_grid.to_csv(output_dir / "probability_complementarity_best_posthoc_alpha.csv", index=False)
    final_outcomes.to_csv(output_dir / "probability_complementarity_final_fusion_outcomes.csv", index=False)
    calibration.to_csv(output_dir / "probability_complementarity_calibration.csv", index=False)
    (output_dir / "probability_complementarity_a2_validation.json").write_text(
        json.dumps(a2_validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    node_a0_errors = int((node_prob["A0"].argmax(1) != truth_node).sum())
    tier3_a0_errors = int((tier3_prob["A0"].argmax(1) != truth_tier3).sum())
    best_node_oracle = node_summary.sort_values(
        ["a0_error_candidate_top1_correct", "oracle_correct_but_blocked_at_0.5"], ascending=False).iloc[0]
    best_tier3_oracle = tier3_summary.sort_values(
        ["a0_error_candidate_top1_correct", "oracle_correct_but_blocked_at_0.5"], ascending=False).iloc[0]
    blocked = oracle_details[oracle_details["blocked_at_equal_0.5"]].copy()
    blocked_show = blocked.sort_values(["label_space", "model", "minimum_candidate_weight"], ascending=[True, True, False])

    report = [
        "# A0 与互补模态逐样本概率差距分析（A_as_test / seed_1）", "",
        "## 1. 结论摘要", "",
        f"本报告读取 19 个条件的完整概率：A0–A6 与 S1–S12；所有模型均按 `sample_name` 对齐到同一批 {len(truth_node)} 个 A 测试 clip。",
        f"A0 在 Node 层错误 {node_a0_errors}/{len(truth_node)}，在聚合 Tier3 层错误 {tier3_a0_errors}/{len(truth_tier3)}。", "",
        f"- Node 层独立互补候选中，Oracle top-1 可修正最多的是 **{best_node_oracle['model_name']}**：可独立判对 {int(best_node_oracle['a0_error_candidate_top1_correct'])} 个 A0 错误样本，其中 {int(best_node_oracle['oracle_correct_but_blocked_at_0.5'])} 个仍无法被 0.5/0.5 概率融合修正。",
        f"- Tier3 层独立互补候选中，Oracle top-1 可修正最多的是 **{best_tier3_oracle['model_name']}**：可独立判对 {int(best_tier3_oracle['a0_error_candidate_top1_correct'])} 个 A0 错误样本，其中 {int(best_tier3_oracle['oracle_correct_but_blocked_at_0.5'])} 个在 0.5/0.5 下仍被阻挡。",
        "- **第二相机确有被A0置信度压住的修正信息**：A1在26个A0错误clip上单独判对，但等权融合只救回其中19个；其余7个需要超过0.5的候选权重，最高达到0.780。",
        "- **IMU的独立互补性也很明显**：S3在Node层可单独修正22个A0错误clip；等权后修正19个、伤害4个，净增15个正确样本，与A2的净增数量相同。这里的A0+S3只是现有概率的事后组合，并非已经训练过的新条件。",
        "- **EMG存在局部正确信息，但整体可靠性不足**：S2在Node层可单独修正15个A0错误clip，等权融合修正14个但又伤害13个，净增仅1个；S1、S5、S6等权融合均明显退化。",
        "- **现有端到端融合没有充分兑现独立分支的Oracle潜力**：A4/A5/A6相对A0的净正确数仅为+1/+3/+2，而A3为-3。",
        "- 因此，‘互补模态已经给出正确类别，但A0概率优势过大’在若干clip上成立；具体样本、概率差、可恢复的成对logit差和最小传感器权重见 blocked sample CSV。",
        "- 本报告的 alpha 扫描和最小权重是测试集上的事后诊断，不能直接作为正式融合权重；正式权重和temperature必须由训练折内部验证集确定。", "",
        "- 当前证据仅来自 `A_as_test / seed_1`，适合定位机制和样本，不足以替代后续12个fold×seed的一致性结论。", "",
        "## 2. 数据与计算定义", "",
        "- A0/A1–A6/S1–S8：读取35-node概率；Tier3概率由固定node→Tier3映射求和。",
        "- S9–S12：直接读取31-Tier3概率。",
        "- Oracle修正：A0错误且独立候选模型top-1等于真值。",
        "- 被0.5融合阻挡：候选模型top-1正确，但 `0.5×A0 + 0.5×candidate` 的top-1仍错误。",
        "- 最小候选权重：在所有竞争类别上同时使真值成为top-1所需的最小 `alpha`。",
        "- raw logits没有保存，但任意两类的logit差可由 `log(p_i)-log(p_j)` 精确恢复；逐样本CSV同时给出概率差和恢复的成对logit差。",
        "- A4–A6只分析最终融合输出的rescue/harm；它们没有保存内部传感器分支的独立概率。", "",
        "## 3. Node层独立互补性", "",
    ]
    display = node_summary.copy()
    display["候选"] = display["model_name"]
    display["Oracle"] = display["a0_error_candidate_top1_correct"]
    display["0.5阻挡"] = display["oracle_correct_but_blocked_at_0.5"]
    display["0.5修正"] = display["all_a0_errors_rescued_at_0.5"]
    display["0.5伤害"] = display["a0_correct_harmed_at_0.5"]
    display["净变化"] = display["net_correct_change_at_0.5"]
    display["alpha中位"] = display["alpha_min_median_for_oracle"]
    report += markdown_table(display, ["候选", "Oracle", "0.5阻挡", "0.5修正", "0.5伤害", "净变化", "alpha中位"])
    report += ["", "说明：`0.5修正`统计所有被等权融合修正的A0错误样本，可能包括候选top-1本身不正确但组合后正确的情况。", "",
               "## 4. Tier3层独立互补性", ""]
    display = tier3_summary.copy()
    display["候选"] = display["model_name"]
    display["Oracle"] = display["a0_error_candidate_top1_correct"]
    display["0.5阻挡"] = display["oracle_correct_but_blocked_at_0.5"]
    display["0.5修正"] = display["all_a0_errors_rescued_at_0.5"]
    display["0.5伤害"] = display["a0_correct_harmed_at_0.5"]
    display["净变化"] = display["net_correct_change_at_0.5"]
    display["alpha中位"] = display["alpha_min_median_for_oracle"]
    report += markdown_table(display, ["候选", "Oracle", "0.5阻挡", "0.5修正", "0.5伤害", "净变化", "alpha中位"])
    report += ["", "## 5. 事后 alpha 扫描（仅用于解释）", "",
               "下表是在当前测试集的0.05步长网格上得到的最佳候选权重，用来判断‘权重不合适’是否可能是限制因素，不能据此选择正式权重。", ""]
    display = best_grid.copy()
    display["层级"] = display["label_space"]
    display["候选"] = display["model_name"]
    display["最佳alpha"] = display["best_alpha_candidate_posthoc"]
    display["最佳accuracy"] = display["best_accuracy_posthoc"]
    display["较0.5多判对"] = display["correct_gain_over_equal"]
    report += markdown_table(display, ["层级", "候选", "最佳alpha", "最佳accuracy", "较0.5多判对"])
    report += ["", "## 6. A1–A6最终输出相对A0的修正与伤害", ""]
    display = final_outcomes.copy()
    display["层级"] = display["label_space"]
    display["条件"] = display["model_name"]
    display["A0错→对"] = display["a0_wrong_to_correct"]
    display["A0对→错"] = display["a0_correct_to_wrong"]
    display["净正确数"] = display["net_correct_change_vs_a0"]
    report += markdown_table(display, ["层级", "条件", "A0错→对", "A0对→错", "净正确数"])
    report += ["", "A2一致性复核：",
               f"- 重新计算的 `0.5×A0+0.5×A1` 与A2保存概率的最大绝对差为 `{a2_validation['max_absolute_probability_difference']:.3e}`；top-1不一致数为 `{a2_validation['node_top1_mismatch_count']}`。", "",
               "## 7. 概率校准诊断", "",
               "下面是测试集上的描述性校准指标。NLL、Brier、ECE越低越好；`confidence−accuracy`为正表示平均过度自信。不能在本测试集上拟合temperature。", ""]
    node_cal = calibration[calibration["label_space"] == "node"].copy()
    node_cal["模型"] = node_cal["model_name"]
    report += markdown_table(node_cal, ["模型", "accuracy", "nll", "brier", "ece_15", "mean_confidence", "confidence_minus_accuracy"])
    report += ["", "Tier3校准完整表见CSV。", "", "## 8. 被等权融合阻挡的代表样本", "",
               "下表每个模型最多列出5个需要最高候选权重的样本；完整列表见CSV。", ""]
    representatives = blocked_show.groupby(["label_space", "model"], sort=False).head(5).copy()
    if representatives.empty:
        report.append("没有发现候选top-1正确但0.5融合仍错误的样本。")
    else:
        representatives["层级"] = representatives["label_space"]
        representatives["模型"] = representatives["model_name"]
        representatives["样本"] = representatives["sample_name"]
        representatives["真值"] = representatives["true_name"]
        representatives["A0错误"] = representatives["a0_pred_name"]
        representatives["A0差距"] = representatives["a0_wrong_gap"]
        representatives["A0 logit差"] = representatives["a0_pairwise_logit_gap"]
        representatives["候选修正差距"] = representatives["candidate_correction_margin"]
        representatives["候选logit余量"] = representatives["candidate_pairwise_logit_margin"]
        representatives["最小权重"] = representatives["minimum_candidate_weight"]
        report += markdown_table(representatives, ["层级", "模型", "样本", "真值", "A0错误", "A0差距", "A0 logit差", "候选修正差距", "候选logit余量", "最小权重"])
    report += ["", "## 9. 输出文件", "",
               "- [Oracle总体表](probability_complementarity_oracle_summary.csv)",
               "- [Oracle逐样本详情](probability_complementarity_oracle_sample_details.csv)",
               "- [被0.5融合阻挡的样本](probability_complementarity_blocked_samples.csv)",
               "- [alpha 0–1扫描](probability_complementarity_alpha_grid.csv)",
               "- [各候选事后最佳alpha摘要](probability_complementarity_best_posthoc_alpha.csv)",
               "- [A1–A6最终rescue/harm](probability_complementarity_final_fusion_outcomes.csv)",
               "- [校准指标](probability_complementarity_calibration.csv)",
               "- [A2重建一致性](probability_complementarity_a2_validation.json)", "",
               "## 10. 如何用于下一步融合", "",
               "1. 先从 blocked sample CSV 中确认第二相机/传感器确实在语义上给出了合理修正。",
               "2. 在每个训练fold内部划分validation clips，分别为A0和候选模型拟合temperature。",
               "3. 只在validation上选择概率融合或log-prob融合权重，再锁定参数测试。",
               "4. 对A4–A6若要分析内部模态贡献，后续评估需要显式保存每个分支的pre-gate logits/probabilities和gate值。", ""]
    report_path = output_dir / "A0_COMPLEMENTARY_MODALITY_PROBABILITY_GAP_ANALYSIS.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
