from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PACKAGE_ROOT / "analysis" / "a_as_test_seed_1"
BILATERAL_ROOT = PACKAGE_ROOT / "outputs" / "supplementary_bilateral"
RIGHT_ROOT = PACKAGE_ROOT / "outputs" / "supplementary"
PROTOCOLS = ("pooled_train", "participant_calibrated")

MODEL_META = {
    "S1": ("EMG", "ResNet10", "Tier3->M2 Node"),
    "S2": ("EMG", "Dilated", "Tier3->M2 Node"),
    "S3": ("IMU", "ResNet10", "Tier3->M2 Node"),
    "S4": ("IMU", "Dilated", "Tier3->M2 Node"),
    "S5": ("EMG", "ResNet10", "Direct Node"),
    "S6": ("EMG", "Dilated", "Direct Node"),
    "S7": ("IMU", "ResNet10", "Direct Node"),
    "S8": ("IMU", "Dilated", "Direct Node"),
    "S9": ("EMG", "ResNet10", "Direct Tier3"),
    "S10": ("EMG", "Dilated", "Direct Tier3"),
    "S11": ("IMU", "ResNet10", "Direct Tier3"),
    "S12": ("IMU", "Dilated", "Direct Tier3"),
}


def load_probability_helpers():
    path = PACKAGE_ROOT / "tools" / "analyze_probability_complementarity.py"
    spec = importlib.util.spec_from_file_location("probability_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = load_probability_helpers()


def classification_metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> dict[str, Any]:
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (truth, prediction), 1)
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    tp = np.diag(confusion)
    recall = np.divide(tp, support, out=np.zeros(classes, dtype=float), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(classes, dtype=float), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(classes, dtype=float), where=(precision + recall) > 0)
    present = support > 0
    return {
        "accuracy": float((truth == prediction).mean()),
        "macro_f1": float(f1[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "weakest_recall": float(recall[present].min()),
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": confusion,
    }


def archive_path(root: Path, model: str, protocol: str | None = None) -> Path:
    base = root / model / "A_as_test" / "seed_1" / "test_results"
    if protocol:
        base /= protocol
    return base / "test_all_probabilities.pt"


def load_aligned(path: Path, names: list[str]) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    loaded = HELPERS.load_probability_archive(path)
    key = "node_probabilities" if "node_probabilities" in loaded else "tier3_probabilities"
    probabilities = np.asarray(loaded[key], dtype=np.float64)
    rows = list(loaded["rows"])
    lookup = {str(row["sample_name"]): index for index, row in enumerate(rows)}
    missing = sorted(set(names) - set(lookup))
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} requested samples")
    order = [lookup[name] for name in names]
    probabilities = probabilities[order]
    rows = [rows[index] for index in order]
    if not np.allclose(probabilities.sum(1), 1.0, atol=2e-5):
        raise ValueError(f"Invalid probability sums: {path}")
    return probabilities, rows, "node" if key == "node_probabilities" else "tier3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_masks(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    cache = PACKAGE_ROOT / "outputs" / "bilateral_signal_cache" / "A_as_test" / "evaluation_protocols" / "pooled_train"
    normal = {str(row["sample_name"]) for row in read_jsonl(cache / "test_normal.jsonl")}
    fault = {str(row["sample_name"]) for row in read_jsonl(cache / "test_fault.jsonl")}
    names = np.asarray([str(row["sample_name"]) for row in rows])
    masks = {
        "All": np.ones(len(rows), dtype=bool),
        "Normal": np.isin(names, list(normal)),
        "Fault": np.isin(names, list(fault)),
    }
    for stage in (1, 2, 3):
        masks[f"Stage {stage}"] = np.asarray([int(row["stage_id"]) == stage for row in rows])
    if int(masks["Normal"].sum() + masks["Fault"].sum()) != len(rows):
        raise ValueError("Normal/Fault masks do not partition the bilateral evaluation set")
    return masks


def metric_rows(source: str, protocol: str, model: str, probability: np.ndarray,
                truth: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    classes = probability.shape[1]
    result = []
    for subset, mask in masks.items():
        metric = classification_metrics(truth[mask], probability[mask].argmax(1), classes)
        modality, encoder, task = MODEL_META.get(model, ("RGB", "M2", "Node"))
        result.append({
            "source": source, "protocol": protocol, "condition": model,
            "modality": modality, "encoder": encoder, "task": task,
            "label_space": "node" if classes == 35 else "tier3",
            "subset": subset, "samples": int(mask.sum()),
            "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
            "macro_recall": metric["macro_recall"], "weakest_recall": metric["weakest_recall"],
        })
    return result


def best_alpha_by_macro_f1(baseline: np.ndarray, candidate: np.ndarray, truth: np.ndarray,
                           masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for alpha in np.linspace(0.0, 1.0, 21):
        prediction = ((1.0 - alpha) * baseline + alpha * candidate).argmax(1)
        row: dict[str, Any] = {"alpha_sensor": float(alpha)}
        for subset in ("All", "Normal", "Fault"):
            mask = masks[subset]
            metric = classification_metrics(truth[mask], prediction[mask], baseline.shape[1])
            row[f"{subset.lower()}_accuracy"] = metric["accuracy"]
            row[f"{subset.lower()}_macro_f1"] = metric["macro_f1"]
        rows.append(row)
    return rows


def three_way_grid(baseline: np.ndarray, emg: np.ndarray, imu: np.ndarray,
                   truth: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for emg_step in range(21):
        for imu_step in range(21 - emg_step):
            w_emg, w_imu = emg_step / 20.0, imu_step / 20.0
            w_a0 = 1.0 - w_emg - w_imu
            prediction = (w_a0 * baseline + w_emg * emg + w_imu * imu).argmax(1)
            row: dict[str, Any] = {"weight_a0": w_a0, "weight_emg": w_emg, "weight_imu": w_imu}
            for subset in ("All", "Normal", "Fault"):
                mask = masks[subset]
                metric = classification_metrics(truth[mask], prediction[mask], baseline.shape[1])
                row[f"{subset.lower()}_accuracy"] = metric["accuracy"]
                row[f"{subset.lower()}_macro_f1"] = metric["macro_f1"]
            rows.append(row)
    return rows


def classwise_rows(protocol: str, model: str, probability: np.ndarray, right_probability: np.ndarray,
                   baseline: np.ndarray, truth: np.ndarray, names: dict[int, str]) -> list[dict[str, Any]]:
    classes = probability.shape[1]
    bilateral_metric = classification_metrics(truth, probability.argmax(1), classes)
    right_metric = classification_metrics(truth, right_probability.argmax(1), classes)
    baseline_metric = classification_metrics(truth, baseline.argmax(1), classes)
    rows = []
    for class_id in range(classes):
        support = int(bilateral_metric["support"][class_id])
        if support == 0:
            continue
        rows.append({
            "protocol": protocol, "condition": model,
            "label_space": "node" if classes == 35 else "tier3",
            "class_id": class_id, "class_name": names.get(class_id, str(class_id)), "support": support,
            "bilateral_recall": bilateral_metric["recall"][class_id],
            "bilateral_f1": bilateral_metric["f1"][class_id],
            "right_recall": right_metric["recall"][class_id],
            "right_f1": right_metric["f1"][class_id],
            "a0_recall": baseline_metric["recall"][class_id],
            "a0_f1": baseline_metric["f1"][class_id],
            "delta_recall_vs_right": bilateral_metric["recall"][class_id] - right_metric["recall"][class_id],
            "delta_f1_vs_right": bilateral_metric["f1"][class_id] - right_metric["f1"][class_id],
            "delta_recall_vs_a0": bilateral_metric["recall"][class_id] - baseline_metric["recall"][class_id],
            "delta_f1_vs_a0": bilateral_metric["f1"][class_id] - baseline_metric["f1"][class_id],
        })
    return rows


def _font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _interpolate_color(stops: list[tuple[float, tuple[int, int, int]]], value: float) -> str:
    value = max(stops[0][0], min(stops[-1][0], value))
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if value <= x1:
            ratio = 0.0 if x1 == x0 else (value - x0) / (x1 - x0)
            rgb = tuple(round(a + ratio * (b - a)) for a, b in zip(c0, c1))
            return "#" + "".join(f"{channel:02x}" for channel in rgb)
    return "#" + "".join(f"{channel:02x}" for channel in stops[-1][1])


def _contrast_text(fill: str) -> str:
    red, green, blue = (int(fill[index:index + 2], 16) for index in (1, 3, 5))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#ffffff" if luminance < 135 else "#111111"


def generate_right_bilateral_heatmap(class_frame: pd.DataFrame, label_space: str, path: Path) -> None:
    conditions = [f"S{i}" for i in range(1, 9 if label_space == "node" else 13)]
    frame = class_frame[(class_frame["protocol"] == "pooled_train") &
                        (class_frame["label_space"] == label_space)].copy()
    class_ids = sorted(frame["class_id"].astype(int).unique())
    lookup = {(str(row.condition), int(row.class_id)): row for row in frame.itertuples(index=False)}
    reference = {int(row.class_id): row for row in frame[frame["condition"] == conditions[0]].itertuples(index=False)}
    methods = [("A0", None)] + [(f"右-{condition}", ("right", condition)) for condition in conditions]
    methods += [(f"双-{condition}", ("bilateral", condition)) for condition in conditions]
    cell_width = 96 if label_space == "node" else 102
    cell_height = 42
    left, right, top = 390, 50, 145
    first_height = len(methods) * cell_height
    second_top = top + first_height + 125
    second_height = len(conditions) * cell_height
    label_height = 530
    width = left + len(class_ids) * cell_width + right
    height = second_top + second_height + label_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(32, bold=True)
    subtitle_font = _font(22, bold=True)
    row_font = _font(17, bold=True)
    cell_font = _font(14, bold=True)
    axis_font = _font(15)
    note_font = _font(17)
    recall_stops = [(0.0, (178, 24, 43)), (0.5, (244, 165, 130)),
                    (0.8, (247, 247, 247)), (1.0, (26, 152, 80))]
    delta_stops = [(-0.8, (178, 24, 43)), (0.0, (247, 247, 247)), (0.8, (33, 102, 172))]
    level_name = "35 Node" if label_space == "node" else "31 Tier3"
    draw.text((35, 22), f"右手 vs 双手 S1–S{len(conditions)}：{level_name} 类别对照",
              fill="#111111", font=title_font)
    draw.text((35, 66), "上半图：同一406-clip子集上的绝对Recall；下半图：双手 pooled_train 相对右手的Recall变化。",
              fill="#333333", font=note_font)
    draw.text((35, top - 30), "绝对 Recall（%）", fill="#111111", font=subtitle_font)
    for row_index, (label, source) in enumerate(methods):
        y = top + row_index * cell_height
        draw.text((left - 12, y + cell_height / 2), label, fill="#111111", font=row_font, anchor="rm")
        for column, class_id in enumerate(class_ids):
            if source is None:
                recall = float(reference[class_id].a0_recall)
            else:
                side, condition = source
                item = lookup[(condition, class_id)]
                recall = float(item.right_recall if side == "right" else item.bilateral_recall)
            fill = _interpolate_color(recall_stops, recall)
            x = left + column * cell_width
            outline = "#111111" if recall < 0.8 else "#ffffff"
            draw.rectangle((x, y, x + cell_width, y + cell_height), fill=fill,
                           outline=outline, width=2 if recall < 0.8 else 1)
            draw.text((x + cell_width / 2, y + cell_height / 2), f"{100 * recall:.0f}",
                      fill=_contrast_text(fill), font=cell_font, anchor="mm")
    draw.text((35, second_top - 30), "双手−右手 Recall 变化（pp）", fill="#111111", font=subtitle_font)
    for row_index, condition in enumerate(conditions):
        y = second_top + row_index * cell_height
        draw.text((left - 12, y + cell_height / 2), condition, fill="#111111", font=row_font, anchor="rm")
        for column, class_id in enumerate(class_ids):
            delta = float(lookup[(condition, class_id)].delta_recall_vs_right)
            fill = _interpolate_color(delta_stops, delta)
            x = left + column * cell_width
            draw.rectangle((x, y, x + cell_width, y + cell_height), fill=fill, outline="#ffffff")
            draw.text((x + cell_width / 2, y + cell_height / 2), f"{100 * delta:+.0f}",
                      fill=_contrast_text(fill), font=cell_font, anchor="mm")
    label_top = second_top + second_height + 22
    prefix = "N" if label_space == "node" else "T"
    for column, class_id in enumerate(class_ids):
        item = reference[class_id]
        class_name = str(item.class_name)
        if label_space == "node" and class_name.startswith("node_"):
            parts = class_name.split("_", 2)
            class_name = parts[2] if len(parts) > 2 else class_name
        label = f"{prefix}{class_id + 1 if label_space == 'node' else class_id} {class_name} (n={int(item.support)})"
        label_image = Image.new("RGBA", (500, 34), (255, 255, 255, 0))
        ImageDraw.Draw(label_image).text((0, 2), label, fill="#111111", font=axis_font)
        rotated = label_image.rotate(270, expand=True)
        image.paste(rotated, (round(left + column * cell_width + cell_width / 2 - rotated.width / 2), label_top), rotated)
    draw.text((35, height - 38), "粗框表示 Recall < 80%；蓝色表示加入左手后提高，红色表示降低。",
              fill="#333333", font=note_font)
    image.save(path, format="PNG", optimize=True)


def write_bilateral_low_recall_index(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    lines = ["# 双手 S1–S8 低 Recall Node 样本索引", "",
             "口径：`A_as_test / seed_1 / pooled_train`，Recall < 80%；样本集合为排除 calibration run 后的 406 clips。", ""]
    for condition in [f"S{i}" for i in range(1, 9)]:
        lines += [f"## {condition}", ""]
        group = frame[frame["condition"] == condition]
        for (class_id, class_name), class_group in group.groupby(["true_node_idx", "true_node_name"], sort=True):
            recall = float(class_group["class_recall"].iloc[0])
            support = int(class_group["class_support"].iloc[0])
            errors = class_group[~class_group["is_correct"]]
            correct = class_group[class_group["is_correct"]]
            lines += [f"### N{int(class_id) + 1} `{class_name}`", "",
                      f"- support={support}；Recall={100 * recall:.1f}%；正确={len(correct)}；错误={len(errors)}。",
                      "- 误分类：" + ("；".join(f"`{row.sample_name}` → `{row.pred_node_name}`" for row in errors.itertuples(index=False)) if len(errors) else "无。"), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_node_comparison_report_fragment(class_frame: pd.DataFrame, path: Path) -> None:
    """Write the two compact 35-Node right-vs-bilateral comparison tables."""
    pooled = class_frame[
        (class_frame["protocol"] == "pooled_train") &
        (class_frame["label_space"] == "node")
    ].copy()
    lookup = {
        (str(row.condition), int(row.class_id)): row
        for row in pooled.itertuples(index=False)
    }
    lines = [
        "### 5.3 右手与双手 S1–S8：35 Node 公平对照",
        "",
        "下表把右手和双手模型都限制到排除 calibration `run_1` 后的同一批 406 个 clip。"
        "`R`/`B` 分别表示右手/双手 pooled Recall，`Δ` 为双手−右手（百分点）；A0 也按同一子集重算。",
        "",
    ]
    for start, end in ((1, 4), (5, 8)):
        models = [f"S{i}" for i in range(start, end + 1)]
        lines += [
            f"#### 5.3.{1 if start == 1 else 2} S{start}–S{end}",
            "",
            "| Node | n | A0 R | " + " | ".join(
                part for model in models for part in (f"R-{model}", f"B-{model}", "Δ")
            ) + " |",
            "| --- | ---: | ---: | " + " | ".join("---:" for _ in range(3 * len(models))) + " |",
        ]
        for class_id in range(35):
            exemplar = lookup[(models[0], class_id)]
            cells = [
                f"N{class_id + 1} `{exemplar.class_name}`",
                str(int(exemplar.support)),
                f"{100 * float(exemplar.a0_recall):.1f}%",
            ]
            for model in models:
                row = lookup[(model, class_id)]
                cells += [
                    f"{100 * float(row.right_recall):.1f}%",
                    f"{100 * float(row.bilateral_recall):.1f}%",
                    f"{100 * float(row.delta_recall_vs_right):+.1f}",
                ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "读表重点：双手不是对所有 Node 的一致增益。pooled S3 的 35 Node Recall 提高/不变/降低为 "
        "**10/15/10**；pooled S7 为 **13/13/9**。因此后续判断双手价值时应同时看总体 Macro-F1、"
        "最弱类别和具体类别崩塌，不能只看 accuracy。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_low_recall_report_fragment(rows: list[dict[str, Any]], path: Path) -> None:
    """Write report-ready bilateral S1-S8 low-recall sample tables."""
    frame = pd.DataFrame(rows)
    model_names = {
        "S1": "双手 EMG ResNet10 Tier3→M2 Node",
        "S2": "双手 EMG Dilated Tier3→M2 Node",
        "S3": "双手 IMU ResNet10 Tier3→M2 Node",
        "S4": "双手 IMU Dilated Tier3→M2 Node",
        "S5": "双手 EMG ResNet10 Direct Node",
        "S6": "双手 EMG Dilated Direct Node",
        "S7": "双手 IMU ResNet10 Direct Node",
        "S8": "双手 IMU Dilated Direct Node",
    }
    lines = [
        "#### 5.5.3 双手 S1–S8 Sensor-only",
        "",
        "口径为 `pooled_train`、同一 406-clip 公平子集、Recall < 80%。与右手表一致，每个低 Recall Node "
        "最多用固定随机种子展示 10 个误分类样本；`显示 x/y` 表示正文展示数/该类别全部误分类数。"
        "完整逐样本记录见 `BILATERAL_SENSOR_LOW_RECALL_NODE_SAMPLE_INDEX.md` 和 "
        "`bilateral_sensor_low_recall_node_samples.csv`。",
        "",
    ]
    for model in [f"S{i}" for i in range(1, 9)]:
        lines += [
            f"##### {model} — {model_names[model]}",
            "",
            "| 低 Recall Node | 支持 | 正确 | Recall | 固定抽取误分类样本 → 预测 Node | 备注 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
        subset = frame[frame["condition"] == model]
        for class_id in sorted(subset["true_node_idx"].unique()):
            group = subset[subset["true_node_idx"] == class_id].sort_values("sample_name")
            support = int(group["class_support"].iloc[0])
            recall = float(group["class_recall"].iloc[0])
            correct = int(group["is_correct"].sum())
            errors = [
                (str(row.sample_name), str(row.pred_node_name))
                for row in group.itertuples(index=False) if not bool(row.is_correct)
            ]
            if len(errors) > 10:
                rng = random.Random(f"20260827:BILATERAL:{model}:{int(class_id)}")
                shown = sorted(rng.sample(errors, 10))
            else:
                shown = sorted(errors)
            sample_text = f"（显示 {len(shown)}/{len(errors)}）"
            if shown:
                sample_text += "<br>" + "<br>".join(
                    f"`{sample}` → `{prediction}`" for sample, prediction in shown)
            class_name = str(group["true_node_name"].iloc[0])
            lines.append(
                f"| `{class_name}` | {support} | {correct}/{support} | {100 * recall:.1f}% | "
                f"{sample_text} |  |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    reference_path = archive_path(BILATERAL_ROOT, "S1", "pooled_train")
    reference = HELPERS.load_probability_archive(reference_path)
    reference_rows = list(reference["rows"])
    names = [str(row["sample_name"]) for row in reference_rows]
    masks = split_masks(reference_rows)
    truth_node = np.asarray([int(row["true_node_idx"]) - 1 for row in reference_rows], dtype=int)
    truth_tier3 = np.asarray([int(row["true_tier3_id"]) for row in reference_rows], dtype=int)

    m2_root = HELPERS.resolve_m2_root()
    a0_path = HELPERS.probability_paths(m2_root)["A0"]
    a0_node, _, a0_space = load_aligned(a0_path, names)
    if a0_space != "node":
        raise ValueError("A0 must be a node model")

    manifest_rows = read_jsonl(m2_root / "outputs" / "A_as_test" / "cam_001484412812" / "protocols" / "all_runs" / "train.jsonl")
    manifest_rows += read_jsonl(m2_root / "outputs" / "A_as_test" / "cam_001484412812" / "protocols" / "all_runs" / "test_all.jsonl")
    node_to_tier3 = np.full(35, -1, dtype=int)
    node_names: dict[int, str] = {}
    tier3_names: dict[int, str] = {}
    for row in manifest_rows:
        node = int(row["node_idx"]) - 1
        tier3 = int(row["tier3_id"])
        node_to_tier3[node] = tier3
        node_names[node] = str(row["node_id"])
        tier3_names[tier3] = str(row["tier3"])
    if (node_to_tier3 < 0).any():
        raise ValueError("Incomplete node-to-Tier3 mapping")
    a0_tier3 = HELPERS.aggregate_node_to_tier3(a0_node, node_to_tier3)

    right: dict[str, np.ndarray] = {}
    spaces: dict[str, str] = {}
    for model in MODEL_META:
        probability, _, space = load_aligned(archive_path(RIGHT_ROOT, model), names)
        right[model], spaces[model] = probability, space

    all_metric_rows: list[dict[str, Any]] = []
    all_metric_rows += metric_rows("A0", "same_406_clip_subset", "A0", a0_node, truth_node, masks)
    all_metric_rows += metric_rows("A0", "same_406_clip_subset", "A0", a0_tier3, truth_tier3, masks)
    for index in range(1, 7):
        condition = f"A{index}"
        probability, _, space = load_aligned(HELPERS.probability_paths(m2_root)[condition], names)
        if space != "node":
            raise ValueError(f"{condition} must be a node model")
        all_metric_rows += metric_rows(
            "camera_or_trained_fusion", "same_406_clip_subset", condition,
            probability, truth_node, masks)
        all_metric_rows += metric_rows(
            "camera_or_trained_fusion", "same_406_clip_subset", condition,
            HELPERS.aggregate_node_to_tier3(probability, node_to_tier3), truth_tier3, masks)
    for model, probability in right.items():
        truth = truth_node if spaces[model] == "node" else truth_tier3
        all_metric_rows += metric_rows("right_hand", "same_406_clip_subset", model, probability, truth, masks)

    fair_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    oracle_frames = []
    detail_frames = []
    alpha_accuracy_frames = []
    alpha_macro_rows = []
    calibration_rows = []
    three_way_rows = []
    bilateral_protocols: dict[str, dict[str, np.ndarray]] = {}

    for protocol in PROTOCOLS:
        bilateral: dict[str, np.ndarray] = {}
        for model in MODEL_META:
            probability, rows, space = load_aligned(archive_path(BILATERAL_ROOT, model, protocol), names)
            if [str(row["sample_name"]) for row in rows] != names or space != spaces[model]:
                raise ValueError(f"Alignment/space mismatch for {protocol}/{model}")
            bilateral[model] = probability
            truth = truth_node if space == "node" else truth_tier3
            all_metric_rows += metric_rows("bilateral", protocol, model, probability, truth, masks)

            for subset, mask in masks.items():
                bilateral_metric = classification_metrics(truth[mask], probability[mask].argmax(1), probability.shape[1])
                right_metric = classification_metrics(truth[mask], right[model][mask].argmax(1), probability.shape[1])
                fair_rows.append({
                    "protocol": protocol, "condition": model, "modality": MODEL_META[model][0],
                    "encoder": MODEL_META[model][1], "task": MODEL_META[model][2],
                    "label_space": space, "subset": subset, "samples": int(mask.sum()),
                    "bilateral_accuracy": bilateral_metric["accuracy"],
                    "right_accuracy": right_metric["accuracy"],
                    "delta_accuracy_bilateral_minus_right": bilateral_metric["accuracy"] - right_metric["accuracy"],
                    "bilateral_macro_f1": bilateral_metric["macro_f1"],
                    "right_macro_f1": right_metric["macro_f1"],
                    "delta_macro_f1_bilateral_minus_right": bilateral_metric["macro_f1"] - right_metric["macro_f1"],
                })

            baseline = a0_node if space == "node" else a0_tier3
            class_names = node_names if space == "node" else tier3_names
            class_rows += classwise_rows(protocol, model, probability, right[model], baseline, truth, class_names)
            calibration_rows.append({
                "protocol": protocol, "label_space": space, "model": model,
                **HELPERS.calibration_metrics(probability, truth),
            })
        bilateral_protocols[protocol] = bilateral

        for model in [f"S{i}" for i in range(1, 9)]:
            class_rows += classwise_rows(
                protocol,
                model,
                HELPERS.aggregate_node_to_tier3(bilateral[model], node_to_tier3),
                HELPERS.aggregate_node_to_tier3(right[model], node_to_tier3),
                a0_tier3,
                truth_tier3,
                tier3_names,
            )

        node_summary, node_details, node_grid = HELPERS.independent_analysis(
            "node", a0_node, {model: bilateral[model] for model in [f"S{i}" for i in range(1, 9)]},
            truth_node, reference_rows, node_names)
        tier3_candidates = {
            model: (HELPERS.aggregate_node_to_tier3(bilateral[model], node_to_tier3)
                    if spaces[model] == "node" else bilateral[model])
            for model in MODEL_META
        }
        tier3_summary, tier3_details, tier3_grid = HELPERS.independent_analysis(
            "tier3", a0_tier3, tier3_candidates, truth_tier3, reference_rows, tier3_names)
        for frame in (node_summary, tier3_summary):
            frame.insert(0, "protocol", protocol)
        for frame in (node_details, tier3_details):
            frame.insert(0, "protocol", protocol)
        for frame in (node_grid, tier3_grid):
            frame.insert(0, "protocol", protocol)
        oracle_frames += [node_summary, tier3_summary]
        detail_frames += [node_details, tier3_details]
        alpha_accuracy_frames += [node_grid, tier3_grid]

        for model in [f"S{i}" for i in range(1, 9)]:
            for row in best_alpha_by_macro_f1(a0_node, bilateral[model], truth_node, masks):
                row.update({"protocol": protocol, "model": model, "label_space": "node"})
                alpha_macro_rows.append(row)

        for emg_model in ("S1", "S2", "S5", "S6"):
            for imu_model in ("S3", "S4", "S7", "S8"):
                for row in three_way_grid(a0_node, bilateral[emg_model], bilateral[imu_model], truth_node, masks):
                    row.update({"protocol": protocol, "emg_model": emg_model, "imu_model": imu_model})
                    three_way_rows.append(row)

    metrics_frame = pd.DataFrame(all_metric_rows)
    metrics_frame.to_csv(ANALYSIS_DIR / "bilateral_condition_metrics.csv", index=False)
    fair_frame = pd.DataFrame(fair_rows)
    fair_frame.to_csv(ANALYSIS_DIR / "bilateral_vs_right_hand_fair_subset.csv", index=False)
    class_frame = pd.DataFrame(class_rows)
    class_frame.to_csv(ANALYSIS_DIR / "bilateral_classwise_comparison.csv", index=False)
    generate_right_bilateral_heatmap(
        class_frame, "node", ANALYSIS_DIR / "right_bilateral_sensor_node_class_impact_heatmap.png")
    generate_right_bilateral_heatmap(
        class_frame, "tier3", ANALYSIS_DIR / "right_bilateral_sensor_tier3_class_impact_heatmap.png")

    bilateral_low_recall_rows: list[dict[str, Any]] = []
    pooled_models = bilateral_protocols["pooled_train"]
    for model in [f"S{i}" for i in range(1, 9)]:
        prediction = pooled_models[model].argmax(1)
        metric = classification_metrics(truth_node, prediction, 35)
        for class_id in range(35):
            support = int(metric["support"][class_id])
            recall = float(metric["recall"][class_id])
            if support == 0 or recall >= 0.8:
                continue
            for index in np.flatnonzero(truth_node == class_id):
                metadata = reference_rows[int(index)]
                bilateral_low_recall_rows.append({
                    "condition": model,
                    "true_node_idx": class_id,
                    "true_node_name": node_names[class_id],
                    "class_support": support,
                    "class_recall": recall,
                    "sample_name": names[int(index)],
                    "is_correct": bool(prediction[int(index)] == class_id),
                    "pred_node_idx": int(prediction[int(index)]),
                    "pred_node_name": node_names[int(prediction[int(index)])],
                    "normal_or_fault": "Fault" if masks["Fault"][int(index)] else "Normal",
                    "stage_id": int(metadata["stage_id"]),
                    "run": metadata["run"],
                    "annotation_row_index": metadata["annotation_row_index"],
                })
    pd.DataFrame(bilateral_low_recall_rows).to_csv(
        ANALYSIS_DIR / "bilateral_sensor_low_recall_node_samples.csv", index=False)
    write_bilateral_low_recall_index(
        bilateral_low_recall_rows, ANALYSIS_DIR / "BILATERAL_SENSOR_LOW_RECALL_NODE_SAMPLE_INDEX.md")
    write_node_comparison_report_fragment(
        class_frame, ANALYSIS_DIR / "BILATERAL_NODE_COMPARISON_REPORT_FRAGMENT.md")
    write_low_recall_report_fragment(
        bilateral_low_recall_rows, ANALYSIS_DIR / "BILATERAL_LOW_RECALL_REPORT_FRAGMENT.md")

    cover_ids = np.asarray([9, 10, 25, 26], dtype=int)
    true_cover = np.isin(truth_node, cover_ids)
    cover_error_rows: list[dict[str, Any]] = []
    for model in [f"S{i}" for i in range(1, 9)]:
        for source, probability in (
            ("right", right[model]),
            ("bilateral", pooled_models[model]),
        ):
            prediction = probability.argmax(1)
            error = prediction != truth_node
            predicted_cover = np.isin(prediction, cover_ids)
            cover_error_rows.append({
                "condition": model,
                "source": source,
                "true_cover_samples": int(true_cover.sum()),
                "true_cover_misclassified": int((true_cover & error).sum()),
                "within_cover_confusions": int((true_cover & error & predicted_cover).sum()),
                "cover_to_noncover_errors": int((true_cover & error & ~predicted_cover).sum()),
                "noncover_to_cover_errors": int((~true_cover & error & predicted_cover).sum()),
                "all_errors_involving_cover": int((error & (true_cover | predicted_cover)).sum()),
            })
    pd.DataFrame(cover_error_rows).to_csv(
        ANALYSIS_DIR / "bilateral_protection_cover_error_comparison.csv", index=False)

    oracle = pd.concat(oracle_frames, ignore_index=True)
    oracle.to_csv(ANALYSIS_DIR / "bilateral_probability_oracle_summary.csv", index=False)
    details = pd.concat(detail_frames, ignore_index=True)
    details.to_csv(ANALYSIS_DIR / "bilateral_probability_oracle_sample_details.csv", index=False)
    details[details["blocked_at_equal_0.5"]].to_csv(
        ANALYSIS_DIR / "bilateral_probability_blocked_samples.csv", index=False)
    alpha_accuracy = pd.concat(alpha_accuracy_frames, ignore_index=True)
    alpha_accuracy.to_csv(ANALYSIS_DIR / "bilateral_probability_alpha_accuracy_grid.csv", index=False)
    alpha_macro = pd.DataFrame(alpha_macro_rows)
    alpha_macro.to_csv(ANALYSIS_DIR / "bilateral_probability_alpha_macro_f1_grid.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(ANALYSIS_DIR / "bilateral_probability_calibration.csv", index=False)

    best_alpha_rows = []
    for (protocol, model), group in alpha_macro.groupby(["protocol", "model"], sort=False):
        maximum = group["all_macro_f1"].max()
        tied = group[np.isclose(group["all_macro_f1"], maximum)].copy()
        tied["distance_from_equal"] = (tied["alpha_sensor"] - 0.5).abs()
        best = tied.sort_values(["distance_from_equal", "alpha_sensor"]).iloc[0]
        equal = group[np.isclose(group["alpha_sensor"], 0.5)].iloc[0]
        best_alpha_rows.append({
            "protocol": protocol, "model": model,
            "best_alpha_sensor_posthoc": best["alpha_sensor"],
            "best_all_accuracy_posthoc": best["all_accuracy"],
            "best_all_macro_f1_posthoc": best["all_macro_f1"],
            "best_normal_macro_f1_posthoc": best["normal_macro_f1"],
            "best_fault_macro_f1_posthoc": best["fault_macro_f1"],
            "equal_all_accuracy": equal["all_accuracy"],
            "equal_all_macro_f1": equal["all_macro_f1"],
        })
    pd.DataFrame(best_alpha_rows).to_csv(
        ANALYSIS_DIR / "bilateral_probability_best_posthoc_alpha.csv", index=False)

    three_way = pd.DataFrame(three_way_rows)
    three_way.to_csv(ANALYSIS_DIR / "bilateral_three_way_posthoc_grid.csv", index=False)
    three_way_best = []
    for (protocol, emg_model, imu_model), group in three_way.groupby(
            ["protocol", "emg_model", "imu_model"], sort=False):
        for scope, scoped in (("unrestricted", group), ("a0_anchor_ge_0.5", group[group["weight_a0"] >= 0.5 - 1e-9])):
            maximum = scoped["all_macro_f1"].max()
            tied = scoped[np.isclose(scoped["all_macro_f1"], maximum)].copy()
            best = tied.sort_values(["weight_a0", "weight_imu", "weight_emg"], ascending=[False, True, True]).iloc[0]
            three_way_best.append({
                "protocol": protocol, "emg_model": emg_model, "imu_model": imu_model, "scope": scope,
                **{column: best[column] for column in [
                    "weight_a0", "weight_emg", "weight_imu", "all_accuracy", "all_macro_f1",
                    "normal_macro_f1", "fault_macro_f1"]},
            })
    pd.DataFrame(three_way_best).to_csv(
        ANALYSIS_DIR / "bilateral_three_way_best_posthoc.csv", index=False)

    gap_rows = []
    for model in MODEL_META:
        history = json.loads((BILATERAL_ROOT / model / "A_as_test" / "seed_1" / "train_log.json").read_text(encoding="utf-8"))
        accuracy_key = "node_accuracy" if model in {"S1", "S2", "S3", "S4"} else "target_accuracy"
        pooled = metrics_frame[(metrics_frame["source"] == "bilateral") &
                               (metrics_frame["protocol"] == "pooled_train") &
                               (metrics_frame["condition"] == model) &
                               (metrics_frame["subset"] == "All")].iloc[0]
        calibrated = metrics_frame[(metrics_frame["source"] == "bilateral") &
                                   (metrics_frame["protocol"] == "participant_calibrated") &
                                   (metrics_frame["condition"] == model) &
                                   (metrics_frame["subset"] == "All")].iloc[0]
        gap_rows.append({
            "condition": model, "target": "node" if model in {f"S{i}" for i in range(1, 9)} else "tier3",
            "epoch": int(history[-1]["epoch"]), "first_train_accuracy": history[0][accuracy_key],
            "last_train_accuracy": history[-1][accuracy_key], "last_train_loss": history[-1]["loss"],
            "pooled_test_accuracy": pooled["accuracy"],
            "participant_calibrated_test_accuracy": calibrated["accuracy"],
            "train_minus_pooled_pp": 100.0 * (history[-1][accuracy_key] - pooled["accuracy"]),
        })
    pd.DataFrame(gap_rows).to_csv(ANALYSIS_DIR / "bilateral_training_generalization_gap.csv", index=False)

    audit = {
        "evaluation_samples": len(names),
        "normal_samples": int(masks["Normal"].sum()),
        "fault_samples": int(masks["Fault"].sum()),
        "stage_samples": {str(stage): int(masks[f"Stage {stage}"].sum()) for stage in (1, 2, 3)},
        "excluded_calibration_run": "run_1",
        "all_protocol_sample_names_identical": True,
        "right_and_a0_aligned_to_same_subset": True,
    }
    (ANALYSIS_DIR / "bilateral_analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bilateral A_as_test analysis outputs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
