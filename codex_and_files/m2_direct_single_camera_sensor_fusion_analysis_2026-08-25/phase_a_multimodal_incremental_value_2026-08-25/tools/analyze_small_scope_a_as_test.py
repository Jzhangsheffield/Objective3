from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "phase_a.json").read_text(encoding="utf-8"))
M2_ROOT = Path(CONFIG["m2_project_root"])
TEST_KEY = "A_as_test"
SEED = 1
CONDITIONS = ["A0", "A1", "A2", "A4", "A5", "A6"]
MAIN_CONDITIONS = ["A0", "A1", "A2", "A4", "A5", "A6"]
COMPARE_CONDITIONS = ["A1", "A2", "A4", "A5", "A6"]
CONDITION_NAMES = {
    "A0": "主相机 M2-Direct",
    "A1": "第二相机单独 M2-Direct",
    "A2": "双相机 0.5/0.5 概率后融合",
    "A4": "主相机 + 右手 IMU",
    "A5": "主相机 + 右手 EMG",
    "A6": "主相机 + 右手 EMG + IMU",
}
REPORT_PATH = ROOT / "A_AS_TEST_SMALL_SCOPE_FUSION_ANALYSIS_2026-08-26.md"
ANALYSIS_DIR = ROOT / "analysis" / "a_as_test_seed_1"
LOW_RECALL_THRESHOLD = 0.80


def prediction_path(condition: str, split: str = "test_all") -> Path:
    if condition == "A0":
        return (
            M2_ROOT
            / "outputs"
            / TEST_KEY
            / f"cam_{CONFIG['primary_camera_id']}"
            / f"seed_{SEED}"
            / "history_models"
            / "direct_head_fusion"
            / "all_runs"
            / "m2_direct"
            / "test_results"
            / f"{split}_predictions.csv"
        )
    return ROOT / "outputs" / condition / TEST_KEY / f"seed_{SEED}" / "test_results" / f"{split}_predictions.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_ids: list[int]) -> dict:
    rows = []
    correct_total = int((y_true == y_pred).sum())
    for class_id in class_ids:
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        tp = int((true_mask & pred_mask).sum())
        support = int(true_mask.sum())
        predicted = int(pred_mask.sum())
        recall = tp / support if support else float("nan")
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if support and precision + recall else 0.0
        rows.append(
            {
                "class_id": class_id,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    present = [row for row in rows if row["support"] > 0]
    return {
        "n": int(len(y_true)),
        "accuracy": correct_total / len(y_true) if len(y_true) else float("nan"),
        "macro_f1": float(np.mean([row["f1"] for row in present])) if present else float("nan"),
        "macro_recall": float(np.mean([row["recall"] for row in present])) if present else float("nan"),
        "weakest_recall": min((row["recall"] for row in present), default=float("nan")),
        "per_class": rows,
    }


def metric_scalar(y_true: np.ndarray, y_pred: np.ndarray, class_ids: list[int], metric: str) -> float:
    if metric == "accuracy":
        return float(np.mean(y_true == y_pred))
    vals = []
    for class_id in class_ids:
        true_mask = y_true == class_id
        support = int(true_mask.sum())
        if not support:
            continue
        tp = int((true_mask & (y_pred == class_id)).sum())
        recall = tp / support
        if metric == "recall":
            vals.append(recall)
        elif metric == "f1":
            predicted = int((y_pred == class_id).sum())
            precision = tp / predicted if predicted else 0.0
            vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        else:
            raise ValueError(metric)
    if metric == "weakest_recall":
        return min(vals) if vals else float("nan")
    return float(np.mean(vals)) if vals else float("nan")


def pct(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100.0 * value:.{digits}f}%"


def pp(value: float, digits: int = 2, sign: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100.0 * value:+.{digits}f}" if sign else f"{100.0 * value:.{digits}f}"


def md_table(headers: list[str], rows: list[list[object]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["---"] * len(headers)
    clean = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join(aligns) + " |"]
    lines.extend("| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def pretty_node(node_id: str) -> str:
    return node_id


def bootstrap_ci(
    truth: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
    is_fault: np.ndarray,
    stages: np.ndarray,
    node_ids: list[int],
    tier3_ids: list[int],
    repetitions: int = 10000,
) -> dict:
    rng = np.random.default_rng(int(CONFIG["bootstrap_seed"]))
    strata = []
    for fault_value in (False, True):
        for stage in sorted(set(stages.tolist())):
            idx = np.where((is_fault == fault_value) & (stages == stage))[0]
            if len(idx):
                strata.append(idx)

    metric_specs = [
        ("node_accuracy", "node", "accuracy", None),
        ("node_macro_f1", "node", "f1", None),
        ("node_macro_recall", "node", "recall", None),
        ("tier3_accuracy", "tier3", "accuracy", None),
        ("tier3_macro_f1", "tier3", "f1", None),
        ("normal_node_macro_f1", "node", "f1", False),
        ("fault_node_macro_f1", "node", "f1", True),
    ]
    values = {condition: {name: np.empty(repetitions, dtype=np.float64) for name, *_ in metric_specs} for condition in COMPARE_CONDITIONS}
    for repetition in range(repetitions):
        sampled = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata])
        sampled_fault = is_fault[sampled]
        for name, level, metric, fault_filter in metric_specs:
            metric_idx = sampled if fault_filter is None else sampled[sampled_fault == fault_filter]
            classes = node_ids if level == "node" else tier3_ids
            base_value = metric_scalar(truth[level][metric_idx], predictions["A0"][level][metric_idx], classes, metric)
            for condition in COMPARE_CONDITIONS:
                candidate_value = metric_scalar(truth[level][metric_idx], predictions[condition][level][metric_idx], classes, metric)
                values[condition][name][repetition] = candidate_value - base_value

    result = {}
    for condition in COMPARE_CONDITIONS:
        result[condition] = {}
        for name, samples in values[condition].items():
            result[condition][name] = {
                "mean_delta": float(samples.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "probability_positive": float(np.mean(samples > 0)),
            }
    return result


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Counter:
    return Counter((int(t), int(p)) for t, p in zip(y_true, y_pred) if t != p)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def generate_node_impact_svg(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["A0_recall"]), int(row["class_id"])))
    methods = MAIN_CONDITIONS
    candidates = COMPARE_CONDITIONS
    cell_width = 116
    cell_height = 54
    left = 330
    right = 70
    top = 150
    panel_gap = 145
    label_height = 620
    first_height = len(methods) * cell_height
    second_top = top + first_height + panel_gap
    second_height = len(candidates) * cell_height
    width = left + len(ordered) * cell_width + right
    height = second_top + second_height + label_height
    recall_stops = [
        (0.0, (178, 24, 43)),
        (0.5, (244, 165, 130)),
        (0.8, (247, 247, 247)),
        (1.0, (26, 152, 80)),
    ]
    delta_stops = [
        (-0.5, (178, 24, 43)),
        (0.0, (247, 247, 247)),
        (0.5, (33, 102, 172)),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">35 Node 类别 Recall 与 F1 变化热图</title>',
        '<desc id="desc">上图是每个模型的绝对 Recall，下图是各候选相对 A0 的 F1 变化。类别按 A0 Recall 从低到高排列。</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#111} .title{font-size:34px;font-weight:600}.subtitle{font-size:23px;font-weight:600}.row{font-size:21px;font-weight:600}.cell{font-size:17px;font-weight:600}.axis{font-size:18px}.note{font-size:18px;fill:#333}</style>',
        '<text class="title" x="40" y="48">35 Node 类别预测影响总览（A_as_test, seed 1）</text>',
        '<text class="note" x="40" y="79">类别按 A0 Recall 从低到高排序；粗框表示 Recall &lt; 80%；单元格为百分比/百分点。</text>',
        f'<text class="subtitle" x="40" y="{top - 18}">绝对 Recall（%）：看该类别自身样本识别率</text>',
    ]

    for row_index, method in enumerate(methods):
        y = top + row_index * cell_height
        svg.append(f'<text class="row" x="{left - 18}" y="{y + 35}" text-anchor="end">{html.escape(method + "  " + CONDITION_NAMES[method])}</text>')
        for column, row in enumerate(ordered):
            recall = float(row["A0_recall"] if method == "A0" else row[f"{method}_recall"])
            fill = _interpolate_color(recall_stops, recall)
            stroke = "#111111" if recall < LOW_RECALL_THRESHOLD else "#ffffff"
            stroke_width = 3 if recall < LOW_RECALL_THRESHOLD else 1
            x = left + column * cell_width
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
            svg.append(f'<text class="cell" x="{x + cell_width / 2}" y="{y + 34}" text-anchor="middle" style="fill:{_contrast_text(fill)}">{100 * recall:.0f}</text>')

    svg.append(f'<text class="subtitle" x="40" y="{second_top - 18}">相对 A0 的 F1 变化（pp）：Recall 不变时，变化来自 Precision/误报数量</text>')
    for row_index, method in enumerate(candidates):
        y = second_top + row_index * cell_height
        svg.append(f'<text class="row" x="{left - 18}" y="{y + 35}" text-anchor="end">{html.escape(method + "  " + CONDITION_NAMES[method])}</text>')
        for column, row in enumerate(ordered):
            delta = float(row[f"{method}_delta_f1"])
            fill = _interpolate_color(delta_stops, delta)
            x = left + column * cell_width
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill}" stroke="#ffffff" stroke-width="1"/>')
            svg.append(f'<text class="cell" x="{x + cell_width / 2}" y="{y + 34}" text-anchor="middle" style="fill:{_contrast_text(fill)}">{100 * delta:+.1f}</text>')

    label_y = second_top + second_height + 28
    for column, row in enumerate(ordered):
        x = left + column * cell_width + cell_width / 2
        label = f"N{row['class_id']} {row['class_name'].split('_', 2)[-1]} (n={row['support']})"
        svg.append(f'<text class="axis" transform="translate({x},{label_y}) rotate(62)" text-anchor="start">{html.escape(label)}</text>')

    legend_y = height - 68
    legend_x = 45
    svg.append(f'<text class="note" x="{legend_x}" y="{legend_y - 12}">颜色：Recall 红=低、绿=高；ΔF1 红=下降、白≈不变、蓝=提升。粗框类别的样本索引见报告下方。</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n</svg>\n", encoding="utf-8")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def generate_node_impact_png(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["A0_recall"]), int(row["class_id"])))
    methods = MAIN_CONDITIONS
    candidates = COMPARE_CONDITIONS
    cell_width, cell_height = 112, 56
    left, right, top = 340, 60, 150
    first_height = len(methods) * cell_height
    second_top = top + first_height + 150
    second_height = len(candidates) * cell_height
    label_height = 560
    width = left + len(ordered) * cell_width + right
    height = second_top + second_height + label_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    subtitle_font = _font(23, bold=True)
    row_font = _font(20, bold=True)
    cell_font = _font(17, bold=True)
    axis_font = _font(17)
    note_font = _font(18)
    recall_stops = [
        (0.0, (178, 24, 43)),
        (0.5, (244, 165, 130)),
        (0.8, (247, 247, 247)),
        (1.0, (26, 152, 80)),
    ]
    delta_stops = [
        (-0.5, (178, 24, 43)),
        (0.0, (247, 247, 247)),
        (0.5, (33, 102, 172)),
    ]
    draw.text((40, 28), "35 Node 类别预测影响总览（A_as_test, seed 1）", fill="#111111", font=title_font)
    draw.text((40, 73), "类别按 A0 Recall 从低到高排序；粗框表示 Recall < 80%；单元格为百分比/百分点。", fill="#333333", font=note_font)
    draw.text((40, top - 33), "绝对 Recall（%）：该类别自身样本的识别率", fill="#111111", font=subtitle_font)

    for row_index, method in enumerate(methods):
        y = top + row_index * cell_height
        draw.text((left - 15, y + cell_height / 2), f"{method}  {CONDITION_NAMES[method]}", fill="#111111", font=row_font, anchor="rm")
        for column, row in enumerate(ordered):
            recall = float(row["A0_recall"] if method == "A0" else row[f"{method}_recall"])
            fill_hex = _interpolate_color(recall_stops, recall)
            x = left + column * cell_width
            outline = "#111111" if recall < LOW_RECALL_THRESHOLD else "#ffffff"
            width_outline = 3 if recall < LOW_RECALL_THRESHOLD else 1
            draw.rectangle((x, y, x + cell_width, y + cell_height), fill=fill_hex, outline=outline, width=width_outline)
            draw.text((x + cell_width / 2, y + cell_height / 2), f"{100 * recall:.0f}", fill=_contrast_text(fill_hex), font=cell_font, anchor="mm")

    draw.text((40, second_top - 34), "相对 A0 的 F1 变化（pp）：Recall 不变时，变化来自 Precision/误报数量", fill="#111111", font=subtitle_font)
    for row_index, method in enumerate(candidates):
        y = second_top + row_index * cell_height
        draw.text((left - 15, y + cell_height / 2), f"{method}  {CONDITION_NAMES[method]}", fill="#111111", font=row_font, anchor="rm")
        for column, row in enumerate(ordered):
            delta = float(row[f"{method}_delta_f1"])
            fill_hex = _interpolate_color(delta_stops, delta)
            x = left + column * cell_width
            draw.rectangle((x, y, x + cell_width, y + cell_height), fill=fill_hex, outline="#ffffff", width=1)
            draw.text((x + cell_width / 2, y + cell_height / 2), f"{100 * delta:+.1f}", fill=_contrast_text(fill_hex), font=cell_font, anchor="mm")

    label_top = second_top + second_height + 28
    for column, row in enumerate(ordered):
        label = f"N{row['class_id']} {row['class_name'].split('_', 2)[-1]} (n={row['support']})"
        label_image = Image.new("RGBA", (520, 38), (255, 255, 255, 0))
        ImageDraw.Draw(label_image).text((0, 3), label, fill="#111111", font=axis_font)
        rotated = label_image.rotate(270, expand=True)
        image.paste(rotated, (round(left + column * cell_width + cell_width / 2 - rotated.width / 2), label_top), rotated)

    draw.text((40, height - 45), "Recall：红=低、绿=高；ΔF1：红=下降、白≈不变、蓝=提升。", fill="#333333", font=note_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def main() -> None:
    protocol_path = (
        M2_ROOT
        / "outputs"
        / TEST_KEY
        / f"cam_{CONFIG['primary_camera_id']}"
        / "protocols"
        / "all_runs"
        / "test_all.jsonl"
    )
    protocol = {row["sample_name"]: row for row in read_jsonl(protocol_path)}
    raw = {condition: read_csv(prediction_path(condition)) for condition in CONDITIONS}
    by_name = {condition: {row["sample_name"]: row for row in rows} for condition, rows in raw.items()}
    sample_names = [row["sample_name"] for row in raw["A0"]]
    reference_set = set(sample_names)
    for condition in CONDITIONS:
        if set(by_name[condition]) != reference_set:
            missing = sorted(reference_set - set(by_name[condition]))
            extra = sorted(set(by_name[condition]) - reference_set)
            raise RuntimeError(f"{condition} sample mismatch; missing={missing[:5]}, extra={extra[:5]}")

    normal_names = {row["sample_name"] for row in read_csv(prediction_path("A0", "test_normal"))}
    fault_names = {row["sample_name"] for row in read_csv(prediction_path("A0", "test_fault"))}
    if normal_names | fault_names != reference_set or normal_names & fault_names:
        raise RuntimeError("Normal/Fault split does not form an exact partition of test_all")

    truth = {
        "node": np.array([int(by_name["A0"][name]["true_node_idx"]) for name in sample_names], dtype=np.int64),
        "tier3": np.array([int(by_name["A0"][name]["true_tier3_id"]) for name in sample_names], dtype=np.int64),
    }
    stages = np.array([int(by_name["A0"][name]["stage_id"]) for name in sample_names], dtype=np.int64)
    is_fault = np.array([name in fault_names for name in sample_names], dtype=bool)
    predictions = {
        condition: {
            "node": np.array([int(by_name[condition][name]["pred_node_idx"]) for name in sample_names], dtype=np.int64),
            "tier3": np.array([int(by_name[condition][name]["pred_tier3_id"]) for name in sample_names], dtype=np.int64),
        }
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        for level, true_values in truth.items():
            csv_true = np.array(
                [int(by_name[condition][name]["true_node_idx" if level == "node" else "true_tier3_id"]) for name in sample_names],
                dtype=np.int64,
            )
            if not np.array_equal(csv_true, true_values):
                raise RuntimeError(f"{condition} {level} ground truth mismatch")

    node_ids = sorted(set(truth["node"].tolist()))
    tier3_ids = sorted(set(truth["tier3"].tolist()))
    node_names = {}
    tier3_names = {}
    for name in sample_names:
        row = protocol[name]
        node_names[int(row["node_idx"])] = row["node_id"]
        tier3_names[int(row["tier3_id"])] = row["tier3"]

    masks = {"总体": np.ones(len(sample_names), dtype=bool), "Normal": ~is_fault, "Fault": is_fault}
    for stage in sorted(set(stages.tolist())):
        masks[f"Stage {stage}"] = stages == stage

    metrics = defaultdict(dict)
    for condition in CONDITIONS:
        for subset, mask in masks.items():
            metrics[condition][subset] = {
                level: class_metrics(truth[level][mask], predictions[condition][level][mask], node_ids if level == "node" else tier3_ids)
                for level in ("node", "tier3")
            }

    bootstrap = bootstrap_ci(truth, predictions, is_fault, stages, node_ids, tier3_ids)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "paired_bootstrap_exploratory.json").write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    class_exports = {}
    for level, ids, names in (("node", node_ids, node_names), ("tier3", tier3_ids, tier3_names)):
        base_rows = {row["class_id"]: row for row in metrics["A0"]["总体"][level]["per_class"]}
        rows = []
        for class_id in ids:
            base = base_rows[class_id]
            row = {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "support": base["support"],
                "A0_recall": base["recall"],
                "A0_f1": base["f1"],
                "A0_correct": base["tp"],
            }
            for condition in COMPARE_CONDITIONS:
                candidate = {r["class_id"]: r for r in metrics[condition]["总体"][level]["per_class"]}[class_id]
                row[f"{condition}_recall"] = candidate["recall"]
                row[f"{condition}_delta_recall"] = candidate["recall"] - base["recall"]
                row[f"{condition}_f1"] = candidate["f1"]
                row[f"{condition}_delta_f1"] = candidate["f1"] - base["f1"]
                row[f"{condition}_delta_correct"] = candidate["tp"] - base["tp"]
            rows.append(row)
        class_exports[level] = rows
        write_csv(ANALYSIS_DIR / f"{level}_classwise_deltas_vs_A0.csv", rows)

    impact_figure_path = ANALYSIS_DIR / "node_class_impact_heatmap.svg"
    generate_node_impact_svg(class_exports["node"], impact_figure_path)
    generate_node_impact_png(class_exports["node"], ANALYSIS_DIR / "node_class_impact_heatmap.png")

    low_recall_summary = defaultdict(list)
    low_recall_sample_rows = []
    for condition in MAIN_CONDITIONS:
        per_class = {row["class_id"]: row for row in metrics[condition]["总体"]["node"]["per_class"]}
        for class_id in node_ids:
            class_row = per_class[class_id]
            if class_row["recall"] >= LOW_RECALL_THRESHOLD:
                continue
            class_samples = [name for index, name in enumerate(sample_names) if truth["node"][index] == class_id]
            misclassified = []
            correct_names = []
            for name in class_samples:
                index = sample_names.index(name)
                predicted_id = int(predictions[condition]["node"][index])
                is_correct = predicted_id == class_id
                if is_correct:
                    correct_names.append(name)
                else:
                    misclassified.append((name, predicted_id))
                protocol_row = protocol[name]
                low_recall_sample_rows.append(
                    {
                        "condition": condition,
                        "true_node_idx": class_id,
                        "true_node_name": node_names[class_id],
                        "class_support": class_row["support"],
                        "class_recall": class_row["recall"],
                        "sample_name": name,
                        "is_correct": is_correct,
                        "pred_node_idx": predicted_id,
                        "pred_node_name": node_names[predicted_id],
                        "normal_or_fault": "Fault" if name in fault_names else "Normal",
                        "stage_id": int(protocol_row["stage_id"]),
                        "run": protocol_row["run"],
                        "annotation_row_index": protocol_row["annotation_row_index"],
                        "dataset_relative_path": f"samples/{name}",
                    }
                )
            low_recall_summary[condition].append(
                {
                    "class_id": class_id,
                    "class_name": node_names[class_id],
                    "support": class_row["support"],
                    "tp": class_row["tp"],
                    "recall": class_row["recall"],
                    "misclassified": misclassified,
                    "correct_names": correct_names,
                }
            )
    write_csv(ANALYSIS_DIR / "low_recall_node_samples.csv", low_recall_sample_rows)

    sample_index_lines = [
        "# Low-Recall Node 样本索引",
        "",
        f"> 范围：A_as_test、seed_1；低 Recall 定义为 `< {100 * LOW_RECALL_THRESHOLD:.0f}%`。每个样本均标注该方法下是否正确以及错误时的预测类别。",
        "",
    ]
    for condition in MAIN_CONDITIONS:
        sample_index_lines.extend([f"## {condition} — {CONDITION_NAMES[condition]}", ""])
        for item in low_recall_summary[condition]:
            sample_index_lines.extend(
                [
                    f"### {item['class_name']} — Recall {pct(item['recall'])} ({item['tp']}/{item['support']})",
                    "",
                    "误分类样本：",
                    "",
                ]
            )
            sample_index_lines.extend(
                f"- `{name}` → `{node_names[predicted_id]}`" for name, predicted_id in item["misclassified"]
            )
            sample_index_lines.extend(["", "正确分类样本：", ""])
            sample_index_lines.extend(f"- `{name}`" for name in item["correct_names"])
            sample_index_lines.append("")
    (ANALYSIS_DIR / "LOW_RECALL_NODE_SAMPLE_INDEX.md").write_text(
        "\n".join(sample_index_lines) + "\n", encoding="utf-8"
    )

    correction_rows = []
    base_correct = predictions["A0"]["node"] == truth["node"]
    for condition in COMPARE_CONDITIONS:
        candidate_correct = predictions[condition]["node"] == truth["node"]
        fixed = int((~base_correct & candidate_correct).sum())
        harmed = int((base_correct & ~candidate_correct).sum())
        correction_rows.append(
            {
                "condition": condition,
                "fixed_A0_errors": fixed,
                "harmed_A0_correct": harmed,
                "net_correct": fixed - harmed,
                "both_correct": int((base_correct & candidate_correct).sum()),
                "both_wrong": int((~base_correct & ~candidate_correct).sum()),
                "prediction_changed": int((predictions[condition]["node"] != predictions["A0"]["node"]).sum()),
            }
        )
    write_csv(ANALYSIS_DIR / "node_correction_flow_vs_A0.csv", correction_rows)

    # Per-true-class rescue/harm counts provide a direct interpretation of recall movement.
    rescue_harm_rows = []
    for class_id in node_ids:
        class_mask = truth["node"] == class_id
        for condition in COMPARE_CONDITIONS:
            candidate_correct = predictions[condition]["node"] == truth["node"]
            rescue_harm_rows.append(
                {
                    "node_id": class_id,
                    "node_name": node_names[class_id],
                    "support": int(class_mask.sum()),
                    "condition": condition,
                    "rescued": int((class_mask & ~base_correct & candidate_correct).sum()),
                    "harmed": int((class_mask & base_correct & ~candidate_correct).sum()),
                }
            )
    write_csv(ANALYSIS_DIR / "node_rescue_harm_by_true_class.csv", rescue_harm_rows)

    lines = []
    lines.append("# A_as_test 小范围 Phase A 融合实验整合分析")
    lines.append("")
    lines.append("> 生成日期：2026-08-26；测试范围：`A_as_test`、`seed_1`、`all_runs`。完整比较 A0、A1、A2、A4、A5、A6。")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")

    ranked = sorted(COMPARE_CONDITIONS, key=lambda c: metrics[c]["总体"]["node"]["macro_f1"], reverse=True)
    best = ranked[0]
    base_node = metrics["A0"]["总体"]["node"]
    best_node = metrics[best]["总体"]["node"]
    fault_delta = metrics[best]["Fault"]["node"]["macro_f1"] - metrics["A0"]["Fault"]["node"]["macro_f1"]
    lines.append(
        f"- 在这一次单 fold、单 seed 测试中，按总体 Node Macro-F1 排名最高的是 **{best}（{CONDITION_NAMES[best]}）**："
        f"{pct(best_node['macro_f1'])}，相对 A0 的 {pct(base_node['macro_f1'])} 为 **{pp(best_node['macro_f1'] - base_node['macro_f1'])} pp**；"
        f"其 Fault Node Macro-F1 变化为 **{pp(fault_delta)} pp**。"
    )
    for condition in COMPARE_CONDITIONS:
        node = metrics[condition]["总体"]["node"]
        tier3 = metrics[condition]["总体"]["tier3"]
        lines.append(
            f"- **{condition}**：Node Macro-F1 {pp(node['macro_f1'] - base_node['macro_f1'])} pp，"
            f"Node accuracy {pp(node['accuracy'] - base_node['accuracy'])} pp；"
            f"Tier3 Macro-F1 {pp(tier3['macro_f1'] - metrics['A0']['总体']['tier3']['macro_f1'])} pp。"
        )
    a1_node = metrics["A1"]["总体"]["node"]
    lines.append(
        f"- **A2 的增益来源是一个本身就很强且与主视角互补的第二视角**：A1 单独已达到 "
        f"{pct(a1_node['accuracy'])} accuracy / {pct(a1_node['macro_f1'])} Macro-F1，分别比 A0 "
        f"{pp(a1_node['accuracy'] - base_node['accuracy'])} / {pp(a1_node['macro_f1'] - base_node['macro_f1'])} pp；"
        f"A2 又比 A1 高 {pp(metrics['A2']['总体']['node']['accuracy'] - a1_node['accuracy'])} / "
        f"{pp(metrics['A2']['总体']['node']['macro_f1'] - a1_node['macro_f1'])} pp。"
    )
    a1_stage_deltas = {
        stage: metrics["A1"][f"Stage {stage}"]["node"]["macro_f1"] - metrics["A0"][f"Stage {stage}"]["node"]["macro_f1"]
        for stage in sorted(set(stages.tolist()))
    }
    lines.append(
        f"- **A1 单独视角的提升也具有子集差异**：Normal/Fault Node Macro-F1 相对 A0 分别为 "
        f"{pp(metrics['A1']['Normal']['node']['macro_f1'] - metrics['A0']['Normal']['node']['macro_f1'])} / "
        f"{pp(metrics['A1']['Fault']['node']['macro_f1'] - metrics['A0']['Fault']['node']['macro_f1'])} pp；"
        + "，".join(f"Stage {stage} 为 {pp(delta)} pp" for stage, delta in a1_stage_deltas.items())
        + f"；最弱类 Recall 为 {pct(a1_node['weakest_recall'])}。"
    )
    lines.append(
        f"- **A5 是当前最有希望的可穿戴条件**：总体/Fault Node Macro-F1 分别比 A0 "
        f"{pp(metrics['A5']['总体']['node']['macro_f1'] - base_node['macro_f1'])} / "
        f"{pp(metrics['A5']['Fault']['node']['macro_f1'] - metrics['A0']['Fault']['node']['macro_f1'])} pp，"
        f"最弱类 Recall 从 {pct(base_node['weakest_recall'])} 提到 {pct(metrics['A5']['总体']['node']['weakest_recall'])}；"
        f"但 Stage 1 Macro-F1 下降 {pp(metrics['A5']['Stage 1']['node']['macro_f1'] - metrics['A0']['Stage 1']['node']['macro_f1'])} pp。"
    )
    lines.append(
        f"- **A4 的信号较弱且存在指标分歧**：Fault Macro-F1 为 "
        f"{pp(metrics['A4']['Fault']['node']['macro_f1'] - metrics['A0']['Fault']['node']['macro_f1'])} pp，"
        f"满足当前以 Macro-F1 定义的单次非劣方向；但 Fault accuracy 为 "
        f"{pp(metrics['A4']['Fault']['node']['accuracy'] - metrics['A0']['Fault']['node']['accuracy'])} pp，不能概括为全面改善。"
    )
    lines.append(
        f"- **A6 没有表现出 EMG+IMU 的简单叠加收益**：总体 Macro-F1 只比 A0 "
        f"{pp(metrics['A6']['总体']['node']['macro_f1'] - base_node['macro_f1'])} pp，低于 A5，最弱类 Recall 仍为 "
        f"{pct(metrics['A6']['总体']['node']['weakest_recall'])}；Stage 3 Macro-F1 还下降 "
        f"{pp(metrics['A6']['Stage 3']['node']['macro_f1'] - metrics['A0']['Stage 3']['node']['macro_f1'])} pp。"
    )
    lines.append(
        "- 这些数字只能回答“在 A 被留作测试者且 seed=1 时有没有迹象”，尚不能回答“传感器是否稳定有价值”。"
        "原验收规则要求 12 个 fold×seed 多数正增益、最弱类 Recall 与 Node Macro-F1 同升、Fault 不退化、压力测试和硬件预算均通过。"
    )

    lines.append("")
    lines.append("## 2. 数据完整性与可比性")
    lines.append("")
    lines.append(
        f"六组预测均逐 `sample_name` 对齐到同一组 **{len(sample_names)} clips**："
        f"Normal {int((~is_fault).sum())}、Fault {int(is_fault.sum())}；"
        + "、".join(f"Stage {stage} {int((stages == stage).sum())}" for stage in sorted(set(stages.tolist())))
        + "。所有条件保存的 node/Tier3 真值完全一致。"
    )
    lines.append("")
    fallback_rows = []
    for condition in ("A4", "A5", "A6"):
        completed = json.loads((ROOT / "outputs" / condition / TEST_KEY / f"seed_{SEED}" / "completed.json").read_text(encoding="utf-8"))
        fallback_rows.append([condition, f"{completed['fallback_equivalence_max_abs_error_before_training']:.3e}", f"{completed['fallback_equivalence_max_abs_error_after_training']:.3e}"])
    lines.append(md_table(["条件", "训练前无传感器回退最大误差", "训练后无传感器回退最大误差"], fallback_rows))
    lines.append("")
    lines.append("A4–A6 的误差量级约为浮点计算误差，支持“传感器缺失时回到 A0 路径”的实现正确性；但本次目录中没有压力测试输出，尚不能证明时间偏移、部分缺失等情形的鲁棒性。")

    lines.append("")
    lines.append("## 3. 总体结果")
    lines.append("")
    overall_rows = []
    for condition in MAIN_CONDITIONS:
        node = metrics[condition]["总体"]["node"]
        tier3 = metrics[condition]["总体"]["tier3"]
        overall_rows.append([
            condition,
            CONDITION_NAMES[condition],
            pct(node["accuracy"]),
            pp(node["accuracy"] - base_node["accuracy"]) if condition != "A0" else "—",
            pct(node["macro_f1"]),
            pp(node["macro_f1"] - base_node["macro_f1"]) if condition != "A0" else "—",
            pct(node["weakest_recall"]),
            pct(tier3["accuracy"]),
            pct(tier3["macro_f1"]),
            pp(tier3["macro_f1"] - metrics["A0"]["总体"]["tier3"]["macro_f1"]) if condition != "A0" else "—",
        ])
    lines.append(md_table(
        ["条件", "输入/融合", "Node Acc", "ΔAcc pp", "Node Macro-F1", "ΔF1 pp", "最弱 Node Recall", "Tier3 Acc", "Tier3 Macro-F1", "ΔF1 pp"],
        overall_rows,
    ))
    lines.append("")
    lines.append("A1 已作为完整候选纳入后续所有子集、类别、混淆和 bootstrap 表；A2 的增益需要同时相对 A0 与 A1 判断，才能区分“第二相机本身更强”和“双视角互补”两种来源。")

    lines.append("")
    lines.append("### 3.1 A0 错误修正与新引入错误")
    lines.append("")
    correction_md = []
    for row in correction_rows:
        correction_md.append([row["condition"], row["prediction_changed"], row["fixed_A0_errors"], row["harmed_A0_correct"], f"{row['net_correct']:+d}", row["both_wrong"]])
    lines.append(md_table(["条件", "相对 A0 改变预测", "修正 A0 错误", "破坏 A0 正确", "净正确数", "两者都错"], correction_md))
    lines.append("")
    lines.append("“净正确数”直接对应 accuracy 的净变化；Macro-F1 还会受到这些修正/损害落在哪些类别以及预测精度变化的影响。")

    lines.append("")
    lines.append("## 4. Normal / Fault 与 Stage 分解")
    lines.append("")
    nf_rows = []
    for subset in ("Normal", "Fault"):
        for condition in MAIN_CONDITIONS:
            node = metrics[condition][subset]["node"]
            tier3 = metrics[condition][subset]["tier3"]
            nf_rows.append([
                subset,
                condition,
                node["n"],
                pct(node["accuracy"]),
                pp(node["accuracy"] - metrics["A0"][subset]["node"]["accuracy"]) if condition != "A0" else "—",
                pct(node["macro_f1"]),
                pp(node["macro_f1"] - metrics["A0"][subset]["node"]["macro_f1"]) if condition != "A0" else "—",
                pct(tier3["macro_f1"]),
            ])
    lines.append(md_table(["子集", "条件", "N", "Node Acc", "ΔAcc pp", "Node Macro-F1", "ΔF1 pp", "Tier3 Macro-F1"], nf_rows))
    lines.append("")
    stage_rows = []
    for stage in sorted(set(stages.tolist())):
        subset = f"Stage {stage}"
        for condition in MAIN_CONDITIONS:
            node = metrics[condition][subset]["node"]
            stage_rows.append([
                stage,
                condition,
                node["n"],
                pct(node["accuracy"]),
                pp(node["accuracy"] - metrics["A0"][subset]["node"]["accuracy"]) if condition != "A0" else "—",
                pct(node["macro_f1"]),
                pp(node["macro_f1"] - metrics["A0"][subset]["node"]["macro_f1"]) if condition != "A0" else "—",
            ])
    lines.append(md_table(["Stage", "条件", "N", "Node Acc", "ΔAcc pp", "Node Macro-F1", "ΔF1 pp"], stage_rows))

    lines.append("")
    lines.append("## 5. 类别级影响：35 Node")
    lines.append("")
    lines.append("### 5.1 类别影响总览图")
    lines.append("")
    lines.append("![35 Node Recall 与 F1 类别影响热图](analysis/a_as_test_seed_1/node_class_impact_heatmap.png)")
    lines.append("")
    lines.append(
        "上半图是各方法的绝对 Recall，类别按 A0 Recall 从低到高排列，粗框表示 Recall 低于 80%。"
        "下半图是候选方法相对 A0 的 F1 变化：蓝色为提高、红色为下降。"
        "如果上半图两个方法的 Recall 数字相同，而下半图 F1 仍有颜色变化，表示该类别正确数没有变，但其他类别误报进来的数量发生了变化，从而改变了 Precision 和 F1。"
    )
    lines.append("")
    lines.append("### 5.2 逐类别数值表")
    lines.append("")
    lines.append("下表以真实类别为行。每个候选单元格为 `Recall变化 / F1变化 / 正确数净变化`；前两项单位均为百分点。小支持度类别的一两个 clip 就会造成很大的百分点波动，应同时看 support 与正确数。")
    lines.append("")
    node_table_rows = []
    for row in class_exports["node"]:
        node_table_rows.append([
            row["class_id"], row["class_name"], row["support"], pct(row["A0_recall"], 1), pct(row["A0_f1"], 1),
            *[f"{pp(row[f'{c}_delta_recall'], 1)} / {pp(row[f'{c}_delta_f1'], 1)} / {row[f'{c}_delta_correct']:+d}" for c in COMPARE_CONDITIONS],
        ])
    lines.append(md_table(
        ["ID", "Node", "支持", "A0 R", "A0 F1"] + [f"{condition} ΔR/ΔF1/Δ正确" for condition in COMPARE_CONDITIONS],
        node_table_rows,
    ))

    lines.append("")
    lines.append("### 5.3 各融合方法最明显的 Node 增益与退化")
    lines.append("")
    for condition in COMPARE_CONDITIONS:
        sorted_gain = sorted(class_exports["node"], key=lambda r: (r[f"{condition}_delta_recall"], r["support"]), reverse=True)
        gains = [r for r in sorted_gain if r[f"{condition}_delta_recall"] > 0][:6]
        losses = [r for r in reversed(sorted_gain) if r[f"{condition}_delta_recall"] < 0][:6]
        lines.append(f"**{condition}（{CONDITION_NAMES[condition]}）**")
        lines.append("")
        lines.append("- Recall 增益最大：" + ("；".join(f"`{r['class_name']}` (n={r['support']}, {pp(r[f'{condition}_delta_recall'], 1)} pp, Δ正确={r[f'{condition}_delta_correct']:+d})" for r in gains) if gains else "无。"))
        lines.append("- Recall 退化最大：" + ("；".join(f"`{r['class_name']}` (n={r['support']}, {pp(r[f'{condition}_delta_recall'], 1)} pp, Δ正确={r[f'{condition}_delta_correct']:+d})" for r in losses) if losses else "无。"))
        improved = sum(r[f"{condition}_delta_recall"] > 0 for r in class_exports["node"])
        degraded = sum(r[f"{condition}_delta_recall"] < 0 for r in class_exports["node"])
        tied = len(class_exports["node"]) - improved - degraded
        lines.append(f"- 35 个受支持 Node 中：Recall 改善 {improved} 类、退化 {degraded} 类、不变 {tied} 类。")
        lines.append("")

    lines.append("### 5.4 各方法低 Recall Node 与误分类样本名称")
    lines.append("")
    lines.append(
        f"这里将低 Recall 预定义为 **Recall < {100 * LOW_RECALL_THRESHOLD:.0f}%**。"
        "表中列出造成低 Recall 的全部误分类样本，并给出该方法实际预测成的 Node。"
        "如需同时查看这些类别中预测正确的样本，可打开 `analysis/a_as_test_seed_1/LOW_RECALL_NODE_SAMPLE_INDEX.md`；"
        "便于筛选的逐样本表为 `analysis/a_as_test_seed_1/low_recall_node_samples.csv`。"
    )
    lines.append("")
    for condition in MAIN_CONDITIONS:
        lines.append(f"#### {condition} — {CONDITION_NAMES[condition]}")
        lines.append("")
        low_rows = []
        for item in low_recall_summary[condition]:
            errors = "<br>".join(
                f"`{name}` → `{node_names[predicted_id]}`" for name, predicted_id in item["misclassified"]
            )
            low_rows.append([
                item["class_name"],
                item["support"],
                f"{item['tp']}/{item['support']}",
                pct(item["recall"], 1),
                errors,
            ])
        lines.append(md_table(["低 Recall Node", "支持", "正确", "Recall", "误分类样本 → 预测 Node"], low_rows))
        lines.append("")

    lines.append("## 6. 类别级影响：31 Tier3")
    lines.append("")
    tier3_table_rows = []
    for row in class_exports["tier3"]:
        tier3_table_rows.append([
            row["class_id"], row["class_name"], row["support"], pct(row["A0_recall"], 1), pct(row["A0_f1"], 1),
            *[f"{pp(row[f'{c}_delta_recall'], 1)} / {pp(row[f'{c}_delta_f1'], 1)} / {row[f'{c}_delta_correct']:+d}" for c in COMPARE_CONDITIONS],
        ])
    lines.append(md_table(
        ["ID", "Tier3", "支持", "A0 R", "A0 F1"] + [f"{condition} ΔR/ΔF1/Δ正确" for condition in COMPARE_CONDITIONS],
        tier3_table_rows,
    ))

    lines.append("")
    lines.append("## 7. 混淆对变化")
    lines.append("")
    base_conf = confusion_counts(truth["node"], predictions["A0"]["node"])
    confs = {c: confusion_counts(truth["node"], predictions[c]["node"]) for c in MAIN_CONDITIONS}
    top_base = base_conf.most_common(12)
    confusion_rows = []
    for (true_id, pred_id), count in top_base:
        confusion_rows.append([
            f"{node_names[true_id]} → {node_names[pred_id]}", count,
            *[f"{confs[c][(true_id, pred_id)]} ({confs[c][(true_id, pred_id)] - count:+d})" for c in COMPARE_CONDITIONS],
        ])
    lines.append("### 7.1 A0 当前前 12 个 Node 混淆对在各融合中的变化")
    lines.append("")
    lines.append(md_table(
        ["真实 → 预测", "A0"] + [f"{condition} 数量(Δ)" for condition in COMPARE_CONDITIONS],
        confusion_rows,
    ))
    lines.append("")
    lines.append("### 7.2 各方法新引入/放大的主要混淆")
    lines.append("")
    for condition in COMPARE_CONDITIONS:
        deltas = []
        for pair in set(confs[condition]) | set(base_conf):
            delta = confs[condition][pair] - base_conf[pair]
            if delta > 0:
                deltas.append((delta, confs[condition][pair], pair))
        deltas.sort(reverse=True)
        text = "；".join(
            f"`{node_names[t]} → {node_names[p]}` {count} 次（比 A0 {delta:+d}）" for delta, count, (t, p) in deltas[:6]
        )
        lines.append(f"- **{condition}**：{text if text else '没有增加的混淆对。'}")

    lines.append("")
    lines.append("## 8. 右手 IMU 与 EMG 的互补性")
    lines.append("")
    correct = {c: predictions[c]["node"] == truth["node"] for c in ("A0", "A4", "A5", "A6")}
    comp_rows = [
        ["A4 对、A5 错", int((correct["A4"] & ~correct["A5"]).sum()), "偏向 IMU 有利样本"],
        ["A5 对、A4 错", int((correct["A5"] & ~correct["A4"]).sum()), "偏向 EMG 有利样本"],
        ["A4/A5 都错，A6 修正", int((~correct["A4"] & ~correct["A5"] & correct["A6"]).sum()), "传感器内部互补的直接证据"],
        ["A4/A5 至少一个对，A6 变错", int(((correct["A4"] | correct["A5"]) & ~correct["A6"]).sum()), "联合融合覆盖单传感器优势的代价"],
        ["A0 错且 A4/A5/A6 至少一个修正", int((~correct["A0"] & (correct["A4"] | correct["A5"] | correct["A6"])).sum()), "可穿戴信息池的潜在上限线索"],
    ]
    lines.append(md_table(["样本关系", "clips", "含义"], comp_rows))
    lines.append("")
    lines.append("A6 是否优于 A4/A5，不能只看总分；关键是它能否保留两种单传感器各自修正的类别，同时减少联合后新引入的错误。上表与第 5 节的类别净正确数应结合解读。")

    lines.append("")
    lines.append("## 9. 探索性 paired clip-level bootstrap")
    lines.append("")
    lines.append(
        f"使用 {CONFIG['bootstrap_repetitions']} 次配对 clip bootstrap；每次在 `Normal/Fault × Stage` 联合层内有放回抽样，保持各层样本量，"
        "并在同一次抽样中同时计算候选与 A0。CI 是本次 A_as_test/seed_1 测试集上的采样不确定性，不包含换测试者、换 seed 或重新训练的不确定性。"
    )
    lines.append("")
    boot_rows = []
    boot_names = [
        ("node_accuracy", "Node accuracy"),
        ("node_macro_f1", "Node Macro-F1"),
        ("node_macro_recall", "Node Macro-Recall"),
        ("tier3_macro_f1", "Tier3 Macro-F1"),
        ("normal_node_macro_f1", "Normal Node Macro-F1"),
        ("fault_node_macro_f1", "Fault Node Macro-F1"),
    ]
    for condition in COMPARE_CONDITIONS:
        for key, label in boot_names:
            item = bootstrap[condition][key]
            boot_rows.append([
                condition, label, pp(item["mean_delta"]), f"[{pp(item['ci_low'])}, {pp(item['ci_high'])}]", pct(item["probability_positive"], 1)
            ])
    lines.append(md_table(["条件", "指标", "平均 Δ pp", "95% CI (pp)", "P(Δ>0)"], boot_rows))
    lines.append("")
    lines.append("若 CI 跨 0，应视为本次测试集不足以区分候选与 A0；即便 CI 不跨 0，也仍需完成其余 11 个 fold×seed，才能判断训练稳定性与跨参与者泛化。")

    lines.append("")
    lines.append("## 10. 当前证据对 Phase A 门槛的回答")
    lines.append("")
    single_run_gate_rows = []
    joint_pass_conditions = []
    joint_fail_conditions = []
    for condition in COMPARE_CONDITIONS:
        macro_delta = metrics[condition]["总体"]["node"]["macro_f1"] - base_node["macro_f1"]
        weak_delta = metrics[condition]["总体"]["node"]["weakest_recall"] - base_node["weakest_recall"]
        fault_macro_delta = metrics[condition]["Fault"]["node"]["macro_f1"] - metrics["A0"]["Fault"]["node"]["macro_f1"]
        joint = macro_delta > 0 and weak_delta > 0
        fault_ok = fault_macro_delta >= -float(CONFIG["fault_noninferiority_margin_pp"]) / 100.0
        (joint_pass_conditions if joint else joint_fail_conditions).append(condition)
        single_run_gate_rows.append([
            condition,
            pp(macro_delta),
            pp(weak_delta),
            pp(fault_macro_delta),
            "是" if joint else "否",
            "是" if fault_ok else "否",
        ])
    lines.append("### 10.1 只针对当前一次运行的方向性检查")
    lines.append("")
    lines.append(md_table(
        ["条件", "Δ总体 Node Macro-F1 pp", "Δ最弱 Node Recall pp", "ΔFault Node Macro-F1 pp", "Macro-F1+最弱Recall同升", "Fault非劣(−0.5 pp)"],
        single_run_gate_rows,
    ))
    lines.append("")
    lines.append(
        f"{', '.join(joint_pass_conditions) if joint_pass_conditions else '没有条件'} 在这一运行中满足 Macro-F1 与最弱 Recall 同升；"
        f"{', '.join(joint_fail_conditions) if joint_fail_conditions else '没有条件'} 不满足。"
        "这不是正式通过：正式门槛要求上述方向在 12 个 fold×seed 中至少 7 个成立，并同时检查 Fault 非劣。"
    )
    lines.append("")
    lines.append("### 10.2 完整 Phase A 状态")
    lines.append("")
    gate_rows = [
        ["12 个 fold×seed 中多数正增益", "未满足/未评估", "当前只有 A_as_test × seed_1（1/12）"],
        ["Node Macro-F1 与最弱类别 Recall 同时改善", "可做单次检查", "总体表给出本次结果；仍需 12 次一致性"],
        ["Fault 不退化", "可做单次检查", "Normal/Fault 表给出本次变化；正式阈值为 -0.5 pp 非劣界"],
        ["缺失模态/时间偏差仍回退接近 A0", "部分满足", "A4-A6 无传感器回退数值等价已验证；A2 缺第二相机及各条件压力测试尚缺；A1 为第二相机单独模型，本身不具备回退 A0 的输入路径"],
        ["延迟与吞吐满足硬件预算", "未评估", "配置中的目标硬件、P95 延迟和最低吞吐预算仍为空"],
    ]
    lines.append(md_table(["门槛", "当前状态", "说明"], gate_rows))

    lines.append("")
    lines.append("## 11. 建议的下一步")
    lines.append("")
    lines.append("1. 先把 A_as_test 的 seed 2、42 补齐，观察本报告中最显著的类别增益是否换 seed 后仍存在；若类别方向反复翻转，暂不扩大到四折。")
    lines.append("2. 对 A2 增加第二相机缺失/失步测试，并对 A4/A5/A6 运行缺失模态与 ±5%、±10%、±20% 时间偏移压力测试；重点检查总体、Fault、边界相关类别及 A0 回退差距。")
    lines.append("3. 本次 A1 自身强于 A0，且 A2 进一步强于 A1，已经形成继续测试 A3 gated residual/cross-view 的充分探索性理由；固定 0.5/0.5 应保留为无调参基线，不要在测试集上搜索权重。")
    lines.append("4. 若 A4 与 A5 各自改善不同类别、而 A6 未能保留两者优势，下一版应加入按 clip/时间步的传感器 gate、modality dropout 和显式缺失标记，而不是简单同时叠加 adapter。")
    lines.append("5. 在扩展到 12 个 fold×seed 前填写目标硬件预算，并记录端到端 RGB 解码/主干/融合/后处理延迟；当前 benchmark 仅覆盖缓存特征后的融合与 head。")

    lines.append("")
    lines.append("## 12. 可复核产物")
    lines.append("")
    lines.append("- `analysis/a_as_test_seed_1/node_classwise_deltas_vs_A0.csv`：35 Node 完整类别指标与差值。")
    lines.append("- `analysis/a_as_test_seed_1/tier3_classwise_deltas_vs_A0.csv`：31 Tier3 完整类别指标与差值。")
    lines.append("- `analysis/a_as_test_seed_1/node_correction_flow_vs_A0.csv`：修正/损害/净正确数。")
    lines.append("- `analysis/a_as_test_seed_1/node_rescue_harm_by_true_class.csv`：按真实 Node 的修正与损害计数。")
    lines.append("- `analysis/a_as_test_seed_1/node_class_impact_heatmap.png`：报告内嵌的 35 Node Recall/F1 类别影响总览图。")
    lines.append("- `analysis/a_as_test_seed_1/node_class_impact_heatmap.svg`：同一图的可无限放大矢量版本。")
    lines.append("- `analysis/a_as_test_seed_1/LOW_RECALL_NODE_SAMPLE_INDEX.md`：每个方法低 Recall 类别的完整样本名，区分正确/错误。")
    lines.append("- `analysis/a_as_test_seed_1/low_recall_node_samples.csv`：低 Recall 类别逐样本明细，可按方法、类别、Normal/Fault、Stage、run 筛选。")
    lines.append("- `analysis/a_as_test_seed_1/paired_bootstrap_exploratory.json`：探索性 bootstrap 原始汇总。")
    lines.append("- 复现命令：`python tools/analyze_small_scope_a_as_test.py`。")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "analysis_dir": str(ANALYSIS_DIR), "samples": len(sample_names)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
