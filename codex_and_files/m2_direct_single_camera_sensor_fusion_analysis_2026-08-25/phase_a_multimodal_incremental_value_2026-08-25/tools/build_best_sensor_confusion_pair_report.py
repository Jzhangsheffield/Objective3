from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.build_low_recall_multimodal_pilot import (
    CAMERAS,
    CONDITION_NAMES,
    DATASET_ROOT,
    MANIFEST_PATH,
    RAW_RGB_ROOT,
    font,
    read_jsonl,
)
from tools.build_low_recall_run_multimodal_pilot import (
    RAW_XDF_ROOT,
    frame_epoch,
    format_clock,
    moving_abs_envelope,
    robust_limits,
    windows_explorer_link,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "a_as_test_seed_1"
MASTER_REPORT = ANALYSIS_DIR / "BEST_SENSOR_MODELS_CONFUSION_PAIR_SIGNAL_ANALYSIS.md"
REPORT_DIR = ANALYSIS_DIR / "pair_reports"
ASSET_DIR = ANALYSIS_DIR / "pair_assets"
SELECTION_JSON = ASSET_DIR / "selection_manifest.json"
SEGMENT_LOG = DATASET_ROOT / "mindrove_seg_log.csv"
PROTOCOL_PATH = (
    ROOT.parents[1]
    / "graph_history_rgb_cross_person_ADM_2026-07-22"
    / "outputs"
    / "A_as_test"
    / "cam_001484412812"
    / "protocols"
    / "all_runs"
    / "test_all.jsonl"
)

LOW_RECALL_THRESHOLD = 0.80
MAX_TARGETS_PER_MODEL = 10
PRIMARY_CAMERA = "001484412812"
SECONDARY_CAMERA = "001528512812"
WAVE_COLOR = "#5b9fa3"
ENVELOPE_COLOR = "#7c3aed"
IMU_COLOR = "#5f86b3"
RGB_COLOR = "#d97706"
TARGET_FILL = "#fee2e2"
REFERENCE_FILL = "#dbeafe"
GRID_COLOR = "#d1d5db"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#4b5563"


MODEL_GROUPS = (
    {"group": "M2-Direct Node", "modality": "EMG", "level": "node", "candidates": ("S1", "S2")},
    {"group": "M2-Direct Node", "modality": "IMU", "level": "node", "candidates": ("S3", "S4")},
    {"group": "Direct Node", "modality": "EMG", "level": "node", "candidates": ("S5", "S6")},
    {"group": "Direct Node", "modality": "IMU", "level": "node", "candidates": ("S7", "S8")},
    {"group": "Direct Tier3", "modality": "EMG", "level": "tier3", "candidates": ("S9", "S10")},
    {"group": "Direct Tier3", "modality": "IMU", "level": "tier3", "candidates": ("S11", "S12")},
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_path(condition: str) -> Path:
    return ROOT / "outputs" / "supplementary" / condition / "A_as_test" / "seed_1" / "test_results" / "test_all_metrics.json"


def prediction_path(condition: str) -> Path:
    return ROOT / "outputs" / "supplementary" / condition / "A_as_test" / "seed_1" / "test_results" / "test_all_predictions.csv"


def confidence(row: dict[str, str], level: str) -> float:
    key = "node_confidence" if level == "node" else "tier3_confidence"
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def label_columns(level: str) -> tuple[str, str]:
    return ("true_node_idx", "pred_node_idx") if level == "node" else ("true_tier3_id", "pred_tier3_id")


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def load_run_signal(path: Path, cache: dict[Path, dict]) -> dict:
    path = path.resolve()
    if path in cache:
        return cache[path]
    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().strip().split(",")
    matrix = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    columns = {name: index for index, name in enumerate(header)}
    board_ts = matrix[:, columns["board_ts"]]
    result = {
        "board_ts": board_ts,
        "emg": matrix[:, [columns[f"emg{index}"] for index in range(1, 9)]].astype(np.float32),
        "imu": matrix[:, [columns[name] for name in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")]].astype(np.float32),
        "duration": float(board_ts[-1] - board_ts[0]),
        "start": float(board_ts[0]),
        "end": float(board_ts[-1]),
        "path": path,
    }
    result["emg_envelope"] = moving_abs_envelope(
        result["emg"], window=max(1, round(0.100 * len(board_ts) / max(result["duration"], 1e-9)))
    )
    cache[path] = result
    return result


def select_winners(metrics: dict[str, dict]) -> list[dict]:
    winners = []
    for group in MODEL_GROUPS:
        level = group["level"]
        ranked = sorted(
            group["candidates"],
            key=lambda condition: (
                metrics[condition][level]["macro_f1"],
                metrics[condition][level]["weakest_class_recall"],
                metrics[condition][level]["accuracy"],
            ),
            reverse=True,
        )
        winner = dict(group)
        winner["condition"] = ranked[0]
        winner["runner_up"] = ranked[1]
        winners.append(winner)
    return winners


def sample_duration(sample_name: str, protocol: dict[str, dict]) -> float:
    row = protocol[sample_name]
    return float(row["mindrove_right_end_board_ts"]) - float(row["mindrove_right_start_board_ts"])


def choose_targets(
    condition: str,
    level: str,
    rows: list[dict[str, str]],
    metrics: dict,
) -> list[dict[str, str]]:
    true_key, pred_key = label_columns(level)
    # Supplementary node metrics store class_id as the zero-based classifier
    # position, while prediction CSVs/protocol rows retain node_idx 1..35.
    node_offset = 1 if level == "node" else 0
    recalls = {int(item["class_id"]) + node_offset: float(item["recall"]) for item in metrics[level]["per_class"]}
    errors = [
        row for row in rows
        if int(row[true_key]) != int(row[pred_key]) and recalls[int(row[true_key])] < LOW_RECALL_THRESHOLD
    ]
    pair_counts = Counter((int(row[true_key]), int(row[pred_key])) for row in errors)

    def row_key(row: dict[str, str]) -> tuple:
        true_id, pred_id = int(row[true_key]), int(row[pred_key])
        conf = confidence(row, level)
        return (
            recalls[true_id],
            -pair_counts[(true_id, pred_id)],
            -(conf if math.isfinite(conf) else -1.0),
            row["sample_name"],
        )

    ordered = sorted(errors, key=row_key)
    selected: list[dict[str, str]] = []
    used_samples: set[str] = set()
    used_true: set[int] = set()
    used_pairs: set[tuple[int, int]] = set()

    for pass_name in ("true", "pair", "fill"):
        for row in ordered:
            if len(selected) >= MAX_TARGETS_PER_MODEL:
                break
            if row["sample_name"] in used_samples:
                continue
            true_id, pred_id = int(row[true_key]), int(row[pred_key])
            pair = (true_id, pred_id)
            if pass_name == "true" and true_id in used_true:
                continue
            if pass_name == "pair" and pair in used_pairs:
                continue
            selected.append(row)
            used_samples.add(row["sample_name"])
            used_true.add(true_id)
            used_pairs.add(pair)
    return selected


def choose_reference(
    target: dict[str, str],
    rows: list[dict[str, str]],
    level: str,
    protocol: dict[str, dict],
    used_references: set[str],
) -> tuple[dict[str, str], str]:
    true_key, pred_key = label_columns(level)
    predicted_id = int(target[pred_key])
    pool = [row for row in rows if int(row[true_key]) == predicted_id and row["sample_name"] != target["sample_name"]]
    if not pool:
        raise RuntimeError(f"No reference class sample for {target['sample_name']} predicted={predicted_id}")
    correct = [row for row in pool if int(row[pred_key]) == predicted_id]
    base = correct if correct else pool
    target_duration = sample_duration(target["sample_name"], protocol)
    target_run = target["run"]

    def ref_key(row: dict[str, str]) -> tuple:
        conf = confidence(row, level)
        return (
            row["sample_name"] in used_references,
            row["run"] != target_run,
            abs(sample_duration(row["sample_name"], protocol) - target_duration),
            -(conf if math.isfinite(conf) else -1.0),
            row["sample_name"],
        )

    reference = min(base, key=ref_key)
    status = "同模型正确识别的预测类别参考" if correct else "预测类别无正确样本；使用该真实类别的最近参考"
    used_references.add(reference["sample_name"])
    return reference, status


def make_clip_info(
    sample_name: str,
    protocol: dict[str, dict],
    segment_rows: dict[str, dict[str, str]],
    cache: dict[str, dict],
) -> dict:
    if sample_name in cache:
        return cache[sample_name]
    row = protocol[sample_name]
    seg = segment_rows[sample_name]
    sensor_start = float(row["mindrove_right_start_board_ts"])
    sensor_end = float(row["mindrove_right_end_board_ts"])
    cameras = {}
    for camera_id in (PRIMARY_CAMERA, SECONDARY_CAMERA):
        directory = RAW_RGB_ROOT / sample_name / camera_id
        files = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise FileNotFoundError(f"No RGB frames: {directory}")
        cameras[camera_id] = {
            "dir": directory,
            "first": files[0],
            "last": files[-1],
            "count": len(files),
            "start": frame_epoch(files[0], seg["annotation_start_raw"], float(seg["annotation_start_sec"])),
            "end": frame_epoch(files[-1], seg["annotation_start_raw"], float(seg["annotation_start_sec"])),
        }
    primary = cameras[PRIMARY_CAMERA]
    result = {
        "sample_name": sample_name,
        "run": row["run"],
        "run_number": int(str(row["run"]).split("_")[-1]),
        "annotation_row_index": int(row["annotation_row_index"]),
        "tier3": row["tier3"],
        "node_id": row["node_id"],
        "sensor_start": sensor_start,
        "sensor_end": sensor_end,
        "rgb_start": primary["start"],
        "rgb_end": primary["end"],
        "cameras": cameras,
        "right_csv": Path(seg["right_csv"]),
        "segment_pt": Path(seg["output_pt"]),
    }
    cache[sample_name] = result
    return result


def x_map(value: float, domain_start: float, domain_end: float, left: int, right: int) -> int:
    return round(left + (value - domain_start) / max(domain_end - domain_start, 1e-9) * (right - left))


def y_map(value: float, low: float, high: float, top: int, bottom: int) -> int:
    clipped = float(np.clip(value, low, high))
    return round(bottom - (clipped - low) / max(high - low, 1e-9) * (bottom - top))


def dashed_vertical(draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, color: str, width: int = 2) -> None:
    for y in range(top, bottom, 10):
        draw.line((x, y, x, min(y + 6, bottom)), fill=color, width=width)


def draw_grid(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    domain_start: float,
    domain_end: float,
    label_ticks: bool,
) -> None:
    left, top, right, bottom = box
    for fraction in np.linspace(0.0, 1.0, 5):
        x = round(left + fraction * (right - left))
        draw.line((x, top, x, bottom), fill=GRID_COLOR, width=1)
        if label_ticks:
            value = domain_start + fraction * (domain_end - domain_start)
            draw.text((x, bottom + 2), f"{value:.1f}s", fill=MUTED_COLOR, font=font(12), anchor="ma")


def draw_minmax(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    box: tuple[int, int, int, int],
    low: float,
    high: float,
    color: str,
) -> None:
    left, top, right, bottom = box
    width = max(1, right - left)
    if not len(values):
        return
    edges = np.linspace(0, len(values), width + 1, dtype=int)
    for pixel in range(width):
        chunk = values[edges[pixel]:edges[pixel + 1]]
        if not len(chunk):
            continue
        lo, hi = float(np.nanmin(chunk)), float(np.nanmax(chunk))
        draw.line((left + pixel, y_map(lo, low, high, top, bottom), left + pixel, y_map(hi, low, high, top, bottom)), fill=color, width=1)


def draw_line(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    box: tuple[int, int, int, int],
    low: float,
    high: float,
    color: str,
    width: int = 2,
) -> None:
    left, top, right, bottom = box
    if len(values) < 2:
        return
    count = max(2, right - left)
    indices = np.linspace(0, len(values) - 1, count, dtype=int)
    points = [(left + index, y_map(float(values[source]), low, high, top, bottom)) for index, source in enumerate(indices)]
    draw.line(points, fill=color, width=width)


def zoom_arrays(signal: dict, clip: dict, modality: str, domain_start: float, domain_end: float) -> tuple[np.ndarray, np.ndarray | None]:
    start = clip["sensor_start"] + domain_start
    end = clip["sensor_start"] + domain_end
    mask = (signal["board_ts"] >= start) & (signal["board_ts"] <= end)
    values = signal[modality.lower()][mask]
    envelope = signal["emg_envelope"][mask] if modality == "EMG" else None
    return values, envelope


def draw_interval(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    domain_start: float,
    domain_end: float,
    sensor_start: float,
    sensor_end: float,
    rgb_start: float,
    rgb_end: float,
    origin: float,
    fill: str,
) -> None:
    left, top, right, bottom = box
    interval_start = sensor_start - origin
    interval_end = sensor_end - origin
    xs = max(left, min(right, x_map(interval_start, domain_start, domain_end, left, right)))
    xe = max(left, min(right, x_map(interval_end, domain_start, domain_end, left, right)))
    draw.rectangle((min(xs, xe), top, max(xs + 1, xe), bottom), fill=fill)
    for value in (rgb_start - origin, rgb_end - origin):
        if domain_start <= value <= domain_end:
            dashed_vertical(draw, x_map(value, domain_start, domain_end, left, right), top, bottom, RGB_COLOR, width=2)


def make_pair_plot(
    condition: str,
    model_name: str,
    modality: str,
    level: str,
    target_row: dict[str, str],
    reference_row: dict[str, str],
    target_clip: dict,
    reference_clip: dict,
    target_signal: dict,
    reference_signal: dict,
    label_names: dict[int, str],
    output_path: Path,
) -> None:
    true_key, pred_key = label_columns(level)
    target_true = label_names[int(target_row[true_key])]
    target_pred = label_names[int(target_row[pred_key])]
    ref_true = label_names[int(reference_row[true_key])]
    ref_pred = label_names[int(reference_row[pred_key])]
    channel_names = [f"EMG {index}" for index in range(1, 9)] if modality == "EMG" else ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
    channels = len(channel_names)

    width = 2400
    header_h = 180
    row_h = 92
    bottom_h = 50
    height = header_h + channels * row_h * 2 + bottom_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label_right = 300
    full_left, full_right = 310, 1530
    zoom_left, zoom_right = 1640, width - 28
    max_duration = max(target_signal["duration"], reference_signal["duration"])
    target_clip_duration = target_clip["sensor_end"] - target_clip["sensor_start"]
    reference_clip_duration = reference_clip["sensor_end"] - reference_clip["sensor_start"]
    zoom_start = -0.5
    zoom_end = max(target_clip_duration, reference_clip_duration) + 0.5

    draw.text((24, 14), f"{condition} · {model_name} · {modality} confusion pair", fill=TEXT_COLOR, font=font(29, bold=True))
    draw.text((24, 54), f"TARGET {target_clip['sample_name']}: {target_true} → {target_pred} · {target_clip['run']}", fill="#b91c1c", font=font(20, bold=True))
    draw.text((24, 84), f"REFERENCE {reference_clip['sample_name']}: {ref_true} → {ref_pred} · {reference_clip['run']}", fill="#1d4ed8", font=font(20, bold=True))
    draw.text((full_left, 122), "Full runs · shared seconds from each run start", fill=TEXT_COLOR, font=font(20, bold=True))
    draw.text((zoom_left, 122), "Synchronized local zoom · t=0 at each MindRove clip start", fill=TEXT_COLOR, font=font(20, bold=True))
    legend = "light teal=filtered waveform; purple=100 ms envelope; orange dashes=A0 RGB boundary" if modality == "EMG" else "blue=filtered IMU; orange dashes=A0 RGB boundary"
    draw.text((full_left, 150), legend, fill=MUTED_COLOR, font=font(15))
    draw.rectangle((zoom_left, 146, zoom_left + 20, 163), fill=TARGET_FILL)
    draw.text((zoom_left + 27, 154), "target clip", fill=MUTED_COLOR, font=font(14), anchor="lm")
    draw.rectangle((zoom_left + 145, 146, zoom_left + 165, 163), fill=REFERENCE_FILL)
    draw.text((zoom_left + 172, 154), "reference clip", fill=MUTED_COLOR, font=font(14), anchor="lm")

    target_values = target_signal[modality.lower()]
    reference_values = reference_signal[modality.lower()]
    target_zoom, target_zoom_env = zoom_arrays(target_signal, target_clip, modality, zoom_start, zoom_end)
    reference_zoom, reference_zoom_env = zoom_arrays(reference_signal, reference_clip, modality, zoom_start, zoom_end)

    for channel, channel_name in enumerate(channel_names):
        full_low, full_high = robust_limits(np.concatenate((target_values[:, channel], reference_values[:, channel])), symmetric=modality == "EMG")
        zoom_combined = np.concatenate((target_zoom[:, channel], reference_zoom[:, channel]))
        zoom_low, zoom_high = robust_limits(zoom_combined, symmetric=modality == "EMG")
        for row_index, (role, clip, signal, values, zoom_values, zoom_env, fill) in enumerate((
            ("TARGET", target_clip, target_signal, target_values, target_zoom, target_zoom_env, TARGET_FILL),
            ("REFERENCE", reference_clip, reference_signal, reference_values, reference_zoom, reference_zoom_env, REFERENCE_FILL),
        )):
            top = header_h + (channel * 2 + row_index) * row_h
            bottom = top + row_h - 12
            full_box = (full_left, top + 7, full_right, bottom - 15)
            zoom_box = (zoom_left, top + 7, zoom_right, bottom - 15)
            draw.rectangle((10, top, width - 10, bottom), outline="#cbd5e1", width=1)
            role_color = "#b91c1c" if role == "TARGET" else "#1d4ed8"
            draw.text((20, top + 9), f"{channel_name} · {role}", fill=role_color, font=font(16, bold=True))
            draw.text((20, top + 34), f"{clip['sample_name']} · {clip['run']}", fill=TEXT_COLOR, font=font(14))
            draw.text((label_right - 8, top + 58), f"{full_high:.3g} / {full_low:.3g}", fill=MUTED_COLOR, font=font(12), anchor="ra")

            draw_grid(draw, full_box, 0.0, max_duration, label_ticks=row_index == 1)
            draw_interval(
                draw, full_box, 0.0, max_duration,
                clip["sensor_start"], clip["sensor_end"], clip["rgb_start"], clip["rgb_end"], signal["start"], fill,
            )
            actual_right = x_map(signal["duration"], 0.0, max_duration, full_left, full_right)
            actual_box = (full_left, full_box[1], max(full_left + 2, actual_right), full_box[3])
            if modality == "EMG":
                draw_minmax(draw, values[:, channel], actual_box, full_low, full_high, WAVE_COLOR)
                draw_line(draw, signal["emg_envelope"][:, channel], actual_box, full_low, full_high, ENVELOPE_COLOR, width=2)
            else:
                draw_line(draw, values[:, channel], actual_box, full_low, full_high, IMU_COLOR, width=2)

            draw_grid(draw, zoom_box, zoom_start, zoom_end, label_ticks=row_index == 1)
            draw_interval(
                draw, zoom_box, zoom_start, zoom_end,
                clip["sensor_start"], clip["sensor_end"], clip["rgb_start"], clip["rgb_end"], clip["sensor_start"], fill,
            )
            if modality == "EMG":
                draw_minmax(draw, zoom_values[:, channel], zoom_box, zoom_low, zoom_high, WAVE_COLOR)
                if zoom_env is not None:
                    draw_line(draw, zoom_env[:, channel], zoom_box, zoom_low, zoom_high, ENVELOPE_COLOR, width=2)
            else:
                draw_line(draw, zoom_values[:, channel], zoom_box, zoom_low, zoom_high, IMU_COLOR, width=2)

        separator_y = header_h + (channel + 1) * 2 * row_h - 4
        draw.line((10, separator_y, width - 10, separator_y), fill="#94a3b8", width=2)

    draw.text(((full_left + full_right) // 2, height - 30), "Seconds from each run start", fill=TEXT_COLOR, font=font(16), anchor="ma")
    draw.text(((zoom_left + zoom_right) // 2, height - 30), "Seconds relative to each clip start", fill=TEXT_COLOR, font=font(16), anchor="ma")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def raw_xdf_for_run(run_number: int) -> Path | None:
    matches = sorted(RAW_XDF_ROOT.glob(f"sub-A/ses-*/emg/*_run-{run_number:03d}_emg.xdf"))
    return matches[0] if len(matches) == 1 else None


def rgb_links(clip: dict) -> str:
    parts = []
    for label, camera_id in (("A0", PRIMARY_CAMERA), ("cam2", SECONDARY_CAMERA)):
        camera = clip["cameras"][camera_id]
        parts.append(f"[{label} 原始目录]({windows_explorer_link(camera['dir'])})")
    return " / ".join(parts)


def alignment_row(role: str, clip: dict, signal: dict) -> str:
    return (
        f"| {role} | `{clip['sample_name']}` | `{format_clock(clip['sensor_start'])}`–`{format_clock(clip['sensor_end'])}` | "
        f"`{format_clock(clip['rgb_start'])}`–`{format_clock(clip['rgb_end'])}` | "
        f"{clip['sensor_start'] - signal['start']:.3f}–{clip['sensor_end'] - signal['start']:.3f}s | "
        f"{clip['rgb_start'] - signal['start']:.3f}–{clip['rgb_end'] - signal['start']:.3f}s | "
        f"{1000 * (clip['rgb_start'] - clip['sensor_start']):+.1f}/{1000 * (clip['rgb_end'] - clip['sensor_end']):+.1f} ms |"
    )


def main() -> None:
    required = (MANIFEST_PATH, SEGMENT_LOG, PROTOCOL_PATH)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    protocol_rows = read_jsonl(PROTOCOL_PATH)
    protocol = {row["sample_name"]: row for row in protocol_rows}
    segment_rows = {row["sample_name"]: row for row in read_csv(SEGMENT_LOG) if row.get("status") == "success"}
    metrics = {f"S{index}": json.loads(metric_path(f"S{index}").read_text(encoding="utf-8")) for index in range(1, 13)}
    predictions = {f"S{index}": read_csv(prediction_path(f"S{index}")) for index in range(1, 13)}
    winners = select_winners(metrics)

    node_names = {int(row["node_idx"]): str(row["node_id"]) for row in protocol_rows}
    tier3_names = {int(row["tier3_id"]): str(row["tier3"]) for row in protocol_rows}
    clip_cache: dict[str, dict] = {}
    signal_cache: dict[Path, dict] = {}
    selection_payload = {"low_recall_threshold": LOW_RECALL_THRESHOLD, "max_targets_per_model": MAX_TARGETS_PER_MODEL, "models": []}
    report_links = []

    for winner in winners:
        condition = winner["condition"]
        level = winner["level"]
        modality = winner["modality"]
        label_names = node_names if level == "node" else tier3_names
        true_key, pred_key = label_columns(level)
        model_metrics = metrics[condition][level]
        targets = choose_targets(condition, level, predictions[condition], metrics[condition])
        node_offset = 1 if level == "node" else 0
        recall_by_id = {int(row["class_id"]) + node_offset: float(row["recall"]) for row in model_metrics["per_class"]}
        used_references: set[str] = set()
        pairs = []
        model_asset_dir = ASSET_DIR / condition

        for pair_index, target in enumerate(targets, 1):
            reference, reference_status = choose_reference(target, predictions[condition], level, protocol, used_references)
            target_clip = make_clip_info(target["sample_name"], protocol, segment_rows, clip_cache)
            reference_clip = make_clip_info(reference["sample_name"], protocol, segment_rows, clip_cache)
            target_signal = load_run_signal(target_clip["right_csv"], signal_cache)
            reference_signal = load_run_signal(reference_clip["right_csv"], signal_cache)
            plot_name = f"p{pair_index:02d}_{target_clip['sample_name'][-6:]}_{reference_clip['sample_name'][-6:]}.png"
            plot_path = model_asset_dir / plot_name
            make_pair_plot(
                condition, CONDITION_NAMES[condition], modality, level,
                target, reference, target_clip, reference_clip, target_signal, reference_signal, label_names, plot_path,
            )
            pairs.append({
                "index": pair_index,
                "target": target,
                "reference": reference,
                "reference_status": reference_status,
                "target_clip": target_clip,
                "reference_clip": reference_clip,
                "target_signal": target_signal,
                "reference_signal": reference_signal,
                "plot_path": plot_path,
                "true_name": label_names[int(target[true_key])],
                "pred_name": label_names[int(target[pred_key])],
                "reference_true_name": label_names[int(reference[true_key])],
                "reference_pred_name": label_names[int(reference[pred_key])],
                "class_recall": recall_by_id[int(target[true_key])],
            })

        report_path = REPORT_DIR / f"{condition}.md"
        relative_report = report_path.relative_to(ANALYSIS_DIR).as_posix()
        report_links.append((winner, report_path, relative_report, pairs))
        lines = [
            f"# {condition} · {CONDITION_NAMES[condition]}：混淆片段 run 信号对比", "",
            f"> A_as_test、seed 1；任务层级：{level.upper()}；右手 {modality}。低 Recall 定义为 Recall < {pct(LOW_RECALL_THRESHOLD)}；最多选择 {MAX_TARGETS_PER_MODEL} 个目标错误样本。", "",
            "## 1. 模型与样本选择", "",
            f"- Macro-F1：{pct(model_metrics['macro_f1'])}；Accuracy：{pct(model_metrics['accuracy'])}；Macro Recall：{pct(model_metrics['macro_recall'])}。",
            f"- 同组候选：{winner['candidates'][0]} 与 {winner['candidates'][1]}；按目标层级 Macro-F1 选择 {condition}。",
            "- 目标样本优先覆盖不同低 Recall 真值类别，再覆盖不同混淆方向，最后按混淆频次和错误置信度补足。",
            "- 参考样本优先选取同模型正确识别的预测类别样本；同等条件优先同 run、相近片段时长和较高置信度。当前预测文件不含倒数第二层 embedding，因此本报告未使用特征空间最近邻。", "",
            "| # | 目标样本 | 真值 → 预测值 | 类别 Recall | 目标置信度 | 参考样本 | 参考真值 → 预测值 | 参考状态 |",
            "| ---: | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
        for pair in pairs:
            target_conf = confidence(pair["target"], level)
            lines.append(
                f"| {pair['index']} | `{pair['target_clip']['sample_name']}` | `{pair['true_name']}` → `{pair['pred_name']}` | "
                f"{pct(pair['class_recall'])} | {pct(target_conf) if math.isfinite(target_conf) else 'NA'} | "
                f"`{pair['reference_clip']['sample_name']}` | `{pair['reference_true_name']}` → `{pair['reference_pred_name']}` | {pair['reference_status']} |"
            )

        lines.extend(["", "## 2. 成对 run 与同步片段对比", ""])
        for pair in pairs:
            target_clip = pair["target_clip"]
            reference_clip = pair["reference_clip"]
            target_signal = pair["target_signal"]
            reference_signal = pair["reference_signal"]
            relative_plot = Path("..") / pair["plot_path"].relative_to(ANALYSIS_DIR)
            lines.extend([
                f"### 2.{pair['index']} `{target_clip['sample_name']}`：`{pair['true_name']}` → `{pair['pred_name']}`", "",
                "| 角色 | 样本 | run | 真值 | 预测值 | 片段时长 | 模型置信度 |",
                "| --- | --- | --- | --- | --- | ---: | ---: |",
                f"| 目标错误 | `{target_clip['sample_name']}` | `{target_clip['run']}` | `{pair['true_name']}` | `{pair['pred_name']}` | {target_clip['sensor_end'] - target_clip['sensor_start']:.3f}s | {pct(confidence(pair['target'], level))} |",
                f"| 预测类别参考 | `{reference_clip['sample_name']}` | `{reference_clip['run']}` | `{pair['reference_true_name']}` | `{pair['reference_pred_name']}` | {reference_clip['sensor_end'] - reference_clip['sensor_start']:.3f}s | {pct(confidence(pair['reference'], level))} |", "",
                "**A0 RGB 与 MindRove 时间对齐（沿用 Pilot v3）**", "",
                "| 角色 | 样本 | MindRove 绝对时间 | A0 RGB 绝对时间 | MindRove（run 秒） | A0 RGB（run 秒） | RGB−MindRove start/end |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                alignment_row("目标", target_clip, target_signal),
                alignment_row("参考", reference_clip, reference_signal), "",
                f"![{condition} pair {pair['index']}]({relative_plot.as_posix()})", "",
                "**原始 RGB 与信号数据**", "",
                f"- 目标 `{target_clip['sample_name']}`：{rgb_links(target_clip)}；右手 CSV：`{target_clip['right_csv']}`；片段：`{target_clip['segment_pt']}`。",
                f"- 参考 `{reference_clip['sample_name']}`：{rgb_links(reference_clip)}；右手 CSV：`{reference_clip['right_csv']}`；片段：`{reference_clip['segment_pt']}`。",
                f"- 目标原始 XDF：`{raw_xdf_for_run(target_clip['run_number']) or '未唯一定位'}`。",
                f"- 参考原始 XDF：`{raw_xdf_for_run(reference_clip['run_number']) or '未唯一定位'}`。", "",
                "**人工检查备注**", "",
                "- full-run 共同背景或整 run 信号质量：",
                "- 同步局部片段的共同点：",
                "- 同步局部片段的差异：",
                "- 这些差异是否能够解释模型混淆：", "",
            ])
        report_path.write_text("\n".join(lines), encoding="utf-8")

        selection_payload["models"].append({
            "group": winner["group"],
            "modality": modality,
            "level": level,
            "condition": condition,
            "runner_up": winner["runner_up"],
            "metrics": {key: model_metrics[key] for key in ("macro_f1", "accuracy", "macro_recall", "weakest_class_recall")},
            "report": str(report_path),
            "pairs": [{
                "index": pair["index"],
                "target_sample": pair["target_clip"]["sample_name"],
                "target_run": pair["target_clip"]["run"],
                "true": pair["true_name"],
                "predicted": pair["pred_name"],
                "class_recall": pair["class_recall"],
                "reference_sample": pair["reference_clip"]["sample_name"],
                "reference_run": pair["reference_clip"]["run"],
                "reference_true": pair["reference_true_name"],
                "reference_predicted": pair["reference_pred_name"],
                "reference_status": pair["reference_status"],
                "plot": str(pair["plot_path"]),
            } for pair in pairs],
        })

    master_lines = [
        "# 最佳右手 EMG/IMU 模型：混淆片段成对信号分析", "",
        "> A_as_test、seed 1。本文先从三种训练路线中分别选择最佳 EMG 与 IMU 模型，再为每个模型选择最多10个低 Recall 错误样本，并与其预测类别的参考片段进行 full-run 和同步局部放大比较。", "",
        "## 1. 最佳模型选择", "",
        "| 路线 | 模态 | 候选 | 入选模型 | 入选 Macro-F1 | 另一候选 Macro-F1 | 详细报告 |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for winner, report_path, relative_report, pairs in report_links:
        level = winner["level"]
        condition = winner["condition"]
        runner_up = winner["runner_up"]
        master_lines.append(
            f"| {winner['group']} | {winner['modality']} | {winner['candidates'][0]} / {winner['candidates'][1]} | "
            f"**{condition} · {CONDITION_NAMES[condition]}** | {pct(metrics[condition][level]['macro_f1'])} | "
            f"{pct(metrics[runner_up][level]['macro_f1'])} | [打开 {condition} 报告]({relative_report}) |"
        )

    master_lines.extend([
        "", "## 2. 选择与配对口径", "",
        f"- 低 Recall 阈值沿用既有分析：Recall < {pct(LOW_RECALL_THRESHOLD)}。",
        f"- 每个入选模型最多 {MAX_TARGETS_PER_MODEL} 个目标错误样本；先覆盖不同真值类别，再覆盖不同混淆方向。",
        "- 每个目标 `真实 A → 预测 B` 配一个真实类别 B 的参考样本；优先同模型正确预测为 B，并优先同 run、相近动作时长和较高置信度。",
        "- full-run 左列：目标/参考按同一通道上下排列，两行共享时间尺度和Y轴范围。",
        "- 同步局部右列：两片段均以各自 MindRove 起点设为 t=0，并共享局部时间窗和Y轴范围。",
        "- A0 RGB 边界仍由原始 JPG 文件名时间计算，橙色虚线及表格毫秒差均沿用 Pilot v3 对齐方法。",
        "- EMG 使用浅青绿色滤波波形和紫色100 ms整流包络；IMU使用蓝色滤波信号。", "",
        "## 3. 入选目标样本总览", "",
    ])
    for winner, report_path, relative_report, pairs in report_links:
        condition = winner["condition"]
        master_lines.extend([
            f"### {condition} · {CONDITION_NAMES[condition]}", "",
            "| # | 目标样本 | run | 真值 → 预测值 | Recall | 参考样本 | 参考 run |",
            "| ---: | --- | --- | --- | ---: | --- | --- |",
        ])
        for pair in pairs:
            master_lines.append(
                f"| {pair['index']} | `{pair['target_clip']['sample_name']}` | `{pair['target_clip']['run']}` | "
                f"`{pair['true_name']}` → `{pair['pred_name']}` | {pct(pair['class_recall'])} | "
                f"`{pair['reference_clip']['sample_name']}` | `{pair['reference_clip']['run']}` |"
            )
        master_lines.extend(["", f"[打开 {condition} 的完整成对信号报告]({relative_report})", ""])

    MASTER_REPORT.write_text("\n".join(master_lines), encoding="utf-8")
    SELECTION_JSON.write_text(json.dumps(selection_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "master_report": str(MASTER_REPORT),
        "model_reports": len(report_links),
        "pair_plots": sum(len(item[3]) for item in report_links),
        "winners": [item[0]["condition"] for item in report_links],
        "selection_manifest": str(SELECTION_JSON),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
