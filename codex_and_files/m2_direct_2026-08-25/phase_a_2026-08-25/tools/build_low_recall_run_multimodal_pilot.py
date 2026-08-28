from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw

from tools.build_low_recall_multimodal_pilot import (
    ANALYSIS_DIR,
    ASSET_DIR,
    CAMERAS,
    CONDITION_NAMES,
    DATASET_ROOT,
    MANIFEST_PATH,
    OUTPUT_PATH,
    RAW_RGB_ROOT,
    asset_image_path,
    asset_link,
    direct_tier3_candidates,
    font,
    parse_report_sensor_candidates,
    read_jsonl,
)


SEGMENT_LOG = DATASET_ROOT / "mindrove_seg_log.csv"
RAW_XDF_ROOT = Path("G:/Dataset_thermal_crimper_stage_2_raw/mindrove_unstructured")
RGB_JUNCTION = ASSET_DIR / "rgb"
FRAME_INDEX_DIR = ASSET_DIR / "frame_indexes"
SELECTION_PATH = ASSET_DIR / "pilot_selection.json"
PALETTE = (
    "#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2",
    "#be123c", "#4f46e5", "#65a30d", "#c026d3", "#0f766e", "#b45309",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_timestamp_text(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d_%H%M%S_%f")


def frame_epoch(frame_path: Path, anchor_text: str, anchor_epoch: float) -> float:
    return float(anchor_epoch) + (parse_timestamp_text(frame_path.stem) - parse_timestamp_text(anchor_text)).total_seconds()


def format_clock(epoch: float) -> str:
    # All current A pilot runs are before the 2026 UK daylight-saving transition, so UTC=Europe/London.
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def windows_explorer_link(path: Path) -> str:
    """Return a Windows Search URI that is handled by Explorer, not the browser."""
    encoded_path = quote(str(path.resolve()), safe="")
    return f"search-ms:query=%2A&crumb=folder%3A{encoded_path}"


def blend_with_white(color: str, ratio: float = 0.86) -> str:
    values = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    mixed = [round(value * (1.0 - ratio) + 255 * ratio) for value in values]
    return "#" + "".join(f"{value:02x}" for value in mixed)


def x_pixel(value: float, start: float, end: float, left: int, right: int) -> int:
    return round(left + (value - start) / max(end - start, 1e-9) * (right - left))


def dashed_vertical(draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, color: str, width: int = 2) -> None:
    for y in range(top, bottom, 12):
        draw.line((x, y, x, min(y + 7, bottom)), fill=color, width=width)


def robust_limits(values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return -1.0, 1.0
    if symmetric:
        limit = float(np.quantile(np.abs(finite), 0.995))
        return (-max(limit, 1e-6), max(limit, 1e-6))
    low, high = np.quantile(finite, [0.005, 0.995])
    if math.isclose(float(low), float(high)):
        return float(low - 1.0), float(high + 1.0)
    padding = 0.06 * (high - low)
    return float(low - padding), float(high + padding)


def moving_abs_envelope(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    result = np.empty_like(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    for channel in range(values.shape[1]):
        result[:, channel] = np.convolve(np.abs(values[:, channel]), kernel, mode="same")
    return result


def draw_interval_overlays(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], run_start: float, run_end: float,
    clips: list[dict],
) -> None:
    left, top, right, bottom = box
    for clip in clips:
        xs = x_pixel(clip["sensor_start"], run_start, run_end, left, right)
        xe = x_pixel(clip["sensor_end"], run_start, run_end, left, right)
        draw.rectangle((xs, top, max(xs + 1, xe), bottom), fill=blend_with_white(clip["color"], 0.90))
        xr0 = x_pixel(clip["rgb_start"], run_start, run_end, left, right)
        xr1 = x_pixel(clip["rgb_end"], run_start, run_end, left, right)
        dashed_vertical(draw, xr0, top, bottom, "#d97706", width=2)
        dashed_vertical(draw, xr1, top, bottom, "#d97706", width=2)


def draw_time_grid(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], run_start: float, run_end: float,
) -> None:
    left, top, right, bottom = box
    duration = run_end - run_start
    for fraction in np.linspace(0.0, 1.0, 6):
        x = round(left + fraction * (right - left))
        draw.line((x, top, x, bottom), fill="#d1d5db", width=1)
        draw.text((x, bottom + 4), f"{fraction * duration:.1f}s", fill="#4b5563", font=font(14), anchor="ma")


def draw_minmax_wave(
    draw: ImageDraw.ImageDraw, values: np.ndarray, box: tuple[int, int, int, int],
    y_low: float, y_high: float, color: str,
) -> None:
    left, top, right, bottom = box
    width = max(1, right - left)
    edges = np.linspace(0, len(values), width + 1, dtype=int)
    for pixel in range(width):
        chunk = values[edges[pixel]:edges[pixel + 1]]
        if not len(chunk):
            continue
        low, high = float(np.nanmin(chunk)), float(np.nanmax(chunk))
        y0 = bottom - (np.clip(low, y_low, y_high) - y_low) / max(y_high - y_low, 1e-9) * (bottom - top)
        y1 = bottom - (np.clip(high, y_low, y_high) - y_low) / max(y_high - y_low, 1e-9) * (bottom - top)
        draw.line((left + pixel, round(y0), left + pixel, round(y1)), fill=color, width=1)


def draw_line_wave(
    draw: ImageDraw.ImageDraw, values: np.ndarray, box: tuple[int, int, int, int],
    y_low: float, y_high: float, color: str, width: int = 2,
) -> None:
    left, top, right, bottom = box
    count = max(2, right - left)
    indices = np.linspace(0, len(values) - 1, count, dtype=int)
    clipped = np.clip(np.nan_to_num(values[indices]), y_low, y_high)
    ys = bottom - (clipped - y_low) / max(y_high - y_low, 1e-9) * (bottom - top)
    points = [(left + index, round(value)) for index, value in enumerate(ys)]
    draw.line(points, fill=color, width=width)


def draw_timeline(
    draw: ImageDraw.ImageDraw, clips: list[dict], run_start: float, run_end: float,
    top: int, left: int, right: int,
) -> int:
    lane_h = 54
    draw.text((20, top), "Selected low-recall clips in this run", fill="#111111", font=font(22, bold=True))
    top += 38
    for index, clip in enumerate(clips):
        lane_top = top + index * lane_h
        center = lane_top + 20
        draw.text((20, center), f"{clip['sample_name']} ({'/'.join(clip['conditions'])})", fill="#111111", font=font(17, bold=True), anchor="lm")
        draw.line((left, center, right, center), fill="#d1d5db", width=2)
        xs = x_pixel(clip["sensor_start"], run_start, run_end, left, right)
        xe = x_pixel(clip["sensor_end"], run_start, run_end, left, right)
        draw.rectangle((xs, center - 9, max(xs + 2, xe), center + 9), fill=clip["color"])
        xr0 = x_pixel(clip["rgb_start"], run_start, run_end, left, right)
        xr1 = x_pixel(clip["rgb_end"], run_start, run_end, left, right)
        draw.rectangle((xr0, center - 13, max(xr0 + 2, xr1), center + 13), outline="#d97706", width=3)
        sensor_rel = (clip["sensor_start"] - run_start, clip["sensor_end"] - run_start)
        rgb_rel = (clip["rgb_start"] - run_start, clip["rgb_end"] - run_start)
        draw.text((right - 4, center), f"M {sensor_rel[0]:.3f}–{sensor_rel[1]:.3f}s | RGB {rgb_rel[0]:.3f}–{rgb_rel[1]:.3f}s", fill="#111111", font=font(15), anchor="rm")
    bottom = top + len(clips) * lane_h
    draw.text((left, bottom + 2), "filled band = MindRove clip", fill="#4b5563", font=font(15))
    draw.text((left + 255, bottom + 2), "orange outline/dashes = A0 RGB first–last frame", fill="#d97706", font=font(15))
    return bottom + 34


def make_run_plot(
    run_name: str, board_ts: np.ndarray, values: np.ndarray, labels: list[str], clips: list[dict],
    output_path: Path, modality: str, source_path: Path,
) -> None:
    run_start, run_end = float(board_ts[0]), float(board_ts[-1])
    channels = values.shape[1]
    width = 2000
    header_h = 112
    timeline_h = 38 + len(clips) * 54 + 34
    panel_h = 190 if modality == "EMG" else 220
    plot_left, plot_right = 310, width - 26
    height = header_h + timeline_h + channels * panel_h + 62
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    duration = run_end - run_start
    draw.text((24, 16), f"A {run_name} · right-hand {modality} · full run", fill="#111111", font=font(32, bold=True))
    draw.text((24, 57), f"Run {format_clock(run_start)} → {format_clock(run_end)} · duration {duration:.2f}s · stored points {len(board_ts)}", fill="#4b5563", font=font(19))
    draw.text((24, 84), f"Source: filtered run CSV exported from raw XDF · {source_path.name}", fill="#4b5563", font=font(16))
    panel_top = draw_timeline(draw, clips, run_start, run_end, header_h, plot_left, plot_right)

    envelope = moving_abs_envelope(values, window=max(1, round(0.100 * len(board_ts) / max(duration, 1e-9)))) if modality == "EMG" else None
    for channel in range(channels):
        outer_top = panel_top + channel * panel_h
        outer_bottom = outer_top + panel_h - 10
        plot_top, plot_bottom = outer_top + 30, outer_bottom - 28
        plot_box = (plot_left, plot_top, plot_right, plot_bottom)
        draw.rectangle((12, outer_top, width - 12, outer_bottom), outline="#9ca3af", width=1)
        draw.text((24, outer_top + 5), labels[channel], fill="#111111", font=font(18, bold=True))
        draw_interval_overlays(draw, plot_box, run_start, run_end, clips)
        draw_time_grid(draw, plot_box, run_start, run_end)
        y_low, y_high = robust_limits(values[:, channel], symmetric=modality == "EMG")
        if modality == "EMG":
            # A light teal waveform leaves the softly tinted clip intervals visible;
            # purple keeps the rectified envelope separate from both waveform and RGB boundaries.
            draw_minmax_wave(draw, values[:, channel], plot_box, y_low, y_high, "#5b9fa3")
            draw_line_wave(draw, envelope[:, channel], plot_box, y_low, y_high, "#7c3aed", width=2)
            legend = "light teal=min/max filtered waveform; purple=100 ms rectified envelope"
        else:
            draw_line_wave(draw, values[:, channel], plot_box, y_low, y_high, "#2563eb", width=2)
            legend = "filtered run signal; displayed at stored timeline resolution"
        draw.text((plot_left - 8, plot_top), f"{y_high:.3g}", fill="#4b5563", font=font(13), anchor="ra")
        draw.text((plot_left - 8, plot_bottom), f"{y_low:.3g}", fill="#4b5563", font=font(13), anchor="ra")
        if channel == 0:
            draw.text((plot_right - 4, outer_top + 6), legend, fill="#4b5563", font=font(14), anchor="ra")
    draw.text((width // 2, height - 34), "Time from right-hand MindRove run start (seconds)", fill="#111111", font=font(18), anchor="ma")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def write_frame_index(sample_name: str, camera_id: str, files: list[Path], output_path: Path) -> None:
    camera_number = CAMERAS.index(camera_id) + 1
    relative_prefix = f"../rgb/{sample_name}/{camera_id}"
    first, middle, last = files[0], files[len(files) // 2], files[-1]
    lines = [
        f"# {sample_name} · camera {camera_id} 原始 RGB 帧", "",
        f"[返回 Pilot 文档](../../{OUTPUT_PATH.name})", "",
        f"> 原始目录：`{files[0].parent}`；共 {len(files)} 张。该页通过 analysis 目录内的只读 `rgb` 联接使用相对路径，避免 VS Code 无法解析 `file:///...` 文件夹 URI。", "",
        "## 首、中、末帧预览", "",
        f"![first]({relative_prefix}/{first.name})", "",
        f"![middle]({relative_prefix}/{middle.name})", "",
        f"![last]({relative_prefix}/{last.name})", "",
        "## 全部原始帧", "",
    ]
    for index, path in enumerate(files, 1):
        lines.append(f"- {index:03d}: [{path.name}]({relative_prefix}/{path.name})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def raw_xdf_for_run(run_number: int) -> Path:
    matches = sorted(RAW_XDF_ROOT.glob(f"sub-A/ses-*/emg/*_run-{run_number:03d}_emg.xdf"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one raw XDF for run {run_number}, found {matches}")
    return matches[0]


def main() -> None:
    if not RGB_JUNCTION.is_dir():
        raise FileNotFoundError(f"RGB junction is missing: {RGB_JUNCTION}")
    manifest_rows = read_jsonl(MANIFEST_PATH)
    manifest = {row["sample_name"]: row for row in manifest_rows}
    tier3_names = {int(row["tier3_id"]): str(row["tier3"]) for row in manifest_rows}
    candidates = parse_report_sensor_candidates()
    for condition in ("S9", "S10", "S11", "S12"):
        candidates[condition] = direct_tier3_candidates(condition, tier3_names)

    selected = {}
    used_samples: set[str] = set()
    for condition in [f"S{index}" for index in range(1, 13)]:
        options = candidates[condition]
        chosen = next((item for item in options if item["sample_name"] not in used_samples), options[0])
        selected[condition] = chosen
        used_samples.add(chosen["sample_name"])

    segment_rows = {row["sample_name"]: row for row in read_csv_rows(SEGMENT_LOG) if row.get("status") == "success"}
    conditions_by_sample: dict[str, list[str]] = defaultdict(list)
    for condition, item in selected.items():
        conditions_by_sample[item["sample_name"]].append(condition)

    clip_info = {}
    frame_indexes = {}
    for palette_index, sample_name in enumerate(sorted(used_samples)):
        row = manifest[sample_name]
        seg = segment_rows[sample_name]
        sensor_start = float(row["mindrove_right_start_board_ts"])
        sensor_end = float(row["mindrove_right_end_board_ts"])
        camera_info = {}
        for camera_index, camera_id in enumerate(CAMERAS, 1):
            frame_dir = RAW_RGB_ROOT / sample_name / camera_id
            files = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not files:
                raise FileNotFoundError(f"No RGB frames: {frame_dir}")
            index_path = FRAME_INDEX_DIR / f"{sample_name}_cam{camera_index}.md"
            write_frame_index(sample_name, camera_id, files, index_path)
            camera_info[camera_id] = {
                "dir": frame_dir, "files": files, "index": index_path,
                "start": frame_epoch(files[0], seg["annotation_start_raw"], float(seg["annotation_start_sec"])),
                "end": frame_epoch(files[-1], seg["annotation_start_raw"], float(seg["annotation_start_sec"])),
            }
        primary = camera_info[CAMERAS[0]]
        clip_info[sample_name] = {
            "sample_name": sample_name, "conditions": conditions_by_sample[sample_name],
            "run": row["run"], "run_number": int(str(row["run"]).split("_")[-1]),
            "annotation_row_index": int(row["annotation_row_index"]), "tier3": row["tier3"],
            "sensor_start": sensor_start, "sensor_end": sensor_end,
            "rgb_start": primary["start"], "rgb_end": primary["end"],
            "color": PALETTE[palette_index % len(PALETTE)], "cameras": camera_info,
            "segment_pt": DATASET_ROOT / row["mindrove"], "right_csv": Path(seg["right_csv"]),
        }

    clips_by_run: dict[int, list[dict]] = defaultdict(list)
    for clip in clip_info.values():
        clips_by_run[clip["run_number"]].append(clip)
    for clips in clips_by_run.values():
        clips.sort(key=lambda clip: clip["sensor_start"])

    run_outputs = {}
    run_metadata = {}
    for run_number, clips in sorted(clips_by_run.items()):
        right_csv = clips[0]["right_csv"]
        if not right_csv.is_file():
            raise FileNotFoundError(f"Filtered full-run CSV not found: {right_csv}")
        header = right_csv.open("r", encoding="utf-8-sig").readline().strip().split(",")
        matrix = np.loadtxt(right_csv, delimiter=",", skiprows=1, dtype=np.float64)
        columns = {name: index for index, name in enumerate(header)}
        board_ts = matrix[:, columns["board_ts"]]
        emg = matrix[:, [columns[f"emg{index}"] for index in range(1, 9)]].astype(np.float32)
        imu = matrix[:, [columns[name] for name in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")]].astype(np.float32)
        xdf = raw_xdf_for_run(run_number)
        run_name = f"run_{run_number}"
        emg_path = ASSET_DIR / f"run{run_number:03d}_right_emg.png"
        imu_path = ASSET_DIR / f"run{run_number:03d}_right_imu.png"
        make_run_plot(run_name, board_ts, emg, [f"EMG {index}" for index in range(1, 9)], clips, emg_path, "EMG", right_csv)
        make_run_plot(run_name, board_ts, imu, ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"], clips, imu_path, "IMU", right_csv)
        run_outputs[run_number] = {"emg": emg_path, "imu": imu_path}
        run_metadata[run_number] = {
            "run_start": float(board_ts[0]), "run_end": float(board_ts[-1]),
            "points": len(board_ts), "right_csv": right_csv, "raw_xdf": xdf,
        }

    selection_by_sample = {
        item["sample_name"]: (condition, item) for condition, item in selected.items()
    }

    def frame_links(clip: dict, camera: dict, camera_id: str) -> str:
        first, last = camera["files"][0], camera["files"][-1]
        prefix = f"low_recall_multimodal_pilot_assets/rgb/{clip['sample_name']}/{camera_id}"
        explorer_uri = windows_explorer_link(camera["dir"])
        return (
            f"{asset_link('逐帧索引', camera['index'])} ｜ "
            f"[首帧]({prefix}/{first.name}) ｜ [末帧]({prefix}/{last.name}) ｜ "
            f"[资源管理器]({explorer_uri})（{len(camera['files'])} 张）"
        )

    lines = [
        "# S1–S12 低 Recall 样本：run 级多模态质量检查（Pilot v3）", "",
        "> 范围：A_as_test、seed 1；每个 S 条件选择 1 个低 Recall 误分类样本。本文以 run 为分析单位，同一 run 的全部所选片段共享一组 EMG/IMU 图，并在同一时间轴上联合标注。", "",
        "## 1. 所选低 Recall 样本", "",
        "| 条件 | 模型 | 样本 | run | 真实 Tier3 | 错误层级 | 真值 | 预测值 | 真实类别 Recall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for condition, item in selected.items():
        clip = clip_info[item["sample_name"]]
        lines.append(
            f"| {condition} | {CONDITION_NAMES[condition]} | `{clip['sample_name']}` | `{clip['run']}` | "
            f"`{clip['tier3']}` | {item['level']} | `{item['true_label']}` | `{item['predicted_label']}` | {item['class_recall']} |"
        )

    lines.extend([
        "", "## 2. 数据与时间对齐口径", "",
        f"- 原始 MindRove/XDF 根目录：`{RAW_XDF_ROOT}`。",
        "- run 图读取由原始 XDF 按参考流程导出的右手全 run CSV；该 CSV 使用 EMG 50–450 Hz band-pass + 50 Hz notch、IMU 20 Hz low-pass，与生成训练片段 `mindrove.pt` 的 run 数据源一致。图中不再次滤波。",
        "- 彩色填充区间：`mindrove.pt` 中保存的右手 board timestamp 起止。橙色边界：A0 主摄像头 `001484412812` 原始 JPG 的第一帧与最后一帧文件名时间。",
        "- run 图横轴是相对右手 MindRove run 起点的秒数；表格保留绝对时间、run-relative 时间以及 RGB−MindRove 边界差。",
        "- 同一 run 的全部所选低 Recall 片段会同时标在该 run 的 EMG/IMU 图上。时间对齐算法与 Pilot v2 完全相同。", "",
        "## 3. 打开原始 RGB 帧", "",
        "- `逐帧索引` 提供首/中/末帧预览以及每一张原始 JPG 的链接。",
        "- `首帧`、`末帧` 使用实验目录内的相对链接。",
        "- `资源管理器` 使用 Windows 内置 `search-ms:` 协议直接定位原始 RGB 文件夹；浏览器第一次调用时可能要求确认。", "",
        "## 4. 时间对齐总览", "",
        "| 条件 | 样本 | run | MindRove 起止（run 秒） | A0 RGB 起止（run 秒） | RGB−MindRove start/end |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for condition, item in selected.items():
        clip = clip_info[item["sample_name"]]
        meta = run_metadata[clip["run_number"]]
        run_start = meta["run_start"]
        lines.append(
            f"| {condition} | `{clip['sample_name']}` | `{clip['run']}` | "
            f"{clip['sensor_start'] - run_start:.3f}–{clip['sensor_end'] - run_start:.3f}s | "
            f"{clip['rgb_start'] - run_start:.3f}–{clip['rgb_end'] - run_start:.3f}s | "
            f"{1000 * (clip['rgb_start'] - clip['sensor_start']):+.1f}/{1000 * (clip['rgb_end'] - clip['sensor_end']):+.1f} ms |"
        )
    lines.extend(["", "## 5. 按 run 联合检查", ""])

    for run_index, (run_number, clips) in enumerate(sorted(clips_by_run.items()), 1):
        meta = run_metadata[run_number]
        run_start = meta["run_start"]
        run_label = clips[0]["run"]
        lines.extend([
            f"### 5.{run_index} `{run_label}` — {len(clips)} 个所选低 Recall 样本", "",
            "**该 run 中的真值与预测值**", "",
            "| 条件 | 模型 | 样本 | 真实 Tier3 | 错误层级 | 真值 | 预测值 | Recall |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: |",
        ])
        for clip in clips:
            condition, item = selection_by_sample[clip["sample_name"]]
            lines.append(
                f"| {condition} | {CONDITION_NAMES[condition]} | `{clip['sample_name']}` | `{clip['tier3']}` | "
                f"{item['level']} | `{item['true_label']}` | `{item['predicted_label']}` | {item['class_recall']} |"
            )

        lines.extend([
            "", "**A0 RGB 与 MindRove 时间边界（原对齐方式保持不变）**", "",
            "| 样本 | MindRove 绝对时间 | A0 RGB 绝对时间 | MindRove（run 秒） | A0 RGB（run 秒） | RGB−MindRove start/end |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for clip in clips:
            lines.append(
                f"| `{clip['sample_name']}` | `{format_clock(clip['sensor_start'])}`–`{format_clock(clip['sensor_end'])}` | "
                f"`{format_clock(clip['rgb_start'])}`–`{format_clock(clip['rgb_end'])}` | "
                f"{clip['sensor_start'] - run_start:.3f}–{clip['sensor_end'] - run_start:.3f}s | "
                f"{clip['rgb_start'] - run_start:.3f}–{clip['rgb_end'] - run_start:.3f}s | "
                f"{1000 * (clip['rgb_start'] - clip['sensor_start']):+.1f}/{1000 * (clip['rgb_end'] - clip['sensor_end']):+.1f} ms |"
            )

        lines.extend([
            "", "**整个 run 的右手 EMG**", "",
            f"![{run_label} right EMG]({asset_image_path(run_outputs[run_number]['emg'])})", "",
            "**整个 run 的右手 IMU**", "",
            f"![{run_label} right IMU]({asset_image_path(run_outputs[run_number]['imu'])})", "",
            "**两个视角的原始 RGB 检查入口**", "",
            "| 条件 | 样本 | 主相机 001484412812 | 第二相机 001528512812 | 九帧联系图 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for clip in clips:
            condition, _ = selection_by_sample[clip["sample_name"]]
            primary = clip["cameras"][CAMERAS[0]]
            secondary = clip["cameras"][CAMERAS[1]]
            primary_contact = ASSET_DIR / f"{clip['sample_name']}_cam1.png"
            secondary_contact = ASSET_DIR / f"{clip['sample_name']}_cam2.png"
            lines.append(
                f"| {condition} | `{clip['sample_name']}` | {frame_links(clip, primary, CAMERAS[0])} | "
                f"{frame_links(clip, secondary, CAMERAS[1])} | "
                f"{asset_link('主相机', primary_contact)} / {asset_link('第二相机', secondary_contact)} |"
            )

        lines.extend([
            "", "**数据源定位**", "",
            f"- 原始 XDF：`{meta['raw_xdf']}`",
            f"- 已滤波全 run 右手 CSV：`{meta['right_csv']}`",
            "- 片段 `mindrove.pt`：",
        ])
        for clip in clips:
            condition, _ = selection_by_sample[clip["sample_name"]]
            lines.append(f"  - {condition} `{clip['sample_name']}`：`{clip['segment_pt']}`")

        lines.extend([
            "", "**人工检查备注**", "",
            "| 条件 | 样本 | RGB 清晰度/遮挡/边界 | EMG 激活 | IMU 方向/相位 | 时间对齐 | 其他信号质量问题 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for clip in clips:
            condition, _ = selection_by_sample[clip["sample_name"]]
            lines.append(f"| {condition} | `{clip['sample_name']}` |  |  |  |  |  |")
        lines.extend([
            "", "- 整 run 是否存在平线、饱和、漂移或孤立尖峰：", "",
        ])

    lines.extend([
        "## 6. Pilot 通过后的批量规则", "",
        "1. 按 run 去重生成信号图，而不是每个 clip 重复生成；一个 run 内所有抽中的低 Recall 片段同时标注。",
        "2. 每个 clip 保留 A0 RGB 第一/最后帧时间、MindRove board timestamp 和两者毫秒差。",
        "3. 每个摄像头生成原始逐帧索引；文档只链接原始 JPG，不复制原始数据。",
        "4. 继续保留 run 级 EMG 100 ms 包络和六轴 IMU，以便观察动作前后上下文及整 run 信号质量。", "",
        "复现命令：`python tools/build_low_recall_run_multimodal_pilot.py`。", "",
    ])
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    (ASSET_DIR / "run_pilot_alignment.json").write_text(
        json.dumps({
            "selected": selected,
            "runs": {str(key): {name: str(value) if isinstance(value, Path) else value for name, value in metadata.items()} for key, metadata in run_metadata.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "document": str(OUTPUT_PATH), "runs": len(clips_by_run),
        "conditions": len(selected), "unique_samples": len(used_samples),
        "frame_indexes": len(used_samples) * len(CAMERAS),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
