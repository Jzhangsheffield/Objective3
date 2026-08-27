from __future__ import annotations

import csv
import io
import json
import math
import pickle
import re
import zipfile
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "a_as_test_seed_1"
ASSET_DIR = ANALYSIS_DIR / "low_recall_multimodal_pilot_assets"
REPORT_PATH = ROOT / "A_AS_TEST_SMALL_SCOPE_FUSION_ANALYSIS_2026-08-26.md"
OUTPUT_PATH = ANALYSIS_DIR / "S1_S12_LOW_RECALL_MULTIMODAL_QUALITY_CHECK_PILOT.md"
DATASET_ROOT = Path("C:/Junxi_data_for_training_speedup/Stage_2_Mapstyle_Dataset")
MANIFEST_PATH = DATASET_ROOT / "3_camera_mindrove_manifest.jsonl"
RAW_RGB_ROOT = Path(
    "D:/Junxi_data/MULTISENSOR_DATA_COLLECTION_Stage2_structured_data/"
    "Action_Recognition_Dataset/Samples"
)
CAMERAS = ("001484412812", "001528512812")
CONDITION_NAMES = {
    "S1": "EMG ResNet10 Tier3→M2 Node", "S2": "EMG Dilated Tier3→M2 Node",
    "S3": "IMU ResNet10 Tier3→M2 Node", "S4": "IMU Dilated Tier3→M2 Node",
    "S5": "EMG ResNet10 Direct Node", "S6": "EMG Dilated Direct Node",
    "S7": "IMU ResNet10 Direct Node", "S8": "IMU Dilated Direct Node",
    "S9": "EMG ResNet10 Direct Tier3", "S10": "EMG Dilated Direct Tier3",
    "S11": "IMU ResNet10 Direct Tier3", "S12": "IMU Dilated Direct Tier3",
}
FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class _Storage:
    values: np.ndarray


def _rebuild_tensor(storage: _Storage, offset: int, size: tuple[int, ...], stride: tuple[int, ...], *_) -> np.ndarray:
    itemsize = storage.values.dtype.itemsize
    return np.ndarray(
        shape=tuple(size), dtype=storage.values.dtype, buffer=storage.values,
        offset=int(offset) * itemsize, strides=tuple(int(value) * itemsize for value in stride),
    ).copy()


class _TorchZipUnpickler(pickle.Unpickler):
    DTYPE_BY_STORAGE = {
        "ByteStorage": np.uint8, "CharStorage": np.int8, "ShortStorage": np.int16,
        "IntStorage": np.int32, "LongStorage": np.int64, "HalfStorage": np.float16,
        "FloatStorage": np.float32, "DoubleStorage": np.float64, "BoolStorage": np.bool_,
    }

    def __init__(self, stream: io.BytesIO, archive: zipfile.ZipFile, prefix: str):
        super().__init__(stream)
        self.archive = archive
        self.prefix = prefix
        self.cache: dict[str, _Storage] = {}

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch" and name in self.DTYPE_BY_STORAGE:
            return type(name, (), {"numpy_dtype": self.DTYPE_BY_STORAGE[name]})
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, identifier):
        if not isinstance(identifier, tuple) or identifier[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {identifier!r}")
        _, storage_type, key, _, count = identifier[:5]
        key = str(key)
        if key not in self.cache:
            raw = self.archive.read(f"{self.prefix}/data/{key}")
            values = np.frombuffer(raw, dtype=storage_type.numpy_dtype, count=int(count)).copy()
            self.cache[key] = _Storage(values)
        return self.cache[key]


def load_torch_zip(path: Path):
    with zipfile.ZipFile(path) as archive:
        data_pickle = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        prefix = data_pickle.rsplit("/", 1)[0]
        return _TorchZipUnpickler(io.BytesIO(archive.read(data_pickle)), archive, prefix).load()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold and FONT_BOLD.is_file() else FONT_REGULAR
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def as_rgb_frame(frames: np.ndarray, index: int) -> Image.Image:
    frame = np.asarray(frames[index])
    if frame.ndim != 3:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    if frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        finite = frame[np.isfinite(frame)]
        if finite.size and finite.max() <= 1.5:
            frame = frame * 255.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def make_contact_sheet(camera_path: Path, output_path: Path, sample_name: str, camera_id: str) -> dict:
    loaded = load_torch_zip(camera_path)
    frames = np.asarray(loaded["frames"] if isinstance(loaded, dict) and "frames" in loaded else loaded)
    if frames.ndim != 4:
        raise ValueError(f"Camera tensor must be 4D, got {frames.shape} at {camera_path}")
    indices = np.linspace(0, len(frames) - 1, num=min(9, len(frames)), dtype=int).tolist()
    tile_w, tile_h, caption_h = 360, 203, 30
    columns, rows = 3, math.ceil(len(indices) / 3)
    header_h = 72
    canvas = Image.new("RGB", (columns * tile_w, header_h + rows * (tile_h + caption_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"{sample_name} · camera {camera_id} · {len(frames)} frames", fill="#111111", font=font(26, bold=True))
    draw.text((18, 43), "Uniformly sampled frames; use the document link to open the full original JPG folder.", fill="#4b5563", font=font(17))
    for position, index in enumerate(indices):
        frame = as_rgb_frame(frames, index)
        frame.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (position % columns) * tile_w + (tile_w - frame.width) // 2
        y = header_h + (position // columns) * (tile_h + caption_h) + (tile_h - frame.height) // 2
        canvas.paste(frame, (x, y))
        draw.text((position % columns * tile_w + 12, header_h + (position // columns) * (tile_h + caption_h) + tile_h + 3),
                  f"frame {index + 1}/{len(frames)}", fill="#111111", font=font(17))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return {"shape": list(frames.shape), "frames": len(frames), "indices": indices}


def _nice_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return -1.0, 1.0
    low, high = np.quantile(finite, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or math.isclose(float(low), float(high)):
        center = float(np.nanmean(finite)) if finite.size else 0.0
        return center - 1.0, center + 1.0
    padding = 0.08 * (high - low)
    return float(low - padding), float(high + padding)


def _plot_series(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], x: np.ndarray, y: np.ndarray,
                 title: str, color: str, y_label: str = "stored units") -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#9ca3af", width=1)
    draw.text((left + 8, top + 5), title, fill="#111111", font=font(18, bold=True))
    plot_left, plot_top, plot_right, plot_bottom = left + 72, top + 34, right - 18, bottom - 32
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#6b7280", width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#6b7280", width=1)
    y_low, y_high = _nice_range(y)
    clipped = np.clip(np.nan_to_num(y, nan=0.0), y_low, y_high)
    if len(x) > plot_right - plot_left:
        take = np.linspace(0, len(x) - 1, plot_right - plot_left, dtype=int)
        x, clipped = x[take], clipped[take]
    x_low, x_high = float(x[0]), float(x[-1]) if len(x) > 1 else float(x[0] + 1.0)
    xp = plot_left + (x - x_low) / max(x_high - x_low, 1e-9) * (plot_right - plot_left)
    yp = plot_bottom - (clipped - y_low) / max(y_high - y_low, 1e-9) * (plot_bottom - plot_top)
    points = [(int(a), int(b)) for a, b in zip(xp, yp)]
    if len(points) > 1:
        draw.line(points, fill=color, width=2)
    draw.text((plot_left, plot_bottom + 5), "0", fill="#4b5563", font=font(14))
    draw.text((plot_right, plot_bottom + 5), f"{x_high:.2f}s", fill="#4b5563", font=font(14), anchor="ra")
    draw.text((plot_left - 7, plot_top), f"{y_high:.3g}", fill="#4b5563", font=font(13), anchor="ra")
    draw.text((plot_left - 7, plot_bottom), f"{y_low:.3g}", fill="#4b5563", font=font(13), anchor="ra")
    draw.text((right - 8, top + 6), y_label, fill="#4b5563", font=font(14), anchor="ra")


def make_emg_figure(emg: np.ndarray, duration: float, output_path: Path, sample_name: str) -> None:
    if emg.ndim != 2 or emg.shape[1] != 8:
        raise ValueError(f"right_emg expected [L,8], got {emg.shape}")
    width, header_h, panel_w, panel_h = 1800, 105, 900, 265
    canvas = Image.new("RGB", (width, header_h + 4 * panel_h + 50), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), f"{sample_name} · Right-hand EMG", fill="#111111", font=font(32, bold=True))
    draw.text((28, 58), f"8 channels · native acquisition 500 Hz · stored points {len(emg)} · clip duration {duration:.3f}s · no extra filtering", fill="#4b5563", font=font(20))
    time = np.linspace(0.0, duration, num=len(emg), endpoint=False)
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be123c", "#4f46e5")
    for channel in range(8):
        row, column = divmod(channel, 2)
        box = (column * panel_w + 10, header_h + row * panel_h, (column + 1) * panel_w - 10, header_h + (row + 1) * panel_h - 10)
        _plot_series(draw, box, time, emg[:, channel], f"EMG channel {channel + 1}", colors[channel])
    draw.text((width // 2, canvas.height - 32), "Time from clip start (seconds)", fill="#111111", font=font(18), anchor="ma")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def make_imu_figure(acc: np.ndarray, gyro: np.ndarray, duration: float, output_path: Path, sample_name: str) -> dict:
    if acc.shape != gyro.shape or acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"right IMU expected matched [L,3], got {acc.shape}, {gyro.shape}")
    # The saved arrays are aligned to the EMG-length clip. Plot a 50 Hz-equivalent view without filtering again.
    target_points = max(2, round(duration * 50.0))
    indices = np.linspace(0, len(acc) - 1, num=min(target_points, len(acc)), dtype=int)
    time = np.linspace(0.0, duration, num=len(indices), endpoint=False)
    acc_view, gyro_view = acc[indices], gyro[indices]
    width, header_h, panel_w, panel_h = 1800, 120, 600, 340
    canvas = Image.new("RGB", (width, header_h + 2 * panel_h + 55), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), f"{sample_name} · Right-hand IMU", fill="#111111", font=font(32, bold=True))
    draw.text((28, 58), f"Acc + Gyro · native acquisition 50 Hz · stored points {len(acc)} · plotted points {len(indices)} · clip duration {duration:.3f}s", fill="#4b5563", font=font(20))
    draw.text((28, 86), "The .pt arrays are clip-aligned and equal-length with EMG; this view samples them at a 50 Hz-equivalent grid and does not filter again.", fill="#4b5563", font=font(17))
    colors = ("#2563eb", "#dc2626", "#059669")
    for row_index, (kind, values) in enumerate((("Acceleration", acc_view), ("Gyroscope", gyro_view))):
        for axis in range(3):
            box = (axis * panel_w + 10, header_h + row_index * panel_h, (axis + 1) * panel_w - 10, header_h + (row_index + 1) * panel_h - 12)
            _plot_series(draw, box, time, values[:, axis], f"{kind} {'XYZ'[axis]}", colors[axis])
    draw.text((width // 2, canvas.height - 34), "Time from clip start (seconds)", fill="#111111", font=font(18), anchor="ma")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return {"stored_points": len(acc), "plotted_points": len(indices)}


def parse_report_sensor_candidates() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    condition = None
    in_sensor_section = False
    for line in REPORT_PATH.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("#### 5.4.2"):
            in_sensor_section = True
            continue
        if in_sensor_section and line.startswith("## 6."):
            break
        if not in_sensor_section:
            continue
        heading = re.match(r"^#####\s+(S[1-8])\s+—", line)
        if heading:
            condition = heading.group(1)
            continue
        if condition is None or not line.startswith("| node_"):
            continue
        cells = [cell.strip() for cell in line.strip().split("|")[1:-1]]
        if len(cells) < 6:
            continue
        samples = re.findall(r"`(sample_\d+)`\s*→\s*`([^`]+)`", cells[4])
        for sample_name, predicted in samples:
            result[condition].append({
                "sample_name": sample_name, "level": "Node", "true_label": cells[0],
                "predicted_label": predicted, "class_recall": cells[3], "source": "主报告 5.4.2 抽样表",
            })
    return result


def direct_tier3_candidates(condition: str, tier3_names: dict[int, str]) -> list[dict]:
    path = ROOT / "outputs" / "supplementary" / condition / "A_as_test" / "seed_1" / "test_results" / "test_all_predictions.csv"
    rows = read_csv(path)
    counts = Counter(int(row["true_tier3_id"]) for row in rows)
    correct = Counter(int(row["true_tier3_id"]) for row in rows if row["true_tier3_id"] == row["pred_tier3_id"])
    low_ids = sorted(counts, key=lambda class_id: (correct[class_id] / counts[class_id], class_id))
    result = []
    for class_id in low_ids:
        recall = correct[class_id] / counts[class_id]
        if recall >= 0.80:
            continue
        for row in sorted(rows, key=lambda value: value["sample_name"]):
            if int(row["true_tier3_id"]) != class_id or row["true_tier3_id"] == row["pred_tier3_id"]:
                continue
            predicted_id = int(row["pred_tier3_id"])
            result.append({
                "sample_name": row["sample_name"], "level": "Tier3",
                "true_label": tier3_names[class_id], "predicted_label": tier3_names[predicted_id],
                "class_recall": f"{100 * recall:.1f}%", "source": "Direct Tier3 低 Recall 类别（由 test_all_predictions.csv 计算）",
            })
    return result


def asset_link(label: str, path: Path) -> str:
    """Use a document-relative path so VS Code Markdown Preview can open it reliably."""
    relative = path.resolve().relative_to(OUTPUT_PATH.parent.resolve()).as_posix()
    return f"[{label}]({relative})"


def raw_file_link(label: str, path: Path) -> str:
    """Use a standard file URI for source tensors stored on another Windows drive."""
    return f"[{label}](<{path.resolve().as_uri()}>)"


def asset_image_path(path: Path) -> str:
    return path.resolve().relative_to(OUTPUT_PATH.parent.resolve()).as_posix()


def main() -> None:
    if not DATASET_ROOT.is_dir() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")
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
        if not options:
            raise RuntimeError(f"No low-recall candidates found for {condition}")
        chosen = next((item for item in options if item["sample_name"] not in used_samples), options[0])
        selected[condition] = chosen
        used_samples.add(chosen["sample_name"])

    generated = {}
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for sample_name in sorted(used_samples):
        row = manifest[sample_name]
        duration = float(row["mindrove_right_end_board_ts"]) - float(row["mindrove_right_start_board_ts"])
        mindrove_path = DATASET_ROOT / row["mindrove"]
        signals = load_torch_zip(mindrove_path)
        emg, acc, gyro = (np.asarray(signals[key]) for key in ("right_emg", "right_acc", "right_gyro"))
        emg_path = ASSET_DIR / f"{sample_name}_right_emg.png"
        imu_path = ASSET_DIR / f"{sample_name}_right_imu.png"
        make_emg_figure(emg, duration, emg_path, sample_name)
        imu_info = make_imu_figure(acc, gyro, duration, imu_path, sample_name)
        camera_outputs = {}
        for camera_number, camera_id in enumerate(CAMERAS, 1):
            raw_camera_path = DATASET_ROOT / row[f"{camera_id}_rgb"]
            original_frames_dir = RAW_RGB_ROOT / sample_name / camera_id
            if not original_frames_dir.is_dir():
                raise FileNotFoundError(f"Original RGB frame directory not found: {original_frames_dir}")
            original_frame_count = sum(1 for path in original_frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            # Keep generated names short enough for legacy Windows MAX_PATH handling.
            sheet_path = ASSET_DIR / f"{sample_name}_cam{camera_number}.png"
            sheet_info = make_contact_sheet(raw_camera_path, sheet_path, sample_name, camera_id)
            camera_outputs[camera_id] = {
                "tensor": raw_camera_path, "original_frames_dir": original_frames_dir,
                "original_frame_count": original_frame_count, "sheet": sheet_path, **sheet_info,
            }
        generated[sample_name] = {
            "row": row, "duration": duration, "mindrove": mindrove_path,
            "emg": emg_path, "imu": imu_path, "emg_shape": list(emg.shape),
            "acc_shape": list(acc.shape), "gyro_shape": list(gyro.shape),
            "imu_info": imu_info, "cameras": camera_outputs,
        }

    lines = [
        "# S1–S12 低 Recall 样本多模态质量检查（Pilot）", "",
        "> 范围：A_as_test、seed 1。该 Pilot 每个 S 条件只选 1 个误分类样本，用于先确认视频帧链接、信号时间轴和版式；尚不是全部低 Recall 样本包。", "",
        "## 1. 选择与可视化口径", "",
        "- S1–S8：从主报告 5.4.2 已经展示的低 Recall Node 误分类样本中选择；为减少重复，优先让不同条件使用不同样本。",
        "- S9–S12：这些模型没有 Node 输出，因此从各自 Direct Tier3 的低 Recall 类别中选择误分类样本。",
        "- 两个视角：主相机 `001484412812` 与第二相机 `001528512812`。每个相对链接打开九帧联系图；`打开原始 RGB 帧文件夹` 则定位到结构化数据集中的逐帧 JPG 目录。",
        "- EMG：右手 8 通道，按 500 Hz 原生采集频率说明，使用 `mindrove.pt` 中已保存信号，不额外滤波或归一化。",
        "- IMU：右手 Acc XYZ + Gyro XYZ，原生采集频率 50 Hz。由于 `mindrove.pt` 中 IMU 与 EMG 已保存为等长、clip 对齐数组，图中按 clip 时长抽取 50 Hz 等效显示点，不再次滤波。",
        "- 所有纵轴均为 `mindrove.pt` 的存储值，当前数据没有在 manifest 中声明物理单位，因此不把数值误标为 mV、g 或 °/s。", "",
        "### VS Code 打开方式", "",
        "1. 在 VS Code 中打开本 `.md`，按 `Ctrl+Shift+V` 进入 Markdown Preview；在源码编辑器中则需要按住 `Ctrl` 再点击链接。",
        "2. `打开 9 帧联系图` 使用相对路径，应直接在 VS Code 中打开 PNG。EMG/IMU 图片也使用相对路径，可在预览中直接显示。",
        "3. `打开原始 RGB 帧文件夹` 使用 `file:///D:/...` 文件夹 URI，目标是 `Action_Recognition_Dataset/Samples/sample_xxxxxx/摄像头ID/`，可逐张检查原始 JPG 帧。",
        "4. `mindrove.pt` 是 PyTorch 二进制文件，链接只用于定位信号源文件；EMG/IMU 请直接查看文档内嵌图片。",
        "5. 如果 VS Code 的 Workspace Trust 阻止打开跨工作区文件夹，可复制链接旁显示的原始帧目录到文件资源管理器；这不影响联系图和信号图。", "",
        "## 2. Pilot 样本总览", "",
        "| 条件 | 模型 | 样本 | 判断层级 | 真实类别 → 预测类别 | 该真实类别 Recall |", "| --- | --- | --- | --- | --- | --- |",
    ]
    for condition, item in selected.items():
        lines.append(f"| {condition} | {CONDITION_NAMES[condition]} | `{item['sample_name']}` | {item['level']} | `{item['true_label']}` → `{item['predicted_label']}` | {item['class_recall']} |")
    lines.extend(["", "## 3. 按条件检查", ""])
    for condition, item in selected.items():
        sample_name = item["sample_name"]
        info = generated[sample_name]
        row = info["row"]
        primary, secondary = info["cameras"][CAMERAS[0]], info["cameras"][CAMERAS[1]]
        lines.extend([
            f"### {condition} — {CONDITION_NAMES[condition]}", "",
            f"- 样本：`{sample_name}`；participant `{row['participant']}`；run `{row['run']}`；annotation row `{row['annotation_row_index']}`；clip 时长 `{info['duration']:.3f} s`。",
            f"- 模型错误：{item['level']} `{item['true_label']}` → `{item['predicted_label']}`；该真实类别 Recall `{item['class_recall']}`。",
            f"- 选择来源：{item['source']}。", "",
            "**两个视角的视频帧**", "",
            f"- 主相机 `{CAMERAS[0]}`：{asset_link('打开 9 帧联系图', primary['sheet'])} ｜ {raw_file_link('打开原始 RGB 帧文件夹', primary['original_frames_dir'])}（{primary['original_frame_count']} 张）｜ `{primary['original_frames_dir']}`",
            f"- 第二相机 `{CAMERAS[1]}`：{asset_link('打开 9 帧联系图', secondary['sheet'])} ｜ {raw_file_link('打开原始 RGB 帧文件夹', secondary['original_frames_dir'])}（{secondary['original_frame_count']} 张）｜ `{secondary['original_frames_dir']}`",
            f"- 右手原始信号：{raw_file_link('mindrove.pt', info['mindrove'])}", "",
            "**右手 EMG**", "", f"![{condition} {sample_name} right EMG]({asset_image_path(info['emg'])})", "",
            "**右手 IMU**", "", f"![{condition} {sample_name} right IMU]({asset_image_path(info['imu'])})", "",
            "**人工检查备注**", "",
            "- 两视角清晰度/遮挡：",
            "- 动作边界与标签是否一致：",
            "- EMG 是否存在饱和、平线、尖峰或明显通道异常：",
            "- IMU 是否存在漂移、平线、尖峰或轴异常：",
            "- 两视角与信号的时间一致性：", "",
        ])
    lines.extend([
        "## 4. Pilot 通过后建议的批量规则", "",
        "1. 继续沿用主报告的抽样口径：每个低 Recall 类别最多 10 个误分类样本。",
        "2. 同一 `sample_name` 若同时被多个 S 条件选中，只生成一套视频联系图、EMG 和 IMU 图片，在各条件章节复用，避免重复占用空间。",
        "3. 批量前先由人工确认本 Pilot 中：相机颜色/方向正确、两视角选择正确、EMG/IMU 时间轴合理、图片尺寸适合逐样本检查。",
        "4. 批量文档中保留条件→类别→样本三级结构，并增加可填写的质量备注字段。", "",
    ])
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    summary_path = ASSET_DIR / "pilot_selection.json"
    summary_path.write_text(json.dumps({"selected": selected, "unique_samples": sorted(used_samples)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"document": str(OUTPUT_PATH), "assets": str(ASSET_DIR), "conditions": len(selected), "unique_samples": len(used_samples)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
