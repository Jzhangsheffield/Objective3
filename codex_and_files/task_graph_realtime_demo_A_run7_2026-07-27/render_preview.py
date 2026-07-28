from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from demo_core import DEMO_ROOT, load_demo_data, resolve


WIDTH, HEIGHT = 1600, 980
COLORS = {
    "page": "#0B1220",
    "panel": "#111C2E",
    "panel_alt": "#16243A",
    "border": "#2A3B54",
    "text": "#EAF0F8",
    "muted": "#91A3BA",
    "blue": "#46A0FF",
    "green": "#3DDC97",
    "red": "#FF6B6B",
    "amber": "#F5C451",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(
        box, radius=12, fill=COLORS["panel"], outline=COLORS["border"], width=2
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def choose_result(rows: list[dict], requested_sample: str | None) -> dict:
    if requested_sample:
        return next(row for row in rows if row["sample_name"] == requested_sample)
    preferred = [
        row
        for row in rows
        if row["m3"]["correct"]
        and (not row["m0"]["correct"] or not row["e2e"]["correct"])
    ]
    return preferred[0] if preferred else rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one demo profile preview")
    parser.add_argument("--config", type=Path, default=DEMO_ROOT / "config.json")
    parser.add_argument("--sample")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    data = load_demo_data(config_path)
    output_dir = resolve(data.config["paths"]["outputs_dir"])
    rows = read_jsonl(output_dir / "validation_predictions.jsonl")
    result = choose_result(rows, args.sample)
    segment = next(
        row
        for row in data.segments
        if row.get("original_action_sample_name") == result["sample_name"]
    )
    frame_rows = data.frames_by_segment[int(segment["segment_no"])]
    frame_row = frame_rows[len(frame_rows) // 2]
    with Image.open(data.raw_frames_dir / frame_row["frame_name"]) as source:
        source = source.convert("RGB")
        video = ImageOps.contain(source, (960, 540), Image.Resampling.LANCZOS)

    demo = data.config["demo"]
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["page"])
    draw = ImageDraw.Draw(image)
    draw.text(
        (30, 22),
        "TASK GRAPH HISTORY • REAL-TIME REPLAY",
        fill=COLORS["blue"],
        font=font(17, True),
    )
    draw.text(
        (30, 50),
        (
            f"{demo['participant']} / {demo['source_run']}  •  "
            f"camera {demo['camera_id']}  •  all-runs seed {demo['seed']}"
        ),
        fill=COLORS["text"],
        font=font(31, True),
    )
    draw.text(
        (30, 93),
        (
            f"Action {result['annotation_row_index']} complete • "
            f"M0 N{result['m0']['pred_node_idx']} • "
            f"M3 N{result['m3']['pred_node_idx']} • "
            f"E2E N{result['e2e']['pred_node_idx']} • 1× playback"
        ),
        fill=COLORS["green"],
        font=font(17),
    )

    left = (30, 130, 1010, 780)
    panel(draw, left)
    draw.text((48, 145), "LIVE FRAME STREAM", fill=COLORS["muted"], font=font(15, True))
    video_canvas = Image.new("RGB", (960, 540), "#030712")
    video_canvas.paste(video, ((960 - video.width) // 2, (540 - video.height) // 2))
    image.paste(video_canvas, (40, 180))
    draw.text(
        (48, 738),
        (
            f"Frame {frame_row['frame_idx']} / {len(data.frames)}  •  "
            f"H.264 display  •  {frame_row['frame_name']}"
        ),
        fill=COLORS["muted"],
        font=font(13),
    )

    right_x = 1030
    panel(draw, (right_x, 130, 1570, 220))
    draw.text((right_x + 18, 144), "CURRENT SEGMENT", fill=COLORS["muted"], font=font(13, True))
    draw.text(
        (right_x + 18, 170),
        f"Action {result['annotation_row_index']} • prediction complete",
        fill=COLORS["green"],
        font=font(21, True),
    )
    draw.text(
        (right_x + 18, 199),
        "M0, M3, E2E and ground truth shown below",
        fill=COLORS["text"],
        font=font(13),
    )

    def prediction_card(y: int, title: str, key: str, accent: str) -> None:
        model = result[key]
        panel(draw, (right_x, y, 1570, y + 135))
        draw.rectangle((right_x, y, 1570, y + 4), fill=accent)
        draw.text((right_x + 18, y + 13), title, fill=accent, font=font(12, True))
        draw.text(
            (right_x + 18, y + 36),
            f"Action {result['annotation_row_index']} • Predicted Node {model['pred_node_idx']}",
            fill=accent,
            font=font(18, True),
        )
        draw.text(
            (right_x + 18, y + 62),
            model["pred_node_id"],
            fill=COLORS["text"],
            font=font(12, True),
        )
        verdict = (
            f"CORRECT • matches Ground-truth Node {result['true_node_idx']}"
            if model["correct"]
            else f"INCORRECT • Ground truth is Node {result['true_node_idx']}"
        )
        draw.text(
            (right_x + 18, y + 84),
            verdict,
            fill=COLORS["green"] if model["correct"] else COLORS["red"],
            font=font(12, True),
        )
        occurrence = (
            f" • occurrence {model['occurrence']}"
            if model["occurrence"] is not None
            else ""
        )
        draw.text(
            (right_x + 18, y + 108),
            (
                f"Tier-3: {model['label']}{occurrence} • "
                f"confidence {model['confidence']:.3f}"
            ),
            fill=COLORS["muted"],
            font=font(11),
        )

    prediction_card(235, "M0 • CURRENT FROZEN RGB FEATURE", "m0", COLORS["amber"])
    prediction_card(380, "M3 • GRAPH-VALID HISTORY", "m3", COLORS["green"])
    prediction_card(525, "E2E-NODE-SCRATCH • CURRENT RGB ONLY", "e2e", COLORS["blue"])

    panel(draw, (right_x, 670, 1570, 780))
    draw.text(
        (right_x + 18, 683),
        "REVEALED AFTER ALL THREE PREDICTIONS",
        fill=COLORS["muted"],
        font=font(12, True),
    )
    truth_occurrence = (
        f" • occurrence {result['true_occurrence']}"
        if result["true_occurrence"] is not None
        else ""
    )
    draw.text(
        (right_x + 18, 707),
        f"Action {result['annotation_row_index']} • Ground-truth Node {result['true_node_idx']}{truth_occurrence}",
        fill=COLORS["text"],
        font=font(16, True),
    )
    draw.text(
        (right_x + 18, 733),
        result["true_node_id"],
        fill=COLORS["text"],
        font=font(12, True),
    )
    draw.text(
        (right_x + 18, 755),
        f"Tier-3: {result['true_label']} • history before current: {result['history_length_before_current']}",
        fill=COLORS["muted"],
        font=font(11),
    )

    panel(draw, (30, 800, 1570, 950))
    draw.text(
        (48, 815),
        "COMPLETED ACTION HISTORY • SELECTED ROW",
        fill=COLORS["muted"],
        font=font(13, True),
    )
    headers = ["#", "Ground truth", "M0 prediction", "M3 prediction", "E2E prediction", "Inference"]
    x_positions = [50, 105, 470, 770, 1070, 1440]
    for x, text in zip(x_positions, headers):
        draw.text((x, 848), text, fill=COLORS["muted"], font=font(12, True))
    values = [
        str(result["annotation_row_index"]),
        result["true_display_name"],
        f"N{result['m0']['pred_node_idx']} {'CORRECT' if result['m0']['correct'] else 'WRONG'}",
        f"N{result['m3']['pred_node_idx']} {'CORRECT' if result['m3']['correct'] else 'WRONG'}",
        f"N{result['e2e']['pred_node_idx']} {'CORRECT' if result['e2e']['correct'] else 'WRONG'}",
        f"{result['inference_ms']:.0f} ms",
    ]
    for x, text in zip(x_positions, values):
        draw.text((x, 882), text, fill=COLORS["text"], font=font(13))

    profile_id = demo.get("profile_id", f"{demo['participant']}_{demo['source_run']}")
    output = args.output or DEMO_ROOT / "previews" / f"{profile_id}_preview.png"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
