from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from demo_core import DEMO_ROOT, load_demo_data, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one demo profile")
    parser.add_argument("--config", type=Path, default=DEMO_ROOT / "config.json")
    args = parser.parse_args()
    data = load_demo_data(args.config.resolve())
    sample_count = min(600, len(data.frames))
    started = time.perf_counter()
    checksum = 0
    for row in data.frames[:sample_count]:
        with Image.open(data.raw_frames_dir / row["frame_name"]) as image:
            resized = image.convert("RGB").resize((960, 540), Image.Resampling.LANCZOS)
            checksum += int(np.asarray(resized)[0, 0, 0])
    jpeg_seconds = time.perf_counter() - started

    video_path = resolve(data.config["paths"]["display_video"])
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    started = time.perf_counter()
    decoded_frames = 0
    while decoded_frames < sample_count:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        checksum += int(rgb[0, 0, 0])
        decoded_frames += 1
    video_seconds = time.perf_counter() - started
    capture.release()

    report = {
        "schema_version": "task-graph-demo-playback-benchmark-v1",
        "sample_frames": sample_count,
        "jpeg_open_decode_lanczos_resize_ms_per_frame": 1000
        * jpeg_seconds
        / sample_count,
        "h264_sequential_decode_rgb_ms_per_frame": 1000
        * video_seconds
        / decoded_frames,
        "display_source_stage_speedup": jpeg_seconds / video_seconds,
        "target_frame_budget_at_effective_fps_ms": 1000
        / ((len(data.frames) - 1) / data.duration_seconds),
        "checksum": checksum,
    }
    output = resolve(data.config["paths"]["display_video_report"]).with_name(
        "playback_benchmark.json"
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
