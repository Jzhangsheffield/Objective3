from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from demo_core import DEMO_ROOT, load_demo_data, resolve
from prepare_demo_metadata import prepare


def ffconcat_quote(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", r"'\''")


def probe_video(ffprobe: str, video_path: Path) -> dict:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(video_path),
    ]
    return json.loads(subprocess.check_output(command, text=True, encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a profile display video")
    parser.add_argument("--config", type=Path, default=DEMO_ROOT / "config.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    prepare(config_path)
    data = load_demo_data(config_path)
    settings = data.config["display_video"]
    video_path = resolve(data.config["paths"]["display_video"])
    report_path = resolve(data.config["paths"]["display_video_report"])
    concat_path = video_path.with_name("display_frames.ffconcat")
    temporary_video = video_path.with_name(f"{video_path.stem}.building.mp4")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and ffprobe must be available on PATH.")

    fps = (len(data.frames) - 1) / data.duration_seconds
    concat_lines = ["ffconcat version 1.0"]
    concat_lines.extend(
        f"file '{ffconcat_quote(data.raw_frames_dir / row['frame_name'])}'"
        for row in data.frames
    )
    concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary_video.exists():
        temporary_video.unlink()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-r",
        f"{fps:.12f}",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-an",
        "-vf",
        (
            f"scale={int(settings['width'])}:{int(settings['height'])}"
            ":flags=fast_bilinear,format=yuv420p"
        ),
        "-c:v",
        str(settings["codec"]),
        "-preset",
        str(settings["preset"]),
        "-crf",
        str(settings["crf"]),
        "-r",
        f"{fps:.12f}",
        "-frames:v",
        str(len(data.frames)),
        "-movflags",
        "+faststart",
        str(temporary_video),
    ]

    started = time.perf_counter()
    subprocess.run(command, check=True)
    build_seconds = time.perf_counter() - started
    os.replace(temporary_video, video_path)

    probe = probe_video(ffprobe, video_path)
    stream = probe["streams"][0]
    encoded_frames = int(stream["nb_read_frames"])
    if encoded_frames != len(data.frames):
        raise RuntimeError(
            f"Encoded frame count mismatch: expected {len(data.frames)}, got {encoded_frames}"
        )
    if int(stream["width"]) != int(settings["width"]) or int(stream["height"]) != int(
        settings["height"]
    ):
        raise RuntimeError("Encoded display video resolution does not match config.json.")

    report = {
        "schema_version": "task-graph-demo-display-video-v1",
        "display_only": True,
        "model_inference_uses_original_jpegs": True,
        "source_frames": len(data.frames),
        "encoded_frames": encoded_frames,
        "source_duration_seconds": data.duration_seconds,
        "effective_fps": fps,
        "video_path": str(video_path),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "encoded_duration_seconds": float(probe["format"]["duration"]),
        "bytes": int(probe["format"]["size"]),
        "build_seconds": build_seconds,
        "ffmpeg_command": command,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
