from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from sensor_boundary.config import format_path, load_config
from sensor_boundary.engine import infer_run, load_checkpoint
from sensor_boundary.online import run_state_machine
from sensor_boundary.utils import read_json, resolve_device, safe_torch_load, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one sensor run with the causal online decoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint-name", default="best.pth")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    output = format_path(cfg["paths"]["output_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    model, _ = load_checkpoint(output / args.checkpoint_name, device)
    online_cfg = cfg["online"]
    calibrated = output / "calibrated_online.json"
    if calibrated.is_file():
        online_cfg = read_json(calibrated)["selected_online"]
    cache = safe_torch_load(Path(cfg["paths"]["cache_root"]) / f"{args.run}.pt")
    probabilities = infer_run(model, cache, device, int(cfg["evaluation"]["inference_chunk_steps"]))
    segments = run_state_machine(**probabilities, settings=online_cfg)
    times = cache["times"].numpy()
    online_root = output / "online_pipeline"
    online_root.mkdir(parents=True, exist_ok=True)
    stream_path = online_root / f"{args.run}_stream.jsonl"
    with stream_path.open("w", encoding="utf-8") as handle:
        for index, (state, start, end) in enumerate(zip(probabilities["state_probability"], probabilities["start_probability"], probabilities["end_probability"])):
            handle.write(json.dumps({
                "index": index,
                "timestamp": float(times[index]),
                "state_probability": float(state),
                "start_probability": float(start),
                "end_probability": float(end),
            }, separators=(",", ":")) + "\n")
    enriched = []
    for segment in segments:
        enriched.append({
            **segment,
            "start_time": float(times[segment["start_index"]]),
            "end_time": float(times[segment["end_index"]]),
            "emitted_at_time": float(times[min(segment["emitted_at_index"], len(times) - 1)]),
        })
    write_json(online_root / f"{args.run}_segments.json", {"sample_name": args.run, "segments": enriched})
    print(f"Wrote {len(enriched)} segments and {len(times)} causal decisions to {online_root}")


if __name__ == "__main__":
    main()
