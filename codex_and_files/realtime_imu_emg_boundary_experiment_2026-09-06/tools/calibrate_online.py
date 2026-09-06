from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from sensor_boundary.config import format_path, load_config
from sensor_boundary.engine import infer_run, load_checkpoint
from sensor_boundary.metrics import evaluate_run
from sensor_boundary.online import run_state_machine
from sensor_boundary.utils import read_json, resolve_device, safe_torch_load, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune online decoding only on training-participant validation runs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--checkpoint-name", default="best.pth")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    output = format_path(cfg["paths"]["output_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    resolved = read_json(output / "resolved_config.json")
    validation_runs = [str(x) for x in resolved["validation_runs"]]
    model, _ = load_checkpoint(output / args.checkpoint_name, device)
    cached = {}
    for run in validation_runs:
        cache = safe_torch_load(Path(cfg["paths"]["cache_root"]) / f"{run}.pt")
        cached[run] = (cache, infer_run(model, cache, device, int(cfg["evaluation"]["inference_chunk_steps"])))
    search = cfg["calibration"]
    candidates = []
    for threshold, start_db, end_db, minimum, merge in itertools.product(
        search["probability_thresholds"], search["start_debounce_steps"], search["end_debounce_steps"],
        search["min_action_steps"], search["merge_gap_steps"],
    ):
        settings = {
            "start_threshold": float(threshold),
            "end_threshold": float(threshold),
            "action_threshold": float(threshold),
            "start_debounce_steps": int(start_db),
            "end_debounce_steps": int(end_db),
            "min_action_steps": int(minimum),
            "merge_gap_steps": int(merge),
        }
        scores = []
        for cache, probabilities in cached.values():
            times = cache["times"].numpy()
            segments = run_state_machine(**probabilities, settings=settings)
            pred_state = np.zeros(len(times), dtype=np.int64)
            pairs = []
            for segment in segments:
                pair = (int(segment["start_index"]), int(segment["end_index"]))
                pairs.append(pair)
                pred_state[pair[0] : pair[1] + 1] = 1
            gt_starts = times[np.flatnonzero(cache["exact_start"].numpy() > 0)].tolist()
            gt_ends = times[np.flatnonzero(cache["exact_end"].numpy() > 0)].tolist()
            metrics = evaluate_run(
                cache["state"].numpy(), pred_state, cache["segment_id"].numpy(), gt_starts, gt_ends,
                pairs, [float(times[x[0]]) for x in pairs], [float(times[x[1]]) for x in pairs],
                [int(search["objective_boundary_tolerance_ms"])], float(times[-1] - times[0]),
            )
            tolerance = str(int(search["objective_boundary_tolerance_ms"]))
            iou = str(int(float(search["objective_segment_iou"]) * 100))
            scores.append(float(np.mean([
                metrics["boundary"][tolerance]["start"]["f1"],
                metrics["boundary"][tolerance]["end"]["f1"],
                metrics["segmental_f1"][iou]["f1"],
            ])))
        candidates.append({"score": float(np.mean(scores)), "online": settings})
    candidates.sort(key=lambda row: row["score"], reverse=True)
    result = {
        "heldout": args.heldout,
        "seed": args.seed,
        "scope": args.scope,
        "validation_runs": validation_runs,
        "objective": "mean(start_F1@200ms, end_F1@200ms, segmental_F1@50)",
        "selected_score": candidates[0]["score"],
        "selected_online": candidates[0]["online"],
        "top_10": candidates[:10],
        "candidate_count": len(candidates),
    }
    write_json(output / "calibrated_online.json", result)
    print(result["selected_score"], result["selected_online"])


if __name__ == "__main__":
    main()
