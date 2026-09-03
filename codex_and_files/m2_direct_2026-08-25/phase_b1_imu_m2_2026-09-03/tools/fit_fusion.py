from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import torch.nn.functional as F

from phase_b1_imu_m2.common import (
    DEFAULT_CONFIG, add_phase_b_to_path, fusion_root, inner_m2_root,
    inner_phase_b_root, load_config, outer_m2_root, outer_protocol,
    seed_everything, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit B1 with cam0 M2 + cam1 M2 + IMU M2")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--outer", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    add_phase_b_to_path(config)
    from phase_b.calibration import apply_static_fusion, fit_static_fusion
    from phase_b.evaluation import align_probability_files, write_probability_evaluation
    from phase_b.io import read_jsonl
    from phase_b.metrics import derive_node_to_tier3
    from phase_b.paths import a0_probabilities, b0_condition_root

    inner_probabilities, rows = [], []
    inner_participants = [value for value in config["participants"] if value != args.outer]
    for inner in inner_participants:
        base = inner_phase_b_root(config, args.outer, inner, args.seed)
        files = [
            base / "cam0_m2" / "all_runs" / "m2_direct" / "test_results" / "test_all_probabilities.pt",
            base / "cam1_m2" / "all_runs" / "m2_direct" / "test_results" / "test_all_probabilities.pt",
            inner_m2_root(config, args.outer, inner, args.seed) / "test_results" / "test_all_probabilities.pt",
        ]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing OOF expert probabilities:\n" + "\n".join(missing))
        aligned_rows, values = align_probability_files(files)
        rows.extend(aligned_rows)
        inner_probabilities.append(torch.stack(values, dim=1).float())
    probabilities = torch.cat(inner_probabilities)
    if len({str(row["sample_name"]) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate samples in strict OOF predictions")
    targets = torch.tensor([int(row.get("true_node_idx", row.get("node_idx"))) - 1 for row in rows])

    settings = config["fusion"]
    fitted = fit_static_fusion(
        probabilities, targets, steps=int(settings["steps"]),
        learning_rate=float(settings["learning_rate"]),
        uniform_l2=float(settings["weight_l2_to_uniform"]),
        epsilon=float(settings["probability_epsilon"]),
    )
    output = fusion_root(config, args.outer, args.seed)
    completed = output / "completed.json"
    if completed.is_file() and not args.overwrite:
        print(f"SKIP completed: {completed}")
        return
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite partial output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, output / "fusion_parameters.pt")
    write_json(output / "fit_summary.json", {
        "condition": "B1_IMU_M2", "training": "strict_inner_loso_out_of_fold",
        "outer_participant": args.outer, "inner_participants": inner_participants,
        "oof_samples": len(rows), "seed": args.seed,
        "experts": settings["experts"],
        "temperatures": fitted["log_temperature"].exp().tolist(),
        "weights": F.softmax(fitted["weight_logits"], dim=0).tolist(),
        "history": fitted["history"],
    })

    mapping = derive_node_to_tier3(read_jsonl(outer_protocol(config, args.outer) / "train.jsonl"))
    for split in config["evaluation"]["splits"]:
        files = [
            a0_probabilities(config["_phase_b_config"], args.outer, args.seed, split),
            b0_condition_root(config["_phase_b_config"], "A1", args.outer, args.seed)
            / "test_results" / f"{split}_probabilities.pt",
            outer_m2_root(config, args.outer, args.seed) / "test_results" / f"{split}_probabilities.pt",
        ]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing outer expert probabilities:\n" + "\n".join(missing))
        aligned_rows, values = align_probability_files(files)
        outer_values = torch.stack(values, dim=1).float()
        fused = apply_static_fusion(
            outer_values, fitted["log_temperature"], fitted["weight_logits"],
            float(settings["probability_epsilon"]),
        )
        write_probability_evaluation(aligned_rows, fused, mapping, output / "test_results", split)
    write_json(completed, {
        "condition": "B1_IMU_M2", "outer": args.outer, "seed": args.seed,
        "experts": settings["experts"], "splits": config["evaluation"]["splits"],
    })
    print(f"Saved B1_IMU_M2 fusion: {output}")


if __name__ == "__main__":
    main()
