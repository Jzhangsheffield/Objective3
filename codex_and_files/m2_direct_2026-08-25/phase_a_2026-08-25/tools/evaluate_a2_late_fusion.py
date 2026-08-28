from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_a.config import load_config
from phase_a.io import read_jsonl, write_json
from phase_a.metrics import derive_node_to_tier3
from phase_a.paths import a0_result_dir, model_dir, protocol_dir
from phase_a.saved_predictions import align_probability_files, write_probability_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 parameter-free paired late probability fusion")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = model_dir(config, "A2", args.participant, args.seed)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocols = protocol_dir(config, args.participant)
    mapping = derive_node_to_tier3(read_jsonl(protocols / "train.jsonl"))
    weight = float(config["late_fusion_weight_primary"])
    for split in ("test_all", "test_normal", "test_fault"):
        a0 = a0_result_dir(config, args.participant, args.seed) / f"{split}_probabilities.pt"
        a1 = model_dir(config, "A1", args.participant, args.seed) / "test_results" / f"{split}_probabilities.pt"
        rows, probabilities = align_probability_files([a0, a1])
        fused = weight * probabilities[0] + (1.0 - weight) * probabilities[1]
        write_probability_evaluation(rows, fused, mapping, output / "test_results", split)
    write_json(output / "completed.json", {
        "condition": "A2", "participant": args.participant, "seed": args.seed,
        "fusion": f"{weight:.6f} * A0 node probability + {1.0 - weight:.6f} * A1 node probability",
        "tuned_on_test": False,
    })


if __name__ == "__main__":
    main()
