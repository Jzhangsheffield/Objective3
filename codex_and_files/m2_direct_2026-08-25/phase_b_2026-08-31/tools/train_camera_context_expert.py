from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unchanged M2-direct camera expert")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--protocol-parent", required=True,
                        help="Directory containing the all_runs protocol directory")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    project = Path(config["m2_project_root"])
    command = [
        sys.executable, str(project / "tools" / "train_direct_history_model.py"),
        "--model", "m2_direct", "--train-scope", "all_runs",
        "--protocol-root", str(Path(args.protocol_parent)),
        "--train-cache", args.train_cache, "--test-cache", args.test_cache,
        "--task-graph", str(project / "assets" / "integrated_task_graph_latest.json"),
        "--relation-matrix", str(project / "assets" / "integrated_feature_history_matrix.json"),
        "--output-root", args.output_root,
        "--epochs", str(config["crossfit"]["camera_context_epochs"]),
        "--batch-size", "64", "--num-workers", str(args.num_workers),
        "--learning-rate", "0.001", "--weight-decay", "0.0001",
        "--d-model", str(config["d_model"]), "--num-heads", str(config["num_heads"]),
        "--max-history", str(config["max_history"]), "--dropout", str(config["dropout"]),
        "--seed", str(args.seed), "--device", args.device,
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
