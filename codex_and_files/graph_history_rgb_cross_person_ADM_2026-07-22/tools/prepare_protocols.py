from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

_PACKAGE_ROOT = _Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from graph_history.protocols import prepare_protocols


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one LOSO fold with normal-only and all-run manifests")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--test-participant", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Skip a complete existing protocol and reject a partial one instead of overwriting it.",
    )
    args = parser.parse_args()

    output_root = _Path(args.output_root)
    required = [
        output_root / scope / f"{split}.jsonl"
        for scope in ("normal_only", "all_runs")
        for split in ("train", "test_normal", "test_fault", "test_all")
    ] + [output_root / "protocol_report.json"]
    existing = [path for path in required if path.is_file()]
    if args.if_missing and len(existing) == len(required):
        print(f"Complete protocol already exists; skipping without overwrite: {output_root}")
        return
    if args.if_missing and existing:
        missing = [str(path) for path in required if not path.is_file()]
        raise FileExistsError(
            "Protocol directory is partial; refusing to mix or overwrite files. "
            f"Missing: {missing}"
        )
    report = prepare_protocols(args.dataset_root, output_root, args.test_participant)
    for scope, splits in report["protocols"].items():
        print(f"[{scope}]")
        for split_name, summary in splits.items():
            print(
                f"  {split_name}: samples={summary['samples']} runs={summary['runs']} "
                f"nodes={summary['present_node_count']}/35 tier3={summary['present_tier3_count']}/31"
            )


if __name__ == "__main__":
    main()
