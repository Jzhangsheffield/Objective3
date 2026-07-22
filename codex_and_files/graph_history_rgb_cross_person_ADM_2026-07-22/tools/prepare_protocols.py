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
    args = parser.parse_args()
    report = prepare_protocols(args.dataset_root, args.output_root, args.test_participant)
    for scope, splits in report["protocols"].items():
        print(f"[{scope}]")
        for split_name, summary in splits.items():
            print(
                f"  {split_name}: samples={summary['samples']} runs={summary['runs']} "
                f"nodes={summary['present_node_count']}/35 tier3={summary['present_tier3_count']}/31"
            )


if __name__ == "__main__":
    main()
