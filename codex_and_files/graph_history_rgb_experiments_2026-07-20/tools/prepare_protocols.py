from __future__ import annotations

import argparse

from graph_history.protocols import prepare_protocols


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare J-as-test normal-only and all-run manifests")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--test-participant", default="J", choices=["A", "D", "J", "M"])
    args = parser.parse_args()
    report = prepare_protocols(args.dataset_root, args.output_root, args.test_participant)
    for scope, splits in report["protocols"].items():
        print(f"[{scope}]")
        for split_name, summary in splits.items():
            print(f"  {split_name}: samples={summary['samples']} runs={summary['runs']}")


if __name__ == "__main__":
    main()

