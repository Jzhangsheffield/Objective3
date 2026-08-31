from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b.config import load_config
from phase_b.io import read_jsonl, sha256, write_json, write_jsonl
from phase_b.paths import crossfit_protocol, outer_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict inner-LOSO manifests for B1/B2 stacking")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--outer", choices=list("ADJM"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    outer_values = [args.outer] if args.outer else list(config["participants"])
    summaries = []
    for outer in outer_values:
        source = outer_protocol(config, outer) / "train.jsonl"
        rows = read_jsonl(source)
        inner_participants = sorted({str(row["participant"]) for row in rows})
        expected = sorted(set(config["participants"]) - {outer})
        if inner_participants != expected:
            raise ValueError(f"Outer {outer}: expected train participants {expected}, got {inner_participants}")
        for inner in inner_participants:
            destination = crossfit_protocol(config, outer, inner)
            if destination.exists() and any(destination.iterdir()) and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {destination}")
            train = [row for row in rows if str(row["participant"]) != inner]
            test_all = [row for row in rows if str(row["participant"]) == inner]
            test_normal = [row for row in test_all if "normal" in str(row.get("run", "")).lower()]
            test_fault = [row for row in test_all if row not in test_normal]
            # The source protocol has authoritative normal/fault membership; use sample sets instead of name heuristics.
            source_normal = {str(row["sample_name"]) for row in read_jsonl(outer_protocol(config, inner) / "test_normal.jsonl")}
            test_normal = [row for row in test_all if str(row["sample_name"]) in source_normal]
            test_fault = [row for row in test_all if str(row["sample_name"]) not in source_normal]
            for name, selected in (
                ("train", train), ("test_all", test_all),
                ("test_normal", test_normal), ("test_fault", test_fault),
            ):
                write_jsonl(destination / f"{name}.jsonl", selected)
            metadata = {
                "outer_test_participant": outer,
                "inner_heldout_participant": inner,
                "inner_train_participants": sorted(set(inner_participants) - {inner}),
                "source_outer_train_manifest": str(source),
                "source_sha256": sha256(source),
                "counts": {
                    "train": len(train), "test_all": len(test_all),
                    "test_normal": len(test_normal), "test_fault": len(test_fault),
                },
                "purpose": "out-of-fold predictions for B1/B2 meta-fusion only",
                "outer_test_samples_used": false,
            }
            write_json(destination / "metadata.json", metadata)
            summaries.append(metadata)
    write_json(Path(config["output_root"]) / "crossfit_protocols" / "summary.json", summaries)
    print(f"Created {len(summaries)} strict inner-LOSO protocol sets")


if __name__ == "__main__":
    main()
