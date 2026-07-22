from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect M0-M6 metrics into one CSV")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for path in sorted(Path(args.results_root).rglob("*_metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if "node" not in metrics or "tier3" not in metrics:
            continue
        relative = path.relative_to(args.results_root)
        parts = relative.parts
        scope = parts[0] if parts else "unknown"
        model = parts[1] if len(parts) > 1 else "unknown"
        rows.append(
            {
                "train_scope": scope,
                "model": model,
                "split": metrics.get("split", path.stem.replace("_metrics", "")),
                "samples": metrics.get("samples", 0),
                "node_accuracy": metrics["node"]["accuracy"],
                "node_macro_f1": metrics["node"]["macro_f1"],
                "tier3_accuracy": metrics["tier3"]["accuracy"],
                "tier3_macro_f1": metrics["tier3"]["macro_f1"],
                "tier3_balanced_accuracy": metrics["tier3"]["balanced_accuracy"],
                "metrics_path": str(path),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
