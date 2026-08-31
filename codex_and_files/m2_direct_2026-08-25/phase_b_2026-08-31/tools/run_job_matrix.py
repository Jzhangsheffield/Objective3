from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase B matrix sequentially with file-based resume")
    parser.add_argument("--matrix", default=str(PACKAGE_ROOT / "scripts" / "phase_b_job_matrix.csv"))
    parser.add_argument("--from-stage", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with Path(args.matrix).open("r", encoding="utf-8-sig", newline="") as handle:
        jobs = list(csv.DictReader(handle))
    started = args.from_stage is None
    summary = {"completed": [], "skipped": [], "failed": []}
    for job in jobs:
        if not started:
            started = job["stage"] == args.from_stage
            if not started:
                continue
        expected = Path(job["expected_output"])
        label = f"job={job['job_id']} stage={job['stage']} outer={job['outer']} inner={job['inner']} seed={job['seed']}"
        if expected.exists():
            print(f"SKIP {label}", flush=True)
            summary["skipped"].append(job["job_id"])
            continue
        print(f"RUN {label}\n{job['command']}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(job["command"], shell=True)
        if result.returncode:
            summary["failed"].append({"job_id": job["job_id"], "returncode": result.returncode})
            break
        if not expected.exists():
            summary["failed"].append({"job_id": job["job_id"], "reason": "expected_output_missing"})
            break
        summary["completed"].append(job["job_id"])
    report = PACKAGE_ROOT / "outputs" / "job_runner_status.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["failed"]:
        raise SystemExit(f"Phase B stopped: {summary['failed'][-1]}")


if __name__ == "__main__":
    main()
