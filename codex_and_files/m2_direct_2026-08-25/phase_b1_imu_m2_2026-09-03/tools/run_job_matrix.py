from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = PACKAGE_ROOT / "scripts" / "job_matrix.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/resume the B1_IMU_M2 job matrix")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--start-job", type=int, default=1)
    parser.add_argument("--stop-job", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with Path(args.matrix).open("r", encoding="utf-8-sig", newline="") as handle:
        jobs = list(csv.DictReader(handle))
    if not jobs:
        raise ValueError("Empty job matrix")
    overwrite_stages = {"01_inner_imu_features", "02_inner_imu_m2", "03_outer_imu_m2", "04_fit_b1_imu_m2"}
    for job in jobs:
        job_id = int(job["job_id"])
        if job_id < args.start_job or (args.stop_job is not None and job_id > args.stop_job):
            continue
        expected = Path(job["expected_output"]) if job["expected_output"] else None
        if expected is not None and expected.is_file() and not args.overwrite:
            print(f"SKIP job={job_id} stage={job['stage']} expected={expected}", flush=True)
            continue
        command = json.loads(job["command_json"])
        if command[0] == "python":
            command[0] = sys.executable
        if args.overwrite and job["stage"] in overwrite_stages:
            command.append("--overwrite")
        print(
            f"RUN job={job_id} stage={job['stage']} outer={job['outer']} "
            f"inner={job['inner']} seed={job['seed']}", flush=True,
        )
        print(subprocess.list2cmdline(command), flush=True)
        result = subprocess.run(command, cwd=PACKAGE_ROOT)
        if result.returncode:
            raise SystemExit(f"Experiment stopped at job {job_id}: returncode={result.returncode}")
    print("B1_IMU_M2 selected job range completed", flush=True)


if __name__ == "__main__":
    main()
