from __future__ import annotations

import csv
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_a.config import load_config


def main() -> None:
    config = load_config(PACKAGE_ROOT / "config" / "phase_a.json")
    jobs, job_id = [], 0
    def add(stage, command, depends_on=""):
        nonlocal job_id
        job_id += 1
        jobs.append({"job_id": job_id, "stage": stage, "depends_on_stage": depends_on, "command": command})
    for participant in config["participants"]:
        add("signal_cache", f"python tools/build_signal_cache.py --participant {participant}", "audit")
        for seed in config["seeds"]:
            add("secondary_upstream", f"python tools/prepare_secondary_camera.py --participant {participant} --seed {seed} --device cuda --execute", "audit")
            for condition in ("A1", "A3", "A4", "A5", "A6", "A7"):
                dependency = "signal_cache+secondary_upstream" if condition in {"A1", "A3", "A7"} else "signal_cache"
                add("train", f"python tools/train_condition.py --condition {condition} --participant {participant} --seed {seed} --device cuda", dependency)
            add("A2", f"python tools/evaluate_a2_late_fusion.py --participant {participant} --seed {seed}", "A1")
            for condition in ("A3", "A4", "A5", "A6", "A7"):
                add("stress", f"python tools/run_stress_tests.py --condition {condition} --participant {participant} --seed {seed} --device cuda", condition)
    for condition in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        add("bootstrap", f"python tools/paired_bootstrap.py --condition {condition}", "all fold×seed results")
    for condition in ("A1", "A3", "A4", "A5", "A6", "A7"):
        add("latency", f"python tools/benchmark_latency.py --condition {condition} --participant A --seed 1 --device cuda", condition)
    add("summary", "python tools/summarize_phase_a.py", "bootstrap+stress+latency")
    output = PACKAGE_ROOT / "scripts" / "phase_a_job_matrix.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(jobs[0])); writer.writeheader(); writer.writerows(jobs)
    print(f"wrote {len(jobs)} jobs to {output}")


if __name__ == "__main__":
    main()
