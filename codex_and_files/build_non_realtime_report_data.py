from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GH = ROOT / "codex_and_files" / "graph_history_rgb_cross_person_ADM_2026-07-22"
AT = ROOT / "codex_and_files" / "atomic_tail_A0_A8_windows_2026-08-19"
OUT = ROOT / "codex_and_files" / "non_realtime_experiment_summary_2026-08-24"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None, "nan", "NaN"):
        return None
    return float(value)


def metric_record(row: dict[str, str], family: str, scope: str, model: str, split: str) -> dict:
    return {
        "family": family,
        "scope": scope,
        "model": model,
        "split": split.replace("test_", ""),
        "node_accuracy": fnum(row, "mean_node_accuracy"),
        "node_macro_f1": fnum(row, "mean_node_macro_f1"),
        "node_balanced_accuracy": fnum(row, "mean_node_balanced_accuracy"),
        "tier3_accuracy": fnum(row, "mean_tier3_accuracy"),
        "tier3_macro_f1": fnum(row, "mean_tier3_macro_f1"),
        "tier3_balanced_accuracy": fnum(row, "mean_tier3_balanced_accuracy"),
        "node_accuracy_sd": fnum(row, "std_node_accuracy"),
        "node_macro_f1_sd": fnum(row, "std_node_macro_f1"),
        "tier3_accuracy_sd": fnum(row, "std_tier3_accuracy"),
        "tier3_macro_f1_sd": fnum(row, "std_tier3_macro_f1"),
        "aggregation": "participant-first: mean over 3 seeds within participant, then equal mean over A/D/J/M",
    }


records: list[dict] = []

# Historical J-as-test feasibility pilot (existing Tier-3 backbone, seeds 1/2/3).
pilot_root = ROOT / "codex_and_files" / "graph_history_rgb_experiments_2026-07-20" / "outputs" / "J_as_test" / "cam_001484412812" / "history_models"
pilot_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
for path in sorted(pilot_root.glob("existing_last*/experiment_summary.csv")):
    for row in read_csv(path):
        pilot_by_key[(row["train_scope"], row["model"], row["split"])].append(row)
for (scope, model, split), rows in sorted(pilot_by_key.items()):
    def pilot_stat(key: str):
        values = [float(r[key]) for r in rows if r.get(key, "") not in ("", "nan", "NaN")]
        if not values:
            return None, None
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0
    na, na_sd = pilot_stat("node_accuracy")
    nf, _ = pilot_stat("node_macro_f1")
    _, nf_sd = pilot_stat("node_macro_f1")
    nb, _ = pilot_stat("node_balanced_accuracy")
    ta, ta_sd = pilot_stat("tier3_accuracy")
    tf, _ = pilot_stat("tier3_macro_f1")
    _, tf_sd = pilot_stat("tier3_macro_f1")
    tb, _ = pilot_stat("tier3_balanced_accuracy")
    records.append({
        "family": "j_pilot_existing_backbone",
        "scope": scope,
        "model": model,
        "split": split.replace("test_", ""),
        "node_accuracy": na,
        "node_macro_f1": nf,
        "node_balanced_accuracy": nb,
        "tier3_accuracy": ta,
        "tier3_macro_f1": tf,
        "tier3_balanced_accuracy": tb,
        "node_accuracy_sd": na_sd,
        "node_macro_f1_sd": nf_sd,
        "tier3_accuracy_sd": ta_sd,
        "tier3_macro_f1_sd": tf_sd,
        "n_runs": len(rows),
        "aggregation": "historical J-only feasibility pilot; mean over seeds 1/2/3; existing Tier-3 backbone; not strict four-fold evidence",
    })

# Original M0-M6 and E2E models.
for scope in ("normal_only", "all_runs"):
    src = GH / "outputs" / f"cross_person_summary_{scope}_ADJM_3seeds" / "all_model_cross_person_aggregate.csv"
    for row in read_csv(src):
        records.append(metric_record(row, "original_and_e2e", scope, row["model"], row["split"]))

# Direct fusion models. Choose the M0-reference rows to avoid duplicated performance rows.
src = GH / "outputs" / "direct_head_fusion_summary_ADJM_3seeds" / "direct_head_aggregate.csv"
for row in read_csv(src):
    if row["reference_model"] == "m0":
        records.append(metric_record(row, "direct_head_fusion", row["train_scope"], row["model"], row["split"]))

# Dynamic models. Choose one canonical reference row per model.
src = GH / "outputs" / "dynamic_epoch_shuffle_summary_ADJM_3seeds" / "dynamic_epoch_shuffle_aggregate.csv"
dynamic_rows = read_csv(src)
for model in sorted({r["model"] for r in dynamic_rows}):
    for scope in ("normal_only", "all_runs"):
        for split in ("test_normal", "test_fault", "test_all"):
            candidates = [r for r in dynamic_rows if r["model"] == model and r["train_scope"] == scope and r["split"] == split]
            if candidates:
                records.append(metric_record(candidates[0], "dynamic_epoch_shuffle", scope, model, split))

# Original atomic-tail direct fusion, actual-order evaluation.
src = GH / "outputs" / "at_actual" / "atomic_actual_order_aggregate.csv"
for row in read_csv(src):
    records.append({
        "family": "atomic_tail_refresh_grid_actual_eval",
        "scope": row["train_scope"],
        "model": row["refresh_policy"],
        "split": row["split"].replace("test_", ""),
        "node_accuracy": fnum(row, "mean_atomic_node_accuracy"),
        "node_macro_f1": None,
        "node_balanced_accuracy": None,
        "tier3_accuracy": fnum(row, "mean_atomic_tier3_accuracy"),
        "tier3_macro_f1": None,
        "tier3_balanced_accuracy": None,
        "node_accuracy_sd": fnum(row, "participant_std_atomic_node_accuracy"),
        "node_macro_f1_sd": None,
        "tier3_accuracy_sd": fnum(row, "participant_std_atomic_tier3_accuracy"),
        "tier3_macro_f1_sd": None,
        "delta_node_vs_m2_direct": fnum(row, "mean_delta_node_accuracy"),
        "delta_tier3_vs_m2_direct": fnum(row, "mean_delta_tier3_accuracy"),
        "aggregation": "participant-first: mean over 3 seeds within participant, then equal mean over A/D/J/M",
    })

# A0-A4 package. Recompute 12 fold-seed mean/SD for every split.
experiments = ["A0", "A1", "A2", "A3", "A3-DualPos", "A4-DualPos"]
for exp in experiments:
    for split in ("normal", "fault", "all"):
        rows = []
        for path in sorted((AT / "outputs" / exp).glob(f"all_runs/*_as_test/seed_*/test_results_actual_order/test_{split}_metrics.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        if not rows:
            continue
        def mean_sd(values):
            return statistics.mean(values), statistics.stdev(values)
        na, na_sd = mean_sd([r["node"]["accuracy"] for r in rows])
        nf, nf_sd = mean_sd([r["node"]["macro_f1"] for r in rows])
        nb, _ = mean_sd([r["node"]["balanced_accuracy"] for r in rows])
        ta, ta_sd = mean_sd([r["tier3"]["accuracy"] for r in rows])
        tf, tf_sd = mean_sd([r["tier3"]["macro_f1"] for r in rows])
        tb, _ = mean_sd([r["tier3"]["balanced_accuracy"] for r in rows])
        records.append({
            "family": "atomic_tail_A0_A4_dualpos",
            "scope": "all_runs",
            "model": exp,
            "split": split,
            "node_accuracy": na,
            "node_macro_f1": nf,
            "node_balanced_accuracy": nb,
            "tier3_accuracy": ta,
            "tier3_macro_f1": tf,
            "tier3_balanced_accuracy": tb,
            "node_accuracy_sd": na_sd,
            "node_macro_f1_sd": nf_sd,
            "tier3_accuracy_sd": ta_sd,
            "tier3_macro_f1_sd": tf_sd,
            "n_runs": len(rows),
            "aggregation": "equal mean over 12 fold-seed runs (A/D/J/M x seeds 1/2/42)",
        })


OUT.mkdir(parents=True, exist_ok=True)

json_path = OUT / "performance_summary.json"
json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

csv_path = OUT / "performance_summary.csv"
columns = [
    "family", "scope", "model", "split", "node_accuracy", "node_accuracy_sd",
    "node_macro_f1", "node_macro_f1_sd", "node_balanced_accuracy", "tier3_accuracy", "tier3_accuracy_sd",
    "tier3_macro_f1", "tier3_macro_f1_sd", "tier3_balanced_accuracy", "delta_node_vs_m2_direct",
    "delta_tier3_vs_m2_direct", "n_runs", "aggregation",
]
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

print(json.dumps({
    "records": len(records),
    "families": sorted({r["family"] for r in records}),
    "json": str(json_path),
    "csv": str(csv_path),
}, ensure_ascii=False, indent=2))
