from __future__ import annotations

import json
import statistics
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    "M2-Direct-RealOrder",
    "A1-Legacy-Once",
    "A1-Legacy-Every10-Replace",
    "A3-DualPos-Once",
    "A3-DualPos-Every10",
]
PARTICIPANTS = ["A", "D", "J", "M"]
SEEDS = [1, 2, 42]
SPLITS = ["normal", "fault", "all"]
METRICS = [
    "node_accuracy", "node_macro_f1", "node_balanced_accuracy",
    "tier3_accuracy", "tier3_macro_f1", "tier3_balanced_accuracy",
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_sd(values):
    values = [float(v) for v in values]
    return {
        "n": len(values), "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def main() -> None:
    config = read_json(PACKAGE_ROOT / "config" / "experiment_config.json")
    records = []
    integrity = {"history_jobs": [], "upstream_jobs": []}
    for experiment in EXPERIMENTS:
        for participant in PARTICIPANTS:
            for seed in SEEDS:
                root = PACKAGE_ROOT / "outputs" / "history_models" / experiment / "all_runs" / f"{participant}_as_test" / f"seed_{seed}"
                required = [root / "completed.json", root / "last.pth", root / "train_log.json", root / "resolved_run_config.json"]
                missing = [str(path) for path in required if not path.is_file()]
                train_log = read_json(root / "train_log.json") if (root / "train_log.json").is_file() else []
                integrity["history_jobs"].append({
                    "experiment": experiment, "participant": participant, "seed": seed,
                    "missing": missing, "epochs": len(train_log),
                    "final_train_accuracy": train_log[-1].get("train_node_accuracy") if train_log else None,
                })
                for split in SPLITS:
                    metrics = read_json(root / "test_results_actual_order" / f"test_{split}_metrics.json")
                    row = {"experiment": experiment, "participant": participant, "seed": seed, "split": split, "samples": metrics["samples"]}
                    for level in ("node", "tier3"):
                        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                            row[f"{level}_{metric}"] = metrics[level][metric]
                    for stage, stage_metrics in metrics.get("per_stage", {}).items():
                        row[f"stage_{stage}_tier3_accuracy"] = stage_metrics["tier3"]["accuracy"]
                        row[f"stage_{stage}_tier3_macro_f1"] = stage_metrics["tier3"]["macro_f1"]
                    records.append(row)

    backbone_records = []
    for participant in PARTICIPANTS:
        for seed in SEEDS:
            root = PACKAGE_ROOT / "outputs" / "upstream" / f"{participant}_as_test" / f"cam_{config['grid']['camera_id']}" / f"seed_{seed}"
            backbone_root = root / "backbone" / "all_runs"
            feature_root = root / "features" / "retrained_all_runs"
            train_log = read_json(backbone_root / "train_log.json")
            required = [
                backbone_root / "completed.json", backbone_root / "last.pth", backbone_root / "train_log.json",
                feature_root / "completed.json", feature_root / "train_all.pt", feature_root / "test_all.pt",
                feature_root / "train_all.metadata.json", feature_root / "test_all.metadata.json",
            ]
            integrity["upstream_jobs"].append({
                "participant": participant, "seed": seed,
                "missing": [str(path) for path in required if not path.is_file()],
                "backbone_epochs": len(train_log), "first_lr": train_log[0]["lr"], "last_lr": train_log[-1]["lr"],
                "final_train_accuracy": train_log[-1]["accuracy"],
            })
            for split in SPLITS:
                metrics = read_json(backbone_root / "test_results" / f"test_{split}_metrics.json")
                backbone_records.append({
                    "participant": participant, "seed": seed, "split": split, "samples": metrics["samples"],
                    "tier3_accuracy": metrics["accuracy"], "tier3_macro_f1": metrics["macro_f1"],
                    "tier3_balanced_accuracy": metrics["balanced_accuracy"],
                })

    aggregates, fold_means = {}, {}
    for experiment in EXPERIMENTS:
        aggregates[experiment], fold_means[experiment] = {}, {}
        for split in SPLITS:
            subset = [r for r in records if r["experiment"] == experiment and r["split"] == split]
            aggregates[experiment][split] = {metric: mean_sd([r[metric] for r in subset]) for metric in METRICS}
            fold_means[experiment][split] = {
                participant: {metric: statistics.fmean([r[metric] for r in subset if r["participant"] == participant]) for metric in METRICS}
                for participant in PARTICIPANTS
            }

    lookup = {(r["experiment"], r["participant"], r["seed"], r["split"]): r for r in records}
    deltas_vs_m2 = {}
    for experiment in EXPERIMENTS[1:]:
        deltas_vs_m2[experiment] = {}
        for split in SPLITS:
            delta_rows = []
            for participant in PARTICIPANTS:
                for seed in SEEDS:
                    row, base = lookup[(experiment, participant, seed, split)], lookup[(EXPERIMENTS[0], participant, seed, split)]
                    delta_rows.append({
                        "participant": participant, "seed": seed,
                        "tier3_accuracy": row["tier3_accuracy"] - base["tier3_accuracy"],
                        "tier3_macro_f1": row["tier3_macro_f1"] - base["tier3_macro_f1"],
                        "node_accuracy": row["node_accuracy"] - base["node_accuracy"],
                    })
            entry = {}
            for metric in ("tier3_accuracy", "tier3_macro_f1", "node_accuracy"):
                values = [r[metric] for r in delta_rows]
                fold_values = [statistics.fmean([r[metric] for r in delta_rows if r["participant"] == p]) for p in PARTICIPANTS]
                entry[metric] = {
                    **mean_sd(values), "wins": sum(v > 1e-12 for v in values), "ties": sum(abs(v) <= 1e-12 for v in values),
                    "losses": sum(v < -1e-12 for v in values), "fold_mean_deltas": dict(zip(PARTICIPANTS, fold_values)),
                    "fold_wins": sum(v > 1e-12 for v in fold_values), "fold_ties": sum(abs(v) <= 1e-12 for v in fold_values),
                    "fold_losses": sum(v < -1e-12 for v in fold_values),
                }
            deltas_vs_m2[experiment][split] = entry

    refresh_pairs = {
        "A1 Every10 - Once": ("A1-Legacy-Every10-Replace", "A1-Legacy-Once"),
        "DualPos Every10 - Once": ("A3-DualPos-Every10", "A3-DualPos-Once"),
    }
    refresh_deltas = {}
    for label, (later, once) in refresh_pairs.items():
        refresh_deltas[label] = {}
        for split in SPLITS:
            delta_rows = []
            for participant in PARTICIPANTS:
                for seed in SEEDS:
                    a, b = lookup[(later, participant, seed, split)], lookup[(once, participant, seed, split)]
                    delta_rows.append({"participant": participant, "seed": seed,
                                       "tier3_accuracy": a["tier3_accuracy"] - b["tier3_accuracy"],
                                       "tier3_macro_f1": a["tier3_macro_f1"] - b["tier3_macro_f1"]})
            refresh_deltas[label][split] = {}
            for metric in ("tier3_accuracy", "tier3_macro_f1"):
                values = [r[metric] for r in delta_rows]
                fold_values = [statistics.fmean([r[metric] for r in delta_rows if r["participant"] == p]) for p in PARTICIPANTS]
                refresh_deltas[label][split][metric] = {
                    **mean_sd(values), "wins": sum(v > 1e-12 for v in values), "ties": sum(abs(v) <= 1e-12 for v in values),
                    "losses": sum(v < -1e-12 for v in values), "fold_mean_deltas": dict(zip(PARTICIPANTS, fold_values)),
                }

    backbone_aggregates = {}
    for split in SPLITS:
        subset = [r for r in backbone_records if r["split"] == split]
        backbone_aggregates[split] = {
            metric: mean_sd([r[metric] for r in subset])
            for metric in ("tier3_accuracy", "tier3_macro_f1", "tier3_balanced_accuracy")
        }
        backbone_aggregates[split]["fold_means"] = {
            participant: {metric: statistics.fmean([r[metric] for r in subset if r["participant"] == participant])
                          for metric in ("tier3_accuracy", "tier3_macro_f1", "tier3_balanced_accuracy")}
            for participant in PARTICIPANTS
        }

    stage_aggregates = {}
    for experiment in EXPERIMENTS:
        subset = [r for r in records if r["experiment"] == experiment and r["split"] == "all"]
        stage_aggregates[experiment] = {
            stage: {metric: mean_sd([r[f"stage_{stage}_{metric}"] for r in subset])
                    for metric in ("tier3_accuracy", "tier3_macro_f1")}
            for stage in ("1", "2", "3")
        }

    runtime_audits = {}
    for experiment in EXPERIMENTS:
        rows = [read_json(PACKAGE_ROOT / "outputs" / "history_models" / experiment / "all_runs" / f"{p}_as_test" / f"seed_{s}" / "augmentation_audit.json")
                for p in PARTICIPANTS for s in SEEDS]
        runtime_audits[experiment] = {
            key: mean_sd([r[key] for r in rows])
            for key in ("augmentation_changed_fraction", "mean_normalized_kendall_distance", "tail_aux_eligible_fraction",
                        "shifted_history_token_fraction", "mean_absolute_position_shift")
        }

    coverage_raw = read_json(PACKAGE_ROOT / "inputs" / "augmentation_coverage_all_folds.json")["results"]
    coverage = {}
    for experiment in EXPERIMENTS[1:]:
        rows = [r for r in coverage_raw if r["experiment_id"] == experiment]
        coverage[experiment] = {
            "by_fold": {r["participant"]: {key: r[key] for key in (
                "changed_fraction", "actual_test_prefix_coverage", "augmented_test_prefix_coverage",
                "test_prefixes_newly_covered_by_augmentation", "generated_augmented_histories")}
                        for r in rows},
            "mean_changed_fraction": statistics.fmean([r["changed_fraction"] for r in rows]),
            "mean_actual_test_prefix_coverage": statistics.fmean([r["actual_test_prefix_coverage"] for r in rows]),
            "mean_augmented_test_prefix_coverage": statistics.fmean([r["augmented_test_prefix_coverage"] for r in rows]),
            "total_new_test_prefixes_covered": sum(r["test_prefixes_newly_covered_by_augmentation"] for r in rows),
        }

    isolation_raw = read_json(PACKAGE_ROOT / "inputs" / "sequence_disjoint_all_folds_report.json")["folds"]
    isolation = {}
    for fold in isolation_raw:
        p = fold["test_participant"]
        isolation[p] = {
            "source_train_samples": fold["source_train"]["samples"], "source_train_runs": fold["source_train"]["runs"],
            "filtered_train_samples": fold["filtered_train"]["samples"], "filtered_train_runs": fold["filtered_train"]["runs"],
            "removed_samples": fold["source_train"]["samples"] - fold["filtered_train"]["samples"],
            "removed_runs": fold["source_train"]["runs"] - fold["filtered_train"]["runs"],
            "remaining_exact_overlap": fold.get("verification", {}).get("remaining_exact_node_sequence_overlap", 0),
            "normal_test_samples": fold["test_splits"]["test_normal"]["samples"],
            "fault_test_samples": fold["test_splits"]["test_fault"]["samples"],
            "all_test_samples": fold["test_splits"]["test_all"]["samples"],
            "missing_nodes": fold["filtered_train"]["missing_node_idx"], "missing_tier3": fold["filtered_train"]["missing_tier3_id"],
        }

    output = {
        "config": config, "integrity": integrity, "isolation": isolation, "records": records,
        "aggregates": aggregates, "fold_means": fold_means, "deltas_vs_m2": deltas_vs_m2,
        "refresh_deltas": refresh_deltas, "backbone_records": backbone_records,
        "backbone_aggregates": backbone_aggregates, "stage_aggregates": stage_aggregates,
        "runtime_augmentation_audits": runtime_audits, "coverage": coverage,
    }
    with (REPORT_ROOT / "analysis_data.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Integrity:", sum(not j["missing"] and j["epochs"] == 50 for j in integrity["history_jobs"]), "/ 60 history jobs")
    print("Upstream:", sum(not j["missing"] and j["backbone_epochs"] == 100 for j in integrity["upstream_jobs"]), "/ 12 jobs")
    for split in SPLITS:
        print("\n", split.upper(), "Tier3 accuracy / macro-F1")
        for experiment in EXPERIMENTS:
            a, f = aggregates[experiment][split]["tier3_accuracy"], aggregates[experiment][split]["tier3_macro_f1"]
            print(f"{experiment:30s} {a['mean']*100:6.2f}±{a['sd']*100:5.2f}  {f['mean']*100:6.2f}±{f['sd']*100:5.2f}")
    print("\nPaired Tier3 accuracy deltas vs M2")
    for experiment in EXPERIMENTS[1:]:
        print(experiment, {s: round(deltas_vs_m2[experiment][s]["tier3_accuracy"]["mean"]*100, 3) for s in SPLITS})
    print("\nRefresh deltas")
    for label in refresh_deltas:
        print(label, {s: round(refresh_deltas[label][s]["tier3_accuracy"]["mean"]*100, 3) for s in SPLITS})


if __name__ == "__main__":
    main()
