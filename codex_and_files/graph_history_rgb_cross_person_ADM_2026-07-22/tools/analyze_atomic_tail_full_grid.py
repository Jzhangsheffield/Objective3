from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PARTICIPANTS = ("A", "D", "J", "M")
SEEDS = (1, 2, 42)
SCOPES = ("normal_only", "all_runs")
POLICIES = ("refresh_every_1", "refresh_every_10", "refresh_once")
SPLITS = ("test_normal", "test_fault", "test_all")
REFERENCES = ("m2_direct", "m3_direct", "m3_dynamic_direct_fusion")
REPEATED_NODES = {14, 15, 16, 17, 19, 20, 21, 22}


def io_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(io_path(path).read_text(encoding="utf-8"))


def graph_metadata(path: Path) -> tuple[dict[int, str], dict[int, int], set[int]]:
    payload = read_json(path)
    labels: dict[int, str] = {}
    tier3: dict[int, int] = {}
    immediate: set[int] = set()
    for node in payload["nodes"]:
        node_idx = int(node["node_idx"])
        if not 1 <= node_idx <= 35:
            continue
        labels[node_idx] = str(node["action_label_tier3"])
        tier3[node_idx] = int(node["action_id_tier3"])
        if node["execution_constraints"].get("must_immediately_previous_node") is not None:
            immediate.add(node_idx)
    return labels, tier3, immediate


def atomic_path(root: Path, participant: str, seed: int, scope: str, policy: str, split: str) -> Path:
    return (
        root / "at_ad" / f"{participant}_s{seed}" / scope / policy
        / "m3_atomic_tail_direct_fusion" / "test_results" / f"{split}_predictions.csv"
    )


def reference_path(root: Path, participant: str, seed: int, scope: str, model: str, split: str) -> Path:
    seed_root = root / f"{participant}_as_test" / "cam_001484412812" / f"seed_{seed}" / "history_models"
    family = "dynamic_epoch_shuffle" if model.startswith("m3_dynamic") else "direct_head_fusion"
    return seed_root / family / scope / model / "test_results" / f"{split}_predictions.csv"


def enrich(frame: pd.DataFrame, *, participant: str, seed: int, scope: str, split: str,
           model: str, policy: str = "") -> pd.DataFrame:
    frame = frame.copy()
    frame["held_out_participant"] = participant
    frame["seed"] = seed
    frame["train_scope"] = scope
    frame["test_split"] = split
    frame["model"] = model
    frame["policy"] = policy
    frame["node_correct"] = (frame["true_node_idx"] == frame["pred_node_idx"]).astype(float)
    frame["tier3_correct"] = (frame["true_tier3_id"] == frame["pred_tier3_id"]).astype(float)
    frame["sample_key"] = participant + ":" + frame["sample_name"].astype(str)
    return frame


def load_predictions(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atomic_frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    missing: list[str] = []
    for participant in PARTICIPANTS:
        for seed in SEEDS:
            for scope in SCOPES:
                for policy in POLICIES:
                    for split in SPLITS:
                        path = atomic_path(root, participant, seed, scope, policy, split)
                        if not io_path(path).is_file():
                            missing.append(str(path))
                            continue
                        frame = enrich(pd.read_csv(io_path(path)), participant=participant, seed=seed,
                                       scope=scope, split=split, model="atomic_direct", policy=policy)
                        atomic_frames.append(frame)
                        inventory.append({"kind": "atomic", "participant": participant, "seed": seed,
                                          "scope": scope, "policy": policy, "split": split,
                                          "rows": len(frame), "path": str(path)})
                for model in REFERENCES:
                    for split in SPLITS:
                        path = reference_path(root, participant, seed, scope, model, split)
                        if not io_path(path).is_file():
                            missing.append(str(path))
                            continue
                        frame = enrich(pd.read_csv(io_path(path)), participant=participant, seed=seed,
                                       scope=scope, split=split, model=model)
                        reference_frames.append(frame)
                        inventory.append({"kind": "reference", "participant": participant, "seed": seed,
                                          "scope": scope, "model": model, "split": split,
                                          "rows": len(frame), "path": str(path)})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction files:\n" + "\n".join(missing[:30]))
    return (pd.concat(atomic_frames, ignore_index=True),
            pd.concat(reference_frames, ignore_index=True), pd.DataFrame(inventory))


def seed_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return frame.groupby(group_cols + ["held_out_participant", "seed"], as_index=False).agg(
        node_accuracy=("node_correct", "mean"), tier3_accuracy=("tier3_correct", "mean"),
        samples=("sample_name", "size"))


def participant_first(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    seeds = seed_metrics(frame, group_cols)
    participants = seeds.groupby(group_cols + ["held_out_participant"], as_index=False).agg(
        node_accuracy=("node_accuracy", "mean"), tier3_accuracy=("tier3_accuracy", "mean"))
    return participants.groupby(group_cols, as_index=False).agg(
        node_accuracy=("node_accuracy", "mean"), node_participant_sd=("node_accuracy", "std"),
        tier3_accuracy=("tier3_accuracy", "mean"), tier3_participant_sd=("tier3_accuracy", "std"),
        participants=("held_out_participant", "nunique"))


def paired_metrics(atomic: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    a = seed_metrics(atomic[atomic.test_split == "test_all"], ["train_scope", "policy"])
    r = seed_metrics(references[references.test_split == "test_all"], ["train_scope", "model"])
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for policy in POLICIES:
            left = a[(a.train_scope == scope) & (a.policy == policy)]
            for model in REFERENCES:
                right = r[(r.train_scope == scope) & (r.model == model)]
                merged = left.merge(right, on=["train_scope", "held_out_participant", "seed"], suffixes=("_a", "_r"))
                for metric in ("node_accuracy", "tier3_accuracy"):
                    delta = (merged[f"{metric}_a"] - merged[f"{metric}_r"]) * 100
                    sd = float(delta.std(ddof=1))
                    half_width = 2.200985 * sd / np.sqrt(len(delta))
                    rows.append({"train_scope": scope, "policy": policy, "reference": model,
                                 "metric": metric, "mean_delta_pp": delta.mean(),
                                 "delta_sd_pp": sd, "descriptive_95ci_low_pp": delta.mean() - half_width,
                                 "descriptive_95ci_high_pp": delta.mean() + half_width,
                                 "positive": int((delta > 1e-12).sum()), "tie": int((delta.abs() <= 1e-12).sum()),
                                 "negative": int((delta < -1e-12).sum()), "pairs": len(delta),
                                 "min_delta_pp": delta.min(), "max_delta_pp": delta.max()})
    return pd.DataFrame(rows)


def seed_summary(atomic: pd.DataFrame) -> pd.DataFrame:
    seeds = seed_metrics(atomic[atomic.test_split == "test_all"], ["train_scope", "policy"])
    return seeds.groupby(["train_scope", "policy", "seed"], as_index=False).agg(
        node_accuracy=("node_accuracy", "mean"), tier3_accuracy=("tier3_accuracy", "mean"))


def sample_flips(atomic: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    a = atomic[atomic.test_split == "test_all"]
    r = references[references.test_split == "test_all"]
    rows: list[dict[str, Any]] = []
    keys = ["held_out_participant", "seed", "sample_name"]
    for scope in SCOPES:
        for policy in POLICIES:
            left = a[(a.train_scope == scope) & (a.policy == policy)]
            for model in REFERENCES:
                right = r[(r.train_scope == scope) & (r.model == model)]
                merged = left.merge(right, on=keys, suffixes=("_a", "_r"))
                for metric in ("node_correct", "tier3_correct"):
                    fixed = int(((merged[f"{metric}_a"] == 1) & (merged[f"{metric}_r"] == 0)).sum())
                    regressed = int(((merged[f"{metric}_a"] == 0) & (merged[f"{metric}_r"] == 1)).sum())
                    rows.append({"train_scope": scope, "policy": policy, "reference": model,
                                 "metric": metric, "rows": len(merged), "fixed": fixed,
                                 "regressed": regressed, "net_fixed": fixed - regressed})
    return pd.DataFrame(rows)


def policy_pairs(atomic: pd.DataFrame) -> pd.DataFrame:
    a = seed_metrics(atomic[atomic.test_split == "test_all"], ["train_scope", "policy"])
    comparisons = (("refresh_once", "refresh_every_1"), ("refresh_once", "refresh_every_10"),
                   ("refresh_every_10", "refresh_every_1"))
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for lhs, rhs in comparisons:
            l = a[(a.train_scope == scope) & (a.policy == lhs)]
            r = a[(a.train_scope == scope) & (a.policy == rhs)]
            m = l.merge(r, on=["train_scope", "held_out_participant", "seed"], suffixes=("_l", "_r"))
            for metric in ("node_accuracy", "tier3_accuracy"):
                d = (m[f"{metric}_l"] - m[f"{metric}_r"]) * 100
                rows.append({"train_scope": scope, "lhs": lhs, "rhs": rhs, "metric": metric,
                             "mean_delta_pp": d.mean(), "positive": int((d > 1e-12).sum()),
                             "tie": int((d.abs() <= 1e-12).sum()), "negative": int((d < -1e-12).sum()),
                             "pairs": len(d)})
    return pd.DataFrame(rows)


def scope_pairs(atomic: pd.DataFrame) -> pd.DataFrame:
    a = seed_metrics(atomic[atomic.test_split == "test_all"], ["train_scope", "policy"])
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        l = a[(a.train_scope == "all_runs") & (a.policy == policy)]
        r = a[(a.train_scope == "normal_only") & (a.policy == policy)]
        m = l.merge(r, on=["policy", "held_out_participant", "seed"], suffixes=("_all", "_normal"))
        for metric in ("node_accuracy", "tier3_accuracy"):
            d = (m[f"{metric}_all"] - m[f"{metric}_normal"]) * 100
            rows.append({"policy": policy, "metric": metric, "mean_delta_pp": d.mean(),
                         "positive": int((d > 1e-12).sum()), "tie": int((d.abs() <= 1e-12).sum()),
                         "negative": int((d < -1e-12).sum()), "pairs": len(d)})
    return pd.DataFrame(rows)


def participant_summary(atomic: pd.DataFrame) -> pd.DataFrame:
    a = atomic[atomic.test_split == "test_all"]
    return a.groupby(["train_scope", "policy", "held_out_participant", "seed"], as_index=False).agg(
        node_accuracy=("node_correct", "mean"), tier3_accuracy=("tier3_correct", "mean")).groupby(
        ["train_scope", "policy", "held_out_participant"], as_index=False).agg(
        node_accuracy=("node_accuracy", "mean"), node_seed_sd=("node_accuracy", "std"),
        tier3_accuracy=("tier3_accuracy", "mean"), tier3_seed_sd=("tier3_accuracy", "std"))


def stage_summary(atomic: pd.DataFrame, immediate: set[int]) -> pd.DataFrame:
    a = atomic[atomic.test_split == "test_all"].copy()
    a["is_immediate"] = a.true_node_idx.isin(immediate)
    a["is_repeated"] = a.true_node_idx.isin(REPEATED_NODES)
    pieces: list[pd.DataFrame] = []
    for kind, selector in (("stage_1", a.stage_id == 1), ("stage_2", a.stage_id == 2),
                           ("stage_3", a.stage_id == 3),
                           ("immediate", a.is_immediate), ("non_immediate", ~a.is_immediate),
                           ("repeated_node", a.is_repeated)):
        s = a[selector].groupby(["train_scope", "policy", "held_out_participant", "seed"], as_index=False).agg(
            accuracy=("node_correct", "mean"))
        s = s.groupby(["train_scope", "policy"], as_index=False).agg(accuracy=("accuracy", "mean"))
        s["subset"] = kind
        pieces.append(s)
    return pd.concat(pieces, ignore_index=True)


def node_failures(frame: pd.DataFrame, labels: dict[int, str], tier3: dict[int, int],
                  group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        prefix = dict(zip(group_cols, keys))
        for node, ng in group.groupby("true_node_idx"):
            errors = ng[ng.node_correct == 0]
            top = errors.pred_node_idx.value_counts()
            pred = int(top.index[0]) if len(top) else -1
            rows.append({**prefix, "true_node_idx": int(node), "label": labels.get(int(node), ""),
                         "support": len(ng), "recall": ng.node_correct.mean(), "errors": len(errors),
                         "top_pred_node_idx": pred, "top_pred_label": labels.get(pred, ""),
                         "top_confusion_count": int(top.iloc[0]) if len(top) else 0,
                         "top_confusion_same_tier3": bool(pred >= 0 and tier3.get(int(node)) == tier3.get(pred))})
    return pd.DataFrame(rows)


def confusion_summary(frame: pd.DataFrame, labels: dict[int, str], tier3: dict[int, int],
                      group_cols: list[str]) -> pd.DataFrame:
    errors = frame[frame.node_correct == 0]
    out = errors.groupby(group_cols + ["true_node_idx", "pred_node_idx"], as_index=False).size()
    out = out.rename(columns={"size": "count"}).sort_values(group_cols + ["count"], ascending=[True] * len(group_cols) + [False])
    out["true_label"] = out.true_node_idx.map(labels)
    out["pred_label"] = out.pred_node_idx.map(labels)
    out["same_tier3"] = [tier3.get(int(t)) == tier3.get(int(p)) for t, p in zip(out.true_node_idx, out.pred_node_idx)]
    return out


def error_characteristics(frame: pd.DataFrame, tier3: dict[int, int], immediate: set[int],
                          group_cols: list[str]) -> pd.DataFrame:
    errors = frame[frame.node_correct == 0].copy()
    errors["same_tier3_pair"] = [tier3.get(int(t)) == tier3.get(int(p)) for t, p in zip(errors.true_node_idx, errors.pred_node_idx)]
    errors["immediate_true"] = errors.true_node_idx.isin(immediate)
    errors["repeated_true"] = errors.true_node_idx.isin(REPEATED_NODES)
    rows: list[dict[str, Any]] = []
    for keys, g in errors.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(group_cols, keys)), "errors": len(g),
                     "same_tier3_error_fraction": g.same_tier3_pair.mean(),
                     "cross_tier3_error_fraction": 1 - g.same_tier3_pair.mean(),
                     "high_confidence_ge_0_9_fraction": (g.node_confidence >= .9).mean(),
                     "mean_error_confidence": g.node_confidence.mean(),
                     "immediate_error_fraction": g.immediate_true.mean(),
                     "repeated_node_error_fraction": g.repeated_true.mean()})
    return pd.DataFrame(rows)


def consensus_errors(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols + ["held_out_participant", "sample_name", "true_node_idx"], as_index=False).agg(
        seeds=("seed", "nunique"), correct_seeds=("node_correct", "sum"),
        mean_confidence=("node_confidence", "mean"), stage_id=("stage_id", "first"))
    return grouped[(grouped.seeds == 3) & (grouped.correct_seeds == 0)].copy()


def audit_and_config(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    max_metric_diff = 0.0
    for participant in PARTICIPANTS:
        for seed in SEEDS:
            for scope in SCOPES:
                for policy in POLICIES:
                    run = root / "at_ad" / f"{participant}_s{seed}" / scope / policy / "m3_atomic_tail_direct_fusion"
                    completed = read_json(run / "completed.json")
                    config = read_json(run / "experiment_config.json")
                    audit = read_json(run / "shuffle_audit.json")
                    expected = {"model": "m3_atomic_tail_direct_fusion", "train_scope": scope,
                                "uses_current_target_for_reordering": False, "uses_m0_checkpoint": False,
                                "uses_logit_delta": False, "node_head_initialization": "random_trainable"}
                    for key, value in expected.items():
                        actual = completed.get(key, config.get(key))
                        if actual != value:
                            violations.append(f"{run}: {key}={actual!r}, expected={value!r}")
                    if int(config.get("epochs", -1)) != 50:
                        violations.append(f"{run}: epochs={config.get('epochs')}")
                    pred = pd.read_csv(io_path(run / "test_results" / "test_all_predictions.csv"))
                    metric = read_json(run / "test_results" / "test_all_metrics.json")
                    recalculated = (pred.true_node_idx == pred.pred_node_idx).mean()
                    max_metric_diff = max(max_metric_diff, abs(recalculated - float(metric["node"]["accuracy"])))
                    rows.append({"participant": participant, "seed": seed, "train_scope": scope, "policy": policy,
                                 "atomic_tail_applied_fraction": audit["atomic_tail_applied_fraction"],
                                 "multiple_orders_fraction": audit["samples_with_multiple_orders_fraction"],
                                 "different_from_actual_fraction": audit["samples_ever_different_from_actual_fraction"],
                                 "atomic_tail_violations": audit["atomic_tail_violations"],
                                 "epochs_audited": audit["epochs_audited"],
                                 "unique_refresh_rounds": len(audit["unique_refresh_rounds"])})
    return pd.DataFrame(rows), {"config_violations": violations, "max_test_all_metric_recalculation_diff": max_metric_diff}


def save(frame: pd.DataFrame, output: Path, name: str) -> None:
    frame.to_csv(io_path(output / name), index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = project / "outputs"
    labels, tier3, immediate = graph_metadata(project / "assets" / "integrated_task_graph_latest.json")
    atomic, references, inventory = load_predictions(root)
    audit, audit_check = audit_and_config(root)

    overall = participant_first(atomic, ["train_scope", "policy", "test_split"])
    reference_overall = participant_first(references, ["train_scope", "model", "test_split"])
    pairs = paired_metrics(atomic, references)
    policy_comparisons = policy_pairs(atomic)
    scope_comparisons = scope_pairs(atomic)
    seeds = seed_summary(atomic)
    flips = sample_flips(atomic, references)
    participants = participant_summary(atomic)
    stages = stage_summary(atomic, immediate)
    test_all = atomic[atomic.test_split == "test_all"]
    nodes = node_failures(test_all, labels, tier3, ["train_scope", "policy"])
    participant_nodes = node_failures(test_all, labels, tier3, ["train_scope", "policy", "held_out_participant"])
    confusions = confusion_summary(test_all, labels, tier3, ["train_scope", "policy"])
    participant_confusions = confusion_summary(test_all, labels, tier3, ["train_scope", "policy", "held_out_participant"])
    error_chars = error_characteristics(test_all, tier3, immediate, ["train_scope", "policy"])
    consensus = consensus_errors(test_all, ["train_scope", "policy"])

    for name, frame in (("inventory.csv", inventory), ("atomic_overall.csv", overall),
                        ("reference_overall.csv", reference_overall), ("paired_references.csv", pairs),
                        ("paired_policies.csv", policy_comparisons), ("paired_scopes.csv", scope_comparisons),
                        ("seed_summary.csv", seeds), ("sample_flips.csv", flips),
                        ("participant_summary.csv", participants), ("stage_and_mechanism.csv", stages),
                        ("node_failures.csv", nodes), ("participant_node_failures.csv", participant_nodes),
                        ("confusions.csv", confusions), ("participant_confusions.csv", participant_confusions),
                        ("error_characteristics.csv", error_chars), ("consensus_errors.csv", consensus),
                        ("shuffle_audit.csv", audit)):
        save(frame, output, name)

    best = overall[overall.test_split == "test_all"].sort_values(
        ["train_scope", "node_accuracy"], ascending=[True, False]).groupby("train_scope", as_index=False).first()
    summary = {
        "atomic_training_units": 72,
        "atomic_prediction_files": 216,
        "atomic_prediction_rows": int(len(atomic)),
        "reference_prediction_files": 216,
        "best_policy_by_scope": best[["train_scope", "policy", "node_accuracy", "tier3_accuracy"]].to_dict("records"),
        "audit": audit_check,
        "tail_violations_total": int(audit.atomic_tail_violations.sum()),
        "tail_applied_fraction_mean": float(audit.atomic_tail_applied_fraction.mean()),
        "tail_applied_fraction_min": float(audit.atomic_tail_applied_fraction.min()),
        "tail_applied_fraction_max": float(audit.atomic_tail_applied_fraction.max()),
    }
    io_path(output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
