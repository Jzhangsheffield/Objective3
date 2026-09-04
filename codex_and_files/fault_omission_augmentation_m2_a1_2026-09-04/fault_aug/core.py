from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from atomic_tail_exp.augmentation import augment_history, stable_seed
from atomic_tail_exp.graph import TaskGraph


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed_for(*parts):
    return int(digest(parts)[:15], 16)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path):
    c = read_json(path)
    if c["grid"]["train_scope"] != "all_runs":
        raise ValueError("This package implements the agreed ADM all_runs protocol only")
    a = c["fault_augmentation"]
    probs = a["stage2_with_stage3"]["error_count_probabilities"]
    if set(probs) != {"0", "1", "2", "3"} or any(v < 0 or v > 1 for v in probs.values()) or not math.isclose(sum(probs.values()), 1):
        raise ValueError("0/1/2/3-error probabilities must be in [0,1] and sum to 1")
    for key in ("E1_probability", "E3_probability"):
        if not 0 <= a["stage2_without_stage3"][key] <= 1:
            raise ValueError(key)
    expected = {"E1": list(range(16,23)), "E3": [23], "E4": [28], "E5": [29], "E6": [30], "E7": [33], "E8": [26,27], "E9": [31], "E10": [34,35]}
    if a["error_nodes"] != expected:
        raise ValueError("Error/node definitions do not match the agreed E1, E3-E10 protocol")
    if c["a1"] != {"shuffle_refresh": "once", "position_mode": "presented", "active_tail_only": False, "sampling": "uniform", "fault_then_shuffle": True}:
        raise ValueError("A1 defaults must use once/presented-position uniform shuffle; Every20 is a group override")
    required_true = [a["stage2_without_stage3"]["independent"], a["stage2_without_stage3"]["include_stage1_stage2"], a["stage2_with_stage3"]["select_uniform_without_replacement"], a["exclude_synthetic_E2"], a["keep_real_fault_runs"], a["replace_run_not_append"], a["delete_targets_and_history"], a["same_plan_across_model_families"]]
    if not all(required_true) or a["refresh"] != "each_epoch_per_run" or a["partial_normal_stage3_policy"] != "error":
        raise ValueError("Unsupported augmentation mode; descriptive config fields cannot silently change semantics")
    t = c["training"]
    if t["batch_size"] < 1 or t["num_workers"] < 0 or t["save_every_epochs"] < 1:
        raise ValueError("Invalid batch/workers/checkpoint interval")
    for p,runs in c["confirmed_normal_corrections"].items():
        if set(runs) & set(c["fault_runs"].get(p,[])):
            raise ValueError("Confirmed normal runs cannot remain in the fault registry")
    if set(c["grid"]["participants"]) != {"A","D","J","M"}:
        raise ValueError("Prepared ADM protocol requires all four participants; use CLI --folds to select runs")
    if not 0 < t["min_learning_rate"] <= t["learning_rate"] or not 0 <= t["warmup_epochs"] < t["epochs"] or not 0 < t["warmup_start_factor"] <= 1:
        raise ValueError("Invalid LR schedule")
    if t["amp"] or t["retrain_backbone"] or not t["backbone_frozen"] or t["warm_start_checkpoint"] is not None:
        raise ValueError("FP32, frozen cached features, random head initialization are required")
    if t["scheduler"] != "linear_warmup_cosine_epoch" or t["checkpoint_policy"] != "last_epoch":
        raise ValueError("Unsupported scheduler/checkpoint policy")
    if c["evaluation"] != {"history_order": "actual", "fault_augmentation": False, "splits": ["Normal", "Fault", "All"], "corrected_fault_labels": True, "include_real_E2_nodes": True}:
        raise ValueError("Evaluation must retain the agreed real-history corrected-label protocol")
    if len({g["id"] for g in c["groups"]}) != 6 or {(g["variant"],g["fault_augmentation"]) for g in c["groups"]} != {(v,a) for v in ("M2-RealOrder","A1-Legacy-Once","A1-Legacy-Every20") for a in (False,True)}:
        raise ValueError("Exactly six paired groups (M2, A1 Once, A1 Every20) are required")
    for g in c["groups"]:
        expected_model = "M2" if g["variant"]=="M2-RealOrder" else "A1"
        expected_refresh = 20 if g["variant"]=="A1-Legacy-Every20" else "once"
        if g["model"]!=expected_model or g.get("shuffle_refresh","once")!=expected_refresh:
            raise ValueError("Group model/variant/shuffle_refresh mismatch")
    return c


def is_fault(c, row):
    p = row["participant"]
    if p not in c["fault_runs"]:
        raise ValueError(f"Unknown participant {p}; explicit fault registry required")
    return int(str(row["run"]).split("_")[-1]) in c["fault_runs"][p]


def group_runs(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["participant"], row["run"])].append(row)
    return [(key, sorted(value, key=lambda r:int(r["annotation_row_index"]))) for key,value in sorted(groups.items())]


def run_type(rows):
    nodes = {int(r["node_idx"]) for r in rows}
    if not set(range(12,26)).issubset(nodes):
        raise ValueError("Run labelled normal does not contain complete Stage 2")
    if nodes.intersection(range(26,36)):
        if not set(range(26,36)).issubset(nodes):
            raise ValueError("Partial Stage 3 in a normal run: do not silently treat absent nodes as errors")
        return "stage2_with_stage3"
    return "stage2_without_stage3"


def sample_errors(c, category, rng):
    a = c["fault_augmentation"]
    if category == "stage2_without_stage3":
        p = a[category]
        return [e for e in ("E1", "E3") if rng.random() < p[e + "_probability"]]
    if category != "stage2_with_stage3":
        raise ValueError(category)
    p = a[category]["error_count_probabilities"]
    count = rng.choices([0,1,2,3], weights=[p[str(k)] for k in range(4)], k=1)[0]
    return sorted(rng.sample(list(a["error_nodes"]), count), key=lambda x:int(x[1:]))


def edit_run(c, rows, seed, epoch, fold, enabled):
    first = rows[0]
    fault = is_fault(c, first)
    category = "real_fault_unchanged" if fault else run_type(rows)
    rng = random.Random(seed_for("fault-mask-v1", seed, epoch, fold, first["participant"], first["run"]))
    errors = sample_errors(c, category, rng) if enabled and not fault else []
    removed_nodes = {node for e in errors for node in c["fault_augmentation"]["error_nodes"][e]}
    kept = [r for r in rows if int(r["node_idx"]) not in removed_nodes]
    removed = [r for r in rows if int(r["node_idx"]) in removed_nodes]
    return kept, {
        "epoch": epoch, "fold": fold, "seed": seed, "participant": first["participant"], "run": first["run"],
        "is_real_fault": fault, "category": category, "errors": errors, "error_count": len(errors),
        "deleted_node_ids": sorted(removed_nodes), "deleted_sample_names": [r["sample_name"] for r in removed],
        "before_nodes": [int(r["node_idx"]) for r in rows], "after_nodes": [int(r["node_idx"]) for r in kept],
        "original_targets": len(rows), "retained_targets": len(kept), "deleted_targets": len(removed)
    }


def order_history(history, graph, model, seed):
    if model == "M2":
        return list(history), "real_order", False
    result = augment_history(history, graph, seed, False, "uniform", None, 16, .75, .35, 2, 1)
    return list(result.rows), result.decision.reason, result.changed


def shuffle_round(group, epoch):
    interval = group.get("shuffle_refresh", "once")
    return 0 if interval == "once" else (max(1,epoch)-1)//int(interval)


def build_examples(c, rows, graph, group, seed, epoch, fold, training):
    examples, plans = [], []
    reasons, class_support = Counter(), Counter()
    changed = 0
    for _, originals in group_runs(rows):
        if training:
            kept, plan = edit_run(c, originals, seed, epoch, fold, group["fault_augmentation"])
            plans.append(plan)
        else:
            kept = originals
        for i,current in enumerate(kept):
            actual = kept[:i]
            presented, reason, different = order_history(actual, graph, group["model"] if training else "M2", stable_seed(seed, shuffle_round(group,epoch), current["sample_name"]))
            assert len(presented) == len(actual)
            assert {r["sample_name"] for r in presented} == {r["sample_name"] for r in actual}
            assert all(int(r["annotation_row_index"]) < int(current["annotation_row_index"]) for r in presented)
            examples.append({"current": current, "history": presented, "actual_history": actual, "tail_reason": reason})
            reasons[reason] += 1
            class_support[int(current["node_idx"])] += 1
            changed += int(different)
    return examples, plans, {"targets":len(examples), "shuffle_refresh_round":shuffle_round(group,epoch), "shuffle_changed_targets":changed, "shuffle_changed_fraction":changed / max(1,len(examples)), "tail_reason_counts":dict(reasons), "retained_node_support":dict(sorted(class_support.items()))}


def lr_multiplier(index, training):
    """LambdaLR index 0 corresponds to epoch 1; warmup ends at peak LR."""
    e = index + 1
    total, warm = training["epochs"], training["warmup_epochs"]
    if warm and e <= warm:
        if warm == 1:
            return 1.0
        return training["warmup_start_factor"] + (1-training["warmup_start_factor"]) * (e-1)/(warm-1)
    progress = min(1.0, max(0.0, (e-warm)/(total-warm)))
    minimum = training["min_learning_rate"] / training["learning_rate"]
    return minimum + (1-minimum) * .5 * (1+math.cos(math.pi*progress))
