"""Quality checks only. Does not start formal training or modify original artifacts."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from fault_aug.core import TaskGraph, build_examples, dump, load_config, read_json, read_jsonl, resolve
from fault_aug.runtime import cache_path, check, file_hash, forward, initialize, loader, load_pt, prepared_path
from graph_history.metrics import aggregate_node_probabilities


def main():
    c = load_config(ROOT/"config/experiment_config.json")
    report = {}
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_*.py")
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report["unit_tests"] = {"run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"log":stream.getvalue()}
    if not result.wasSuccessful():
        dump(ROOT/"verification/validation_summary.json",report)
        raise RuntimeError(stream.getvalue())
    checks = check(c,c["grid"]["participants"],c["grid"]["seeds"])
    report["feature_caches_checked"] = len(checks)
    source_paths = {}
    adm = Path(c["paths"]["adm_root"])
    for name in ("__init__.py","constants.py","models.py","metrics.py"):
        source_paths[f"vendor/graph_history/{name}"] = adm/"graph_history"/name
    atomic = adm.parent/"atomic_tail_A0_A8_windows_2026-08-19"
    for name in ("__init__.py","graph.py","augmentation.py"):
        source_paths[f"vendor/atomic_tail_exp/{name}"] = atomic/"atomic_tail_exp"/name
    source_paths["assets/integrated_task_graph_latest.json"] = atomic/"assets/integrated_task_graph_latest.json"
    report["vendor_integrity"] = [{"local":name,"original":str(source),"sha256":file_hash(ROOT/name),"matches_original":file_hash(ROOT/name)==file_hash(source)} for name,source in source_paths.items()]
    assert all(r["matches_original"] for r in report["vendor_integrity"])
    counts = Counter()
    for row in read_json(ROOT/"verification/augmentation_plan_audit.json"):
        counts.update(row["error_count_by_category"])
    report["mask_audit_100_epochs"] = {}
    for category in ("stage2_without_stage3","stage2_with_stage3","real_fault_unchanged"):
        values = {int(k.split("|")[1]):v for k,v in counts.items() if k.split("|")[0]==category}
        total = sum(values.values())
        report["mask_audit_100_epochs"][category] = {"run_epoch_draws":total,"count_histogram":values,"frequencies":{k:v/total for k,v in values.items()}}
    smoke = []
    for group in c["groups"]:
        directory = resolve(c["paths"]["smoke_output_root"])/group["id"]/"A_as_test/seed_1"
        done = read_json(directory/"completed.json")
        logs = read_json(directory/"train_log.json")
        assert done["smoke_only"] and done["epochs"]==2
        smoke.append({"group":group["id"],"initialization_sha256":done["initialization_sha256"],
                      "epoch_mask_sha256":[r["run_mask_sha256"] for r in logs],"trained_batches":[r["optimizer_steps"] for r in logs],
                      "test_samples":done["test_samples"]})
    assert len({r["initialization_sha256"] for r in smoke})==1
    augmented = [r for r in smoke if r["group"].endswith("FaultAug")]
    assert len({tuple(r["epoch_mask_sha256"]) for r in augmented})==1
    report["smoke_runs"] = smoke
    graph = TaskGraph.load(resolve(c["paths"]["task_graph"]))
    # Exercise actual histories across refresh boundaries without training 100 epochs.
    train_rows = read_jsonl(prepared_path(c,"A","train"))
    every20 = next(g for g in c["groups"] if g["id"]=="A1-Legacy-Every20-Control")
    refresh_audit,previous = [],None
    for start in [1,21,41,61,81]:
        first,_,stats = build_examples(c,train_rows,graph,every20,1,start,"A",True)
        last,_,_ = build_examples(c,train_rows,graph,every20,1,start+19,"A",True)
        assert first==last
        changed = None if previous is None else sum(a["history"]!=b["history"] for a,b in zip(first,previous))
        if previous is not None: assert changed>0
        refresh_audit.append({"start_epoch":start,"end_epoch":start+19,"round":stats["shuffle_refresh_round"],
                              "targets":len(first),"same_order_within_window":True,"changed_from_previous_window":changed})
        previous = first
    report["every20_shuffle_audit"] = refresh_audit
    dump(ROOT/"verification/every20_shuffle_audit.json",refresh_audit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression = []
    import csv
    with torch.no_grad():
        for fold in c["grid"]["participants"]:
            rows = read_jsonl(prepared_path(c,fold,"test"))
            examples,_,_ = build_examples(c,rows,graph,{"model":"M2","fault_augmentation":False},1,0,fold,False)
            for seed in c["grid"]["seeds"]:
                base = adm/"outputs"/f"{fold}_as_test"/("cam_"+c["grid"]["camera_id"])/f"seed_{seed}"/"history_models/direct_head_fusion/all_runs/m2_direct"
                model = initialize(seed,c["model"]).to(device)
                model.load_state_dict(load_pt(base/"last.pth")["model_state_dict"],strict=True)
                model.eval()
                with (base/"test_results/test_all_predictions.csv").open(encoding="utf-8-sig",newline="") as f:
                    old = {r["sample_name"]:r for r in csv.DictReader(f)}
                node_diff,tier_diff,n = 0,0,0
                for batch in loader(c,load_pt(cache_path(c,fold,seed,"test")),examples,False,seed):
                    prob = forward(model,batch,device).softmax(-1)
                    tier = aggregate_node_probabilities(prob,torch.tensor(graph.node_to_tier3,device=device),31)
                    pn,pt = prob.argmax(-1),tier.argmax(-1)
                    for i,r in enumerate(batch["rows"]):
                        expected = old[r["sample_name"]]
                        node_diff += int(int(pn[i])+1 != int(expected["pred_node_idx"]))
                        tier_diff += int(int(pt[i]) != int(expected["pred_tier3_id"]))
                        n += 1
                regression.append({"fold":fold,"seed":seed,"samples":n,"node_prediction_mismatches":node_diff,"tier3_prediction_mismatches":tier_diff})
                assert node_diff==tier_diff==0, regression[-1]
    report["old_m2_checkpoint_regression"] = regression
    report["status"] = "passed"
    report["formal_training_started"] = any(resolve(c["paths"]["output_root"]).glob("*/**/completed.json"))
    dump(ROOT/"verification/validation_summary.json",report)
    print(f"PASS: {result.testsRun} unit tests, {len(checks)} caches, {len(smoke)} smoke runs, {sum(r['samples'] for r in regression)} legacy predictions reproduced")


if __name__=="__main__": main()
