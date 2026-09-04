from __future__ import annotations

import copy
import csv
import hashlib
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .core import (ROOT, TaskGraph, build_examples, digest, dump, edit_run, group_runs,
                   is_fault, jsonl, lr_multiplier, read_json, read_jsonl, resolve, run_type, seed_for)
from graph_history.models import build_direct_context_model
from graph_history.metrics import aggregate_node_probabilities, classification_metrics


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def prepared_path(c, fold, split):
    return resolve(c["paths"]["input_root"]) / f"{fold}_as_test" / f"{split}.jsonl"


def cache_path(c, fold, seed, split):
    return Path(c["paths"]["adm_root"]) / "outputs" / f"{fold}_as_test" / ("cam_"+c["grid"]["camera_id"]) / f"seed_{seed}" / "features/retrained_all_runs" / f"{split}_all.pt"


def prepare(c):
    report = {"fault_registry": c["fault_runs"], "folds": {}, "warnings": []}
    master_test = {}
    for fold in c["grid"]["participants"]:
        base = Path(c["paths"]["adm_root"]) / "outputs" / f"{fold}_as_test" / ("cam_"+c["grid"]["camera_id"]) / "protocols"
        old_faults = set(read_json(base/"protocol_report.json")["global_fault_runs"])
        splits = {}
        for split, source in (("train","train.jsonl"),("test","test_all.jsonl")):
            rows = read_jsonl(base/"all_runs"/source)
            if len({r["sample_name"] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate sample_name in {fold}/{split}")
            for r in rows:
                r["is_fault"] = is_fault(c,r)
                r["legacy_is_fault"] = (r["participant"]+"|"+r["run"]) in old_faults
                if split == "train" and r["participant"] == fold or split == "test" and r["participant"] != fold:
                    raise ValueError("Cross-person leakage")
                if not 1 <= int(r["node_idx"]) <= 35:
                    raise ValueError("Start/end/background cannot be an action target")
            dest = prepared_path(c,fold,split)
            if dest.exists():
                if read_jsonl(dest) != rows:
                    raise ValueError(f"Existing prepared manifest differs; use a new input_root: {dest}")
            else:
                jsonl(dest,rows)
            splits[split] = rows
        train_names = {r["sample_name"] for r in splits["train"]}
        test_names = {r["sample_name"] for r in splits["test"]}
        if train_names & test_names:
            raise ValueError("Train/test sample overlap")
        for split,rows in splits.items():
            count = Counter()
            for (p,run), rs in group_runs(rows):
                category = "real_fault" if is_fault(c,rs[0]) else run_type(rs)
                count[category] += 1
                if len({r["node_idx"] for r in rs}) != len(rs):
                    report["warnings"].append(f"{fold}/{split}/{p}/{run}: repeated node; A1 keeps real order once repeats enter history")
            report["folds"].setdefault(fold,{})[split] = {"nodes":len(rows), "normal_nodes":sum(not r["is_fault"] for r in rows), "fault_nodes":sum(r["is_fault"] for r in rows), "runs":dict(count), "manifest_sha256":file_hash(prepared_path(c,fold,split))}
        master_test[fold] = splits["test"]
    # Verify all_runs means the exact other-person union, not the sequence-disjoint subset.
    for fold in c["grid"]["participants"]:
        union = {r["sample_name"] for p,rs in master_test.items() if p != fold for r in rs}
        actual = {r["sample_name"] for r in read_jsonl(prepared_path(c,fold,"train"))}
        if actual != union:
            raise ValueError(f"{fold}: ADM train set is not the full other-person union")
    dump(resolve(c["paths"]["input_root"])/"preparation_report.json", report)
    return report


def check(c, folds, seeds):
    graph = TaskGraph.load(resolve(c["paths"]["task_graph"]))
    report = []
    for fold in folds:
        for seed in seeds:
            for split in ("train","test"):
                path = cache_path(c,fold,seed,split)
                cache = load_pt(path)
                rows = read_jsonl(prepared_path(c,fold,split))
                records = cache["records"]
                lookup = {r["sample_name"]:r for r in records}
                if len(lookup) != len(records) or set(lookup) != {r["sample_name"] for r in rows}:
                    raise ValueError(f"Cache/manifest membership mismatch: {path}")
                features = cache["features"]
                if tuple(features.shape) != (len(records),c["model"]["feature_dim"]) or not torch.isfinite(features).all():
                    raise ValueError(f"Invalid features: {path}")
                if str(cache["metadata"]["camera_id"]) != c["grid"]["camera_id"]:
                    raise ValueError("Wrong camera feature cache")
                for r in rows:
                    for k in ("participant","run","node_idx","tier3_id","stage_id","annotation_row_index"):
                        if str(r[k]) != str(lookup[r["sample_name"]][k]):
                            raise ValueError(f"Cache label/provenance mismatch: {r['sample_name']} {k}")
                    if graph.node_to_tier3[int(r["node_idx"])-1] != int(r["tier3_id"]):
                        raise ValueError("Graph-to-Tier3 mapping mismatch")
                report.append({"fold":fold,"seed":seed,"split":split,"path":str(path),"samples":len(rows),"feature_dim":features.shape[1],"sha256":file_hash(path),"metadata":cache["metadata"]})
    dump(ROOT/"verification/cache_preflight.json",report)
    return report


class CachedDataset(Dataset):
    def __init__(self, cache, examples):
        self.features = cache["features"].float()
        self.lookup = {r["sample_name"]:i for i,r in enumerate(cache["records"])}
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,i):
        ex = self.examples[i]
        ids = [self.lookup[r["sample_name"]] for r in ex["history"]]
        return {"current_feature": self.features[self.lookup[ex["current"]["sample_name"]]],
                "history_features": self.features[ids] if ids else self.features.new_zeros((0,self.features.shape[1])),
                "row":ex["current"],"history":ex["history"]}


def collate(items):
    n, width = len(items),items[0]["current_feature"].numel()
    length = max(len(x["history"]) for x in items)
    feat = torch.zeros(n,length,width)
    pos = torch.zeros(n,length,dtype=torch.long)
    mask = torch.ones(n,length,dtype=torch.bool)
    for i,item in enumerate(items):
        k = len(item["history"])
        if k:
            feat[i,:k] = item["history_features"]
            pos[i,:k] = torch.arange(k,0,-1)
            mask[i,:k] = False
    return {"current_feature":torch.stack([x["current_feature"] for x in items]),
            "history_features":feat,"history_position_ids":pos,"history_padding_mask":mask,
            "target":torch.tensor([int(x["row"]["node_idx"])-1 for x in items]),"rows":[x["row"] for x in items]}


def loader(c, cache, examples, shuffle, seed):
    # New dataset/loader each epoch: no stale worker copy of last epoch's deleted nodes.
    return DataLoader(CachedDataset(cache,examples),batch_size=c["training"]["batch_size"],shuffle=shuffle,
                      num_workers=c["training"]["num_workers"],persistent_workers=False,collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


def forward(model,batch,device):
    return model(**{k:batch[k].to(device) for k in ("current_feature","history_features","history_position_ids","history_padding_mask")})[0]


def initialize(seed,config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Both M2 and A1 use exactly this class. Only the training history builder differs.
    return build_direct_context_model("m2_direct",**config)


def state_hash(state):
    h = hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def metric_set(rows):
    return {"samples":len(rows),
            "node":classification_metrics([r["true_node_idx"]-1 for r in rows],[r["pred_node_idx"]-1 for r in rows],35),
            "tier3":classification_metrics([r["true_tier3_id"] for r in rows],[r["pred_tier3_id"] for r in rows],31)}


@torch.no_grad()
def evaluate(c,model,cache,rows,graph,device,directory,fold,seed):
    examples,_,_ = build_examples(c,rows,graph,{"model":"M2","fault_augmentation":False},seed,0,fold,False)
    model.eval()
    predictions, probabilities = [], []
    mapping = torch.tensor(graph.node_to_tier3,device=device)
    for batch in loader(c,cache,examples,False,seed):
        prob = forward(model,batch,device).softmax(-1)
        tier = aggregate_node_probabilities(prob,mapping,31)
        pn,pt = prob.argmax(-1),tier.argmax(-1)
        probabilities.append(prob.cpu())
        for i,row in enumerate(batch["rows"]):
            predictions.append({"sample_name":row["sample_name"],"participant":row["participant"],"run":row["run"],
                                "annotation_row_index":int(row["annotation_row_index"]),"stage_id":int(row["stage_id"]),
                                "is_fault":row["is_fault"],"legacy_is_fault":row["legacy_is_fault"],
                                "true_node_idx":int(row["node_idx"]),"pred_node_idx":int(pn[i])+1,
                                "true_tier3_id":int(row["tier3_id"]),"pred_tier3_id":int(pt[i]),
                                "node_confidence":float(prob[i,pn[i]]),"tier3_confidence":float(tier[i,pt[i]])})
    metrics = {}
    for split in c["evaluation"]["splits"]:
        selected = [r for r in predictions if split == "All" or r["is_fault"] == (split == "Fault")]
        metrics[split] = metric_set(selected)
        metrics[split]["per_stage"] = {str(s):metric_set([r for r in selected if r["stage_id"]==s]) for s in (1,2,3)}
        write_csv(directory/f"{split}_predictions.csv",selected)
    metrics["legacy_label_reference"] = {split:metric_set([r for r in predictions if r["legacy_is_fault"]==(split=="Fault")]) for split in ("Normal","Fault")}
    dump(directory/"metrics.json",metrics)
    atomic_save(directory/"probabilities.pt", {"rows":predictions,"node_probabilities":torch.cat(probabilities)})
    return metrics


def train_one(c,group,fold,seed,resume=False,smoke=False):
    c = copy.deepcopy(c)
    if smoke:
        c["paths"]["output_root"] = c["paths"]["smoke_output_root"]
        c["training"].update(epochs=2,warmup_epochs=1,save_every_epochs=1)
    t = c["training"]
    dest = resolve(c["paths"]["output_root"])/group["id"]/f"{fold}_as_test"/f"seed_{seed}"
    train_path,test_path = cache_path(c,fold,seed,"train"),cache_path(c,fold,seed,"test")
    source_hashes = {str(p.relative_to(ROOT)):file_hash(p) for folder in (ROOT/"fault_aug",ROOT/"vendor") for p in sorted(folder.rglob("*.py"))}
    spec = {"config":c,"group":group,"fold":fold,"seed":seed,"smoke_only":smoke,"source_hashes":source_hashes,
            "train_cache_sha256":file_hash(train_path),"test_cache_sha256":file_hash(test_path),
            "train_manifest_sha256":file_hash(prepared_path(c,fold,"train")),"test_manifest_sha256":file_hash(prepared_path(c,fold,"test")),
            "task_graph_sha256":file_hash(resolve(c["paths"]["task_graph"]))}
    fingerprint = digest(spec)
    if (dest/"completed.json").exists():
        if read_json(dest/"completed.json")["fingerprint"] != fingerprint:
            raise ValueError(f"Completed result configuration differs: {dest}; choose a new output_root")
        print(f"SKIP completed {dest}",flush=True)
        return
    if dest.exists() and any(dest.iterdir()) and not resume:
        raise FileExistsError(f"Incomplete result: {dest}; use --resume or a new output_root")
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"resolved_config.json").exists() and digest(read_json(dest/"resolved_config.json")) != fingerprint:
        raise ValueError("Resume code/config/input fingerprint mismatch")
    dump(dest/"resolved_config.json",spec)
    device = torch.device("cuda" if t["device"]=="auto" and torch.cuda.is_available() else "cpu" if t["device"]=="auto" else t["device"])
    torch.backends.cudnn.deterministic = t["deterministic"]
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(t["deterministic"])
    model = initialize(seed,c["model"]).to(device)
    init_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:lr_multiplier(i,t))
    logs,start = [],1
    if resume and (dest/"last.pth").exists():
        saved = load_pt(dest/"last.pth")
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint/config fingerprint mismatch")
        model.load_state_dict(saved["model_state_dict"],strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start = saved["epoch"]+1
        logs = saved["train_log"]
    graph = TaskGraph.load(resolve(c["paths"]["task_graph"]))
    train_cache = load_pt(train_path)
    train_rows = read_jsonl(prepared_path(c,fold,"train"))
    for epoch in range(start,t["epochs"]+1):
        begun = time.time()
        epoch_seed = seed_for("training-epoch",seed,fold,epoch)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        examples,plans,audit = build_examples(c,train_rows,graph,group,seed,epoch,fold,True)
        if max(len(x["history"]) for x in examples) > c["model"]["max_history"]:
            raise ValueError("History exceeds configured embedding capacity; no silent truncation")
        if c["audit"]["save_run_masks_every_epoch"]:
            jsonl(dest/"run_masks"/f"epoch_{epoch:03d}.jsonl",plans)
        if epoch==1 and c["audit"]["save_example_histories_epoch1"]:
            jsonl(dest/"epoch_001_histories.jsonl",[{"sample_name":x["current"]["sample_name"],"run":x["current"]["run"],"participant":x["current"]["participant"],"tail_reason":x["tail_reason"],"actual_history_samples":[r["sample_name"] for r in x["actual_history"]],"presented_history_samples":[r["sample_name"] for r in x["history"]],"presented_history_nodes":[r["node_idx"] for r in x["history"]],"position_ids":list(range(len(x["history"]),0,-1))} for x in examples])
        model.train()
        loss_sum,total,correct,steps = 0.0,0,0,0
        used_lr = optimizer.param_groups[0]["lr"]
        for batch in loader(c,train_cache,examples,True,epoch_seed):
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(device)
            logits = forward(model,batch,device)
            loss = torch.nn.functional.cross_entropy(logits,target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),t["gradient_clip_norm"],error_if_nonfinite=True)
            optimizer.step()
            total += target.numel(); correct += int((logits.argmax(-1)==target).sum())
            loss_sum += float(loss.detach())*target.numel(); steps += 1
            if smoke and steps>=3:
                break
        scheduler.step()
        audit.update(epoch=epoch,learning_rate=used_lr,train_loss=loss_sum/total,train_node_accuracy=correct/total,
                     optimizer_steps=steps,trained_targets=total,deleted_targets=sum(p["deleted_targets"] for p in plans),
                     normal_error_count_histogram=dict(Counter(p["error_count"] for p in plans if not p["is_real_fault"])),
                     normal_error_count_by_category={category:dict(Counter(p["error_count"] for p in plans if p["category"]==category)) for category in ("stage2_without_stage3","stage2_with_stage3")},
                     error_occurrence_counts=dict(Counter(e for p in plans for e in p["errors"])),
                     run_mask_sha256=digest(plans),seconds=time.time()-begun)
        logs.append(audit)
        dump(dest/"train_log.json",logs)
        if epoch % t["save_every_epochs"]==0 or epoch==t["epochs"]:
            atomic_save(dest/"last.pth",{"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                        "scheduler_state_dict":scheduler.state_dict(),"epoch":epoch,"train_log":logs,
                        "fingerprint":fingerprint,"initialization_sha256":init_hash,"model_config":c["model"]})
        print(f"{group['id']} {fold} seed={seed} epoch={epoch}/{t['epochs']} lr={used_lr:.8f} targets={total}/{len(examples)} deleted={audit['deleted_targets']} shuffle={audit['shuffle_changed_fraction']:.3f} loss={loss_sum/total:.4f}",flush=True)
    # No test-driven selection: inference runs only after final-epoch checkpoint is written.
    metrics = evaluate(c,model,load_pt(test_path),read_jsonl(prepared_path(c,fold,"test")),graph,device,dest/"test_results",fold,seed)
    dump(dest/"completed.json",{"fingerprint":fingerprint,"group":group,"fold":fold,"seed":seed,"epochs":t["epochs"],
         "smoke_only":smoke,"initialization_sha256":init_hash,"test_samples":metrics["All"]["samples"],
         "total_optimizer_steps":sum(x["optimizer_steps"] for x in logs),"checkpoint":"last.pth","status":"complete"})


def audit_plans(c,folds,seeds,epochs=10):
    rows = []
    for fold in folds:
        runs = group_runs(read_jsonl(prepared_path(c,fold,"train")))
        for seed in seeds:
            for epoch in range(1,epochs+1):
                plans = [edit_run(c,rs,seed,epoch,fold,True)[1] for _,rs in runs]
                counts = Counter((p["category"],p["error_count"]) for p in plans)
                rows.append({"fold":fold,"seed":seed,"epoch":epoch,"runs":len(plans),"deleted_targets":sum(p["deleted_targets"] for p in plans),
                             "error_count_by_category":{f"{k[0]}|{k[1]}":v for k,v in counts.items()},"plan_sha256":digest(plans)})
                if epoch==1:
                    jsonl(ROOT/"verification"/f"example_masks_{fold}_seed{seed}.jsonl",plans)
    dump(ROOT/"verification/augmentation_plan_audit.json",rows)
    return rows


def summarize(c):
    root = resolve(c["paths"]["output_root"])
    rows,pooled,pairs,coverage = [],[],[],[]
    for group in c["groups"]:
        for fold in c["grid"]["participants"]:
            for seed in c["grid"]["seeds"]:
                directory = root/group["id"]/f"{fold}_as_test"/f"seed_{seed}"
                complete = (directory/"completed.json").exists()
                coverage.append({"group":group["id"],"fold":fold,"seed":seed,"complete":complete})
                if not complete:
                    continue
                done = read_json(directory/"completed.json")
                if done["smoke_only"]:
                    raise ValueError("Do not summarize smoke outputs as experimental performance")
                metrics = read_json(directory/"test_results/metrics.json")
                for split in c["evaluation"]["splits"]:
                    m = metrics[split]
                    rows.append({"group":group["id"],"model":group["model"],"variant":group["variant"],"fault_augmentation":group["fault_augmentation"],"fold":fold,"seed":seed,"split":split,
                                 "samples":m["samples"],"node_accuracy":m["node"]["accuracy"],"tier3_accuracy":m["tier3"]["accuracy"],
                                 "node_macro_f1":m["node"]["macro_f1"],"tier3_macro_f1":m["tier3"]["macro_f1"],
                                 "node_balanced_accuracy":m["node"]["balanced_accuracy"],"tier3_balanced_accuracy":m["tier3"]["balanced_accuracy"]})
    def aggregation(data,key_fields,value_fields):
        keys = sorted({tuple(r[k] for k in key_fields) for r in data})
        out = []
        for key in keys:
            subset = [r for r in data if tuple(r[k] for k in key_fields)==key]
            row = dict(zip(key_fields,key)); row["n_units"] = len(subset)
            for field in value_fields:
                values = [r[field] for r in subset]
                row[field+"_mean"] = statistics.mean(values)
                row[field+"_sd"] = statistics.stdev(values) if len(values)>1 else 0.0
            out.append(row)
        return out
    fields = ["node_accuracy","tier3_accuracy","node_macro_f1","tier3_macro_f1","node_balanced_accuracy","tier3_balanced_accuracy"]
    for group in c["groups"]:
        for seed in c["grid"]["seeds"]:
            for split in c["evaluation"]["splits"]:
                selected = [r for r in rows if r["group"]==group["id"] and r["seed"]==seed and r["split"]==split]
                if len(selected)==len(c["grid"]["participants"]):
                    n = sum(r["samples"] for r in selected)
                    pooled.append({"group":group["id"],"seed":seed,"split":split,"samples":n,
                                   **{k:sum(r[k]*r["samples"] for r in selected)/n for k in ("node_accuracy","tier3_accuracy")}})
    for variant in ("M2-RealOrder","A1-Legacy-Once","A1-Legacy-Every20"):
        for fold in c["grid"]["participants"]:
            for seed in c["grid"]["seeds"]:
                for split in c["evaluation"]["splits"]:
                    found = {r["fault_augmentation"]:r for r in rows if (r["variant"],r["fold"],r["seed"],r["split"])==(variant,fold,seed,split)}
                    if set(found)=={False,True}:
                        if found[False]["samples"]!=found[True]["samples"]:
                            raise ValueError("Paired test counts differ")
                        pairs.append({"variant":variant,"fold":fold,"seed":seed,"split":split,"samples":found[False]["samples"],
                                      **{k+"_delta_pp":100*(found[True][k]-found[False][k]) for k in fields}})
    dest = root/"summary"
    write_csv(dest/"per_fold_seed.csv",rows)
    write_csv(dest/"mean_sd_12_fold_seed.csv",aggregation(rows,["group","split"],fields))
    write_csv(dest/"pooled_4fold_by_seed.csv",pooled)
    write_csv(dest/"pooled_mean_sd_3seed.csv",aggregation(pooled,["group","split"],["node_accuracy","tier3_accuracy"]))
    write_csv(dest/"paired_deltas.csv",pairs)
    write_csv(dest/"paired_delta_mean_sd.csv",aggregation(pairs,["variant","split"],[k+"_delta_pp" for k in fields]))
    dump(dest/"coverage.json",{"expected_training_runs":len(coverage),"completed":sum(r["complete"] for r in coverage),"partial":not all(r["complete"] for r in coverage),"runs":coverage})
    print(f"Summary: {sum(r['complete'] for r in coverage)}/{len(coverage)} completed (partial status recorded)")
