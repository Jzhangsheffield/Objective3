from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Required by deterministic CUDA GEMM; set before importing torch / creating CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fault_aug.core import load_config
from fault_aug.runtime import prepare, check, audit_plans, train_one, summarize


def main():
    parser = argparse.ArgumentParser(description="M2/A1 paired normal-run omission augmentation experiment")
    parser.add_argument("action",choices=["prepare","check","audit","train","smoke","summarize"])
    parser.add_argument("--config",type=Path,default=ROOT/"config/experiment_config.json")
    parser.add_argument("--folds",nargs="+")
    parser.add_argument("--seeds",type=int,nargs="+")
    parser.add_argument("--groups",nargs="+")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--audit-epochs",type=int,default=10)
    args = parser.parse_args()
    c = load_config(args.config)
    folds = args.folds or ([c["grid"]["participants"][0]] if args.action=="smoke" else c["grid"]["participants"])
    seeds = args.seeds or ([c["grid"]["seeds"][0]] if args.action=="smoke" else c["grid"]["seeds"])
    groups = [g for g in c["groups"] if args.groups is None or g["id"] in args.groups]
    if not set(folds)<=set(c["grid"]["participants"]) or not set(seeds)<=set(c["grid"]["seeds"]) or not groups or args.groups and not set(args.groups)<={g["id"] for g in c["groups"]}:
        parser.error("Unknown fold, seed or group")
    if args.action=="summarize":
        summarize(c); return
    prepare(c)
    if args.action=="prepare":
        print("Prepared corrected all-runs manifests without modifying source protocols"); return
    if args.action=="audit":
        if args.audit_epochs<1: parser.error("--audit-epochs must be positive")
        audit_plans(c,folds,seeds,args.audit_epochs)
        print("Augmentation mask audit complete; no model trained"); return
    check(c,folds,seeds)
    if args.action=="check":
        print("All selected feature caches passed membership, label and camera checks"); return
    for fold in folds:
        for seed in seeds:
            for group in groups:
                train_one(c,group,fold,seed,resume=args.resume,smoke=args.action=="smoke")
    if args.action=="train": summarize(c)
    else: print("Smoke finished: 2 epochs x at most 3 batches; NOT experiment results")


if __name__=="__main__":
    main()
