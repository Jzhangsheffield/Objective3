from __future__ import annotations

import copy
import math
import os
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from fault_aug.core import (TaskGraph, build_examples, edit_run, is_fault, load_config,
                            lr_multiplier, order_history, run_type, sample_errors, shuffle_round)
from fault_aug.runtime import CachedDataset, collate, initialize, state_hash


def records(nodes, p="A", run="run_1"):
    return [{"participant":p,"run":run,"sample_name":f"{p}_{run}_{i}","annotation_row_index":i+1,"node_idx":node,
             "tier3_id":0,"stage_id":1 if node<12 else 2 if node<=25 else 3} for i,node in enumerate(nodes)]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_config(ROOT/"config/experiment_config.json")
        cls.g = TaskGraph.load(ROOT/"assets/integrated_task_graph_latest.json")

    def test_error_mapping(self):
        a = self.c["fault_augmentation"]["error_nodes"]
        self.assertNotIn("E2",a)
        self.assertEqual(a["E1"],list(range(16,23)))
        self.assertEqual(a["E3"],[23])
        self.assertEqual(set(a["E1"]) & set(a["E3"]),set())
        self.assertEqual(a["E8"],[26,27]); self.assertEqual(a["E10"],[34,35])

    def test_independent_stage2_distribution(self):
        rng = random.Random(100)
        n = 30000
        counts = Counter(tuple(sample_errors(self.c,"stage2_without_stage3",rng)) for _ in range(n))
        for outcome,p in [((),.49),(("E1",),.21),(("E3",),.21),(("E1","E3"),.09)]:
            self.assertAlmostEqual(counts[outcome]/n,p,delta=.012)

    def test_stage23_distribution_and_no_replacement(self):
        rng = random.Random(101)
        n = 30000
        counts,errors = Counter(),Counter()
        for _ in range(n):
            chosen = sample_errors(self.c,"stage2_with_stage3",rng)
            self.assertEqual(len(chosen),len(set(chosen)))
            self.assertNotIn("E2",chosen)
            counts[len(chosen)] += 1; errors.update(chosen)
        for k,p in {0:.5,1:.2,2:.2,3:.1}.items():
            self.assertAlmostEqual(counts[k]/n,p,delta=.012)
        # Expected error count = 0.9, so every error has marginal selection probability 0.1.
        for e in self.c["fault_augmentation"]["error_nodes"]:
            self.assertAlmostEqual(errors[e]/n,.1,delta=.012)

    def test_stage_categories(self):
        self.assertEqual(run_type(records(range(12,26))),"stage2_without_stage3")
        self.assertEqual(run_type(records(range(1,26))),"stage2_without_stage3")
        self.assertEqual(run_type(records(range(12,36))),"stage2_with_stage3")
        with self.assertRaises(ValueError): run_type(records(list(range(12,26))+[28]))

    def test_confirmed_normal(self):
        for p,run in [("A",28),("J",31),("J",32),("J",34)]:
            self.assertFalse(is_fault(self.c,{"participant":p,"run":f"run_{run}"}))
        self.assertTrue(is_fault(self.c,{"participant":"J","run":"run_28"}))

    def test_forced_e1_e3_deletion_is_run_level(self):
        c = copy.deepcopy(self.c)
        c["fault_augmentation"]["stage2_without_stage3"].update(E1_probability=1.,E3_probability=1.)
        rows = records(range(12,26))
        kept,plan = edit_run(c,rows,1,1,"D",True)
        self.assertEqual([r["node_idx"] for r in kept],[12,13,14,15,24,25])
        self.assertEqual(plan["deleted_targets"],8)
        ex,_,_ = build_examples(c,rows,self.g,{"model":"M2","fault_augmentation":True},1,1,"D",True)
        self.assertEqual([r["node_idx"] for r in ex[-1]["history"]],[12,13,14,15,24])
        self.assertEqual(len(rows),14)  # original objects not modified

    def test_e1_retains_inspection(self):
        c = copy.deepcopy(self.c)
        c["fault_augmentation"]["stage2_without_stage3"].update(E1_probability=1.,E3_probability=0.)
        kept,_ = edit_run(c,records(range(12,26)),1,1,"D",True)
        self.assertEqual([r["node_idx"] for r in kept],[12,13,14,15,23,24,25])

    def test_fault_never_deleted_including_e2(self):
        rows = records(list(range(12,23))+list(range(16,23))+[23,24,25],p="J",run="run_27")
        for epoch in range(1,101):
            kept,plan = edit_run(self.c,rows,1,epoch,"A",True)
            self.assertEqual(kept,rows); self.assertEqual(plan["errors"],[])

    def test_same_masks_between_m2_and_a1(self):
        rows = records(range(1,36))
        for epoch in range(1,15):
            a,pa,_ = build_examples(self.c,rows,self.g,{"model":"M2","fault_augmentation":True},42,epoch,"D",True)
            b,pb,_ = build_examples(self.c,rows,self.g,{"model":"A1","fault_augmentation":True},42,epoch,"D",True)
            self.assertEqual(pa,pb)
            self.assertEqual([x["current"] for x in a],[x["current"] for x in b])

    def test_a1_once_and_missing_stage2(self):
        rows = records(range(1,36))
        a,_,_ = build_examples(self.c,rows,self.g,{"model":"A1","fault_augmentation":False},1,1,"D",True)
        b,_,_ = build_examples(self.c,rows,self.g,{"model":"A1","fault_augmentation":False},1,100,"D",True)
        self.assertEqual(a,b)
        hist = records([12,13,14,15,24])
        shuffled,reason,changed = order_history(hist,self.g,"A1",123)
        self.assertEqual(reason,"observed_atomic_nodes_not_prefix")
        self.assertEqual(shuffled,hist); self.assertFalse(changed)

    def test_repeated_history_fallback(self):
        hist = records(list(range(12,23))+[16])
        shuffled,reason,changed = order_history(hist,self.g,"A1",123)
        self.assertEqual(reason,"repeated_node_fallback")
        self.assertEqual(hist,shuffled); self.assertFalse(changed)

    def test_every20_round_boundaries(self):
        group = {"shuffle_refresh":20}
        for epoch in range(1,101):
            self.assertEqual(shuffle_round(group,epoch),(epoch-1)//20)
            self.assertEqual(shuffle_round({"shuffle_refresh":"once"},epoch),0)

    def test_every20_orders_and_first_round_match_once(self):
        rows = records(range(1,36))
        group = {"model":"A1","fault_augmentation":False,"shuffle_refresh":20}
        first,_,_ = build_examples(self.c,rows,self.g,group,1,1,"D",True)
        once,_,_ = build_examples(self.c,rows,self.g,{**group,"shuffle_refresh":"once"},1,100,"D",True)
        self.assertEqual(first,once)
        previous = None
        for start in [1,21,41,61,81]:
            a,_,audit = build_examples(self.c,rows,self.g,group,1,start,"D",True)
            b,_,_ = build_examples(self.c,rows,self.g,group,1,start+19,"D",True)
            self.assertEqual(a,b)
            self.assertEqual(audit["shuffle_refresh_round"],(start-1)//20)
            if previous is not None: self.assertNotEqual(a,previous)
            previous = a

    def test_all_three_variants_share_masks(self):
        rows = records(range(1,36))
        groups = [g for g in self.c["groups"] if g["fault_augmentation"]]
        self.assertEqual(len(groups),3)
        for epoch in [1,20,21,40,41,60,61,80,81,100]:
            outputs = [build_examples(self.c,rows,self.g,g,42,epoch,"D",True) for g in groups]
            for examples,plan,_ in outputs[1:]:
                self.assertEqual(plan,outputs[0][1])
                self.assertEqual([e["current"] for e in examples],[e["current"] for e in outputs[0][0]])

    def test_evaluation_never_augments(self):
        rows = records(range(1,36))
        a,plans,_ = build_examples(self.c,rows,self.g,{"model":"A1","fault_augmentation":True},1,3,"D",False)
        self.assertFalse(plans)
        for i,ex in enumerate(a): self.assertEqual(ex["history"],rows[:i])

    def test_features_positions_padding_and_current_unchanged(self):
        rows = records(range(12,26))
        c = copy.deepcopy(self.c)
        c["fault_augmentation"]["stage2_without_stage3"].update(E1_probability=1.,E3_probability=1.)
        ex,_,_ = build_examples(c,rows,self.g,{"model":"M2","fault_augmentation":True},1,1,"D",True)
        features = torch.arange(14*512,dtype=torch.float).reshape(14,512)
        ds = CachedDataset({"features":features,"records":rows},ex)
        batch = collate([ds[0],ds[-1]])
        self.assertTrue(torch.equal(ds[-1]["current_feature"],features[-1]))
        self.assertEqual(batch["history_position_ids"][1].tolist(),[5,4,3,2,1])
        self.assertTrue(batch["history_padding_mask"][0].all())

    def test_same_model_initialization_and_empty_history(self):
        a = initialize(42,self.c["model"]); b = initialize(42,self.c["model"])
        self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
        logits,_ = a(torch.zeros(2,512),torch.zeros(2,0,512),torch.zeros(2,0,dtype=torch.long),torch.zeros(2,0,dtype=torch.bool))
        self.assertEqual(tuple(logits.shape),(2,35)); self.assertTrue(torch.isfinite(logits).all())

    def test_scheduler_100_epochs(self):
        t = self.c["training"]
        rates = [t["learning_rate"]*lr_multiplier(i,t) for i in range(100)]
        self.assertAlmostEqual(rates[0],.0001)
        self.assertAlmostEqual(rates[4],.001)
        self.assertAlmostEqual(rates[-1],.00001)
        self.assertTrue(all(rates[i+1]<=rates[i] for i in range(4,99)))

    def test_scheduler_resume(self):
        t = self.c["training"]
        m = torch.nn.Linear(1,1)
        opt = torch.optim.AdamW(m.parameters(),lr=t["learning_rate"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt,lambda i:lr_multiplier(i,t))
        for _ in range(10): opt.step(); sched.step()
        a = opt.state_dict(); b = sched.state_dict()
        opt2 = torch.optim.AdamW(m.parameters(),lr=t["learning_rate"])
        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2,lambda i:lr_multiplier(i,t))
        opt2.load_state_dict(a); sched2.load_state_dict(b)
        self.assertEqual(opt.param_groups[0]["lr"],opt2.param_groups[0]["lr"])
        opt.step(); sched.step(); opt2.step(); sched2.step()
        self.assertEqual(opt.param_groups[0]["lr"],opt2.param_groups[0]["lr"])


if __name__=="__main__":
    unittest.main(verbosity=2)
