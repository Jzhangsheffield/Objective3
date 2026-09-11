"""Load all eight bundled weights; no feature extraction or training."""
import _bootstrap
from pathlib import Path
import torch
from boundary_experiment.config import load_config, format_path
from boundary_experiment.features import build_frozen_backbone
from boundary_experiment.m3_adapter import M3AtomicTailOnlineAdapter
from boundary_experiment.utils import write_json

def main():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/base.json')
    device=torch.device('cpu')
    reports={}
    for fold in cfg['data']['participants']:
        backbone, report=build_frozen_backbone(cfg['paths']['atomic_project_root'],
            format_path(cfg['paths']['backbone_checkpoint_template'],heldout=fold,seed=1,scope='all_runs'),device)
        del backbone
        adapter=M3AtomicTailOnlineAdapter(cfg['paths']['atomic_project_root'],
            format_path(cfg['paths']['m3_checkpoint_template'],heldout=fold,seed=1,scope='all_runs'),device,
            task_graph_path=cfg['paths']['task_graph'])
        first=adapter.predict(torch.zeros(512));second=adapter.predict(torch.ones(512))
        assert first['history_length']==0 and second['history_length']==1
        reports[fold]={'backbone':report,'m3':adapter.load_report,'synthetic_history_test':'passed'}
        del adapter
        print(f'{fold}: backbone + M3 exact load and history test passed')
    write_json(Path(cfg['paths']['validation_root'])/'weight_checks.json',reports)

if __name__=='__main__':main()
