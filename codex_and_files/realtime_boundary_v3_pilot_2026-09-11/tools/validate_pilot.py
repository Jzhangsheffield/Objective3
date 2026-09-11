"""Preflight every pilot fold and v3 node correspondence without model training."""
import os, json, csv, hashlib
from pathlib import Path
from collections import defaultdict
import _bootstrap
from boundary_experiment.config import load_config, format_path
from boundary_experiment.annotations import load_run_index, load_frame_table
from evaluate_end_to_end import _node_rows_by_run, _ground_truth_segments

def main():
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs/base.json')
    lineage=json.loads(Path(cfg['paths']['node_lineage']).read_text())
    index=load_run_index(cfg['paths']['dataset_root'],cfg['paths']['annotation_root'],cfg['data']['camera_id'])
    report={'runs':len(index),'frames':0,'gt_actions':0,'folds':{},'problems':[]}
    for fold in cfg['data']['participants']:
        nodes=_node_rows_by_run(format_path(cfg['paths']['atomic_protocol_template'],heldout=fold,scope='all_runs'))
        count=0
        for info in index.values():
            if info.participant!=fold:continue
            try:
                table=load_frame_table(info)
                gt=_ground_truth_segments(table,nodes[(info.participant,info.source_run)],lineage[info.sample_name])
                assert len(gt)==len(lineage[info.sample_name]['segments'])
                for entry in lineage[info.sample_name]['segments'].values():
                    n=nodes[(info.participant,info.source_run)][entry['v2_action_ordinal']]
                    assert n['tier1']==entry['action'], (info.sample_name,n,entry)
                report['frames']+=len(table['rows']);report['gt_actions']+=len(gt);count+=len(gt)
                for side in ['left','right']:
                    assert (Path(cfg['paths']['dataset_root'])/'raw'/info.sample_name/f'mindrove_{side}.csv').is_file()
            except Exception as e:report['problems'].append(f'{info.sample_name}: {e}')
        report['folds'][fold]={'gt_actions':count}
        for seed in cfg['data']['seeds']:
            for key in ['backbone_checkpoint_template','m3_checkpoint_template']:
                p=format_path(cfg['paths'][key],heldout=fold,scope='all_runs',seed=seed)
                if not p.is_file():report['problems'].append(f'Missing: {p}')
    report['status']='failed' if report['problems'] else 'ok'
    dest=Path(cfg['paths']['validation_root']);dest.mkdir(parents=True,exist_ok=True)
    (dest/'pilot_preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
    if report['problems']:raise SystemExit(1)

if __name__=='__main__':main()
