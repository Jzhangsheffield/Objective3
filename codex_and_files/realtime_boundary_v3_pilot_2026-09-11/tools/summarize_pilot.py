"""Collect RGB split metrics; missing conditions remain explicitly missing."""
import argparse, json, csv
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outputs',type=Path,required=True);args=ap.parse_args()
    result=[]
    for fold in 'ADJM':
        base=args.outputs/f'{fold}_as_test/all_runs/seed_1/causal_boundary_tcn_canonical_v3'
        for split in ['test_normal','test_fault','test_all']:
            r={'heldout':fold,'seed':1,'scope':'all_runs','split':split}
            bp=base/'evaluation'/split/'metrics.json';np=base/'end_to_end'/split/'metrics.json'
            r['boundary_status']='complete' if bp.exists() else 'missing'
            r['node_status']='complete' if np.exists() else 'missing'
            if bp.exists():
                m=json.loads(bp.read_text())['macro']
                r.update(frame_f1=m['frame_state']['f1'],frame_accuracy=m['frame_state']['accuracy'],binary_edit=m['edit_score'],segment_f1_50=m['segmental_f1']['50']['f1'])
                for tol,val in m['boundary'].items():
                    for edge in ['start','end']:r[f'{edge}_f1_tol{tol}']=val[edge]['f1']
            if np.exists():r.update(json.loads(np.read_text()))
            result.append(r)
    args.outputs.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in result for k in r))
    with (args.outputs/'pilot_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(result)
    print(f'Saved {len(result)} condition rows to {args.outputs / "pilot_summary.csv"}')

if __name__=='__main__':main()
