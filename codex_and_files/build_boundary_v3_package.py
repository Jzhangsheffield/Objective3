"""Create an isolated, portable boundary experiment from audited local sources."""
from pathlib import Path
import csv, json, shutil, hashlib
from collections import defaultdict

BASE = Path(__file__).resolve().parent
OUT = BASE / 'realtime_boundary_v3_pilot_2026-09-11'
RGB = BASE / 'realtime_action_boundary_experiment_2026-08-07'
SENSOR = BASE / 'realtime_imu_emg_boundary_experiment_2026-09-06'
DATA = BASE / 'multimodal_training_repaired_2026-09-10'
ANN = Path('D:/Junxi_data/MULTISENSOR_DATA_COLLECTION_Stage2_structured_data/Action_Segmentation_Dataset/annotations')
ATOMIC = BASE / 'graph_history_rgb_cross_person_ADM_2026-07-22'

def read(p):
    return p.read_text(encoding='utf-8-sig')

def write(p,s):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(s,encoding='utf-8')

def js(p,obj):
    write(p,json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def sha(p):
    return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    for source,target,folders in [(RGB,OUT,['boundary_experiment','tools','tests','configs','scripts']),
                                 (SENSOR,OUT/'sensor_baselines',['sensor_boundary','tools','tests','configs','scripts','protocols'])]:
        for folder in folders:
            shutil.copytree(source/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ['config_windows.bat','requirements.txt']:
            shutil.copy2(source/name,target/name)
    shutil.copy2(RGB/'README.md',OUT/'README_previous_rgb.md')
    shutil.copy2(SENSOR/'README.md',OUT/'sensor_baselines/README_previous.md')
    # Self-contained sensor data; images remain a separately relocatable dependency.
    shutil.copytree(DATA/'raw',OUT/'data/raw')
    shutil.copytree(ANN/'action_recognition_timestamps_canonical_v3',OUT/'data/annotations')
    for name in ['sensor_time_quality.csv','fully_repaired_runs.txt','runs_with_unrepaired_rows.txt','source_integrity.json']:
        shutil.copy2(DATA/name,OUT/'data'/name)
    manifest=[json.loads(l) for l in read(DATA/'manifest.jsonl').splitlines() if l.strip()]
    for m in manifest:
        m['camera_dirs']={c:f"raw/{m['sample_name']}/{c}" for c in m['camera_ids']}
        m['annotation_version']='action_recognition_timestamps_canonical_v3'
    write(OUT/'data/manifest.jsonl',''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in manifest))
    # Stable node lineage: v3 -> v2 action ordinal -> original recognition protocol row.
    lineage={}
    for m in manifest:
        name=m['sample_name']
        old=[r for r in rows(ANN/'action_recognition_timestamps_canonical_v2'/f'{name}_segmentation_annotation.csv') if r['action']!='background']
        new=[r for r in rows(OUT/'data/annotations'/f'{name}_segmentation_annotation.csv') if r['action']!='background']
        used=set(); entries={}
        for r in new:
            matches=[(i,x) for i,x in enumerate(old) if (x['action'],x['object'],x['mark'])==(r['action'],r['object'],r['mark']) and max(int(x['start_idx']),int(r['start_idx']))<=min(int(x['end_idx']),int(r['end_idx']))]
            if len(matches)!=1 or matches[0][0] in used:
                raise ValueError(f'Ambiguous lineage: {name} {r}')
            i,x=matches[0]; used.add(i)
            entries[r['No']]={'v2_action_ordinal':i,'action':r['action'],'object':r['object'],'mark':r['mark']}
        if sorted(used)!=list(range(len(old))):
            assert name=='run_sample_000083' and len(old)-len(new)==1
        lineage[name]={'v2_action_count':len(old),'segments':entries}
    js(OUT/'assets/v3_node_lineage.json',lineage)
    # Include only inference source/assets and pilot checkpoints/protocols.
    for folder in ['graph_history','assets']:
        shutil.copytree(ATOMIC/folder,OUT/'atomic_dependency'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    missing=[]
    for fold in 'ADJM':
        protocol=Path('outputs')/f'{fold}_as_test'/'cam_001484412812/protocols'
        shutil.copytree(ATOMIC/protocol,OUT/'atomic_dependency'/protocol)
        for relative in [Path('outputs')/f'{fold}_as_test'/'cam_001484412812/seed_1/backbone/all_runs/last.pth',
                         Path('outputs/at_ad')/f'{fold}_s1'/'all_runs/refresh_once/m3_atomic_tail_direct_fusion/last.pth']:
            if not (ATOMIC/relative).is_file(): missing.append(str(relative)); continue
            (OUT/'atomic_dependency'/relative).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ATOMIC/relative,OUT/'atomic_dependency'/relative)
    # Portable defaults; separate dataset root and RGB image root.
    p=OUT/'config_windows.bat'; s=read(p)
    s=s.replace('D:\\Junxi_data\\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\\Action_Segmentation_Dataset','%PACKAGE_ROOT%\\data')
    s=s.replace('D:\\junxi_data\\Objective3\\codex_and_files\\graph_history_rgb_cross_person_ADM_2026-07-22','%PACKAGE_ROOT%\\atomic_dependency')
    s=s.replace('%DATASET_ROOT%\\annotations\\action_recognition_timestamps_canonical_v2','%DATASET_ROOT%\\annotations')
    s=s.replace('canonical_v2','canonical_v3').replace('RECOMMENDED_SEEDS=1 2 42','RECOMMENDED_SEEDS=1').replace('RECOMMENDED_SCOPES=normal_only all_runs','RECOMMENDED_SCOPES=all_runs')
    s=s.replace('NUM_WORKERS=8','NUM_WORKERS=0').replace('EXTRACT_NUM_WORKERS=14','EXTRACT_NUM_WORKERS=4')
    s+='\n' if not s.endswith('\n') else ''
    s=s.replace('REM Environment aliases', 'if not defined RGB_DATASET_ROOT set "RGB_DATASET_ROOT=D:\\Junxi_data\\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\\Action_Segmentation_Dataset"\nset "RAB_RGB_DATASET_ROOT=%RGB_DATASET_ROOT%"\n\nREM Environment aliases')
    write(p,s)
    p=OUT/'configs/base.json'; c=json.loads(read(p)); c['experiment_name']='boundary_v3_pilot'; c['data'].update(seeds=[1],train_scopes=['all_runs'],annotation_policy='action_recognition_timestamps_canonical_v3')
    c['paths']['output_template']=c['paths']['output_template'].replace('canonical_v2','canonical_v3')
    c['paths']['node_lineage']='%RAB_EXPERIMENT_ROOT%/assets/v3_node_lineage.json'
    js(p,c)
    p=OUT/'configs/smoke_stride4.json'; c=json.loads(read(p)); c['training']['num_workers']=0; js(p,c)
    p=OUT/'boundary_experiment/annotations.py'; s=read(p).replace('import numpy as np','import numpy as np\nimport os')
    s=s.replace('camera_dir=dataset_root / str(camera_rel),','camera_dir=Path(os.environ.get("RAB_RGB_DATASET_ROOT", str(dataset_root))) / str(camera_rel),');write(p,s)
    p=OUT/'tools/evaluate_end_to_end.py';s=read(p)
    s=s.replace('def _ground_truth_segments(table: dict, node_rows: list[dict])','def _ground_truth_segments(table: dict, node_rows: list[dict], lineage: dict)')
    s=s.replace('if len(action_segment_ids) != len(node_rows):','if lineage["v2_action_count"] != len(node_rows):')
    s=s.replace('for segment_id, node in zip(action_segment_ids, node_rows):','for segment_id in action_segment_ids:\n        entry = lineage["segments"][str(segment_id)]\n        node = node_rows[entry["v2_action_ordinal"]]')
    s=s.replace('node_rows = _node_rows_by_run(atomic_protocol)','node_rows = _node_rows_by_run(atomic_protocol)\n    lineage = json.loads(Path(cfg["paths"]["node_lineage"]).read_text(encoding="utf-8"))')
    s=s.replace('_ground_truth_segments(table, node_rows[(info.participant, info.source_run)])','_ground_truth_segments(table, node_rows[(info.participant, info.source_run)], lineage[run])');write(p,s)
    # Sensor context loss fix, matching RGB semantics.
    p=OUT/'sensor_baselines/sensor_boundary/data.py';s=read(p)
    s=s.replace('self.chunk_length = int(chunk_length)','self.chunk_length = int(chunk_length)\n        self.chunk_overlap = int(chunk_overlap)\n        if self.chunk_overlap < 0: raise ValueError("negative overlap")')
    s=s.replace('"start_offset": start,','"start_offset": start,\n            "context_steps": 0 if start == 0 else min(self.chunk_overlap, end-start),')
    s=s.replace('mask[i, :length] = True','mask[i, int(row.get("context_steps", 0)):length] = True');write(p,s)
    # Use the verified RGB pending decoder, keeping sensor settings names.
    s=read(OUT/'boundary_experiment/online.py').replace('start_debounce: int = 2','start_debounce_steps: int = 2').replace('end_debounce: int = 2','end_debounce_steps: int = 2').replace('self.start_debounce = start_debounce','self.start_debounce = start_debounce_steps').replace('self.end_debounce = end_debounce','self.end_debounce = end_debounce_steps')
    write(OUT/'sensor_baselines/sensor_boundary/online.py',s)
    p=OUT/'sensor_baselines/config_windows.bat';s=read(p)
    s=s.replace('D:\\Junxi_data\\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\\Action_Segmentation_Dataset','%PACKAGE_ROOT%\\..\\data').replace('%DATASET_ROOT%\\annotations\\action_recognition_timestamps_canonical_v2','%DATASET_ROOT%\\annotations').replace('canonical_v2','canonical_v3').replace('RECOMMENDED_SEEDS=1 2 42','RECOMMENDED_SEEDS=1').replace('RECOMMENDED_SCOPES=normal_only all_runs','RECOMMENDED_SCOPES=all_runs');write(p,s)
    for p in (OUT/'sensor_baselines/configs').rglob('*.json'):
        s=read(p).replace('canonical_v2','canonical_v3');write(p,s)
    p=OUT/'sensor_baselines/configs/common.json';c=json.loads(read(p));c['data']['seeds']=[1];c['data']['train_scopes']=['all_runs'];c['online']['merge_gap_steps']=0;js(p,c)
    source_hashes={str(p.relative_to(RGB)):sha(p) for folder in ['boundary_experiment','tools','tests'] for p in (RGB/folder).glob('*.py')}
    js(OUT/'PACKAGE_PROVENANCE.json',{'rgb_source':str(RGB),'rgb_source_sha256':source_hashes,'sensor_source':str(SENSOR),'sensor_data_source':str(DATA),'annotation_source':str(ANN/'action_recognition_timestamps_canonical_v3'),'annotation_summary':json.loads(read(OUT/'data/annotations/generation_summary.json')),'missing_pilot_checkpoints':missing,'pilot':{'folds':list('ADJM'),'seeds':[1],'scope':'all_runs','epochs':40,'stride':1},'training_executed':False})
    print(json.dumps({'output':str(OUT),'missing_checkpoints':missing,'runs':len(manifest)},indent=2))

if __name__=='__main__': main()
