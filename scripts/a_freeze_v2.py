"""Freeze all candidates before scoring the new IDMT external recordings."""
from pathlib import Path
import json,hashlib,shutil,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=Path('reports/v2_external_protocol.json')
if out.exists():raise RuntimeError('External protocol already frozen; do not overwrite it')
runs={};folder=Path('runs/frozen_v2');folder.mkdir(exist_ok=True)
for name in ('a6_sounding','a7_sounding_techs','a8_sounding_full'):
 r=json.loads(Path(f'reports/{name}_val.json').read_text());model=Path(r['checkpoint'])
 if r['split']!='val' or sha(model)!=r['checkpoint_sha256']:raise ValueError('Validation checkpoint mismatch')
 frozen=folder/(name+'.pt');shutil.copy2(model,frozen)
 runs[name]={'checkpoint':str(frozen),'checkpoint_sha256':sha(frozen),'validation_note_f1':r['macro_note_f1'],'thresholds':{k:r[k] for k in ('threshold','frame_threshold','offset_threshold')}}
best=max(runs,key=lambda k:runs[k]['validation_note_f1'])
protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label_semantics':'sounding_pitch_v2','external_manifest':'data/manifest_external.json','external_manifest_sha256':sha('data/manifest_external.json'),'external_tracks':22,'runs':runs,'selected_by_validation':best,'selection_uses_external_scores':False,'known_limits':'17 bass recordings, one bass/player; 5 guitar recordings, 4 composition groups, one guitar/player. Public model pretraining overlap cannot be ruled out. Solo recordings, not full bands.'}
out.write_text(json.dumps(protocol,indent=2));print('FROZEN_EXTERNAL_PROTOCOL',best,flush=True)
