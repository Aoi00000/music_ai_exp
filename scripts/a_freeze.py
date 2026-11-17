"""Azure: freeze validation-selected settings before reading held-out test scores."""
from pathlib import Path
import json,hashlib,datetime
out=Path('reports/test_protocol.json')
if out.exists():raise RuntimeError('Protocol already frozen; keep it immutable')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
runs={}
for name in ('a0','a1'):
    result=json.loads(Path(f'reports/{name}_val.json').read_text());model=f'runs/{name}/best.pt'
    if result['split']!='val' or result['macro_note_f1']<=.01:raise RuntimeError('Diagnose validation first')
    if result['checkpoint_sha256']!=sha(model):raise RuntimeError('Checkpoint changed after validation')
    history=json.loads(Path(f'runs/{name}/history.json').read_text())
    runs[name]={'checkpoint':model,'checkpoint_sha256':sha(model),'thresholds':{k:result[k] for k in ('threshold','frame_threshold','offset_threshold')},'completed_updates':history[-1]['epoch']*100,'validation_note_f1':result['macro_note_f1']}
if runs['a0']['completed_updates']!=runs['a1']['completed_updates']:raise RuntimeError('Unequal training budget')
protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':17,'manifest':'data/manifest_clean.json','manifest_sha256':sha('data/manifest_clean.json'),'runs':runs,'baseline':'Basic Pitch 0.4.0, published default onset/frame; minimum length 30ms; output B0-E6','selection':'BCE-best epoch per run; thresholds maximize validation macro Note F1; test is not used for selection'}
out.write_text(json.dumps(protocol,indent=2));print('Frozen',out)
