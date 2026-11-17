"""Frozen MuScriptor cache; test inference is separate from train/val inference."""
from pathlib import Path
import argparse,hashlib,json,shutil,time
from muscriptor import TranscriptionModel

p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--manifest',default='data/manifest_duration.json');a=p.parse_args()
if a.test and not Path('reports/duration_selection.json').exists():raise RuntimeError('Freeze the validation-selected model before test inference')
rows=json.loads(Path(a.manifest).read_text());out=Path('reports/duration_base');out.mkdir(exist_ok=True)
weights=Path('models/muscriptor-medium/model.safetensors');sha=hashlib.sha256(weights.read_bytes()).hexdigest()
rows=[r for r in rows if (r['split']=='test')==a.test]
model=None
for i,it in enumerate(rows):
    path=out/(it['id']+'.mid');meta=path.with_suffix('.json')
    if meta.exists() and path.exists():
        old=json.loads(meta.read_text())
        if old['audio_sha256']==it['audio_sha256'] and old['model_sha256']==sha:continue
        raise ValueError('Stale prediction: '+it['id'])
    if hashlib.sha256(Path(it['audio']).read_bytes()).hexdigest()!=it['audio_sha256']:raise ValueError('Audio changed')
    reuse=Path(it.get('reuse_prediction','missing'))
    if reuse.is_file():
        old=json.loads(reuse.with_suffix('.json').read_text())
        if old['audio_sha256']!=it['audio_sha256'] or old['model_sha256']!=sha:raise ValueError('Stale reusable prediction')
        shutil.copy2(reuse,path);meta.write_text(json.dumps(old));print('REUSED',it['id'],flush=True);continue
    if model is None:model=TranscriptionModel.load_model(weights,device='cuda',dtype='float16')
    start=time.monotonic();midi,_=model.transcribe_and_postprocess(it['audio'],detect_tempo=False)
    tmp=path.with_suffix('.tmp');tmp.write_bytes(midi);tmp.replace(path)
    meta.write_text(json.dumps({'audio_sha256':it['audio_sha256'],'model_sha256':sha,'cfg_coef':1.,'beam_size':1,'seconds':time.monotonic()-start}))
    print('PREDICTED',i+1,'/',len(rows),it['id'],time.monotonic()-start,flush=True)
