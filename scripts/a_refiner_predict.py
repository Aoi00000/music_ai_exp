"""Cache all pitched MuScriptor events, including non-guitar/bass assignments."""
from pathlib import Path
import json,hashlib,time
from muscriptor import TranscriptionModel

rows=json.loads(Path('data/manifest_refiner.json').read_text());out=Path('reports/refiner_base');out.mkdir(exist_ok=True)
weights=Path('models/muscriptor-medium/model.safetensors');sha=hashlib.sha256(weights.read_bytes()).hexdigest()
model=TranscriptionModel.load_model(weights,device='cuda',dtype='float16')
for i,it in enumerate(rows):
    path=out/(it['id']+'.mid');meta=path.with_suffix('.json')
    if meta.exists() and path.exists():
        old=json.loads(meta.read_text())
        if old['audio_sha256']==it['audio_sha256'] and old['model_sha256']==sha:continue
    start=time.monotonic();midi,_=model.transcribe_and_postprocess(it['audio'],detect_tempo=False)
    tmp=path.with_suffix('.tmp');tmp.write_bytes(midi);tmp.replace(path)
    meta.write_text(json.dumps({'audio_sha256':it['audio_sha256'],'model_sha256':sha,'cfg_coef':1.,'beam_size':1,'seconds':time.monotonic()-start}))
    print(i+1,'/',len(rows),it['id'],time.monotonic()-start,flush=True)
