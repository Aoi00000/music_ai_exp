"""Frozen MuScriptor inference, isolated environment, no beat quantization."""
import argparse,json,time,hashlib,importlib.metadata
from pathlib import Path
from muscriptor import TranscriptionModel
from band2tab.score import read_midi
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);a=p.parse_args()
weights=Path('models/muscriptor-medium/model.safetensors');out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
config={'engine':'MuScriptor','version':importlib.metadata.version('muscriptor'),'model_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'instruments':None,'beat_quantization':False,'pitch_range_midi':[23,88],'minimum_note_seconds':.03,'cfg_coef':1.0,'beam_size':1}
m=TranscriptionModel.load_model(weights,device='cuda',dtype='float16')
items=[it for it in json.loads(Path(a.manifest).read_text()) if it['split']==a.split]
audio_cache={}
for i,it in enumerate(items):
    dest=out/(it['id']+'.json');audio_hash=hashlib.sha256(Path(it['audio']).read_bytes()).hexdigest()
    if dest.exists():
        obj=json.loads(dest.read_text())
        if obj['config']==config and obj['audio_sha256']==audio_hash:continue
    start=time.monotonic()
    if audio_hash in audio_cache:mid=audio_cache[audio_hash]
    else:
        mid,_=m.transcribe_and_postprocess(it['audio'],detect_tempo=False);audio_cache[audio_hash]=mid
    midi_path=out/(it['id']+'.mid');midi_path.write_bytes(mid)
    parts=read_midi(midi_path);notes=[n for n in parts.get(it['part'],[]) if 23<=n['pitch']<=88 and n['offset']-n['onset']>=.03]
    obj={'id':it['id'],'audio_sha256':audio_hash,'config':config,'notes':notes,'seconds':time.monotonic()-start}
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(dest)
    print(i+1,'/',len(items),it['id'],len(notes),obj['seconds'],flush=True)
