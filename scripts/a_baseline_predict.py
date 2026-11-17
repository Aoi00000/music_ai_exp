"""Run only in the isolated Basic Pitch environment; atomic/resumable predictions."""
import argparse,json,time,hashlib,importlib.metadata
from pathlib import Path
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model,predict
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
items=[it for it in json.loads(Path(a.manifest).read_text()) if it['split']==a.split]
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
config={'engine':'Basic Pitch','version':importlib.metadata.version('basic-pitch'),'onset_threshold':.5,'frame_threshold':.3,'minimum_note_length_ms':30,'pitch_range_midi':[23,88],'thresholds':'published defaults; duration/output range matched to CRNN'}
model=Model(ICASSP_2022_MODEL_PATH)
for i,it in enumerate(items):
    dest=out/(it['id']+'.json');audio_hash=hashlib.sha256(Path(it['audio']).read_bytes()).hexdigest()
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['audio_sha256']==audio_hash and old['config']==config:continue
    start=time.monotonic();_,midi,events=predict(it['audio'],model,minimum_note_length=30)
    notes=[{'pitch':int(n[2]),'onset':float(n[0]),'offset':float(n[1])} for n in events if 23<=n[2]<=88]
    midi.write(str(out/(it['id']+'.mid')))
    result={'id':it['id'],'audio_sha256':audio_hash,'config':config,'notes':notes,'seconds':time.monotonic()-start}
    temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(dest)
    print(f'{i+1}/{len(items)} {it["id"]} notes={len(notes)} seconds={result["seconds"]:.1f}',flush=True)
