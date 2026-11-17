"""Fixed 30-second validation excerpts, chosen without model scores."""
from pathlib import Path
import json,hashlib
import soundfile as sf
import numpy as np
from band2tab.audio import features,note_targets
root=Path('data');items=[i for i in json.loads((root/'manifest_clean.json').read_text()) if i['split']=='val'];chosen=[]
for part in ('guitar','bass'):
 for mixed in (False,True):
  pool=[it for it in items if it['part']==part and (it['dataset']=='remix')==mixed]
  pool.sort(key=lambda i:hashlib.sha256(i['id'].encode()).hexdigest());chosen+=pool[:4]
folder=root/'valclips';folder.mkdir(exist_ok=True);result=[]
for it in chosen:
 ident='clip_'+it['id'];path=folder/(ident+'.wav');y,sr=sf.read(it['audio'],dtype='float32',frames=sf.info(it['audio']).samplerate*30);sf.write(path,y,sr,subtype='PCM_24');duration=len(y)/sr
 notes=[{**n,'offset':min(duration,n['offset'])} for n in it['notes'] if 0<=n['onset']<duration]
 cache=root/'cache'/(ident+'.npz');x=features(path);targets,outside=note_targets(notes,len(x));assert not outside;np.savez_compressed(cache,x=x,y=targets)
 result.append({**it,'id':ident,'audio':str(path.resolve()),'cache':str(cache.resolve()),'notes':notes,'excerpt_start':0,'excerpt_seconds':duration})
(root/'manifest_valclips.json').write_text(json.dumps(result,indent=2));print('VALCLIPS_READY',len(result),flush=True)
