"""Fixed, source-disjoint pilot; no IDMT references or test clips in fitting."""
from pathlib import Path
import json,hashlib
from collections import defaultdict,Counter
import soundfile as sf
from band2tab.data import slakh_items

root=Path('data');dest=root/'refiner_clips';dest.mkdir(exist_ok=True)
manifest=root/'manifest_refiner.json'
if manifest.exists():raise RuntimeError('Refiner protocol already frozen; retain this manifest')
rows=[];sources=defaultdict(list)
for it in slakh_items(root/'raw/baby','baby'):sources[it['group']].append(it)
def clip(ident,audio,group,split,kind,parts):
    info=sf.info(audio);start=max(0,(info.duration-12)*.35)
    with sf.SoundFile(audio) as f:
        f.seek(round(start*info.samplerate));y=f.read(round(12*info.samplerate),dtype='float32')
    path=dest/(ident+'.wav');sf.write(path,y,info.samplerate,subtype='PCM_16');end=start+len(y)/info.samplerate
    # Boundary-spanning notes lack a genuine onset inside the clip: do not turn
    # artificial starts into onset targets. Offset at clip end is censored.
    truth={part:[{**n,'onset':n['onset']-start,'offset':min(end,n['offset'])-start} for n in ns if start<=n['onset']<end and 23<=n['pitch']<=88] for part,ns in parts.items()}
    rows.append({'id':ident,'audio':str(path.resolve()),'group':group,'split':split,'kind':kind,
                 'parts':truth,'excerpt_start':start,'duration':len(y)/info.samplerate,
                 'boundary_policy':'onsets inside excerpt; truncated offsets; first 0.25s excluded from candidate/GT scoring',
                 'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
for group,items in sorted(sources.items()):
    split=items[0]['split']
    if split=='test':continue
    audio=Path(items[0]['audio']).parent.parent/'mix.wav'
    clip('ref_mix_'+audio.parent.name,audio,group,split,'mixture',
         {part:[n for it in items if it['part']==part for n in it['notes']] for part in ('guitar','bass')})
base=json.loads((root/'manifest_v2_techs_full.json').read_text())
for split,kind,count in [('train','guitarset',12),('train','guitar_techs',8),('val','guitarset',4)]:
    pool=[it for it in base if it['dataset']==kind and it['split']==split]
    pool.sort(key=lambda it:hashlib.sha256(it['id'].encode()).hexdigest())
    for it in pool[:count]:clip('ref_'+it['id'],it['audio'],it['group'],split,'solo',{'guitar':it['notes'],'bass':[]})
groups=defaultdict(set)
for it in rows:groups[it['group']].add(it['split'])
assert all(len(s)==1 for s in groups.values())
manifest.write_text(json.dumps(rows,indent=2))
Path('reports/refiner_protocol.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'seed':17,'epochs':[25,50,100,150],'thresholds':[.6,.75,.9],'margins':[.2,.4],
    'training_target':'3 classes: matching guitar, matching bass, no matching target; onset tolerance 50ms',
    'selection':'validation macro Note F1 across guitar+bass, ties prefer identity; all changes review-only pending independent acceptance',
    'test_references_used_for_fitting':False,'automatic_application_approved':False,
    'scope':'small pilot, existing synthetic band and solo sources; no new natural-band test','counts':{str(k):v for k,v in Counter((r['split'],r['kind']) for r in rows).items()}},indent=2))
print('REFINER_DATA',len(rows),Counter((it['split'],it['kind']) for it in rows),flush=True)
