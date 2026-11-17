"""Versioned sounding-pitch correction; original manifests/caches are retained."""
from pathlib import Path
import json,hashlib
import numpy as np
from .data import slakh_items
from .audio import note_targets

def run(root='data'):
 root=Path(root);sources={it['id']:it for it in slakh_items(root/'raw/baby','baby')};ledger=[];cache_dir=root/'cache_v2';cache_dir.mkdir(exist_ok=True)
 for name in ('clean','robust','techs','techs_full','valclips'):
  src=root/('manifest_'+name+'.json')
  if not src.exists():continue
  original=json.loads(src.read_text());rows=[]
  for it in original:
   if it['part']=='bass':
    rawid=it['id'].removeprefix('clip_').removesuffix('_remix');source=sources[rawid];shift=source['bass_octave_semitones']
    if it.get('label_semantics')!='sounding_pitch_v2':
     for n in it['notes']:n['written_midi_pitch']=n['pitch'];n['pitch']+=shift
    it.update(label_semantics='sounding_pitch_v2',bass_octave_semitones=shift,plugin_name=source['plugin_name'])
    if any(not 23<=n['pitch']<=88 for n in it['notes']):
     ledger.append({'id':it['id'],'manifest':name,'excluded':'corrected_pitch_outside_B0_E6'});continue
    cache=cache_dir/(it['id']+'.npz');sig=hashlib.sha256(json.dumps(it['notes'],sort_keys=True).encode()).hexdigest()
    with np.load(it['cache']) as z:x=z['x']
    y,outside=note_targets(it['notes'],len(x));assert not outside
    if not cache.exists():
     tmp=cache.with_suffix('.tmp.npz');np.savez_compressed(tmp,x=x,y=y,label_sha256=np.array(sig));tmp.replace(cache)
    else:
     with np.load(cache) as z:
      if str(z['label_sha256'])!=sig:raise ValueError('Corrected cache label mismatch')
    it['cache']=str(cache.resolve());ledger.append({'id':it['id'],'manifest':name,'shift':shift,'plugin':source['plugin_name']})
   rows.append(it)
  dest=root/('manifest_v2_'+name+'.json');dest.write_text(json.dumps(rows,indent=2));print('CORRECTED',name,len(rows),flush=True)
 (root/'v2_label_corrections.json').write_text(json.dumps({'source':'https://huggingface.co/spaces/mimbres/YourMT3/blob/main/amt/src/utils/preprocess/preprocess_slakh.py','changes':ledger,'original_data_retained':True},indent=2))
if __name__=='__main__':run()
