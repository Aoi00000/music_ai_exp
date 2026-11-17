"""Synthetic full-mixture diagnostic, not a fresh real-band test."""
from pathlib import Path
import json,argparse
import soundfile as sf
from collections import defaultdict
from band2tab.data import slakh_items
from band2tab.learn import note_metrics,pitch_set_metrics
p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','score']);a=p.parse_args()
root=Path('data');path=root/'manifest_mix_diagnostic.json'
if a.command=='prepare':
 groups=defaultdict(list)
 for it in slakh_items(root/'raw/baby','baby'):
  if it['split']=='test':groups[it['group']].append(it)
 rows=[]
 for group,items in sorted(groups.items()):
  for part in ('guitar','bass'):
   selected=[i for i in items if i['part']==part];audio=Path(selected[0]['audio']).parent.parent/'mix.wav';assert audio.exists()
   y,sr=sf.read(audio,dtype='float32',frames=sf.info(audio).samplerate*30)
   clips=root/'mix_diagnostic_clips';clips.mkdir(exist_ok=True);clip=clips/(audio.parent.name+'.wav');sf.write(clip,y,sr,subtype='PCM_16');duration=len(y)/sr
   notes=[{**n,'offset':min(duration,n['offset'])} for it in selected for n in it['notes'] if 23<=n['pitch']<=88 and n['onset']<duration]
   audio=clip
   rows.append({'id':'mix30_'+audio.stem+'_'+part,'dataset':'synthetic_full_mix','source_dataset':'baby','label_semantics':'sounding_pitch_v2','part':part,'group':group,'split':'test','audio':str(audio.resolve()),'notes':notes})
 if len(groups)!=3:raise ValueError('Expected the original three held-out BabySlakh source songs')
 path.write_text(json.dumps(rows,indent=2));print('MIX_DIAGNOSTIC_READY',len(rows),flush=True)
else:
 rows=json.loads(path.read_text());groups=defaultdict(list)
 for it in rows:
  pred=json.loads((Path('reports/muscriptor_mix30_predictions')/(it['id']+'.json')).read_text())['notes'];groups[it['group']].append((it,pred))
 results=[]
 for g,pairs in groups.items():
  entry={'source_group':g,'parts':{it['part']:note_metrics(it['notes'],pred) for it,pred in pairs}}
  entry['guitar_bass_union']=pitch_set_metrics([n for it,_ in pairs for n in it['notes']],[n for _,pred in pairs for n in pred]);results.append(entry)
 out={'scope':'3 synthetic full-mixture 30-second excerpts, historical test fold; not new real-band generalization; public model pretraining overlap possible','tracks':results,'mean_union_frame_f1':sum(r['guitar_bass_union']['frame_f1'] for r in results)/len(results),'mean_union_active_exact_match':sum(r['guitar_bass_union']['active_exact_match'] for r in results)/len(results)}
 Path('reports/mix_diagnostic.json').write_text(json.dumps(out,indent=2));print(out['scope'],out['mean_union_frame_f1'],out['mean_union_active_exact_match'],flush=True)
