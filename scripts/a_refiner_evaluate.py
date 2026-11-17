"""One post-selection check on existing comparisons; no subsequent tuning."""
from pathlib import Path
import json,hashlib
from collections import defaultdict
import numpy as np
from band2tab.refine import refine
from band2tab.learn import note_metrics,pitch_set_metrics

checkpoint=Path('models/muscriptor_refiner.pt');frozen_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
dest=Path('reports/refiner_existing_comparisons.json')
if dest.exists():raise RuntimeError('Comparison already recorded; do not overwrite/reselect')
results={'model_sha256':frozen_sha,'scope':'previously inspected IDMT and synthetic mix comparisons, not a new untouched external test; no refitting/reselection',
         'automatic_application_approved':False,'notes':'Proposals only; original MIDI is retained.'}
def metrics(truth,parts):
    return {'note_f1':note_metrics(truth,parts)['f1'],'offset_f1':note_metrics(truth,parts,True)['f1'],**pitch_set_metrics(truth,parts)}
rows=[]
for it in json.loads(Path('data/manifest_external.json').read_text()):
    midi=Path('reports/muscriptor_external_predictions')/(it['id']+'.mid')
    obj=refine(it['audio'],midi,checkpoint)
    eligible=lambda ns:[n for n in ns if 23<=n['pitch']<=88 and n['offset']-n['onset']>=.03]
    before=eligible(obj['parts'].get(it['part'],[]));after=eligible(obj['suggested_parts'].get(it['part'],[]))
    rows.append({'id':it['id'],'part':it['part'],'baseline':metrics(it['notes'],before),'proposed':metrics(it['notes'],after),'changes':obj['refinement_changes']})
    print('EXISTING_SOLO',it['id'],rows[-1]['baseline']['note_f1'],rows[-1]['proposed']['note_f1'],flush=True)
results['solo_tracks']=rows
results['solo_macro_note_f1']={key:float(np.mean([r[key]['note_f1'] for r in rows])) for key in ('baseline','proposed')}
groups=defaultdict(list)
for it in json.loads(Path('data/manifest_mix_diagnostic.json').read_text()):groups[it['group']].append(it)
mixed=[]
for group,items in groups.items():
    it=items[0];midi=Path('reports/muscriptor_mix30_predictions')/(it['id']+'.mid');obj=refine(it['audio'],midi,checkpoint)
    truth=[n for it in items for n in it['notes']]
    pair={key:pitch_set_metrics(truth,[n for ns in obj[field].values() for n in ns if 23<=n['pitch']<=88 and n['offset']-n['onset']>=.03]) for key,field in [('baseline','parts'),('proposed','suggested_parts')]}
    mixed.append({'group':group,**pair,'changes':obj['refinement_changes']})
    print('EXISTING_MIX',group,pair['baseline']['frame_f1'],pair['proposed']['frame_f1'],flush=True)
results['mixture_tracks']=mixed
results['mixture_macro_union_frame_f1']={key:float(np.mean([r[key]['frame_f1'] for r in mixed])) for key in ('baseline','proposed')}
if frozen_sha!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():raise ValueError('Refiner changed during evaluation')
dest.write_text(json.dumps(results,indent=2));print('REFINER_COMPARISONS_COMPLETE',results['solo_macro_note_f1'],results['mixture_macro_union_frame_f1'],flush=True)
