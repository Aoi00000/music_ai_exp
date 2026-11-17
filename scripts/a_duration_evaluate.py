"""Frozen new-player holdout and explicitly reused regression comparisons."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json
import numpy as np
import torch
import soundfile as sf
from band2tab.audio import features
from band2tab.refine import candidates_from_midi,event_features,EventRefiner,proposed_parts
from band2tab.refine_duration import instrument_metrics

p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');a=p.parse_args()
torch.set_num_threads(2);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
selection=json.loads(Path('reports/duration_selection.json').read_text());protocol=json.loads(Path('reports/duration_protocol.json').read_text())
if selection['protocol_sha256']!=sha('reports/duration_protocol.json') or protocol['manifest_sha256']!=sha('data/manifest_duration.json'):raise ValueError('Frozen evaluation protocol changed')
out=Path('reports/duration_test_comparison_full_output.json')
if out.exists():raise RuntimeError('Already scored; do not reselect on test')
paths={'v1_onset':'models/muscriptor_refiner.pt',**{k:v['checkpoint'] for k,v in selection['results'].items()}}
models={}
for name,path in paths.items():
    if name in selection['results'] and sha(path)!=selection['results'][name]['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    state=torch.load(path,map_location='cpu',weights_only=True);model=EventRefiner();model.load_state_dict(state['model']);model.eval()
    models[name]=(model,state)
records=[]
for r in json.loads(Path('data/manifest_duration.json').read_text()):
    if r['split']=='test':records.append({**r,'evaluation':r.get('evaluation','new_p3_player'),'midi':str(Path('reports/duration_base')/(r['id']+'.mid'))})
for it in json.loads(Path('data/manifest_external.json').read_text()):
    records.append({**it,'parts':{it['part']:it['notes']},'duration':sf.info(it['audio']).duration,
                    'evaluation':'previously_inspected_idmt22','midi':str(Path('reports/muscriptor_external_predictions')/(it['id']+'.mid'))})
groups=defaultdict(list)
for it in json.loads(Path('data/manifest_mix_diagnostic.json').read_text()):groups[it['group']].append(it)
for group,items in groups.items():
    it=items[0];records.append({**it,'parts':{p:[n for row in items if row['part']==p for n in row['notes']] for p in ('guitar','bass')},
        'duration':sf.info(it['audio']).duration,'evaluation':'previously_inspected_mix3','midi':str(Path('reports/muscriptor_mix30_predictions')/(it['id']+'.mid'))})
results=[];predictions=Path('reports/duration_test_predictions');predictions.mkdir(exist_ok=True)
for r in records:
    notes=candidates_from_midi(r['midi']);x=event_features(features(r['audio']),notes) if not a.replay else None
    original=proposed_parts(notes,np.eye(3)[[n['baseline_class'] for n in notes]],1.,1.)[0]
    start=.25 if r.get('source_dataset','').startswith('filobass') else 0.
    # Excerpt guard windows are intentional. For whole recordings, include
    # hallucinated output AFTER the file ends rather than silently dropping it.
    end=r['duration']-.25 if start else None
    scored={'muscriptor':instrument_metrics(r['parts'],original,start=start,end=end)};changes={}
    raw={'muscriptor':instrument_metrics(r['unshifted_parts'],original)} if 'unshifted_parts' in r else {}
    for name,(model,state) in models.items():
        path=predictions/(r['id']+'_'+name+'.json')
        if a.replay:
            saved=json.loads(path.read_text());proposed=saved['parts'];change=saved['changes']
        else:
            z=torch.tensor(np.clip((x-np.array(state['mean']))/np.array(state['std']),-8,8),dtype=torch.float32)
            with torch.no_grad():probability=model(z,torch.tensor([n['baseline_class'] for n in notes],dtype=torch.long)).softmax(-1).numpy()
            proposed,change=proposed_parts(notes,probability,**state['policy'])
        changes[name]=change
        scored[name]=instrument_metrics(r['parts'],proposed,start=start,end=end)
        if raw:raw[name]=instrument_metrics(r['unshifted_parts'],proposed)
        if not a.replay:path.write_text(json.dumps({'parts':proposed,'changes':change},indent=2))
    results.append({'id':r['id'],'group':r['group'],'evaluation':r['evaluation'],'metrics':scored,'raw_annotation_metrics':raw,'changes':changes})
    print('SCORED',r['id'],{k:round(v['frame_f1'],4) for k,v in scored.items()},flush=True)
summary={};raw_summary={}
for evaluation in sorted({r['evaluation'] for r in results}):
    use=[r for r in results if r['evaluation']==evaluation]
    summary[evaluation]={name:{k:float(np.mean([r['metrics'][name][k] for r in use])) for k in ('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')} for name in ('muscriptor',*models)}
    if evaluation=='new_p3_player':raw_summary={name:{k:float(np.mean([r['raw_annotation_metrics'][name][k] for r in use])) for k in ('note_f1','offset_f1','frame_f1')} for name in ('muscriptor',*models)}
report={'selected_by_validation':selection['selected_by_validation'],'test_used_for_selection':False,'automatic_application_approved':False,
        'whole_recording_metrics':'include all output, including predictions after the audio ends; FiloBass excerpts alone use fixed 0.25s edge guards',
        'replayed_saved_predictions':a.replay,
        'scope':'P3 and held-out FiloBass artists unused in our training/selection; public-model pretraining overlap unknown; constructed mixes are not natural bands',
        'models':{name:{'path':p,'sha256':sha(p)} for name,p in paths.items()},'summary':summary,'unshifted_p3_annotations':raw_summary,'tracks':results}
out.write_text(json.dumps(report,indent=2));print('DURATION_TEST_COMPLETE',summary,flush=True)
