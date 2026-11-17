"""Duration loss ablation on small vs expanded sources, strict validation gates."""
from pathlib import Path
from collections import Counter
import copy,hashlib,json
import numpy as np
import torch
from band2tab.audio import features
from band2tab.refine import candidates_from_midi,event_features,EventRefiner,proposed_parts
from band2tab.refine_duration import duration_targets,instrument_metrics,summarize_records,SCHEMA

torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol_path=Path('reports/duration_protocol.json');protocol=json.loads(protocol_path.read_text())
manifest=Path('data/manifest_duration.json')
if sha(manifest)!=protocol['manifest_sha256']:raise ValueError('Frozen manifest changed')
if Path('reports/duration_selection.json').exists():raise RuntimeError('Already selected; do not overwrite')
rows=[r for r in json.loads(manifest.read_text()) if r['split']!='test'];records=[]
cache=Path('data/duration_features');cache.mkdir(exist_ok=True)
baseline_sha=sha('models/muscriptor-medium/model.safetensors')
for it in rows:
    midi=Path('reports/duration_base')/(it['id']+'.mid');meta=json.loads(midi.with_suffix('.json').read_text())
    if meta['audio_sha256']!=it['audio_sha256'] or meta['model_sha256']!=baseline_sha:raise ValueError('Stale baseline')
    notes=candidates_from_midi(midi);path=cache/(it['id']+'.npz')
    if path.exists():
        with np.load(path) as z:
            if str(z['midi_sha256'])!=sha(midi) or str(z['audio_sha256'])!=it['audio_sha256']:raise ValueError('Stale feature cache')
            x=z['x'];targets=z['targets']
    else:
        x=event_features(features(it['audio']),notes);targets=duration_targets(notes,it['parts'],protocol['occupancy_weight'])
        np.savez_compressed(path,x=x,targets=targets,midi_sha256=np.array(sha(midi)),audio_sha256=np.array(it['audio_sha256']))
    records.append({**it,'notes':notes,'x':x,'targets':targets})
    print('FEATURES',it['id'],len(notes),flush=True)
validation=[r for r in records if r['split']=='val']
def score(predictions):
    return summarize_records([{'id':r['id'],'group':r['group'],'kind':r['kind'],**instrument_metrics(r['parts'],p,start=.25,end=r['duration']-.25)} for r,p in zip(validation,predictions)])
identity=[proposed_parts(r['notes'],np.eye(3)[[n['baseline_class'] for n in r['notes']]],1.,1.)[0] for r in validation]
baseline=score(identity)
def feasible(candidate):
    for key in ('by_domain','by_instrument'):
        for group,metrics in baseline[key].items():
            for metric in ('note_f1','frame_f1'):
                if candidate[key][group][metric]<metrics[metric]-1e-6:return False
    return True
summaries={}
for condition in protocol['training_conditions']:
    torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed'])
    train=[r for r in records if r['split']=='train' and (condition=='duration_expanded' or r['small_training']) and len(r['notes'])]
    trainx=np.concatenate([r['x'] for r in train]);mean=trainx.mean(0);std=np.maximum(trainx.std(0),.05)
    x=torch.tensor(np.clip((trainx-mean)/std,-8,8),dtype=torch.float32);targets=torch.tensor(np.concatenate([r['targets'] for r in train]))
    base=torch.tensor([n['baseline_class'] for r in train for n in r['notes']],dtype=torch.long)
    domains=Counter(r['kind'] for r in train);weights=[]
    for r in train:
        duration=np.array([min(3.,max(.03,n['offset']-n['onset'])) for n in r['notes']]);weights.extend(duration/duration.sum()/domains[r['kind']])
    weights=torch.tensor(weights,dtype=torch.float32);weights/=weights.mean()
    model=EventRefiner();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=baseline;policy={'threshold':1.,'margin':1.,'allow_removal':False};best_state=None;selected_epoch=0;history=[];candidates=[]
    folder=Path('runs')/condition;folder.mkdir(exist_ok=True)
    for epoch in range(1,max(protocol['epochs'])+1):
        model.train();optimizer.zero_grad()
        logp=model(x,base).log_softmax(-1);loss=(-(targets*logp).sum(-1)*weights).mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();history.append({'epoch':epoch,'loss':float(loss.item())})
        if epoch not in protocol['epochs']:continue
        model.eval()
        with torch.no_grad():
            probs=[model(torch.tensor(np.clip((r['x']-mean)/std,-8,8),dtype=torch.float32),torch.tensor([n['baseline_class'] for n in r['notes']],dtype=torch.long)).softmax(-1).numpy() for r in validation]
        for threshold in protocol['thresholds']:
            for margin in protocol['margins']:
                for remove in protocol['allow_removal']:
                    p={'threshold':threshold,'margin':margin,'allow_removal':remove}
                    result=score([proposed_parts(r['notes'],v,**p)[0] for r,v in zip(validation,probs)])
                    ok=feasible(result);candidates.append({'epoch':epoch,'policy':p,'feasible':ok,**{k:v for k,v in result.items() if k!='tracks'}})
                    if ok and result['selection_score']>best['selection_score']+1e-6:
                        best=result;policy=p;best_state=copy.deepcopy(model.state_dict());selected_epoch=epoch
        torch.save({'model':model.state_dict(),'schema':SCHEMA,'mean':mean.tolist(),'std':std.tolist(),'policy':policy,'automatic_application_approved':False},folder/f'epoch_{epoch}.pt')
        print('TRAIN',condition,epoch,loss.item(),'baseline',baseline['selection_score'],'best',best['selection_score'],flush=True)
    if best_state is None:best_state=model.state_dict()
    state={'model':best_state,'schema':SCHEMA,'mean':mean.tolist(),'std':std.tolist(),'policy':policy,'automatic_application_approved':False,
           'condition':condition,'selected_epoch':selected_epoch,'protocol_sha256':sha(protocol_path),'baseline_model_sha256':baseline_sha}
    torch.save(state,folder/'selected.pt');(folder/'history.json').write_text(json.dumps(history,indent=2))
    result={'condition':condition,'training_records':len(train),'training_candidates':len(trainx),'training_domains':dict(domains),
            'baseline':baseline,'selected':best,'policy':policy,'selected_epoch':selected_epoch,'all_validation_policies':candidates,
            'checkpoint':str(folder/'selected.pt'),'checkpoint_sha256':sha(folder/'selected.pt')}
    Path('reports',condition+'_validation.json').write_text(json.dumps(result,indent=2));summaries[condition]=result
selected=max(summaries,key=lambda name:summaries[name]['selected']['selection_score'])
state=torch.load(summaries[selected]['checkpoint'],map_location='cpu',weights_only=True)
torch.save(state,'models/muscriptor_refiner_duration.pt')
selection={'selected_by_validation':selected,'test_used_for_selection':False,'protocol_sha256':sha(protocol_path),
           'model_sha256':sha('models/muscriptor_refiner_duration.pt'),'automatic_application_approved':False,
           'results':{key:{k:v for k,v in result.items() if k!='all_validation_policies'} for key,result in summaries.items()}}
Path('reports/duration_selection.json').write_text(json.dumps(selection,indent=2));print('DURATION_SELECTION_FROZEN',selected,flush=True)
