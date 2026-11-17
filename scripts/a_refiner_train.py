"""Train an event refiner; choose policies on validation, never on IDMT."""
from pathlib import Path
import json,hashlib,copy
import numpy as np
import torch
import mir_eval
from band2tab.audio import features
from band2tab.refine import candidates_from_midi,event_features,EventRefiner,proposed_parts,SCHEMA
from band2tab.learn import note_metrics,pitch_set_metrics

torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
protocol_path=Path('reports/refiner_protocol.json');protocol=json.loads(protocol_path.read_text())
manifest=Path('data/manifest_refiner.json')
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=protocol['manifest_sha256']:raise ValueError('Frozen data changed')
out=Path('runs/muscriptor_refiner');out.mkdir(parents=True,exist_ok=True)
if (out/'selected.pt').exists():raise RuntimeError('Refiner already fitted; do not overwrite')
records=[];allx=[];ally=[];allbase=[];allweight=[];prediction_ledger=[]
baseline_sha=hashlib.sha256(Path('models/muscriptor-medium/model.safetensors').read_bytes()).hexdigest()
for it in json.loads(manifest.read_text()):
    path=Path('reports/refiner_base')/(it['id']+'.mid')
    meta=json.loads(path.with_suffix('.json').read_text())
    if meta['audio_sha256']!=it['audio_sha256'] or meta['model_sha256']!=baseline_sha:
        raise ValueError('Stale baseline prediction: '+it['id'])
    if hashlib.sha256(Path(it['audio']).read_bytes()).hexdigest()!=it['audio_sha256']:
        raise ValueError('Audio changed: '+it['id'])
    prediction_ledger.append({'id':it['id'],'midi_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    notes=[n for n in candidates_from_midi(path) if n['onset']>=.25]
    truth={part:[n for n in it['parts'][part] if n['onset']>=.25] for part in ('guitar','bass')}
    x=event_features(features(it['audio']),notes);labels=np.zeros(len(notes),np.int64);matched=[]
    for part in ('guitar','bass'):
        ref=truth[part]
        if not ref or not notes:matched.append({});continue
        intervals=lambda ns:np.array([[n['onset'],n['offset']] for n in ns])
        pitches=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
        pairs=mir_eval.transcription.match_notes(intervals(ref),pitches(ref),intervals(notes),pitches(notes),onset_tolerance=.05,offset_ratio=None)
        matched.append({e:r for r,e in pairs})
    for j,n in enumerate(notes):
        valid=[i+1 for i,m in enumerate(matched) if j in m]
        labels[j]=valid[0] if len(valid)==1 else n['baseline_class'] if n['baseline_class'] in valid else -1 if len(valid)>1 else 0
    records.append({**it,'notes':notes,'truth':truth,'x':x})
    if it['split']=='train' and len(notes):
        ok=labels>=0;allx.append(x[ok]);ally.append(labels[ok]);allbase.extend(n['baseline_class'] for n,k in zip(notes,ok) if k)
        allweight.extend([1/max(1,int(ok.sum()))]*int(ok.sum()))
    print('FEATURES',it['id'],len(notes),np.bincount(labels[labels>=0],minlength=3).tolist(),flush=True)
train=np.concatenate(allx);mean=train.mean(0);std=np.maximum(train.std(0),.05)
for r in records:r['z']=torch.tensor(np.clip((r['x']-mean)/std,-8,8),dtype=torch.float32)
x=torch.tensor(np.clip((train-mean)/std,-8,8),dtype=torch.float32);y=torch.tensor(np.concatenate(ally));base=torch.tensor(allbase)
weight=torch.tensor(allweight,dtype=torch.float32);weight/=weight.mean()
model=EventRefiner();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
np.savez_compressed(out/'training_features.npz',x=train,y=y.numpy(),baseline=base.numpy(),weights=weight.numpy())
validation=[r for r in records if r['split']=='val']
def score(preds):
    rows=[]
    for r,parts in zip(validation,preds):
        part_scores={part:note_metrics(r['truth'][part],parts[part])['f1'] for part in ('guitar','bass')}
        union=pitch_set_metrics([n for ns in r['truth'].values() for n in ns],[n for ns in parts.values() for n in ns])
        rows.append({'id':r['id'],'group':r['group'],'kind':r['kind'],'part_note_f1':part_scores,'union_frame_f1':union['frame_f1']})
    # Equal weight for mixture and solo domains, not equal number of excerpts.
    domains={kind:float(np.mean([v for r in rows if r['kind']==kind for v in r['part_note_f1'].values()])) for kind in ('mixture','solo')}
    return {'macro_note_f1':float(np.mean(list(domains.values()))),'by_domain_note_f1':domains,
            'mixture_union_frame_f1':float(np.mean([r['union_frame_f1'] for r in rows if r['kind']=='mixture'])),'tracks':rows}
identity=[proposed_parts(r['notes'],np.eye(3)[[n['baseline_class'] for n in r['notes']]],1.,1.)[0] for r in validation]
baseline_score=score(identity);best_score=baseline_score;best_state=None;best_policy={'threshold':1.,'margin':1.};history=[];policies=[]
for epoch in range(1,151):
    model.train();optimizer.zero_grad();loss=(torch.nn.functional.cross_entropy(model(x,base),y,reduction='none')*weight).mean()
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
    history.append({'epoch':epoch,'train_cross_entropy':float(loss.item())})
    if epoch not in protocol['epochs']:continue
    model.eval()
    with torch.no_grad():probs=[model(r['z'],torch.tensor([n['baseline_class'] for n in r['notes']],dtype=torch.long)).softmax(-1).numpy() for r in validation]
    for threshold in protocol['thresholds']:
        for margin in protocol['margins']:
            policy={'threshold':threshold,'margin':margin}
            scored=score([proposed_parts(r['notes'],p,**policy)[0] for r,p in zip(validation,probs)])
            policies.append({'epoch':epoch,'policy':policy,**scored})
            if scored['macro_note_f1']>best_score['macro_note_f1']+1e-6:
                best_state=copy.deepcopy(model.state_dict());best_score=scored;best_policy=policy;selected_epoch=epoch
    print('EPOCH',epoch,'loss',loss.item(),'baseline',baseline_score['macro_note_f1'],'best',best_score['macro_note_f1'],flush=True)
    torch.save({'model':model.state_dict(),'schema':SCHEMA,'mean':mean.tolist(),'std':std.tolist(),'policy':{'threshold':.75,'margin':.2},'automatic_application_approved':False},out/f'epoch_{epoch}.pt')
if best_state is None:
    best_state=model.state_dict();best_policy={'threshold':1.,'margin':1.};selected_epoch=150
state={'schema':SCHEMA,'model':best_state,'mean':mean.tolist(),'std':std.tolist(),'policy':best_policy,'selected_epoch':selected_epoch,
       'automatic_application_approved':False,'selected_on':'validation only','protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
       'baseline_model_sha256':baseline_sha}
torch.save(state,out/'selected.pt');torch.save(state,Path('models/muscriptor_refiner.pt'))
report={'scope':'validation pilot, not a new external test; tiny synthetic mixture groups and solo excerpts',
        'training_candidates':len(y),'training_label_counts':np.bincount(y.numpy(),minlength=3).tolist(),
        'parameters':sum(p.numel() for p in model.parameters()),'baseline':baseline_score,'selected':best_score,'policy':best_policy,
        'epoch':selected_epoch,'selection_uses_test':False,'automatic_application_approved':False,
        'model_sha256':hashlib.sha256(Path('models/muscriptor_refiner.pt').read_bytes()).hexdigest(),'all_validation_policies':policies,
        'baseline_prediction_checksums':prediction_ledger}
Path('reports/refiner_validation.json').write_text(json.dumps(report,indent=2));(out/'history.json').write_text(json.dumps(history,indent=2))
Path('models/muscriptor_refiner.json').write_text(json.dumps({k:v for k,v in report.items() if k not in ('baseline','selected','all_validation_policies')},indent=2))
print('REFINER_TRAINING_COMPLETE',report['baseline']['macro_note_f1'],report['selected']['macro_note_f1'],report['policy'],flush=True)
