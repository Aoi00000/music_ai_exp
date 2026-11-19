"""Learn to accept only complementary guitar notes; compare unfiltered union."""
from pathlib import Path
from collections import defaultdict
import json,random,time,argparse
import numpy as np,torch
from band2tab.addition import SCHEMA,AdditionRanker,add_notes
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['linear','mlp'],required=True)
    parser.add_argument('--experiment',choices=['addition_v1','addition_balanced_v2','addition_jazz_v3_control','addition_jazz_v3'],default='addition_v1');a=parser.parse_args()
    positive_weight=2. if a.experiment=='addition_v1' else 8.
    report=Path('reports')/a.experiment;folder=Path('runs')/a.experiment/a.kind
    if (report/'selection.json').exists() or folder.exists():raise ValueError('Do not overwrite/further tune frozen experiments')
    jazz_experiment=a.experiment.startswith('addition_jazz_')
    manifest=Path('data/addition_jazz_v3/manifest.json' if jazz_experiment else 'data/addition_v1/manifest.json');rows=json.load(open(manifest));cache={};metadata={}
    if any(r['split'] not in ('train','val') for r in rows):raise ValueError('No held-out test access')
    protocol=dict(schema=SCHEMA,models=['linear','mlp'],steps=1000,monitor_steps=[250,1000],seed=19,thresholds=[.5,.8,.95,1.1],
        baseline='Same frozen source-known separated Mu; never delete or retime any baseline note',proposal_model='Frozen augmentation YourMT3 adapter, source-known guitar stem',
        purpose='Learn true missing notes instead of adding all proposals',primary='0.5 Note F1 + 0.3 Frame F1 + 0.2 pitch-set F1',
        gates='Per clean/noisy dataset Note and Frame >= baseline; aggregate active-exact >= baseline',test_used=False,manifest_sha256=sha(manifest))
    if a.experiment!='addition_v1':protocol.update(positive_loss_weight=positive_weight,hypothesis='Only 109/2024 training candidates were missing true notes. Increase positive loss weight from 2 to 8; same data/order/steps/models/thresholds.')
    if jazz_experiment:
        if a.kind!='linear':raise ValueError('Targeted jazz experiment preregisters the linear model only')
        protocol.update(models=['linear'],hypothesis='Compare original data vs 12 jazz recordings; identical common normalization, steps, model and positive weight. With-jazz gives jazz 20% of samples.',
            normalization='Original band training features only, shared by both conditions',jazz_training=a.experiment=='addition_jazz_v3',jazz_validation='Six Jazz3 recordings; correction-head holdout only; older upstream/model families may have seen these sources')
    p=report/'protocol.json'
    if p.exists() and json.load(open(p))!=protocol:raise ValueError('Addition protocol changed')
    save(p,protocol)
    for r in rows:
        f=Path(r.get('addition_cache',str(Path('data/addition_v1/cache')/(r['id']+'.npz'))))
        with np.load(f) as data:cache[r['id']]={k:data[k].copy() for k in data.files}
        metadata[r['id']]=json.load(open(f.with_suffix('.json')))
    train=[r for r in rows if r['split']=='train' and len(cache[r['id']]['x']) and not (a.experiment=='addition_jazz_v3_control' and r['dataset']=='jazz_guitarset')];val=[r for r in rows if r['split']=='val']
    normalization=[r for r in train if not jazz_experiment or r['dataset']!='jazz_guitarset']
    x=np.concatenate([cache[r['id']]['x'] for r in normalization]);mean=x.mean(0);std=x.std(0).clip(.05)
    for c in cache.values():c['x']=np.clip((c['x']-mean)/std,-8,8).astype(np.float32)
    grouped=defaultdict(list)
    for r in train:grouped[r['group']].append(r)
    def evaluate(predictions,threshold):
        items=[];tracks=[]
        for r in val:
            m=metadata[r['id']];parts=add_notes(m['baseline'],m['pool'],predictions[r['id']],threshold);s=metrics(r,parts);items.append((r,s));tracks.append(dict(id=r['id'],dataset=r['dataset'],metrics=s,parts=parts))
        s=aggregate(items);s['addition_score']=.5*s['note_f1']+.3*s['frame_f1']+.2*s['union_frame_f1']
        return dict(threshold=threshold,summary=s,tracks=tracks)
    ones={r['id']:np.ones(len(cache[r['id']]['x'])) for r in val};control=evaluate(ones,1.1);union=evaluate(ones,0.)
    torch.set_num_threads(2);torch.manual_seed(19);np.random.seed(19);rng=random.Random(19);model=AdditionRanker(a.kind)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);history=[];started=time.monotonic()
    config=dict(kind=a.kind,protocol_sha256=sha(p),train_records=len(train),train_candidates=sum(len(cache[r['id']]['x']) for r in train),train_positive=int(sum(cache[r['id']]['y'].sum() for r in train)),parameters=sum(p.numel() for p in model.parameters()))
    save(folder/'config.json',config)
    for step in range(1,1001):
        batch=[]
        for _ in range(8):
            available=sorted(grouped)
            if a.experiment=='addition_jazz_v3':
                jazz=rng.random()<.2;available=[g for g in available if g.startswith('guitarset_Jazz')==jazz]
            g=rng.choice(available);r=rng.choice(grouped[g]);c=cache[r['id']];ix=np.random.randint(0,len(c['x']),32);batch.append({k:v[ix] for k,v in c.items()})
        t={k:torch.from_numpy(np.concatenate([b[k] for b in batch])) for k in batch[0]}
        model.train();opt.zero_grad(set_to_none=True);raw=torch.nn.functional.binary_cross_entropy_with_logits(model(t['x']),t['y'],reduction='none',pos_weight=torch.tensor(positive_weight));loss=(raw*t['mask']).sum()/t['mask'].sum().clamp_min(1)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step not in protocol['monitor_steps']:continue
        model.eval()
        with torch.no_grad():pred={r['id']:model(torch.from_numpy(cache[r['id']]['x'])).sigmoid().numpy() for r in val}
        choices=[evaluate(pred,t) for t in protocol['thresholds']];path=folder/f'step_{step}.pt'
        torch.save(dict(schema=SCHEMA,kind=a.kind,model=model.state_dict(),mean=mean.tolist(),std=std.tolist(),config=config,step=step),path)
        history.append(dict(step=step,checkpoint=str(path),sha256=sha(path),choices=choices));save(folder/'history.json',history)
        print('ADDITION_VALIDATED',a.kind,step,max(c['summary']['addition_score'] for c in choices),flush=True)
    save(folder/'completed.json',dict(config=config,control=control,unfiltered_union=union,history=history,seconds=time.monotonic()-started));print('ADDITION_TRAIN_COMPLETE',a.kind,flush=True)


if __name__=='__main__':main()
