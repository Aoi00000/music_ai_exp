"""CPU supervised residual correction, selected against identical Mu inputs."""
from pathlib import Path
from collections import defaultdict
import argparse,json,random,time,itertools
import numpy as np,torch
from band2tab.stem_refiner import SCHEMA,StemNoteCorrector,correction_loss,suggest
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,KEYS


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--width',type=int,default=128)
    p.add_argument('--positive-weight',type=float,default=1.5);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=19)
    p.add_argument('--band-only',action='store_true');p.add_argument('--monitor-steps',type=int,nargs='+');a=p.parse_args()
    if Path('reports/stem_refiner_v1/selection.json').exists():raise ValueError('Final stem selection frozen; do not tune on final tests')
    root=Path('backups/azure');manifest=root/'data/stem_refiner_v1/manifest.json';rows=json.load(open(manifest))
    if any(r['split'] not in ('train','val') for r in rows):raise ValueError('No test access during training')
    cache={};metadata={}
    for r in rows:
        path=root/'data/stem_refiner_v1/cache'/(r['id']+'.npz')
        with np.load(path) as f:cache[r['id']]={k:f[k].copy() for k in f.files}
        metadata[r['id']]=json.load(open(path.with_suffix('.json')))
    train=[r for r in rows if r['split']=='train' and len(cache[r['id']]['x']) and (not a.band_only or r['dataset'].startswith('stem_band_'))];validation=[r for r in rows if r['split']=='val']
    x=np.concatenate([cache[r['id']]['x'] for r in train]);mean=x.mean(0);std=x.std(0).clip(.05);del x
    for c in cache.values():c['x']=np.clip((c['x']-mean)/std,-8,8).astype(np.float32)
    grouped=defaultdict(lambda:defaultdict(list))
    for r in train:grouped[r['dataset']][r['group']].append(r)
    folder=Path('runs/stem_refiner_v1')/a.name
    if folder.exists():raise ValueError('Run already exists')
    config={**vars(a),'manifest_sha256':sha(manifest),'train_records':len(train),'validation_records':len(validation),
        'cache_sha256':{r['id']:sha(root/'data/stem_refiner_v1/cache'/(r['id']+'.npz')) for r in rows},
        'selection':'Same-MIDI frozen source-known Mu, dataset macro across clean/noisy validation. IDMT not used for selection.',
        'test_used_for_selection':False}
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);rng=random.Random(a.seed)
    model=StemNoteCorrector(a.width);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01);config['parameters']=sum(p.numel() for p in model.parameters());save(folder/'config.json',config)
    controls=aggregate([(r,metrics(r,metadata[r['id']]['baseline_parts'])) for r in validation]);best=None;history=[];started=time.monotonic()
    for step in range(1,a.steps+1):
        batch=[]
        for _ in range(8):
            dataset=rng.choice(sorted(grouped));group=rng.choice(sorted(grouped[dataset]));row=rng.choice(grouped[dataset][group]);c=cache[row['id']]
            indices=np.random.randint(0,len(c['x']),size=32);batch.append({k:v[indices] for k,v in c.items()})
        tensors={k:torch.from_numpy(np.concatenate([b[k] for b in batch])) for k in batch[0]}
        model.train();opt.zero_grad(set_to_none=True);loss=correction_loss(model(tensors['x']),tensors,a.positive_weight);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:print('STEM_CORRECTOR_TRAIN',a.name,step,float(loss.detach()),flush=True)
        if step not in sorted(set(a.monitor_steps or [min(500,a.steps),min(1000,a.steps),a.steps])):continue
        model.eval();predictions={}
        with torch.no_grad():
            for r in validation:
                z=model(torch.from_numpy(cache[r['id']]['x']));predictions[r['id']]={'probability':z['logit'].sigmoid().numpy(),'delta':z['delta'].numpy()}
        choices=[]
        # Include the unchanged control and low removal rates: errors in
        # source-separated audio do not justify deleting uncertain true notes.
        for threshold,retime in itertools.product((0.,.01,.05,.1,.2,.35,.5,.65,.8),(False,True)):
            policy={'threshold':threshold,'retime':retime};scores=[];tracks=[]
            for r in validation:
                meta=metadata[r['id']];pred=predictions[r['id']];parts=suggest(meta['pool'],pred['probability'],pred['delta'],r['duration'],unmodelled=meta['unmodelled'],**policy)
                score=metrics(r,parts);scores.append((r,score));tracks.append({'id':r['id'],'dataset':r['dataset'],'parts':parts,'metrics':{k:score[k] for k in KEYS}})
            choices.append({'policy':policy,'summary':aggregate(scores),'tracks':tracks})
        selected=max(choices,key=lambda r:r['summary']['score']);checkpoint=folder/('step_'+str(step)+'.pt')
        torch.save({'schema':SCHEMA,'model':model.state_dict(),'width':a.width,'mean':mean.tolist(),'std':std.tolist(),'config':config,'step':step,'policy':selected['policy'],'automatic_application_approved':False},checkpoint)
        record={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'step':step,**selected}
        history.append({'step':step,'choices':[{k:v for k,v in r.items() if k!='tracks'} for r in choices]});save(folder/'history.json',history)
        if best is None or selected['summary']['score']>best['summary']['score']:best=record
        save(folder/'progress.json',{'best':best,'latest':record,'controls':controls})
        print('STEM_CORRECTOR_VALIDATION',a.name,step,selected['policy'],{k:round(selected['summary'][k],4) for k in (*KEYS,'score')},flush=True)
    save(folder/'completed.json',{'best':best,'controls':controls,'config':config,'seconds':time.monotonic()-started,'test_used_for_selection':False})
    print('STEM_CORRECTOR_COMPLETE',a.name,flush=True)


if __name__=='__main__':main()
