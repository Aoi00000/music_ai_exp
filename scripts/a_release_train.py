"""New experiment: release learning with an exact onset/count protection rule."""
from pathlib import Path
from collections import defaultdict
import argparse,json,random,time
import numpy as np,torch
from band2tab.release import SCHEMA,ReleaseCorrector,release_loss,apply_release,predict_release
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate

ROOT=Path('backups/azure');REPORT=Path('reports/release_v2');RUN=Path('runs/release_v2')


def load():
    manifest=ROOT/'data/stem_refiner_v1/manifest.json';rows=json.load(open(manifest));cache={};metadata={}
    if any(r['split'] not in ('train','val') for r in rows):raise ValueError('No held-out labels during selection')
    for r in rows:
        p=ROOT/'data/stem_refiner_v1/cache'/(r['id']+'.npz')
        with np.load(p) as f:cache[r['id']]={k:f[k].copy() for k in f.files}
        metadata[r['id']]=json.load(open(p.with_suffix('.json')))
    return manifest,rows,cache,metadata


def evaluate(rows,metadata,predictions,scale):
    scores=[];tracks=[]
    for r in rows:
        m=metadata[r['id']];parts=apply_release(m['pool'],predictions[r['id']],r['duration'],scale,m['unmodelled']);s=metrics(r,parts)
        original=metrics(r,m['baseline_parts'])
        if s['note_f1']!=original['note_f1']:raise ValueError('Protected onset/count invariant failed')
        scores.append((r,s));tracks.append(dict(id=r['id'],dataset=r['dataset'],metrics=s,parts=parts))
    summary=aggregate(scores);summary['release_score']=.6*summary['frame_f1']+.4*summary['union_frame_f1']
    return dict(scale=scale,summary=summary,tracks=tracks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--kind',choices=['linear','mlp'],default='linear');p.add_argument('--data',choices=['band','all'],default='band');p.add_argument('--seed',type=int,default=19);p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    if a.name!=a.kind+'_'+a.data or a.seed!=19 or a.steps!=1000:raise ValueError('Arguments differ from the preregistered four-run protocol')
    if (REPORT/'selection.json').exists():raise ValueError('Release protocol is frozen')
    manifest,rows,cache,metadata=load();validation=[r for r in rows if r['split']=='val']
    train=[r for r in rows if r['split']=='train' and len(cache[r['id']]['x']) and (a.data=='all' or r['dataset'].startswith('stem_band_'))]
    protocol=dict(schema=SCHEMA,manifest_sha256=sha(manifest),source_cache_schema='source_isolated_unique_assignment_v2',
        runs=['linear_band','linear_all','mlp_band','mlp_all'],steps=1000,seed=19,monitors=[250,1000],scales=[0.,.25,.5,1.],
        primary='0.6 instrument Frame F1 + 0.4 pitch-set F1; preserve exact Note F1; active exact >= baseline - 0.005',
        transformation='Only release; retain every pitch, role, onset and note. Bound by 150ms and half original note duration.',
        final_test='Unopened Slakh source UUIDs (11); old GOAT heldout is now a diagnostic regression, never a new test',test_used_for_selection=False)
    path=REPORT/'protocol.json'
    if path.exists() and json.load(open(path))!=protocol:raise ValueError('Release protocol changed')
    save(path,protocol)
    folder=RUN/a.name
    if folder.exists():raise ValueError('Do not overwrite a previous release trial')
    x=np.concatenate([cache[r['id']]['x'] for r in train]);mean=x.mean(0);std=x.std(0).clip(.05)
    for c in cache.values():c['x']=np.clip((c['x']-mean)/std,-8,8).astype(np.float32)
    grouped=defaultdict(lambda:defaultdict(list))
    for r in train:grouped[r['dataset']][r['group']].append(r)
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);rng=random.Random(a.seed)
    model=ReleaseCorrector(a.kind);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.1)
    config={**vars(a),'manifest_sha256':sha(manifest),'protocol_sha256':sha(path),'train_records':len(train),'train_note_candidates':len(x),'parameters':sum(p.numel() for p in model.parameters()),'normalization_training_only':True}
    save(folder/'config.json',config);history=[];started=time.monotonic()
    controls=evaluate(validation,metadata,{r['id']:np.zeros(len(cache[r['id']]['x'])) for r in validation},0.)
    for step in range(1,a.steps+1):
        batch=[]
        for _ in range(8):
            d=rng.choice(sorted(grouped));g=rng.choice(sorted(grouped[d]));r=rng.choice(grouped[d][g]);c=cache[r['id']];ix=np.random.randint(0,len(c['x']),32);batch.append({k:v[ix] for k,v in c.items()})
        t={k:torch.from_numpy(np.concatenate([b[k] for b in batch])) for k in batch[0]}
        model.train();opt.zero_grad(set_to_none=True);loss=release_loss(model(t['x']),t);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step not in (250,a.steps):continue
        model.eval()
        with torch.no_grad():pred={r['id']:model(torch.from_numpy(cache[r['id']]['x'])).numpy() for r in validation}
        choices=[evaluate(validation,metadata,pred,s) for s in protocol['scales']]
        checkpoint=folder/f'step_{step}.pt'
        torch.save(dict(schema=SCHEMA,kind=a.kind,model=model.state_dict(),mean=mean.tolist(),std=std.tolist(),step=step,config=config),checkpoint)
        history.append(dict(step=step,checkpoint=str(checkpoint),sha256=sha(checkpoint),choices=choices));save(folder/'history.json',history)
        print('RELEASE_VALIDATED',a.name,step,max(c['summary']['release_score'] for c in choices),flush=True)
    save(folder/'completed.json',dict(config=config,history=history,control=controls,seconds=time.monotonic()-started));print('RELEASE_TRAIN_COMPLETE',a.name,flush=True)


if __name__=='__main__':main()
