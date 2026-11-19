"""Preregistered source-expansion and sounding-quality additions, 2026-11-20."""
from pathlib import Path
from collections import Counter
import argparse,copy,json,random,time,hashlib
import numpy as np,soundfile as sf,torch
from band2tab.audio import features
from band2tab.stem_refiner import rms_envelope,read_candidates
from band2tab.residual import SCHEMA,DIMENSION,ResidualNet,candidate_pool,residual_features,teacher,apply,probabilities
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate
from scripts.a_stress_data import local
DATA=Path('data/residual_v1');REPORT=Path('reports/residual_v1');RUN=Path('runs/residual_v1');BACK=Path('backups/azure')
ARMS={'your_only':(False,False,False),'octave_onset':(True,False,False),'octave_sound':(True,True,False),'octave_diverse':(True,True,True)}
SEEDS=(17,29,43)

def prepare():
    if (DATA/'protocol.json').exists():raise ValueError('Protocol frozen')
    rows=copy.deepcopy(json.load(open('data/sound_keep_v6/manifest.json')))
    for r in rows:r['base_cache']=r['cache'];r['extra_domain']=False
    old=json.load(open(BACK/'data/candidate_v2/manifest.json'));more=[r for r in old if r['split']=='train' and r['dataset'] in ('guitarset','guitar_techs_clean')]
    for r in more:
        r=copy.deepcopy(r);r.update(phase='train',extra_domain=True,base_cache=None);rows.append(r)
    train_groups={r['group'] for r in rows if r['phase']=='train'};cal_groups={r['group'] for r in rows if r['phase']=='calibration'}
    if train_groups&cal_groups:raise ValueError('Source overlap')
    save(DATA/'manifest.json',rows)
    save(DATA/'protocol.json',dict(created='2026-11-20',schema=SCHEMA,arms=ARMS,seeds=SEEDS,steps=800,learning_rate=.0003,weight_decay=.1,
       normalization='Original training domains only, common to arms; no calibration',sampling='Common GOAT band rows with a valid Your proposal; diverse substitutes 25% with other train rows (GuitarSet/TECHS/Slakh/augmented bass). Same row/index uniforms.',
       positive_weight=8.,quality_loss_weight=.5,teacher='Strict missing-note one-to-one onset target; optional nonnegative marginal Frame gain target.',
       proposals='Frozen Your guitar notes and +/-12 semitones of Mu G/B; no truth used in proposal/features; all originals preserved.',
       thresholds=[.5,.65,.8,.9,.95,.99,1.1],selection='Internal24 only, per-dataset Note/Frame/exact/offset all >= baseline; choose .5 Note+.3 Frame+.2 exact; identity allowed.',
       rounds='4 arms x 3 seeds, final800 only, no checkpoint search',calibration_groups=sorted(cal_groups),training_records=len(rows)-24,
       dataset_counts=dict(Counter((r['dataset']+':'+r['phase']) for r in rows)),manifest_sha256=sha(DATA/'manifest.json'),
       implementation_sha256={p:sha(p) for p in ['band2tab/residual.py','scripts/a_residual_cycle.py']},unopened_test_used=False,automatic_application_approved=False))
    print('RESIDUAL_REGISTERED',len(rows),len(more),flush=True)

def baseline_and_support(r):
    if r.get('base_cache'):
        m=json.load(open(Path(r['base_cache']).with_suffix('.json')));base=m['baseline_parts']
    else:
        base={p:[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in read_candidates(BACK/'reports/candidate_v2/midi'/r['id']/(p+'.mid'),p,r['duration'])] for p in ('guitar','bass')}
    support={p:[] for p in ('guitar','bass')};your=Path('reports/addition_v1/midi')/(r['id']+'.mid')
    if your.exists():support['guitar']=read_candidates(your,'guitar',r['duration'])
    return base,support

def cache_rows(rows):
    result=[]
    for r in rows:
        path=DATA/'cache'/(r['id']+'.npz');meta=path.with_suffix('.json');base,support=baseline_and_support(r);pool=candidate_pool(base,support,r['duration'],r['guard_seconds'])
        inputs={p:local(BACK,v) for p,v in r['inputs'].items() if p in ('mix','guitar','bass')}
        fp=dict(audio={p:sha(v) for p,v in inputs.items()},baseline_sha256=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest(),pool_sha256=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest(),implementation_sha256=sha('band2tab/residual.py'))
        if meta.exists():
            if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale residual cache')
        else:
            views={p:features(v) for p,v in inputs.items()};env={}
            for p,v in inputs.items():y,sr=sf.read(v,dtype='float32',always_2d=True);env[p]=rms_envelope(y,sr)
            x=residual_features(views,env,pool,base,r['duration']);target=teacher(pool,base,r['parts'],r['duration'],r['guard_seconds'])
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,x=x,**target,origin=np.array([n['origin'][0] for n in pool],np.float32))
            save(meta,dict(fingerprint=fp,pool=pool,baseline_parts=base))
        result.append({**r,'residual_cache':str(path)})
        print('RESIDUAL_CACHED',r['id'],len(pool),flush=True)
    return result

def load():
    rows=json.load(open(DATA/'manifest.json'));cache={};meta={}
    for r in rows:
        p=Path(r['residual_cache'])
        with np.load(p) as z:cache[r['id']]={k:z[k].copy() for k in z.files}
        meta[r['id']]=json.load(open(p.with_suffix('.json')))
    return rows,cache,meta

def train(device):
    protocol=json.load(open(DATA/'protocol.json'))
    for p,h in protocol['implementation_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen implementation changed')
    rows,cache,meta=load();base=[r for r in rows if r['phase']=='train' and r['dataset'].startswith('stem_band_') and np.any((cache[r['id']]['origin']>.5)&(cache[r['id']]['mask']>.5))];base_ids={r['id'] for r in base};extra=[r for r in rows if r['phase']=='train' and r['id'] not in base_ids and np.any(cache[r['id']]['mask']>.5)]
    x=np.concatenate([cache[r['id']]['x'] for r in base]);mean=x.mean(0);std=x.std(0).clip(.05)
    for c in cache.values():c['z']=np.clip((c['x']-mean)/std,-8,8).astype(np.float32)
    torch.set_num_threads(2)
    for arm,(expanded,quality,diverse) in ARMS.items():
      for seed in SEEDS:
        folder=RUN/arm/('seed_'+str(seed))
        if (folder/'completed.json').exists():continue
        if folder.exists():raise ValueError('Incomplete run must not be overwritten')
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=random.Random(seed);gen=np.random.default_rng(seed)
        model=ResidualNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);history=[];draw=hashlib.sha256();started=time.monotonic()
        cfg=dict(arm=arm,seed=seed,steps=800,parameters=sum(p.numel() for p in model.parameters()),quality=quality,expanded=expanded,diverse=diverse,protocol_sha256=sha(DATA/'protocol.json'))
        save(folder/'config.json',cfg)
        for step in range(1,801):
            batch=[]
            for _ in range(8):
                u=rng.random();ib=rng.randrange(len(base));ie=rng.randrange(len(extra));r=extra[ie] if diverse and u<.25 else base[ib];c=cache[r['id']]
                candidates=np.flatnonzero((c['mask']>.5)&(np.ones(len(c['x']),bool) if expanded else c['origin']>.5))
                uniforms=gen.random(32);draw.update(np.array([u,ib,ie],np.float64).tobytes()+uniforms.tobytes())
                if not len(candidates):continue
                ix=candidates[np.minimum(len(candidates)-1,(uniforms*len(candidates)).astype(int))]
                batch.append({k:c[k][ix] for k in ('z','y','sound','mask')})
            if not batch:raise ValueError('Empty learning batch')
            t={k:torch.as_tensor(np.concatenate([b[k] for b in batch]),device=device) for k in batch[0]};model.train();opt.zero_grad(set_to_none=True);out=model(t['z'])
            loss=torch.nn.functional.binary_cross_entropy_with_logits(out[:,0],t['y'],pos_weight=torch.tensor(8.,device=device))
            if quality:loss=loss+.5*torch.nn.functional.binary_cross_entropy_with_logits(out[:,1],t['sound'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%200==0:history.append(dict(step=step,loss=float(loss.detach())));print('RESIDUAL_TRAIN',arm,seed,step,float(loss.detach()),flush=True)
        state=dict(schema=SCHEMA,members=[{k:v.cpu() for k,v in model.state_dict().items()}],mean=mean.tolist(),std=std.tolist(),quality=quality,expanded=expanded,config=cfg)
        torch.save(state,folder/'final.pt');save(folder/'completed.json',dict(config=cfg,seconds=time.monotonic()-started,draw_sha256=draw.hexdigest(),history=history,checkpoint_sha256=sha(folder/'final.pt')))
    save(REPORT/'training_completed.json',dict(runs=12,steps=9600,test_used=False))

def evaluate(rows,cache,meta,qs,thresholds):
    items=[];tracks=[]
    for r in rows:
        m=meta[r['id']];parts=apply(m['baseline_parts'],m['pool'],qs[r['id']],thresholds);s=metrics(r,parts);items.append((r,s));tracks.append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],metrics=s,parts=parts))
    return dict(summary=aggregate(items),tracks=tracks)

def select():
    if (REPORT/'selection.json').exists():raise ValueError('Selection frozen')
    rows,cache,meta=load();cal=[r for r in rows if r['phase']=='calibration'];torch.set_num_threads(2)
    baseline=evaluate(cal,cache,meta,{r['id']:np.zeros(len(cache[r['id']]['x'])) for r in cal},(1.1,1.1));choices=[]
    for arm in ARMS:
        states=[torch.load(RUN/arm/('seed_'+str(s))/'final.pt',map_location='cpu',weights_only=True) for s in SEEDS];bundle={**states[0],'members':[s['members'][0] for s in states]}
        qs={r['id']:probabilities(bundle,cache[r['id']]['x']) for r in cal}
        if not ARMS[arm][0]:
            for r in cal:qs[r['id']]=np.where(cache[r['id']]['origin']>.5,qs[r['id']],0.)
        arm_choices=[]
        import itertools
        for thresholds in itertools.product((.5,.65,.8,.9,.95,.99,1.1),repeat=2):
            value=evaluate(cal,cache,meta,qs,thresholds);s=value['summary'];b=baseline['summary'];eligible=all(s['datasets'][d][k]>=v[k]-1e-10 for d,v in b['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'));score=.5*s['note_f1']+.3*s['frame_f1']+.2*s['active_exact_match']
            arm_choices.append(dict(thresholds=thresholds,eligible=eligible,score=score,summary=s))
        allowed=[c for c in arm_choices if c['eligible']];best=max(allowed,key=lambda c:c['score']);bundle['thresholds']=best['thresholds'];torch.save(bundle,RUN/arm/'selected_bundle.pt');choices.append(dict(arm=arm,best=best,choices=arm_choices,export=str(RUN/arm/'selected_bundle.pt'),sha256=sha(RUN/arm/'selected_bundle.pt')))
        print('RESIDUAL_SELECTED',arm,best['thresholds'],best['summary'],flush=True)
    save(REPORT/'selection.json',dict(baseline=baseline['summary'],arms=choices,selected_before_review=True,test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','cache','train','select']);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='cache':save(DATA/'manifest.json',cache_rows(json.load(open(DATA/'manifest.json'))))
    elif a.command=='train':train(a.device)
    else:select()
