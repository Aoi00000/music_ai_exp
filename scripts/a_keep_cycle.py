"""A preregistered keep-only experiment, with internal-source calibration first."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,random,hashlib,time
import numpy as np,torch
from band2tab.keep import SCHEMA,KeepNet,keep_targets,probabilities,prune
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import aggregate,metrics,KEYS

DATA=Path('data/keep_cycle_v1');REPORT=Path('reports/keep_cycle_v1');RUN=Path('runs/keep_cycle_v1')
ARMS={'onset_linear':('onset','linear'),'joint_linear':('joint','linear'),'joint_mlp':('joint','mlp')}
SEEDS=[17,29,43]

def prepare():
    from scripts.a_release_train import load
    if DATA.exists():raise ValueError('Never overwrite a frozen preparation')
    manifest,rows,cache,meta=load();train=[r for r in rows if r['split']=='train']
    ggroups=sorted({r['group'] for r in train if r['dataset'].startswith('stem_band_')},key=lambda g:hashlib.sha256(('keep-source-calibration-v1-'+g).encode()).hexdigest())
    if len(ggroups)!=47:raise ValueError('Expected 47 training guitar performances')
    calibration=set(ggroups[:10]);records=[];all_targets=Counter()
    for r in train:
        subset='calibration' if r['group'] in calibration else 'train';m=meta[r['id']]
        targets=keep_targets(m['pool'],r['parts'],m['baseline_parts'],r['guard_seconds'],r['duration'])
        path=DATA/'cache'/(r['id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,x=cache[r['id']]['x'],**targets)
        save(path.with_suffix('.json'),m)
        records.append({**r,'phase':subset,'cache':str(path)})
        if subset=='train':
            use=targets['mask']>0;all_targets.update(records=1,candidates=int(use.sum()),onset_positive=int(targets['onset'][use].sum()),joint_positive=int(targets['joint'][use].sum()))
    save(DATA/'manifest.json',records)
    protocol=dict(schema=SCHEMA,created='2026-11-19',arms=ARMS,seeds=SEEDS,steps=800,
        batch_records=8,notes_per_record=32,learning_rate=.0003,weight_decay=.1,positive_keep_weight=3.,
        sampling={'band':.8,'augmented_bass':.2},normalization='Fit-training only; calibration excluded',
        calibration_guitar_groups=sorted(calibration),calibration_conditions=sum(r['phase']=='calibration' for r in records),
        calibration_scope='Guitar performance held out within existing training corpus. Bass recordings may overlap fit; upstream training overlap not excluded.',
        purpose='Compare exact-onset target against onset OR individually harmful frame deletion target. Never retime, repitch, or reassign roles.',
        thresholds=[0.,.05,.1,.2,.35,.5,.65,.8],selection='Calibration only; clean/noisy Note+Frame and aggregate offset+active exact must not decrease; maximize .5 Note+.3 Frame+.2 exact.',
        selection_checkpoint_step=800,development16_used_for_selection=False,regression14_used_for_selection=False,unopened_test_access=False,
        manifest_sha256=sha(DATA/'manifest.json'),source_manifest_sha256=sha(manifest),training_target_counts=dict(all_targets),
        dataset_counts={str(k):v for k,v in Counter((r['phase'],r['dataset']) for r in records).items()},
        implementation_sha256={p:sha(p) for p in ('band2tab/keep.py','scripts/a_keep_cycle.py')})
    save(DATA/'protocol.json',protocol)
    print('KEEP_DATA_READY',protocol['dataset_counts'],dict(all_targets),flush=True)

def load():
    protocol=json.load(open(DATA/'protocol.json'))
    if sha(DATA/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    for p,h in protocol['implementation_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen implementation changed: '+p)
    rows=json.load(open(DATA/'manifest.json'));cache={};meta={}
    for r in rows:
        p=Path(r['cache'])
        with np.load(p) as f:cache[r['id']]={k:f[k].copy() for k in f.files}
        meta[r['id']]=json.load(open(p.with_suffix('.json')))
    return protocol,rows,cache,meta

def train(device):
    protocol,rows,cache,_=load()
    fit=[r for r in rows if r['phase']=='train' and cache[r['id']]['mask'].any()]
    # Neither labels nor normalization from the internal calibration set.
    xx=np.concatenate([cache[r['id']]['x'][cache[r['id']]['mask']>0] for r in fit]);mean=xx.mean(0);std=xx.std(0).clip(.05)
    groups=defaultdict(list)
    for r in fit:groups[r['group']].append(r)
    band=sorted(g for g in groups if any(r['dataset'].startswith('stem_band_') for r in groups[g]));bass=sorted(set(groups)-set(band))
    z={r['id']:torch.tensor(np.clip((cache[r['id']]['x']-mean)/std,-8,8),device=device) for r in fit}
    targets={r['id']:{k:torch.tensor(cache[r['id']][k],device=device) for k in ('onset','joint','mask')} for r in fit}
    torch.set_num_threads(2);started=time.monotonic()
    for arm,(target,kind) in ARMS.items():
        for seed in SEEDS:
            folder=RUN/arm/f'seed_{seed}'
            if folder.exists():raise ValueError('Run exists: '+str(folder))
            torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
            model=KeepNet(kind).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1)
            config=dict(arm=arm,target=target,kind=kind,seed=seed,steps=800,parameters=sum(p.numel() for p in model.parameters()),device=device,
                training_records=len(fit),training_guitar_groups=len(band),protocol_sha256=sha(DATA/'protocol.json'))
            save(folder/'config.json',config);history=[]
            for step in range(1,801):
                xs=[];ys=[];masks=[]
                for _ in range(8):
                    candidates=band if rng.random()<.8 else bass;g=rng.choice(candidates);r=rng.choice(groups[g]);ident=r['id']
                    ix=np.random.randint(0,len(z[ident]),32);xs.append(z[ident][ix]);ys.append(targets[ident][target][ix]);masks.append(targets[ident]['mask'][ix])
                x=torch.cat(xs);y=torch.cat(ys);mask=torch.cat(masks);model.train();opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y,reduction='none',pos_weight=torch.tensor(3.,device=device))
                loss=(loss*mask).sum()/mask.sum().clamp_min(1);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if step%200==0:
                    history.append(dict(step=step,loss=float(loss.detach().cpu())));print('KEEP_TRAIN',arm,seed,step,history[-1]['loss'],flush=True)
            weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(dict(schema=SCHEMA,kind=kind,members=[weights],mean=mean.tolist(),std=std.tolist(),config=config),folder/'final.pt')
            save(folder/'completed.json',dict(config=config,history=history,checkpoint_sha256=sha(folder/'final.pt')))
    save(REPORT/'training_completed.json',dict(seconds=time.monotonic()-started,device=device,runs=9,updates_per_run=800,test_used=False))

def evaluate(rows,cache,meta,state,threshold):
    items=[];tracks=[]
    for r in rows:
        m=meta[r['id']];q=probabilities(state,cache[r['id']]['x'])
        parts=prune(m['pool'],q,threshold,m['unmodelled'],r['duration'],r['guard_seconds']);s=metrics(r,parts)
        items.append((r,s));tracks.append(dict(id=r['id'],dataset=r['dataset'],metrics=s,parts=parts,
            removed=sum(len(v) for v in m['baseline_parts'].values())-sum(len(v) for v in parts.values())))
    s=aggregate(items);s['keep_score']=.5*s['note_f1']+.3*s['frame_f1']+.2*s['active_exact_match']
    return dict(summary=s,tracks=tracks)

def select():
    if (REPORT/'selection.json').exists():raise ValueError('Selection already frozen')
    protocol,rows,cache,meta=load();rows=[r for r in rows if r['phase']=='calibration'];choices=[];torch.set_num_threads(2)
    for arm,(_,kind) in ARMS.items():
        members=[torch.load(RUN/arm/f'seed_{s}/final.pt',map_location='cpu',weights_only=True) for s in SEEDS]
        state={**members[0],'members':[v['members'][0] for v in members]}
        if any(v['mean']!=state['mean'] or v['std']!=state['std'] for v in members):raise ValueError('Ensemble normalization differs')
        baseline=evaluate(rows,cache,meta,state,0.);b=baseline['summary']
        for t in protocol['thresholds']:
            score=evaluate(rows,cache,meta,state,t);s=score['summary']
            eligible=all(s[k]>=b[k]-1e-10 for k in ('note_f1','frame_f1','active_exact_match','offset_f1')) and all(s['datasets'][d][k]>=v[k]-1e-10 for d,v in b['datasets'].items() for k in ('note_f1','frame_f1'))
            choices.append(dict(arm=arm,threshold=t,eligible=eligible,**score))
        save(REPORT/'calibration'/f'{arm}.json',dict(baseline=baseline,choices=[v for v in choices if v['arm']==arm]))
        print('KEEP_CALIBRATED',arm,max(x['summary']['keep_score'] for x in choices if x['arm']==arm),flush=True)
    allowed=[v for v in choices if v['eligible'] and v['threshold']>0 and v['summary']['keep_score']>b['keep_score']+1e-9]
    best=max(allowed,key=lambda v:v['summary']['keep_score']) if allowed else None
    if best:
        members=[torch.load(RUN/best['arm']/f'seed_{s}/final.pt',map_location='cpu',weights_only=True) for s in SEEDS]
        state={**members[0],'members':[v['members'][0] for v in members],'threshold':best['threshold'],
            'seeds':SEEDS,'selection_scope':'Internal held-out guitar performances only','automatic_application_approved':False}
        export=RUN/'calibration_selected.pt';torch.save(state,export)
    else:export=None
    save(REPORT/'selection.json',dict(baseline=baseline,best=best,choices=choices,identity_selected=best is None,
        export=str(export) if export else None,export_sha256=sha(export) if export else None,
        development16_used_for_selection=False,regression14_used_for_selection=False,test_used=False))
    print('KEEP_SELECTION_FROZEN',None if best is None else (best['arm'],best['threshold'],best['summary']),flush=True)

def review():
    from scripts.a_release_train import load as old_load
    selection=json.load(open(REPORT/'selection.json'));_,rows,cache,meta=old_load();rows=[r for r in rows if r['split']=='val']
    state=torch.load(selection['export'],map_location='cpu',weights_only=True) if selection['export'] else torch.load(RUN/'onset_linear/seed_17/final.pt',map_location='cpu',weights_only=True)
    threshold=selection['best']['threshold'] if selection['best'] else 0.
    baseline=evaluate(rows,cache,meta,state,0.);selected=evaluate(rows,cache,meta,state,threshold)
    # Frozen threshold, with independent member outputs for seed sensitivity.
    seeds=[]
    for i,s in enumerate(state.get('seeds',[17])):
        member={**state,'members':[state['members'][i]]};seeds.append(dict(seed=s,**evaluate(rows,cache,meta,member,threshold)))
    save(REPORT/'development.json',dict(baseline=baseline,selected=selected,seeds=seeds,
        selected_on_internal_calibration=True,development_used_for_retuning=False,unopened_test_access=False))
    print('KEEP_DEVELOPMENT_REVIEW',baseline['summary'],selected['summary'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','select','review']);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='train':train(a.device)
    elif a.command=='select':select()
    else:review()
