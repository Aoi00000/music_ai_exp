"""Fourth feedback hypothesis, separately controlled against zeroed evidence."""
from pathlib import Path
from collections import defaultdict
import argparse,json,random,hashlib,itertools
import numpy as np,torch
from band2tab.keep_agreement import SCHEMA,AgreementKeepNet,agreement_features,probabilities
from band2tab.stem_refiner import read_candidates
from scripts.a_candidate_data import sha,save
from scripts.a_keep_cycle import load as original,SEEDS
from scripts.a_keep_policy_cycle import evaluate

DATA=Path('data/keep_agreement_v4');REPORT=Path('reports/keep_agreement_v4');RUN=Path('runs/keep_agreement_v4')

def prepare():
    if DATA.exists():raise ValueError('Preparation exists')
    _,rows,arrays,meta=original();records=[]
    for r in rows:
        m=meta[r['id']];path=Path('reports/addition_v1/midi')/(r['id']+'.mid')
        support=read_candidates(path,'guitar',r['duration']) if path.exists() else []
        if r['dataset'].startswith('stem_band_') and not path.exists():raise ValueError('Missing fixed Your proposals')
        a=arrays[r['id']];x=np.concatenate([a['x'],agreement_features(m['pool'],support)],axis=1)
        dest=DATA/'cache'/(r['id']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**{**a,'x':x})
        save(dest.with_suffix('.json'),m);records.append({**r,'cache':str(dest),'support_sha256':sha(path) if path.exists() else None})
    save(DATA/'manifest.json',records)
    save(DATA/'protocol.json',dict(hypothesis='CQT keep-only removed true guitars. Existing frozen YourMT3 LoRA agrees with many removed true notes. Learn this complementary evidence instead of treating CQT alone as sufficient.',
        control='Same358-input network with final8 normalized features set to zero; same initialization, draws, labels and800 updates.',
        expanded='Eight inference-only agreement features: onset50/120ms, onset distance, overlap, held note, octave peer, neighbor count, available flag.',
        seeds=SEEDS,steps=800,arms=['zero_evidence','agreement'],guard_variants=[False,True],
        guard='Optional protection of Mu guitar notes corroborated by Your at onset50ms. Select on internal20 only; it can preserve correlated errors.',
        calibration='Zero calibration true-onset loss and nondecreasing Note/Frame/offset/exact; max10% removals perpart perclip.',
        original_fit_and_calibration_split=True,normalization_training_only=True,proposal_adapter_sha256=sha('models/yourmt3_lora_augmentation_review.pt'),
        upstream_overlap='Your LoRA already saw original training performances including internal calibration; split is only held out for this keep head, not the full system.',
        manifest_sha256=sha(DATA/'manifest.json'),test_used=False,
        implementation_sha256={p:sha(p) for p in ('scripts/a_keep_agreement_cycle.py','band2tab/keep_agreement.py','scripts/a_keep_policy_cycle.py')}))
    print('KEEP_AGREEMENT_PREPARED',len(records),flush=True)

def load():
    protocol=json.load(open(DATA/'protocol.json'))
    if sha(DATA/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    for p,h in protocol['implementation_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen implementation changed')
    rows=json.load(open(DATA/'manifest.json'));arrays={};meta={}
    for r in rows:
        with np.load(r['cache']) as f:arrays[r['id']]={k:f[k].copy() for k in f.files}
        meta[r['id']]=json.load(open(Path(r['cache']).with_suffix('.json')))
    return rows,arrays,meta

def train():
    rows,arrays,_=load();fit=[r for r in rows if r['phase']=='train' and arrays[r['id']]['mask'].any()]
    x=np.concatenate([arrays[r['id']]['x'][arrays[r['id']]['mask']>0] for r in fit]);mean=x.mean(0);std=x.std(0).clip(.05)
    z={r['id']:torch.tensor(np.clip((arrays[r['id']]['x']-mean)/std,-8,8),device='cuda') for r in fit}
    y={r['id']:torch.tensor(arrays[r['id']]['joint'],device='cuda') for r in fit};mask={r['id']:torch.tensor(arrays[r['id']]['mask'],device='cuda') for r in fit}
    by=defaultdict(list)
    for r in fit:by[r['group']].append(r)
    band=sorted(g for g in by if any(r['dataset'].startswith('stem_band_') for r in by[g]));bass=sorted(set(by)-set(band));torch.set_num_threads(2)
    for seed in SEEDS:
        traces=[]
        for arm in ('zero_evidence','agreement'):
            folder=RUN/arm/f'seed_{seed}'
            if folder.exists():raise ValueError('Run exists')
            torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed);trace=hashlib.sha256();model=AgreementKeepNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1)
            save(folder/'config.json',dict(seed=seed,arm=arm,steps=800,parameters=sum(p.numel() for p in model.parameters()),protocol_sha256=sha(DATA/'protocol.json')))
            for step in range(1,801):
                xs=[];ys=[];ms=[]
                for _ in range(8):
                    groups=band if rng.random()<.8 else bass;g=rng.choice(groups);r=rng.choice(by[g]);i=r['id'];ix=np.random.randint(0,len(z[i]),32);trace.update(i.encode()+ix.tobytes());xx=z[i][ix].clone()
                    if arm=='zero_evidence':xx[:,350:]=0
                    xs.append(xx);ys.append(y[i][ix]);ms.append(mask[i][ix])
                model.train();opt.zero_grad(set_to_none=True);m=torch.cat(ms);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.cat(xs)),torch.cat(ys),reduction='none',pos_weight=torch.tensor(3.,device='cuda'))
                loss=(loss*m).sum()/m.sum().clamp_min(1);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if step%200==0:print('KEEP_AGREEMENT_TRAIN',arm,seed,step,float(loss.detach()),flush=True)
            state=dict(schema=SCHEMA,use_agreement=arm=='agreement',agreement_guard=False,members=[{k:v.detach().cpu() for k,v in model.state_dict().items()}],mean=mean.tolist(),std=std.tolist())
            torch.save(state,folder/'final.pt');traces.append(trace.hexdigest());save(folder/'completed.json',dict(base_draw_sha256=trace.hexdigest(),checkpoint_sha256=sha(folder/'final.pt')))
        if len(set(traces))!=1:raise ValueError('Paired samples differ')
    save(REPORT/'training_completed.json',dict(runs=6,steps=800,seeds=SEEDS,paired_draws_verified=True))

def select():
    rows,arrays,meta=load();rows=[r for r in rows if r['phase']=='calibration'];torch.set_num_threads(2)
    for arm in ('zero_evidence','agreement'):
        if (REPORT/arm/'selection.json').exists():raise ValueError('Selection exists')
        states=[torch.load(RUN/arm/f'seed_{s}/final.pt',map_location='cpu',weights_only=True) for s in SEEDS];base_state={**states[0],'members':[x['members'][0] for x in states],'seeds':SEEDS};choices=[]
        for guard in ([False,True] if arm=='agreement' else [False]):
            state={**base_state,'agreement_guard':guard};qs={r['id']:probabilities(state,arrays[r['id']]['x']) for r in rows};base=evaluate(rows,arrays,meta,qs,(0,0));b=base['summary']
            for thresholds in itertools.product((0.,.05,.1,.2,.35,.5,.65,.8),repeat=2):
                v=evaluate(rows,arrays,meta,qs,thresholds);s=v['summary'];eligible=not any(v['lost_true_onsets'].values()) and all(s[k]>=b[k]-1e-10 for k in ('note_f1','frame_f1','offset_f1','active_exact_match'))
                v.update(thresholds=thresholds,guard=guard,eligible=eligible);choices.append(v)
        allowed=[v for v in choices if v['eligible'] and v['summary']['keep_score']>b['keep_score']+1e-9];best=max(allowed,key=lambda v:v['summary']['keep_score']) if allowed else None
        base_state.update(thresholds=best['thresholds'] if best else (0.,0.),agreement_guard=best['guard'] if best else False,removal_cap_fraction=.1,automatic_application_approved=False)
        export=RUN/arm/'selected_ensemble.pt';torch.save(base_state,export)
        save(REPORT/arm/'selection.json',dict(best=best,baseline=base,identity_selected=best is None,export=str(export),export_sha256=sha(export),choices=[{k:v for k,v in x.items() if k!='tracks'} for x in choices]))
        print('KEEP_AGREEMENT_SELECTED',arm,base_state['thresholds'],base_state['agreement_guard'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','select']);a=p.parse_args();globals()[a.command]()
