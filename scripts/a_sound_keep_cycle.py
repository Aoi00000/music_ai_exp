"""Additional feedback after the application fixture lost sounding frames.

Compare original fit data with additional Slakh full mixtures. Calibration now
also covers source-disjoint Slakh mixtures; exclude all old views of those UUIDs.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,random,hashlib,math
import numpy as np,soundfile as sf,torch
from band2tab.keep import SCHEMA,KeepNet,keep_targets,probabilities
from band2tab.stem_refiner import read_candidates,rms_envelope,feature_matrix
from band2tab.audio import features,PITCH_MIN,PITCH_MAX
from band2tab.bass_keep import SCHEMA as BUNDLE_SCHEMA,apply
from scripts.a_keep_cycle import load as old_load,SEEDS
from scripts.a_stress_data import local
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,KEYS
from scripts.a_error_taxonomy import classify

DATA=Path('data/sound_keep_v6');REPORT=Path('reports/sound_keep_v6');RUN=Path('runs/sound_keep_v6')

def sound_targets(pool,truth,baseline,guard,duration):
    target=keep_targets(pool,truth,baseline,guard,duration);sound=target['onset'].copy();frames=max(1,math.ceil((duration-2*guard)/.02))
    def interval(n):return max(0,math.ceil((n['onset']-guard)/.02-1e-9)),min(frames,math.ceil((n['offset']-guard)/.02-1e-9))
    for part in ('guitar','bass'):
        counts=np.zeros((frames,66),np.int32);rr=np.zeros_like(counts,dtype=bool)
        for n in baseline[part]:
            if PITCH_MIN<=n['pitch']<=PITCH_MAX:
                a,b=interval(n)
                if a<b:counts[a:b,n['pitch']-PITCH_MIN]+=1
        for n in truth[part]:
            if PITCH_MIN<=n['pitch']<=PITCH_MAX:
                a,b=interval(n)
                if a<b:rr[a:b,n['pitch']-PITCH_MIN]=True
        for i,n in enumerate(pool):
            if n['part']!=part:continue
            a,b=interval(n);k=n['pitch']-PITCH_MIN
            if ((counts[a:b,k]==1)&rr[a:b,k]).any():sound[i]=1.
    target['sound']=sound;return target

def prepare():
    if DATA.exists():raise ValueError('Preparation exists')
    _,old,cache,meta=old_load();root=Path('backups/azure').resolve()
    allrows=[r for r in json.load(open(root/'data/candidate_v2/manifest.json')) if r['split']=='train' and r['dataset']=='slakh_candidate_new']
    bad=set(json.load(open(root/'reports/candidate_v2/source_quality.json'))['excluded_groups']);allrows=[r for r in allrows if r['group'] not in bad]
    groups=sorted({r['group'] for r in allrows},key=lambda g:hashlib.sha256(('sound-keep-v6-'+g).encode()).hexdigest());held=set(groups[:4]);fit=set(groups[4:16])
    rows=[];dropped=[]
    for r in old:
        if r['phase']=='train' and r['group'] in held:dropped.append(r['id']);continue
        m=meta[r['id']];target=sound_targets(m['pool'],r['parts'],m['baseline_parts'],r['guard_seconds'],r['duration'])
        p=DATA/'cache'/(r['id']+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,x=cache[r['id']]['x'],**target);save(p.with_suffix('.json'),m);rows.append({**r,'cache':str(p),'extra':False})
    for group in groups[:16]:
        r=next(r for r in allrows if r['group']==group and r['id'].endswith('_1'));pool=[];outside={p:[] for p in ('guitar','bass')};baseline={};inputs={k:local(root,p) for k,p in r['inputs'].items()}
        for part in ('guitar','bass'):
            ns=read_candidates(root/'reports/candidate_v2/midi'/r['id']/(part+'.mid'),part,r['duration']);baseline[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns]
            pool += [n for n in ns if PITCH_MIN<=n['pitch']<=PITCH_MAX];outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
        views={};env={}
        for part,path in inputs.items():views[part]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);env[part]=rms_envelope(y,sr)
        x=feature_matrix(views,env,pool,r['duration']);target=sound_targets(pool,r['parts'],baseline,r['guard_seconds'],r['duration']);p=DATA/'cache'/(r['id']+'.npz');np.savez_compressed(p,x=x,**target)
        save(p.with_suffix('.json'),dict(pool=pool,unmodelled=outside,baseline_parts=baseline))
        rows.append({**r,'phase':'calibration' if group in held else 'train','cache':str(p),'extra':True,
            'source_fingerprint':{'audio':{k:sha(v) for k,v in inputs.items()},'midi':{k:sha(root/'reports/candidate_v2/midi'/r['id']/(k+'.mid')) for k in ('guitar','bass')}}})
        print('SOUND_KEEP_FULL_BAND_CACHED',r['id'],flush=True)
    if {r['group'] for r in rows if r['phase']=='train'}&{r['group'] for r in rows if r['phase']=='calibration'}:raise ValueError('Source leakage')
    save(DATA/'manifest.json',rows);save(DATA/'protocol.json',dict(hypothesis='An additional Slakh application fixture lost true sounding frames despite fewer onset FP. Add full-mixture domains and protect every unique true frame in teachers and calibration.',
        arms=['control','full_band'],seeds=SEEDS,steps=800,positive_keep_weight=3.,extra_draw_probability=.2,
        new_fullband_fit_groups=12,new_fullband_calibration_groups=4,old_views_removed_to_protect_calibration=dropped,
        source='Already acquired, prediction-blind-QC-passed Slakh training UUIDs only; none of11 reserved test UUIDs.',
        control='Original fit subset with all held-Slakh views removed. Same normalization, draws, initializer and updates; expanded arm substitutes20% of draws with full-mixture records.',
        teachers='Keep unique onset-matched notes OR any note contributing a unique correct20ms frame. More conservative than marginal-F1 teacher.',
        calibration='Old20 plus4 Slakh mixtures. Bass-only; no lost bass onset TP or true frames; four full/bass metrics per dataset nondecreasing; max10% removals.',
        normalization='Original fit subset only, identical in both arms',manifest_sha256=sha(DATA/'manifest.json'),counts={str(k):v for k,v in Counter((r['phase'],r['dataset']) for r in rows).items()},
        test_used=False,upstream_note='Cached Mu predictions and original sound-labelled data; differing cached generation regimes are part of the added-domain package, not an isolated waveform-only effect.',
        implementation_sha256={p:sha(p) for p in ('scripts/a_sound_keep_cycle.py','band2tab/keep.py','band2tab/bass_keep.py')}))
    print('SOUND_KEEP_PREPARED',len(rows),len(dropped),flush=True)

def load():
    protocol=json.load(open(DATA/'protocol.json'))
    if sha(DATA/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    for p,h in protocol['implementation_sha256'].items():
        if sha(p)!=h:raise ValueError('Implementation changed')
    rows=json.load(open(DATA/'manifest.json'));cc={};mm={}
    for r in rows:
        with np.load(r['cache']) as f:cc[r['id']]={k:f[k].copy() for k in f.files}
        mm[r['id']]=json.load(open(Path(r['cache']).with_suffix('.json')))
    return rows,cc,mm

def train():
    rows,cc,_=load();old=[r for r in rows if r['phase']=='train' and not r['extra'] and cc[r['id']]['mask'].any()];extra=[r for r in rows if r['phase']=='train' and r['extra'] and cc[r['id']]['mask'].any()]
    xx=np.concatenate([cc[r['id']]['x'][cc[r['id']]['mask']>0] for r in old]);mean=xx.mean(0);std=xx.std(0).clip(.05)
    z={r['id']:torch.tensor(np.clip((cc[r['id']]['x']-mean)/std,-8,8),device='cuda') for r in old+extra};y={r['id']:torch.tensor(cc[r['id']]['sound'],device='cuda') for r in old+extra};mask={r['id']:torch.tensor(cc[r['id']]['mask'],device='cuda') for r in old+extra}
    by=defaultdict(list)
    for r in old:by[r['group']].append(r)
    band=sorted(g for g in by if any(r['dataset'].startswith('stem_band_') for r in by[g]));bass=sorted(set(by)-set(band));torch.set_num_threads(2)
    for seed in SEEDS:
        hashes=[]
        for arm in ('control','full_band'):
            folder=RUN/arm/f'seed_{seed}'
            if folder.exists():raise ValueError('Run exists')
            torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed);extra_rng=random.Random(seed+60191);trace=hashlib.sha256();used=0
            model=KeepNet('mlp').cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);save(folder/'config.json',dict(arm=arm,seed=seed,steps=800,protocol_sha256=sha(DATA/'protocol.json')))
            for step in range(1,801):
                xs=[];ys=[];ms=[]
                for _ in range(8):
                    g=rng.choice(band if rng.random()<.8 else bass);r=rng.choice(by[g]);u=np.random.random(32);trace.update(r['id'].encode()+u.tobytes())
                    swap=extra_rng.random()<.2;other=extra_rng.choice(extra);chosen=other if arm=='full_band' and swap else r;used+=chosen is other;i=chosen['id'];ix=np.minimum(len(z[i])-1,(u*len(z[i])).astype(int))
                    xs.append(z[i][ix]);ys.append(y[i][ix]);ms.append(mask[i][ix])
                model.train();opt.zero_grad(set_to_none=True);m=torch.cat(ms);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.cat(xs)),torch.cat(ys),reduction='none',pos_weight=torch.tensor(3.,device='cuda'))
                loss=(loss*m).sum()/m.sum().clamp_min(1);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if step%200==0:print('SOUND_KEEP_TRAIN',arm,seed,step,float(loss.detach()),flush=True)
            state=dict(schema=SCHEMA,kind='mlp',members=[{k:v.detach().cpu() for k,v in model.state_dict().items()}],mean=mean.tolist(),std=std.tolist())
            torch.save(state,folder/'final.pt');hashes.append(trace.hexdigest());save(folder/'completed.json',dict(base_draw_sha256=trace.hexdigest(),fullband_draws=used,checkpoint_sha256=sha(folder/'final.pt')))
        if len(set(hashes))!=1:raise ValueError('Paired base draws differ')
    save(REPORT/'training_completed.json',dict(runs=6,steps=800,seeds=SEEDS,paired_draws_verified=True))

def select():
    rows,cc,mm=load();rows=[r for r in rows if r['phase']=='calibration'];torch.set_num_threads(2)
    for arm in ('control','full_band'):
        if (REPORT/arm/'selection.json').exists():raise ValueError('Selection exists')
        states=[torch.load(RUN/arm/f'seed_{s}/final.pt',map_location='cpu',weights_only=True) for s in SEEDS];learner={**states[0],'members':[x['members'][0] for x in states],'seeds':SEEDS};choices=[]
        def evaluate(threshold):
            bundle=dict(schema=BUNDLE_SCHEMA,learner=learner,threshold=threshold,max_remove_fraction=.1);full=[];bass=[];lost=frames=0
            for r in rows:
                m=mm[r['id']];parts=apply(bundle,cc[r['id']]['x'],m['pool'],m['unmodelled'],r['duration'],r['guard_seconds']);s=metrics(r,parts);b=metrics(r,m['baseline_parts']);full.append((r,s));frames+=max(0,b['frame_tp']-s['frame_tp'])
                old,_=classify(r,m['baseline_parts']);new,_=classify(r,parts);lost+=max(0,old.get('bass_tp',0)-new.get('bass_tp',0))
                br={**r,'parts':{'guitar':[],'bass':r['parts']['bass']}};bass.append((r,metrics(br,{'guitar':[],'bass':parts['bass']})))
            f=aggregate(full);b=aggregate(bass);return dict(full=f,bass=b,lost_true_onsets=lost,lost_true_frames=frames,bass_score=.5*b['note_f1']+.3*b['frame_f1']+.2*b['active_exact_match'])
        base=evaluate(0.)
        for t in (0.,.05,.1,.2,.35,.5,.65,.8):
            v=evaluate(t);eligible=not v['lost_true_onsets'] and not v['lost_true_frames'] and all(v[s]['datasets'][d][k]>=ref[k]-1e-10 for s in ('full','bass') for d,ref in base[s]['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'))
            v.update(threshold=t,eligible=eligible);choices.append(v)
        good=[v for v in choices if v['eligible'] and v['bass_score']>base['bass_score']+1e-9];best=max(good,key=lambda v:v['bass_score']) if good else None
        bundle=dict(schema=BUNDLE_SCHEMA,learner=learner,threshold=best['threshold'] if best else 0.,max_remove_fraction=.1,part='bass',mu_sha256='ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec',automatic_application_approved=False,inference_requires_yourmt3=False)
        export=RUN/arm/'selected_bundle.pt';torch.save(bundle,export);save(REPORT/arm/'selection.json',dict(best=best,baseline=base,choices=choices,export=str(export),export_sha256=sha(export),identity_selected=best is None))
        print('SOUND_KEEP_SELECTED',arm,bundle['threshold'],None if not best else best['bass'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','select']);a=p.parse_args();globals()[a.command]()
