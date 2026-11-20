"""Targeted real/synthetic contrast after57/58 octave FP already had correct roots."""
from pathlib import Path
from collections import Counter
import argparse,copy,json,random,hashlib,time
import numpy as np,soundfile as sf,torch
from band2tab.audio import features,SR,HOP,PITCH_MIN,N_BINS
from band2tab.octave_guard import SCHEMA,OctaveGuard,peers,patches,probabilities,prune,keep_labels
from band2tab.addition import _assign
from band2tab.separation import resample
from scripts.a_stress_data import local
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import metrics,aggregate
DATA=Path('data/octave_guard_v1');REPORT=Path('reports/octave_guard_v1');RUN=Path('runs/octave_guard_v1');BACK=Path('backups/azure')
SEEDS=(17,29,43);ARMS=('real_only','synth_clean','synth_drive')

def baseline_pool(base):return [{**n,'part':p} for p in ('guitar','bass') for n in base[p]]

def prepare_real():
    rows=copy.deepcopy(json.load(open('data/residual_v1/manifest.json')));out=[]
    for r in rows:
        # This cache contains the exact frozen source-known Mu notes, not GT.
        old=json.load(open(Path(r['residual_cache']).with_suffix('.json')));base=old['baseline_parts'];pool=baseline_pool(base);pair=peers(pool,r['duration'],r['guard_seconds']);dest=DATA/'real'/(r['id']+'.npz');meta=dest.with_suffix('.json')
        source=Path(r['inputs']['guitar']);source=source if source.is_file() else local(BACK,source)
        if not meta.exists():
            x,m=patches(features(source),pool,pair) if pair else (np.empty((0,24,12),np.float32),np.empty((0,4),np.float32))
            y,mask=keep_labels(pool,pair,r['parts'],r['duration'],r['guard_seconds'])
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,x=x,meta=m,y=y,mask=mask);save(meta,dict(pool=pool,pairs=pair,baseline_parts=base,guitar_audio_sha256=sha(source)))
        else:
            with np.load(dest) as z:old={k:z[k].copy() for k in z.files}
            y,mask=keep_labels(pool,pair,r['parts'],r['duration'],r['guard_seconds']);np.savez_compressed(dest,x=old['x'],meta=old['meta'],y=y,mask=mask)
        out.append({**r,'guard_cache':str(dest)})
        print('OCTAVE_GUARD_REAL',r['id'],len(pair),flush=True)
    save(DATA/'manifest.json',out)

def prepare_synthetic():
    import librosa
    real=json.load(open(DATA/'manifest.json'));fit={r['group'] for r in real if r['phase']=='train'};held={r['group'] for r in real if r['phase']=='calibration'}
    rows=[r for r in json.load(open(BACK/'data/goat_v2/manifest.json')) if r['split']=='train' and r['view']=='di' and r['group'] in fit]
    rows += [r for r in json.load(open(BACK/'data/candidate_v2/manifest.json')) if r['split']=='train' and r['dataset'] in ('guitar_techs_clean','guitarset')]
    donors=[];seen=set()
    for r in rows:
      for n in r['parts'].get('guitar',[]):
        a=n['onset']+.025;b=a+.3
        if n.get('censored_onset') or b>n['offset']-.01 or a<.05 or b>r['duration']-.05:continue
        if any(t['pitch']!=n['pitch'] and t['onset']<b and t['offset']>a for t in r['parts']['guitar']):continue
        key=(r['group'],n['pitch'],round(a+r.get('excerpt_start',0),3))
        if key in seen:continue
        seen.add(key);source=Path(r['audio']);source=source if source.is_file() else local(BACK,source)
        with sf.SoundFile(source) as f:
            sr=f.samplerate;f.seek(round(a*sr));y=f.read(round(.3*sr),dtype='float32',always_2d=True).mean(1)
        y=resample(y,sr,SR);y=y[:round(.3*SR)]
        if len(y)!=round(.3*SR) or not np.isfinite(y).all() or np.std(y)<1e-5:continue
        fade=min(110,len(y)//4);env=np.ones(len(y),np.float32);env[:fade]=np.linspace(0,1,fade);env[-fade:]=np.linspace(1,0,fade);y*=env;y-=y.mean();y/=max(1e-7,float(np.sqrt((y*y).mean())))
        donors.append(dict(id=r['id'],group=r['group'],pitch=n['pitch'],start=a,audio=y,source_sha256=sha(source)))
    by={p:[i for i,d in enumerate(donors) if d['pitch']==p] for p in range(23,89)};low=[i for i,d in enumerate(donors) if 36<=d['pitch']<=76 and by.get(d['pitch']+12)];rng=np.random.default_rng(3120);selected=rng.choice(low,min(128,len(low)),replace=False);records=[];xx=[];mm=[];yy=[];tone=[]
    for pair_id,lo in enumerate(selected):
        a=donors[int(lo)];options=[i for i in by[a['pitch']+12] if donors[i]['group']!=a['group']] or by[a['pitch']+12];b=donors[int(rng.choice(options))]
        if a['group'] in held or b['group'] in held:raise ValueError('Synthetic donor leaked calibration source')
        gain=float(rng.choice([-9.,-3.,3.]));drive=float(rng.choice([2.,4.,8.]));bias=float(rng.choice([.15,.4,.8]));offset=1.25;length=round(3.*SR);start=round(offset*SR)
        for colour in (0,1):
          for present in (0,1):
            body=.1*a['audio'].copy()
            if present:body+=.1*10**(gain/20)*b['audio']
            if colour:body=np.tanh(body*drive*5+bias)-np.tanh(bias);body-=body.mean()
            body*=.1/max(1e-7,float(np.sqrt((body*body).mean())));audio=np.zeros(length,np.float32);audio[start:start+len(body)]=body
            c=np.log1p(10*np.abs(librosa.cqt(audio,sr=SR,hop_length=HOP,fmin=librosa.midi_to_hz(PITCH_MIN),n_bins=N_BINS,bins_per_octave=36))).astype(np.float32).T
            pool=[dict(part='guitar',pitch=a['pitch'],onset=1.25,offset=1.55,velocity=100),dict(part='guitar',pitch=a['pitch']+12,onset=1.25,offset=1.55,velocity=100)];x,m=patches(c,pool,[(1,0)]);xx.append(x[0]);mm.append(m[0]);yy.append(present);tone.append(colour)
            records.append(dict(pair=pair_id,upper_present=bool(present),drive=bool(colour),lower_id=a['id'],upper_id=b['id'],lower_group=a['group'],upper_group=b['group'],lower_pitch=a['pitch'],upper_pitch=b['pitch'],lower_start=a['start'],upper_start=b['start'],upper_db=gain,drive_gain=drive,drive_bias=bias,lower_sha256=a['source_sha256'],upper_sha256=b['source_sha256'],audio_sha256=hashlib.sha256(audio.tobytes()).hexdigest()))
    np.savez_compressed(DATA/'synthetic.npz',x=np.array(xx),meta=np.array(mm),y=np.array(yy,np.float32),tone=np.array(tone));save(DATA/'synthetic.json',records);print('OCTAVE_GUARD_SYNTHETIC',len(donors),len(selected),len(records),flush=True)

def freeze():
    if (DATA/'protocol.json').exists():raise ValueError('Protocol already frozen')
    rows=json.load(open(DATA/'manifest.json'));cache={}
    for r in rows:
        with np.load(r['guard_cache']) as z:cache[r['id']]={k:z[k] for k in z.files}
    counts={phase:dict(records=sum(r['phase']==phase for r in rows),eligible=sum(len(cache[r['id']]['y']) for r in rows if r['phase']==phase),correct_or_sounding=sum(int((cache[r['id']]['y']*cache[r['id']]['mask']).sum()) for r in rows if r['phase']==phase)) for phase in ('train','calibration')}
    import shutil,inspect
    sources=['band2tab/octave_guard.py','scripts/a_octave_guard_cycle.py','scripts/a_candidate_train.py','band2tab/refine_duration.py','band2tab/temporal.py','band2tab/score.py']
    for file in sources:
        target=DATA/'frozen_sources'/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    save(DATA/'protocol.json',dict(created='2026-11-20',schema=SCHEMA,arms=ARMS,seeds=SEEDS,steps=1200,learning_rate=.0003,positive_keep_weight=3.,synthetic_sampling_probability=.5,
        reason='57/58 octave-apart guitar FP already had corresponding correct notes. Focus only on upper candidates and distinguish a distorted single from genuine octave pair.',
        synthetic='128 train-only donor pairs x single/double x clean/asymmetric drive. Same donor pair, pitch, onset, length and output RMS for both labels. Independently recorded actual octave pitches, no pitch shifting.',
        limitations='Reference-monophonic crops may contain tails/unlabelled noise; synthetic attack/fades are constructed, not genuine new performances. No new independent players. No Mu inference on synthetic notes.',
        normalization='Per-example90th-percentile harmonic patch, no calibration statistics',counts=counts,
        teachers='Keep exact-onset TP or any corresponding correct20ms frame on the same evaluation grid. Censored attacks masked.',
        selection='Internal24 only. Zero observed lost guitar onset TP and all4 per-dataset metrics nondecreasing. Cap deletions10% baseline guitar count per clip. Strict gain else explicit identity.',
        thresholds=[0.,.01,.02,.05,.1,.2,.35,.5,.65],manifest_sha256=sha(DATA/'manifest.json'),synthetic_sha256=sha(DATA/'synthetic.npz'),synthetic_metadata_sha256=sha(DATA/'synthetic.json'),workload_sha256=sha('scripts/a_octave_guard_gpu.py'),training_dtype='CUDA float32; deterministic CuDNN; no mixed precision',
        implementation_sha256={p:sha(p) for p in sources},patch_function_sha256=hashlib.sha256(inspect.getsource(patches).encode()).hexdigest(),
        cache_sha256={str(p):sha(p) for p in (DATA/'real').iterdir() if p.suffix in ('.npz','.json')},test_used=False,automatic_application_approved=False))
    print('OCTAVE_GUARD_FROZEN',counts,flush=True)

def load():
    rows=json.load(open(DATA/'manifest.json'));cache={};meta={}
    for r in rows:
        p=Path(r['guard_cache'])
        with np.load(p) as z:cache[r['id']]={k:z[k].copy() for k in z.files}
        meta[r['id']]=json.load(open(p.with_suffix('.json')))
    return rows,cache,meta

def train():
    p=json.load(open(DATA/'protocol.json'))
    if any(sha(f)!=h for f,h in p['implementation_sha256'].items()):raise ValueError('Frozen source changed')
    rows,c,_=load();fit=[r for r in rows if r['phase']=='train' and c[r['id']]['mask'].any()];z={i:{k:torch.as_tensor(v,device='cuda') for k,v in d.items()} for i,d in c.items()}
    with np.load(DATA/'synthetic.npz') as a:synth={k:torch.as_tensor(a[k],device='cuda') for k in ('x','meta','y')};tones=a['tone'].copy()
    clean=np.flatnonzero(tones==0);driven=np.flatnonzero(tones==1);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    for seed in SEEDS:
      hashes=[]
      for arm in ARMS:
        folder=RUN/arm/f'seed_{seed}'
        if folder.exists():raise ValueError('Do not overwrite run')
        torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed);model=OctaveGuard().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);trace=hashlib.sha256();count=0;started=time.monotonic()
        save(folder/'config.json',dict(arm=arm,seed=seed,parameters=sum(v.numel() for v in model.parameters()),protocol_sha256=sha(DATA/'protocol.json')))
        for step in range(1,1201):
            xs=[];ms=[];ys=[]
            for _ in range(8):
                r=rng.choice(fit);u=np.random.random(32);sw=rng.random()<.5;si=np.random.randint(0,len(clean),32);trace.update(r['id'].encode()+u.tobytes()+si.tobytes()+bytes([sw]))
                if arm!='real_only' and sw:
                    ix=(driven if arm=='synth_drive' else clean)[si];xs.append(synth['x'][ix]);ms.append(synth['meta'][ix]);ys.append(synth['y'][ix]);count+=1
                else:
                    d=z[r['id']];valid=torch.nonzero(d['mask']>.5).flatten();ix=valid[torch.as_tensor(np.minimum(len(valid)-1,(u*len(valid)).astype(int)),device='cuda')];xs.append(d['x'][ix]);ms.append(d['meta'][ix]);ys.append(d['y'][ix])
            model.train();opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.cat(xs),torch.cat(ms)),torch.cat(ys),pos_weight=torch.tensor(3.,device='cuda'))
            if not torch.isfinite(loss):raise ValueError('Nonfinite guard loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%300==0:print('OCTAVE_GUARD_TRAIN',arm,seed,step,float(loss.detach()),flush=True)
        state=dict(schema=SCHEMA,members=[{k:v.detach().cpu() for k,v in model.state_dict().items()}],arm=arm,seed=seed);torch.save(state,folder/'final.pt');hashes.append(trace.hexdigest());save(folder/'completed.json',dict(draw_sha256=trace.hexdigest(),synthetic_draws=count,steps=1200,seconds=time.monotonic()-started,checkpoint_sha256=sha(folder/'final.pt')))
      if len(set(hashes))!=1:raise ValueError('Paired draws differ')
    save(REPORT/'training_completed.json',dict(runs=9,steps=1200,seeds=SEEDS,paired_draws_verified=True));print('OCTAVE_GUARD_TRAINING_COMPLETE',flush=True)

def evaluate(rows,c,m,qs,t):
    items=[];tracks=[];lost=0
    for r in rows:
        meta=m[r['id']];parts=prune(meta['baseline_parts'],meta['pool'],meta['pairs'],qs[r['id']],t);s=metrics(r,parts);before=metrics(r,meta['baseline_parts']);lost+=max(0,before['note_tp']-s['note_tp']);items.append((r,s));tracks.append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=parts,metrics=s))
    return dict(summary=aggregate(items),tracks=tracks,lost_true_onsets=lost)

def select():
    if (REPORT/'selection.json').exists():raise ValueError('Selection frozen')
    p=json.load(open(DATA/'protocol.json'));rows,c,m=load();rows=[r for r in rows if r['phase']=='calibration'];torch.set_num_threads(2);base=evaluate(rows,c,m,{r['id']:np.ones(len(c[r['id']]['x'])) for r in rows},0.);b=base['summary'];base_score=.5*b['note_f1']+.3*b['frame_f1']+.2*b['active_exact_match'];arms=[]
    for arm in ARMS:
        states=[torch.load(RUN/arm/f'seed_{s}/final.pt',weights_only=True,map_location='cpu') for s in SEEDS];state={**states[0],'members':[a['members'][0] for a in states]};qs={r['id']:probabilities(state,c[r['id']]['x'],c[r['id']]['meta']) for r in rows};choices=[]
        for t in p['thresholds']:
            v=evaluate(rows,c,m,qs,t);s=v['summary'];ok=not v['lost_true_onsets'] and all(s['datasets'][d][k]>=ref[k]-1e-10 for d,ref in b['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'));choices.append(dict(threshold=t,eligible=ok,score=.5*s['note_f1']+.3*s['frame_f1']+.2*s['active_exact_match'],summary=s,lost_true_onsets=v['lost_true_onsets']))
        allowed=[a for a in choices if a['eligible'] and a['score']>base_score+1e-9];best=max(allowed,key=lambda a:a['score']) if allowed else None;state['threshold']=best['threshold'] if best else 0.;state['cap_fraction']=.1;export=RUN/arm/'selected_bundle.pt';torch.save(state,export);arms.append(dict(arm=arm,best=best,choices=choices,export=str(export),sha256=sha(export)));print('OCTAVE_GUARD_SELECTED',arm,best,flush=True)
    good=[a for a in arms if a['best']];chosen=max(good,key=lambda a:a['best']['score'])['arm'] if good else 'identity';save(REPORT/'selection.json',dict(baseline=b,arms=arms,chosen=chosen,test_used=False,automatic_application_approved=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['real','synthetic','freeze','train','select']);a=p.parse_args();{'real':prepare_real,'synthetic':prepare_synthetic,'freeze':freeze,'train':train,'select':select}[a.command]()
