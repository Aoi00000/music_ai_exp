"""Third feedback cycle: matched real amplifier views for the keep learner.

Only existing fit performances are expanded. Internal calibration and external
development examples receive no new training views. Controls share draws,
initialization, normalization, update count and candidate-sampling uniforms.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,copy,json,random,time,gc,os,hashlib
import numpy as np,soundfile as sf,torch
from band2tab.keep import SCHEMA,KeepNet,keep_targets,probabilities
from band2tab.separation import write_input,separate_inputs
from band2tab.stem_refiner import read_candidates,rms_envelope,feature_matrix
from band2tab.audio import features,PITCH_MIN,PITCH_MAX
from scripts.a_candidate_data import save,sha
from scripts.a_stress_data import read_stereo
from scripts.a_keep_cycle import load as load_original,SEEDS
from scripts.a_keep_policy_cycle import evaluate

DATA=Path('data/keep_amp_v3');REPORT=Path('reports/keep_amp_v3');RUN=Path('runs/keep_amp_v3')

def prepare():
    if DATA.exists():raise ValueError('Preparation already exists')
    _,rows,cache,meta=load_original();fit=[r for r in rows if r['phase']=='train'];by=defaultdict(list);ranks=defaultdict(lambda:[0,0])
    for r in fit:
        if not r['dataset'].startswith('stem_band_'):continue
        by[r['group']].append(r);m=meta[r['id']];z=cache[r['id']]
        for i,n in enumerate(m['pool']):
            if n['part']=='guitar' and z['mask'][i]:ranks[r['group']][int(z['onset'][i]>.5)]+=1
    order=sorted(by,key=lambda g:(-ranks[g][0]*min(1,ranks[g][1]/20),g))[:12]
    root=Path('backups/azure').resolve();goat={r['id']:r for r in json.load(open(root/'data/goat_v2/manifest.json'))};new=[];rejected=[]
    for group in order:
        for r in by[group]:
            g=goat[r['components']['guitar']];alt='amp5' if g['view']!='amp5' else 'amp1'
            item=group.removeprefix('goat_');folder=root/'data/raw/goat/GOAT/data'/item
            old_file=folder/(item+('' if g['view']=='di' else '_amp_'+g['view'][3:])+'.wav')
            new_file=folder/(item+'_amp_'+alt[3:]+'.wav')
            if abs(sf.info(old_file).duration-sf.info(new_file).duration)>.05:
                rejected.append(dict(id=r['id'],reason='Amplifier source duration differs by more than50ms'));continue
            start=g['excerpt_start']+(g['duration']-r['duration'])*.35
            oldg=read_stereo(old_file,start,r['duration']);newg=read_stereo(new_file,start,r['duration'])
            # Preserve this exact mixture: replace only its guitar component.
            current=read_stereo(Path(r['audio'].replace('/home/u6069685234/music_ai_exp',str(root))),0,r['duration'])
            old_peak=max(1e-7,float(np.abs(current).max()))
            # Rebuild from original components, avoiding unknown peak scaling.
            bass_id=r['components']['bass'];b=next(x for x in json.load(open(root/'data/yourmt3_lora_extra/manifest.json')) if x['id']==bass_id)
            title=b['id'].removeprefix('full_filobass_').rsplit('_',1)[0]
            ba=read_stereo(root/'data/raw/filobass/FiloBass ISMIR Publication/audio_bass_stems'/(title+'.mp3'),b['excerpt_start']+(b['duration']-r['duration'])*.35,r['duration'])
            rms=lambda y:max(1e-7,float(np.sqrt((y*y).mean())))
            levels=r['mixing'];clean=.1*newg/rms(newg)+.1*ba/rms(ba)*10**(levels['bass_to_guitar_rms_db']/20);y=clean
            if r['dataset'].endswith('noisy'):
                bg=next(x for x in json.load(open(root/'reports/band_stress_v1/acquisition.json'))['selected'] if x['group']==r['components']['background'])
                paths={Path(m).name:root/'data/raw/musdb_7s'/m for m in bg['members']}
                v=read_stereo(paths['vocals.wav'],0,r['duration']);d=read_stereo(paths['drums.wav'],0,r['duration'])
                y=clean+v/rms(v)*rms(clean)*10**(levels['vocals_to_target_rms_db']/20)+d/rms(d)*rms(clean)*10**(levels['drums_to_target_rms_db']/20)
            ident='ampview_'+r['id'];dest=DATA/'audio'/ident;dest.mkdir(parents=True,exist_ok=True);native=dest/'native.wav'
            y=y*min(1.,.95/max(1e-7,float(np.abs(y).max())));sf.write(native,y,44100,subtype='FLOAT');write_input(dest/'mix.wav',y,44100)
            new.append({**r,'id':ident,'base_id':r['id'],'audio':str(native),'inputs':{p:str(dest/(p+'.wav')) for p in ('mix','guitar','bass')},
                'audio_sha256':sha(native),'new_amplifier':alt,'old_amplifier':g['view'],'source_same_note_excerpt':True,
                'parts':r['parts'],'phase':'train','cache':str(DATA/'cache'/(ident+'.npz'))})
    if not new:raise ValueError('No valid alternate amplifier sources')
    if {r['group'] for r in new}&{r['group'] for r in rows if r['phase']=='calibration'}:raise ValueError('Calibration source leakage')
    save(DATA/'manifest.json',new)
    save(DATA/'protocol.json',dict(created='2026-11-19',hypothesis='Global pruning lost true guitars on a different performer. Expand real amplifier timbres of hard fit performances, preserving known note timing and bass/background.',
        fit_only=True,independent_new_performers=0,derived_conditions=len(new),source_performances=len({r['group'] for r in new}),
        group_selection='Training guitar false-onset count times min(1, true-onsets/20); top12. No development selection.',rejected=rejected,
        controls=['control','alternate_amp'],steps=800,seeds=SEEDS,target='joint',architecture='KeepNet mlp64',
        new_view_sampling_probability=.5,scope='Only when a sampled fit record has an alternate amp. Common base sampling80% band20% bass.',
        normalization='Original fit candidates only, identical in both arms',
        thresholds=[0.,.05,.1,.2,.35,.5,.65,.8],removal_cap_fraction=.1,
        calibration='Original internal20; no lost true onsets; four aggregate metrics preserved. Same selection for both arms.',
        inference_numeric='Original and extra training Mu: CUDA float16, source-known; downstream development remains Mac CPU float32.',
        manifest_sha256=sha(DATA/'manifest.json'),
        implementation_sha256={p:sha(p) for p in ('scripts/a_keep_amp_cycle.py','band2tab/keep.py','scripts/a_keep_policy_cycle.py')},
        test_used=False))
    print('KEEP_AMP_DATA_PREPARED',len(new),len(order),flush=True)

def frozen():
    protocol=json.load(open(DATA/'protocol.json'))
    if sha(DATA/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    for p,h in protocol['implementation_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen implementation changed: '+p)
    return protocol,json.load(open(DATA/'manifest.json'))

def infer():
    _,rows=frozen();torch.set_num_threads(1);os.environ['TORCH_HOME']=str(Path('models/torch').resolve())
    from demucs.pretrained import get_model
    separator=get_model('htdemucs_6s').to('cuda').eval()
    for r in rows:
        dest=Path(r['audio']).parent
        torch.manual_seed(int(r['audio_sha256'][:8],16)%2**31);random.seed(int(r['audio_sha256'][:8],16)%2**31)
        result=separate_inputs(r['audio'],dest,separator)
        save(dest/'separation.json',result);print('KEEP_AMP_SEPARATED',r['id'],flush=True)
    del separator;gc.collect();torch.cuda.empty_cache()
    from muscriptor import TranscriptionModel
    weight=Path('models/muscriptor-medium/model.safetensors');model=TranscriptionModel.load_model(weight,device='cuda',dtype='float16')
    for r in rows:
        dest=REPORT/'midi'/r['id'];dest.mkdir(parents=True,exist_ok=True)
        for part in ('guitar','bass'):
            midi,_=model.transcribe_and_postprocess(r['inputs'][part],detect_tempo=False);out=dest/(part+'.mid');out.write_bytes(midi)
            save(out.with_suffix('.json'),dict(audio_sha256=sha(r['inputs'][part]),model_sha256=sha(weight),dtype='float16'))
        print('KEEP_AMP_MU',r['id'],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()

def cache():
    _,rows=frozen()
    for r in rows:
        views={};env={};pool=[];baseline={};outside={p:[] for p in ('guitar','bass')}
        for part in ('mix','guitar','bass'):
            views[part]=features(r['inputs'][part]);y,sr=sf.read(r['inputs'][part],dtype='float32',always_2d=True);env[part]=rms_envelope(y,sr)
        for part in ('guitar','bass'):
            ns=read_candidates(REPORT/'midi'/r['id']/(part+'.mid'),part,r['duration']);baseline[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns]
            pool += [n for n in ns if PITCH_MIN<=n['pitch']<=PITCH_MAX]
            outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
        x=feature_matrix(views,env,pool,r['duration']);target=keep_targets(pool,r['parts'],baseline,r['guard_seconds'],r['duration']);dest=Path(r['cache']);dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,x=x,**target);save(dest.with_suffix('.json'),dict(pool=pool,baseline_parts=baseline,unmodelled=outside))
        print('KEEP_AMP_CACHED',r['id'],len(pool),flush=True)

def train():
    protocol,extra=frozen();_,rows,old,_=load_original();fit=[r for r in rows if r['phase']=='train' and old[r['id']]['mask'].any()];extra_by={r['base_id']:r for r in extra};arrays=dict(old)
    for r in extra:
        with np.load(r['cache']) as f:arrays[r['id']]={k:f[k].copy() for k in f.files}
    xx=np.concatenate([old[r['id']]['x'][old[r['id']]['mask']>0] for r in fit]);mean=xx.mean(0);std=xx.std(0).clip(.05)
    grouped=defaultdict(list)
    for r in fit:grouped[r['group']].append(r)
    band=sorted(g for g in grouped if any(r['dataset'].startswith('stem_band_') for r in grouped[g]));bass=sorted(set(grouped)-set(band))
    ids=[r['id'] for r in fit+extra];z={i:torch.tensor(np.clip((arrays[i]['x']-mean)/std,-8,8),device='cuda') for i in ids}
    y={i:torch.tensor(arrays[i]['joint'],device='cuda') for i in ids};m={i:torch.tensor(arrays[i]['mask'],device='cuda') for i in ids};torch.set_num_threads(2)
    for seed in SEEDS:
        digests=[]
        for arm in ('control','alternate_amp'):
            folder=RUN/arm/f'seed_{seed}'
            if folder.exists():raise ValueError('Run already exists')
            torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed);aug_rng=random.Random(seed+91827);trace=hashlib.sha256();aug_count=0
            model=KeepNet('mlp').cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1)
            save(folder/'config.json',dict(seed=seed,arm=arm,steps=800,protocol_sha256=sha(DATA/'protocol.json'),parameters=sum(p.numel() for p in model.parameters())))
            for step in range(1,801):
                bx=[];by=[];bm=[]
                for _ in range(8):
                    choices=band if rng.random()<.8 else bass;g=rng.choice(choices);r=rng.choice(grouped[g]);use_aug=aug_rng.random()<.5
                    # Identical base IDs and uniform indices in each pair; the new view alone differs.
                    u=np.random.random(32);trace.update(r['id'].encode()+u.tobytes());chosen=extra_by[r['id']] if arm=='alternate_amp' and r['id'] in extra_by and use_aug else r
                    ident=chosen['id'];aug_count+=ident!=r['id']
                    if not len(z[ident]):ident=r['id']
                    ix=np.minimum(len(z[ident])-1,(u*len(z[ident])).astype(int));bx.append(z[ident][ix]);by.append(y[ident][ix]);bm.append(m[ident][ix])
                model.train();opt.zero_grad(set_to_none=True);mask=torch.cat(bm);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.cat(bx)),torch.cat(by),reduction='none',pos_weight=torch.tensor(3.,device='cuda'))
                loss=(loss*mask).sum()/mask.sum().clamp_min(1);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if step%200==0:print('KEEP_AMP_TRAIN',arm,seed,step,float(loss.detach()),flush=True)
            state=dict(schema=SCHEMA,kind='mlp',members=[{k:v.detach().cpu() for k,v in model.state_dict().items()}],mean=mean.tolist(),std=std.tolist())
            torch.save(state,folder/'final.pt');digests.append(trace.hexdigest())
            save(folder/'completed.json',dict(seed=seed,arm=arm,steps=800,base_draw_sha256=trace.hexdigest(),alternate_amp_batch_draws=aug_count,checkpoint_sha256=sha(folder/'final.pt')))
        if len(set(digests))!=1:raise ValueError('Paired base samples diverged')
    save(REPORT/'training_completed.json',dict(runs=6,steps=800,seeds=SEEDS,paired_base_draws_verified=True,extra_conditions=len(extra)))

def select():
    frozen();_,rows,arrays,meta=load_original();rows=[r for r in rows if r['phase']=='calibration'];torch.set_num_threads(2)
    for arm in ('control','alternate_amp'):
        if (REPORT/arm/'selection.json').exists():raise ValueError('Selection frozen')
        states=[torch.load(RUN/arm/f'seed_{s}/final.pt',map_location='cpu',weights_only=True) for s in SEEDS]
        state={**states[0],'members':[s['members'][0] for s in states],'seeds':SEEDS}
        qs={r['id']:probabilities(state,arrays[r['id']]['x']) for r in rows};base=evaluate(rows,arrays,meta,qs,(0,0));choices=[];b=base['summary']
        for thresholds in __import__('itertools').product((0.,.05,.1,.2,.35,.5,.65,.8),repeat=2):
            v=evaluate(rows,arrays,meta,qs,thresholds);s=v['summary'];eligible=not any(v['lost_true_onsets'].values()) and all(s[k]>=b[k]-1e-10 for k in ('note_f1','frame_f1','offset_f1','active_exact_match'))
            v.update(thresholds=thresholds,eligible=eligible);choices.append(v)
        allowed=[v for v in choices if v['eligible'] and v['summary']['keep_score']>b['keep_score']+1e-9];best=max(allowed,key=lambda v:v['summary']['keep_score']) if allowed else None
        state.update(thresholds=best['thresholds'] if best else (0.,0.),removal_cap_fraction=.1,automatic_application_approved=False)
        export=RUN/arm/'selected_ensemble.pt';torch.save(state,export)
        save(REPORT/arm/'selection.json',dict(best=best,baseline=base,export=str(export),export_sha256=sha(export),identity_selected=best is None,choices=[{k:v for k,v in x.items() if k!='tracks'} for x in choices]))
        print('KEEP_AMP_SELECTED',arm,state['thresholds'],None if best is None else best['summary'],flush=True)

def audit():
    paths=[p for folder in (DATA,REPORT,RUN) for p in folder.rglob('*') if p.is_file() and p.name!='audit.json']
    save(REPORT/'audit.json',dict(files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]))
    print('KEEP_AMP_AUDIT_READY',len(paths),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','infer','cache','train','select','audit']);a=p.parse_args()
    globals()[a.command]()
