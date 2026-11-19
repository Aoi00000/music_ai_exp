"""Fixed model comparison and source-block uncertainty; never open final tests."""
from pathlib import Path
from collections import defaultdict
import argparse,json
import numpy as np,soundfile as sf,torch
from band2tab.audio import features,PITCH_MIN,PITCH_MAX
from band2tab.stem_refiner import read_candidates,rms_envelope,feature_matrix
from band2tab.keep import probabilities as keep_prob,prune
from band2tab.keep_agreement import probabilities as agree_prob,agreement_features
from scripts.a_keep_policy_cycle import apply
from scripts.a_candidate_train import metrics,aggregate,KEYS
from scripts.a_candidate_data import sha,save

DATA=Path('data/keep_feedback_regression_v1');REPORT=Path('reports/keep_feedback_20261119')

def cache_regression():
    if (DATA/'manifest.json').exists():raise ValueError('Regression cache is frozen')
    manifest=Path('data/stem_refiner_holdout_v1/manifest.json');rows=json.load(open(manifest))
    previous=json.load(open('reports/octave_regression_v1/comparison.json'))['tracks']['known_regression']
    if {r['id'] for r in rows}!={r['id'] for r in previous}:raise ValueError('Only previously inspected fourteen conditions allowed')
    records=[]
    for r in rows:
        pool=[];outside={p:[] for p in ('guitar','bass')};baseline={}
        for part in ('guitar','bass'):
            ns=read_candidates(Path('reports/stem_refiner_holdout_v1/midi')/r['id']/(part+'.mid'),part,r['duration'])
            baseline[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns];pool += [n for n in ns if PITCH_MIN<=n['pitch']<=PITCH_MAX]
            outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
        views={};env={}
        for part,path in r['inputs'].items():
            views[part]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);env[part]=rms_envelope(y,sr)
        x=feature_matrix(views,env,pool,r['duration']);support=read_candidates(Path('reports/addition_regression_v1/midi')/(r['id']+'.mid'),'guitar',r['duration'])
        ax=np.concatenate([x,agreement_features(pool,support)],axis=1);p=DATA/'cache'/(r['id']+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,x=x,agreement=ax)
        save(p.with_suffix('.json'),dict(pool=pool,unmodelled=outside,baseline_parts=baseline));records.append({**r,'cache':str(p)})
        print('KEEP_REGRESSION_FEATURES',r['id'],flush=True)
    save(DATA/'manifest.json',records);save(DATA/'protocol.json',dict(scope='Previously inspected14; cache features only. No new test access or model selection.',source_manifest_sha256=sha(manifest)))

def bootstrap(rows,baseline,after,seed=216,replicates=10000):
    by=defaultdict(list)
    for r,b,a in zip(rows,baseline,after):by[r['group']].append([a['metrics'][k]-b['metrics'][k] for k in KEYS])
    values=np.array([np.mean(v,axis=0) for v in by.values()]);rng=np.random.default_rng(seed);indices=rng.integers(0,len(values),(replicates,len(values)));draw=values[indices].mean(1)
    return dict(groups=len(values),replicates=replicates,seed=seed,
        method='Paired guitar-performance block bootstrap, clean/noisy kept together. Descriptive on reused development/regression; performers/compositions not fully independent; no selection correction.',
        differences={k:dict(mean=float(values[:,i].mean()),ci025=float(np.quantile(draw[:,i],.025)),ci975=float(np.quantile(draw[:,i],.975))) for i,k in enumerate(KEYS)})

def review():
    if (REPORT/'comparison.json').exists():raise ValueError('Comparison already frozen')
    from scripts.a_release_train import load
    _,allrows,cache,meta=load();development=[r for r in allrows if r['split']=='val']
    rrows=json.load(open(DATA/'manifest.json'));rc={};rm={}
    for r in rrows:
        with np.load(r['cache']) as f:rc[r['id']]={k:f[k].copy() for k in f.files}
        rm[r['id']]=json.load(open(Path(r['cache']).with_suffix('.json')))
    variants={}
    v1=json.load(open('reports/keep_cycle_v1/selection.json'));v2=json.load(open('reports/keep_policy_v2/selection.json'))
    variants['global_keep_v1']=dict(state=torch.load(v1['export'],map_location='cpu',weights_only=True),global_threshold=v1['best']['threshold'])
    variants['risk_policy_v2']=dict(state=torch.load(v2['source_checkpoint'],map_location='cpu',weights_only=True),thresholds=v2['best']['thresholds'] if v2['best'] else (0,0))
    for folder in ('keep_amp_v3','keep_agreement_v4'):
        arms=('control','alternate_amp') if folder=='keep_amp_v3' else ('zero_evidence','agreement')
        for arm in arms:
            s=json.load(open(Path('reports')/folder/arm/'selection.json'));state=torch.load(s['export'],map_location='cpu',weights_only=True)
            variants[folder+'_'+arm]=dict(state=state,thresholds=state['thresholds'],selection_sha256=sha(Path('reports')/folder/arm/'selection.json'))
    # Prespecified component ablations motivated by v1's guitar loss/bass gain.
    # Keep calibrated thresholds; do not tune them on the review sets.
    for name in ('risk_policy_v2','keep_amp_v3_alternate_amp','keep_agreement_v4_agreement'):
        variants[name+'_bass_only']={**variants[name],'bass_only':True}
    save(REPORT/'review_protocol.json',dict(variants={k:{q:v for q,v in d.items() if q!='state'} for k,d in variants.items()},
        all_models_and_thresholds_frozen_before_review=True,scopes=['Repeated development16','Previously inspected regression14'],test_used=False))
    torch.set_num_threads(2);summaries={};tracks={};intervals={}
    for scope,rows,cc,mm in [('development',development,cache,meta),('known_regression',rrows,rc,rm)]:
        results={'baseline':[]};items={'baseline':[]}
        for r in rows:
            m=mm[r['id']];s=metrics(r,m['baseline_parts']);results['baseline'].append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=m['baseline_parts'],metrics=s,removed=0));items['baseline'].append((r,s))
        for name,method in variants.items():
            state=method['state'];values=[];scores=[]
            for r in rows:
                m=mm[r['id']];x=cc[r['id']]['x']
                if name.startswith('keep_agreement'):
                    if scope=='development':
                        support=read_candidates(Path('reports/addition_v1/midi')/(r['id']+'.mid'),'guitar',r['duration']);ax=np.concatenate([x,agreement_features(m['pool'],support)],axis=1)
                    else:ax=cc[r['id']]['agreement']
                    q=agree_prob(state,ax)
                else:q=keep_prob(state,x)
                if method.get('bass_only'):
                    q=np.where([n['part']=='guitar' for n in m['pool']],1.,q)
                if 'global_threshold' in method:parts=prune(m['pool'],q,method['global_threshold'],m['unmodelled'],r['duration'],r['guard_seconds'])
                else:parts=apply(m['pool'],q,method['thresholds'],m['unmodelled'],r['duration'],r['guard_seconds'])
                s=metrics(r,parts);scores.append((r,s));values.append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=parts,metrics=s,
                    removed=sum(len(v) for v in m['baseline_parts'].values())-sum(len(v) for v in parts.values())))
            results[name]=values;items[name]=scores;print('KEEP_FEEDBACK_MEASURED',scope,name,{k:round(v,6) for k,v in aggregate(scores).items() if k in KEYS},flush=True)
        summaries[scope]={k:aggregate(v) for k,v in items.items()};tracks[scope]=results
        intervals[scope]={k:bootstrap(rows,results['baseline'],v) for k,v in results.items() if k!='baseline'}
    save(REPORT/'comparison.json',dict(summary=summaries,tracks=tracks,bootstrap=intervals,test_used=False,
        automatic_application_approved=False,scope='Exploratory feedback on previously-used development and regression. No fresh-test superiority claim.'))
    print('KEEP_FEEDBACK_REVIEW_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['cache','review']);a=p.parse_args()
    if a.command=='cache':cache_regression()
    else:review()
