"""Frozen residual policies on reused16/reused14/application fixture, no test."""
from pathlib import Path
from collections import Counter
import copy,json,time
import numpy as np,soundfile as sf,torch
from band2tab.audio import features
from band2tab.stem_refiner import rms_envelope,read_candidates
from band2tab.residual import candidate_pool,residual_features,teacher,apply,probabilities
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import metrics,aggregate
from scripts.a_stress_data import local
from scripts.a_keep_feedback_review import bootstrap
DATA=Path('data/residual_review_v1');REPORT=Path('reports/residual_fixed_review_v1');BACK=Path('backups/azure')

def cache(r,base,support):
    path=DATA/'cache'/(r['id']+'.npz');pool=candidate_pool(base,support,r['duration'],r['guard_seconds']);inputs={p:(Path(v) if Path(v).is_file() else local(BACK,v)) for p,v in r['inputs'].items() if p in ('mix','guitar','bass')}
    if not path.exists():
        views={p:features(v) for p,v in inputs.items()};env={}
        for p,v in inputs.items():y,sr=sf.read(v,dtype='float32',always_2d=True);env[p]=rms_envelope(y,sr)
        x=residual_features(views,env,pool,base,r['duration']);target=teacher(pool,base,r['parts'],r['duration'],r['guard_seconds']);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,x=x,**target)
        save(path.with_suffix('.json'),dict(pool=pool,baseline_parts=base,audio_sha256={p:sha(v) for p,v in inputs.items()},implementation_sha256=sha('band2tab/residual.py')))
    with np.load(path) as z:cc={k:z[k].copy() for k in z.files}
    mm=json.load(open(path.with_suffix('.json')))
    if mm['pool']!=pool or mm['baseline_parts']!=base:raise ValueError('Stale review candidates')
    return cc,mm

def main():
    if (REPORT/'comparison.json').exists():raise ValueError('Review frozen')
    selected=json.load(open('reports/residual_identity_v1/selection.json'));chosen=max(selected['arms'],key=lambda x:x['best']['score'])
    save(REPORT/'protocol.json',dict(selection_sha256=sha('reports/residual_identity_v1/selection.json'),chosen_before_review=chosen['arm'],
      candidate_sets_and_all_thresholds_frozen=True,scope='Repeated development16 and known regression14; application fixture is already used training source, not an independent test.',
      composition='Also compose chosen addition policy with previous C7. Never choose thresholds from these outcomes.',test_used=False))
    states={a['arm']:torch.load(a['export'],map_location='cpu',weights_only=True) for a in selected['arms']};torch.set_num_threads(2)
    old=json.load(open('reports/sound_keep_conservative_v7/comparison.json'))['tracks'];all_s={};all_t={};all_diag={};ci={}
    for scope,rows in [('development',[r for r in json.load(open(BACK/'data/stem_refiner_v1/manifest.json')) if r['split']=='val']),('known_regression',json.load(open('data/stem_refiner_holdout_v1/manifest.json')))]:
        items={k:[] for k in ['baseline','unfiltered_union','candidate_oracle_diagnostic',*states,'prior_c7','chosen_plus_c7']};tracks={k:[] for k in items};diag=Counter()
        previous={r['id']:r for r in old[scope]['combined']}
        for r in rows:
            if scope=='development':base=json.load(open(BACK/'data/stem_refiner_v1/cache'/(r['id']+'.json')))['baseline_parts'];support_path=Path('reports/addition_v1/midi')/(r['id']+'.mid')
            else:base=json.load(open('data/keep_feedback_regression_v1/cache/'+r['id']+'.json'))['baseline_parts'];support_path=Path('reports/addition_regression_v1/midi')/(r['id']+'.mid')
            support={'guitar':read_candidates(support_path,'guitar',r['duration']),'bass':[]};c,m=cache(r,base,support);pool=m['pool'];diag['candidates']+=len(pool);diag['strict_missing_matches']+=int(c['y'].sum());diag['positive_and_frame_nonnegative']+=int((c['y']*c['sound']).sum())
            variants={'baseline':base,'unfiltered_union':apply(base,pool,np.ones(len(pool)),(0.,0.)),
              'candidate_oracle_diagnostic':apply(base,pool,c['y'],(.5,.5)),'prior_c7':previous[r['id']]['parts']}
            for arm,bundle in states.items():
                q=probabilities(bundle,c['x'])
                if not bundle['expanded']:q=np.where([n['origin'][0]>.5 for n in pool],q,0.)
                variants[arm]=apply(base,pool,q,bundle['thresholds'])
                if arm==chosen['arm']:variants['chosen_plus_c7']=apply(previous[r['id']]['parts'],pool,q,bundle['thresholds'])
            for name,p in variants.items():
                s=metrics(r,p);items[name].append((r,s));tracks[name].append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],metrics=s,parts=p))
            print('RESIDUAL_REVIEW',scope,r['id'],flush=True)
        all_s[scope]={k:aggregate(v) for k,v in items.items()};all_t[scope]=tracks;all_diag[scope]=dict(diag);ci[scope]=bootstrap(rows,tracks['baseline'],tracks['chosen_plus_c7'])
    export=Path('models/residual_additions_review.pt');export.parent.mkdir(exist_ok=True);torch.save(states[chosen['arm']],export)
    save(REPORT/'comparison.json',dict(summary=all_s,tracks=all_t,coverage=all_diag,bootstrap=ci,chosen_arm=chosen['arm'],export=str(export),export_sha256=sha(export),test_used=False,automatic_application_approved=False))
    print('RESIDUAL_REVIEW_COMPLETE',chosen['arm'],all_s,flush=True)
if __name__=='__main__':main()
