"""Reproduce public aggregate evidence without copying private audio/note arrays."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import torch
from band2tab.addition import _assign
from band2tab.octave_guard import keep_labels
from band2tab.octave_harmonics import probabilities
from scripts.a_candidate_data import sha,save
ROOT=Path('data/octave_guard_v1')

def main():
    protocol=json.load(open(ROOT/'protocol.json'));ph=sha(ROOT/'protocol.json');checks=0
    assert sha('scripts/a_octave_guard_gpu.py')==protocol['workload_sha256'];checks+=1
    for name,expected in protocol['implementation_sha256'].items():
        assert sha(name)==expected,(name,'source changed')
        assert sha(ROOT/'frozen_sources'/name)==expected;checks+=2
    for name,expected in protocol['cache_sha256'].items():assert sha(name)==expected;checks+=1
    for key,name in [('manifest_sha256','manifest.json'),('synthetic_sha256','synthetic.npz'),('synthetic_metadata_sha256','synthetic.json')]:assert sha(ROOT/name)==protocol[key];checks+=1
    for seed in (17,29,43):
        traces=[]
        for arm in ('real_only','synth_clean','synth_drive'):
            folder=Path('runs/octave_guard_v1')/arm/f'seed_{seed}';cfg=json.load(open(folder/'config.json'));d=json.load(open(folder/'completed.json'))
            assert cfg['protocol_sha256']==ph;assert d['steps']==1200;assert sha(folder/'final.pt')==d['checkpoint_sha256']
            assert sha(folder/'final.pt')==sha(Path('backups/azure')/folder/'final.pt');traces.append(d['draw_sha256']);checks+=4
        assert len(set(traces))==1
    experiments={}
    for name in ('octave_guard_v1','octave_harmonics_v1','octave_six_v1','octave_tree_v1'):
        d=json.load(open('reports/'+name+'/selection.json'));models=d.get('arms',d.get('models'));experiments[name]=dict(baseline=d['baseline'],models=[])
        for m in models:
            assert sha(m['export'])==m['sha256']
            experiments[name]['models'].append({k:m[k] for k in ('arm','name','best','choices') if k in m})
        if name!='octave_guard_v1':
            p=json.load(open('reports/'+name+'/protocol.json'));src=p.get('implementation_sha256',p.get('source_sha256',{}))
            for file,expected in src.items():assert sha(file)==expected;checks+=1
            for f in (Path('runs')/name).glob('*/completed.json'):
                r=json.load(open(f));assert r['protocol_sha256']==sha('reports/'+name+'/protocol.json');checks+=1
                checkpoint=f.parent/('selected.pkl' if name=='octave_tree_v1' else 'final.pt');assert sha(checkpoint)==r['checkpoint_sha256'];checks+=1
    reviews={}
    source_rows={'development':[r for r in json.load(open('backups/azure/data/stem_refiner_v1/manifest.json')) if r['split']=='val'],'known_regression':json.load(open('data/stem_refiner_holdout_v1/manifest.json'))}
    diagnosis={}
    state=torch.load('models/upper_octave_guard_review.pt',weights_only=True)
    for scope,rows in source_rows.items():
        count=Counter();qs=[]
        for r in rows:
            p=Path('data/octave_guard_review_v1/cache')/(r['id']+'.json');meta=json.load(open(p));pool=meta['pool'];pairs=meta['pairs'];ys,mask=keep_labels(pool,pairs,r['parts'],r['duration'],r['guard_seconds']);g=r['guard_seconds']
            ix=[i for i,n in enumerate(pool) if n['part']=='guitar' and g<=n['onset']<r['duration']-g]
            refs=[n for n in r['parts']['guitar'] if not n.get('censored_onset') and g<=n['onset']<r['duration']-g]
            correct={ix[j] for i,j in _assign(refs,[pool[i] for i in ix])}
            for k,(i,j) in enumerate(pairs):
                count['eligible']+=1
                count['onset_true' if i in correct else ('onset_false_but_true_frames' if ys[k] else 'onset_false_no_true_frames')]+=1
                count['teacher_censored']+=int(not mask[k])
            with np.load(p.with_suffix('.npz')) as z:qs.extend(probabilities(state,z['x'],z['meta']).tolist())
        diagnosis[scope]=dict(counts=dict(count),linear_score_quantiles=np.quantile(qs,[0,.01,.1,.5,.9,1]).tolist(),linear_threshold=.05,candidates_below_threshold=sum(v<.05 for v in qs))
    for name in ('octave_guard_joint_v1','octave_six_review_v1','octave_tree_review_v1'):
        d=json.load(open('reports/'+name+'/comparison.json'));pooled={};changes={}
        for scope,methods in d['tracks'].items():
            pooled[scope]={};changes[scope]={}
            for method,tracks in methods.items():
                pooled[scope][method]={k:sum(t['metrics'][k] for t in tracks) for k in ('note_tp','note_fp','note_fn','frame_tp','frame_fp','frame_fn')}
                changes[scope][method]={p:sum(len(a['parts'][p])-len(b['parts'][p]) for a,b in zip(methods['baseline'],tracks)) for p in ('guitar','bass')}
        reviews[name]=dict(chosen_before_review=d['chosen_before_review'],summary=d['summary'],pooled=pooled,deleted_notes=changes,bootstrap=d['bootstrap'])
    azure=json.load(open('reports/azure_octave_final_20261120.json'));assert azure['deallocated']
    evidence=dict(date='2026-11-20',scope='Repeated development16 + known regression14; exploratory, not fresh independent tests. Final11 Slakh sources remain unopened/unscored.',new_learning=dict(cnn_runs=9,cnn_parameters=26081,cnn_updates=10800,linear70_fits=9,linear35_fits=9,boosted_tree_fits=9,total_fits=36),protocol_sha256=ph,counts=protocol['counts'],derived_examples=512,experiments=experiments,reviews=reviews,diagnosis=diagnosis,integrity_checks=checks,azure=azure,app_default_changed=False,automatic_application_approved=False,decision='No new guitar policy adopted. Preserve baseline and previous optional C7 with explicit TP-loss warning. Do not replace prior review by new identity policy.',test_used=False)
    save('docs/measurements/octave_guard_20261120.json',evidence);print(json.dumps(dict(checks=checks,diagnosis=diagnosis,azure=azure),ensure_ascii=False))
if __name__=='__main__':main()
