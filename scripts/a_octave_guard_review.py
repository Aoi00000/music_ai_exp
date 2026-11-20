"""Frozen upper-octave policies on reused16/14, plus prior bass-only review."""
from pathlib import Path
from collections import Counter
import json,copy
import numpy as np,torch
from band2tab.audio import features
from band2tab.octave_guard import peers,patches,probabilities,prune
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import metrics,aggregate
from scripts.a_stress_data import local
from scripts.a_keep_feedback_review import bootstrap
DATA=Path('data/octave_guard_review_v1');REPORT=Path('reports/octave_guard_review_v1');BACK=Path('backups/azure')

def cache(r,baseline):
    pool=[{**n,'part':p} for p in ('guitar','bass') for n in baseline[p]];pair=peers(pool,r['duration'],r['guard_seconds']);p=DATA/'cache'/(r['id']+'.npz');meta=p.with_suffix('.json');source=Path(r['inputs']['guitar']);source=source if source.is_file() else local(BACK,source)
    fingerprint=dict(guitar_sha256=sha(source),implementation_sha256=sha('band2tab/octave_guard.py'))
    if not p.exists():
        x,m=patches(features(source),pool,pair) if pair else (np.empty((0,24,12),np.float32),np.empty((0,4),np.float32));p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,x=x,meta=m);save(meta,dict(pool=pool,pairs=pair,baseline_parts=baseline,fingerprint=fingerprint))
    m=json.load(open(meta))
    if m['pool']!=pool or m['pairs']!=[list(v) for v in pair] or m['fingerprint']!=fingerprint:raise ValueError('Stale review patch')
    with np.load(p) as z:c={k:z[k].copy() for k in z.files}
    return c,m

def main():
    if (REPORT/'comparison.json').exists():raise ValueError('Review already frozen')
    selection=json.load(open('reports/octave_guard_v1/selection.json'));chosen=selection['chosen'];torch.set_num_threads(2)
    save(REPORT/'protocol.json',dict(selected_before_review=chosen,selection_sha256=sha('reports/octave_guard_v1/selection.json'),
      scope='Reused16 and known14; not fresh tests',composition='New selected G replaces prior octave filter; optionally preserve previous sounding-aware bass proposals. If internal gate chose identity, retain previous C7 as app fallback.',
      test_used=False,automatic_application_approved=False))
    states={a['arm']:torch.load(a['export'],map_location='cpu',weights_only=True) for a in selection['arms']};prior=json.load(open('reports/sound_keep_conservative_v7/comparison.json'))['tracks'];summaries={};tracks={};cis={}
    for scope,rows in [('development',[r for r in json.load(open(BACK/'data/stem_refiner_v1/manifest.json')) if r['split']=='val']),('known_regression',json.load(open('data/stem_refiner_holdout_v1/manifest.json')))]:
        old={r['id']:r for r in prior[scope]['combined']};items={k:[] for k in ['baseline','previous_c7',*states,'selected_plus_bass','adoption_candidate']};out={k:[] for k in items}
        for r in rows:
            if scope=='development':base=json.load(open(BACK/'data/stem_refiner_v1/cache'/(r['id']+'.json')))['baseline_parts']
            else:base=json.load(open('data/keep_feedback_regression_v1/cache/'+r['id']+'.json'))['baseline_parts']
            c,m=cache(r,base);variants={'baseline':base,'previous_c7':old[r['id']]['parts']}
            for name,state in states.items():variants[name]=prune(base,m['pool'],m['pairs'],probabilities(state,c['x'],c['meta']),state['threshold'])
            selected=base if chosen=='identity' else variants[chosen];variants['selected_plus_bass']=dict(guitar=selected['guitar'],bass=old[r['id']]['parts']['bass']);variants['adoption_candidate']=old[r['id']]['parts'] if chosen=='identity' else variants['selected_plus_bass']
            for name,p in variants.items():
                if p['bass']!=base['bass'] and name not in ('previous_c7','selected_plus_bass','adoption_candidate'):raise ValueError('Protected bass changed')
                s=metrics(r,p);items[name].append((r,s));out[name].append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=p,metrics=s))
            print('OCTAVE_GUARD_REVIEW',scope,r['id'],flush=True)
        summaries[scope]={k:aggregate(v) for k,v in items.items()};tracks[scope]=out;cis[scope]=bootstrap(rows,out['baseline'],out['adoption_candidate'])
    export=None
    if chosen!='identity':
        export=Path('models/upper_octave_guard_review.pt');torch.save(states[chosen],export)
    save(REPORT/'comparison.json',dict(summary=summaries,tracks=tracks,bootstrap=cis,chosen_before_review=chosen,export=str(export) if export else None,export_sha256=sha(export) if export else None,test_used=False,automatic_application_approved=False))
    print('OCTAVE_GUARD_REVIEW_COMPLETE',chosen,flush=True)
if __name__=='__main__':main()
