"""Select CNN/linear guards on internal data, then freeze reused-set comparison."""
from pathlib import Path
import json,copy
import numpy as np,torch
from band2tab.octave_guard import probabilities as cnn_prob,prune
from band2tab.octave_six import probabilities as harmonic_prob,SCHEMA as HARMONIC_SCHEMA
from scripts.a_octave_guard_review import cache
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import aggregate,metrics
from scripts.a_keep_feedback_review import bootstrap
REPORT=Path('reports/octave_six_review_v1');BACK=Path('backups/azure')

def predict(state,x,meta):return harmonic_prob(state,x,meta) if state['schema']==HARMONIC_SCHEMA else cnn_prob(state,x,meta)

def main():
    if (REPORT/'comparison.json').exists():raise ValueError('Joint review frozen')
    b=json.load(open('reports/octave_six_v1/selection.json'))
    models=[{**v,'name':'six_'+v['name']} for v in b['models']];good=[v for v in models if v['best']];winner=max(good,key=lambda v:v['best']['score']) if good else None;chosen=winner['name'] if winner else 'identity'
    save(REPORT/'selection.json',dict(chosen=chosen,winner=winner,identity_selected=winner is None,source_sha256={p:sha(p) for p in ['reports/octave_six_v1/selection.json','scripts/a_octave_six_review.py','band2tab/octave_six.py']},scope='Internal24 only, BEFORE any reused16/14 scores; fallback previousC7 if identity',test_used=False))
    states={v['name']:torch.load(v['export'],weights_only=True,map_location='cpu') for v in models};torch.set_num_threads(2);prior=json.load(open('reports/sound_keep_conservative_v7/comparison.json'))['tracks'];summaries={};tracks={};cis={}
    for scope,rows in [('development',[r for r in json.load(open(BACK/'data/stem_refiner_v1/manifest.json')) if r['split']=='val']),('known_regression',json.load(open('data/stem_refiner_holdout_v1/manifest.json')))]:
        old={r['id']:r for r in prior[scope]['combined']};items={k:[] for k in ['baseline','previous_c7',*states,'selected_plus_bass','adoption_candidate']};out={k:[] for k in items}
        for r in rows:
            parent=BACK/'data/stem_refiner_v1/cache' if scope=='development' else Path('data/keep_feedback_regression_v1/cache');base=json.load(open(parent/(r['id']+'.json')))['baseline_parts'];c,m=cache(r,base)
            variants={'baseline':base,'previous_c7':old[r['id']]['parts']}
            for name,state in states.items():variants[name]=prune(base,m['pool'],m['pairs'],predict(state,c['x'],c['meta']),state['threshold'])
            selected=base if chosen=='identity' else variants[chosen];variants['selected_plus_bass']=dict(guitar=selected['guitar'],bass=old[r['id']]['parts']['bass']);variants['adoption_candidate']=old[r['id']]['parts'] if chosen=='identity' else variants['selected_plus_bass']
            for name,p in variants.items():
                s=metrics(r,p);items[name].append((r,s));out[name].append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=p,metrics=s))
            print('OCTAVE_JOINT_REVIEW',scope,r['id'],flush=True)
        summaries[scope]={k:aggregate(v) for k,v in items.items()};tracks[scope]=out;cis[scope]=bootstrap(rows,out['baseline'],out['adoption_candidate'])
    export=None
    if winner:
        export=Path('models/upper_octave_six_review.pt');package={**states[chosen],'mu_sha256':'ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec','source_protocol_sha256':sha('data/octave_guard_v1/protocol.json'),'joint_selection_sha256':sha(REPORT/'selection.json'),'automatic_application_approved':False};torch.save(package,export)
    save(REPORT/'comparison.json',dict(summary=summaries,tracks=tracks,bootstrap=cis,chosen_before_review=chosen,export=str(export) if export else None,export_sha256=sha(export) if export else None,test_used=False,automatic_application_approved=False));print('OCTAVE_JOINT_COMPLETE',chosen,flush=True)
if __name__=='__main__':main()
