"""Select the bass component on internal calibration, then freeze composition."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from band2tab.bass_keep import SCHEMA,apply
from scripts.a_keep_cycle import load
from scripts.a_keep_feedback_review import DATA as REG_DATA,bootstrap
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import metrics,aggregate,KEYS
from scripts.a_error_taxonomy import classify

REPORT=Path('reports/bass_keep_v5')
SOURCES=['runs/keep_cycle_v1/calibration_selected.pt','runs/keep_amp_v3/control/selected_ensemble.pt',
    'runs/keep_amp_v3/alternate_amp/selected_ensemble.pt','runs/keep_agreement_v4/zero_evidence/selected_ensemble.pt',
    'runs/keep_agreement_v4/agreement/selected_ensemble.pt']

def select():
    if (REPORT/'selection.json').exists():raise ValueError('Selection is frozen')
    save(REPORT/'protocol.json',dict(hypothesis='Guitar pruning did not transfer consistently. Keep the previous guitar correction fixed; select bass-only learner on internal20 with zero observed TP loss and nondecreasing four metrics. No new training.',
        sources={p:sha(p) for p in SOURCES},thresholds=[0.,.05,.1,.2,.35,.5,.65,.8],removal_cap_fraction=.1,
        selection='Bass-only .5Note+.3Frame+.2Exact; all four bass and full-mixture calibration metrics >= baseline; zero bass TP loss.',
        previous_guitar='Frozen octave/addition from before this batch. Composition checked only after bass selection.',
        upstream_overlap='Internal calibration excludes keep-head training performances; the earlier Your model may have trained on those recordings.',
        review_scope='Exploratory, previously inspected16/14. No fresh test opened.',test_used=False))
    _,rows,cc,mm=load();rows=[r for r in rows if r['phase']=='calibration'];torch.set_num_threads(2);choices=[]
    def evaluate(bundle):
        full=[];bass=[];lost=0
        for r in rows:
            m=mm[r['id']];parts=apply(bundle,cc[r['id']]['x'],m['pool'],m['unmodelled'],r['duration'],r['guard_seconds'])
            full.append((r,metrics(r,parts)))
            br={**r,'parts':{'guitar':[],'bass':r['parts']['bass']}};bass.append((r,metrics(br,{'guitar':[],'bass':parts['bass']})))
            old,_=classify(r,m['baseline_parts']);new,_=classify(r,parts);lost+=max(0,old.get('bass_tp',0)-new.get('bass_tp',0))
        return dict(full=aggregate(full),bass=aggregate(bass),lost_true_bass_onsets=lost)
    for source in SOURCES:
        learner=torch.load(source,map_location='cpu',weights_only=True)
        base_bundle=dict(schema=SCHEMA,learner=learner,max_remove_fraction=.1,threshold=0.)
        baseline=evaluate(base_bundle)
        for threshold in (0.,.05,.1,.2,.35,.5,.65,.8):
            bundle={**base_bundle,'threshold':threshold};result=evaluate(bundle)
            eligible=not result['lost_true_bass_onsets'] and all(result[scope][k]>=baseline[scope][k]-1e-10 for scope in ('full','bass') for k in ('note_f1','frame_f1','offset_f1','active_exact_match'))
            result.update(source=source,threshold=threshold,eligible=eligible,bass_score=.5*result['bass']['note_f1']+.3*result['bass']['frame_f1']+.2*result['bass']['active_exact_match'])
            choices.append(result)
    good=[v for v in choices if v['eligible'] and v['threshold']>0];best=max(good,key=lambda v:v['bass_score']) if good else None
    if not best:raise ValueError('No eligible bass learner')
    learner=torch.load(best['source'],map_location='cpu',weights_only=True)
    bundle=dict(schema=SCHEMA,learner=learner,threshold=best['threshold'],max_remove_fraction=.1,part='bass',
        automatic_application_approved=False,inference_requires_yourmt3=False,
        mu_sha256=sha('models/muscriptor-medium/model.safetensors'),selected_on='Internal calibration20, bass-only',source_checkpoint_sha256=sha(best['source']))
    export=Path('models/bass_keep_review.pt');torch.save(bundle,export)
    save(REPORT/'selection.json',dict(best=best,choices=choices,export=str(export),export_sha256=sha(export),
        no_new_training=True,no_new_test=True,automatic_application_approved=False))
    print('BASS_COMPONENT_SELECTED',best['source'],best['threshold'],best['bass'],flush=True)

def compose():
    if (REPORT/'composition.json').exists():raise ValueError('Composition review already exists')
    selected=json.load(open(REPORT/'selection.json'));bundle=torch.load(selected['export'],map_location='cpu',weights_only=True)
    from scripts.a_release_train import load as development
    _,drows,dc,dm=development();drows=[r for r in drows if r['split']=='val'];rrows=json.load(open(REG_DATA/'manifest.json'));rc={};rm={}
    for r in rrows:
        with np.load(r['cache']) as f:rc[r['id']]={k:f[k].copy() for k in f.files}
        rm[r['id']]=json.load(open(Path(r['cache']).with_suffix('.json')))
    previous=json.load(open('reports/octave_regression_v1/comparison.json'))['tracks'];summaries={};all_tracks={};cis={}
    from collections import Counter
    key=lambda n:(n['pitch'],n['onset'],n['offset'],n.get('velocity'))
    for scope,rows,cc,mm in [('development',drows,dc,dm),('known_regression',rrows,rc,rm)]:
        old={r['id']:r for r in previous[scope]};scores={k:[] for k in ('baseline','previous_guitar','bass_only','combined')};tracks={k:[] for k in scores}
        for r in rows:
            m=mm[r['id']];base=m['baseline_parts'];older=old[r['id']]['variants']
            for p in ('guitar','bass'):
                if Counter(map(key,base[p]))!=Counter(map(key,older['baseline']['parts'][p])):raise ValueError('Different baseline composition')
            bass=apply(bundle,cc[r['id']]['x'],m['pool'],m['unmodelled'],r['duration'],r['guard_seconds'])
            if Counter(map(key,bass['guitar']))!=Counter(map(key,base['guitar'])):raise ValueError('Protected guitar changed')
            g=older['combined']['parts'];combined=dict(guitar=g['guitar'],bass=bass['bass'])
            for name,parts in [('baseline',base),('previous_guitar',g),('bass_only',bass),('combined',combined)]:
                s=metrics(r,parts);scores[name].append((r,s));tracks[name].append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],metrics=s,parts=parts))
        summaries[scope]={k:aggregate(v) for k,v in scores.items()};all_tracks[scope]=tracks
        cis[scope]={k:bootstrap(rows,tracks['baseline'],v) for k,v in tracks.items() if k!='baseline'}
    save(REPORT/'composition.json',dict(summary=summaries,tracks=all_tracks,bootstrap=cis,unopened_test_access=False,
        prior_guitar_model_unchanged=True,bass_selection_sha256=sha(REPORT/'selection.json'),automatic_application_approved=False))
    print('BASS_GUITAR_COMPOSITION',summaries,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','compose']);a=p.parse_args();globals()[a.command]()
