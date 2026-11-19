"""Second feedback cycle: calibration-only per-role risk limits, no new learner.

The hypothesis follows the observed v1 development regression. Those repeated
development examples are diagnostic, not a new unseen test.
"""
from pathlib import Path
import json,itertools
import numpy as np,torch
from band2tab.keep import probabilities,prune
from scripts.a_keep_cycle import DATA,REPORT as OLD,load
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate
from scripts.a_error_taxonomy import classify

REPORT=Path('reports/keep_policy_v2')

def apply(pool,q,thresholds,unmodelled,duration,guard=.25):
    # Score ranks only; never pretend that it is calibrated correctness.
    adjusted=np.ones(len(pool),np.float32)
    for part,t in zip(('guitar','bass'),thresholds):
        eligible=[i for i,n in enumerate(pool) if n['part']==part and guard<=n['onset']<duration-guard]
        maximum=int(len(eligible)*.1)
        chosen=sorted((i for i in eligible if q[i]<t),key=lambda i:float(q[i]))[:maximum]
        adjusted[chosen]=0.
    return prune(pool,adjusted,.5,unmodelled,duration,guard)

def evaluate(rows,cache,meta,qs,thresholds):
    items=[];tracks=[];lost_tp={p:0 for p in ('guitar','bass')};removed=0
    for r in rows:
        m=meta[r['id']];parts=apply(m['pool'],qs[r['id']],thresholds,m['unmodelled'],r['duration'],r['guard_seconds'])
        s=metrics(r,parts);items.append((r,s));before,_=classify(r,m['baseline_parts']);after,_=classify(r,parts)
        for part in lost_tp:lost_tp[part]+=max(0,before.get(part+'_tp',0)-after.get(part+'_tp',0))
        count=sum(len(v) for v in m['baseline_parts'].values())-sum(len(v) for v in parts.values());removed+=count
        tracks.append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=parts,metrics=s,removed=count))
    summary=aggregate(items);summary['keep_score']=.5*summary['note_f1']+.3*summary['frame_f1']+.2*summary['active_exact_match']
    return dict(summary=summary,tracks=tracks,lost_true_onsets=lost_tp,removed=removed)

def select():
    protocol_path=REPORT/'protocol.json'
    if not protocol_path.exists():raise ValueError('Write protocol before selecting')
    if (REPORT/'selection.json').exists():raise ValueError('Selection frozen')
    _,rows,cache,meta=load();rows=[r for r in rows if r['phase']=='calibration']
    selected=json.load(open(OLD/'selection.json'))
    if not selected['export']:raise ValueError('No source model to calibrate')
    state=torch.load(selected['export'],map_location='cpu',weights_only=True);torch.set_num_threads(2)
    qs={r['id']:probabilities(state,cache[r['id']]['x']) for r in rows}
    base=evaluate(rows,cache,meta,qs,(0.,0.));b=base['summary'];choices=[]
    for thresholds in itertools.product((0.,.05,.1,.2,.35,.5,.65,.8),repeat=2):
        v=evaluate(rows,cache,meta,qs,thresholds);s=v['summary']
        eligible=all(s[k]>=b[k]-1e-10 for k in ('note_f1','frame_f1','active_exact_match','offset_f1')) and not any(v['lost_true_onsets'].values())
        v.update(thresholds=thresholds,eligible=eligible);choices.append(v)
    candidates=[v for v in choices if v['eligible'] and v['summary']['keep_score']>b['keep_score']+1e-9]
    best=max(candidates,key=lambda v:v['summary']['keep_score']) if candidates else None
    save(REPORT/'selection.json',dict(baseline=base,best=best,identity_selected=best is None,
        choices=[{k:v for k,v in x.items() if k!='tracks'} for x in choices],source_checkpoint=selected['export'],
        source_checkpoint_sha256=sha(selected['export']),no_new_training=True,development_used_for_selection=False))
    print('KEEP_ROLE_POLICY_SELECTED',None if best is None else (best['thresholds'],best['removed'],best['summary']),flush=True)

def review():
    from scripts.a_release_train import load as old_load
    if (REPORT/'development.json').exists():raise ValueError('Do not overwrite fixed review')
    selection=json.load(open(REPORT/'selection.json'));state=torch.load(selection['source_checkpoint'],map_location='cpu',weights_only=True)
    _,rows,cache,meta=old_load();rows=[r for r in rows if r['split']=='val'];torch.set_num_threads(2)
    qs={r['id']:probabilities(state,cache[r['id']]['x']) for r in rows};thresholds=selection['best']['thresholds'] if selection['best'] else (0,0)
    baseline=evaluate(rows,cache,meta,qs,(0,0));selected=evaluate(rows,cache,meta,qs,thresholds)
    save(REPORT/'development.json',dict(baseline=baseline,selected=selected,thresholds=thresholds,
        note='Repeated development, hypothesis informed by v1 failure; no unseen-test claim; no further tuning here.'))
    print('KEEP_ROLE_DEVELOPMENT',baseline['summary'],selected['summary'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','select','review']);a=p.parse_args()
    if a.command=='register':
        if (REPORT/'protocol.json').exists():raise ValueError('Protocol exists')
        save(REPORT/'protocol.json',dict(hypothesis='The global threshold deleted true guitars despite improving bass. Calibrate per-role with zero observed calibration TP loss and cap removals to 10% per role per clip.',
            source_selection_sha256=sha(OLD/'selection.json'),source_model_frozen=True,no_new_training=True,
            thresholds=[0.,.05,.1,.2,.35,.5,.65,.8],removal_cap_fraction=.1,calibration='Same internal20; zero true-onset loss both roles; all four macro metrics nondecreasing.',
            inference='Features and predictions only; no reference labels. Pitch/onset/offset/role immutable.',
            development16_informed_hypothesis=True,development16_used_to_select_threshold=False,unopened_test_access=False))
        print('KEEP_ROLE_POLICY_REGISTERED')
    elif a.command=='select':select()
    else:review()
