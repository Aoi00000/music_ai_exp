"""Registered nonlinear control for the fixed upper-octave materials."""
from pathlib import Path
import json,time,pickle
import numpy as np,torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from band2tab.octave_harmonics import summarize
from scripts.a_octave_guard_cycle import load,evaluate,DATA
from scripts.a_candidate_data import save,sha
REPORT=Path('reports/octave_tree_v1');RUN=Path('runs/octave_tree_v1')

def main():
    if REPORT.exists():raise ValueError('Already registered')
    arms=['real_only','synth_clean','synth_drive'];leaves=[3,7,15]
    save(REPORT/'protocol.json',dict(reason='CNN and linear guards produced no G edits on reused review. Test nonlinear harmonic-feature interactions without relaxing protection gates.',
      arms=arms,max_leaf_nodes=leaves,max_iter=150,learning_rate=.05,min_samples_leaf=20,l2_regularization=10.,seed=17,
      positive_weight=3.,synthetic_weight=.5,normalization='Fit real-only mean/std, fixed across arms; clip8SD',
      selection='Internal24 only. Zero onsetTP loss, four metrics protected per dataset. Strict gain else identity. Same thresholds/cap as frozen CNN.',
      fixed_review='Choose on internal24 before scoring reused16/14. No external score threshold tuning.',
      source_sha256={p:sha(p) for p in ['scripts/a_octave_tree_cycle.py','band2tab/octave_harmonics.py','band2tab/octave_guard.py','scripts/a_octave_guard_cycle.py','scripts/a_candidate_train.py']},source_protocol_sha256=sha(DATA/'protocol.json'),test_used=False))
    rows,c,m=load();fit=[r for r in rows if r['phase']=='train' and c[r['id']]['mask'].any()];cal=[r for r in rows if r['phase']=='calibration'];xx=[];yy=[];ww=[]
    for r in fit:
        d=c[r['id']];v=d['mask']>.5;xx.append(summarize(d['x'][v],d['meta'][v]));yy.append(d['y'][v]);ww.append(np.ones(v.sum())/len(fit)/v.sum())
    real=np.concatenate(xx);yr=np.concatenate(yy);wr=np.concatenate(ww);mean=real.mean(0);std=real.std(0).clip(.05)
    with np.load(DATA/'synthetic.npz') as z:sy={k:z[k].copy() for k in z.files}
    base=evaluate(cal,c,m,{r['id']:np.ones(len(c[r['id']]['x'])) for r in cal},0.)['summary'];bs=.5*base['note_f1']+.3*base['frame_f1']+.2*base['active_exact_match'];results=[]
    with threadpool_limits(limits=2):
      for arm in arms:
        if arm=='real_only':x=real;y=yr;w=wr.copy()
        else:
            take=sy['tone']==int(arm=='synth_drive');sx=summarize(sy['x'][take],sy['meta'][take]);x=np.concatenate([real,sx]);y=np.r_[yr,sy['y'][take]];w=np.r_[.5*wr,np.full(len(sx),.5/len(sx))]
        w*=np.where(y>.5,3.,1.);w*=len(real)/w.sum();x=np.clip((x-mean)/std,-8,8)
        for leaf in leaves:
            name=f'{arm}_leaves{leaf}';folder=RUN/name;folder.mkdir(parents=True);start=time.monotonic()
            model=HistGradientBoostingClassifier(max_leaf_nodes=leaf,max_iter=150,learning_rate=.05,min_samples_leaf=20,l2_regularization=10.,random_state=17,early_stopping=False)
            model.fit(x,y,sample_weight=w);q={}
            for r in cal:
                d=c[r['id']];z=np.clip((summarize(d['x'],d['meta'])-mean)/std,-8,8);q[r['id']]=model.predict_proba(z)[:,1] if len(z) else np.empty(0)
            choices=[]
            for t in json.load(open(DATA/'protocol.json'))['thresholds']:
                v=evaluate(cal,c,m,q,t);s=v['summary'];ok=not v['lost_true_onsets'] and all(s['datasets'][d][k]>=ref[k]-1e-10 for d,ref in base['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'));choices.append(dict(threshold=t,eligible=ok,score=.5*s['note_f1']+.3*s['frame_f1']+.2*s['active_exact_match'],summary=s,lost_true_onsets=v['lost_true_onsets']))
            good=[v for v in choices if v['eligible'] and v['score']>bs+1e-9];best=max(good,key=lambda v:v['score']) if good else None;dest=folder/'selected.pkl'
            with open(dest,'wb') as f:pickle.dump(dict(model=model,mean=mean,std=std,threshold=best['threshold'] if best else 0.,schema='upper_octave_harmonic_tree_v1'),f)
            save(folder/'completed.json',dict(seconds=time.monotonic()-start,candidates=len(x),iterations=150,leaves=leaf,protocol_sha256=sha(REPORT/'protocol.json'),checkpoint_sha256=sha(dest)))
            results.append(dict(name=name,arm=arm,leaves=leaf,best=best,choices=choices,export=str(dest),sha256=sha(dest)));print('OCTAVE_TREE',name,None if best is None else best['threshold'],flush=True)
    good=[v for v in results if v['best']];winner=max(good,key=lambda v:v['best']['score']) if good else None
    save(REPORT/'selection.json',dict(baseline=base,models=results,winner=winner,test_used=False,automatic_application_approved=False));print('OCTAVE_TREE_COMPLETE',None if winner is None else winner['name'],flush=True)
if __name__=='__main__':main()
