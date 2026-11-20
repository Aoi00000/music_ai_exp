"""Same candidate task/materials, small harmonic-summary logistic controls."""
from pathlib import Path
import json,copy,time
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from band2tab.octave_harmonics import SCHEMA,summarize,probabilities
from scripts.a_octave_guard_cycle import load,evaluate,DATA
from scripts.a_candidate_data import save,sha
REPORT=Path('reports/octave_harmonics_v1');RUN=Path('runs/octave_harmonics_v1')

def main():
    if REPORT.exists():raise ValueError('Harmonic experiment already registered')
    grid=[.03,.3,3.];arms=['real_only','synth_clean','synth_drive'];source=json.load(open(DATA/'protocol.json'))
    save(REPORT/'protocol.json',dict(reason='1214 real octave candidates are small for26k CNN weights. Compare70 harmonic/envelope features with L2 logistic regression and targeted augmentation.',
       arms=arms,regularization_C=grid,positive_keep_weight=3.,synthetic_weight=.5,normalization='Original fit real-only, identical for all arms; clipped8SD',
       row_weight='Equal original row sampling; augmented substitutes50% with paired synthetic cases. Weight sums held fixed across arms.',
       selection='Same internal24, zero onsetTP loss and all4 per-dataset protected; strict gain else identity.',
       joint_selection='Before any reused16/14 review, compare CNN and harmonic logistic by same internal score; do not pick from reused outcomes.',
       source_protocol_sha256=sha(DATA/'protocol.json'),implementation_sha256={p:sha(p) for p in ['band2tab/octave_harmonics.py','scripts/a_octave_harmonic_cycle.py']},test_used=False))
    rows,c,m=load();fit=[r for r in rows if r['phase']=='train' and c[r['id']]['mask'].any()];cal=[r for r in rows if r['phase']=='calibration'];real=[];yy=[];weights=[]
    for r in fit:
        d=c[r['id']];valid=d['mask']>.5;real.append(summarize(d['x'][valid],d['meta'][valid]));yy.append(d['y'][valid]);weights.append(np.ones(valid.sum())/len(fit)/valid.sum())
    real=np.concatenate(real);yreal=np.concatenate(yy);wreal=np.concatenate(weights);mean=real.mean(0);std=real.std(0).clip(.05)
    with np.load(DATA/'synthetic.npz') as z:synthetic={k:z[k].copy() for k in z.files}
    base=evaluate(cal,c,m,{r['id']:np.ones(len(c[r['id']]['x'])) for r in cal},0.);b=base['summary'];baseline_score=.5*b['note_f1']+.3*b['frame_f1']+.2*b['active_exact_match'];results=[]
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
      for arm in arms:
        if arm=='real_only':x=real;y=yreal;w=wreal.copy()
        else:
            pick=synthetic['tone']==(1 if arm=='synth_drive' else 0);sx=summarize(synthetic['x'][pick],synthetic['meta'][pick]);x=np.concatenate([real,sx]);y=np.concatenate([yreal,synthetic['y'][pick]]);w=np.r_[.5*wreal,np.full(len(sx),.5/len(sx))]
        w=w*np.where(y>.5,3.,1.);w=w*len(real)/w.sum();z=np.clip((x-mean)/std,-8,8)
        for strength in grid:
            ident=arm+'_C'+str(strength);folder=RUN/ident;folder.mkdir(parents=True);started=time.monotonic();model=LogisticRegression(C=strength,max_iter=2000,solver='lbfgs',tol=1e-8);model.fit(z,y,sample_weight=w)
            state=dict(schema=SCHEMA,mean=mean.tolist(),std=std.tolist(),coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),C=strength,arm=arm,cap_fraction=.1)
            path=folder/'final.pt';torch.save(state,path);save(folder/'completed.json',dict(candidates=len(x),parameters=len(model.coef_[0])+1,iterations=int(model.n_iter_[0]),seconds=time.monotonic()-started,protocol_sha256=sha(REPORT/'protocol.json'),checkpoint_sha256=sha(path)))
            qs={r['id']:probabilities(state,c[r['id']]['x'],c[r['id']]['meta']) for r in cal};choices=[]
            for t in source['thresholds']:
                v=evaluate(cal,c,m,qs,t);s=v['summary'];ok=not v['lost_true_onsets'] and all(s['datasets'][d][k]>=ref[k]-1e-10 for d,ref in b['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'));choices.append(dict(threshold=t,eligible=ok,score=.5*s['note_f1']+.3*s['frame_f1']+.2*s['active_exact_match'],summary=s,lost_true_onsets=v['lost_true_onsets']))
            allowed=[v for v in choices if v['eligible'] and v['score']>baseline_score+1e-9];best=max(allowed,key=lambda a:a['score']) if allowed else None;state['threshold']=best['threshold'] if best else 0.;export=folder/'selected_bundle.pt';torch.save(state,export);results.append(dict(arm=arm,kind='harmonic',name=ident,best=best,choices=choices,export=str(export),sha256=sha(export)));print('HARMONIC_GUARD_SELECTED',ident,best,flush=True)
    good=[r for r in results if r['best']];winner=max(good,key=lambda a:a['best']['score']) if good else None
    save(REPORT/'selection.json',dict(baseline=b,models=results,winner=winner,test_used=False,automatic_application_approved=False));print('HARMONIC_GUARD_COMPLETE',None if winner is None else winner['name'],flush=True)
if __name__=='__main__':main()
