"""Final calibration feedback: rare positives may score below0.5, not probabilities.

Registered after internal tie diagnoses. No new learner or use of final test.
"""
from pathlib import Path
import json,itertools
import numpy as np,torch,joblib
from threadpoolctl import threadpool_limits
from band2tab.residual import probabilities
from scripts.a_residual_cycle import load,evaluate
from scripts.a_residual_tree import scores
from scripts.a_candidate_data import save,sha
ROOT=Path('reports/residual_calibration_v3')

def main():
    if (ROOT/'selection.json').exists():raise ValueError('Calibration frozen')
    thresholds=[.05,.1,.2,.35,.5,.65,.8,.95,1.1]
    save(ROOT/'protocol.json',dict(reason='Original NN search .5..1.1 produced no strict calibration gains. Rare-positive scores are not calibrated probabilities; broaden operating-point search using internal24 only.',
       thresholds=thresholds,criterion='Unchanged: all4 per-dataset nondecreasing and strict macro score gain, else explicit identity.',no_new_training=True,no_new_test=True,reused_outcomes_used_to_choose_threshold=False,implementation_sha256=sha(__file__)))
    rows,c,m=load();rows=[r for r in rows if r['phase']=='calibration'];torch.set_num_threads(2)
    original=json.load(open('reports/residual_v1/selection.json'));tree=json.load(open('reports/residual_tree_v2/selection.json'));learners=[]
    for a in original['arms']:
        state=torch.load(a['export'],map_location='cpu',weights_only=True);q={r['id']:probabilities(state,c[r['id']]['x']) for r in rows}
        if not state['expanded']:
            for r in rows:q[r['id']]=np.where(c[r['id']]['origin']>.5,q[r['id']],0.)
        learners.append(('nn_'+a['arm'],q,a['export'],'nn'))
    with threadpool_limits(limits=2):
      for a in tree['arms']:
        state=joblib.load(a['checkpoint']);q={r['id']:scores(state,c[r['id']]['x']) for r in rows};learners.append(('tree_'+a['arm'],q,a['checkpoint'],'tree'))
    baseline=evaluate(rows,c,m,{r['id']:np.zeros(len(c[r['id']]['x'])) for r in rows},(1.1,1.1))['summary'];base_score=.5*baseline['note_f1']+.3*baseline['frame_f1']+.2*baseline['active_exact_match'];arms=[]
    for name,q,path,kind in learners:
        choices=[]
        for ts in itertools.product(thresholds,repeat=2):
            v=evaluate(rows,c,m,q,ts);s=v['summary'];ok=all(s['datasets'][ds][k]>=ref[k]-1e-10 for ds,ref in baseline['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'));choices.append(dict(thresholds=ts,eligible=ok,score=.5*s['note_f1']+.3*s['frame_f1']+.2*s['active_exact_match'],summary=s))
        good=[v for v in choices if v['eligible'] and v['score']>base_score+1e-9];best=max(good,key=lambda v:v['score']) if good else None;arms.append(dict(name=name,kind=kind,checkpoint=path,checkpoint_sha256=sha(path),best=best,choices=choices));print('LOW_SCORE_CALIBRATION',name,None if best is None else (best['thresholds'],best['summary']),flush=True)
    good=[a for a in arms if a['best']];winner=max(good,key=lambda a:a['best']['score']) if good else None
    save(ROOT/'selection.json',dict(baseline=baseline,winner=winner,arms=arms,identity_selected=winner is None,test_used=False,automatic_application_approved=False));print('LOW_SCORE_CALIBRATION_COMPLETE',None if winner is None else winner['name'],flush=True)
if __name__=='__main__':main()
