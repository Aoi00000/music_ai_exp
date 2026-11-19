"""Registered small-data tree alternative using the SAME residual features."""
from pathlib import Path
import argparse,json,time,itertools
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from scripts.a_residual_cycle import load,evaluate
from scripts.a_candidate_data import save,sha
DATA=Path('data/residual_v1');REPORT=Path('reports/residual_tree_v2');RUN=Path('runs/residual_tree_v2')

def register():
    if (REPORT/'protocol.json').exists():raise ValueError('Protocol exists')
    save(REPORT/'protocol.json',dict(reason='Residual training is a small, rare-positive tabular problem. Compare gradient-boosted trees against the small NN without more audio or new-test access.',
      arms=['original','diverse'],seeds=[17,29,43],max_iter=200,max_leaf_nodes=15,max_depth=5,min_samples_leaf=20,l2_regularization=1.,learning_rate=.05,
      early_stopping=False,positive_weight=8.,row_weight='Equal expected row draws, original:base only; diverse:75% base/25% extra',
      features='Same366; all candidates including Mu octave and Your support; no targets in input',selection='Same internal24 per-dataset all4 nondecreasing; same49 threshold pairs; select before reused16/14 review',
      source_protocol_sha256=sha(DATA/'protocol.json'),implementation_sha256=sha(__file__),test_used=False))

def scores(state,x):
    # Average three independently fitted models, not calibrated probabilities.
    return np.mean([m.predict_proba(x)[:,1] for m in state['members']],axis=0)

def train():
    p=json.load(open(REPORT/'protocol.json'))
    if sha(__file__)!=p['implementation_sha256']:raise ValueError('Implementation changed')
    rows,c,m=load();base=[r for r in rows if r['phase']=='train' and r['dataset'].startswith('stem_band_') and np.any((c[r['id']]['origin']>.5)&(c[r['id']]['mask']>.5))];ids={r['id'] for r in base};extra=[r for r in rows if r['phase']=='train' and r['id'] not in ids and np.any(c[r['id']]['mask']>.5)]
    with threadpool_limits(limits=2):
      for arm in p['arms']:
        selected=base if arm=='original' else base+extra;x=[];y=[];w=[]
        for r in selected:
            z=c[r['id']];ix=z['mask']>.5;count=ix.sum();x.append(z['x'][ix]);y.append(z['y'][ix]);domain=.75/len(base) if r['id'] in ids else .25/len(extra)
            if arm=='original':domain=1/len(base)
            w.append(np.where(z['y'][ix]>.5,8.,1.)*domain/count)
        x=np.concatenate(x);y=np.concatenate(y);w=np.concatenate(w);w=w/w.mean();members=[]
        for seed in p['seeds']:
            folder=RUN/arm/('seed_'+str(seed))
            if folder.exists():raise ValueError('Do not overwrite tree run')
            started=time.monotonic();model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,max_depth=5,min_samples_leaf=20,l2_regularization=1.,learning_rate=.05,early_stopping=False,random_state=seed)
            model.fit(x,y,sample_weight=w);folder.mkdir(parents=True);joblib.dump(model,folder/'final.joblib');members.append(model)
            save(folder/'completed.json',dict(rows=len(selected),candidates=len(x),positive=int(y.sum()),iterations=int(model.n_iter_),seconds=time.monotonic()-started,protocol_sha256=sha(REPORT/'protocol.json'),checkpoint_sha256=sha(folder/'final.joblib')));print('RESIDUAL_TREE_TRAINED',arm,seed,flush=True)
        joblib.dump(dict(schema='residual_tree_v2',members=members),RUN/arm/'ensemble.joblib')

def select():
    if (REPORT/'selection.json').exists():raise ValueError('Selection frozen')
    rows,c,m=load();rows=[r for r in rows if r['phase']=='calibration'];baseline=evaluate(rows,c,m,{r['id']:np.zeros(len(c[r['id']]['x'])) for r in rows},(1.1,1.1));arms=[]
    with threadpool_limits(limits=2):
      for arm in ('original','diverse'):
        path=RUN/arm/'ensemble.joblib';state=joblib.load(path);q={r['id']:scores(state,c[r['id']]['x']) for r in rows};choices=[]
        for thresholds in itertools.product((.5,.65,.8,.9,.95,.99,1.1),repeat=2):
            value=evaluate(rows,c,m,q,thresholds);s=value['summary'];b=baseline['summary'];eligible=all(s['datasets'][d][k]>=v[k]-1e-10 for d,v in b['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'))
            choices.append(dict(thresholds=thresholds,eligible=eligible,score=.5*s['note_f1']+.3*s['frame_f1']+.2*s['active_exact_match'],summary=s))
        best=max((v for v in choices if v['eligible']),key=lambda v:v['score']);arms.append(dict(arm=arm,best=best,choices=choices,checkpoint=str(path),sha256=sha(path)));print('RESIDUAL_TREE_SELECTED',arm,best,flush=True)
    save(REPORT/'selection.json',dict(baseline=baseline['summary'],arms=arms,test_used=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','train','select']);a=p.parse_args();globals()[a.command]()
