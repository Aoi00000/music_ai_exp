"""Same-condition validation of YourMT3+, Mu and learned candidate correction."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from band2tab.score import read_midi
from band2tab.candidate import CandidateRanker,predict
from scripts.a_candidate_data import rows_for,REPORT,save,sha
from scripts.a_candidate_train import metrics,aggregate,baselines,bound,load,candidate_parts,KEYS
from scripts.a_candidate_review import errors

def main():
    p=argparse.ArgumentParser();p.add_argument('--ready',action='store_true');a=p.parse_args()
    rows=[r for r in rows_for(False) if r['split']=='val'];completed=[];missing=[]
    for r in rows:
        path=Path('reports/yourmt3/midi')/(r['id']+'.mid')
        if path.exists() and path.with_suffix('.json').exists():completed.append(r)
        else:missing.append(r['id'])
    if missing and not a.ready:raise ValueError('Validation inference incomplete '+str(missing[:3]))
    trials=[json.loads(p.read_text()) for p in Path('runs/candidate_v2').glob('*/completed.json')]
    selected=max((t['best'] for t in trials),key=lambda t:t['validation']['score'])
    if sha(selected['checkpoint'])!=selected['checkpoint_sha256']:raise ValueError('Candidate checkpoint changed')
    state=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True);model=CandidateRanker(context=state['context']).to('cuda');model.load_state_dict(state['model']);model.eval();torch.set_num_threads(4)
    records=[]
    for r in completed:
        path=Path('reports/yourmt3/midi')/(r['id']+'.mid');meta=json.loads(path.with_suffix('.json').read_text())
        if meta['audio_sha256']!=sha(r['inputs']['mix']):raise ValueError('Stale YourMT3 inference')
        parts=baselines(r);original=read_midi(path);parts['YourMT3']=bound({p:original.get(p,[]) for p in ('guitar','bass')},r['duration'])
        data=load(r);prediction=predict(model,data['x'],data['prior'],np.array(state['mean'],np.float32),np.array(state['std'],np.float32),'cuda')
        parts['Mu_learned']=candidate_parts(r,data,prediction,selected['policy'],parts['Mu'])
        records.append({'id':r['id'],'dataset':r['dataset'],'metrics':{k:{z:s[z] for z in KEYS} for k,v in parts.items() for s in [metrics(r,v)]},'errors':{k:errors(r,v) for k,v in parts.items()}})
    summary={m:aggregate([(r,t['metrics'][m]) for r,t in zip(completed,records)]) for m in ('Mu','six_Mu','Mu_learned','YourMT3')}
    save('reports/yourmt3/'+('validation_progress.json' if missing else 'validation_comparison.json'),
        {'complete':not missing,'cases':len(completed),'missing':missing,'candidate_run':selected['run'],'summary':summary,'tracks':records,
         'scope':'Validation/development only; public YourMT3 training includes Slakh and GuitarSet. Not evidence of independent pretrained-model generalization.'})
    print('YOURMT3_VALIDATION',len(completed),'complete',not missing)
    for m,s in summary.items():print(m,{k:round(s[k],4) for k in (*KEYS,'score')},flush=True)

if __name__=='__main__':main()
