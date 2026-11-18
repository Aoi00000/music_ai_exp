"""Validation-only candidate coverage and development error analysis."""
from pathlib import Path
import argparse,json
from collections import defaultdict
import numpy as np,torch,mir_eval
from band2tab.candidate import CandidateRanker,predict
from scripts.a_candidate_data import REPORT,sha,save,rows_for
from scripts.a_candidate_train import load,baselines,metrics,candidate_parts,KEYS

def matching(ref,est,tolerance=.05):
    if not ref or not est:return []
    intervals=lambda ns:np.array([[n['onset'],n['offset']] for n in ns],float).reshape(-1,2)
    pitch=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
    return mir_eval.transcription.match_notes(intervals(ref),pitch(ref),intervals(est),pitch(est),onset_tolerance=tolerance,offset_ratio=None)

def errors(r,parts):
    g=r['guard_seconds'];end=r['duration']-g if g else float('inf');out={}
    for part in ('guitar','bass'):
        refs=[n for n in r['parts'].get(part,[]) if g<=n['onset']<end and not n.get('censored_onset')]
        est=[n for n in parts.get(part,[]) if g<=n['onset']<end];pairs=matching(refs,est);used={b for a,b in pairs};extra=[n for i,n in enumerate(est) if i not in used]
        out[part]={'reference_notes':len(refs),'matched':len(pairs),'missed':len(refs)-len(pairs),'extra':len(extra),
            'sustain_reattack_overlap':sum(any(n['pitch']==b['pitch'] and b['onset']+.05<n['onset']<b['offset'] for b in refs) for n in extra),
            'offset_mismatches':int(sum(not refs[a].get('censored_offset') and abs(refs[a]['offset']-est[b]['offset'])>max(.05,.2*(refs[a]['offset']-refs[a]['onset'])) for a,b in pairs))}
    return out

def coverage(r,pool):
    g=r['guard_seconds'];end=r['duration']-g if g else float('inf');counts=defaultdict(int)
    estimates=[n for n in pool if g<=n['onset']<end]
    for part in ('guitar','bass'):
        refs=[n for n in r['parts'].get(part,[]) if g<=n['onset']<end and not n.get('censored_onset')]
        counts['reference_notes']+=len(refs)
        for tolerance in (.05,.12):counts[str(tolerance)]+=len(matching(refs,estimates,tolerance))
    return dict(counts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run');a=p.parse_args();rows=[r for r in rows_for(False) if r['split']=='val'];model=None
    if a.run:
        selection=json.loads((Path('runs/candidate_v2')/a.run/'completed.json').read_text())['best']
        if sha(selection['checkpoint'])!=selection['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        state=torch.load(selection['checkpoint'],map_location='cpu',weights_only=True);model=CandidateRanker(context=state['context']).to('cuda');model.load_state_dict(state['model']);model.eval();torch.set_num_threads(4)
    records=[]
    for r in rows:
        data=load(r);parts=baselines(r)
        if model is not None:
            pred=predict(model,data['x'],data['prior'],np.array(state['mean'],np.float32),np.array(state['std'],np.float32),'cuda')
            parts['learned']=candidate_parts(r,data,pred,selection['policy'],parts['Mu'])
        scores={k:metrics(r,v) for k,v in parts.items()}
        records.append({'id':r['id'],'dataset':r['dataset'],'coverage':coverage(r,data['pool']),
            'metrics':{k:{z:s[z] for z in KEYS} for k,s in scores.items()},'errors':{k:errors(r,v) for k,v in parts.items()}})
    summary={}
    for ds in sorted({r['dataset'] for r in records}):
        selected=[r for r in records if r['dataset']==ds];totals={k:sum(r['coverage'][k] for r in selected) for k in selected[0]['coverage']}
        summary[ds]={'records':len(selected),'candidate_coverage_pooled':totals,
            'coverage_strict':totals['0.05']/max(1,totals['reference_notes']), 'coverage_with_120ms_adjustment':totals['0.12']/max(1,totals['reference_notes']),
            'metrics':{m:{k:float(np.mean([r['metrics'][m][k] for r in selected])) for k in KEYS} for m in selected[0]['metrics']}}
    save(REPORT/'reviews'/((a.run or 'coverage')+'.json'),{'validation_only':True,'summary':summary,'tracks':records,
        'coverage_limit':'All-program candidate pitch/onset retrieval, assuming perfect role identification, not an achievable accuracy claim or precision. Timing-adjusted coverage ignores duration constraints.'})
    for ds,r in summary.items():print('REVIEW',ds,'coverage',round(r['coverage_strict'],3),round(r['coverage_with_120ms_adjustment'],3),'F1',r['metrics'].get('learned',{}),flush=True)

if __name__=='__main__':main()
