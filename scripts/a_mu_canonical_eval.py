"""Matched-device greedy evaluation, selection then frozen reused-set review."""
from pathlib import Path
import argparse,json,gc,time,copy
import torch
from muscriptor import TranscriptionModel
from band2tab.mu_adapter import restore_mu
from band2tab.lora import disabled
from band2tab.stem_refiner import read_candidates
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import metrics,aggregate
ROOT=Path('reports/mu_canonical_v3')

def evaluate(trans,rows,name):
    items=[];tracks=[]
    for r in rows:
        parts={};finger={};started=time.monotonic()
        for part in ('guitar','bass'):
            out=ROOT/'midi'/name/r['id']/(part+'.mid');out.parent.mkdir(parents=True,exist_ok=True)
            mid,_=trans.transcribe_and_postprocess(r['inputs'][part],detect_tempo=False,cfg_coef=1.)
            out.write_bytes(mid);ns=read_candidates(out,part,r['duration']);parts[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns];finger[part]=sha(out)
        s=metrics(r,parts);items.append((r,s));tracks.append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=parts,metrics=s,midi_sha256=finger,seconds=time.monotonic()-started));print('MU_ADAPT_EVAL',name,r['id'],round(s['note_f1'],4),flush=True)
    return dict(summary=aggregate(items),tracks=tracks)

def monitor():
    if (ROOT/'selection.json').exists():raise ValueError('Selection frozen')
    torch.set_num_threads(2);rows=json.load(open('data/mu_canonical_v3/monitor.json'));trans=TranscriptionModel.load_model('models/muscriptor-medium/model.safetensors',device='cuda',dtype='float32');results={'baseline':evaluate(trans,rows,'monitor_baseline')}
    adapters=None
    for arm in ('gold','gold_tone'):
        state=torch.load(Path('runs/mu_canonical_v3')/arm/'final.pt',map_location='cpu',weights_only=True)
        if adapters is None:adapters=restore_mu(trans._model,state)
        else:
            with torch.no_grad():
                for n,m in adapters.items():m.a.copy_(state['adapters'][n]['a']);m.b.copy_(state['adapters'][n]['b'])
        trans._model.eval();results[arm]=evaluate(trans,rows,'monitor_'+arm)
    b=results['baseline']['summary'];base=.5*b['note_f1']+.3*b['frame_f1']+.2*b['active_exact_match'];choices=[]
    for arm in ('gold','gold_tone'):
        s=results[arm]['summary'];eligible=all(s['datasets'][d][k]>=v[k]-1e-10 for d,v in b['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'));score=.5*s['note_f1']+.3*s['frame_f1']+.2*s['active_exact_match'];choices.append(dict(arm=arm,eligible=eligible,score=score,summary=s))
    allowed=[v for v in choices if v['eligible'] and v['score']>base+1e-9];chosen=max(allowed,key=lambda x:x['score'])['arm'] if allowed else 'baseline'
    save(ROOT/'selection.json',dict(chosen=chosen,baseline=results['baseline']['summary'],choices=choices,tracks={k:v['tracks'] for k,v in results.items()},scope='5 internal conditions, source holdout of Mu adapter only; upstream overlap possible',test_used=False,same_device_dtype='CUDA float32 weights / float16 autocast',no_checkpoint_search=True))
    print('MU_ADAPT_SELECTED',chosen,flush=True)

def review():
    if (ROOT/'review.json').exists():raise ValueError('Review frozen')
    selected=json.load(open(ROOT/'selection.json'));chosen=selected['chosen'];rows=json.load(open('data/mu_canonical_v3/review_manifest.json'));torch.set_num_threads(2)
    trans=TranscriptionModel.load_model('models/muscriptor-medium/model.safetensors',device='cuda',dtype='float32');all_results={'baseline':evaluate(trans,rows,'review_baseline')}
    # Fixed hypotheses remain visible even if the strict gate rejected adoption.
    adapters=None
    for arm in ('gold','gold_tone'):
        state=torch.load(Path('runs/mu_canonical_v3')/arm/'final.pt',map_location='cpu',weights_only=True)
        if adapters is None:adapters=restore_mu(trans._model,state)
        else:
            with torch.no_grad():
                for n,m in adapters.items():m.a.copy_(state['adapters'][n]['a']);m.b.copy_(state['adapters'][n]['b'])
        trans._model.eval();all_results[arm]=evaluate(trans,rows,'review_'+arm)
    save(ROOT/'review.json',dict(chosen_before_review=chosen,results=all_results,test_used=False,scope='Repeated development16 only. No additional selection from these outcomes.',automatic_application_approved=False))
    print('MU_ADAPT_REVIEW_COMPLETE',chosen,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['monitor','review']);a=p.parse_args();globals()[a.command]()
