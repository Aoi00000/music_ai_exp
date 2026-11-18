"""One frozen final comparison; new source holdout and old regression separated.

Never choose a model or threshold from this report. Further development after
opening it needs a new independent holdout (these cases become regression).
"""
from pathlib import Path
import json,time
import numpy as np,torch
from band2tab.lora import restore
from band2tab.score import read_midi
from band2tab.candidate import CandidateRanker,predict
from scripts.a_candidate_data import rows_for,save,sha
from scripts.a_candidate_train import metrics,aggregate,bound,baselines,load,candidate_parts,KEYS
from scripts.a_candidate_review import errors
from scripts.a_yourmt3_predict import load_model,transcribe

REPORT=Path('reports/yourmt3_final')

def main():
    selection=Path('reports/yourmt3_lora/selection.json')
    selected=json.loads(selection.read_text())
    if sha('data/candidate_v2/manifest.json')!=selected['test_manifest_sha256']:raise ValueError('Manifest changed after selection')
    rows=rows_for(True);torch.set_num_threads(2)
    if not rows:raise ValueError('Missing frozen test')
    model,info=load_model()
    for name in ('YourMT3','YourMT3_LoRA'):
        adapter_sha=None
        if name.endswith('_LoRA'):
            package=Path('models/yourmt3_lora_review.pt')
            if sha(package)!=selected['adapter_sha256']:raise ValueError('Adapter changed')
            state=torch.load(package,map_location='cpu',weights_only=True)
            if info['checkpoint_sha256']!=state['base_sha256']:raise ValueError('Wrong base')
            restore(model,state);model.eval();adapter_sha=sha(package)
        for r in rows:
            out=REPORT/'midi'/name/(r['id']+'.mid');meta=out.with_suffix('.json')
            fingerprint={'audio_sha256':sha(r['inputs']['mix']),'adapter_sha256':adapter_sha,
                'base_sha256':info['checkpoint_sha256'],'selection_sha256':sha(selection)}
            if out.exists() and meta.exists():
                if json.loads(meta.read_text())['fingerprint']!=fingerprint:raise ValueError('Stale final MIDI')
                continue
            started=time.monotonic();decode=transcribe(model,r['inputs']['mix'],out)
            raw=read_midi(out);parts=bound({p:raw.get(p,[]) for p in ('guitar','bass')},r['duration'])
            score=metrics(r,parts)
            save(meta,{'id':r['id'],'fingerprint':fingerprint,'seconds':time.monotonic()-started,
                'metrics':{k:score[k] for k in KEYS},'parts':parts,'errors':errors(r,parts),'decoding_errors':decode})
            print('FROZEN_FINAL',name,r['id'],round(score['note_f1'],4),flush=True)
    del model
    torch.cuda.empty_cache()
    mu_selection=json.loads(Path('reports/candidate_v2/selection.json').read_text());best=mu_selection['research_selected']
    if sha(best['checkpoint'])!=best['checkpoint_sha256']:raise ValueError('Mu correction changed')
    state=torch.load(best['checkpoint'],map_location='cpu',weights_only=True)
    corrector=CandidateRanker(context=state['context']).to('cuda');corrector.load_state_dict(state['model']);corrector.eval()
    tracks=[]
    for r in rows:
        parts=baselines(r);data=load(r)
        prob=predict(corrector,data['x'],data['prior'],np.array(state['mean'],np.float32),np.array(state['std'],np.float32),'cuda')
        parts['Mu_learned']=candidate_parts(r,data,prob,best['policy'],parts['Mu'])
        for name in ('YourMT3','YourMT3_LoRA'):
            parts[name]=json.loads((REPORT/'midi'/name/(r['id']+'.json')).read_text())['parts']
        tracks.append({'id':r['id'],'dataset':r['dataset'],'split':r['split'],'group':r['group'],
            'metrics':{m:{k:s[k] for k in KEYS} for m,v in parts.items() for s in [metrics(r,v)]},
            'errors':{m:errors(r,v) for m,v in parts.items()}})
    summary={}
    for split in ('test','regression'):
        rr=[r for r in rows if r['split']==split];tt=[r for r in tracks if r['split']==split]
        summary[split]={m:aggregate([(r,t['metrics'][m]) for r,t in zip(rr,tt)]) for m in tracks[0]['metrics']}
    save(REPORT/'comparison.json',{'selection':selected,'cases':len(rows),'counts':{s:sum(r['split']==s for r in rows) for s in ('test','regression')},
        'summary':summary,'tracks':tracks,'test_used_for_selection':False,'automatic_application_approved':False,
        'scope':'test: new source UUID Slakh only. regression: previously observed evaluation cases. Upstream pretraining overlap is not excluded. Natural band and practical TAB edit time remain unvalidated.',
        'practical_claim_supported':False})
    for split,result in summary.items():
        for name,s in result.items():print('FINAL_SUMMARY',split,name,{k:round(s[k],4) for k in KEYS},flush=True)

if __name__=='__main__':main()
