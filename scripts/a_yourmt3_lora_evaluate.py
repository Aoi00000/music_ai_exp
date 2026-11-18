"""Autoregressive full validation of a fixed monitor-selected LoRA checkpoint.

Test evaluation is deliberately separate and requires a frozen selection.
"""
from pathlib import Path
import argparse,json,torch
from band2tab.lora import restore
from band2tab.score import read_midi
from scripts.a_candidate_data import rows_for,sha,save
from scripts.a_candidate_train import aggregate,metrics,bound,KEYS
from scripts.a_candidate_review import errors
from scripts.a_yourmt3_predict import load_model,transcribe

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    folder=Path('runs/yourmt3_lora')/a.run
    selected=json.loads((folder/'completed.json').read_text())['best']
    if sha(selected['checkpoint'])!=selected['checkpoint_sha256']:raise ValueError('Adapter changed')
    state=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
    model,info=load_model()
    if info['checkpoint_sha256']!=state['base_sha256']:raise ValueError('Wrong base checkpoint')
    restore(model,state);model.eval()
    rows=[r for r in rows_for(False) if r['split']=='val'];records=[]
    target=folder/'full_validation'
    for r in rows:
        out=target/'midi'/(r['id']+'.mid');meta=out.with_suffix('.json')
        fingerprint={'audio_sha256':sha(r['inputs']['mix']),'adapter_sha256':sha(selected['checkpoint']),
            'base_sha256':info['checkpoint_sha256']}
        if out.exists() and meta.exists():
            if json.loads(meta.read_text())['fingerprint']!=fingerprint:raise ValueError('Stale full validation')
        else:
            transcribe(model,r['inputs']['mix'],out);save(meta,{'fingerprint':fingerprint})
        raw=read_midi(out);parts=bound({p:raw.get(p,[]) for p in ('guitar','bass')},r['duration'])
        score=metrics(r,parts)
        record={'id':r['id'],'dataset':r['dataset'],'metrics':{k:score[k] for k in KEYS},
                'errors':errors(r,parts),'parts':parts,'fingerprint':fingerprint}
        save(meta,record);records.append(record)
        print('LORA_FULL_VAL',a.run,r['id'],round(score['note_f1'],4),flush=True)
    report={'run':a.run,'selected':selected,'cases':len(rows),'validation_only':True,
        'summary':aggregate([(r,v['metrics']) for r,v in zip(rows,records)]),'tracks':records,
        'test_used_for_selection':False,'automatic_application_approved':False}
    save(target/'comparison.json',report)
    print('LORA_FULL_COMPLETE',a.run,{k:round(report['summary'][k],4) for k in (*KEYS,'score')},flush=True)

if __name__=='__main__':main()
