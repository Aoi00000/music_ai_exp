"""Confirm local CPU inference on the backed-up development validation cases."""
from pathlib import Path
import argparse,json,time,torch
from band2tab.lora import restore
from band2tab.score import read_midi
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,bound,KEYS
from scripts.a_yourmt3_predict import load_model,transcribe

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path);p.add_argument('--limit',type=int);a=p.parse_args()
    root=Path('backups/azure').resolve();manifest=root/'data/candidate_v2/manifest.json'
    rows=[r for r in json.loads(manifest.read_text()) if r['split']=='val']
    if a.limit:rows=rows[:a.limit]
    torch.set_num_threads(4);model,info=load_model('cpu');adapter_sha=None
    if a.adapter:
        state=torch.load(a.adapter,map_location='cpu',weights_only=True)
        if state['base_sha256']!=info['checkpoint_sha256']:raise ValueError('Adapter/base mismatch')
        restore(model,state);model.eval();adapter_sha=sha(a.adapter)
    folder=Path('reports/yourmt3/mac_validation')/('learned' if a.adapter else 'base');items=[];seconds=[]
    for r in rows:
        audio=root/('data/'+r['inputs']['mix'].split('/music_ai_exp/data/',1)[1])
        out=folder/'midi'/(r['id']+'.mid');meta=out.with_suffix('.json')
        fingerprint={'audio_sha256':sha(audio),'base_sha256':info['checkpoint_sha256'],'adapter_sha256':adapter_sha}
        if out.exists() and meta.exists():
            record=json.loads(meta.read_text())
            if record['fingerprint']!=fingerprint:raise ValueError('Stale Mac output')
        else:
            start=time.monotonic();decode=transcribe(model,audio,out,device='cpu');elapsed=time.monotonic()-start
            raw=read_midi(out);parts=bound({p:raw.get(p,[]) for p in ('guitar','bass')},r['duration']);score=metrics(r,parts)
            record={'id':r['id'],'dataset':r['dataset'],'seconds':elapsed,'duration':r['duration'],'fingerprint':fingerprint,
                'metrics':{k:score[k] for k in KEYS},'parts':parts,'decoding_errors':decode}
            save(meta,record)
        items.append((r,record['metrics']));seconds.append(record['seconds'])
        print('MAC_VALIDATION',r['id'],round(record['metrics']['note_f1'],4),round(record['seconds'],2),flush=True)
    save(folder/'comparison.json',{'device':'cpu','dtype':'float32','cases':len(rows),'validation_only':True,
        'summary':aggregate(items),'inference_seconds':sum(seconds),'audio_seconds':sum(r['duration'] for r in rows),
        'adapter_sha256':adapter_sha,'base':info,'manifest_sha256':sha(manifest),'test_used_for_selection':False})
    print('MAC_VALIDATION_COMPLETE',len(rows),sum(seconds),flush=True)

if __name__=='__main__':main()
