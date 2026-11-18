"""Local supplemental validation of interference and estimated separation."""
from pathlib import Path
import argparse,json,time,torch
from band2tab.lora import restore
from band2tab.score import read_midi
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,bound,KEYS
from scripts.a_candidate_review import errors
from scripts.a_yourmt3_predict import load_model,transcribe

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path);p.add_argument('--tag',required=True)
    p.add_argument('--routes',nargs='+',choices=['raw','four_clean','six_clean','six_parts'],default=['raw']);a=p.parse_args()
    root=Path('backups/azure').resolve();manifest=root/'data/band_stress_v1/manifest.json';rows=json.load(open(manifest))
    if any(r['split']!='val' for r in rows):raise ValueError('Only supplemental validation is allowed')
    torch.set_num_threads(4);model,info=load_model('cpu');adapter_sha=None
    if a.adapter:
        state=torch.load(a.adapter,map_location='cpu',weights_only=True)
        if state['base_sha256']!=info['checkpoint_sha256']:raise ValueError('Adapter/base mismatch')
        restore(model,state);model.eval();adapter_sha=sha(a.adapter)
    summary={};records=[]
    for route in a.routes:
        items=[]
        for r in rows:
            stems={part:root/'data/band_stress_v1/separated'/('six_'+part)/(r['id']+'.wav') for part in ('guitar','bass')} if route=='six_parts' else None
            audio=root/r['inputs']['mix'] if route=='raw' else root/'data/band_stress_v1/separated'/route/(r['id']+'.wav')
            out=Path('reports/band_stress_v1')/a.tag/route/'midi'/(r['id']+'.mid');meta=out.with_suffix('.json')
            fingerprint={'audio_sha256':{part:sha(path) for part,path in stems.items()} if stems else sha(audio),'base_sha256':info['checkpoint_sha256'],'adapter_sha256':adapter_sha,'manifest_sha256':sha(manifest)}
            if meta.exists() and (out.exists() or stems):
                record=json.load(open(meta))
                if record['fingerprint']!=fingerprint:raise ValueError('Stale stress evaluation')
            else:
                started=time.monotonic()
                if stems:
                    raw={};decode={}
                    for part,path in stems.items():
                        stem_out=out.parent/(r['id']+'_'+part+'.mid')
                        decode[part]=transcribe(model,path,stem_out,'cpu')
                        raw[part]=read_midi(stem_out).get(part,[])
                else:
                    decode=transcribe(model,audio,out,'cpu');raw=read_midi(out)
                parts=bound({p:raw.get(p,[]) for p in ('guitar','bass')},r['duration']);s=metrics(r,parts)
                record={'id':r['id'],'dataset':r['dataset'],'route':route,'parts':parts,'metrics':{k:s[k] for k in KEYS},
                    'errors':errors(r,parts),'fingerprint':fingerprint,'seconds':time.monotonic()-started,'decoding_errors':decode}
                save(meta,record)
            items.append((r,record['metrics']));records.append(record)
            print('STRESS_VALIDATION',a.tag,route,r['id'],round(record['metrics']['note_f1'],4),flush=True)
        summary[route]=aggregate(items)
    save(Path('reports/band_stress_v1')/a.tag/'comparison.json',{'validation_only':True,'natural_band_recording':False,
        'summary':summary,'tracks':records,'adapter_sha256':adapter_sha,'test_used_for_selection':False,
        'scope':'Constructed unrelated guitar/bass performances plus real vocal/drum stems; 16 conditions share 8 guitar sources'})
    print('STRESS_VALIDATION_COMPLETE',a.tag,flush=True)

if __name__=='__main__':main()
