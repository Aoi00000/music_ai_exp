"""Raw MuScriptor control on GOAT validation, with bounded GPU memory."""
from pathlib import Path
import json,time,torch
from muscriptor import TranscriptionModel
from band2tab.score import read_midi
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,bound,KEYS

def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.4)
    rows=[r for r in json.load(open('data/goat_v1/manifest.json')) if r['split']=='val']
    model_path=Path('models/muscriptor-medium/model.safetensors');model_sha=sha(model_path)
    model=TranscriptionModel.load_model(model_path,device='cuda',dtype='float16');items=[]
    for r in rows:
        out=Path('reports/goat/mu_validation/midi')/(r['id']+'.mid');meta=out.with_suffix('.json')
        fp={'audio_sha256':sha(r['inputs']['mix']),'model_sha256':model_sha}
        if out.exists() and meta.exists():
            record=json.loads(meta.read_text())
            if record['fingerprint']!=fp:raise ValueError('Stale Mu validation')
        else:
            started=time.monotonic();mid,_=model.transcribe_and_postprocess(r['inputs']['mix'],detect_tempo=False)
            out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(mid);raw=read_midi(out)
            parts=bound({p:raw.get(p,[]) for p in ('guitar','bass')},r['duration']);score=metrics(r,parts)
            record={'id':r['id'],'parts':parts,'fingerprint':fp,'metrics':{k:score[k] for k in KEYS},'seconds':time.monotonic()-started}
            save(meta,record)
        items.append((r,record['metrics']));print('GOAT_MU_VAL',r['id'],round(record['metrics']['note_f1'],4),flush=True)
    save('reports/goat/mu_validation/comparison.json',{'cases':len(rows),'summary':aggregate(items),
        'validation_only':True,'test_used_for_selection':False,'model_sha256':model_sha,'device':'cuda float16'})

if __name__=='__main__':main()
