"""Frozen MuScriptor controls for raw and individually separated validation."""
from pathlib import Path
import argparse,json,time,torch
from muscriptor import TranscriptionModel
from band2tab.score import read_midi
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,bound,KEYS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();root=a.root.resolve()
    torch.set_num_threads(1 if a.device=='cuda' else 4)
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.4)
    manifest=root/'data/band_stress_v1/manifest.json';rows=json.load(open(manifest))
    if any(r['split']!='val' for r in rows):raise ValueError('Development validation only')
    weight=Path('models/muscriptor-medium/model.safetensors');weight_sha=sha(weight)
    dtype='float16' if a.device=='cuda' else 'float32'
    model=TranscriptionModel.load_model(weight,device=a.device,dtype=dtype);summary={};tracks=[]
    folder=Path('reports/band_hybrid_v1')/('mu' if a.device=='cuda' else 'mu_mac')
    for route in ('raw','six_parts'):
        items=[]
        for row in rows:
            paths={'mix':root/row['inputs']['mix']} if route=='raw' else {p:root/'data/band_stress_v1/separated'/('six_'+p)/(row['id']+'.wav') for p in ('guitar','bass')}
            variants={'strict':{},'source_known':{}}
            for part,audio in paths.items():
                out=folder/route/(row['id']+'_'+part+'.mid');meta=out.with_suffix('.json')
                fp={'audio_sha256':sha(audio),'model_sha256':weight_sha,'manifest_sha256':sha(manifest),'tempo_quantization':False}
                if a.device!='cuda':fp['device_dtype']=a.device+' '+dtype
                if out.exists() and meta.exists():
                    record=json.load(open(meta))
                    if record['fingerprint']!=fp:raise ValueError('Stale Mu hybrid control')
                else:
                    start=time.monotonic();mid,_=model.transcribe_and_postprocess(audio,detect_tempo=False)
                    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(mid)
                    record={'fingerprint':fp,'parts':read_midi(out),'seconds':time.monotonic()-start}
                    if part!='mix':record['source_known_notes']=read_midi(out,force_part=part).get(part,[])
                    save(meta,record)
                if part=='mix':
                    variants['strict']={p:record['parts'].get(p,[]) for p in ('guitar','bass')};variants.pop('source_known')
                else:
                    variants['strict'][part]=record['parts'].get(part,[])
                    variants['source_known'][part]=record['source_known_notes']
            for policy,parts in variants.items():
                parts=bound(parts,row['duration']);score=metrics(row,parts)
                result={'id':row['id'],'dataset':row['dataset'],'route':route,'policy':policy,'parts':parts,'metrics':{k:score[k] for k in KEYS}}
                tracks.append(result);items.append((row,policy,score))
            print('MU_STRESS_VALIDATED',route,row['id'],flush=True)
        summary[route]={policy:aggregate([(r,s) for r,p,s in items if p==policy]) for policy in sorted({p for r,p,s in items})}
    save(folder/'comparison.json',{'summary':summary,'tracks':tracks,'model_sha256':weight_sha,'validation_only':True,'natural_band_recording':False,'test_used_for_selection':False,'device':a.device+' '+dtype})
    print('MU_STRESS_COMPLETE',flush=True)


if __name__=='__main__':main()
