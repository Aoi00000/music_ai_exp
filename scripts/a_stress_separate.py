"""Estimated 4/6-source cleaning on native-bandwidth controlled validation."""
from pathlib import Path
import argparse,json,os,random,time
import numpy as np,soundfile as sf,torch
from band2tab.separation import resample,write_input
from scripts.a_candidate_data import save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--save-individual',action='store_true');a=p.parse_args();root=a.root.resolve()
    os.environ['TORCH_HOME']=str(root/'models/torch');torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3)
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    rows=json.load(open(root/'data/band_stress_v1/manifest.json'))
    weights={p.name:sha(p) for p in (root/'models/torch').rglob('*.th')}
    for route,name,keep in (('four_clean','htdemucs_ft',('bass','other')),('six_clean','htdemucs_6s',('bass','guitar'))):
        model=get_model(name).to('cuda').eval()
        for r in rows:
            source=root/r['audio'];out=root/'data/band_stress_v1/separated'/route/(r['id']+'.wav');meta=out.with_suffix('.json')
            fingerprint={'audio_sha256':sha(source),'model':name,'keep':list(keep),'cached_weight_sha256':weights,
                'shifts':1,'overlap':.25,'segment':7,'native_input':True}
            individual={part:root/'data/band_stress_v1/separated'/('six_'+part)/(r['id']+'.wav') for part in ('guitar','bass')} if a.save_individual and route=='six_clean' else {}
            cached=out.exists() and meta.exists()
            if cached:
                if json.load(open(meta))['fingerprint']!=fingerprint:raise ValueError('Stale separated validation')
                complete=True
                for part,path in individual.items():
                    if not path.exists() or not path.with_suffix('.json').exists():complete=False;continue
                    if json.load(open(path.with_suffix('.json')))['fingerprint']!={**fingerprint,'individual_source':part}:raise ValueError('Stale individual separation')
                if complete:continue
            seed=int(fingerprint['audio_sha256'][:8],16)%2**31;torch.manual_seed(seed);random.seed(seed);start=time.monotonic()
            y,sr=sf.read(source,dtype='float32',always_2d=True);y=resample(y,sr,model.samplerate)
            wav=torch.from_numpy(np.ascontiguousarray(y.T));reference=wav.mean(0);mean=reference.mean();std=reference.std().clamp_min(1e-8)
            with torch.inference_mode():parts=apply_model(model,((wav-mean)/std)[None],device='cuda',shifts=1,split=True,overlap=.25,segment=7)[0].cpu()*std+mean
            selected=[model.sources.index(p) for p in keep]
            if not cached:
                write_input(out,parts[selected].sum(0).numpy().T,model.samplerate)
                save(meta,{'fingerprint':fingerprint,'output_sha256':sha(out),'seconds':time.monotonic()-start,'sources':list(model.sources)})
            for part,path in individual.items():
                write_input(path,parts[model.sources.index(part)].numpy().T,model.samplerate)
                save(path.with_suffix('.json'),{'fingerprint':{**fingerprint,'individual_source':part},'output_sha256':sha(path),'seconds':time.monotonic()-start})
            print('STRESS_SEPARATED',route,r['id'],round(time.monotonic()-start,2),flush=True)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
