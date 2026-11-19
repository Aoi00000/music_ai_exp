"""Frozen separator/Mu predictions for training a residual note corrector."""
from pathlib import Path
import json,time,random,gc,torch
from band2tab.separation import separate_inputs
from band2tab.score import read_midi
from scripts.a_candidate_data import save,sha


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.4)
    import os
    os.environ['TORCH_HOME']=str(Path('models/torch').resolve())
    manifest=Path('data/stem_refiner_v1/manifest.json');rows=json.load(open(manifest))
    from demucs.pretrained import get_model
    model=get_model('htdemucs_6s').to('cuda').eval()
    for r in rows:
        if r['split']!='train' or r.get('known_part'):continue
        dest=Path(r['inputs']['mix']).parent;meta=dest/'separation.json';fingerprint={'audio_sha256':sha(r['audio']),'model':'htdemucs_6s','shifts':1,'overlap':.25,'segment':7}
        if meta.exists():
            if json.load(open(meta))['fingerprint']!=fingerprint:raise ValueError('Stale training separation')
            continue
        seed=int(fingerprint['audio_sha256'][:8],16)%2**31;torch.manual_seed(seed);random.seed(seed)
        result=separate_inputs(r['audio'],dest,model);save(meta,{'fingerprint':fingerprint,'separation':result,'stem_sha256':{p:sha(r['inputs'][p]) for p in ('guitar','bass')}})
        print('STEM_TRAIN_SEPARATED',r['id'],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    from muscriptor import TranscriptionModel
    weight=Path('models/muscriptor-medium/model.safetensors');weight_sha=sha(weight);model=TranscriptionModel.load_model(weight,device='cuda',dtype='float16')
    for r in rows:
        if r['split']!='train':continue
        for part in ('guitar','bass'):
            out=Path('reports/stem_refiner_v1/midi')/r['id']/(part+'.mid');meta=out.with_suffix('.json');fp={'audio_sha256':sha(r['inputs'][part]),'model_sha256':weight_sha,'tempo_quantization':False}
            if meta.exists():
                if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale training Mu MIDI')
                continue
            out.parent.mkdir(parents=True,exist_ok=True)
            if r.get('known_part') and part!=r['known_part']:
                save(meta,{'fingerprint':fp,'known_silent_source':True,'parts':{part:[]}});continue
            start=time.monotonic();mid,_=model.transcribe_and_postprocess(r['inputs'][part],detect_tempo=False);out.write_bytes(mid)
            save(meta,{'fingerprint':fp,'parts':read_midi(out,force_part=part),'seconds':time.monotonic()-start})
            print('STEM_TRAIN_MU',r['id'],part,flush=True)
    print('STEM_REFINER_PREDICTIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
