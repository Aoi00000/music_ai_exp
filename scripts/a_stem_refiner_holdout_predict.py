"""Native Mac frozen separation and Mu controls, once selection is locked."""
from pathlib import Path
import json,os,time,random,gc
import torch
from scripts.a_candidate_data import save,sha
from band2tab.separation import separate_inputs


def main():
    torch.set_num_threads(4);selection=Path('reports/stem_refiner_v1/selection.json');manifest=Path('data/stem_refiner_holdout_v1/manifest.json')
    protocol=json.load(open(manifest.parent/'protocol.json'));s=json.load(open(selection))
    if protocol['selection_sha256']!=sha(selection) or protocol['manifest_sha256']!=sha(manifest) or sha(s['export'])!=s['export_sha256']:raise ValueError('Frozen held-out protocol changed')
    rows=json.load(open(manifest));os.environ['TORCH_HOME']=str(Path('models/torch').resolve())
    from demucs.pretrained import get_model
    separator=get_model('htdemucs_6s').eval()
    for r in rows:
        folder=Path(r['audio']).parent;meta=folder/'separation.json';fp=dict(audio_sha256=sha(r['audio']),model='htdemucs_6s',device='cpu',shifts=1,segment=7,overlap=.25)
        if meta.exists():
            if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale held-out stems')
        else:
            seed=int(fp['audio_sha256'][:8],16)%2**31;torch.manual_seed(seed);random.seed(seed);started=time.monotonic()
            info=separate_inputs(r['audio'],folder,separator,'cpu');save(meta,dict(fingerprint=fp,separation=info,seconds=time.monotonic()-started,stem_sha256={p:sha(r['inputs'][p]) for p in ('guitar','bass')}))
        print('HELDOUT_MAC_SEPARATED',r['id'],flush=True)
    del separator;gc.collect()
    from muscriptor import TranscriptionModel
    weight=Path('models/muscriptor-medium/model.safetensors');weight_sha=sha(weight);model=TranscriptionModel.load_model(weight,device='cpu',dtype='float32')
    for r in rows:
        for part in ('guitar','bass'):
            audio=r['inputs'][part];out=Path('reports/stem_refiner_holdout_v1/midi')/r['id']/(part+'.mid');meta=out.with_suffix('.json')
            fp=dict(audio_sha256=sha(audio),model_sha256=weight_sha,selection_sha256=sha(selection),tempo_quantization=False,device_dtype='cpu float32')
            if meta.exists():
                if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale held-out Mu predictions')
            else:
                started=time.monotonic();midi,_=model.transcribe_and_postprocess(audio,detect_tempo=False);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(midi)
                save(meta,dict(fingerprint=fp,seconds=time.monotonic()-started))
            print('HELDOUT_MAC_MU',r['id'],part,flush=True)
    print('HELDOUT_PREDICTIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
