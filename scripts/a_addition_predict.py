"""Frozen adapted YourMT3 proposals for missing guitar notes; CPU only."""
from pathlib import Path
import json,time,torch
from band2tab.lora import restore
from scripts.a_stress_data import local
from scripts.a_candidate_data import sha,save
from scripts.a_yourmt3_predict import load_model,transcribe


def main():
    root=Path('backups/azure').resolve();manifest=root/'data/stem_refiner_v1/manifest.json';rows=[r for r in json.load(open(manifest)) if r['dataset'].startswith(('stem_band_','band_stress_'))]
    if any(r['split'] not in ('train','val') for r in rows):raise ValueError('No final test during proposal creation')
    adapter=Path('models/yourmt3_lora_augmentation_review.pt');torch.set_num_threads(2);model,info=load_model('cpu');state=torch.load(adapter,map_location='cpu',weights_only=True)
    if state['base_sha256']!=info['checkpoint_sha256']:raise ValueError('Proposal adapter/base mismatch')
    restore(model,state);model.eval()
    for r in rows:
        audio=local(root,r['inputs']['guitar']);out=Path('reports/addition_v1/midi')/(r['id']+'.mid');meta=out.with_suffix('.json')
        fp=dict(audio_sha256=sha(audio),adapter_sha256=sha(adapter),base_sha256=info['checkpoint_sha256'],manifest_sha256=sha(manifest),device='cpu float32')
        if meta.exists():
            if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale guitar proposals')
        else:
            start=time.monotonic();transcribe(model,audio,out,'cpu');save(meta,dict(fingerprint=fp,seconds=time.monotonic()-start))
        print('GUITAR_PROPOSALS_READY',r['id'],flush=True)
    print('GUITAR_PROPOSALS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
