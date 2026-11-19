"""Frozen complementary proposal model for the targeted jazz experiment."""
from pathlib import Path
import json,time,torch
from band2tab.lora import restore
from scripts.a_candidate_data import save,sha
from scripts.a_yourmt3_predict import load_model,transcribe


def main():
    rows=json.load(open('data/jazz_guitar_v1/manifest.json'));adapter=Path('models/yourmt3_lora_augmentation_review.pt');torch.set_num_threads(2);model,info=load_model('cpu');state=torch.load(adapter,map_location='cpu',weights_only=True);restore(model,state);model.eval()
    for r in rows:
        audio=r['inputs']['guitar'];out=Path('reports/jazz_guitar_v1/midi/your')/(r['id']+'.mid');meta=out.with_suffix('.json');fp=dict(audio_sha256=sha(audio),adapter_sha256=sha(adapter),device='cpu float32')
        if meta.exists():
            if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale jazz proposal')
        else:
            start=time.monotonic();transcribe(model,audio,out,'cpu');save(meta,dict(fingerprint=fp,seconds=time.monotonic()-start))
        print('JAZZ_YOUR_READY',r['id'],flush=True)
    print('JAZZ_YOUR_COMPLETE',flush=True)


if __name__=='__main__':main()
