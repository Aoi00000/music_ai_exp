"""Measure Mu note recognition on labelled jazz; no claim of genre weakness."""
from pathlib import Path
import json,time,torch
from muscriptor import TranscriptionModel
from band2tab.stem_refiner import read_candidates
from scripts.a_mu_conditioning import GROUPS,polyphony_counts
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate


def main():
    torch.set_num_threads(2);rows=json.load(open('data/jazz_guitar_v1/manifest.json'));folder=Path('reports/jazz_guitar_v1');weight=Path('models/muscriptor-medium/model.safetensors');digest=sha(weight)
    model=TranscriptionModel.load_model(weight,device='cpu',dtype='float32');tracks=[];summaries={}
    for row in rows:
        for tag,groups in [('unconditioned',None),('conditioned',GROUPS['guitar'])]:
            out=folder/'midi'/tag/(row['id']+'.mid');meta=out.with_suffix('.json');audio=row['inputs']['guitar'];fp=dict(audio_sha256=sha(audio),model_sha256=digest,instruments=groups,tempo_quantization=False)
            if meta.exists():
                if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale jazz Mu output')
            else:
                start=time.monotonic();mid,_=model.transcribe_and_postprocess(audio,instruments=groups,detect_tempo=False);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(mid);save(meta,dict(fingerprint=fp,seconds=time.monotonic()-start))
            ns=read_candidates(out,'guitar',row['duration']);parts={'guitar':[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns],'bass':[]}
            tracks.append(dict(id=row['id'],split=row['split'],method=tag,parts=parts,metrics=metrics(row,parts),polyphony=polyphony_counts(row,parts)))
        print('JAZZ_MU_LABELLED',row['id'],flush=True)
    for split in ('train','val'):
        for tag in ('unconditioned','conditioned'):
            pairs=[(r,next(t['metrics'] for t in tracks if t['id']==r['id'] and t['method']==tag)) for r in rows if r['split']==split];summaries[split+'_'+tag]=aggregate(pairs)
    save(folder/'mu_comparison.json',dict(summary=summaries,tracks=tracks,test_used=False,own_learning=False,warning='The jazz validation progression is held out only from this new correction training, not guaranteed unseen by older pretrained/model families.'))
    print('JAZZ_MU_COMPLETE',flush=True)


if __name__=='__main__':main()
