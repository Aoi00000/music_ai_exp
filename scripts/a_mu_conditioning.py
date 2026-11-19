"""Test native instrument conditioning, distinct from post-hoc role assignment."""
from pathlib import Path
import json,time
import numpy as np,torch
from muscriptor import TranscriptionModel
from band2tab.stem_refiner import read_candidates
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import metrics,aggregate
from scripts.a_stress_data import local

GROUPS={'guitar':['acoustic_guitar','clean_electric_guitar','distorted_electric_guitar'],
        'bass':['acoustic_bass','electric_bass','contrabass']}


def polyphony_counts(row,parts):
    start=row['guard_seconds'];stop=row['duration']-start;t=np.arange(start,stop,.02);truth=np.zeros((len(t),2,128),bool);pred=np.zeros_like(truth)
    for a,notes in ((truth,row['parts']),(pred,parts)):
        for j,p in enumerate(('guitar','bass')):
            for n in notes[p]:a[(t>=n['onset'])&(t<n['offset']),j,int(n['pitch'])]=True
    count=truth.sum((1,2));result={}
    for name,mask in [('one',count==1),('two',count==2),('three',count==3),('four_plus',count>=4)]:
        r,e=truth[mask],pred[mask];result[name]=dict(frames=int(mask.sum()),tp=int((r&e).sum()),fp=int((~r&e).sum()),fn=int((r&~e).sum()))
    return result


def main():
    torch.set_num_threads(4);root=Path('backups/azure');manifest=root/'data/stem_refiner_v1/manifest.json';rows=[r for r in json.load(open(manifest)) if r['split']=='val']
    folder=Path('reports/mu_conditioning_v1');weight=Path('models/muscriptor-medium/model.safetensors');digest=sha(weight)
    save(folder/'protocol.json',dict(model_sha256=digest,manifest_sha256=sha(manifest),groups=GROUPS,cfg_values=[1.,2.],
        baseline='Cached native Mac Mu, cfg=1, no native instrument condition; source-known non-drum parser on estimated stems',
        purpose='Measure native conditioning and polyphony; genre weakness is a hypothesis, not an established fact',test_used=False))
    model=TranscriptionModel.load_model(weight,device='cpu',dtype='float32');tracks=[];scores={'unconditioned':[],'conditioned_cfg1':[],'conditioned_cfg2':[]}
    for r in rows:
        baseline=json.load(open(root/'data/stem_refiner_v1/cache'/(r['id']+'.json')))['baseline_parts'];variants={'unconditioned':baseline}
        for cfg in (1.,2.):
            tag='conditioned_cfg'+str(int(cfg));parts={}
            for part in ('guitar','bass'):
                audio=local(root,r['inputs'][part]);out=folder/'midi'/tag/(r['id']+'_'+part+'.mid');meta=out.with_suffix('.json')
                fp=dict(audio_sha256=sha(audio),model_sha256=digest,instruments=GROUPS[part],cfg_coef=cfg,tempo_quantization=False,device='cpu float32')
                if meta.exists():
                    if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale conditioning result')
                else:
                    start=time.monotonic();midi,_=model.transcribe_and_postprocess(audio,instruments=GROUPS[part],cfg_coef=cfg,detect_tempo=False)
                    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(midi);save(meta,dict(fingerprint=fp,seconds=time.monotonic()-start))
                ns=read_candidates(out,part,r['duration']);parts[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns]
            variants[tag]=parts
        for tag,parts in variants.items():
            s=metrics(r,parts);scores[tag].append((r,s));tracks.append(dict(id=r['id'],dataset=r['dataset'],method=tag,metrics=s,parts=parts,polyphony_counts=polyphony_counts(r,parts)))
        print('NATIVE_MU_CONDITIONING_VALIDATED',r['id'],flush=True)
    poly={}
    for tag in scores:
        poly[tag]={}
        for bucket in ('one','two','three','four_plus'):
            counts={k:sum(t['polyphony_counts'][bucket][k] for t in tracks if t['method']==tag) for k in ('frames','tp','fp','fn')}
            counts['frame_f1']=2*counts['tp']/max(1,2*counts['tp']+counts['fp']+counts['fn']);poly[tag][bucket]=counts
    save(folder/'comparison.json',dict(summary={k:aggregate(v) for k,v in scores.items()},polyphony=poly,tracks=tracks,
        own_learning=False,pretrained_inference_configuration_only=True,test_used_for_selection=False,
        caveat='Polyphony bins describe known simultaneous note counts, not chord names or jazz genre. The 16 conditions are repeated development data.'))
    print('NATIVE_MU_CONDITIONING_COMPLETE',flush=True)


if __name__=='__main__':main()
