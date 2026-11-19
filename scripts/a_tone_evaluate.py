"""Same-device, same-67-case controls after paired tone training."""
from pathlib import Path
import json,gc,time
import torch
from band2tab.lora import restore
from band2tab.score import read_midi
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,bound,KEYS
from scripts.a_yourmt3_predict import load_model,transcribe


def main():
    torch.set_num_threads(2);protocol=json.load(open('reports/tone_v1/protocol.json'))
    choices={'parent':Path(protocol['initial_adapter'])}
    for tag,name in [('control','tone_control_r8_600'),('tone','tone_colour_r8_600')]:
        choices[tag]=Path(json.load(open(Path('runs/yourmt3_lora')/name/'completed.json'))['best']['checkpoint'])
    samples=[list(map(json.loads,(Path('runs/yourmt3_lora')/name/'samples.jsonl').read_text().splitlines())) for name in ('tone_control_r8_600','tone_colour_r8_600')]
    fields=('id','start_sample','original_frames','pitch_shift_semitones','time_scale','tokens')
    if len(samples[0])!=600 or any(tuple(a[k] for k in fields)!=tuple(b[k] for k in fields) for a,b in zip(*samples)):raise ValueError('Matched source/crop/token control failed')
    rows=[r for name in ('candidate_v2','goat_v1') for r in json.load(open(Path('data')/name/'manifest.json')) if r['split']=='val']
    if len(rows)!=67:raise ValueError('Expected common 67 development cases')
    result={};all_tracks={}
    for tag,path in choices.items():
        model,info=load_model('cuda');state=torch.load(path,map_location='cpu',weights_only=True)
        if state['base_sha256']!=info['checkpoint_sha256']:raise ValueError('Base/adapter mismatch')
        restore(model,state);model.eval();scores=[];tracks=[]
        for r in rows:
            audio=Path(r['inputs']['mix']);out=Path('reports/tone_v1/midi')/tag/(r['id']+'.mid');meta=out.with_suffix('.json')
            fp=dict(audio_sha256=sha(audio),adapter_sha256=sha(path),base_sha256=info['checkpoint_sha256'],device='cuda float16 autocast')
            if meta.exists():
                record=json.load(open(meta))
                if record['fingerprint']!=fp:raise ValueError('Stale tone evaluation')
            else:
                start=time.monotonic();transcribe(model,audio,out,'cuda');raw=read_midi(out);parts=bound({p:raw.get(p,[]) for p in ('guitar','bass')},r['duration'])
                score=metrics(r,parts);record=dict(fingerprint=fp,id=r['id'],group=r['group'],dataset=r['dataset'],parts=parts,metrics={k:score[k] for k in KEYS},seconds=time.monotonic()-start);save(meta,record)
            scores.append((r,record['metrics']));tracks.append(record);print('TONE_COMMON_VALIDATED',tag,r['id'],flush=True)
        result[tag]=dict(summary=aggregate(scores),adapter=str(path),adapter_sha256=sha(path));all_tracks[tag]=tracks
        del model;gc.collect();torch.cuda.empty_cache()
    save('reports/tone_v1/comparison.json',dict(results=result,tracks=all_tracks,matched_source_crop_and_tokens=True,cases=67,test_used_for_selection=False,device='Azure T4 CUDA float16; do not equate with Mac float32 numbers'))
    print('TONE_PAIRED_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
