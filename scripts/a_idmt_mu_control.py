"""Strong frozen Mu controls on the previously inspected bass regression."""
from pathlib import Path
import json,time,torch
from muscriptor import TranscriptionModel
from band2tab.score import read_midi
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,bound


def main():
    torch.set_num_threads(4);folder=Path('reports/idmt_bass_lines/mu_control');manifest=Path('reports/idmt_bass_lines/manifest.json')
    acquisition=json.load(open(manifest.parent/'acquisition.json'))
    if sha(manifest)!=acquisition['manifest_sha256'] or any(sha(r['path'])!=r['sha256'] for r in acquisition['files']):raise ValueError('Bass evaluation source changed')
    rows=json.load(open(manifest));weight=Path('models/muscriptor-medium/model.safetensors');weight_sha=sha(weight)
    model=TranscriptionModel.load_model(weight,device='cpu',dtype='float32');tracks=[];scores={'strict':[],'source_known':[]}
    for r in rows:
        audio=Path(r['audio']);out=folder/'midi'/(r['id']+'.mid');meta=out.with_suffix('.json')
        fp=dict(audio_sha256=sha(audio),model_sha256=weight_sha,manifest_sha256=sha(manifest),device_dtype='cpu float32',tempo_quantization=False)
        if meta.exists():
            rec=json.load(open(meta))
            if rec['fingerprint']!=fp:raise ValueError('Stale Mu regression')
        else:
            start=time.monotonic();midi,_=model.transcribe_and_postprocess(audio,detect_tempo=False);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(midi)
            rec=dict(fingerprint=fp,seconds=time.monotonic()-start);save(meta,rec)
        policies={'strict':read_midi(out),'source_known':read_midi(out,force_part='bass')}
        track=dict(id=r['id'])
        for policy,parts in policies.items():
            parts=bound({p:parts.get(p,[]) for p in ('guitar','bass')},r['duration']);score=metrics(r,parts);scores[policy].append((r,score));track[policy]=dict(parts=parts,metrics=score)
        tracks.append(track);print('MU_SEEN_BASS_REGRESSION',r['id'],flush=True)
    save(folder/'comparison.json',dict(summary={p:aggregate(items) for p,items in scores.items()},tracks=tracks,
        previously_inspected_regression=True,test_used_for_selection=False,model_sha256=weight_sha,
        note='Known solo bass: source_known uses dataset-level instrument identity; strict uses predicted GM roles. Do not compare different role policies as a pure model gain.'))
    print('MU_SEEN_BASS_COMPLETE',flush=True)


if __name__=='__main__':main()
