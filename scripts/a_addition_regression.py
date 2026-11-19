"""Frozen addition models on the previously inspected 14-condition band set."""
from pathlib import Path
from collections import Counter
import json,time,argparse
import numpy as np,soundfile as sf,torch
from band2tab.lora import restore
from band2tab.audio import features
from band2tab.stem_refiner import read_candidates,rms_envelope
from band2tab.addition import proposals,addition_features,add_notes,predict_additions
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import metrics,aggregate
from scripts.a_yourmt3_predict import load_model,transcribe


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiments',nargs='+',default=['addition_v1','addition_balanced_v2'],choices=['addition_v1','addition_balanced_v2','addition_jazz_v3_control','addition_jazz_v3']);parser.add_argument('--out',type=Path,default=Path('reports/addition_regression_v1'));parser.add_argument('--reuse-proposals',action='store_true');a=parser.parse_args()
    rows=json.load(open('data/stem_refiner_holdout_v1/manifest.json'));folder=a.out;proposal_folder=Path('reports/addition_regression_v1');adapter=Path('models/yourmt3_lora_augmentation_review.pt')
    if (folder/'comparison.json').exists():raise ValueError('Regression is frozen; do not overwrite')
    torch.set_num_threads(2)
    if not a.reuse_proposals:
        model,info=load_model('cpu');s=torch.load(adapter,map_location='cpu',weights_only=True);restore(model,s);model.eval()
    for r in rows:
        audio=r['inputs']['guitar'];out=proposal_folder/'midi'/(r['id']+'.mid');meta=out.with_suffix('.json');fp=dict(audio_sha256=sha(audio),adapter_sha256=sha(adapter),device='cpu float32')
        if meta.exists():
            if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale regression proposal')
        else:
            if a.reuse_proposals:raise ValueError('Frozen proposals are missing')
            start=time.monotonic();transcribe(model,audio,out,'cpu');save(meta,dict(fingerprint=fp,seconds=time.monotonic()-start))
        print('ADDITION_REGRESSION_PROPOSED',r['id'],flush=True)
    if not a.reuse_proposals:del model
    selections={k:json.load(open(Path('reports')/k/'selection.json')) for k in a.experiments}
    states={k:torch.load(v['export'],map_location='cpu',weights_only=True) for k,v in selections.items() if not v['identity_selected']}
    results={k:[] for k in ['baseline','unfiltered_union',*states]};tracks=[]
    for r in rows:
        baseline={};base_notes={}
        for part in ('guitar','bass'):
            notes=read_candidates(Path('reports/stem_refiner_holdout_v1/midi')/r['id']/(part+'.mid'),part,r['duration']);base_notes[part]=notes
            baseline[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes]
        candidates=proposals(read_candidates(proposal_folder/'midi'/(r['id']+'.mid'),'guitar',r['duration']),baseline['guitar']);views={};env={}
        for p,path in r['inputs'].items():views[p]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);env[p]=rms_envelope(y,sr)
        x=addition_features(views,env,candidates,baseline['guitar'],r['duration']);variants={'baseline':baseline,'unfiltered_union':add_notes(baseline,candidates,np.ones(len(candidates)),0.)}
        for k,state in states.items():variants[k]=add_notes(baseline,candidates,predict_additions(state,x),state['threshold'])
        track=dict(id=r['id'],group=r['group'],dataset=r['dataset'],variants={})
        for k,parts in variants.items():
            for part in ('guitar','bass'):
                key=lambda n:(n['pitch'],n['onset'],n['offset'],n['velocity'])
                if Counter(map(key,baseline[part]))-Counter(map(key,parts[part])):raise ValueError('Original note changed')
            score=metrics(r,parts);results[k].append((r,score));track['variants'][k]=dict(metrics=score,parts=parts)
        tracks.append(track)
    save(folder/'comparison.json',dict(summary={k:aggregate(v) for k,v in results.items()},tracks=tracks,previously_inspected_regression=True,test_used_for_selection=False,
        selection_hashes={k:sha(Path('reports')/k/'selection.json') for k in selections},original_notes_preserved=True,
        warning='These GOAT performances were inspected in a previous experiment. This is a frozen regression check, not a fresh held-out test.'))
    print('ADDITION_REGRESSION_COMPLETE',flush=True)


if __name__=='__main__':main()
