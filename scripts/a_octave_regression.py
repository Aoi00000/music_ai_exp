"""Frozen octave-pruning/addition composition on development and known regression."""
from pathlib import Path
import json
import numpy as np,soundfile as sf,torch
from band2tab.audio import features,PITCH_MIN,PITCH_MAX
from band2tab.stem_refiner import read_candidates,rms_envelope,feature_matrix
from band2tab.octave import predict_octaves,prune_octaves
from band2tab.addition import proposals,addition_features,predict_additions,add_notes
from scripts.a_candidate_data import sha,save
from scripts.a_candidate_train import metrics,aggregate
from scripts.a_release_train import load


def main():
    folder=Path('reports/octave_regression_v1')
    if folder.exists():raise ValueError('Do not retune this frozen comparison')
    selection=Path('reports/octave_v1/selection.json');chosen=json.load(open(selection))['best']
    if chosen is None:raise ValueError('No eligible octave learner')
    state=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=True);addition_selection=Path('reports/addition_jazz_v3/selection.json');ap=json.load(open(addition_selection));addition=torch.load(ap['export'],map_location='cpu',weights_only=True)
    save(folder/'protocol.json',dict(octave_selection_sha256=sha(selection),addition_selection_sha256=sha(addition_selection),
        composition='Both fixed before this comparison. Classifier features use unchanged Mu baseline. Prune only allowed octave neighbours; add only independently selected missing candidates.',
        scope='16 development conditions and 14 previously inspected regression conditions; not a fresh test',test_opened=False))
    torch.set_num_threads(2);_,rows,cache,metadata=load();summaries={};all_tracks={}
    for split,rs in [('development',[r for r in rows if r['split']=='val']),('known_regression',json.load(open('data/stem_refiner_holdout_v1/manifest.json')))]:
        scores={k:[] for k in ('baseline','octave_only','addition_only','combined')};tracks=[]
        for r in rs:
            if split=='development':
                meta=metadata[r['id']];x=cache[r['id']]['x'];pool=meta['pool'];baseline=meta['baseline_parts'];outside=meta['unmodelled']
                add_meta=json.load(open(Path('data/addition_v1/cache')/(r['id']+'.json')));newpool=add_meta['pool']
                with np.load(Path('data/addition_v1/cache')/(r['id']+'.npz')) as f:ax=f['x'].copy()
            else:
                pool=[];outside={p:[] for p in ('guitar','bass')};baseline={}
                for part in ('guitar','bass'):
                    ns=read_candidates(Path('reports/stem_refiner_holdout_v1/midi')/r['id']/(part+'.mid'),part,r['duration']);baseline[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns];pool+=[n for n in ns if PITCH_MIN<=n['pitch']<=PITCH_MAX];outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
                views={};env={}
                for p,path in r['inputs'].items():views[p]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);env[p]=rms_envelope(y,sr)
                x=feature_matrix(views,env,pool,r['duration']);newpool=proposals(read_candidates(Path('reports/addition_regression_v1/midi')/(r['id']+'.mid'),'guitar',r['duration']),baseline['guitar']);ax=addition_features(views,env,newpool,baseline['guitar'],r['duration'])
            probability,eligible=predict_octaves(state,x,pool);pruned=prune_octaves(pool,probability,eligible,chosen['threshold'],outside);addprob=predict_additions(addition,ax)
            variants=dict(baseline=baseline,octave_only=pruned,addition_only=add_notes(baseline,newpool,addprob,addition['threshold']),combined=add_notes(pruned,newpool,addprob,addition['threshold']))
            record=dict(id=r['id'],dataset=r['dataset'],variants={})
            for name,parts in variants.items():
                if parts['bass']!=baseline['bass'] and sorted(parts['bass'],key=lambda n:(n['onset'],n['pitch']))!=sorted(baseline['bass'],key=lambda n:(n['onset'],n['pitch'])):raise ValueError('Bass notes changed')
                s=metrics(r,parts);scores[name].append((r,s));record['variants'][name]=dict(metrics=s,parts=parts)
            tracks.append(record);print('OCTAVE_REGRESSION_MEASURED',split,r['id'],flush=True)
        summaries[split]={k:aggregate(v) for k,v in scores.items()};all_tracks[split]=tracks
    save(folder/'comparison.json',dict(summary=summaries,tracks=all_tracks,previously_inspected_regression=True,automatic_application_approved=False,test_used=False))
    print('OCTAVE_REGRESSION_COMPLETE',summaries,flush=True)


if __name__=='__main__':main()
