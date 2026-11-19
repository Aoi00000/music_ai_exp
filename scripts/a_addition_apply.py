"""Review learned missing-note suggestions on an existing local Mu analysis."""
from pathlib import Path
from collections import Counter
import argparse,json,time,gc
import numpy as np,soundfile as sf,torch
from band2tab.lora import restore
from band2tab.audio import features
from band2tab.stem_refiner import read_candidates,rms_envelope
from band2tab.addition import proposals,addition_features,predict_additions,add_notes
from band2tab.octave import predict_octaves,prune_octaves
from band2tab.stem_refiner import feature_matrix
from band2tab.audio import PITCH_MIN,PITCH_MAX
from scripts.a_candidate_data import save,sha
from scripts.a_yourmt3_predict import load_model,transcribe

PROPOSAL_SHA='60c100d4c55f6b816b5ef9ef291da91052498a93e1b8a9df68d47d720b3dad5b'


def main():
    p=argparse.ArgumentParser();p.add_argument('analysis',type=Path,help='Output directory from a_band_pipeline');p.add_argument('--out',required=True,type=Path);p.add_argument('--ranker',type=Path,default=Path('models/addition_jazz_v3_review.pt'));p.add_argument('--octave-filter',type=Path);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new review directory')
    source=a.analysis/'notes.json';base=json.load(open(source))
    if base.get('corrector_applied') and 'raw_parts' not in base:raise ValueError('Uncorrected Mu notes are required')
    adapter=Path('models/yourmt3_lora_augmentation_review.pt')
    if sha(adapter)!=PROPOSAL_SHA:raise ValueError('Proposal adapter must match the training protocol')
    state=torch.load(a.ranker,map_location='cpu',weights_only=True);torch.set_num_threads(2);started=time.monotonic()
    model,info=load_model('cpu');s=torch.load(adapter,map_location='cpu',weights_only=True)
    if s['base_sha256']!=info['checkpoint_sha256']:raise ValueError('Base/adapter mismatch')
    restore(model,s);model.eval();a.out.mkdir(parents=True)
    audio=a.analysis/'stems'/'guitar.wav';out=a.out/'proposal_guitar.mid';transcribe(model,audio,out,'cpu');del model;gc.collect()
    baseline=base.get('raw_parts',base['parts']);pool=proposals(read_candidates(out,'guitar',base['duration']),baseline['guitar']);views={};env={}
    for part in ('mix','guitar','bass'):
        path=a.analysis/'stems'/(part+'.wav');views[part]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);env[part]=rms_envelope(y,sr)
    x=addition_features(views,env,pool,baseline['guitar'],base['duration']);prob=predict_additions(state,x);working=baseline;removed_count=0
    if a.octave_filter:
        octave=torch.load(a.octave_filter,map_location='cpu',weights_only=True);original=[];outside={p:[] for p in ('guitar','bass')}
        for part in ('guitar','bass'):
            ns=read_candidates(a.analysis/('raw_'+part+'.mid'),part,base['duration'])
            key=lambda n:(n['pitch'],n['onset'],n['offset'],n.get('velocity'))
            if Counter(map(key,ns))!=Counter(map(key,baseline[part])):raise ValueError('Raw MIDI and baseline JSON differ; use the original analysis')
            original += [n for n in ns if PITCH_MIN<=n['pitch']<=PITCH_MAX];outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
        ox=feature_matrix(views,env,original,base['duration']);keep,eligible=predict_octaves(octave,ox,original);working=prune_octaves(original,keep,eligible,octave['threshold'],outside);removed_count=len(baseline['guitar'])-len(working['guitar'])
    suggested=add_notes(working,pool,prob,state['threshold'])
    for part in ('guitar','bass'):
        key=lambda n:(n['pitch'],n['onset'],n['offset'],n.get('velocity'))
        if (part=='bass' or not a.octave_filter) and Counter(map(key,baseline[part]))-Counter(map(key,suggested[part])):raise ValueError('A protected baseline note was changed')
    save(a.out/'notes.json',dict(parts=baseline,raw_parts=baseline,suggested_parts=suggested,requires_review=True,automatic_application_approved=False,
        duration=base['duration'],source_analysis=str(source.resolve()),source_analysis_sha256=sha(source),ranker_sha256=sha(a.ranker),proposal_adapter_sha256=PROPOSAL_SHA,
        proposal_count=len(pool),added_count=len(suggested['guitar'])-len(working['guitar']),removed_count=removed_count,octave_filter_sha256=sha(a.octave_filter) if a.octave_filter else None,seconds=time.monotonic()-started,
        policy='Original Mu notes preserved in parts/raw_parts. Separate suggestions may prune learned octave candidates only when requested. No automatic replacement or invented playing techniques.'))
    print('GUITAR_ADDITION_REVIEW_READY',a.out/'notes.json',flush=True)


if __name__=='__main__':main()
