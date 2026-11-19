"""Apply a frozen bass-only review to A's unchanged local Mu analysis."""
from pathlib import Path
from collections import Counter
import argparse,json,time
import numpy as np,soundfile as sf,torch
from band2tab.audio import features,PITCH_MIN,PITCH_MAX
from band2tab.stem_refiner import read_candidates,rms_envelope,feature_matrix
from band2tab.bass_keep import apply,SCHEMA
from scripts.a_candidate_data import save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--checkpoint',default=Path('models/bass_sound_keep_review.pt'),type=Path,
        help='Frozen sounding-aware review; original notes are always preserved')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--guitar-review',type=Path,help='Optional notes.json from the fixed guitar addition/octave review of this exact baseline')
    g.add_argument('--octave-filter',type=Path,help='Optional existing frozen guitar octave-only filter; no added notes');a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    base=json.load(open(a.analysis/'notes.json'));bundle=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if bundle['schema']!=SCHEMA or base.get('muscriptor_sha256')!=bundle['mu_sha256']:raise ValueError('This review requires the frozen Mu medium baseline, not small or edited output')
    baseline=base.get('raw_parts',base['parts']);pool=[];outside={p:[] for p in ('guitar','bass')}
    key=lambda n:(n['pitch'],n['onset'],n['offset'],n.get('velocity'))
    for part in ('guitar','bass'):
        ns=read_candidates(a.analysis/('raw_'+part+'.mid'),part,base['duration'])
        if Counter(map(key,ns))!=Counter(map(key,baseline[part])):raise ValueError('Raw MIDI and baseline JSON differ')
        pool += [n for n in ns if PITCH_MIN<=n['pitch']<=PITCH_MAX]
        outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
    torch.set_num_threads(2);started=time.monotonic();views={};envelopes={}
    for part in ('mix','guitar','bass'):
        path=a.analysis/'stems'/(part+'.wav');views[part]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);envelopes[part]=rms_envelope(y,sr)
    x=feature_matrix(views,envelopes,pool,base['duration']);suggested=apply(bundle,x,pool,outside,base['duration'])
    if Counter(map(key,suggested['guitar']))!=Counter(map(key,baseline['guitar'])):raise ValueError('Protected guitar changed')
    if Counter(map(key,suggested['bass']))-Counter(map(key,baseline['bass'])):raise ValueError('Pitch/time/role changed')
    if a.guitar_review:
        guitar=json.load(open(a.guitar_review))
        if guitar.get('source_analysis_sha256')!=sha(a.analysis/'notes.json') or 'suggested_parts' not in guitar:raise ValueError('Guitar review must come from the same original analysis')
        for part in ('guitar','bass'):
            if Counter(map(key,guitar.get('raw_parts',{}).get(part,[])))!=Counter(map(key,baseline[part])):raise ValueError('Guitar review has a different baseline')
        suggested['guitar']=guitar['suggested_parts']['guitar']
    if a.octave_filter:
        from band2tab.octave import predict_octaves,prune_octaves
        octave=torch.load(a.octave_filter,map_location='cpu',weights_only=True);q,eligible=predict_octaves(octave,x,pool)
        g=prune_octaves(pool,q,eligible,octave['threshold'],outside);suggested['guitar']=g['guitar']
    result={**base,'parts':baseline,'raw_parts':baseline,'suggested_parts':suggested,'requires_review':True,
        'automatic_application_approved':False,'corrector_applied':False,'review_model_sha256':sha(a.checkpoint),
        'source_analysis_sha256':sha(a.analysis/'notes.json'),'removed_bass_notes':len(baseline['bass'])-len(suggested['bass']),
        'guitar_unchanged':Counter(map(key,suggested['guitar']))==Counter(map(key,baseline['guitar'])),
        'guitar_review_sha256':sha(a.guitar_review) if a.guitar_review else None,
        'octave_filter_sha256':sha(a.octave_filter) if a.octave_filter else None,
        'bass_pitch_onset_offset_unchanged':True,'seconds_for_bass_review':time.monotonic()-started,
        'review_scope':'Learned bass deletion proposals only; keep original Mu in parts/raw_parts. Not calibrated probabilities or verified musical truth.'}
    save(a.out/'notes.json',result);print('BASS_KEEP_REVIEW_READY',a.out/'notes.json',result['removed_bass_notes'],result['seconds_for_bass_review'],flush=True)
if __name__=='__main__':main()
