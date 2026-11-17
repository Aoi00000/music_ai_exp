"""Train-only octave diagnostic; never infer a transposition from test scores."""
from pathlib import Path
from math import gcd
import argparse,json
import numpy as np
import librosa,soundfile as sf
from scipy.signal import resample_poly
from band2tab.filobass import records

p=argparse.ArgumentParser();p.add_argument('--root',default='data/raw/filobass/FiloBass ISMIR Publication')
p.add_argument('--out',default='reports/filobass_train_pitch_qc.json');a=p.parse_args()
result=[]
for it in [r for r in records(a.root) if r['split']=='train'][:3]:
    with sf.SoundFile(it['audio']) as f:sr=f.samplerate;y=f.read(sr*60,dtype='float32',always_2d=True).mean(1)
    d=gcd(sr,16000);y=resample_poly(y,16000//d,sr//d)
    pitch=librosa.hz_to_midi(librosa.yin(y,fmin=30,fmax=400,sr=16000,frame_length=2048,hop_length=256));deltas=[]
    for n in it['notes']:
        if n['offset']-n['onset']<.3 or n['offset']>len(y)/16000:continue
        start=round((n['onset']+.08)*16000/256);end=round((n['offset']-.04)*16000/256)
        if start<end:deltas.append(float(np.median(pitch[start:end])-n['pitch']))
    result.append({'id':it['id'],'source_programs':it['source_programs'],'midi_min':min(n['pitch'] for n in it['notes']),
                   'midi_max':max(n['pitch'] for n in it['notes']),'notes_checked':len(deltas),
                   'median_semitone_error_yin_minus_gt':float(np.median(deltas)),
                   'fraction_within_one_semitone':float(np.mean(np.abs(deltas)<=1.))})
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'training_sources_only':True,'method':'YIN on first 60s, interiors of notes >=300ms; diagnostic, not full annotation verification','semitone_shift_applied':0,'tracks':result},indent=2))
print('Saved train-only FiloBass pitch diagnostic:',out)
