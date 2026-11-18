"""Train-only octave/time sanity checks, not a replacement for manual GT audit."""
from pathlib import Path
import json,hashlib
import numpy as np,librosa,soundfile as sf
from band2tab.data import slakh_items

selected=json.loads(Path('reports/temporal_v1/source_acquisition.json').read_text())['selected']
allowed={r['group'] for r in selected if r['split']=='train'}
rows=[r for r in slakh_items('data/raw/slakh_temporal','slakh_temporal') if r['group'] in allowed and r['part']=='bass']
rows.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest());results=[]
for r in rows[:6]:
    samples=[];info=sf.info(r['audio'])
    if max([n['offset'] for n in r['notes']]+[0.])>info.duration+.1:raise ValueError('MIDI extends beyond source audio')
    long=[n for n in r['notes'] if n['offset']-n['onset']>=.6 and 23<=n['pitch']<=64]
    for n in long[::max(1,len(long)//12)][:12]:
        if any(other is not n and other['onset']<n['offset'] and other['offset']>n['onset'] for other in r['notes']):continue
        with sf.SoundFile(r['audio']) as f:
            f.seek(round((n['onset']+.12)*info.samplerate));y=f.read(round(.4*info.samplerate),dtype='float32',always_2d=True).mean(1)
        y=librosa.resample(y,orig_sr=info.samplerate,target_sr=16000)
        if np.sqrt(np.mean(y*y))<1e-5:continue
        f0=librosa.yin(y,sr=16000,fmin=librosa.midi_to_hz(23),fmax=librosa.midi_to_hz(65),frame_length=4096,hop_length=512)
        error=float(np.median(librosa.hz_to_midi(f0))-n['pitch']);samples.append(error)
    results.append({'id':r['id'],'plugin':r['plugin_name'],'pitch_shift':r['bass_octave_semitones'],
        'checked_notes':len(samples),'median_semitone_error':float(np.median(samples)) if samples else None,
        'within_one_semitone':float(np.mean(np.abs(samples)<=1)) if samples else None})
    print('OCTAVE_QC',results[-1],flush=True)
Path('reports/temporal_v1/label_qc.json').write_text(json.dumps({'split':'train only','results':results,
    'scope':'YIN may itself make octave errors; no test-based transposition or selection; fixed published Slakh correction retained'},indent=2))
