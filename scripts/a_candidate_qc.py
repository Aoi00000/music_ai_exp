"""Training-only sample-based pitch-label audit; never changes MIDI or tests."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np,librosa,soundfile as sf
from band2tab.data import slakh_items
from scripts.a_candidate_data import REPORT,save

def main():
    records=json.loads((REPORT/'source_acquisition.json').read_text())['selected']
    allowed={r['group'] for r in records if r['split']=='train'};by=defaultdict(list)
    for r in slakh_items('data/raw/slakh_candidate_v2','c2qc'):
        if r['group'] in allowed and r['part']=='bass':by[r['plugin_name']].append(r)
    result=[]
    for plugin,rows in sorted(by.items()):
        rows.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest());errors=[];sources=[];checks=[]
        for r in rows[:2]:
            info=sf.info(r['audio']);candidates=[n for n in r['notes'] if n['offset']-n['onset']>=.4 and 23<=n['pitch']<=60]
            valid=[n for n in candidates if not any(b is not n and max(n['onset'],b['onset'])<min(n['offset'],b['offset']) for b in r['notes'])]
            for n in valid[::max(1,len(valid)//6)][:6]:
                with sf.SoundFile(r['audio']) as f:
                    f.seek(round((n['onset']+.08)*info.samplerate));y=f.read(round(.25*info.samplerate),dtype='float32',always_2d=True).mean(1)
                y=librosa.resample(y,orig_sr=info.samplerate,target_sr=16000)
                if np.sqrt(np.mean(y*y))<1e-5:continue
                f0=librosa.yin(y,sr=16000,fmin=librosa.midi_to_hz(23),fmax=librosa.midi_to_hz(65),frame_length=2048,hop_length=256)
                error=float(np.median(librosa.hz_to_midi(f0))-n['pitch']);errors.append(error);sources.append(r['id'])
                spectrum=np.abs(np.fft.rfft(y*np.hanning(len(y))));freq=np.fft.rfftfreq(len(y),1/16000);hz=float(librosa.midi_to_hz(n['pitch']));inside=(freq>=25)&(freq<1000)
                local=lambda f:float(spectrum[np.abs(freq-f)<=3].max())
                checks.append({'source':r['id'],'onset':n['onset'],'pitch':n['pitch'],'yin_error':error,
                    'rms':float(np.sqrt(np.mean(y*y))),'expected_hz':hz,
                    'peak_hz':float(freq[np.where(inside)[0][np.argmax(spectrum[inside])]]),
                    'expected_peak':local(hz),'half_frequency_peak':local(hz/2)})
        d={'plugin':plugin,'sampled_sources':sorted(set(sources)),'notes':len(errors),
           'errors_semitones':errors,'median_error':float(np.median(errors)) if errors else None,
           'within_one_semitone':float(np.mean(np.abs(errors)<=1)) if errors else None,'checks':checks}
        result.append(d);print('TRAIN_LABEL_QC',plugin,'notes',len(errors),'median',d['median_error'],flush=True)
    save(REPORT/'train_label_qc.json',{'split':'train only','results':result,
        'limits':'A sampled YIN check can itself have octave errors and does not validate all annotations. No MIDI transposition or test-based filtering.'})

if __name__=='__main__':main()
