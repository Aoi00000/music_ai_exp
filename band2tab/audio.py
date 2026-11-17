from pathlib import Path
import subprocess
import numpy as np
import librosa
SR=22050
HOP=256
N_BINS=264
PITCH_MIN=23
PITCH_MAX=88
N_PITCHES=PITCH_MAX-PITCH_MIN+1

def convert(src,dest,seconds=None):
    import imageio_ffmpeg
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-y','-i',str(src),'-vn']
    if seconds:cmd+=['-t',str(seconds)]
    # Separation engines receive stereo audio; each transcriber resamples itself.
    cmd+=['-ac','2','-ar','44100',str(dest)]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    return dest

def features(path):
    y,_=librosa.load(str(path),sr=SR,mono=True)
    # A hop is a sampling interval, not the true low-frequency time resolution.
    c=librosa.cqt(y,sr=SR,hop_length=HOP,fmin=librosa.midi_to_hz(PITCH_MIN),
                  n_bins=N_BINS,bins_per_octave=36)
    x=np.log1p(10*np.abs(c)).astype('float32')
    return x.T

def note_targets(notes,frames):
    out=np.zeros((frames,3,N_PITCHES),np.float32); outside=0
    for n in notes:
        p=int(round(n['pitch']))-PITCH_MIN
        if not 0<=p<N_PITCHES:outside+=1;continue
        a=max(0,round(float(n['onset'])*SR/HOP));b=min(frames,round(float(n['offset'])*SR/HOP))
        if a>=frames or b<=a:continue
        out[a,0,p]=1;out[a:b,1,p]=1;out[max(a,b-1),2,p]=1
    return out,outside

def decode(prob,onset=.5,frame=.5,offset=.5,min_duration=.03):
    """Onset peaks open notes; the next peak/frame-off/offset closes them."""
    notes=[];dt=HOP/SR;length=len(prob)
    if not length:return notes
    for pi in range(prob.shape[-1]):
        o=prob[:,0,pi]
        peaks=np.flatnonzero((o>=onset)&np.r_[True,o[1:]>o[:-1]]&np.r_[o[:-1]>=o[1:],True])
        stops=np.flatnonzero((prob[:,1,pi]<frame)|(prob[:,2,pi]>=offset))
        for j,start in enumerate(peaks):
            stop_index=np.searchsorted(stops,start,side='right')
            end=min(peaks[j+1] if j+1<len(peaks) else length,
                    stops[stop_index] if stop_index<len(stops) else length)
            if (end-start)*dt>=min_duration:
                notes.append({'pitch':pi+PITCH_MIN,'onset':start*dt,'offset':end*dt,
                              'confidence':float(prob[start:end,1,pi].mean()),'velocity':80})
    return sorted(notes,key=lambda n:(n['onset'],n['pitch']))
