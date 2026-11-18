"""Fixed six-source separation, retaining the original for transcription evidence."""
from pathlib import Path
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def resample(y,source_rate,target_rate):
    if source_rate==target_rate:return y
    d=gcd(source_rate,target_rate)
    return resample_poly(y,target_rate//d,source_rate//d,axis=0)

def write_input(path,y,sr):
    """All transcription routes share mono 16 kHz, float WAV and no peak scaling."""
    y=np.asarray(y,dtype=np.float32)
    if y.ndim==2:y=y.mean(axis=1)
    y=resample(y,sr,16000)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,y,16000,subtype='FLOAT')
    return path

def separate_inputs(audio,dest,model,device='cuda'):
    import torch
    from demucs.apply import apply_model
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    y,sr=sf.read(audio,dtype='float32',always_2d=True)
    if y.shape[1] not in (1,2):raise ValueError('Only mono/stereo inputs supported')
    if Path(audio).resolve()!=(dest/'mix.wav').resolve():
        write_input(dest/'mix.wav',y,sr)
    elif sr!=16000 or y.shape[1]!=1:
        raise ValueError('An in-place mix input must already be canonical mono 16kHz')
    if y.shape[1]==1:y=np.repeat(y,2,axis=1)
    y=resample(y,sr,model.samplerate)
    wav=torch.from_numpy(np.ascontiguousarray(y.T));ref=wav.mean(0)
    mean=ref.mean();std=ref.std().clamp_min(1e-8)
    # Mean is zero for the synthetic/recorded excerpts to numerical tolerance.
    # Match the official Demucs inference normalization/restoration per source.
    with torch.no_grad():
        stems=apply_model(model,((wav-mean)/std)[None],device=device,
                          shifts=1,split=True,overlap=.25,segment=7)[0].cpu()*std+mean
    keep=[model.sources.index(p) for p in model.sources if p not in ('drums','vocals')]
    write_input(dest/'no_drums_vocals.wav',stems[keep].sum(0).numpy().T,model.samplerate)
    for part in ('guitar','bass'):
        write_input(dest/(part+'.wav'),stems[model.sources.index(part)].numpy().T,model.samplerate)
    return {'sources':list(model.sources),'input_mean':float(mean),'input_std':float(std),
            'frames':len(y),'sample_rate':model.samplerate}
