"""Timing-explicit training augmentations for labelled isolated instruments."""
import numpy as np


def soft_drive(audio,drive=6.,target_rms=.1):
    """Memoryless saturation: harmonics change, sample positions do not move."""
    audio=np.asarray(audio,dtype=np.float32)
    rms=max(1e-7,float(np.sqrt(np.mean(audio*audio))))
    out=np.tanh(audio*(drive/(4*rms)))
    out*=target_rms/max(1e-7,float(np.sqrt(np.mean(out*out))))
    return out.astype(np.float32)


def damp_isolated(audio,sr,notes,start,max_seconds=.12,decay_seconds=.04):
    """Shorten isolated monophonic notes and their labels together.

    Full-recording note times are needed, including attacks before the crop.
    This synthesizes short decay, not a claim to reproduce a real palm mute.
    """
    audio=np.asarray(audio,dtype=np.float32);duration=len(audio)/sr
    relevant=sorted([n for n in notes if n['onset']<start+duration and n['offset']>start],key=lambda n:n['onset'])
    if any(a['offset']>b['onset']+.005 for a,b in zip(relevant,relevant[1:])):
        raise ValueError('Damping requires a monophonic isolated source')
    envelope=np.zeros(len(audio),np.float32);changed=[]
    for old in notes:
        note={**old,'offset':min(old['offset'],old['onset']+max_seconds)};changed.append(note)
        a=round((note['onset']-start)*sr);b=round((note['offset']-start)*sr);lo=max(0,a);hi=min(len(audio),b)
        if hi<=lo:continue
        t=(np.arange(lo,hi)-a)/sr;remaining=(b-np.arange(lo,hi))/sr
        attack=np.clip(t/.002,0,1);release=np.clip(remaining/.01,0,1)
        envelope[lo:hi]=np.maximum(envelope[lo:hi],np.exp(-t/decay_seconds)*attack*release)
    return (audio*envelope).astype(np.float32),changed
