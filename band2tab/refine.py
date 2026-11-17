"""Learned note-level refinement on top of frozen MuScriptor predictions.

Never overwrite the source MIDI. Review mode returns the original parts and
separate proposed changes. This first stage cannot invent missing pitches or
correct timing: it classifies existing events as guitar, bass, or neither.
"""
from pathlib import Path
import argparse,copy,hashlib,json,math
import numpy as np
import torch
from torch import nn
from .audio import features,SR,HOP,PITCH_MIN,PITCH_MAX

SCHEMA='muscriptor_event_refiner_v1'
PARTS=('other','guitar','bass')

def candidates_from_midi(path):
    import pretty_midi
    midi=pretty_midi.PrettyMIDI(str(path));out=[]
    for ins in midi.instruments:
        if ins.is_drum:continue
        part=1 if 24<=ins.program<=31 else 2 if 32<=ins.program<=39 else 0
        for note in ins.notes:
            if PITCH_MIN<=note.pitch<=PITCH_MAX and note.end-note.start>=.03:
                out.append({'pitch':int(note.pitch),'onset':float(note.start),'offset':float(note.end),
                            'velocity':int(note.velocity),'program':int(ins.program),'baseline_class':part})
    return sorted(out,key=lambda n:(n['onset'],n['pitch'],n['program']))

def event_features(x,notes):
    """No labels: pitch, source program, duration, local and harmonic CQT."""
    x=np.asarray(x,np.float32);scale=max(float(np.percentile(x,90)),1e-4)
    x=x/scale;rows=[]
    for n in notes:
        if not (0<=n['onset']<n['offset'] and PITCH_MIN<=n['pitch']<=PITCH_MAX):
            raise ValueError('Invalid candidate note')
        a=min(len(x)-1,max(0,round(n['onset']*SR/HOP)))
        b=min(len(x),max(a+1,round(n['offset']*SR/HOP)))
        body=x[a:b];attack=x[max(0,a-3):min(len(x),a+5)]
        center=3*(int(n['pitch'])-PITCH_MIN)
        # Relative pitch patch, plus harmonics: a large peak alone is insufficient.
        def band(array,c):
            lo=max(0,c-1);hi=min(x.shape[1],c+2)
            return float(array[:,lo:hi].mean()) if lo<hi else 0.
        patch=[band(array,center+3*k) for array in (attack,body) for k in range(-6,7)]
        harmonics=[band(array,center+round(36*math.log2(h))) for array in (attack,body) for h in range(1,9)]
        spectrum=body.mean(0)[:264].reshape(33,8).mean(1).tolist()
        basic=[n['pitch']/127,math.log1p(n['offset']-n['onset']),n['program']/127,
               float(n['velocity'])/127]+[float(n['baseline_class']==i) for i in range(3)]
        rows.append(basic+patch+harmonics+spectrum)
    return np.asarray(rows,np.float32).reshape(len(notes),82)

class EventRefiner(nn.Module):
    def __init__(self,dimension=82):
        super().__init__();self.net=nn.Sequential(nn.Linear(dimension,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,baseline):
        prior=torch.nn.functional.one_hot(baseline,3).to(x.dtype)*2.
        return prior+self.net(x)

def proposed_parts(notes,probability,threshold=.75,margin=.2,allow_removal=True):
    parts={'guitar':[],'bass':[]};changes=[]
    for i,(n,p) in enumerate(zip(notes,probability)):
        old=n['baseline_class'];best=int(np.argmax(p));chosen=old
        if threshold<1. and best!=old and p[best]>=threshold and p[best]-p[old]>=margin and (allow_removal or best!=0):chosen=best
        if chosen!=old:
            changes.append({'candidate':i,'from':PARTS[old],'to':PARTS[chosen],
                            'score':float(p[chosen]),'note':dict(n)})
        if chosen:
            part=PARTS[chosen];note={k:n[k] for k in ('pitch','onset','offset','velocity')}
            note.update(refinement_score=float(p[chosen]),needs_review=chosen!=old)
            parts[part].append(note)
    return parts,changes

def refine(audio,midi,checkpoint=None,mode='review',device='cpu'):
    if mode not in ('review','apply'):raise ValueError('mode must be review or apply')
    from .score import read_midi
    original=read_midi(midi)
    if checkpoint is None:
        return {'parts':original,'baseline_parts':copy.deepcopy(original),'suggested_parts':copy.deepcopy(original),
                'refinement_changes':[],'refinement_mode':'disabled','automatic_application_approved':False}
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['schema'] not in (SCHEMA,'muscriptor_event_refiner_duration_v2'):raise ValueError('Unknown refiner feature schema')
    approved=bool(state.get('automatic_application_approved',False))
    if mode=='apply' and not approved:raise ValueError('Automatic correction is not validated; use review mode')
    notes=candidates_from_midi(midi);x=event_features(features(audio),notes)
    model=EventRefiner().to(device);model.load_state_dict(state['model']);model.eval()
    with torch.no_grad():
        z=np.clip((x-np.array(state['mean']))/np.array(state['std']),-8,8).astype('float32')
        probabilities=model(torch.from_numpy(z).to(device),torch.tensor([n['baseline_class'] for n in notes],dtype=torch.long,device=device)).softmax(-1).cpu().numpy()
    suggested,changes=proposed_parts(notes,probabilities,**state['policy'])
    # Our model has a bounded pitch/duration vocabulary. Preserve all events
    # outside it, even in proposed output; review mode is exactly read_midi.
    for part,ns in original.items():
        suggested[part].extend(copy.deepcopy(n) for n in ns if not PITCH_MIN<=n['pitch']<=PITCH_MAX or n['offset']-n['onset']<.03)
    return {'parts':suggested if mode=='apply' else original,'baseline_parts':copy.deepcopy(original),
            'suggested_parts':suggested,'refinement_changes':changes,'refinement_mode':mode,
            'refiner_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'automatic_application_approved':approved,
            'limitations':'Existing events only. No missing-pitch, onset, offset, fingering or technique correction.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--audio',required=True);p.add_argument('--midi',required=True)
    p.add_argument('--checkpoint',default='models/muscriptor_refiner.pt');p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['review','apply'],default='review');p.add_argument('--device',default='cpu');a=p.parse_args()
    result=refine(a.audio,a.midi,a.checkpoint,a.mode,a.device)
    from .score import midi_timing
    bpm,meter,beats=midi_timing(a.midi);result.update(bpm=bpm,beats_per_bar=meter,beat_times=beats)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('Refinement proposals:',len(result['refinement_changes']),'mode:',a.mode)

if __name__=='__main__':main()
