"""Learned note corrections on the same frozen, source-known Mu baseline."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .audio import PITCH_MIN,PITCH_MAX,SR,HOP
from .candidate import unique_targets
from .refine import event_features

SCHEMA='muscriptor_stem_note_corrector_v1'
PARTS=('guitar','bass')
DIMENSION=350


def read_candidates(path,part,duration):
    import pretty_midi
    notes=[]
    for instrument in pretty_midi.PrettyMIDI(str(path)).instruments:
        if instrument.is_drum:continue
        cls=1 if 24<=instrument.program<=31 else 2 if 32<=instrument.program<=39 else 0
        for n in instrument.notes:
            if not 0<=n.start<duration:continue
            notes.append({'pitch':int(n.pitch),'onset':float(n.start),'offset':min(duration,float(n.end)),
                'velocity':int(n.velocity),'program':int(instrument.program),'baseline_class':cls,'part':part,
                'baseline':[float(part==p) for p in PARTS]})
    return sorted(notes,key=lambda n:(n['onset'],n['pitch'],n['program']))


def rms_envelope(audio,sr):
    from .separation import resample
    x=np.asarray(audio,np.float32)
    if x.ndim==2:x=x.mean(1)
    x=resample(x,sr,SR);x=np.pad(x,(0,HOP-len(x)%HOP))
    return np.log1p(100*np.sqrt((x.reshape(-1,HOP)**2).mean(1)))


def feature_matrix(views,envelopes,pool,duration):
    if not pool:return np.empty((0,DIMENSION),np.float32)
    blocks=[event_features(views[p],pool) for p in ('mix','guitar','bass')];extra=[]
    for n in pool:
        neighbors=sum(m['part']==n['part'] and abs(m['onset']-n['onset'])<.05 for m in pool)
        values=[float(n['part']==p) for p in PARTS]+[n['onset']/max(.01,duration),min(neighbors,12)/12]
        for source in ('mix',n['part']):
            env=envelopes[source]
            for edge in ('onset','offset'):
                center=round(n[edge]*SR/HOP);indices=np.clip(np.arange(center-12,center+13),0,len(env)-1)
                patch=env[indices];values.extend((patch-patch[12]).tolist())
        extra.append(values)
    return np.concatenate([*blocks,np.asarray(extra,np.float32)],axis=1).astype(np.float32)


def supervised_targets(pool,truth,guard,duration):
    # Assign within each estimated source. A bass unison must never steal the
    # guitar's positive target during the global duplicate assignment.
    result={k:np.zeros((len(pool),2) if k in ('delta','timing_mask') else len(pool),np.float32)
            for k in ('y','mask','delta','timing_mask')}
    for role,part in enumerate(PARTS):
        indices=np.array([i for i,n in enumerate(pool) if n['part']==part],dtype=int)
        if not len(indices):continue
        targets=unique_targets([pool[i] for i in indices],{part:truth.get(part,[])})
        for key in result:result[key][indices]=targets[key][:,role]
    valid=np.array([guard<=n['onset']<duration-guard for n in pool],np.float32)
    result['mask']*=valid;result['timing_mask']*=valid[:,None]
    return result


class StemNoteCorrector(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.width=width
        self.body=nn.Sequential(nn.Linear(DIMENSION,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(.15),nn.Linear(width,width//2),nn.GELU())
        self.head=nn.Linear(width//2,3);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        z=self.head(self.body(x));return {'logit':z[...,0]+2.,'delta':z[...,1:].tanh()}


def correction_loss(output,target,positive_weight=1.5):
    weight=torch.where(target['y']>.5,positive_weight,1.)*target['mask']
    ce=torch.nn.functional.binary_cross_entropy_with_logits(output['logit'],target['y'],reduction='none')
    classification=(ce*weight).sum()/weight.sum().clamp_min(1)
    mask=target['timing_mask'];error=torch.nn.functional.smooth_l1_loss(output['delta'],target['delta'],reduction='none')
    return classification+.3*(error*mask).sum()/mask.sum().clamp_min(1)


def suggest(pool,probability,delta,duration,threshold=.5,retime=True,unmodelled=None):
    if len(pool)!=len(probability) or np.asarray(delta).shape!=(len(pool),2):raise ValueError('Correction/event lengths differ')
    if not np.isfinite(probability).all() or not np.isfinite(delta).all():raise ValueError('Non-finite note correction')
    parts={p:list((unmodelled or {}).get(p,[])) for p in PARTS}
    for n,score,change in zip(pool,probability,delta):
        if float(score)<threshold:continue
        a,b=n['onset'],n['offset']
        if retime:
            da,db=np.asarray(change)*[.12,.4];a=max(0.,a+float(da));b=min(duration,b+float(db))
            if b<=a:a,b=n['onset'],n['offset']
        parts[n['part']].append({'pitch':n['pitch'],'onset':a,'offset':b,'velocity':n['velocity']})
    return {p:sorted(ns,key=lambda n:(n['onset'],n['pitch'])) for p,ns in parts.items()}


@torch.inference_mode()
def predict_corrections(state,x):
    if state['schema']!=SCHEMA:raise ValueError('Unsupported stem corrector')
    model=StemNoteCorrector(state['width']).eval();model.load_state_dict(state['model'],strict=True)
    z=np.clip((x-np.asarray(state['mean'],np.float32))/np.asarray(state['std'],np.float32),-8,8).astype(np.float32)
    out=model(torch.from_numpy(z))
    return out['logit'].sigmoid().numpy(),out['delta'].numpy()
