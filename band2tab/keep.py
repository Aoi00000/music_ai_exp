"""Onset-preserving Mu note pruning; labels never enter inference."""
import math, copy
import numpy as np
import torch
from torch import nn
from .audio import PITCH_MIN,PITCH_MAX
from .addition import _assign

SCHEMA='mu_keep_frame_protection_v1'
DIMENSION=350

class KeepNet(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.net=nn.Sequential(nn.Linear(DIMENSION,1)) if kind=='linear' else nn.Sequential(
            nn.Linear(DIMENSION,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,2.)
    def forward(self,x):return self.net(x).squeeze(-1)

def keep_targets(pool,truth,baseline,guard,duration,hop=.02):
    """Strict unique onset labels versus strict labels OR harmful-to-frame deletion.

    The marginal frame calculation uses the same sampled time origin/range as
    instrument_metrics. These are training teachers, never inference features.
    """
    end=duration-guard;frames=max(1,math.ceil((end-guard)/hop));shape=(frames,2,PITCH_MAX-PITCH_MIN+1)
    rr=np.zeros(shape,bool);counts=np.zeros(shape,np.int32)
    interval=lambda n:(max(0,math.ceil((n['onset']-guard)/hop-1e-9)),min(frames,math.ceil((n['offset']-guard)/hop-1e-9)))
    for part_id,part in enumerate(('guitar','bass')):
        for n in truth.get(part,[]):
            if PITCH_MIN<=n['pitch']<=PITCH_MAX:
                a,b=interval(n)
                if a<b:rr[a:b,part_id,n['pitch']-PITCH_MIN]=True
        for n in baseline.get(part,[]):
            if PITCH_MIN<=n['pitch']<=PITCH_MAX:
                a,b=interval(n)
                if a<b:counts[a:b,part_id,n['pitch']-PITCH_MIN]+=1
    er=counts>0;tp=int((rr&er).sum());fp=int((~rr&er).sum());fn=int((rr&~er).sum())
    original=2*tp/max(1,2*tp+fp+fn)
    onset=np.zeros(len(pool),np.float32);mask=np.zeros(len(pool),np.float32);gain=np.zeros(len(pool),np.float32)
    for part_id,part in enumerate(('guitar','bass')):
        indices=[i for i,n in enumerate(pool) if n['part']==part and guard<=n['onset']<end]
        refs=[n for n in truth.get(part,[]) if not n.get('censored_onset') and guard<=n['onset']<end]
        for _,j in _assign(refs,[pool[i] for i in indices]):onset[indices[j]]=1.
        for i in indices:
            n=pool[i]
            if any(r.get('censored_onset') and r['pitch']==n['pitch'] and r['onset']<=n['onset']<r['offset'] for r in truth.get(part,[])):continue
            mask[i]=1.;a,b=interval(n);k=n['pitch']-PITCH_MIN
            unique=counts[a:b,part_id,k]==1;correct=rr[a:b,part_id,k]
            dt=int((unique&correct).sum());df=int((unique&~correct).sum())
            gain[i]=2*(tp-dt)/max(1,2*(tp-dt)+(fp-df)+(fn+dt))-original
    joint=np.maximum(onset,(gain < -1e-9).astype(np.float32))
    return dict(onset=onset,joint=joint,mask=mask,delete_frame_gain=gain)

def prune(pool,probability,threshold,unmodelled,duration,guard=.25):
    if len(pool)!=len(probability) or not np.isfinite(probability).all():raise ValueError('Invalid keep probabilities')
    parts={p:copy.deepcopy(unmodelled.get(p,[])) for p in ('guitar','bass')}
    for n,q in zip(pool,probability):
        eligible=guard<=n['onset']<duration-guard
        if eligible and q<threshold:continue
        parts[n['part']].append({k:n[k] for k in ('pitch','onset','offset','velocity')})
    return {p:sorted(ns,key=lambda n:(n['onset'],n['pitch'])) for p,ns in parts.items()}

@torch.inference_mode()
def probabilities(state,x):
    if state['schema']!=SCHEMA:raise ValueError('Unknown keep checkpoint')
    z=np.clip((x-np.asarray(state['mean'],np.float32))/np.asarray(state['std'],np.float32),-8,8).astype(np.float32)
    outputs=[]
    for weights in state['members']:
        model=KeepNet(state['kind']).eval();model.load_state_dict(weights,strict=True)
        outputs.append(model(torch.from_numpy(z)).sigmoid().numpy())
    return np.mean(outputs,axis=0)
