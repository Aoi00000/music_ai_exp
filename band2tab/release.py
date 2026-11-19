"""Learn releases while keeping every pitch, role, and onset unchanged."""
import numpy as np
import torch
from torch import nn
from .stem_refiner import DIMENSION,PARTS

SCHEMA='onset_protected_release_v2'


class ReleaseCorrector(nn.Module):
    def __init__(self,kind='linear'):
        super().__init__();self.kind=kind
        if kind=='linear':self.net=nn.Sequential(nn.Linear(DIMENSION,1))
        elif kind=='mlp':self.net=nn.Sequential(nn.Linear(DIMENSION,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,1))
        else:raise ValueError('Unknown release model')
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).squeeze(-1).tanh()


def release_loss(prediction,target):
    # Train on uniquely matched notes with reliable attacks and uncensored ends.
    mask=target['timing_mask'][:,1]*(target['delta'][:,0].abs()<=.05/.12)*target['y']
    error=torch.nn.functional.smooth_l1_loss(prediction,target['delta'][:,1],reduction='none')
    return ((error+.05*prediction.square())*mask).sum()/mask.sum().clamp_min(1)


def apply_release(pool,delta,duration,scale=1.,unmodelled=None):
    delta=np.asarray(delta)
    if delta.shape!=(len(pool),) or not np.isfinite(delta).all():raise ValueError('Invalid release predictions')
    if not 0<=scale<=1:raise ValueError('Release scale must be within [0,1]')
    parts={p:list((unmodelled or {}).get(p,[])) for p in PARTS}
    for note,change in zip(pool,delta):
        limit=min(.15,.5*(note['offset']-note['onset']))
        end=min(duration,note['offset']+float(np.clip(.4*scale*change,-limit,limit)))
        if end<=note['onset']:end=note['offset']
        parts[note['part']].append({k:note[k] for k in ('pitch','onset','offset','velocity')})
        parts[note['part']][-1]['offset']=end
    return {p:sorted(ns,key=lambda n:(n['onset'],n['pitch'])) for p,ns in parts.items()}


@torch.inference_mode()
def predict_release(state,x):
    if state['schema']!=SCHEMA:raise ValueError('Unexpected release schema')
    model=ReleaseCorrector(state['kind']).eval();model.load_state_dict(state['model'],strict=True)
    x=np.clip((x-np.asarray(state['mean'],np.float32))/np.asarray(state['std'],np.float32),-8,8).astype(np.float32)
    return model(torch.from_numpy(x)).numpy()
