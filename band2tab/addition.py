"""Classify complementary guitar proposals without editing existing notes."""
import copy
import numpy as np,torch
from torch import nn
from .stem_refiner import feature_matrix
from .audio import PITCH_MIN,PITCH_MAX

SCHEMA='protected_guitar_additions_v1'
DIMENSION=358


def proposals(notes,baseline):
    pool=[]
    for n in notes:
        if not PITCH_MIN<=n['pitch']<=PITCH_MAX:continue
        if any(b['pitch']==n['pitch'] and abs(b['onset']-n['onset'])<=.05 for b in baseline):continue
        if any(b['pitch']==n['pitch'] and abs(b['onset']-n['onset'])<=.02 for b in pool):continue
        pool.append(copy.deepcopy(n))
    return pool


def addition_features(views,envelopes,pool,baseline,duration):
    x=feature_matrix(views,envelopes,pool,duration)
    if not pool:return np.empty((0,DIMENSION),np.float32)
    extra=[]
    for n in pool:
        same=[b for b in baseline if b['pitch']==n['pitch']];length=max(.001,n['offset']-n['onset'])
        close=min([abs(b['onset']-n['onset']) for b in same]+[1.])
        overlap=max([max(0,min(b['offset'],n['offset'])-max(b['onset'],n['onset']))/length for b in same]+[0.])
        active=sum(b['onset']<=n['onset']<b['offset'] for b in same)
        nearby=sum(abs(b['onset']-n['onset'])<=.05 for b in baseline)
        proposal_nearby=sum(abs(b['onset']-n['onset'])<=.05 for b in pool)
        any_close=min([abs(b['onset']-n['onset']) for b in baseline]+[1.])
        extra.append([min(close,1),overlap,min(active,3)/3,min(nearby,12)/12,min(proposal_nearby,12)/12,min(length,2)/2,min(any_close,1),float(bool(same))])
    return np.concatenate([x,np.array(extra,np.float32)],axis=1)


def _assign(refs,notes):
    from scipy.optimize import linear_sum_assignment
    if not refs or not notes:return []
    dummy=2*(len(refs)+1);cost=np.full((len(refs),len(notes)+len(refs)),1e9);cost[:,len(notes):]=dummy
    for i,r in enumerate(refs):
        for j,n in enumerate(notes):
            if r['pitch']==n['pitch'] and abs(r['onset']-n['onset'])<=.05:cost[i,j]=abs(r['onset']-n['onset'])/.05+1e-8*j
    return [(i,j) for i,j in zip(*linear_sum_assignment(cost)) if j<len(notes) and cost[i,j]<dummy]


def targets(pool,baseline,truth,guard,duration):
    refs=[n for n in truth if not n.get('censored_onset') and guard<=n['onset']<duration-guard]
    valid_base=[n for n in baseline if guard<=n['onset']<duration-guard]
    covered={i for i,j in _assign(refs,valid_base)};remaining=[r for i,r in enumerate(refs) if i not in covered]
    y=np.zeros(len(pool),np.float32);mask=np.array([guard<=n['onset']<duration-guard for n in pool],np.float32)
    for i,n in enumerate(pool):
        if any(r.get('censored_onset') and r['pitch']==n['pitch'] and r['onset']<=n['onset']<r['offset'] for r in truth):mask[i]=0
    eligible=[i for i,n in enumerate(pool) if mask[i]]
    for i,j in _assign(remaining,[pool[i] for i in eligible]):y[eligible[j]]=1
    return dict(y=y,mask=mask)


def add_notes(baseline,pool,probability,threshold):
    if len(pool)!=len(probability) or not np.isfinite(probability).all():raise ValueError('Invalid addition probabilities')
    parts=copy.deepcopy(baseline)
    for n,p in zip(pool,probability):
        if p>=threshold:parts['guitar'].append({k:n[k] for k in ('pitch','onset','offset','velocity')})
    return {p:sorted(ns,key=lambda n:(n['onset'],n['pitch'])) for p,ns in parts.items()}


class AdditionRanker(nn.Module):
    def __init__(self,kind='linear'):
        super().__init__();self.kind=kind
        self.net=nn.Sequential(nn.Linear(DIMENSION,1)) if kind=='linear' else nn.Sequential(nn.Linear(DIMENSION,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,-3.)
    def forward(self,x):return self.net(x).squeeze(-1)


@torch.inference_mode()
def predict_additions(state,x):
    if state['schema']!=SCHEMA:raise ValueError('Unexpected addition schema')
    model=AdditionRanker(state['kind']).eval();model.load_state_dict(state['model'],strict=True)
    z=np.clip((x-np.asarray(state['mean'],np.float32))/np.asarray(state['std'],np.float32),-8,8).astype(np.float32)
    return model(torch.from_numpy(z)).sigmoid().numpy()
