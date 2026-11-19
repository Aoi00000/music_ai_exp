"""Learned complementary-model evidence for existing Mu notes."""
import numpy as np,torch
from torch import nn

SCHEMA='mu_keep_your_agreement_v1'

def agreement_features(pool,support):
    out=np.zeros((len(pool),8),np.float32)
    for i,n in enumerate(pool):
        if n['part']!='guitar':continue
        same=[s for s in support if s['pitch']==n['pitch']]
        close=min([abs(s['onset']-n['onset']) for s in same]+[1.])
        overlap=max([max(0,min(n['offset'],s['offset'])-max(n['onset'],s['onset']))/(n['offset']-n['onset']) for s in same]+[0.])
        out[i]=[float(close<=.05),float(close<=.12),min(close,1.),overlap,
            float(any(s['onset']<=n['onset']<s['offset'] for s in same)),
            float(any(abs(s['pitch']-n['pitch'])==12 and abs(s['onset']-n['onset'])<=.05 for s in support)),
            min(12,sum(abs(s['onset']-n['onset'])<=.05 for s in support))/12.,1.]
    return out

class AgreementKeepNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(358,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,2.)
    def forward(self,x):return self.net(x).squeeze(-1)

@torch.inference_mode()
def probabilities(state,x):
    if state['schema']!=SCHEMA:raise ValueError('Unknown agreement schema')
    z=np.clip((x-np.asarray(state['mean'],np.float32))/np.asarray(state['std'],np.float32),-8,8).astype(np.float32)
    if not state['use_agreement']:z[:,350:]=0
    values=[]
    for weights in state['members']:
        model=AgreementKeepNet().eval();model.load_state_dict(weights,strict=True);values.append(model(torch.from_numpy(z)).sigmoid().numpy())
    q=np.mean(values,axis=0)
    if state.get('agreement_guard'):q=np.where(x[:,350]>.5,1.,q)
    return q
