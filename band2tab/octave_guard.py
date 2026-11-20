"""Recognize an upper-octave note versus a lower note's harmonics."""
import copy,math
import numpy as np,torch
from torch import nn
from .audio import SR,HOP,PITCH_MIN,PITCH_MAX
SCHEMA='upper_octave_temporal_guard_v1'

def peers(pool,duration,guard=.25):
    selected=[]
    for i,n in enumerate(pool):
        if n['part']!='guitar' or not PITCH_MIN<=n['pitch']<=PITCH_MAX or not guard<=n['onset']<duration-guard:continue
        below=[(j,b) for j,b in enumerate(pool) if b['part']=='guitar' and b['pitch']==n['pitch']-12 and abs(b['onset']-n['onset'])<=.08 and min(b['offset'],n['offset'])>max(b['onset'],n['onset'])]
        if below:
            j,b=min(below,key=lambda z:abs(z[1]['onset']-n['onset']));selected.append((i,j))
    return selected

def patches(cqt,pool,pairs):
    result=[];meta=[]
    for i,j in pairs:
        n,b=pool[i],pool[j];start=min(n['onset'],b['onset'])-.05
        times=start+np.arange(24)*(.5/24);ti=np.clip(np.round(times*SR/HOP).astype(int),0,len(cqt)-1);p=[]
        for h in range(1,13):
            center=round(3*(b['pitch']-PITCH_MIN)+36*math.log2(h));lo=max(0,center-1);hi=min(cqt.shape[1],center+2)
            p.append(cqt[ti,lo:hi].mean(1) if lo<hi else np.zeros(24,np.float32))
        p=np.stack(p,1);p=p/max(.05,float(np.percentile(p,90)));result.append(np.clip(p,0,8))
        overlap=max(0,min(n['offset'],b['offset'])-max(n['onset'],b['onset']))/max(.001,n['offset']-n['onset'])
        meta.append([n['pitch']/127,(n['onset']-b['onset'])/.08,min(n['offset']-n['onset'],3)/3,overlap])
    return np.asarray(result,np.float32).reshape(-1,24,12),np.asarray(meta,np.float32).reshape(-1,4)

def keep_labels(pool,pairs,truth,duration,guard=.25):
    from .addition import _assign
    refs=[n for n in truth.get('guitar',[]) if not n.get('censored_onset') and guard<=n['onset']<duration-guard]
    ix=[i for i,n in enumerate(pool) if n['part']=='guitar' and guard<=n['onset']<duration-guard]
    correct={ix[j] for i,j in _assign(refs,[pool[i] for i in ix])};ys=[];masks=[]
    frames=math.ceil((duration-2*guard)/.02)
    interval=lambda n:(max(0,math.ceil((n['onset']-guard)/.02-1e-9)),min(frames,math.ceil((n['offset']-guard)/.02-1e-9)))
    for i,j in pairs:
        n=pool[i];a,b=interval(n);sounding=False
        for t in truth.get('guitar',[]):
            if t['pitch']!=n['pitch']:continue
            c,d=interval(t)
            if max(a,c)<min(b,d):sounding=True;break
        ys.append(float(i in correct or sounding));masks.append(float(not any(t.get('censored_onset') and t['pitch']==n['pitch'] and t['onset']<=n['onset']<t['offset'] for t in truth.get('guitar',[]))))
    return np.array(ys,np.float32),np.array(masks,np.float32)

class OctaveGuard(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.GELU(),nn.Conv2d(8,16,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d((4,12)))
        self.head=nn.Sequential(nn.Linear(16*4*12+4,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.constant_(self.head[-1].bias,2.)
    def forward(self,x,meta):return self.head(torch.cat([self.conv(x[:,None]).flatten(1),meta],1)).squeeze(-1)

@torch.inference_mode()
def probabilities(state,x,meta):
    if state['schema']!=SCHEMA:raise ValueError('Wrong octave guard schema')
    if not len(x):return np.empty(0,np.float32)
    q=[]
    for weights in state['members']:
        model=OctaveGuard().eval();model.load_state_dict(weights,strict=True);q.append(model(torch.as_tensor(x),torch.as_tensor(meta)).sigmoid().numpy())
    return np.mean(q,0)

def prune(baseline,pool,pairs,q,threshold,cap_fraction=.1):
    if len(q)!=len(pairs) or not np.isfinite(q).all():raise ValueError('Invalid octave guard predictions')
    maximum=int(len(baseline['guitar'])*cap_fraction);chosen=sorted((k for k,v in enumerate(q) if v<threshold),key=lambda k:float(q[k]))[:maximum]
    remove={pairs[k][0] for k in chosen};out={p:[] for p in ('guitar','bass')}
    for i,n in enumerate(pool):
        if i not in remove:out[n['part']].append({k:n[k] for k in ('pitch','onset','offset','velocity')})
    return out
