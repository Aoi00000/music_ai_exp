"""Audio-backed guitar/bass additions. Never delete or retime original notes."""
import copy, math
import numpy as np
import torch
from torch import nn
from .audio import PITCH_MIN,PITCH_MAX
from .addition import _assign,addition_features
SCHEMA='band2tab_residual_additions_v1'
DIMENSION=366

def candidate_pool(baseline,support,duration,guard=.25):
    pool=[]
    def put(part,n,origin):
        if not PITCH_MIN<=n['pitch']<=PITCH_MAX or not guard<=n['onset']<duration-guard:return
        if not n['onset']<n['offset']<=duration:return
        if any(b['pitch']==n['pitch'] and abs(b['onset']-n['onset'])<=.05 for b in baseline[part]):return
        match=next((b for b in pool if b['part']==part and b['pitch']==n['pitch'] and abs(b['onset']-n['onset'])<=.02),None)
        if match is None:
            match={k:n[k] for k in ('pitch','onset','offset','velocity')};match.update(part=part,program=25 if part=='guitar' else 33,baseline_class=1 if part=='guitar' else 2,baseline=[float(part==p) for p in ('guitar','bass')],origin=[0.,0.,0.,0.]);pool.append(match)
        match['origin'][origin]=1.
    # Keep support timing first; octave candidates may corroborate it, not use GT.
    for part in ('guitar','bass'):
        for n in support.get(part,[]):put(part,n,0)
        for n in baseline[part]:
            for shift,origin in [(-12,1),(12,2)]:put(part,{**n,'pitch':n['pitch']+shift},origin)
    return sorted(pool,key=lambda n:(('guitar','bass').index(n['part']),n['onset'],n['pitch']))

def residual_features(views,envelopes,pool,baseline,duration):
    chunks=[]
    for part in ('guitar','bass'):
        ns=[n for n in pool if n['part']==part]
        x=addition_features(views,envelopes,ns,baseline[part],duration);extra=[]
        for n in ns:
            peer=[b for b in baseline[part] if abs(abs(b['pitch']-n['pitch'])-12)<.1]
            close=min([abs(b['onset']-n['onset']) for b in peer]+[1.])
            other=baseline['bass' if part=='guitar' else 'guitar']
            extra.append(n['origin']+[min(close,1.),float(any(b['pitch']==n['pitch'] and abs(b['onset']-n['onset'])<=.05 for b in other)),min(len(ns),200)/200.,float(part=='bass')])
        chunks.append(np.concatenate([x,np.array(extra,np.float32).reshape(-1,8)],1))
    return np.concatenate(chunks) if chunks else np.empty((0,DIMENSION),np.float32)

def teacher(pool,baseline,truth,duration,guard=.25,hop=.02):
    y=np.zeros(len(pool),np.float32);sound=np.zeros(len(pool),np.float32);mask=np.ones(len(pool),np.float32)
    frames=math.ceil((duration-2*guard)/hop);rr=np.zeros((frames,2,PITCH_MAX-PITCH_MIN+1),bool);ee=rr.copy()
    interval=lambda n:(max(0,math.ceil((n['onset']-guard)/hop-1e-9)),min(frames,math.ceil((n['offset']-guard)/hop-1e-9)))
    for j,part in enumerate(('guitar','bass')):
        for src,roll in [(truth,rr),(baseline,ee)]:
            for n in src.get(part,[]):
                if PITCH_MIN<=n['pitch']<=PITCH_MAX:
                    a,b=interval(n);roll[a:b,j,n['pitch']-PITCH_MIN]=True
        refs=[n for n in truth.get(part,[]) if guard<=n['onset']<duration-guard and not n.get('censored_onset')]
        base=[n for n in baseline[part] if guard<=n['onset']<duration-guard]
        covered={i for i,k in _assign(refs,base)};remaining=[n for i,n in enumerate(refs) if i not in covered]
        ix=[i for i,n in enumerate(pool) if n['part']==part]
        for k,t in _assign(remaining,[pool[i] for i in ix]):y[ix[t]]=1.
    tp=int((rr&ee).sum());fp=int((~rr&ee).sum());fn=int((rr&~ee).sum());score=2*tp/max(1,2*tp+fp+fn)
    gain=np.zeros(len(pool),np.float32)
    for i,n in enumerate(pool):
        j=0 if n['part']=='guitar' else 1;a,b=interval(n);k=n['pitch']-PITCH_MIN
        new=~ee[a:b,j,k];target=rr[a:b,j,k];dt=int((new&target).sum());df=int((new&~target).sum())
        gain[i]=2*(tp+dt)/max(1,2*(tp+dt)+fp+df+fn-dt)-score;sound[i]=float(gain[i]>=-1e-9)
        if any(t.get('censored_onset') and t['pitch']==n['pitch'] and t['onset']<=n['onset']<t['offset'] for t in truth.get(n['part'],[])):mask[i]=0.
    return dict(y=y,sound=sound,mask=mask,frame_gain=gain)

def apply(baseline,pool,q,thresholds=(1.1,1.1),cap_per_second=2.):
    if len(pool)!=len(q) or not np.isfinite(q).all():raise ValueError('Invalid residual scores')
    out=copy.deepcopy(baseline)
    for part,t in zip(('guitar','bass'),thresholds):
        selected=sorted((i for i,n in enumerate(pool) if n['part']==part and q[i]>=t),key=lambda i:-float(q[i]))
        duration=max([n['offset'] for n in pool]+[0.]);limit=max(1,int(duration*cap_per_second))
        for i in selected[:limit]:out[part].append({k:pool[i][k] for k in ('pitch','onset','offset','velocity')})
    return {p:sorted(ns,key=lambda n:(n['onset'],n['pitch'])) for p,ns in out.items()}

class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(DIMENSION,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,-2.)
    def forward(self,x):return self.net(x)

@torch.inference_mode()
def probabilities(bundle,x):
    if bundle['schema']!=SCHEMA:raise ValueError('Unsupported residual schema')
    z=np.clip((x-np.array(bundle['mean'],np.float32))/np.array(bundle['std'],np.float32),-8,8).astype(np.float32);qs=[]
    for weights in bundle['members']:
        model=ResidualNet().eval();model.load_state_dict(weights,strict=True);q=model(torch.from_numpy(z)).sigmoid().numpy();qs.append(np.minimum(q[:,0],q[:,1]) if bundle['quality'] else q[:,0])
    return np.mean(qs,0)
