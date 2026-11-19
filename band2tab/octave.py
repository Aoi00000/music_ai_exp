"""Conservative octave-neighbour review; never shift or retime notes."""
import numpy as np
import torch
from .stem_refiner import PARTS


def octave_context(x,pool):
    extra=np.zeros((len(pool),354),np.float32);eligible=np.zeros(len(pool),bool)
    for i,n in enumerate(pool):
        if n['part']!='guitar':continue
        peers=[(j,q) for j,q in enumerate(pool) if q['part']=='guitar' and abs(q['pitch']-n['pitch'])==12 and abs(q['onset']-n['onset'])<=.08]
        if not peers:continue
        j,q=min(peers,key=lambda pair:abs(pair[1]['onset']-n['onset']));eligible[i]=True
        overlap=max(0,min(n['offset'],q['offset'])-max(n['onset'],q['onset']))/max(.001,n['offset']-n['onset'])
        extra[i,:350]=x[i]-x[j];extra[i,350:]=[(q['pitch']-n['pitch'])/12,(q['onset']-n['onset'])/.08,overlap,min(len(peers),4)/4]
    return np.concatenate([x,extra],axis=1),eligible


def prune_octaves(pool,probability,eligible,threshold,unmodelled=None):
    if len(pool)!=len(probability) or len(pool)!=len(eligible) or not np.isfinite(probability).all():raise ValueError('Invalid octave predictions')
    parts={p:list((unmodelled or {}).get(p,[])) for p in PARTS}
    for n,prob,use in zip(pool,probability,eligible):
        if n['part']=='guitar' and use and prob<threshold:continue
        parts[n['part']].append({k:n[k] for k in ('pitch','onset','offset','velocity')})
    return {p:sorted(ns,key=lambda n:(n['onset'],n['pitch'])) for p,ns in parts.items()}


@torch.inference_mode()
def predict_octaves(state,x,pool):
    context,eligible=octave_context(x,pool);a=context if state['config']['arm']=='octave_context704' else x
    mean=np.asarray(state['mean'],np.float32);std=np.asarray(state['std'],np.float32);model=torch.nn.Linear(a.shape[1],1);model.load_state_dict(state['model'],strict=True);model.eval()
    return model(torch.from_numpy(np.clip((a-mean)/std,-8,8).astype(np.float32))).squeeze(-1).sigmoid().numpy(),eligible
