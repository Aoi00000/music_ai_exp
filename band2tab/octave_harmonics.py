"""Interpretable harmonic/envelope summaries for the octave guard."""
import numpy as np
SCHEMA='upper_octave_harmonic_logistic_v1'

def summarize(x,meta):
    if not len(x):return np.empty((0,70),np.float64)
    mean=x.mean(1);std=x.std(1);change=x[:,16:].mean(1)-x[:,:8].mean(1);z=x-x.mean(1,keepdims=True)
    def corr(a,b):return (z[:,:,a]*z[:,:,b]).sum(1)/np.sqrt((z[:,:,a]**2).sum(1)*(z[:,:,b]**2).sum(1)+1e-8)
    correlations=np.stack([corr(0,j) for j in range(1,12)]+[corr(1,j) for j in (3,5,7,9,11)],1)
    ratios=np.clip(np.log((mean[:,1:]+1e-4)/(mean[:,:1]+1e-4)),-8,8)
    shapes=np.stack([mean[:,1::2].mean(1)-mean[:,::2].mean(1),mean[:,:4].mean(1)-mean[:,4:8].mean(1),mean[:,4:8].mean(1)-mean[:,8:].mean(1)],1)
    out=np.concatenate([mean,std,change,correlations,ratios,shapes,meta],1).astype(np.float64)
    if out.shape[1]!=70 or not np.isfinite(out).all():raise ValueError('Bad harmonic summary')
    return out

def probabilities(state,x,meta):
    if state['schema']!=SCHEMA:raise ValueError('Unknown harmonic guard schema')
    z=np.clip((summarize(x,meta)-np.asarray(state['mean']))/np.asarray(state['std']),-8,8)
    raw=z@np.asarray(state['coef'])+state['intercept'];return 1/(1+np.exp(-np.clip(raw,-60,60)))
