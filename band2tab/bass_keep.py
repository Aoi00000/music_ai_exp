"""Portable bass-only review bundle. No YourMT3 model is needed at inference."""
import numpy as np
from .keep import probabilities as plain_prob
from .keep_agreement import probabilities as agree_prob,SCHEMA as AGREEMENT_SCHEMA
from .keep import prune

SCHEMA='band2tab_bass_keep_review_v1'

def probabilities(bundle,x,pool):
    if bundle['schema']!=SCHEMA:raise ValueError('Unknown bass keep bundle')
    learner=bundle['learner']
    if learner['schema']==AGREEMENT_SCHEMA:
        # Agreement inputs are always zero for bass, by training design.
        # Guitar scores are deliberately unused, so no support model is loaded.
        z=np.concatenate([x,np.zeros((len(x),8),np.float32)],axis=1);q=agree_prob(learner,z)
    else:q=plain_prob(learner,x)
    return np.where([n['part']=='guitar' for n in pool],1.,q)

def apply(bundle,x,pool,unmodelled,duration,guard=.25):
    q=probabilities(bundle,x,pool);chosen=[]
    eligible=[i for i,n in enumerate(pool) if n['part']=='bass' and guard<=n['onset']<duration-guard]
    limit=int(len(eligible)*bundle['max_remove_fraction'])
    chosen=sorted((i for i in eligible if q[i]<bundle['threshold']),key=lambda i:float(q[i]))[:limit]
    p=np.ones(len(pool),np.float32);p[chosen]=0.
    return prune(pool,p,.5,unmodelled,duration,guard)
