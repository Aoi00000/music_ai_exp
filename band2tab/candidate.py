"""Learned guitar/bass selection of existing multi-route note events.

Unlike frame decoding, this model does not manufacture a new attack for every
local probability peak. All non-drum programs remain candidate evidence.
"""
import copy,math
import numpy as np
import torch
from torch import nn
from .audio import HOP,SR,PITCH_MIN,PITCH_MAX
from .fusion import overlap
from .refine import event_features

SCHEMA='multiroute_candidate_v2'
PARTS=('guitar','bass')
ROUTES=('mix','guitar','bass')

def candidate_pool(routes,duration):
    pool=[]
    for route,source in enumerate(ROUTES):
        for note in routes[source]:
            if not 0<=note['onset']<duration:continue
            n={**copy.deepcopy(note),'offset':min(duration,note['offset'])}
            if n['offset']-n['onset']<.03:continue
            matches=[a for a in pool if a['pitch']==n['pitch'] and abs(a['onset']-n['onset'])<=.02 and overlap(a,n)>=.8]
            if matches:a=min(matches,key=lambda x:abs(x['onset']-n['onset']))
            else:
                a={**n,'support':[0.,0.,0.],'classes':[[0.,0.,0.] for _ in ROUTES],
                   'baseline':[0.,0.]};pool.append(a)
            a['support'][route]=1.;a['classes'][route][n['baseline_class']]=1.
            if route==0 and n['baseline_class']:a['baseline'][n['baseline_class']-1]=1.
    return sorted(pool,key=lambda n:(n['onset'],n['pitch'],n['program']))

def candidate_features(views,encoded,pool,duration):
    """246 direct CQT descriptors + 768 frozen acoustic + 16 provenance/time."""
    if not pool:return np.empty((0,1030),np.float32),np.empty((0,2),np.float32)
    blocks=[event_features(view,pool) for view in views];acoustic=[];extra=[];priors=[]
    for n in pool:
        a=min(len(encoded)-1,max(0,round(n['onset']*SR/HOP)))
        b=min(len(encoded),max(a+1,round(n['offset']*SR/HOP)))
        attack=encoded[max(0,a-3):min(len(encoded),a+6)].astype(np.float32).mean(0)
        body=encoded[a:b].astype(np.float32).mean(0)
        acoustic.append(np.concatenate([attack.flatten(),body.flatten()]))
        extra.append(n['support']+np.array(n['classes']).flatten().tolist()+n['baseline']+[n['onset']/max(duration,1e-6),(n['offset']-n['onset'])/max(duration,1e-6)])
        priors.append([2. if n['baseline'][j] else 1. if n['support'][j+1] else -2. for j in (0,1)])
    return np.concatenate([*blocks,np.asarray(acoustic),np.asarray(extra)],1).astype(np.float32),np.asarray(priors,np.float32)

def targets(pool,truth):
    """Independent roles, bounded timing supervision and unknown-cut masks."""
    y=np.zeros((len(pool),2),np.float32);quality=np.zeros_like(y)
    delta=np.zeros((len(pool),2,2),np.float32);timing_mask=np.zeros_like(delta)
    mask=np.ones_like(y);keep_correct=np.zeros_like(y)
    for i,n in enumerate(pool):
        for j,part in enumerate(PARTS):
            refs=[r for r in truth.get(part,[]) if r['pitch']==n['pitch']]
            if any(r.get('censored_onset') and r['onset']<=n['onset']<r['offset'] for r in refs):mask[i,j]=0.
            matches=[r for r in refs if not r.get('censored_onset') and abs(r['onset']-n['onset'])<=.12 and overlap(r,n)>=.15]
            if not matches:continue
            r=min(matches,key=lambda r:(abs(r['onset']-n['onset']),-overlap(r,n)))
            y[i,j]=1.;mask[i,j]=1.;quality[i,j]=overlap(r,n)
            delta[i,j]=np.clip([(r['onset']-n['onset'])/.12,(r['offset']-n['offset'])/.4],-1,1)
            timing_mask[i,j]=[1.,float(not r.get('censored_offset',False))]
            keep_correct[i,j]=float(n['baseline'][j] and abs(r['onset']-n['onset'])<=.05 and overlap(r,n)>=.6)
    return {'y':y,'quality':quality,'delta':delta,'timing_mask':timing_mask,'mask':mask,'keep_correct':keep_correct}

def unique_targets(pool,truth):
    """Select one representative per reference, so duplicate routes aren't all positive.

    A global assignment maximizes target coverage before minimizing onset and
    duration disagreement. Roles are independent, preserving cross-part unison.
    """
    from scipy.optimize import linear_sum_assignment
    result=targets(pool,truth)
    for key in ('y','quality','delta','timing_mask','keep_correct'):result[key].fill(0)
    for j,part in enumerate(PARTS):
        refs=[r for r in truth.get(part,[]) if not r.get('censored_onset')]
        if not refs or not pool:continue
        dummy_cost=2.*(len(refs)+1)
        cost=np.full((len(refs),len(pool)+len(refs)),1e9)
        cost[:,len(pool):]=dummy_cost
        for a,r in enumerate(refs):
            for b,n in enumerate(pool):
                if n['pitch']!=r['pitch'] or abs(n['onset']-r['onset'])>.12 or overlap(r,n)<.15:continue
                cost[a,b]=abs(n['onset']-r['onset'])/.12+.5*(1-overlap(r,n))-.03*n['baseline'][j]+1e-7*b
        for a,b in zip(*linear_sum_assignment(cost)):
            if b>=len(pool) or cost[a,b]>=dummy_cost:continue
            r,n=refs[a],pool[b];result['y'][b,j]=1.;result['mask'][b,j]=1.
            result['quality'][b,j]=overlap(r,n)
            result['delta'][b,j]=np.clip([(r['onset']-n['onset'])/.12,(r['offset']-n['offset'])/.4],-1,1)
            result['timing_mask'][b,j]=[1.,float(not r.get('censored_offset',False))]
            result['keep_correct'][b,j]=float(n['baseline'][j] and abs(r['onset']-n['onset'])<=.05 and overlap(r,n)>=.6)
    return result

class CandidateRanker(nn.Module):
    def __init__(self,dimension=1030,context=False,width=128):
        super().__init__();self.context=context;self.width=width
        self.input=nn.Sequential(nn.Linear(dimension,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(.1))
        if context:
            layer=nn.TransformerEncoderLayer(width,4,width*3,.1,batch_first=True,norm_first=True)
            self.core=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        else:self.core=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Dropout(.1))
        self.head=nn.Linear(width,8)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,prior,valid=None):
        z=self.input(x)
        z=self.core(z,src_key_padding_mask=~valid) if self.context and valid is not None else self.core(z)
        z=self.head(z)
        return {'logits':z[...,:2]+prior,'delta':z[...,2:6].reshape(*z.shape[:-1],2,2).tanh(),
                'quality':z[...,6:].sigmoid()}

def loss(output,target,negative_weight=1.):
    y=target['y'];mask=target['mask']
    weights=torch.where(y>.5,1.+target['keep_correct'],negative_weight)
    ce=torch.nn.functional.binary_cross_entropy_with_logits(output['logits'],y,reduction='none')
    classification=(ce*weights*mask).sum()/mask.sum().clamp_min(1)
    quality=((output['quality']-target['quality']).square()*mask).sum()/mask.sum().clamp_min(1)
    tm=target['timing_mask']
    timing=(torch.nn.functional.smooth_l1_loss(output['delta'],target['delta'],reduction='none')*tm).sum()/tm.sum().clamp_min(1)
    return classification+.3*quality+.3*timing

@torch.no_grad()
def predict(model,x,prior,mean,std,device='cpu',window=384,overlap=96):
    if not len(x):return {'probability':np.empty((0,2)), 'delta':np.empty((0,2,2)), 'quality':np.empty((0,2))}
    total={'probability':np.zeros((len(x),2)), 'delta':np.zeros((len(x),2,2)), 'quality':np.zeros((len(x),2))};count=np.zeros(len(x))
    for start in range(0,len(x),window-overlap):
        stop=min(len(x),start+window);z=np.clip((x[start:stop].astype(np.float32)-mean)/std,-8,8)
        result=model(torch.tensor(z,device=device)[None],torch.tensor(prior[start:stop],device=device)[None])
        result={'probability':result['logits'].sigmoid(), 'delta':result['delta'],'quality':result['quality']}
        for k,v in result.items():total[k][start:stop]+=v[0].cpu().numpy()
        count[start:stop]+=1
    return {k:v/count.reshape((-1,)+(1,)*(v.ndim-1)) for k,v in total.items()}

def propose(pool,prediction,duration,threshold=.5,quality_weight=0.,retime=False,nms=.05):
    parts={p:[] for p in PARTS}
    for j,part in enumerate(PARTS):
        choices=[]
        for i,n in enumerate(pool):
            score=float(prediction['probability'][i,j])*((1-quality_weight)+quality_weight*float(prediction['quality'][i,j]))
            if score<threshold:continue
            a,b=n['onset'],n['offset']
            if retime:
                da,db=prediction['delta'][i,j]*np.array([.12,.4]);a=max(0.,a+float(da));b=min(duration,b+float(db))
            if b-a<.03:continue
            choices.append({'pitch':n['pitch'],'onset':a,'offset':b,'velocity':n['velocity'],
                            'score':score,'baseline':n['baseline'][j]})
        accepted=[]
        for n in sorted(choices,key=lambda n:-(n['score']+.0001*n['baseline'])):
            if any(a['pitch']==n['pitch'] and abs(a['onset']-n['onset'])<=nms and overlap(a,n)>=.3 for a in accepted):continue
            accepted.append(n)
        parts[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in sorted(accepted,key=lambda n:(n['onset'],n['pitch']))]
    return parts

def refine(audio_inputs,midi_inputs,checkpoint,mode='review',device='cpu'):
    from .audio import features
    from .model import NoteCRNN
    from .temporal import encode_views,note_changes
    from .refine import candidates_from_midi
    from .score import read_midi
    if mode not in ('review','apply'):raise ValueError('Unknown mode')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['schema']!=SCHEMA:raise ValueError('Unknown candidate checkpoint')
    if mode=='apply' and not state.get('automatic_application_approved',False):raise ValueError('Unvalidated automatic application; use review')
    import soundfile as sf
    duration=sf.info(audio_inputs['mix']).duration
    original=read_midi(midi_inputs['mix']);pool=candidate_pool({k:candidates_from_midi(midi_inputs[k]) for k in ROUTES},duration)
    views=[features(audio_inputs[k]) for k in ROUTES];encoder=NoteCRNN().to(device);encoder.load_state_dict(state['encoder']);encoder.eval()
    e=encode_views(encoder,views,device);x,prior=candidate_features(views,e,pool,duration)
    model=CandidateRanker(context=state['context']);model.load_state_dict(state['model']);model.to(device).eval()
    prediction=predict(model,x,prior,np.array(state['mean'],np.float32),np.array(state['std'],np.float32),device)
    suggested=propose(pool,prediction,duration,**state['policy'])
    for part,notes in original.items():
        suggested[part].extend(copy.deepcopy(n) for n in notes if not PITCH_MIN<=n['pitch']<=PITCH_MAX or n['offset']-n['onset']<.03)
    return {'parts':original if mode=='review' else suggested,'baseline_parts':copy.deepcopy(original),
            'suggested_parts':suggested,'refinement_changes':note_changes(original,suggested),
            'refinement_mode':mode,'refiner_schema':SCHEMA,'automatic_application_approved':False,
            'limitations':'Existing candidate pitches; candidate retrieval limits recall. No validated bend, fingering or natural-band performance guarantee.'}
