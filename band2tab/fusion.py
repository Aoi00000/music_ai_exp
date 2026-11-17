"""Independent per-instrument note proposals using mix and separated evidence."""
import copy
import numpy as np
import torch
from torch import nn
from .refine import event_features

SCHEMA='separation_event_fusion_v1'
PARTS=('guitar','bass')

def overlap(a,b):
    lo=max(a['onset'],b['onset']);hi=min(a['offset'],b['offset'])
    return max(0.,hi-lo)/max(1e-9,max(a['offset'],b['offset'])-min(a['onset'],b['onset']))

def candidate_pool(original,separated):
    """Retain baseline events; match stem support, proposing only new candidates.

    Separate pools allow a unison pitch in both instruments. Matching never
    overwrites an original note's onset/offset. Every addition has provenance.
    """
    pools={}
    for cls,part in enumerate(PARTS,1):
        notes=[{**copy.deepcopy(n),'part':part,'keep':n['baseline_class']==cls,
                'origin':'mix','support':[1.,0.,0.]} for n in original]
        for n in separated.get(part,[]):
            matches=[(i,a) for i,a in enumerate(notes) if a['pitch']==n['pitch'] and abs(a['onset']-n['onset'])<=.05 and overlap(a,n)>=.3]
            if matches:
                i,a=min(matches,key=lambda pair:(not pair[1]['keep'],abs(pair[1]['onset']-n['onset']),-overlap(pair[1],n)))
                notes[i]['support'][cls]=1.
            else:
                notes.append({**copy.deepcopy(n),'baseline_class':0,'part':part,'keep':False,
                              'origin':'separated_'+part,'support':[0.,float(cls==1),float(cls==2)]})
        pools[part]=notes
    return pools

def acoustic_features(inputs,pools):
    from .audio import HOP,SR
    notes=[n for p in PARTS for n in pools[p]]
    blocks=[event_features(inputs[k],notes) for k in ('mix','guitar','bass')]
    duration=max(HOP/SR,(len(inputs['mix'])-1)*HOP/SR)
    extra=np.array([[float(n['part']==p) for p in PARTS]+[float(n['keep'])]+n['support']+
                    [max(0.,n['onset']-duration)/duration,max(0.,n['offset']-duration)/duration]
                    for n in notes],np.float32).reshape(len(notes),8)
    return np.concatenate([*blocks,extra],axis=1)

class EventFusion(nn.Module):
    def __init__(self,dimension=254):
        super().__init__();self.net=nn.Sequential(nn.Linear(dimension,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,keep):return self.net(x).squeeze(-1)+4*keep-2

def propose(pools,probability,add_threshold=1.,remove_threshold=-1.):
    result={p:[] for p in PARTS};changes=[];i=0
    for part in PARTS:
        for n in pools[part]:
            p=float(probability[i]);i+=1
            keep=n['keep']
            accepted=(p>=remove_threshold) if keep else (add_threshold<1. and p>=add_threshold)
            if accepted:
                result[part].append({k:n[k] for k in ('pitch','onset','offset','velocity')})
            if bool(accepted)!=bool(keep):
                changes.append({'part':part,'action':'add' if accepted else 'remove',
                                'origin':n['origin'],'score':p,'note':{k:n[k] for k in ('pitch','onset','offset')}})
    if i!=len(probability):raise ValueError('Candidate/probability length mismatch')
    return result,changes

def refine(audio_inputs,midi_inputs,checkpoint,mode='review',device='cpu'):
    """Review proposals without changing the source MIDI or default score.

    Both mappings require mix/guitar/bass paths. This API consumes cached
    separation and transcription results; it does not download or run models.
    """
    from pathlib import Path
    import hashlib
    from .audio import features,PITCH_MIN,PITCH_MAX
    from .score import read_midi
    from .refine import candidates_from_midi
    if mode not in ('review','apply'):raise ValueError('Unknown review mode')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['schema']!=SCHEMA:raise ValueError('Unknown fusion schema')
    approved=bool(state.get('automatic_application_approved',False))
    if mode=='apply' and not approved:raise ValueError('Automatic application is not validated; use review')
    original=read_midi(midi_inputs['mix'])
    pools=candidate_pool(candidates_from_midi(midi_inputs['mix']),
                         {p:candidates_from_midi(midi_inputs[p]) for p in PARTS})
    x=acoustic_features({k:features(audio_inputs[k]) for k in ('mix',*PARTS)},pools)
    model=EventFusion().to(device);model.load_state_dict(state['model']);model.eval()
    z=np.clip((x-np.asarray(state['mean']))/np.asarray(state['std']),-8,8).astype('float32')
    keep=[float(n['keep']) for p in PARTS for n in pools[p]]
    with torch.no_grad():probability=model(torch.from_numpy(z).to(device),torch.tensor(keep,device=device)).sigmoid().cpu().numpy()
    suggested,changes=propose(pools,probability,**state['policy'])
    for part,ns in original.items():
        suggested[part].extend(copy.deepcopy(n) for n in ns if not PITCH_MIN<=n['pitch']<=PITCH_MAX or n['offset']-n['onset']<.03)
    return {'parts':suggested if mode=='apply' else original,'baseline_parts':copy.deepcopy(original),
            'suggested_parts':suggested,'refinement_changes':changes,'refinement_mode':mode,
            'automatic_application_approved':approved,'refiner_schema':SCHEMA,
            'refiner_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'limitations':'Missing notes can be proposed from separated transcriptions. No continuous pitch, technique or original timing correction.'}

def main():
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser()
    for kind in ('mix','guitar','bass'):
        p.add_argument('--'+kind+'-audio',required=True);p.add_argument('--'+kind+'-midi',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audio={k:getattr(a,k+'_audio') for k in ('mix',*PARTS)};midi={k:getattr(a,k+'_midi') for k in ('mix',*PARTS)}
    result=refine(audio,midi,a.checkpoint);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print('Proposals',len(result['refinement_changes']))

if __name__=='__main__':main()
