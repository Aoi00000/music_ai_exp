"""Supervised temporal residual transcription on top of frozen MuScriptor.

The A8 acoustic encoder is reused, while a sequence model learns onsets,
sounding pitches and offsets. No reinforcement learning is used.
"""
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .audio import SR,HOP,N_PITCHES,PITCH_MIN,PITCH_MAX,features,decode
from .model import NoteCRNN

SCHEMA='temporal_residual_acr_v1';PARTS=('guitar','bass')

def piano_roll(parts,frames,reference=False):
    y=np.zeros((frames,2,3,N_PITCHES),np.float32)
    for part_index,part in enumerate(PARTS):
        for n in parts.get(part,[]):
            p=n['pitch']-PITCH_MIN
            if not 0<=p<N_PITCHES:continue
            a=round(n['onset']*SR/HOP);b=round(n['offset']*SR/HOP)
            lo=max(0,a);hi=min(frames,b)
            if lo>=hi:continue
            y[lo:hi,part_index,1,p]=1.
            if 0<=a<frames and not (reference and n.get('censored_onset')):y[a,part_index,0,p]=1.
            if 0<b<=frames and not (reference and n.get('censored_offset')):y[b-1,part_index,2,p]=1.
    return y

class TemporalFusion(nn.Module):
    def __init__(self,initial=None,include_ft=True):
        super().__init__();self.include_ft=include_ft
        self.adapter=nn.Linear(3*128+3*N_PITCHES,128)
        self.inst=nn.Embedding(2,8)
        self.gru=nn.GRU(136,128,num_layers=2,batch_first=True,bidirectional=True,dropout=.1)
        self.head=nn.Linear(256,3*N_PITCHES)
        with torch.no_grad():
            self.adapter.weight.zero_();self.adapter.bias.zero_()
            for i in range(3):self.adapter.weight[:,128*i:128*(i+1)]=torch.eye(128)/3
            self.head.weight.zero_();self.head.bias.zero_()
        if initial:
            self.inst.load_state_dict({k.removeprefix('inst.'):v for k,v in initial.items() if k.startswith('inst.')})
            self.gru.load_state_dict({k.removeprefix('gru.'):v for k,v in initial.items() if k.startswith('gru.')})

    def forward(self,encoded,prior,part):
        # encoded: B,T,[mix,guitar,bass,ft_clean],128; prior: B,T,3,66.
        indices=(part+1)[:,None,None,None].expand(-1,encoded.shape[1],1,128)
        separated=encoded.gather(2,indices).squeeze(2)
        clean=encoded[:,:,3] if self.include_ft else encoded[:,:,0]
        z=torch.cat([encoded[:,:,0],separated,clean,prior.flatten(2)],-1)
        z=torch.relu(self.adapter(z));e=self.inst(part)[:,None].expand(-1,z.shape[1],-1)
        z,_=self.gru(torch.cat([z,e],-1))
        residual=self.head(z).reshape(*prior.shape)
        return residual+(prior*2-1)*2.

@torch.no_grad()
def encode_views(encoder,views,device='cpu',batch_frames=344):
    length=min(len(v) for v in views);out=[]
    for view in views:
        blocks=[]
        for start in range(0,length,batch_frames):
            x=torch.as_tensor(view[start:start+batch_frames],device=device)[None,None]
            z=encoder.cnn(x).permute(0,2,1,3).flatten(2)
            blocks.append(torch.relu(encoder.proj(z))[0].cpu().numpy())
        out.append(np.concatenate(blocks))
    return np.stack(out,axis=1).astype(np.float16)

@torch.no_grad()
def predict_scales(model,encoded,prior,scales=(1.,),device='cpu',window=344,overlap=86):
    totals={s:np.zeros_like(prior,dtype=np.float32) for s in scales};count=np.zeros(len(prior),np.float32)
    for start in range(0,len(prior),window-overlap):
        e=torch.tensor(encoded[start:start+window],dtype=torch.float32,device=device)[None]
        for part in (0,1):
            m=torch.tensor(prior[start:start+window,part],dtype=torch.float32,device=device)[None]
            logits=model(e,m,torch.tensor([part],device=device));base=(m*2-1)*2
            for scale in scales:
                adjusted=logits if scale==1. else base+scale*(logits-base)
                p=adjusted.sigmoid()[0].cpu().numpy();totals[scale][start:start+len(p),part]+=p
        count[start:start+len(e[0])]+=1
    return {s:total/np.maximum(count[:,None,None,None],1) for s,total in totals.items()}

def predict(model,encoded,prior,device='cpu',window=344,overlap=86,residual_scale=1.):
    return predict_scales(model,encoded,prior,(residual_scale,),device,window,overlap)[residual_scale]

def decode_parts(probability,policy):
    return {part:decode(probability[:,j],**policy) for j,part in enumerate(PARTS)}

def activity_parts(probability,threshold=.5):
    """Frame-only diagnostic: separate acoustic recognition from note decoding."""
    result={p:[] for p in PARTS};dt=HOP/SR
    for j,part in enumerate(PARTS):
        for k in range(N_PITCHES):
            active=probability[:,j,1,k]>=threshold
            edges=np.diff(np.r_[False,active,False].astype(np.int8))
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                result[part].append({'pitch':int(k+PITCH_MIN),'onset':float(a*dt),'offset':float(b*dt),'velocity':80})
    return result

def note_changes(before,after):
    import mir_eval
    changes=[]
    for part in PARTS:
        old=before.get(part,[]);new=after.get(part,[])
        intervals=lambda ns:np.array([[n['onset'],n['offset']] for n in ns],float).reshape(-1,2)
        pitches=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
        pairs=mir_eval.transcription.match_notes(intervals(old),pitches(old),intervals(new),pitches(new),onset_tolerance=.1,offset_ratio=None) if old and new else []
        used_old={a for a,_ in pairs};used_new={b for _,b in pairs}
        for a,b in pairs:
            if max(abs(old[a]['onset']-new[b]['onset']),abs(old[a]['offset']-new[b]['offset']))>.02:
                changes.append({'part':part,'action':'retime','before':old[a],'after':new[b]})
        changes.extend({'part':part,'action':'remove','before':n} for i,n in enumerate(old) if i not in used_old)
        changes.extend({'part':part,'action':'add','after':n} for i,n in enumerate(new) if i not in used_new)
    return changes

def masked_loss(logits,target,mask):
    pos=torch.tensor([20.,3.,20.],device=logits.device)[None,None,:,None]
    heads=torch.tensor([2.,1.,1.],device=logits.device)[None,None,:,None]
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),target.float(),pos_weight=pos,reduction='none')
    return (loss*mask[:,:,None,None]*heads).sum()/mask.sum().clamp_min(1)/(4*N_PITCHES)

def known_offset_score(truth,estimated,start=0.,end=float('inf')):
    """Do not pretend an excerpt's cut point is the original note release."""
    import mir_eval
    from .learn import note_metrics
    tp=fp=fn=references=0
    for part in PARTS:
        refs=[n for n in truth.get(part,[]) if start<=n['onset']<end and not n.get('censored_onset')]
        est=[n for n in estimated.get(part,[]) if start<=n['onset']<end]
        intervals=lambda ns:np.array([[n['onset'],n['offset']] for n in ns],float).reshape(-1,2)
        pitches=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
        pairs=mir_eval.transcription.match_notes(intervals(refs),pitches(refs),intervals(est),pitches(est),onset_tolerance=.05,offset_ratio=None) if refs and est else []
        ignored={b for a,b in pairs if refs[a].get('censored_offset')}
        refs=[n for n in refs if not n.get('censored_offset')];est=[n for i,n in enumerate(est) if i not in ignored]
        score=note_metrics(refs,est,True,include_frames=False);correct=round(score['precision']*len(est))
        tp+=correct;fp+=len(est)-correct;fn+=len(refs)-correct;references+=len(refs)
    return {'offset_f1':2*tp/max(1,2*tp+fp+fn),'known_offset_reference_notes':references}

def refine(audio_inputs,midi,checkpoint,device='cpu',mode='review'):
    from .score import read_midi
    if mode not in ('review','apply'):raise ValueError('Unknown mode')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['schema']!=SCHEMA:raise ValueError('Unknown temporal checkpoint')
    if mode=='apply' and not state.get('automatic_application_approved',False):raise ValueError('Use review; automatic application is not validated')
    original=read_midi(midi);encoder=NoteCRNN().to(device);encoder.load_state_dict(state['encoder']);encoder.eval()
    views=[features(audio_inputs[k]) for k in ('mix','guitar','bass','ft_clean')]
    encoded=encode_views(encoder,views,device);prior=piano_roll(original,len(encoded))
    model=TemporalFusion(include_ft=state['include_ft']).to(device);model.load_state_dict(state['model']);model.eval()
    p=predict(model,encoded,prior,device,residual_scale=state.get('residual_scale',1.));suggested=decode_parts(p,state['policy'])
    for part,notes in original.items():
        suggested[part].extend(copy.deepcopy(n) for n in notes if not PITCH_MIN<=n['pitch']<=PITCH_MAX or n['offset']-n['onset']<.03)
    return {'parts':suggested if mode=='apply' else original,'baseline_parts':copy.deepcopy(original),
            'suggested_parts':suggested,'refinement_mode':mode,'automatic_application_approved':False,
            'refinement_changes':note_changes(original,suggested),'refiner_schema':SCHEMA,
            'limitations':'Discrete notes in B0-E6. No validated bends, playing techniques, fingering or fully automatic score acceptance.'}

def main():
    """Portable replay of a backed-up case; does not start Azure or a separator."""
    import argparse,json
    p=argparse.ArgumentParser(description='Review temporal corrections from saved audio and MuScriptor MIDI')
    p.add_argument('--manifest',required=True,help='Use the verified manifest_mac.json on Mac')
    p.add_argument('--id',required=True);p.add_argument('--midi',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--mode',choices=('review','apply'),default='review')
    a=p.parse_args();rows=json.loads(Path(a.manifest).read_text())
    row=next((r for r in rows if r['id']==a.id),None)
    if row is None:p.error('Case is absent from the manifest: '+a.id)
    paths={k:row['inputs'][k] for k in ('mix','guitar','bass','ft_clean')}
    missing=[str(path) for path in paths.values() if not Path(path).is_file()]
    if missing:p.error('Missing audio; verify/rebase the Mac backup first: '+', '.join(missing))
    torch.set_num_threads(4)
    result=refine(paths,a.midi,a.checkpoint,device=a.device,mode=a.mode)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('TEMPORAL_REVIEW_SAVED',out,'automatic_application_approved',result['automatic_application_approved'])

if __name__=='__main__':main()
