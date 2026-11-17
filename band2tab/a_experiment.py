"""Checks for operator A. Smoke scores measure memorization, never generalization."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,hashlib,time
import numpy as np
from .audio import SR,HOP,N_BINS,N_PITCHES,decode

def save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))

def audit(manifest,out):
    import soundfile as sf
    items=json.loads(Path(manifest).read_text());errors=[];groups={};stats=Counter();notes=Counter();durations=Counter()
    for it in items:
        g=it['group'];split=it['split'];key=f'{it["dataset"]}/{it["part"]}/{split}'
        if g in groups and groups[g]!=split:errors.append([it['id'],'cross_split_group'])
        groups[g]=split
        info=sf.info(it['audio']);duration=info.frames/info.samplerate;durations[key]+=duration
        with np.load(it['cache']) as z:
            x,y=z['x'],z['y']
            if x.shape!=(len(x),N_BINS) or y.shape!=(len(x),3,N_PITCHES):errors.append([it['id'],'shape'])
            if not np.isfinite(x).all() or not np.isfinite(y).all():errors.append([it['id'],'non_finite'])
            if not (x.max()>0):errors.append([it['id'],'silent_audio_with_notes'])
            if not np.isin(y,[0,1]).all():errors.append([it['id'],'invalid_target'])
            if abs(len(x)*HOP/SR-duration)>.1:errors.append([it['id'],'feature_audio_duration'])
        for n in it['notes']:
            if not(0<=n['onset']<n['offset']<=duration+.1):errors.append([it['id'],'invalid_note_time',n])
            if not 23<=n['pitch']<=88:errors.append([it['id'],'out_of_range_pitch'])
        stats[key]+=1;notes[key]+=len(it['notes'])
    result={'manifest':str(manifest),'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            'tracks':len(items),'independent_groups':len(groups),'counts':dict(stats),
            'note_counts':dict(notes),'audio_seconds':dict(durations),'errors':errors,'passed':not errors}
    save(out,result);print({k:v for k,v in result.items() if k not in ('errors','audio_seconds','note_counts')},flush=True)
    if errors:raise RuntimeError(f'Audit found {len(errors)} errors; inspect {out}')
    return result

def smoke(manifest,out,steps=220,device='cuda'):
    import torch
    from .model import NoteCRNN
    from .learn import note_metrics
    torch.manual_seed(17);torch.set_num_threads(4)
    items=json.loads(Path(manifest).read_text());xs=[];ys=[];masks=[];parts=[];refs=[];chosen=[]
    for part in ('guitar','bass'):
        it=next(i for i in items if i['split']=='train' and i['part']==part)
        with np.load(it['cache']) as z:
            first=int(np.where(z['y'][:,0].any(1))[0][0]);a=max(0,first-8);x=z['x'][a:a+344];y=z['y'][a:a+344]
        valid=len(x);mask=np.zeros(344,np.float32);mask[:valid]=1
        xs.append(np.pad(x,((0,344-valid),(0,0))));ys.append(np.pad(y,((0,344-valid),(0,0),(0,0))));masks.append(mask)
        parts.append(int(part=='bass'));start=a*HOP/SR;end=(a+valid)*HOP/SR
        refs.append([{'pitch':n['pitch'],'onset':n['onset']-start,'offset':min(n['offset'],end)-start}
                     for n in it['notes'] if start<=n['onset']<end and n['offset']>start])
        chosen.append({'id':it['id'],'start':start,'part':part})
    x=torch.tensor(np.array(xs),device=device);y=torch.tensor(np.array(ys),device=device)
    mask=torch.tensor(np.array(masks),device=device);part=torch.tensor(parts,device=device)
    m=NoteCRNN().to(device);opt=torch.optim.AdamW(m.parameters(),lr=.003)
    pos=y.sum((0,1));total=mask.sum();pw=((total-pos)/pos.clamp_min(1)).clamp(1,20)
    hw=torch.tensor([3.,1.,1.],device=device)[None,None,:,None]
    def loss():
        z=m(x,part);b=torch.nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=pw,reduction='none')
        return (b*hw*mask[:,:,None,None]).sum()/(mask.sum()*66*5)
    m.eval();initial=float(loss().detach());m.train();begin=time.monotonic()
    for step in range(steps):
        opt.zero_grad(set_to_none=True);l=loss();l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
        if (step+1)%50==0:print('smoke step',step+1,'loss',float(l.detach()),flush=True)
    m.eval()
    with torch.no_grad():final=float(loss());p=m(x,part).sigmoid().cpu().numpy()
    scores=[]
    for th in (.05,.1,.2,.3,.5):
        metrics=[note_metrics(refs[i],decode(p[i,:int(masks[i].sum())],onset=th,frame=th,offset=th)) for i in range(2)]
        scores.append({'threshold':th,'mean_note_f1':float(np.mean([r['f1'] for r in metrics])),'by_part':metrics})
    best=max(scores,key=lambda s:s['mean_note_f1'])
    result={'type':'TRAINING_SET_MEMORIZATION_CHECK_NOT_TEST_ACCURACY','chosen':chosen,
            'initial_loss':initial,'final_loss':final,'steps':steps,'seconds':time.monotonic()-begin,
            'best':best,'passed':final<initial*.6 and best['mean_note_f1']>=.5}
    save(out,result);print(result,flush=True)
    if not result['passed']:raise RuntimeError('Real-data learnability check did not pass; inspect before full training')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','smoke']);p.add_argument('--manifest',default='data/manifest.json');p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=220);a=p.parse_args()
    audit(a.manifest,a.out) if a.command=='audit' else smoke(a.manifest,a.out,a.steps)
if __name__=='__main__':main()
