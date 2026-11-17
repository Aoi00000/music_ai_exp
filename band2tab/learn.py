"""Supervised learning and note-level evaluation for the experimental CRNN."""
from pathlib import Path
import json, random, hashlib
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .model import NoteCRNN, load, predict
from .audio import decode, SR, HOP

def sample_item(items):
    part=random.choice(sorted({it["part"] for it in items}))
    return random.choice([it for it in items if it["part"]==part])

class Windows(Dataset):
    def __init__(self,items,steps,augment=False):
        self.items=items;self.steps=steps;self.augment=augment
        self.windows=[(i,start) for i,it in enumerate(items)
                      for start in range(0,self.read(it['cache'])[0].shape[0],344)]
    @staticmethod
    @lru_cache(maxsize=4)
    def read(path):
        with np.load(path) as z:return z['x'],z['y']
    def __len__(self):return self.steps if self.augment else len(self.windows)
    def __getitem__(self,index):
        if self.augment:
            it=sample_item(self.items);x,y=self.read(it['cache'])
            start=random.randrange(max(1,len(x)-344+1))
        else:
            i,start=self.windows[index];it=self.items[i];x,y=self.read(it['cache'])
        xx=x[start:start+344].copy();yy=y[start:start+344].copy();mask=np.ones(len(xx),np.float32)
        if self.augment and random.random()<.5:
            # A controlled spectral augmentation; not claimed to reproduce Demucs artifacts.
            xx*=random.uniform(.8,1.2)
            xx+=np.random.normal(0,.03,xx.shape).astype('float32')
            xx=np.maximum(xx,0)
        if len(xx)<344:
            pad=344-len(xx);xx=np.pad(xx,((0,pad),(0,0)));yy=np.pad(yy,((0,pad),(0,0),(0,0)));mask=np.pad(mask,(0,pad))
        return torch.from_numpy(xx),torch.from_numpy(yy),torch.tensor(it['part']=='bass',dtype=torch.long),torch.from_numpy(mask)

def train(manifest='data/manifest.json',out='runs/clean',epochs=5,steps=100,batch=4,lr=.001,device='cuda',init=None,augment=False,dataset='all',resume=None):
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    items=json.loads(Path(manifest).read_text());items=[i for i in items if dataset=='all' or i['dataset']==dataset]
    tr=[i for i in items if i['split']=='train'];va=[i for i in items if i['split']=='val']
    if not tr or not va:raise ValueError('Training and validation data required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Check the assigned VM.')
    if init and resume:raise ValueError('Choose --init or --resume')
    model=load(init or resume,device)[0] if (init or resume) else NoteCRNN().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    # Clip imbalance weights to avoid assigning extreme importance to a few labels.
    pos=np.zeros((3,66),np.float64);total=0
    for it in tr:
        with np.load(it['cache']) as z:pos+=z['y'].sum(0);total+=len(z['y'])
    pw=torch.tensor(np.clip((total-pos)/np.maximum(pos,1),1,20),dtype=torch.float32,device=device)
    headw=torch.tensor([3.,1.,1.],device=device)[None,None,:,None]
    def loss(logits,y,mask):
        l=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pw,reduction='none')
        return (l*mask[:,:,None,None]*headw).sum()/(mask.sum()*66*5)
    # Sampling is random in both conditions; only spectral augmentation differs.
    class Controlled(Windows):
        def __getitem__(self,idx):
            if augment:return super().__getitem__(idx)
            it=sample_item(self.items);x,y=self.read(it['cache']);s=random.randrange(max(1,len(x)-344+1))
            xx=x[s:s+344];yy=y[s:s+344];m=np.ones(len(xx),np.float32)
            if len(xx)<344:
                p=344-len(xx);xx=np.pad(xx,((0,p),(0,0)));yy=np.pad(yy,((0,p),(0,0),(0,0)));m=np.pad(m,(0,p))
            return torch.from_numpy(xx.copy()),torch.from_numpy(yy.copy()),torch.tensor(it['part']=='bass',dtype=torch.long),torch.from_numpy(m)
    dl=DataLoader(Controlled(tr,steps*batch,augment=True),batch_size=batch,num_workers=0)
    vl=DataLoader(Windows(va,0),batch_size=batch,num_workers=0)
    fingerprint=hashlib.sha256(json.dumps([{k:it.get(k) for k in ('id','dataset','part','split','group','notes')} for it in items],sort_keys=True).encode()).hexdigest()
    best=float('inf');history=[];start_epoch=0
    if resume:
        saved=torch.load(resume,map_location=device,weights_only=True)
        if 'optimizer' not in saved:raise ValueError('--resume requires last.pt, not model-only best.pt')
        if saved['dataset_fingerprint']!=fingerprint:raise ValueError('Resume data differs from original run')
        cfg=saved['train_config']
        if (cfg['steps'],cfg['batch'],cfg['augment'],cfg['dataset'])!=(steps,batch,augment,dataset):
            raise ValueError('Resume training settings differ')
        opt.load_state_dict(saved['optimizer']);best=saved['best_validation_bce'];history=saved['history'];start_epoch=saved['epoch']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        nr=saved['numpy_rng'];np.random.set_state((nr['name'],np.array(nr['keys'],dtype=np.uint32),nr['pos'],nr['has_gauss'],nr['cached_gaussian']))
        if torch.cuda.is_available() and saved.get('cuda_rng'):torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        print('Continuing after completed epoch',start_epoch,flush=True)
    def atomic_save(obj,path):
        path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');torch.save(obj,temporary);temporary.replace(path)
    for epoch in range(start_epoch,epochs):
        model.train();running=[]
        for x,y,part,mask in dl:
            opt.zero_grad(set_to_none=True);l=loss(model(x.to(device),part.to(device)),y.to(device),mask.to(device))
            l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();running.append(l.item())
        model.eval();valid=[]
        with torch.no_grad():
            for x,y,part,mask in vl:valid.append(loss(model(x.to(device),part.to(device)),y.to(device),mask.to(device)).item())
        row={'epoch':epoch+1,'train_bce':float(np.mean(running)),'val_bce':float(np.mean(valid))};history.append(row);print(row,flush=True)
        weights={'model':model.state_dict(),'schema':2,'sr':SR,'hop':HOP,'pitch_min':23,'pitch_max':88,'augment':augment,'epoch':epoch+1}
        if row['val_bce']<best:
            best=row['val_bce'];atomic_save(weights,out/'best.pt')
        atomic_save(weights,out/f'epoch_{epoch+1:02d}.pt')
        nr=np.random.get_state()
        state={**weights,'optimizer':opt.state_dict(),'best_validation_bce':best,'history':history,
               'dataset_fingerprint':fingerprint,'train_config':{'steps':steps,'batch':batch,'augment':augment,'dataset':dataset,'lr':lr},
               'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'numpy_rng':{'name':nr[0],'keys':nr[1].tolist(),'pos':nr[2],'has_gauss':nr[3],'cached_gaussian':nr[4]}}
        atomic_save(state,out/'last.pt')
        temp=out/'history.json.tmp';temp.write_text(json.dumps(history,indent=2));temp.replace(out/'history.json')
    return {'checkpoint':str(out/'best.pt'),'validation_bce':best}

def pitch_set_metrics(ref,est,hop=.02):
    end=max([n['offset'] for n in ref+est]+[hop]);frames=max(1,int(np.ceil(end/hop)))
    def roll(notes):
        r=np.zeros((frames,128),bool)
        for n in notes:
            a=max(0,int(np.ceil(n['onset']/hop-1e-9)));b=min(frames,int(np.ceil(n['offset']/hop-1e-9)))
            r[a:b,int(n['pitch'])]=True
        return r
    r,e=roll(ref),roll(est);tp=int((r&e).sum());fp=int((~r&e).sum());fn=int((r&~e).sum())
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);active=(r|e).any(1)
    polyphony={}
    count=r.sum(1)
    for name,selected in [('silence',count==0),('single',count==1),('double',count==2),('poly3plus',count>=3)]:
        rr,ee=r[selected],e[selected];t=int((rr&ee).sum());f=int((~rr&ee).sum());n=int((rr&~ee).sum())
        polyphony[name]={'frames':int(selected.sum()),'tp':t,'fp':f,'fn':n,
                         'frame_f1':2*t/(2*t+f+n) if 2*t+f+n else None}
    return {'frame_precision':precision,'frame_recall':recall,
            'frame_f1':2*tp/max(1,2*tp+fp+fn),
            'active_exact_match':float((r[active]==e[active]).all(1).mean()) if active.any() else 1.,
            'frame_hop_seconds':hop,'by_reference_polyphony':polyphony}

def note_metrics(ref,est,with_offset=False):
    import mir_eval
    ri=np.array([[n['onset'],n['offset']] for n in ref],float).reshape(-1,2)
    ei=np.array([[n['onset'],n['offset']] for n in est],float).reshape(-1,2)
    rp=440*2**((np.array([n['pitch'] for n in ref])-69)/12)
    ep=440*2**((np.array([n['pitch'] for n in est])-69)/12)
    frame=pitch_set_metrics(ref,est)
    if not len(ref):return {'precision':float(not est),'recall':1.,'f1':float(not est),**frame}
    if not len(est):return {'precision':0.,'recall':0.,'f1':0.,**frame}
    p,r,f,_=mir_eval.transcription.precision_recall_f1_overlap(ri,rp,ei,ep,onset_tolerance=.05,
               offset_ratio=.2 if with_offset else None,offset_min_tolerance=.05)
    return {'precision':float(p),'recall':float(r),'f1':float(f),**frame}

def evaluate(checkpoint,manifest='data/manifest.json',split='val',out='reports/eval.json',device='cpu',threshold=.5,tune=False,frame_threshold=None,offset_threshold=None):
    if tune and split!='val':raise ValueError('Tune thresholds on validation only')
    items=[it for it in json.loads(Path(manifest).read_text()) if it['split']==split]
    if not items:raise ValueError('No evaluation examples')
    model,_=load(checkpoint,device);cached=[]
    for it in items:
        with np.load(it['cache']) as z:p=predict(model,z['x'],int(it['part']=='bass'),device)
        cached.append((it,p))
    candidates=[(o,f,.5) for o in (.03,.05,.1,.2,.3,.5,.7) for f in (.1,.3,.5)] if tune else [(threshold,frame_threshold if frame_threshold is not None else threshold,offset_threshold if offset_threshold is not None else threshold)];results=[]
    for th,fr,off in candidates:
        rows=[]
        for it,p in cached:
            est=decode(p,onset=th,frame=fr,offset=off)
            est=[n for n in est if 23<=n['pitch']<=88]
            rows.append({'id':it['id'],'dataset':it['dataset'],'part':it['part'],'group':it['group'],'reference_notes':len(it['notes']),'estimated_notes':len(est),
                         **note_metrics(it['notes'],est),'offset_f1':note_metrics(it['notes'],est,True)['f1']})
        results.append({'threshold':th,'frame_threshold':fr,'offset_threshold':off,'macro_note_f1':float(np.mean([r['f1'] for r in rows])),
                        'macro_offset_f1':float(np.mean([r['offset_f1'] for r in rows])),'tracks':rows})
    best=max(results,key=lambda x:x['macro_note_f1']);best.update(split=split,checkpoint=str(checkpoint),tuned_on_validation=tune,validation_candidates=[{k:v for k,v in r.items() if k!='tracks'} for r in results],manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),pitch_range_midi=[23,88],minimum_note_seconds=.03)
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    predictions=dest.parent/(dest.stem+'_predictions');predictions.mkdir(exist_ok=True)
    for it,p in cached:
        est=decode(p,onset=best['threshold'],frame=best['frame_threshold'],offset=best['offset_threshold'])
        (predictions/(it['id']+'.json')).write_text(json.dumps({'id':it['id'],'notes':est}))
    dest.write_text(json.dumps(best,indent=2))
    print({k:v for k,v in best.items() if k not in ('tracks','validation_candidates')},flush=True)
    return best

def benchmark(engine,binary,manifest='data/manifest_clean.json',split='val',out='reports/baseline.json',model='medium',device='cuda',separator=None,limit=0):
    """Run an unchanged public model against exactly the same note labels."""
    import subprocess
    from .score import read_midi
    items=[it for it in json.loads(Path(manifest).read_text()) if it['split']==split]
    if limit:items=items[:limit]
    if not items:raise ValueError('No benchmark data')
    out=Path(out);cache=out.parent/(out.stem+'_predictions');cache.mkdir(parents=True,exist_ok=True)
    rows=[]
    for it in items:
        folder=cache/it['id'];folder.mkdir(exist_ok=True);source=Path(it['audio'])
        if engine=='muscriptor':
            dest=folder/'notes.mid';cmd=[binary,'transcribe',str(source),'--model',model,'--device',device,
                 '--detect-tempo','false','--output',str(dest)]
            if device.startswith('cuda'):cmd+=['--dtype','float16']
        else:
            if separator:
                subprocess.run([separator,'-m','demucs','-n','htdemucs_6s','--device',device,
                    '--segment','7','--shifts','1','-o',str(folder/'sep'),str(source)],check=True)
                source=folder/'sep'/'htdemucs_6s'/source.stem/(it['part']+'.wav')
            dest=None;cmd=[binary,str(folder/'bp'),str(source)];(folder/'bp').mkdir(exist_ok=True)
        subprocess.run(cmd,check=True)
        if dest is None:dest=next((folder/'bp').glob('*.mid'))
        parts=read_midi(dest,it['part'] if engine=='basic-pitch' else None);est=parts.get(it['part'],[])
        est=[n for n in est if 23<=n['pitch']<=88]
        rows.append({'id':it['id'],'dataset':it['dataset'],'part':it['part'],'group':it['group'],'reference_notes':len(it['notes']),'estimated_notes':len(est),
                     **note_metrics(it['notes'],est),'offset_f1':note_metrics(it['notes'],est,True)['f1']})
    result={'engine':engine,'model':model,'separator':separator,'manifest':str(manifest),'split':split,
            'tracks':rows,'macro_note_f1':float(np.mean([r['f1'] for r in rows])),
            'macro_offset_f1':float(np.mean([r['offset_f1'] for r in rows]))}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print('Saved',out)
    return result
