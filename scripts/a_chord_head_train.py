"""Target a measured chord-vocabulary gap with supervised frozen-backbone transfer."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,random
import numpy as np,soundfile as sf,torch
from chordchart.core import cqt,target,decode,DT,SIZES,signature,dictionary
from chordchart.model import load_original
from chordchart.infer import extract_audio
from scripts.a_candidate_data import save,sha
from scripts.a_stress_data import local
from scripts.a_chord_diagnostic import measure


def main():
    root=Path('backups/azure').resolve();folder=Path('reports/chord_head_v2');run=Path('runs/chord_head_v2');cache_dir=Path('data/chord_head_v2/cache')
    if folder.exists() or run.exists():raise ValueError('Do not overwrite or extend frozen chord-head experiment')
    groups=defaultdict(list)
    for r in json.load(open(root/'data/manifest_v2_techs_full.json')):
        if r['dataset']=='guitarset' and r['split']=='train' and r['id'].endswith('_comp'):groups[(r['group'],r['id'].split('_')[1])].append(r)
    train=[min(v,key=lambda r:hashlib.sha256(('chord-head-v1-'+r['id']).encode()).hexdigest()) for k,v in sorted(groups.items())]
    val_ids=json.load(open('reports/chord_diagnostic_v1/protocol.json'))['selected_ids'];originals={r['id']:r for r in json.load(open(root/'data/manifest_v2_techs_full.json'))};rows=train+[originals[i] for i in val_ids]
    protocol=dict(train_ids=[r['id'] for r in train],validation_ids=val_ids,source='Existing GuitarSet performed-chord annotations. One fixed-hash tempo per group x performer. Project progression split retained.',
        conditions=['uniform','power_only','power_only_weight3'],frozen='Published fold0 CNN and bidirectional LSTM',learned='Compare full 192 -> 112 factor head against only 12 new power-triad rows; existing 100 outputs fixed in power_only arms',steps=500,monitors=[100,500],seed=31,lr=.001,weight_decay=.01,
        window=256,overlap=64,excerpt_seconds=7.,excerpt_start_fraction=.35,guard=.25,training_boundary_mask_seconds=.05,penalty=2.,
        gate='Root and triad weighted accuracy >= published five-model ensemble; improve root+bass+full-pitch-class-set exact time agreement.',
        preprocessing='Identical FFmpeg 22050Hz mono PCM16 conversion as published-ensemble pipeline before CQT. V1 direct-librosa-resampling experiment retained but excluded from same-converter ensemble comparison.',
        rationale='Measured roots are stronger than detailed qualities; baseline excludes power triads. Compare new rows without learning, full-head learning, new-rows-only learning, and new-rows-only power-weighted learning.',
        scope='Repeated project development validation; solo guitar three style/progression2 groups, not fresh natural-band or unseen-composition evidence',own_learning=True,reinforcement_learning=False,test_opened=False)
    save(folder/'protocol.json',protocol);torch.set_num_threads(2);model=load_original(0,power=True).eval();original=load_original(0,power=False).eval();data={};man=[]
    frozen_digest=lambda:hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items() if not k.startswith('final_fc1.'))).hexdigest()
    digest=frozen_digest();all_vocab={signature(l) for l in dictionary()};basic_vocab={signature(l) for l in dictionary() if target(l)[0]<73}
    for r in rows:
        audio=local(root,r['audio']);info=sf.info(audio);start=max(0,(info.duration-7)*.35);dest=cache_dir/(r['id']+'.npz');jam=root/'data/raw/guitarset'/(r['id'].removeprefix('guitarset_')+'.jams');anns=json.load(open(jam))['annotations'];performed=[a for a in anns if a['namespace']=='chord' and a['annotation_metadata'].get('version')==.9]
        if len(performed)!=1:raise ValueError('Expected performed-chord annotation')
        ref=[dict(start=s['time']-start,end=s['time']+s['duration']-start,label=s['value']) for s in performed[0]['data']];fp=dict(source_audio_sha256=sha(audio),annotation_sha256=sha(jam),frozen_digest=digest,start=start,window=256,overlap=64)
        if dest.exists():
            if json.load(open(dest.with_suffix('.json')))['fingerprint']!=fp:raise ValueError('Stale chord feature cache')
            with np.load(dest) as f:item={k:f[k].copy() for k in f.files}
        else:
            wav=cache_dir/(r['id']+'.wav');wav.parent.mkdir(parents=True,exist_ok=True)
            with sf.SoundFile(audio) as f:f.seek(round(start*info.samplerate));y=f.read(round(7*info.samplerate),dtype='float32',always_2d=True)
            sf.write(wav,y,info.samplerate,subtype='FLOAT');canonical=wav.with_name(wav.stem+'_canonical.wav');extract_audio(wav,canonical);feature,_=cqt(canonical);x=feature[:,18:270];n=len(x);starts=sorted(set(list(range(0,max(1,n-256+1),192))+[max(0,n-256)]));zs=[];ys=[];masks=[]
            for a in starts:
                k=min(256,n-a);chunk=np.pad(x[a:a+k],((0,256-k),(0,0)))
                with torch.inference_mode():z=model.lstm1(model.audio_feature_block(torch.from_numpy(chunk[None])))[0][0].numpy()
                tags=np.full((256,6),-100,np.int64);mask=np.zeros(256,bool)
                for t in range(k):
                    when=(a+t)*DT;label=next((s['label'] for s in ref if s['start']<=when<s['end']),'X');tags[t]=target(label)
                    mask[t]=.25<=when<6.75 and all(abs(when-s['start'])>.05 for s in ref)
                zs.append(z);ys.append(tags);masks.append(mask)
            item=dict(z=np.asarray(zs),y=np.asarray(ys),mask=np.asarray(masks),starts=np.asarray(starts),n=np.array(n));np.savez_compressed(dest,**item);save(dest.with_suffix('.json'),dict(fingerprint=fp,reference=ref))
        item['reference']=ref;data[r['id']]=item;man.append(dict(id=r['id'],group=r['group'],split=r['split'],source_sha256=sha(audio),cache=str(dest)))
        print('CHORD_HEAD_FEATURES',r['split'],r['id'],flush=True)
    save('data/chord_head_v2/manifest.json',man)
    def probabilities(head,item,power):
        with torch.no_grad():logits=head(torch.from_numpy(item['z']));chunks=[l.softmax(-1).numpy() for l in torch.split(logits,SIZES if power else (73,13,4,4,3,3),dim=-1)]
        n=int(item['n']);sums=[np.zeros((n,x.shape[-1]),np.float32) for x in chunks];cnt=np.zeros(n,np.float32)
        for j,a in enumerate(item['starts']):
            k=min(256,n-a);w=np.maximum(np.hanning(256),.05)[:k].astype(np.float32);cnt[a:a+k]+=w
            for i,p in enumerate(chunks):sums[i][a:a+k]+=p[j,:k]*w[:,None]
        return [p/cnt[:,None] for p in sums]
    def evaluate(head,power):
        tracks=[];by=defaultdict(list)
        for ident in val_ids:
            item=data[ident];segments=decode(probabilities(head,item,power),penalty=2.);s=measure(item['reference'],segments,.25,6.75);group=originals[ident]['group'];by[group].append(s);tracks.append(dict(id=ident,group=group,metrics=s,segments=segments))
        grouped={g:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for g,rs in by.items()};return dict(summary={k:float(np.mean([v[k] for v in grouped.values()])) for k in tracks[0]['metrics']},groups=grouped,tracks=tracks)
    ensemble=json.load(open('reports/chord_diagnostic_v1/comparison.json'));controls=dict(published_ensemble=ensemble['summary'],published_single=evaluate(original.final_fc1,False)['summary'],extended_without_learning=evaluate(model.final_fc1,True)['summary']);results=[]
    train_by=defaultdict(list)
    for r in train:train_by[r['group']].append(r['id'])
    for name in protocol['conditions']:
        torch.manual_seed(31);np.random.seed(31);rng=random.Random(31)
        if name=='uniform':head=torch.nn.Linear(192,112);head.load_state_dict(model.final_fc1.state_dict());parameters=list(head.parameters())
        else:
            class PowerOnly(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.original=torch.nn.Linear(192,112);self.original.load_state_dict(model.final_fc1.state_dict());self.original.requires_grad_(False);self.power=torch.nn.Linear(192,12)
                    with torch.no_grad():self.power.weight.copy_(self.original.weight[73:85]);self.power.bias.copy_(self.original.bias[73:85])
                def forward(self,x):
                    raw=self.original(x);return torch.cat([raw[...,:73],self.power(x),raw[...,85:]],-1)
            head=PowerOnly();parameters=list(head.power.parameters())
        opt=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01)
        parameter_count=sum(p.numel() for p in parameters);save(run/name/'config.json',dict(parameters=parameter_count,training_recordings=len(train),protocol_sha256=sha(folder/'protocol.json')))
        for step in range(1,501):
            xs=[];ys=[]
            for _ in range(8):
                ident=rng.choice(train_by[rng.choice(sorted(train_by))]);item=data[ident];z=item['z'][item['mask']];y=item['y'][item['mask']];ix=np.random.randint(0,len(z),64);xs.append(z[ix]);ys.append(y[ix])
            z=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys));opt.zero_grad(set_to_none=True);outputs=torch.split(head(z),SIZES,dim=-1);loss=[]
            for j,out in enumerate(outputs):
                if (y[:,j]!=-100).any():
                    weight=torch.ones(out.shape[-1])
                    if j==0 and name=='power_only_weight3':weight[73:85]=3.
                    loss.append(torch.nn.functional.cross_entropy(out,y[:,j],weight=weight,ignore_index=-100))
            total=torch.stack(loss).mean();total.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            if step not in protocol['monitors']:continue
            score=evaluate(head,True);checkpoint=run/name/f'step_{step}.pt';state={k:v.clone() for k,v in model.state_dict().items()}
            if name=='uniform':state.update({'final_fc1.'+k:v.detach().clone() for k,v in head.state_dict().items()})
            else:
                state['final_fc1.weight'][73:85]=head.power.weight.detach().clone();state['final_fc1.bias'][73:85]=head.power.bias.detach().clone()
                if not torch.equal(state['final_fc1.weight'][:73],model.final_fc1.weight[:73]) or not torch.equal(state['final_fc1.weight'][85:],model.final_fc1.weight[85:]):raise ValueError('Protected original chord outputs changed')
            torch.save(dict(model_type='CRNN7_v1',model=state,config=protocol,step=step,condition=name,frozen_base_digest=digest,automatic_application_approved=False),checkpoint)
            s=score['summary'];base=controls['published_ensemble'];eligible=s['root']>=base['root'] and s['triads']>=base['triads'] and s['structure_exact']>base['structure_exact'];results.append(dict(condition=name,step=step,checkpoint=str(checkpoint),sha256=sha(checkpoint),eligible=eligible,**score));print('CHORD_HEAD_VALIDATED',name,step,s,'eligible',eligible,flush=True)
        save(run/name/'completed.json',dict(steps=500,parameters=parameter_count,frozen_backbone_unchanged=frozen_digest()==digest))
    allowed=[r for r in results if r['eligible']];best=max(allowed,key=lambda r:r['summary']['structure_exact']) if allowed else None
    output=dict(controls=controls,choices=results,best=best,automatic_application_approved=False,test_used=False,protocol_sha256=sha(folder/'protocol.json'),training_recordings=len(train),validation_recordings=18)
    if best:
        state=torch.load(best['checkpoint'],map_location='cpu',weights_only=True);out=Path('models/chord_head_review.pt');torch.save(state,out);save(out.with_suffix('.json'),dict(window=256,penalty=2.,review_margin=.5,sha256=sha(out),automatic_application_approved=False));output.update(export=str(out),export_sha256=sha(out))
    save(folder/'selection.json',output);print('CHORD_HEAD_COMPLETE',None if best is None else (best['condition'],best['step'],best['summary']),flush=True)


if __name__=='__main__':main()
