"""Supervised domain adaptation of frozen YourMT3+ with small attention LoRA.

Only guitar/bass token losses are supervised; unrelated instrument channels
are not incorrectly labelled as absent in a full-band mixture.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json,random,hashlib,time
import numpy as np,soundfile as sf,torch
from scipy.signal import resample as speed_resample
from band2tab.lora import SCHEMA,inject,disabled,adapter_state,frozen_parameter_digest
from band2tab.score import read_midi
from scripts.a_candidate_data import rows_for,sha,save
from scripts.a_candidate_train import metrics,aggregate,bound,KEYS
from scripts.a_yourmt3_predict import load_model,transcribe

REPORT=Path('reports/yourmt3_lora');RUN=Path('runs/yourmt3_lora')
ALLOWED=('filobass','filobass_guitarset_mix','guitarset','guitar_techs_clean','slakh_candidate_new')

def role_channels(model):
    mapping=model.task_manager.program2channel_vocab
    channels=[mapping[p]['channel'] for p in (25,33)]
    if len(set(channels))!=2:raise ValueError('Expected independent guitar/bass channels')
    return channels

def prepare_protocol(info):
    rows=rows_for(False);train=[r for r in rows if r['split']=='train' and r['dataset'] in ALLOWED]
    if REPORT.name.endswith('_expanded'):
        extra=json.loads(Path('data/yourmt3_lora_extra/manifest.json').read_text())
        forbidden={r['group'] for r in json.loads(Path('data/candidate_v2/manifest.json').read_text()) if r['split']!='train'}
        if any(r['split']!='train' or r['group'] in forbidden for r in extra):raise ValueError('Expanded bass source leakage')
        train+=extra
    val=[r for r in rows if r['split']=='val'];by=defaultdict(list)
    for r in val:by[r['dataset']].append(r)
    monitor=[]
    for ds,items in sorted(by.items()):monitor+=sorted(items,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:2]
    report={'created':'2026-11-18','source_manifest_sha256':sha('data/candidate_v2/manifest.json'),
        'base_sha256':info['checkpoint_sha256'],'train_ids':[r['id'] for r in train],
        'train_records':len(train),'train_seconds':sum(r['duration'] for r in train),
        'monitor_ids':[r['id'] for r in monitor],'full_validation_ids':[r['id'] for r in val],
        'input_sha256':{r['id']:sha(r['inputs']['mix']) for r in train+val},
        'selection':'Monitor validation dataset macro score, .4 Note + .25 instrument Frame + .25 union Frame + .1 exact. Confirm on full validation before test.',
        'target_channels':'Guitar and bass only, ignore unannotated instrument channels',
        'teacher_training_only':'Author overlap trimming; held notes use ties; source annotations and evaluation labels are unchanged',
        'test_used_for_selection':False,'automatic_application_approved':False}
    if REPORT.name.endswith('_expanded'):report['extra_manifest_sha256']=sha('data/yourmt3_lora_extra/manifest.json')
    path=REPORT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=report:raise ValueError('LoRA protocol or source changed')
    else:save(path,report)
    return train,val,monitor

def sample(row,audio,model,augment=False,audit=False):
    from utils.note_event_dataclasses import Note
    from utils.note2event import trim_overlapping_notes,note2note_event,slice_multiple_note_events_and_ties_to_bundle
    sr=model.audio_cfg['sample_rate'];size=model.audio_cfg['input_frames'];duration=size/sr
    semitone=random.choice((-2,-1,0,0,0,1,2)) if augment and row['dataset']!='slakh_candidate_new' else 0
    original_size=round(size*2**(semitone/12));factor=original_size/size;guard=round(.25*sr)
    if len(audio)<original_size+2*guard:raise ValueError('Training excerpt too short')
    start_sample=random.randint(guard,len(audio)-original_size-guard);start=start_sample/sr;end=start+original_size/sr
    x=audio[start_sample:start_sample+original_size]
    if original_size!=size:x=speed_resample(x,size).astype(np.float32)
    gain=random.uniform(.7,1.2) if augment else 1.;x=(x*gain).astype(np.float32)
    notes=[]
    for part,program in (('guitar',25),('bass',33)):
        for n in row['parts'].get(part,[]):
            if n['onset']<end and n['offset']>start:
                notes.append(Note(False,program,(n['onset']-start)/factor,(n['offset']-start)/factor,n['pitch']+semitone,1))
    # Keep the longest simultaneous duplicate; preserve later attacks according
    # to the author's training-only overlap convention. Raw labels stay intact.
    notes.sort(key=lambda n:(n.program,n.pitch,n.onset,n.offset))
    notes=trim_overlapping_notes(notes)
    events=note2note_event(notes)
    bundle=slice_multiple_note_events_and_ties_to_bundle(events,[0.],duration,tidyup=False)
    tokens=model.task_manager.tokenize_note_events_batch(bundle,start_time_to_zero=True)
    nonzero=np.nonzero(tokens)[2];length=max(2,int(nonzero.max())+1 if len(nonzero) else 2)
    tokens=tokens[:,:,:length].astype(np.int64)
    trace={'id':row['id'],'start_sample':start_sample,'original_frames':original_size,
           'pitch_shift_semitones':semitone,'time_scale':factor,'gain':gain,'tokens':length,'teacher_notes':len(notes)}
    if audit:trace['teacher_reference']=[{'program':n.program,'pitch':n.pitch,'onset':n.onset,'offset':n.offset} for n in notes]
    return x,tokens,trace

def token_roundtrip(model,tokens,reference):
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    decoded=[]
    for c in role_channels(model):
        zipped,_,_=model.task_manager.detokenize_list_batches([tokens[:,c,:]],[0.],return_events=True)
        notes,_=merge_zipped_note_events_and_ties_to_notes(zipped);decoded+=notes
    duration=model.audio_cfg['input_frames']/model.audio_cfg['sample_rate']
    # Fully contained events have known attacks/releases. Boundary ties are
    # separately counted, rather than pretending a cropped attack was observed.
    wanted=[n for n in reference if 0<=n['onset']<n['offset']<duration-.02]
    role=lambda program:'guitar' if 24<=program<=31 else 'bass' if 32<=program<=39 else 'other'
    onset_errors=[];offset_errors=[];used=set()
    for n in wanted:
        matches=[(i,m) for i,m in enumerate(decoded) if i not in used and m.pitch==n['pitch'] and role(m.program)==role(n['program']) and m.onset is not None and abs(m.onset-n['onset'])<=.011]
        if not matches:raise ValueError('Target tokenizer lost or shifted a contained attack')
        i,m=min(matches,key=lambda im:abs(im[1].onset-n['onset']));used.add(i)
        if m.offset is None or abs(m.offset-n['offset'])>.011:raise ValueError('Target tokenizer shifted a contained release')
        onset_errors.append(abs(m.onset-n['onset']));offset_errors.append(abs(m.offset-n['offset']))
    if not wanted:raise ValueError('Preflight needs a contained reference note')
    return {'contained_reference_notes':len(wanted),'matched':len(used),'max_onset_error_seconds':max(onset_errors),
        'max_offset_error_seconds':max(offset_errors),'boundary_tie_references':sum(n['onset']<0<n['offset'] for n in reference)}

def target_loss(logits,labels,channels):
    return torch.stack([torch.nn.functional.cross_entropy(logits[:,c].float().reshape(-1,logits.shape[-1]),labels[:,c].reshape(-1),ignore_index=0) for c in channels]).mean()

def anchor_loss(student,teacher,labels,channels):
    values=[]
    for c in channels:
        mask=labels[:,c]!=0
        kl=torch.nn.functional.kl_div(student[:,c].float().log_softmax(-1),teacher[:,c].float().softmax(-1),reduction='none').sum(-1)
        values.append((kl*mask).sum()/mask.sum().clamp_min(1))
    return torch.stack(values).mean()

def monitor(model,rows,folder):
    model.eval();items=[]
    for r in rows:
        out=folder/'midi'/(r['id']+'.mid');transcribe(model,r['inputs']['mix'],out)
        original=read_midi(out);parts=bound({p:original.get(p,[]) for p in ('guitar','bass')},r['duration']);score=metrics(r,parts)
        items.append((r,score))
        save(out.with_suffix('.json'),{'id':r['id'],'metrics':{k:score[k] for k in KEYS},'parts':parts})
    return aggregate(items)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','train']);p.add_argument('--name',default='decoder');p.add_argument('--scope',choices=['decoder','both'],default='decoder')
    p.add_argument('--rank',type=int,default=8);p.add_argument('--alpha',type=float,default=16.);p.add_argument('--dropout',type=float,default=.05);p.add_argument('--targets',choices=['attention','wide'],default='attention')
    p.add_argument('--lr',type=float,default=.0001);p.add_argument('--steps',type=int,default=600);p.add_argument('--seed',type=int,default=17);p.add_argument('--anchor-kl',type=float,default=0.);p.add_argument('--pitch-augment',action='store_true');p.add_argument('--solo-balanced',action='store_true');p.add_argument('--extra-bass',action='store_true');a=p.parse_args()
    global REPORT
    if a.extra_bass:REPORT=Path('reports/yourmt3_lora_expanded')
    if a.command=='train' and Path('reports/yourmt3_lora/selection.json').exists():
        raise RuntimeError('Final selection frozen; new learning requires a separate experiment and fresh holdout')
    import os
    os.environ['WANDB_MODE']='disabled';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';torch.set_num_threads(2)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    model,info=load_model();train,val,validation=prepare_protocol(info);channels=role_channels(model)
    base_parameters=list(model.parameters());base_digest=frozen_parameter_digest(base_parameters)
    audios={}
    for r in train:
        y,sr=sf.read(r['inputs']['mix'],dtype='float32',always_2d=True)
        if sr!=model.audio_cfg['sample_rate']:raise ValueError('Unexpected input sample rate')
        audios[r['id']]=y.mean(1)
    example=next(r for r in train if r['dataset']=='filobass');x,tokens,trace=sample(example,audios[example['id']],model,audit=True)
    roundtrip=token_roundtrip(model,tokens,trace.pop('teacher_reference'))
    tensor=torch.from_numpy(x)[None,None].to('cuda');labels=torch.from_numpy(tokens).to('cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):before=model(tensor,labels)['logits']
    adapters=inject(model,a.scope,a.rank,a.alpha,a.dropout,a.targets);model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):after=model(tensor,labels)['logits']
    difference=float((before.float()-after.float()).abs().max())
    if difference>1e-5:raise ValueError('Zero adapter changes base outputs '+str(difference))
    parameters=[p for p in model.parameters() if p.requires_grad];config={k:v for k,v in vars(a).items() if k!='command'}
    config.update(base_sha256=info['checkpoint_sha256'],protocol_sha256=sha(REPORT/'protocol.json'),
        trainable_parameters=sum(p.numel() for p in parameters),modules=list(adapters),target_channels=channels,
        zero_adapter_max_logit_difference=difference,token_roundtrip=roundtrip,
        monitor_steps=sorted({min(100,a.steps),min(300,a.steps),min(600,a.steps),a.steps}))
    if a.command=='smoke':
        opt=torch.optim.AdamW(parameters,lr=a.lr)
        with torch.autocast('cuda',dtype=torch.float16):value=target_loss(model(tensor,labels)['logits'],labels,channels)
        value.backward();nonzero=sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0 for p in parameters);opt.step()
        unchanged=frozen_parameter_digest(base_parameters)==base_digest
        if not unchanged or not nonzero:raise ValueError('Adapter gradient or frozen-weight invariant failed')
        save(REPORT/'smoke.json',{'config':config,'trace':trace,'logit_shape':list(after.shape),'loss':float(value.detach()),
            'finite_nonzero_adapter_gradients':nonzero,'frozen_base_parameters_unchanged':unchanged})
        print('LORA_SMOKE_OK',config['trainable_parameters'],channels,'loss',float(value.detach()),flush=True);return
    folder=RUN/a.name
    if folder.exists():raise RuntimeError('Run exists; do not overwrite')
    save(folder/'config.json',config);opt=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    family=defaultdict(lambda:defaultdict(list))
    for r in train:
        domain='bass' if r['dataset'] in ('filobass','filobass_guitarset_mix') else 'synth' if r['dataset']=='slakh_candidate_new' else 'guitar'
        if a.solo_balanced and domain=='bass':domain='bass_solo' if r['dataset']=='filobass' else 'bass_mix'
        family[domain][r['group']].append(r)
    domains,weights=(('bass_solo','bass_mix','guitar','synth'),(.3,.25,.2,.25)) if a.solo_balanced else (('bass','guitar','synth'),(.4,.3,.3))
    history=[];running=[];updates=0;best=None;started=time.monotonic()
    with (folder/'samples.jsonl').open('w') as samples:
        for step in range(1,a.steps+1):
            domain=random.choices(domains,weights)[0];group=random.choice(sorted(family[domain]));r=random.choice(family[domain][group])
            x,tokens,trace=sample(r,audios[r['id']],model,a.pitch_augment);samples.write(json.dumps({'step':step,**trace})+'\n')
            tensor=torch.from_numpy(x)[None,None].to('cuda');labels=torch.from_numpy(tokens).to('cuda')
            for m in adapters.values():m.train()
            opt.zero_grad(set_to_none=True);teacher=None
            if a.anchor_kl and r['dataset'] in ('slakh_candidate_new','guitarset'):
                with disabled(adapters),torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):teacher=model(tensor,labels)['logits']
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(tensor,labels)['logits'];ce=target_loss(logits,labels,channels)
                kl=anchor_loss(logits,teacher,labels,channels) if teacher is not None else torch.zeros((),device='cuda');value=ce+a.anchor_kl*kl
            if not torch.isfinite(value):raise ValueError('Non-finite training loss')
            scaler.scale(value).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(parameters,1.)
            scale=scaler.get_scale();scaler.step(opt);scaler.update();updates+=int(scaler.get_scale()>=scale)
            running.append([float(value.detach()),float(ce.detach()),float(kl.detach())])
            if step%25==0:
                values=np.mean(running,0);running=[];history.append({'step':step,'optimizer_updates':updates,'loss':float(values[0]),'ce':float(values[1]),'kl':float(values[2]),'seconds':time.monotonic()-started})
                save(folder/'history.json',history);print('LORA_TRAIN',a.name,history[-1],flush=True)
            if step not in config['monitor_steps']:continue
            checkpoint=folder/('step_'+str(step)+'.pt')
            state={'schema':SCHEMA,'config':config,'adapters':adapter_state(adapters),'step':step,
                'optimizer_updates':updates,'base_sha256':info['checkpoint_sha256'],'automatic_application_approved':False}
            torch.save(state,checkpoint);score=monitor(model,validation,folder/('step_'+str(step)))
            record={'run':a.name,'step':step,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'validation_monitor':score}
            if best is None or score['score']>best['validation_monitor']['score']:best=record
            save(folder/'progress.json',{'best':best,'latest':record});print('LORA_MONITOR',a.name,step,{k:round(score[k],4) for k in (*KEYS,'score')},flush=True)
    unchanged=frozen_parameter_digest(base_parameters)==base_digest
    if not unchanged:raise ValueError('Frozen base parameters changed')
    save(folder/'completed.json',{'best':best,'config':config,'frozen_base_parameters_unchanged':True,
        'base_parameter_digest':base_digest,'test_used_for_selection':False,'automatic_application_approved':False})
    print('LORA_RUN_COMPLETE',a.name,flush=True)

if __name__=='__main__':main()
