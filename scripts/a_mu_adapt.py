"""Pilot direct Mu LoRA: gold-token training and paired tone augmentation."""
from pathlib import Path
import argparse,json,random,time,copy,hashlib
import numpy as np,soundfile as sf,torch
from band2tab.mu_adapter import SCHEMA,encode_window,roundtrip,inject_mu,restore_mu
from band2tab.lora import adapter_state,disabled,frozen_parameter_digest
from band2tab.augment import tone_colour
from scripts.a_candidate_data import sha,save
from scripts.a_stress_data import local
DATA=Path('data/mu_adapt_v1');REPORT=Path('reports/mu_adapt_v1');RUN=Path('runs/mu_adapt_v1')

def prepare():
    from muscriptor.tokenizer.mt3 import MT3Tokenizer
    from band2tab.separation import resample
    if (DATA/'protocol.json').exists():raise ValueError('Protocol frozen')
    rows=json.load(open('data/residual_v1/manifest.json'));tok=MT3Tokenizer('MT3_FULL_PLUS');out=[];qc=[]
    for r in rows:
      if r['phase']!='train':continue
      for part in ('guitar','bass'):
        ns=r['parts'].get(part,[])
        if not ns:continue
        path=Path(r['inputs'][part]);path=path if path.is_file() else local(Path('backups/azure'),path)
        y,sr=sf.read(path,dtype='float32',always_2d=True);y=resample(y.mean(1),sr,16000);program=26 if part=='guitar' else 33
        # Fixed training-only center crop. No inference outcomes select windows.
        start=max(.25,(r['duration']-5.)*.5);start=min(start,max(0.,r['duration']-5.));audio=y[round(start*16000):round((start+5.)*16000)]
        if len(audio)!=80000:raise ValueError('Mu training window length differs')
        tokens,refs=encode_window(tok,ns,start,5.,program);q=roundtrip(tok,tokens,refs);ident=r['id']+'_'+part
        if len(tokens)>700:qc.append(dict(id=ident,excluded_token_budget=True,tokens=len(tokens)));continue
        if not refs:qc.append(dict(id=ident,excluded_no_contained_notes=True,tokens=len(tokens)));continue
        p=DATA/'cache'/(ident+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,audio=audio,tokens=np.array(tokens,np.int64))
        out.append(dict(id=ident,group=r['group'],part=part,dataset=r['dataset'],cache=str(p),source_audio_sha256=sha(path),source_start=start,program=program,cache_sha256=sha(p),notes=len(refs),tokens=len(tokens)))
        qc.append(dict(id=ident,**q,tokens=len(tokens)))
    cal=[r for r in rows if r['phase']=='calibration'];groups=sorted({r['group'] for r in cal if r['dataset'].startswith('stem_band')},key=lambda x:hashlib.sha256(('mu-adapt-monitor-'+x).encode()).hexdigest())[:2]
    monitor=[r for r in cal if r['group'] in groups]+[r for r in cal if r['dataset']=='slakh_candidate_new'][:1]
    save(DATA/'manifest.json',out);save(DATA/'monitor.json',monitor)
    save(REPORT/'token_preflight.json',dict(records=qc,contained_total=sum(x.get('contained',0) for x in qc),max_error_seconds=max(x.get('max_error_seconds',0) for x in qc),all_contained_roundtrips_passed=True))
    save(DATA/'protocol.json',dict(created='2026-11-20',arms=['gold','gold_tone'],seed=17,steps=120,lr=.00001,anchor_kl=.5,gradient_accumulation=2,
       hypothesis='Candidate pool recovered only35/148 missed attacks on reused16; directly adapt Mu decoder to existing performance-aligned guitar/bass stems.',
       trainable='Last8 attention out_proj LoRA rank8 alpha16 dropout.05; all307M base weights fixed; supervised CE, not RL.',
       augmentation='Paired same windows, draws, labels, initializer and120 updates. Tone arm applies training-only zero-phase EQ/gain, memoryless drive only to known isolated augmented bass.',
       conditioning='Same unconditioned instrument/dataset prefix as baseline; source-known roles preserved at scoring.',teacher='100Hz absolute shift, program26/33 role representatives, tie prelude and EOS, overlap trimmed only in training; original evaluation labels unchanged.',
       monitor='4 existing GOAT clean/noisy plus1 Slakh internal-calibration condition; final120 only, all4 per dataset >= same-device fresh baseline, strict score gain required. Later reused16/14 diagnostic, not test.',
       source_protocol_sha256=sha('data/residual_v1/protocol.json'),manifest_sha256=sha(DATA/'manifest.json'),monitor_sha256=sha(DATA/'monitor.json'),mu_sha256=sha('models/muscriptor-medium/model.safetensors'),
       implementation_sha256={p:sha(p) for p in ['band2tab/mu_adapter.py','scripts/a_mu_adapt.py','band2tab/augment.py']},test_used=False,automatic_application_approved=False))
    print('MU_ADAPT_PREPARED',len(out),'contained_notes',sum(r['notes'] for r in out),flush=True)

def conditions(trans,audio):
    wav=torch.as_tensor(audio,device='cuda')[None,:];attrs=trans._build_conditions(wav,None)
    with torch.no_grad():return trans._model.condition_provider(trans._model.condition_provider.tokenize(attrs))

def train():
    from muscriptor import TranscriptionModel
    p=json.load(open(DATA/'protocol.json'))
    for f,h in p['implementation_sha256'].items():
        if sha(f)!=h:raise ValueError('Frozen implementation changed')
    torch.set_num_threads(2);rows=json.load(open(DATA/'manifest.json'));cache={}
    for r in rows:
        if sha(r['cache'])!=r['cache_sha256']:raise ValueError('Training cache changed')
        with np.load(r['cache']) as z:cache[r['id']]={k:z[k].copy() for k in z.files}
    role={p:[r for r in rows if r['part']==p] for p in ('guitar','bass')}
    for arm in ('gold','gold_tone'):
        folder=RUN/arm
        if folder.exists():raise ValueError('Do not overwrite Mu training')
        torch.manual_seed(17);np.random.seed(17);rng=random.Random(17);trans=TranscriptionModel.load_model('models/muscriptor-medium/model.safetensors',device='cuda',dtype='float32');model=trans._model
        original=list(model.parameters());digest=frozen_parameter_digest(original);adapters=inject_mu(model);model.eval();params=[q for q in model.parameters() if q.requires_grad]
        example=cache[rows[0]['id']];target=torch.tensor(example['tokens'],device='cuda')[None,:];seq=torch.cat([torch.full_like(target[:,:1],model.initial_token_id),target[:,:-1]],1);cond=conditions(trans,example['audio'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            with disabled(adapters):base=model(seq,cond,first_step=True)
            after=model(seq,cond,first_step=True)
        diff=float((base.float()-after.float()).abs().max())
        if diff>1e-5:raise ValueError('Zero LoRA changed logits')
        opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=512.);history=[];draw=hashlib.sha256();started=time.monotonic();successful=0
        save(folder/'config.json',dict(arm=arm,seed=17,parameters=sum(q.numel() for q in params),protocol_sha256=sha(DATA/'protocol.json'),zero_adapter_max_difference=diff))
        for step in range(1,121):
            opt.zero_grad(set_to_none=True);values=[]
            for accum in range(2):
                part='guitar' if rng.random()<.5 else 'bass';r=rng.choice(role[part]);c=cache[r['id']];draw.update(r['id'].encode());audio=c['audio']
                if arm=='gold_tone':audio,_=tone_colour(audio,16000,17+step*2+accum,allow_drive=r['dataset']=='stem_aug_bass')
                cond=conditions(trans,audio);labels=torch.tensor(c['tokens'],device='cuda')[None,:];sequence=torch.cat([torch.full_like(labels[:,:1],model.initial_token_id),labels[:,:-1]],1)
                model.eval()
                with torch.no_grad(),disabled(adapters),torch.autocast('cuda',dtype=torch.float16):teacher_logits=model(sequence,cond,first_step=True).float()
                for m in adapters.values():m.dropout.train()
                with torch.autocast('cuda',dtype=torch.float16):logits=model(sequence,cond,first_step=True).float()
                ce=torch.nn.functional.cross_entropy(logits.reshape(-1,model.card),labels.reshape(-1));kl=torch.nn.functional.kl_div(logits.log_softmax(-1),teacher_logits.softmax(-1),reduction='batchmean')/labels.shape[1];loss=(ce+.5*kl)/2
                if not torch.isfinite(loss):raise ValueError('Nonfinite Mu loss')
                scaler.scale(loss).backward();values.append(float(ce.detach()))
            scaler.unscale_(opt)
            gradients=[q.grad for q in params if q.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Invalid Mu LoRA gradients')
            torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update();successful+=1
            if step%20==0:history.append(dict(step=step,ce=float(np.mean(values)),seconds=time.monotonic()-started));save(folder/'history.json',history);print('MU_ADAPT_TRAIN',arm,history[-1],flush=True)
        unchanged=frozen_parameter_digest(original)==digest
        if not unchanged:raise ValueError('Frozen Mu weights changed')
        model.eval();torch.save(dict(schema=SCHEMA,adapters=adapter_state(adapters),base_sha256=p['mu_sha256'],steps=120,arm=arm),folder/'final.pt')
        save(folder/'completed.json',dict(successful_updates=successful,steps=120,frozen_base_unchanged=unchanged,draw_sha256=draw.hexdigest(),seconds=time.monotonic()-started,checkpoint_sha256=sha(folder/'final.pt')))
        del model,trans,original,adapters,opt,params;import gc;gc.collect();torch.cuda.empty_cache()
    print('MU_ADAPT_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);a=p.parse_args();globals()[a.command]()
