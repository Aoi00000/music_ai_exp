"""Temporal residual learning and validation-only checkpoint/decoder selection."""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib,random,time,copy
import numpy as np
import torch
from band2tab.temporal import SCHEMA,TemporalFusion,predict,decode_parts,masked_loss,known_offset_score
from band2tab.refine_duration import instrument_metrics
from band2tab.refine import candidates_from_midi
ROOT=Path('data/temporal_v1');REPORT=Path('reports/temporal_v1');RUN=Path('runs/temporal_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2));tmp.replace(p)
def original(r):
    notes=candidates_from_midi(REPORT/'midi'/r['id']/'mix.mid')
    return {p:[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes if n['baseline_class']==c] for c,p in ((1,'guitar'),(2,'bass'))}
def clipped(parts,duration):
    return {p:[{**n,'offset':min(duration,n['offset'])} for n in ns if 0<=n['onset']<duration] for p,ns in parts.items()}
def metrics(r,parts):
    g=r['guard_seconds'];end=r['duration']-g if g else None
    result=instrument_metrics(r['parts'],parts,start=g,end=end)
    result['offset_f1_including_excerpt_cuts']=result['offset_f1']
    return {**result,**known_offset_score(r['parts'],parts,g,end if end is not None else float('inf'))}
def aggregate(items):
    grouped=defaultdict(list)
    for r,m in items:grouped[r['dataset']].append(m)
    keys=('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')
    by={d:{k:float(np.mean([v[k] for v in values])) for k in keys} for d,values in grouped.items()}
    mean={k:float(np.mean([v[k] for v in by.values()])) for k in keys}
    return {**mean,'datasets':by,'selection_score':float(np.mean([mean[k] for k in ('note_f1','frame_f1','union_frame_f1')]))}
def eligible(score,baseline):
    return all(score['datasets'][d][k]>=v[k]-1e-8 for d,v in baseline['datasets'].items() for k in ('note_f1','frame_f1','union_frame_f1'))
def load_cache(r):
    with np.load(ROOT/'cache'/(r['id']+'.npz')) as f:return {k:f[k].copy() for k in ('e','m','y')}

def train():
    if (REPORT/'selection.json').exists():raise RuntimeError('Validation selection already frozen')
    protocol=json.loads((REPORT/'protocol.json').read_text())
    if sha(ROOT/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    torch.set_num_threads(4);rows=json.loads((ROOT/'manifest.json').read_text());val=[r for r in rows if r['split']=='val']
    caches={r['id']:load_cache(r) for r in rows if r['split']!='test'}
    baseline=aggregate([(r,metrics(r,original(r))) for r in val])
    controls={'original':baseline,'audio_bounded':aggregate([(r,metrics(r,clipped(original(r),r['duration']))) for r in val])}
    encoder=torch.load('models/acoustic_review.pt',weights_only=True,map_location='cpu')['model']
    if sha('models/acoustic_review.pt')!=protocol['encoder_sha256']:raise ValueError('Encoder changed')
    policies=[{'onset':a,'frame':b,'offset':off,'min_duration':.03} for a in (.3,.5,.7) for b in (.3,.5,.7) for off in (.5,1.1)]
    results={};histories={};comparisons=[]
    for condition in ('standard','hard'):
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        tr=[r for r in rows if r['split']=='train' and (condition=='hard' or not r['hard_training'])]
        groups=defaultdict(list)
        for r in tr:groups[r['group']].append(r)
        group_names=sorted(groups);model=TemporalFusion(encoder,include_ft=True).to('cuda')
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);history=[];best=None
        started=time.monotonic();running=[]
        for step in range(1,3001):
            batch=[]
            for _ in range(8):
                r=random.choice(groups[random.choice(group_names)]);data=caches[r['id']];part=random.randrange(2)
                length=len(data['e']);start=random.randrange(max(1,length-344+1));stop=min(length,start+344)
                e=data['e'][start:stop].astype(np.float32);m=data['m'][start:stop,part].astype(np.float32);y=data['y'][start:stop,part].astype(np.float32)
                mask=np.ones(stop-start,np.float32)
                if len(e)<344:
                    pad=344-len(e);e=np.pad(e,((0,pad),(0,0),(0,0)));m=np.pad(m,((0,pad),(0,0),(0,0)));y=np.pad(y,((0,pad),(0,0),(0,0)));mask=np.pad(mask,(0,pad))
                batch.append((e,m,y,mask,part))
            e,m,y,mask,part=[torch.as_tensor(np.stack([v[i] for v in batch]),device='cuda') for i in range(5)]
            model.train();opt.zero_grad(set_to_none=True)
            # FP32 GRU keeps training/inference portable to CPU and Apple Silicon.
            loss=masked_loss(model(e,m,part.long()),y,mask);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            running.append(float(loss.detach()))
            if step%100==0:
                entry={'step':step,'loss':float(np.mean(running)),'seconds':time.monotonic()-started};running=[];history.append(entry);print('TRAIN',condition,entry,flush=True)
                save(RUN/condition/'history.json',history)
            if step not in protocol['updates']:continue
            model.eval();checkpoint=RUN/condition/('step_'+str(step)+'.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True)
            state={'schema':SCHEMA,'model':{k:v.cpu() for k,v in model.state_dict().items()},'encoder':encoder,
                   'include_ft':True,'step':step,'condition':condition,'seed':17,'protocol_sha256':sha(REPORT/'protocol.json'),
                   'automatic_application_approved':False,'train_ids':[r['id'] for r in tr],
                   'loss':{'positive_weights':[20.,3.,20.],'head_weights':[2.,1.,1.]},
                   'trainable_parameters':sum(p.numel() for p in model.parameters())}
            torch.save(state,checkpoint)
            probabilities={}
            for r in val:
                d=caches[r['id']];probabilities[r['id']]=predict(model,d['e'],d['m'],'cuda')
            for policy in policies:
                score=aggregate([(r,metrics(r,decode_parts(probabilities[r['id']],policy))) for r in val])
                record={'condition':condition,'step':step,'policy':policy,'validation':score,'eligible':eligible(score,baseline)}
                comparisons.append(record)
                if best is None or score['selection_score']>best['validation']['selection_score']:
                    best={**record,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint)}
            save(REPORT/('progress_'+condition+'.json'),{'best':best,'comparisons':comparisons})
            print('VALIDATED',condition,step,'best',best['step'],best['policy'],best['validation']['selection_score'],'eligible',best['eligible'],flush=True)
        results[condition]=best;histories[condition]=history
    research=max(results,key=lambda name:results[name]['validation']['selection_score'])
    selected=results[research]
    deployed=research if selected['eligible'] and selected['validation']['selection_score']>baseline['selection_score'] else 'identity'
    package=torch.load(selected['checkpoint'],weights_only=True,map_location='cpu');package['policy']=selected['policy'];package['validation_adopted']=deployed!='identity'
    path=Path('models/muscriptor_temporal_review.pt');torch.save(package,path)
    save(path.with_suffix('.json'),{'schema':SCHEMA,'sha256':sha(path),'source_checkpoint':selected['checkpoint'],
        'source_sha256':selected['checkpoint_sha256'],'policy':selected['policy'],'validation_adopted':deployed!='identity',
        'automatic_application_approved':False,'trainable_parameters':package['trainable_parameters']})
    save(REPORT/'selection.json',{'protocol_sha256':sha(REPORT/'protocol.json'),'results':results,'research_selected':research,
        'adoption_candidate':deployed,'controls':controls,'comparisons':comparisons,'test_used_for_selection':False,
        'automatic_application_approved':False,'packaged_checkpoint_sha256':sha(path)})
    print('TEMPORAL_SELECTION_FROZEN',research,'adoption_candidate',deployed,flush=True)

if __name__=='__main__':train()
