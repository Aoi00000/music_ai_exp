"""Repeated supervised development rounds, followed by one frozen selection.

Only train/validation rows are accessible here. Tests are handled separately.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json,random,time,itertools
import numpy as np,torch
from band2tab.candidate import SCHEMA,CandidateRanker,loss,predict,propose,unique_targets
from band2tab.score import read_midi
from band2tab.audio import PITCH_MIN,PITCH_MAX
from band2tab.temporal import known_offset_score
from band2tab.refine_duration import instrument_metrics
from scripts.a_candidate_data import ROOT,REPORT,sha,save,rows_for

KEYS=('note_f1','frame_f1','union_frame_f1','active_exact_match','offset_f1')
def metrics(r,parts):
    g=r['guard_seconds'];end=r['duration']-g if g else None
    return {**instrument_metrics(r['parts'],parts,start=g,end=end),
            **known_offset_score(r['parts'],parts,g,end if end is not None else float('inf'))}
def aggregate(items):
    grouped=defaultdict(list)
    for r,s in items:grouped[r['dataset']].append(s)
    by={d:{k:float(np.mean([s[k] for s in values])) for k in KEYS} for d,values in grouped.items()}
    mean={k:float(np.mean([s[k] for s in by.values()])) for k in KEYS}
    mean.update(datasets=by,score=.4*mean['note_f1']+.25*mean['frame_f1']+.25*mean['union_frame_f1']+.1*mean['active_exact_match'])
    return mean
def load(r):
    path=ROOT/'cache'/(r['id']+'.npz')
    with np.load(path) as f:data={k:f[k].copy() for k in f.files}
    data['pool']=json.loads(path.with_suffix('.json').read_text())['pool'];return data
def bound(parts,duration):
    return {p:[{**n,'offset':min(duration,n['offset'])} for n in ns if 0<=n['onset']<duration] for p,ns in parts.items()}
def baselines(r):
    folder=REPORT/'midi'/r['id'];raw=read_midi(folder/'mix.mid')
    return {'Mu':bound({p:raw.get(p,[]) for p in ('guitar','bass')},r['duration']),
            'six_Mu':bound({p:read_midi(folder/(p+'.mid'),force_part=p).get(p,[]) for p in ('guitar','bass')},r['duration'])}
def restore(parts,baseline):
    return {p:list(parts.get(p,[]))+[n for n in baseline.get(p,[]) if not PITCH_MIN<=n['pitch']<=PITCH_MAX or n['offset']-n['onset']<.03] for p in ('guitar','bass')}
def candidate_parts(r,data,pred,policy,baseline):return restore(propose(data['pool'],pred,r['duration'],**policy),baseline)

def train(a):
    if (REPORT/'selection.json').exists():raise RuntimeError('Selection frozen; start a separate experiment instead')
    if sha('models/acoustic_review.pt')!=json.loads((REPORT/'protocol.json').read_text())['encoder_sha256']:raise ValueError('Encoder changed')
    folder=Path('runs/candidate_v2')/a.name
    if folder.exists():raise RuntimeError('Run name exists; do not silently overwrite a previous trial')
    rows=rows_for(False);val=[r for r in rows if r['split']=='val']
    train=[r for r in rows if r['split']=='train' and
           (a.data=='hard' or not r['hard_training']) and (a.data!='small' or not r['fresh_source']) and
           (a.data in ('balanced','hard') or r['dataset']!='guitar_techs_clean')]
    cache={r['id']:load(r) for r in train+val}
    if a.unique_targets:
        for r in train:cache[r['id']].update(unique_targets(cache[r['id']]['pool'],r['parts']))
    valid_train=[r for r in train if len(cache[r['id']]['x'])]
    arrays=[cache[r['id']]['x'].astype(np.float32) for r in valid_train];x=np.concatenate(arrays)
    mean=x.mean(0);std=x.std(0).clip(.05);del x,arrays
    for data in cache.values():data['normalized']=np.clip((data['x'].astype(np.float32)-mean)/std,-8,8)
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    by_dataset=defaultdict(lambda:defaultdict(list))
    for r in valid_train:by_dataset[r['dataset']][r['group']].append(r)
    datasets=sorted(by_dataset);groups={k:sorted(v) for k,v in by_dataset.items()}
    model=CandidateRanker(context=a.context).to('cuda');opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.001)
    encoder=torch.load('models/acoustic_review.pt',map_location='cpu',weights_only=True)['model']
    config={k:v for k,v in vars(a).items() if k!='command'};config.update(train_records=len(train),train_seconds=sum(r['duration'] for r in train),
        trainable_parameters=sum(p.numel() for p in model.parameters()),protocol_sha256=sha(REPORT/'protocol.json'),
        train_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in val],
        validation_policy_grid={'threshold':[.35,.5,.65,.8],'retime':[False,True],'quality_weight':[0.],'nms':[.03]})
    save(folder/'config.json',config);baselines_val={r['id']:baselines(r) for r in val}
    controls={m:aggregate([(r,metrics(r,baselines_val[r['id']][m])) for r in val]) for m in ('Mu','six_Mu')}
    history=[];comparisons=[];best=None;running=[];started=time.monotonic()
    for step in range(1,a.steps+1):
        batch=[]
        for _ in range(8):
            ds=random.choice(datasets);group=random.choice(groups[ds]);r=random.choice(by_dataset[ds][group]);d=cache[r['id']]
            n=len(d['x']);start=random.randrange(max(1,n-192+1));end=min(n,start+192);pad=192-(end-start)
            item={}
            for k in ('prior','y','quality','delta','timing_mask','mask','keep_correct'):
                v=d[k][start:end];item[k]=np.pad(v,((0,pad),)+((0,0),)*(v.ndim-1))
            item['x']=np.pad(d['normalized'][start:end],((0,pad),(0,0)))
            item['valid']=np.arange(192)<end-start;batch.append(item)
        data={k:torch.as_tensor(np.stack([b[k] for b in batch]),device='cuda') for k in batch[0]}
        model.train();opt.zero_grad(set_to_none=True);output=model(data['x'],data['prior'],data['valid'])
        value=loss(output,data,negative_weight=a.negative_weight);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();running.append(float(value.detach()))
        if step%100==0:
            history.append({'step':step,'loss':float(np.mean(running)),'seconds':time.monotonic()-started});running=[]
            save(folder/'history.json',history);print('CANDIDATE_TRAIN',a.name,history[-1],flush=True)
        if step not in sorted({min(500,a.steps),min(1500,a.steps),a.steps}):continue
        model.eval();path=folder/('step_'+str(step)+'.pt')
        state={'schema':SCHEMA,'model':{k:v.cpu() for k,v in model.state_dict().items()},'context':a.context,
            'mean':mean.tolist(),'std':std.tolist(),'encoder':encoder,'config':config,'step':step,
            'automatic_application_approved':False};torch.save(state,path)
        probabilities={r['id']:predict(model,cache[r['id']]['x'],cache[r['id']]['prior'],mean,std,'cuda') for r in val}
        for threshold,retime in itertools.product((.35,.5,.65,.8),(False,True)):
            policy={'threshold':threshold,'retime':retime,'quality_weight':0.,'nms':.03}
            score=aggregate([(r,metrics(r,candidate_parts(r,cache[r['id']],probabilities[r['id']],policy,baselines_val[r['id']]['Mu']))) for r in val])
            record={'run':a.name,'step':step,'policy':policy,'validation':score,'checkpoint':str(path),'checkpoint_sha256':sha(path)}
            comparisons.append(record)
            if best is None or score['score']>best['validation']['score']:best=record
        save(folder/'progress.json',{'best':best,'comparisons':comparisons,'controls':controls})
        print('CANDIDATE_VALIDATION',a.name,step,'best',best['step'],best['policy'],{k:round(best['validation'][k],4) for k in (*KEYS,'score')},flush=True)
    save(folder/'completed.json',{'best':best,'controls':controls,'config':config,'comparisons':comparisons,'test_used_for_selection':False})
    print('CANDIDATE_RUN_COMPLETE',a.name,flush=True)

def freeze():
    path=REPORT/'selection.json'
    if path.exists():raise RuntimeError('Selection frozen')
    trials=[json.loads(p.read_text()) for p in sorted(Path('runs/candidate_v2').glob('*/completed.json'))]
    if not trials:raise ValueError('No completed learning runs')
    best=max((r['best'] for r in trials),key=lambda r:r['validation']['score']);controls=trials[0]['controls']
    for t in trials:
        if t['controls']!=controls:raise ValueError('Validation controls changed between rounds')
    chosen=max([{'method':m,'validation':v} for m,v in controls.items()]+[{'method':'candidate','validation':best['validation']}],key=lambda r:r['validation']['score'])
    package=torch.load(best['checkpoint'],map_location='cpu',weights_only=True);package['policy']=best['policy']
    destination=Path('models/muscriptor_candidate_v2_review.pt');torch.save(package,destination)
    save(destination.with_suffix('.json'),{'schema':SCHEMA,'sha256':sha(destination),'run':best['run'],'policy':best['policy'],
        'parameters':package['config']['trainable_parameters'],'automatic_application_approved':False})
    gates=json.loads((REPORT/'protocol.json').read_text())['practical_targets']
    practical=all(v[k]>=limit for v in best['validation']['datasets'].values() for k,limit in gates.items())
    save(path,{'protocol_sha256':sha(REPORT/'protocol.json'),'research_selected':best,'validation_winner':chosen['method'],
        'controls':controls,'runs':[{'run':t['best']['run'],'best':t['best']} for t in trials],
        'provisional_practical_validation_passed':practical,'test_used_for_selection':False,
        'automatic_application_approved':False,'package_sha256':sha(destination)})
    print('CANDIDATE_SELECTION_FROZEN',best['run'],best['policy'],'winner',chosen['method'],'practical',practical,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','freeze']);p.add_argument('--name');p.add_argument('--data',choices=['small','expanded','balanced','hard'],default='expanded');p.add_argument('--context',action='store_true')
    p.add_argument('--negative-weight',type=float,default=1.);p.add_argument('--steps',type=int,default=3000);p.add_argument('--seed',type=int,default=17);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--unique-targets',action='store_true')
    a=p.parse_args()
    if a.command=='freeze':freeze()
    else:
        if not a.name:p.error('--name is required')
        train(a)
