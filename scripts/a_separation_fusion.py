"""Train a frozen-base addition filter; select only on validation before test."""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib
import numpy as np
import torch
from band2tab.audio import features
from band2tab.refine import candidates_from_midi
from band2tab.refine_duration import duration_targets,instrument_metrics
from band2tab.fusion import SCHEMA,PARTS,EventFusion,candidate_pool,acoustic_features,propose

ROOT=Path('data/separation_v1');REPORT=Path('reports/separation_v1');RUN=Path('runs/separation_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2))

def cache_record(r):
    path=ROOT/'features'/(r['id']+'.npz');meta=path.with_suffix('.json')
    midi={k:REPORT/'midi'/r['id']/(k+'.mid') for k in ('mix','guitar','bass')}
    fingerprint={'audio':{k:sha(r['inputs'][k]) for k in midi},'midi':{k:sha(v) for k,v in midi.items()}}
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved['fingerprint']!=fingerprint:raise ValueError('Cached feature inputs changed')
        data=np.load(path);return saved['pools'],data['x']
    original=candidates_from_midi(midi['mix']);separated={p:candidates_from_midi(midi[p]) for p in PARTS}
    pools=candidate_pool(original,separated);x=acoustic_features({k:features(r['inputs'][k]) for k in midi},pools)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,x=x);save(meta,{'fingerprint':fingerprint,'pools':pools})
    return pools,x

def metrics(r,parts):
    guard=r['guard_seconds']
    return instrument_metrics(r['parts'],parts,start=guard,end=r['duration']-guard if guard else None)

def polyphony(r,parts):
    from band2tab.learn import pitch_set_metrics
    guard=r['guard_seconds'];end=r['duration']-guard if guard else float('inf')
    def cut(notes):
        return [{**n,'onset':max(guard,n['onset'])-guard,'offset':min(end,n['offset'])-guard}
                for n in notes if n['offset']>guard and n['onset']<end]
    result={}
    for name in ('union',*PARTS):
        targets=PARTS if name=='union' else (name,)
        truth=cut([n for p in targets for n in r['parts'].get(p,[])])
        estimated=cut([n for p in targets for n in parts.get(p,[])])
        result[name]=pitch_set_metrics(truth,estimated)['by_reference_polyphony']
    return result

def aggregate(rows):
    groups=defaultdict(list)
    for r,m in rows:groups[r.get('source_dataset','p3')].append(m)
    keys=('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')
    by={g:{k:float(np.mean([m[k] for m in ms])) for k in keys} for g,ms in groups.items()}
    return {**{k:float(np.mean([v[k] for v in by.values()])) for k in keys},'datasets':by}

def eligible(score,baseline):
    # Protect actual pitch recognition as well as labels, separately by dataset.
    return all(score['datasets'][d][k]>=v[k]-1e-8 for d,v in baseline['datasets'].items() for k in ('note_f1','frame_f1','union_frame_f1'))

def train():
    if (REPORT/'selection.json').exists():raise RuntimeError('Selection frozen; never reselect on test')
    torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17)
    rows=[r for r in json.loads((ROOT/'manifest.json').read_text()) if r['split']!='test'];train=[];val=[]
    for r in rows:
        pools,x=cache_record(r);notes=[n for p in PARTS for n in pools[p]]
        keep=np.array([n['keep'] for n in notes],np.float32)
        if r['split']=='train':
            y=np.concatenate([duration_targets(pools[p],{p:r['parts'].get(p,[])})[:,i] for i,p in enumerate(PARTS,1)])
            w=np.array([min(3.,n['offset']-n['onset']) for n in notes],np.float32);w=w/max(1e-9,w.sum())
            train.append((r,pools,x,keep,y,w))
        else:val.append((r,pools,x,keep))
        print('FEATURED',r['split'],r['id'],len(notes),flush=True)
    x=np.concatenate([t[2] for t in train]);mean=x.mean(0);std=x.std(0).clip(.05)
    norm=lambda x:torch.tensor(np.clip((x-mean)/std,-8,8),dtype=torch.float32)
    tx=norm(x);tk=torch.tensor(np.concatenate([t[3] for t in train]));ty=torch.tensor(np.concatenate([t[4] for t in train]))
    counts={kind:sum(t[0]['kind']==kind for t in train) for kind in ('solo','mixture')}
    tw=torch.tensor(np.concatenate([t[5]/counts[t[0]['kind']] for t in train]));tw=tw/tw.sum()
    model=EventFusion();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    baseline=aggregate([(r,metrics(r,propose(pools,np.zeros(len(k)))[0])) for r,pools,x,k in val])
    selected={'method':'M0_original','validation':baseline,'policy':None,'checkpoint':None}
    comparisons=[];curve=[];RUN.mkdir(parents=True,exist_ok=True)
    # Separation routes are also candidates, selected without test access.
    development=json.loads((REPORT/'comparison_development.json').read_text())['tracks']
    mapping={r['id']:r for r in rows if r['split']=='val'}
    for method in ('M1_no_drums_vocals','M2_separated_labels','M2_separated_forced','M3_naive_union'):
        score=aggregate([(mapping[t['id']],t['metrics'][method]) for t in development if t['split']=='val'])
        comparisons.append({'method':method,'validation':score,'eligible':eligible(score,baseline)})
        if eligible(score,baseline) and score['union_frame_f1']+score['note_f1']>selected['validation']['union_frame_f1']+selected['validation']['note_f1']:
            selected={'method':method,'validation':score,'policy':None,'checkpoint':None}
    for epoch in range(1,151):
        model.train();opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(tx,tk),ty,reduction='none')*tw).sum();loss.backward();opt.step()
        curve.append({'epoch':epoch,'loss':float(loss.detach())})
        if epoch not in (25,50,100,150):continue
        model.eval();prob=[]
        with torch.no_grad():
            for r,pools,x,k in val:prob.append(model(norm(x),torch.tensor(k)).sigmoid().numpy())
        checkpoint=RUN/('epoch_'+str(epoch)+'.pt')
        torch.save({'schema':SCHEMA,'model':model.state_dict(),'mean':mean.tolist(),'std':std.tolist(),
                    'epoch':epoch,'seed':17,'automatic_application_approved':False,
                    'protocol_sha256':sha(REPORT/'protocol.json')},checkpoint)
        for add in (.75,.9,.97):
            for remove in (-1.,.1,.25):
                policy={'add_threshold':add,'remove_threshold':remove}
                score=aggregate([(r,metrics(r,propose(pools,p,**policy)[0])) for (r,pools,x,k),p in zip(val,prob)])
                ok=eligible(score,baseline)
                comparisons.append({'method':'learned_fusion','epoch':epoch,'policy':policy,'eligible':ok,'validation':score})
                if ok and score['union_frame_f1']+score['note_f1']>selected['validation']['union_frame_f1']+selected['validation']['note_f1']+1e-8:
                    selected={'method':'learned_fusion','epoch':epoch,'policy':policy,'validation':score,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint)}
        print('TRAINED',epoch,curve[-1],'selected',selected['method'],selected.get('epoch'),flush=True)
    # Fixed diagnostic learned variant even if no candidate meets adoption rules.
    fixed={'method':'learned_fusion','epoch':150,'policy':{'add_threshold':.9,'remove_threshold':-1.},
           'checkpoint':str(RUN/'epoch_150.pt'),'checkpoint_sha256':sha(RUN/'epoch_150.pt')}
    save(REPORT/'training.json',{'train_records':len(train),'train_candidates':len(tx),
        'parameters':sum(p.numel() for p in model.parameters()),'curve':curve,'comparisons':comparisons,
        'train_ids':[t[0]['id'] for t in train],'val_ids':[t[0]['id'] for t in val]})
    package_policy=selected if selected['method']=='learned_fusion' else fixed
    packaged=torch.load(package_policy['checkpoint'],weights_only=True,map_location='cpu')
    packaged.update(policy=package_policy['policy'],automatic_application_approved=False,
                    validation_adopted=selected['method']=='learned_fusion')
    packaged_path=Path('models/muscriptor_separation_fusion_review.pt');packaged_path.parent.mkdir(exist_ok=True)
    torch.save(packaged,packaged_path)
    save(packaged_path.with_suffix('.json'),{'schema':SCHEMA,'sha256':sha(packaged_path),
        'source_checkpoint':package_policy['checkpoint'],'source_checkpoint_sha256':package_policy['checkpoint_sha256'],
        'policy':package_policy['policy'],'parameters':sum(p.numel() for p in model.parameters()),
        'validation_adopted':selected['method']=='learned_fusion','automatic_application_approved':False,
        'scope':'Review proposals only; fixed diagnostic policy if validation did not select learned fusion'})
    save(REPORT/'selection.json',{'protocol_sha256':sha(REPORT/'protocol.json'),'selected':selected,
         'fixed_diagnostic':fixed,'baseline_validation':baseline,'automatic_application_approved':False,
         'packaged_checkpoint':str(packaged_path),'packaged_checkpoint_sha256':sha(packaged_path),
         'test_used_for_selection':False,'selection':'equal dataset macro; maximize union Frame + Note; no per-dataset Note/Frame/union loss vs original'})
    print('SELECTION_FROZEN',json.dumps(selected),flush=True)

def evaluate():
    from scripts.a_separation_compare import methods,get_rows
    selection=json.loads((REPORT/'selection.json').read_text());out=REPORT/'fusion_test.json'
    if out.exists():raise RuntimeError('Already scored, do not reselect')
    if selection['protocol_sha256']!=sha(REPORT/'protocol.json'):raise ValueError('Protocol changed')
    policies={'fixed_learned_additions':selection['fixed_diagnostic']}
    if selection['selected']['method']=='learned_fusion':policies['selected_learned']=selection['selected']
    models={}
    for name,s in policies.items():
        if sha(s['checkpoint'])!=s['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        state=torch.load(s['checkpoint'],weights_only=True,map_location='cpu');model=EventFusion();model.load_state_dict(state['model']);model.eval();models[name]=(model,state,s)
    results=[]
    for r in get_rows(True):
        pools,x=cache_record(r);keep=torch.tensor([float(n['keep']) for p in PARTS for n in pools[p]])
        predicted=methods(r);changes={}
        for name,(model,state,s) in models.items():
            z=torch.tensor(np.clip((x-np.array(state['mean']))/np.array(state['std']),-8,8),dtype=torch.float32)
            with torch.no_grad():p=model(z,keep).sigmoid().numpy()
            parts,change=propose(pools,p,**s['policy']);predicted[name]=parts;changes[name]=change
        selected=selection['selected']['method'];predicted['selected']=predicted['selected_learned' if selected=='learned_fusion' else selected]
        scores={name:{**metrics(r,parts),'polyphony':polyphony(r,parts)} for name,parts in predicted.items()}
        save(REPORT/'predictions'/(r['id']+'.json'),{'parts':predicted,'changes':changes})
        raw={name:instrument_metrics(r['unshifted_parts'],parts) for name,parts in predicted.items()} if 'unshifted_parts' in r else {}
        results.append({'id':r['id'],'group':r['group'],'evaluation':r['evaluation'],'dataset':r.get('source_dataset','p3'),'metrics':scores,'raw_annotation_metrics':raw})
        print('TESTED',r['id'],{k:round(v['union_frame_f1'],4) for k,v in scores.items()},flush=True)
    summary={}
    for ds in sorted({r['dataset'] for r in results}):
        use=[r for r in results if r['dataset']==ds];names=set.intersection(*(set(r['metrics']) for r in use))
        summary[ds]={'n':len(use),'methods':{name:{k:float(np.mean([r['metrics'][name][k] for r in use])) for k in ('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')} for name in sorted(names)}}
    save(out,{'selection':selection,'summary':summary,'tracks':results,'scope':'previously inspected regression; no natural vocal-band generalization claim'})
    print('FUSION_TEST_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate']);a=p.parse_args()
    globals()[a.command]()
