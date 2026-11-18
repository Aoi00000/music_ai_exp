"""New-source splits, preserved-rate separation and candidate feature caches."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,random,os,time,shutil
import numpy as np,soundfile as sf
from band2tab.separation import write_input,separate_inputs
from scripts.a_temporal_data import clipped_truth

ROOT=Path('data/candidate_v2');REPORT=Path('reports/candidate_v2')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2));tmp.replace(p)

def prepare():
    from band2tab.data import slakh_items
    if (ROOT/'manifest.json').exists():raise RuntimeError('Manifest frozen')
    acquisition=json.loads((REPORT/'source_acquisition.json').read_text());rows=[]
    for old in json.loads(Path('data/temporal_v1/manifest.json').read_text()):
        r=dict(old);r['reuse_cache']=str((Path('data/temporal_v1/cache')/(r['id']+'.npz')).resolve())
        r['reuse_midi']={k:str((Path('reports/temporal_v1/midi')/r['id']/(k+'.mid')).resolve()) for k in ('mix','guitar','bass')}
        if r['split']=='test':r['split']='regression';r['dataset']='previous_'+r['dataset']
        r['fresh_source']=False;rows.append(r)
    groups=defaultdict(list)
    for r in slakh_items('data/raw/slakh_candidate_v2','candidate'):
        groups[r['group']].append(r)
    for source in acquisition['selected']:
        path=Path(source['folder'])/'mix.flac';info=sf.info(path);split=source['split']
        duration=min(info.duration,24. if split=='train' else 30. if split=='val' else 40.)
        parts={p:[n for r in groups[source['group']] if r['part']==p for n in r['notes']] for p in ('guitar','bass')}
        if not all(parts.values()):raise ValueError('Missing source annotations '+source['track'])
        fractions=(.15,.45,.75) if split=='train' else (.35,)
        for j,fraction in enumerate(fractions):
            start=(info.duration-duration)*fraction;ident='c2_'+source['track']+'_'+str(j);dest=ROOT/'audio'/ident;dest.mkdir(parents=True,exist_ok=True)
            with sf.SoundFile(path) as f:
                f.seek(round(start*info.samplerate));y=f.read(round(duration*info.samplerate),dtype='float32',always_2d=True)
            # Keep available native bandwidth until the separator. AMT is 16k.
            audio=dest/'native_mix.wav';sf.write(audio,y,info.samplerate,subtype='FLOAT');write_input(dest/'mix.wav',y,info.samplerate)
            actual=sf.info(dest/'mix.wav').duration
            rows.append({'id':ident,'group':source['group'],'split':split,'kind':'mixture',
                'dataset':'slakh_candidate_new','fresh_source':True,'hard_training':False,
                'audio':str(audio.resolve()),'excerpt_start':start,'duration':actual,'guard_seconds':.25,
                'parts':clipped_truth(parts,start,actual),
                'inputs':{k:str((dest/(k+'.wav')).resolve()) for k in ('mix','guitar','bass')}})
    mapping=defaultdict(set)
    for r in rows:
        mapping[r['group']].add(r['split']);r['audio_sha256']=sha(r['audio'])
        for k in ('backing_group','guitar_group'):
            if r.get(k):mapping[r[k]].add(r['split'])
    if any(len(s)>1 for s in mapping.values()):raise ValueError('Group leakage')
    save(ROOT/'manifest.json',rows)
    protocol={'schema':'candidate_v2','created':'2026-11-18','manifest_sha256':sha(ROOT/'manifest.json'),
        'source_acquisition_sha256':sha(REPORT/'source_acquisition.json'),
        'encoder_sha256':sha('models/acoustic_review.pt'),'mu_sha256':sha('models/muscriptor-medium/model.safetensors'),
        'counts':dict(Counter(r['split'] for r in rows)),
        'selection':'Development validation only, mean across datasets of .4 Note + .25 instrument Frame + .25 union Frame + .1 active exact',
        'test_access':'New 12 source UUIDs withheld until selection is frozen; previous tests are regression only',
        'practical_targets':{'note_f1':.85,'frame_f1':.85,'active_exact_match':.7,'offset_f1':.7},
        'target_note':'Provisional engineering gates, not a standard or proof of usability; actual Mac editing time and natural-band validation also required',
        'automatic_application_approved':False,'test_used_for_selection':False}
    save(REPORT/'protocol.json',protocol);print('CANDIDATE_MANIFEST_FROZEN',protocol['counts'],flush=True)

def rows_for(test=False):
    protocol=json.loads((REPORT/'protocol.json').read_text())
    if sha(ROOT/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    if test and not (REPORT/'selection.json').exists():raise RuntimeError('Freeze selection before test access')
    lock=Path('reports/global_development_lock.json')
    if test and lock.exists() and not json.loads(lock.read_text()).get('test_access_enabled',False):raise RuntimeError('Other model families are still being developed; final test remains locked')
    rows=json.loads((ROOT/'manifest.json').read_text())
    return [r for r in rows if (r['split'] in ('test','regression'))==test]

def add_clean():
    """Add already available real clean-guitar training, never a test performer."""
    if list(Path('runs/candidate_v2').glob('*/config.json')):raise RuntimeError('Add data before training')
    rows=json.loads((ROOT/'manifest.json').read_text())
    if any(r['dataset']=='guitar_techs_clean' for r in rows):raise RuntimeError('Clean training already added')
    original=json.loads(Path('data/manifest_duration.json').read_text())
    extra=[r for r in original if r['split']=='train' and r['id'].startswith(('duration_P1_','duration_P2_'))]
    for old in extra:
        r=dict(old);ident='clean_'+old['id'];dest=ROOT/'audio'/ident
        if sha(r['audio'])!=r['audio_sha256']:raise ValueError('Original clean source changed')
        y,sr=sf.read(r['audio'],dtype='float32',always_2d=True);write_input(dest/'mix.wav',y,sr)
        rows.append({**r,'id':ident,'dataset':'guitar_techs_clean','fresh_source':False,'hard_training':False,
            'guard_seconds':.25,'inputs':{k:str((dest/(k+'.wav')).resolve()) for k in ('mix','guitar','bass')}})
    mapping=defaultdict(set)
    for r in rows:mapping[r['group']].add(r['split'])
    if any(len(v)>1 for v in mapping.values()):raise ValueError('Clean source group leakage')
    shutil.copy2(ROOT/'manifest.json',ROOT/'manifest_before_clean.json');shutil.copy2(REPORT/'protocol.json',REPORT/'protocol_before_clean.json')
    save(ROOT/'manifest.json',rows);protocol=json.loads((REPORT/'protocol.json').read_text())
    protocol.update(manifest_sha256=sha(ROOT/'manifest.json'),clean_training_records=len(extra),counts=dict(Counter(r['split'] for r in rows)))
    save(REPORT/'protocol.json',protocol);print('CLEAN_TRAINING_ADDED',len(extra),protocol['counts'],flush=True)

def separate(test=False):
    import torch
    from demucs.pretrained import get_model
    os.environ['TORCH_HOME']=str(Path('models/torch').resolve());torch.set_num_threads(4);model=None
    for r in rows_for(test):
        if all(Path(r['inputs'][p]).exists() for p in ('guitar','bass')):continue
        if model is None:model=get_model('htdemucs_6s').to('cuda').eval()
        seed=int(sha(r['audio'])[:8],16)%2**31;torch.manual_seed(seed);random.seed(seed)
        before=sha(r['audio']);meta=separate_inputs(r['audio'],ROOT/'audio'/r['id'],model)
        if before!=sha(r['audio']):raise ValueError('Source changed during separation')
        save(ROOT/'audio'/r['id']/'separation.json',meta);print('CANDIDATE_SEPARATED',r['id'],flush=True)

def transcribe(test=False,records=None):
    from muscriptor import TranscriptionModel
    model=None;weights=Path('models/muscriptor-medium/model.safetensors');model_sha=sha(weights)
    for r in rows_for(test) if records is None else records:
        for kind in ('mix','guitar','bass'):
            path=REPORT/'midi'/r['id']/(kind+'.mid');meta=path.with_suffix('.json');audio_sha=sha(r['inputs'][kind])
            if path.exists() and meta.exists():
                m=json.loads(meta.read_text())
                if m['audio_sha256']!=audio_sha or m['model_sha256']!=model_sha:raise ValueError('Stale MIDI')
                continue
            path.parent.mkdir(parents=True,exist_ok=True);reuse=Path(r.get('reuse_midi',{}).get(kind,'missing'))
            if reuse.is_file():
                m=json.loads(reuse.with_suffix('.json').read_text())
                if m['audio_sha256']!=audio_sha or m['model_sha256']!=model_sha:raise ValueError('Stale reusable MIDI')
                shutil.copy2(reuse,path);save(meta,m);continue
            if model is None:model=TranscriptionModel.load_model(weights,device='cuda',dtype='float16')
            start=time.monotonic();mid,_=model.transcribe_and_postprocess(r['inputs'][kind],detect_tempo=False);path.write_bytes(mid)
            save(meta,{'audio_sha256':audio_sha,'model_sha256':model_sha,'seconds':time.monotonic()-start})
            print('CANDIDATE_MIDI',r['id'],kind,round(time.monotonic()-start,2),flush=True)

def cache(test=False,ready_only=False):
    import torch
    from band2tab.audio import features
    from band2tab.model import load
    from band2tab.temporal import encode_views
    from band2tab.refine import candidates_from_midi
    from band2tab.candidate import candidate_pool,candidate_features,targets
    if sha('models/acoustic_review.pt')!=json.loads((REPORT/'protocol.json').read_text())['encoder_sha256']:raise ValueError('Encoder changed')
    torch.set_num_threads(4);encoder,_=load('models/acoustic_review.pt','cuda');encoder.eval()
    for r in rows_for(test):
        path=ROOT/'cache'/(r['id']+'.npz');meta=path.with_suffix('.json')
        midis={k:REPORT/'midi'/r['id']/(k+'.mid') for k in ('mix','guitar','bass')}
        if ready_only and not all(p.is_file() and p.with_suffix('.json').is_file() for p in midis.values()):continue
        fingerprint={'audio':{k:sha(r['inputs'][k]) for k in midis},'midi':{k:sha(p) for k,p in midis.items()},'truth_sha256':hashlib.sha256(json.dumps(r['parts'],sort_keys=True).encode()).hexdigest(),
            'encoder_sha256':sha('models/acoustic_review.pt'),'feature_schema':'candidate1030-v1'}
        if path.exists() and meta.exists():
            if json.loads(meta.read_text())['fingerprint']!=fingerprint:raise ValueError('Stale candidate features')
            continue
        pool=candidate_pool({k:candidates_from_midi(p) for k,p in midis.items()},r['duration'])
        views=[features(r['inputs'][k]) for k in midis];reuse=Path(r.get('reuse_cache','missing'))
        if reuse.exists():
            with np.load(reuse) as f:e=f['e'][:,:3].copy()
            if len(e)!=min(map(len,views)):raise ValueError('Latent alignment mismatch')
        else:e=encode_views(encoder,views,'cuda')
        x,prior=candidate_features(views,e,pool,r['duration']);labels=targets(pool,r['parts'])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,x=x.astype(np.float16),prior=prior,**labels)
        save(meta,{'fingerprint':fingerprint,'pool':pool});print('CANDIDATE_CACHED',r['id'],len(pool),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','add_clean','bootstrap','separate','transcribe','transcribe_new','transcribe_clean','cache','cache_ready']);p.add_argument('--test',action='store_true');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='add_clean':add_clean()
    elif a.command=='bootstrap':transcribe(records=[r for r in json.loads(Path('data/temporal_v1/manifest.json').read_text()) if r['split'] in ('train','val')])
    elif a.command=='transcribe_new':transcribe(records=sorted([r for r in rows_for(a.test) if r['fresh_source']],key=lambda r:(r['split']!='val',r['id'])))
    elif a.command=='transcribe_clean':transcribe(records=[r for r in rows_for(False) if r['dataset']=='guitar_techs_clean'])
    elif a.command=='cache_ready':cache(a.test,ready_only=True)
    else:globals()[a.command](a.test)
