"""Controlled frozen-MuScriptor separation ablation. Never select on test outputs."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,os,time
import numpy as np
import soundfile as sf
from band2tab.separation import write_input,resample,separate_inputs

ROOT=Path('data/separation_v1');REPORT=Path('reports/separation_v1')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

def prepare():
    from band2tab.data import slakh_items
    manifest=ROOT/'manifest.json'
    if manifest.exists():raise RuntimeError('Already frozen')
    rows=json.loads(Path('data/manifest_duration.json').read_text());chosen=[]
    for r in rows:
        dataset=r.get('source_dataset','p3')
        if dataset=='baby' and r['kind']=='mixture' and not r['id'].startswith('ref_mix_'):continue
        if r['split']=='train' and r['kind']=='solo':continue
        chosen.append(r)
    # Add modest solo training coverage, fixed by hash before any predictions.
    for dataset in ('guitarset','filobass'):
        pool=[r for r in rows if r['split']=='train' and r['kind']=='solo' and r.get('source_dataset')==dataset]
        chosen+=sorted(pool,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:8]
    groups=defaultdict(list)
    for it in json.loads(Path('data/manifest_mix_diagnostic.json').read_text()):groups[it['group']].append(it)
    for g,items in groups.items():
        r=items[0]
        chosen.append({'id':'historical_'+Path(r['audio']).stem,'group':g,'split':'test','kind':'mixture',
            'audio':r['audio'],'parts':{part:[n for it in items if it['part']==part for n in it['notes']] for part in ('guitar','bass')},
            'duration':sf.info(r['audio']).duration,'excerpt_start':0.,'source_dataset':'baby',
            'audio_sha256':sha(r['audio']),'evaluation':'previously_inspected_mix3'})
    stems=defaultdict(lambda:defaultdict(list))
    for it in slakh_items('data/raw/baby','baby'):stems[it['group']][it['part']].append(it['audio'])
    output=[]
    for r in chosen:
        r=dict(r);r['guard_seconds']=0. if r.get('source_dataset','p3')=='p3' else .25
        r['evaluation']=r.get('evaluation', 'previously_inspected_p3' if r['split']=='test' else r.get('source_dataset','p3'))
        folder=ROOT/'audio'/r['id'];folder.mkdir(parents=True,exist_ok=True)
        r['inputs']={k:str((folder/(k+'.wav')).resolve()) for k in ('mix','no_drums_vocals','guitar','bass')}
        if r.get('source_dataset')=='baby' and r['kind']=='mixture':
            for part in ('guitar','bass'):
                arrays=[]
                for path in stems[r['group']][part]:
                    info=sf.info(path)
                    with sf.SoundFile(path) as f:
                        f.seek(round(r['excerpt_start']*info.samplerate));y=f.read(round(r['duration']*info.samplerate),dtype='float32',always_2d=True)
                    arrays.append(resample(y.mean(1),info.samplerate,16000))
                size=round(r['duration']*16000);combined=np.zeros(size,np.float32)
                for y in arrays:combined[:min(len(y),size)]+=y[:size]
                path=folder/('oracle_'+part+'.wav');write_input(path,combined,16000);r['inputs']['oracle_'+part]=str(path.resolve())
        output.append(r)
    splits=defaultdict(set)
    for r in output:
        splits[r['group']].add(r['split'])
        if r.get('guitar_group'):splits[r['guitar_group']].add(r['split'])
    assert all(len(v)==1 for v in splits.values())
    save(manifest,output)
    protocol={'date':'2026-11-18','manifest_sha256':sha(manifest),'model':'MuScriptor medium frozen',
        'model_sha256':sha('models/muscriptor-medium/model.safetensors'),'muscriptor_version':'0.4.0',
        'separator':'htdemucs_6s','separator_config':{'shifts':1,'overlap':.25,'segment':7},
        'canonical_audio':'mono 16000Hz float WAV, no peak normalization, all routes',
        'inference':{'detect_tempo':False,'cfg_coef':1.,'beam_size':1},
        'counts':{str(k):v for k,v in Counter((r['split'],r['kind']) for r in output).items()},
        'selection':'train/validation only; test is previously inspected regression, not a new holdout',
        'test_warning':'P3/FiloBass/Baby test sets already inspected in previous experiments; pretrained data overlap unknown',
        'vocal_warning':'These sources do not establish performance on real vocal-band mixtures',
        'metrics':'B0-E6, min30ms; 0.25s guards on excerpts, all output on P3 whole recordings'}
    save(REPORT/'protocol.json',protocol);print('FROZEN',protocol['counts'],flush=True)

def get_rows(test):
    protocol=json.loads((REPORT/'protocol.json').read_text())
    if sha(ROOT/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    if test and not (REPORT/'selection.json').exists():raise RuntimeError('Freeze validation selection before test')
    rows=[r for r in json.loads((ROOT/'manifest.json').read_text()) if (r['split']=='test')==test]
    if test and (ROOT/'manifest_amp_condition.json').exists():
        amp_protocol=json.loads((REPORT/'amp_protocol.json').read_text())
        if sha(ROOT/'manifest_amp_condition.json')!=amp_protocol['manifest_sha256']:raise ValueError('Amp condition changed')
        rows+=json.loads((ROOT/'manifest_amp_condition.json').read_text())
    return rows

def prepare_amp():
    """Unused amp-microphone audio, same P3 performances: a new condition only."""
    import zipfile
    from band2tab.adapt import alignment
    manifest=ROOT/'manifest_amp_condition.json'
    if manifest.exists() or (REPORT/'selection.json').exists():raise RuntimeError('Prepare before selection, never overwrite')
    archive=Path('data/raw/guitar_techs/P3_music.zip')
    spec=next(f for f in json.loads(Path('reports/guitar_techs_record.json').read_text())['files'] if f['name']==archive.name)
    if hashlib.md5(archive.read_bytes()).hexdigest()!=spec['checksum'].split(':')[-1]:raise ValueError('Archive checksum mismatch')
    rows=[];qa=[]
    with zipfile.ZipFile(archive) as z:
        for r in json.loads(Path('data/manifest_duration.json').read_text()):
            if not r['id'].startswith('duration_P3_music_'):continue
            name=Path(r['audio']).name.replace('directinput_','micamp_');member='P3_music/audio/micamp/'+name
            path=Path('data/raw/guitar_techs')/member;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(member))
            notes=r['unshifted_parts']['guitar'];lag=alignment(path,notes)
            qa.append({'id':r['id'],'audio':str(path),'accepted':lag['accepted'],**lag})
            if not lag['accepted']:continue
            duration=sf.info(path).duration
            truth=[{**n,'onset':max(0,n['onset']+lag['shift_seconds']),'offset':min(duration,n['offset']+lag['shift_seconds'])} for n in notes]
            if any(n['onset']>=n['offset'] for n in truth):raise ValueError('Invalid amp note time')
            ident='amp_condition_'+r['id'];folder=ROOT/'audio'/ident
            rows.append({'id':ident,'audio':str(path.resolve()),'audio_sha256':sha(path),'group':r['group'],'split':'test','kind':'solo',
                'parts':{'guitar':truth,'bass':[]},'unshifted_parts':r['unshifted_parts'],'annotation_alignment':lag,
                'source_dataset':'guitar_techs_amp_new_condition','evaluation':'new_amp_audio_of_previously_inspected_p3_performances',
                'duration':duration,'excerpt_start':0.,'guard_seconds':0.,
                'inputs':{k:str((folder/(k+'.wav')).resolve()) for k in ('mix','no_drums_vocals','guitar','bass')}})
    save(manifest,rows);save(REPORT/'amp_protocol.json',{'manifest_sha256':sha(manifest),'qa':qa,
        'before_test_inference':True,'before_selection':True,'annotation_qa':'same fixed onset-alignment rule as prior P3; no model output used',
        'scope':'Previously unused micamp signals of the SAME P3 performances; not new independent songs or performers; no vocals/drums'})
    print('AMP_CONDITION_FROZEN',len(rows),flush=True)

def separate(test):
    import torch
    from demucs.pretrained import get_model
    os.environ['TORCH_HOME']=str(Path('models/torch').resolve());torch.set_num_threads(4);torch.manual_seed(17)
    model=get_model('htdemucs_6s').to('cuda').eval()
    for i,r in enumerate(get_rows(test)):
        folder=ROOT/'audio'/r['id'];meta=folder/'separation.json'
        if meta.exists() and all(Path(p).exists() for p in r['inputs'].values()):continue
        if sha(r['audio'])!=r['audio_sha256']:raise ValueError('Source changed')
        start=time.monotonic();info=separate_inputs(r['audio'],folder,model)
        info.update(audio_sha256=r['audio_sha256'],seconds=time.monotonic()-start,
                    input_sha256={k:sha(p) for k,p in r['inputs'].items()})
        save(meta,info);print('SEPARATED',i+1,r['id'],round(info['seconds'],2),flush=True)

def predict(test):
    from muscriptor import TranscriptionModel
    import importlib.metadata
    if importlib.metadata.version('muscriptor')!='0.4.0':raise ValueError('Unexpected model runtime')
    weights=Path('models/muscriptor-medium/model.safetensors');protocol=json.loads((REPORT/'protocol.json').read_text())
    if sha(weights)!=protocol['model_sha256']:raise ValueError('Model changed')
    model=TranscriptionModel.load_model(weights,device='cuda',dtype='float16')
    for i,r in enumerate(get_rows(test)):
        for kind,path in r['inputs'].items():
            dest=REPORT/'midi'/r['id']/(kind+'.mid');meta=dest.with_suffix('.json');h=sha(path)
            if dest.exists() and meta.exists():
                old=json.loads(meta.read_text())
                if old['audio_sha256']!=h:raise ValueError('Cached input changed')
                continue
            start=time.monotonic();mid,_=model.transcribe_and_postprocess(path,detect_tempo=False)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(mid)
            info={'audio_sha256':h,'model_sha256':protocol['model_sha256'],'seconds':time.monotonic()-start}
            save(meta,info);print('PREDICTED',i+1,r['id'],kind,round(info['seconds'],2),flush=True)

def methods(r):
    from band2tab.refine import candidates_from_midi
    def read(kind):
        notes=candidates_from_midi(REPORT/'midi'/r['id']/(kind+'.mid'))
        return {p:[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes if n['baseline_class']==c] for c,p in ((1,'guitar'),(2,'bass'))}
    out={'M0_original':read('mix'),'M1_no_drums_vocals':read('no_drums_vocals'),
         'M2_separated_labels':{p:read(p)[p] for p in ('guitar','bass')}}
    for prefix,name in (('', 'M2_separated_forced'),('oracle_','oracle_forced')):
        if prefix and 'oracle_guitar' not in r['inputs']:continue
        out[name]={p:[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in candidates_from_midi(REPORT/'midi'/r['id']/(prefix+p+'.mid'))] for p in ('guitar','bass')}
    from band2tab.fusion import candidate_pool,propose,PARTS
    pools=candidate_pool(candidates_from_midi(REPORT/'midi'/r['id']/'mix.mid'),
                         {p:candidates_from_midi(REPORT/'midi'/r['id']/(p+'.mid')) for p in PARTS})
    union_probability=[float(n['keep'] or n['support'][cls]>0) for cls,p in enumerate(PARTS,1) for n in pools[p]]
    out['M3_naive_union']=propose(pools,union_probability,add_threshold=.5)[0]
    return out

def score(test):
    from band2tab.refine_duration import instrument_metrics
    rows=[]
    for r in get_rows(test):
        guard=r['guard_seconds'];scored={name:instrument_metrics(r['parts'],parts,start=guard,end=r['duration']-guard if guard else None) for name,parts in methods(r).items()}
        rows.append({'id':r['id'],'group':r['group'],'split':r['split'],'kind':r['kind'],'dataset':r.get('source_dataset','p3'),'metrics':scored})
    summary={}
    for split in sorted({r['split'] for r in rows}):
        for ds in sorted({r['dataset'] for r in rows if r['split']==split}):
            use=[r for r in rows if r['split']==split and r['dataset']==ds];names=set.intersection(*(set(r['metrics']) for r in use))
            summary[split+'/'+ds]={'n':len(use),'methods':{name:{k:float(np.mean([r['metrics'][name][k] for r in use])) for k in ('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')} for name in sorted(names)}}
    path=REPORT/('comparison_test.json' if test else 'comparison_development.json')
    if path.exists():raise RuntimeError('Do not overwrite frozen scoring')
    save(path,{'summary':summary,'tracks':rows});print('COMPARISON',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','prepare_amp','separate','predict','score']);p.add_argument('--test',action='store_true');a=p.parse_args()
    if a.command in ('prepare','prepare_amp'):globals()[a.command]()
    else:globals()[a.command](a.test)
