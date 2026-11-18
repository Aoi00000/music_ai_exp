"""Frozen temporal experiment: new songs, hard remixes, two separators, caches."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib,shutil,os,time,random
import numpy as np
import soundfile as sf
from band2tab.separation import resample,write_input,separate_inputs
ROOT=Path('data/temporal_v1');REPORT=Path('reports/temporal_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2));tmp.replace(p)
def read_clip(path,start,duration):
    info=sf.info(path)
    with sf.SoundFile(path) as f:f.seek(round(start*info.samplerate));y=f.read(round(duration*info.samplerate),dtype='float32',always_2d=True)
    return resample(y.mean(1),info.samplerate,16000)
def clipped_truth(parts,start,duration):
    return {p:[{**n,'onset':max(0.,n['onset']-start),'offset':min(start+duration,n['offset'])-start,
                     'censored_onset':bool(n['onset']<start),'censored_offset':bool(n['offset']>start+duration)}
               for n in parts.get(p,[]) if n['onset']<start+duration and n['offset']>start] for p in ('guitar','bass')}

def prepare():
    import yaml
    from band2tab.data import slakh_items
    if (ROOT/'manifest.json').exists():raise RuntimeError('Temporal manifest already frozen')
    ROOT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(exist_ok=True)
    acquisition=json.loads((REPORT/'source_acquisition.json').read_text());rows=[]
    for path in ('data/separation_v1/manifest.json','data/separation_v1/manifest_amp_condition.json'):
        for old in json.loads(Path(path).read_text()):
            r=dict(old);r['hard_training']=False;r['fresh_source']=False;r['dataset']=r.get('source_dataset','p3')
            r['inputs']=dict(old['inputs']);r['inputs']['ft_clean']=str((ROOT/'audio'/r['id']/'ft_clean.wav').resolve())
            r['reuse_midi']={k:str((Path('reports/separation_v1/midi')/r['id']/(k+'.mid')).resolve()) for k in ('mix','guitar','bass')}
            rows.append(r)
    eligible={r['group'] for r in acquisition['selected']};groups=defaultdict(list)
    for r in slakh_items('data/raw/slakh_temporal','slakh_temporal'):
        if r['group'] in eligible:groups[r['group']].append(r)
    if set(groups)!=eligible:raise ValueError('Missing selected source annotations')
    for source in acquisition['selected']:
        if (REPORT/'label_qc_policy.json').exists() and source['group'] in json.loads((REPORT/'label_qc_policy.json').read_text())['quarantined_groups']:
            continue
        folder=Path(source['folder']);path=folder/'mix.flac';info=sf.info(path);split=source['split']
        duration=min(12. if split=='train' else 24. if split=='val' else 30.,info.duration)
        parts={p:[n for r in groups[source['group']] if r['part']==p for n in r['notes']] for p in ('guitar','bass')}
        for j,fraction in enumerate((.2,.65) if split=='train' else (.35,)):
            start=max(0.,(info.duration-duration)*fraction);ident='new_'+source['track']+'_'+str(j)
            dest=ROOT/'audio'/ident;write_input(dest/'mix.wav',read_clip(path,start,duration),16000)
            actual=sf.info(dest/'mix.wav').duration
            rows.append({'id':ident,'group':source['group'],'split':split,'kind':'mixture','dataset':'slakh_new',
                'audio':str((dest/'mix.wav').resolve()),'duration':actual,'excerpt_start':start,
                'parts':clipped_truth(parts,start,actual),'guard_seconds':.25,'hard_training':False,
                'fresh_source':True,'inputs':{k:str((dest/(k+'.wav')).resolve()) for k in ('mix','guitar','bass','no_drums_vocals','ft_clean')}})
    backing=defaultdict(list)
    for path in Path('data/raw/baby').rglob('metadata.yaml'):
        meta=yaml.safe_load(path.read_text());g='slakh_'+str(meta['UUID'])
        matches=[r for r in rows if r['group']==g]
        if not matches:continue
        split=matches[0]['split'];files=[]
        for stem,info in meta['stems'].items():
            if not (info.get('is_drum',False) or info.get('inst_class','').lower()=='piano'):continue
            f=next((path.parent/'stems'/(stem+ext) for ext in ('.wav','.flac') if (path.parent/'stems'/(stem+ext)).exists()),None)
            if f:files.append(str(f.resolve()))
        if files:backing[split].append({'group':g,'audio':files})
    # Preserve source split and target notes; add only known non-target stems.
    sources=[r for r in rows if r['dataset']=='filobass_guitarset_mix']
    # A small electric-guitar hard-domain supplement, training only.
    duration_rows=json.loads(Path('data/manifest_duration.json').read_text())
    techs=[r for r in duration_rows if r['split']=='train' and r['id'].startswith(('duration_P1_','duration_P2_'))]
    techs.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
    for r in techs[:16]:
        r={**r,'inputs':{'mix':r['audio']},'dataset':'guitar_techs','guard_seconds':.25,'fresh_source':False}
        sources.append(r)
    for r in sources:
        group_list=sorted(backing[r['split']],key=lambda b:hashlib.sha256((r['id']+b['group']).encode()).hexdigest());b=group_list[0]
        y,sr=sf.read(r['inputs']['mix'],dtype='float32',always_2d=True);y=resample(y.mean(1),sr,16000)
        extra=np.zeros_like(y)
        for p in b['audio']:
            start=max(0.,(sf.info(p).duration-len(y)/16000)*.4);z=read_clip(p,start,len(y)/16000)
            extra[:min(len(z),len(y))]+=z[:len(y)]
        rms=lambda a:max(1e-7,float(np.sqrt(np.mean(a*a))))
        h=int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16);level=(-6,0,6)[h%3]
        mixed=y+extra/rms(extra)*rms(y)*10**(level/20)
        # A memoryless drive changes timbre without shifting note times/pitches.
        drive=(1.,2.5)[(h//3)%2];mixed=np.tanh(mixed/rms(mixed)*.15*drive)
        ident='hard_'+r['id'];dest=ROOT/'audio'/ident;write_input(dest/'mix.wav',mixed,16000)
        rows.append({**r,'id':ident,'audio':str((dest/'mix.wav').resolve()),'hard_training':True,
            'dataset':'hard_'+r['dataset'],'backing_group':b['group'],'backing_split':r['split'],
            'augmentation':{'background_to_target_db':level,'drive':drive},
            'inputs':{k:str((dest/(k+'.wav')).resolve()) for k in ('mix','guitar','bass','no_drums_vocals','ft_clean')}})
        rows[-1].pop('reuse_midi',None)
    mapping=defaultdict(set)
    for r in rows:
        r['audio_sha256']=sha(r['audio'])
        mapping[r['group']].add(r['split'])
        for key in ('guitar_group','backing_group'):
            if r.get(key):mapping[r[key]].add(r['split'])
    if any(len(v)>1 for v in mapping.values()):raise ValueError('Source group leakage')
    save(ROOT/'manifest.json',rows)
    protocol={'created':'2026-11-18','manifest_sha256':sha(ROOT/'manifest.json'),
        'source_acquisition_sha256':sha(REPORT/'source_acquisition.json'),'base_model_sha256':sha('models/muscriptor-medium/model.safetensors'),
        'encoder_sha256':sha('models/acoustic_review.pt'),'training_conditions':['standard','hard'],
        'updates':[500,1000,2000,3000],'batch':8,'window_frames':344,'seed':17,
        'decoding_grid':{'onset':[.3,.5,.7],'frame':[.3,.5,.7],'offset':[.5,1.1],'min_duration':[.03]},
        'selection':'Validation only, dataset macro mean(Note F1 + instrument Frame F1 + union Frame F1); identity candidate retained; no automatic deployment',
        'test_used_for_selection':False,'fresh_test':'8 Slakh source UUIDs absent from all previous own training/evaluation; pretrained-model overlap unknown',
        'counts':{str(k):v for k,v in Counter((r['split'],r['hard_training'],r['dataset']) for r in rows).items()},
        'scope':'Synthetic new-source bands and constructed real-instrument remixes. No natural vocal-band claim.'}
    save(REPORT/'protocol.json',protocol);print('TEMPORAL_DATA_FROZEN',protocol['counts'],flush=True)

def quarantine():
    """Ambiguous training-discovered patch is withheld, not transposed by guess."""
    import yaml
    if (REPORT/'selection.json').exists():raise RuntimeError('Cannot change label policy after selection')
    if (REPORT/'label_qc_policy.json').exists():raise RuntimeError('Label policy already fixed')
    qc=json.loads((REPORT/'slap_independent_qc.json').read_text())
    if qc['split']!='train only' or len(qc['results'])<2:raise ValueError('Insufficient training-only QC')
    bad=[]
    for s in json.loads((REPORT/'source_acquisition.json').read_text())['selected']:
        metadata=yaml.safe_load((Path(s['folder'])/'metadata.yaml').read_text())
        if any(v.get('plugin_name')=='scarbee_jay_bass_slap_both.nkm' for v in metadata['stems'].values()):bad.append(s['group'])
    old=json.loads((ROOT/'manifest.json').read_text());kept=[r for r in old if not (r['fresh_source'] and r['group'] in bad)]
    shutil.copy2(ROOT/'manifest.json',ROOT/'manifest_before_label_qc.json');shutil.copy2(REPORT/'protocol.json',REPORT/'protocol_before_label_qc.json')
    policy={'quarantined_groups':bad,'quarantined_plugin':'scarbee_jay_bass_slap_both.nkm',
        'basis':'Training-only YIN and FFT/autocorrelation disagree with published octave exception; annotation/perceptual pitch ambiguity unresolved',
        'action':'Exclude these whole source songs across splits before model prediction/learning. Preserve original MIDI; do not guess a new transposition.',
        'excluded_records':[r['id'] for r in old if r not in kept],'before_test_inference':True}
    save(REPORT/'label_qc_policy.json',policy);save(ROOT/'manifest.json',kept)
    proto=json.loads((REPORT/'protocol.json').read_text());proto.update(manifest_sha256=sha(ROOT/'manifest.json'),
        label_qc_policy_sha256=sha(REPORT/'label_qc_policy.json'),
        counts={str(k):v for k,v in Counter((r['split'],r['hard_training'],r['dataset']) for r in kept).items()})
    save(REPORT/'protocol.json',proto);print('LABEL_QC_FROZEN','quarantined songs',len(bad),'excluded clips',len(old)-len(kept),flush=True)

def rows_for(test):
    proto=json.loads((REPORT/'protocol.json').read_text())
    if sha(ROOT/'manifest.json')!=proto['manifest_sha256']:raise ValueError('Manifest changed')
    if test and not (REPORT/'selection.json').exists():raise RuntimeError('Freeze validation selection before test inference')
    return [r for r in json.loads((ROOT/'manifest.json').read_text()) if (r['split']=='test')==test]

def separate(test):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    torch.set_num_threads(4);os.environ['TORCH_HOME']=str(Path('models/torch').resolve());six=ft=None
    for i,r in enumerate(rows_for(test)):
        folder=ROOT/'audio'/r['id'];meta=folder/'separation.json'
        if meta.exists() and all(Path(p).exists() for p in r['inputs'].values()):continue
        folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();seed=int(sha(r['audio'])[:8],16)%2**31;torch.manual_seed(seed);random.seed(seed)
        if not all(Path(r['inputs'][k]).exists() for k in ('guitar','bass','no_drums_vocals')):
            if six is None:six=get_model('htdemucs_6s').to('cuda').eval()
            separate_inputs(r['audio'],folder,six)
        if not Path(r['inputs']['ft_clean']).exists():
            if ft is None:ft=get_model('htdemucs_ft').to('cuda').eval()
            y,sr=sf.read(r['inputs']['mix'],dtype='float32',always_2d=True)
            if y.shape[1]==1:y=np.repeat(y,2,axis=1)
            y=resample(y,sr,ft.samplerate);wav=torch.from_numpy(np.ascontiguousarray(y.T));ref=wav.mean(0);mean=ref.mean();std=ref.std().clamp_min(1e-8)
            with torch.no_grad():parts=apply_model(ft,((wav-mean)/std)[None],device='cuda',shifts=1,split=True,overlap=.25,segment=7)[0].cpu()*std+mean
            keep=[ft.sources.index(p) for p in ('bass','other')];write_input(r['inputs']['ft_clean'],parts[keep].sum(0).numpy().T,ft.samplerate)
        save(meta,{'source_audio_sha256':sha(r['audio']),'seconds':time.monotonic()-start,
                   'inputs_sha256':{k:sha(p) for k,p in r['inputs'].items()}})
        print('SEPARATED',i+1,r['id'],round(time.monotonic()-start,2),flush=True)

def predict(test):
    from muscriptor import TranscriptionModel
    import yaml
    model=None;weights=Path('models/muscriptor-medium/model.safetensors');model_sha=sha(weights)
    sources={s['group']:s for s in json.loads((REPORT/'source_acquisition.json').read_text())['selected']}
    for i,r in enumerate(rows_for(test)):
        kinds=['mix'] if r['split']=='train' else ['mix','guitar','bass','ft_clean']
        inputs=dict(r['inputs'])
        if r['split']!='train' and r['fresh_source']:
            source=Path(sources[r['group']]['folder']);metadata=yaml.safe_load((source/'metadata.yaml').read_text())
            for part in ('guitar','bass'):
                path=ROOT/'oracle'/r['id']/(part+'.wav');combined=np.zeros(round(r['duration']*16000),np.float32)
                for stem,info in metadata['stems'].items():
                    if info.get('inst_class','').lower()!=part:continue
                    z=read_clip(source/'stems'/(stem+'.flac'),r['excerpt_start'],r['duration'])
                    combined[:min(len(z),len(combined))]+=z[:len(combined)]
                if not path.exists():write_input(path,combined,16000)
                inputs['oracle_'+part]=str(path);kinds.append('oracle_'+part)
        for kind in kinds:
            path=REPORT/'midi'/r['id']/(kind+'.mid');meta=path.with_suffix('.json');path.parent.mkdir(parents=True,exist_ok=True)
            audio_sha=sha(inputs[kind])
            if path.exists() and meta.exists():
                if json.loads(meta.read_text())['audio_sha256']!=audio_sha:raise ValueError('Stale MIDI')
                continue
            reuse=Path(r.get('reuse_midi',{}).get(kind,'missing'))
            if reuse.is_file():
                old=json.loads(reuse.with_suffix('.json').read_text())
                if old['audio_sha256']!=audio_sha or old['model_sha256']!=model_sha:raise ValueError('Stale reusable MIDI')
                shutil.copy2(reuse,path);save(meta,old);continue
            if model is None:model=TranscriptionModel.load_model(weights,device='cuda',dtype='float16')
            start=time.monotonic();mid,_=model.transcribe_and_postprocess(inputs[kind],detect_tempo=False);path.write_bytes(mid)
            save(meta,{'audio_sha256':audio_sha,'model_sha256':model_sha,'seconds':time.monotonic()-start,'detect_tempo':False})
            print('TRANSCRIBED',i+1,r['id'],kind,round(time.monotonic()-start,2),flush=True)

def cache(test):
    import torch
    from band2tab.model import load
    from band2tab.audio import features
    from band2tab.temporal import encode_views,piano_roll
    from band2tab.score import read_midi
    torch.set_num_threads(4);encoder,_=load('models/acoustic_review.pt','cuda');encoder.eval()
    for i,r in enumerate(rows_for(test)):
        path=ROOT/'cache'/(r['id']+'.npz');meta=path.with_suffix('.json')
        midi=REPORT/'midi'/r['id']/'mix.mid';fingerprint={'midi':sha(midi),'encoder':sha('models/acoustic_review.pt'),
            'audio':{k:sha(r['inputs'][k]) for k in ('mix','guitar','bass','ft_clean')}}
        if path.exists() and meta.exists():
            if json.loads(meta.read_text())!=fingerprint:raise ValueError('Stale features')
            continue
        views=[features(r['inputs'][k]) for k in ('mix','guitar','bass','ft_clean')];e=encode_views(encoder,views,'cuda')
        parts=read_midi(midi);m=piano_roll(parts,len(e));y=piano_roll(r['parts'],len(e),True)
        path.parent.mkdir(exist_ok=True);np.savez_compressed(path,e=e,m=m.astype(np.uint8),y=y.astype(np.uint8));save(meta,fingerprint)
        print('CACHED',i+1,r['id'],len(e),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','quarantine','separate','predict','cache']);p.add_argument('--test',action='store_true');a=p.parse_args()
    if a.command in ('prepare','quarantine'):globals()[a.command]()
    else:globals()[a.command](a.test)
