"""Official data downloads, deterministic song-level splits and label adapters."""
from pathlib import Path
import hashlib, json, shutil, tarfile, urllib.request, zipfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .audio import features, note_targets

FILES={
 'guitarset':[(3371780,'annotation.zip','b39b78e63d3446f2e54ddb7a54df9b10'),
              (3371780,'audio_mono-mic.zip','275966d6610ac34999b58426beb119c3')],
 'baby':[(4603870,'babyslakh_16k.tar.gz','311096dc2bde7d61c97e930edbfc7f78')]}
def _json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def _safe(dest,name):
    p=(dest/name).resolve()
    if not p.is_relative_to(dest.resolve()):raise ValueError('Unsafe archive path')
    return p

def fetch_ranges(url,path,partial):
    probe=urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=30)
    with probe:
        if probe.status!=206:return False
        total=int(probe.headers['Content-Range'].split('/')[-1])
    prefix=partial.stat().st_size if partial.exists() else 0
    if prefix>total:raise ValueError('Partial download larger than expected archive')
    if prefix==total:return True
    chunk=(total-prefix+3)//4;ranges=[(a,min(total-1,a+chunk-1)) for a in range(prefix,total,chunk)]
    def one(bounds):
        a,b=bounds;piece=path.with_name(path.name+f'.range-{a}-{b}')
        if piece.exists() and piece.stat().st_size==b-a+1:return piece
        for retry in range(3):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
                with urllib.request.urlopen(req,timeout=90) as response:
                    if response.status!=206:raise ValueError('Range request was not honored')
                    if response.headers.get('Content-Range')!=f'bytes {a}-{b}/{total}':raise ValueError('Unexpected range response')
                    with piece.open('wb') as f:
                        shutil.copyfileobj(response,f,1024**2)
                if piece.stat().st_size!=b-a+1:raise OSError('Incomplete range')
                print('Range complete',path.name,a,b,flush=True);return piece
            except Exception:
                if retry==2:raise
                time.sleep(3*(retry+1))
    with ThreadPoolExecutor(max_workers=4) as pool:pieces=list(pool.map(one,ranges))
    with partial.open('ab') as dest:
        for piece in pieces:
            with piece.open('rb') as src:shutil.copyfileobj(src,dest,1024**2)
    for piece in pieces:piece.unlink()
    return True

def fetch_checked(url,path,md5):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def valid(p):
        if not p.exists():return False
        with p.open('rb') as f:return hashlib.file_digest(f,'md5').hexdigest()==md5
    if valid(path):return
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            if fetch_ranges(url,path,partial):
                if not valid(partial):raise ValueError('Downloaded archive checksum mismatch')
                partial.replace(path);print('Checksum OK:',path.name,flush=True);return
            offset=partial.stat().st_size if partial.exists() else 0
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req,timeout=90) as response:
                resume=offset>0 and response.status==206
                count=offset if resume else 0;last=count//(64*1024**2)
                with partial.open('ab' if resume else 'wb') as dest:
                    while True:
                        block=response.read(1024**2)
                        if not block:break
                        dest.write(block);count+=len(block)
                        if count//(64*1024**2)>last:
                            last=count//(64*1024**2);print(path.name,round(count/1024**2),'MiB',flush=True)
            if not valid(partial):raise ValueError('Downloaded archive checksum mismatch')
            partial.replace(path);print('Checksum OK:',path.name,flush=True);return
        except ValueError:raise
        except Exception as e:
            print('Download retry',attempt+1,type(e).__name__,str(e),flush=True)
            if attempt==2:raise
            time.sleep(3*(attempt+1))

def download(which,root='data'):
    root=Path(root);dest=root/'raw'/which;dest.mkdir(parents=True,exist_ok=True)
    for record,name,md5 in FILES[which]:
        f=dest/name
        print('Fetching',name,flush=True)
        fetch_checked(f'https://zenodo.org/records/{record}/files/{name}?download=1',f,md5)
        if name.endswith('.zip'):
            with zipfile.ZipFile(f) as z:
                for member in z.infolist():_safe(dest,member.filename)
                z.extractall(dest)
        else:
            with tarfile.open(f) as t:
                for m in t:
                    if not m.isfile():continue
                    p=_safe(dest,m.name);p.parent.mkdir(parents=True,exist_ok=True)
                    with t.extractfile(m) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
    print('Ready:',dest)

def download_slakh_subset(root='data',train=100,val=20,test=20):
    """Read the official gzip stream once; save only selected complete songs.

    Network can still approach the full 105 GB archive. Disk avoids the archive
    copy and retains FLAC. Do not pretend that selecting 140 songs reduces
    compressed-stream network cost proportionally.
    """
    dest=Path(root)/'raw'/'slakh';dest.mkdir(parents=True,exist_ok=True)
    caps={'train':train,'validation':val,'test':test};selected={k:set() for k in caps}
    url='https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1'
    print('Large streaming transfer; keep at least 40 GB free. Use BabySlakh first.',flush=True)
    with urllib.request.urlopen(url,timeout=120) as response,tarfile.open(fileobj=response,mode='r|gz') as t:
        for m in t:
            if not m.isfile():continue
            parts=Path(m.name).parts
            split=next((s for s in caps if s in parts),None)
            track=next((s for s in parts if s.startswith('Track') and s[5:].isdigit()),None)
            if not split or not track:continue
            if track not in selected[split] and len(selected[split])<caps[split]:
                selected[split].add(track);print(split,track,flush=True)
            if track not in selected[split]:continue
            if shutil.disk_usage(dest).free<20*1024**3:raise RuntimeError('Less than 20 GiB free; transfer stopped')
            p=_safe(dest,m.name);p.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(m) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
    _json(dest/'subset.json',{k:sorted(v) for k,v in selected.items()})

def group_split(group):
    # Stable grouping across players, renderings, keys and crops.
    v=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10
    return 'test' if v==0 else 'val' if v==1 else 'train'

def guitarset_items(root):
    root=Path(root)
    wavs={p.stem.removesuffix('_mic'):p for p in root.rglob('*.wav')}
    for p in sorted(root.rglob('*.jams')):
        base=p.stem
        if base not in wavs:continue
        j=json.loads(p.read_text()); notes=[]
        for a in j.get('annotations',[]):
            if a['namespace']!='note_midi':continue
            for d in a['data']:
                v=d['value']
                pitch=float(v.get('pitch',v.get('value'))) if isinstance(v,dict) else float(v)
                note={'pitch':round(pitch),'onset':d['time'],'offset':d['time']+d['duration']}
                src=str(a.get('annotation_metadata',{}).get('data_source',''))
                if src.isdigit() and 0<=int(src)<6:
                    idx=int(src);note['gt_string']=6-idx;note['gt_fret']=round(pitch)-[40,45,50,55,59,64][idx]
                notes.append(note)
        # e.g. 00_BN1-129-Eb_comp -> BN1, shared across performers and keys.
        tokens=base.split('_');progression=tokens[1].split('-')[0]
        yield {'id':'guitarset_'+base,'dataset':'guitarset','part':'guitar','audio':str(wavs[base].resolve()),
               'group':'guitarset_'+progression,'split':group_split('guitarset_'+progression),'notes':notes}

def slakh_split_map(root):
    import yaml
    groups={};official={}
    for meta in sorted(Path(root).rglob('metadata.yaml')):
        if 'omitted' in meta.parts:continue
        j=yaml.safe_load(meta.read_text());g='slakh_'+str(j.get('UUID',meta.parent.name));groups[g]=None
        spl=next((x for x in ('train','validation','test') if x in meta.parts),None)
        if spl:
            spl={'validation':'val'}.get(spl,spl)
            if g in official and official[g]!=spl:raise ValueError('Official split has duplicate source MIDI across folds')
            official[g]=spl
    if official:return official
    ordered=sorted(groups,key=lambda g:hashlib.sha256(g.encode()).hexdigest())
    if len(ordered)<3:raise ValueError('Need at least three independent source MIDI groups')
    n=max(1,round(len(ordered)*.15))
    return {g:('test' if i<n else 'val' if i<2*n else 'train') for i,g in enumerate(ordered)}

def slakh_items(root,dataset):
    import yaml,pretty_midi
    split_map=slakh_split_map(root)
    for meta in sorted(Path(root).rglob('metadata.yaml')):
        if 'omitted' in meta.parts:continue
        j=yaml.safe_load(meta.read_text());folder=meta.parent;group=str(j.get('UUID',folder.name))
        split=split_map['slakh_'+group]
        for stem,info in j.get('stems',{}).items():
            cls=info.get('inst_class','').lower()
            part='guitar' if cls=='guitar' else 'bass' if cls=='bass' else None
            if part is None or info.get('is_drum',False):continue
            audio=next((folder/'stems'/f'{stem}{ext}' for ext in ('.flac','.wav') if (folder/'stems'/f'{stem}{ext}').exists()),None)
            midi=folder/'MIDI'/f'{stem}.mid'
            if audio is None or not midi.exists():continue
            notes=[{'pitch':n.pitch,'onset':n.start,'offset':n.end}
                   for ins in pretty_midi.PrettyMIDI(str(midi)).instruments if not ins.is_drum for n in ins.notes]
            yield {'id':dataset+'_'+folder.name+'_'+stem,'dataset':dataset,'part':part,
                   'audio':str(audio.resolve()),'metadata_audio_rendered':info.get('audio_rendered'),'group':'slakh_'+group,'split':split,'notes':notes}

def prepare(root='data',dataset='all'):
    root=Path(root);cache=root/'cache';cache.mkdir(parents=True,exist_ok=True)
    items=[];skipped=[]
    known_bad={'guitarset_04_BN3-154-E_comp','guitarset_04_Jazz1-200-B_comp','guitarset_02_Funk2-119-G_comp'}
    for name in ('guitarset','baby','slakh'):
        if dataset not in ('all',name):continue
        folder=root/'raw'/name
        if not folder.exists():continue
        source=guitarset_items(folder) if name=='guitarset' else slakh_items(folder,name)
        for it in source:
            if it['id'] in known_bad:
                skipped.append({'id':it['id'],'reason':'published_annotation_error','source':'https://github.com/marl/GuitarSet/issues/5 or /4'});continue
            xfile=cache/(it['id']+'.npz')
            if not xfile.exists():
                x=features(it['audio']);y,outside=note_targets(it['notes'],len(x))
                if outside:
                    # Excluding out-of-range examples is explicit, not false silence.
                    skipped.append({'id':it['id'],'reason':'pitch_outside_B0_E6','count':outside});continue
                if y[:,1].sum()==0:skipped.append({'id':it['id'],'reason':'no_in_range_notes'});continue
                np.savez_compressed(xfile,x=x,y=y)
            it['cache']=str(xfile.resolve());items.append(it);print('Cached',it['id'],it['split'],flush=True)
    if not items:raise ValueError('No usable pairs. Run fetch guitarset and/or fetch baby first.')
    # When Baby and redux overlap, retain the redux pair once with its official split.
    unique={}
    for it in sorted(items,key=lambda x:x['dataset']=='slakh'):
        key=(it['group'],it['audio'].split('/stems/')[-1]) if it['dataset']!='guitarset' else it['id']
        unique[key]=it
    items=list(unique.values())
    groups={}
    for it in items:
        if it['group'] in groups and groups[it['group']]!=it['split']:raise ValueError('Group split leakage detected')
        groups[it['group']]=it['split']
    _json(root/'manifest.json',items);_json(root/'excluded.json',skipped)
    counts={s:sum(i['split']==s for i in items) for s in ('train','val','test')}
    print('SPLITS',counts,'excluded',len(skipped))
    if not all(counts.values()):raise ValueError('A split is empty. Add data; do not split crops randomly.')
    return counts

def remix(root='data',limit=30,snr=0.):
    """Controlled mixture tests: real guitar/labelled bass + non-target backing.

    This creates derived mixtures from existing recordings, not new performances.
    No guitar/bass sources are allowed in the backing. It is not claimed to be
    a natural real-band benchmark. Official/group splits remain unchanged.
    """
    import random,librosa,soundfile as sf,yaml
    from .audio import SR
    root=Path(root);base=json.loads((root/'manifest.json').read_text());rng=random.Random(17)
    backing={'train':[],'val':[],'test':[]}
    for folder in ('baby','slakh'):
        if not (root/'raw'/folder).exists():continue
        split_map=slakh_split_map(root/'raw'/folder)
        for meta in sorted((root/'raw'/folder).rglob('metadata.yaml')):
            if 'omitted' in meta.parts:continue
            j=yaml.safe_load(meta.read_text());g='slakh_'+j.get('UUID',meta.parent.name)
            spl=split_map[g]
            for stem,info in j.get('stems',{}).items():
                if info.get('inst_class','').lower() in ('guitar','bass'):continue
                for ext in ('.flac','.wav'):
                    p=meta.parent/'stems'/(stem+ext)
                    if p.exists():backing[spl].append(p);break
    extra=[];dest=root/'derived';dest.mkdir(exist_ok=True)
    for spl in ('train','val','test'):
        candidates=[]
        for part in ('guitar','bass'):
            part_items=[it for it in base if it['split']==spl and it['part']==part];rng.shuffle(part_items);candidates.extend(part_items[:limit])
        if not backing[spl]:raise ValueError(f'No non-guitar/bass backing in {spl}. Fetch and prepare BabySlakh or redux.')
        for it in candidates:
            y,_=librosa.load(it['audio'],sr=SR,mono=True);yrms=float(np.sqrt(np.mean(y*y)))
            if yrms<1e-7:raise ValueError('Target recording is silent: '+it['id'])
            for attempt in range(50):
                bg=rng.choice(backing[spl]);b,_=librosa.load(str(bg),sr=SR,mono=True)
                if len(b)==0:continue
                b=np.roll(b,rng.randrange(len(b)));b=np.tile(b,int(np.ceil(len(y)/len(b))))[:len(y)]
                brms=float(np.sqrt(np.mean(b*b)))
                if brms>=1e-7:break
            else:raise ValueError('Cannot find non-silent background in '+spl)
            b=b*(yrms/brms)*10**(-snr/20)
            mix=y+b;mix=mix/max(1.,float(np.max(np.abs(mix)))/.98)
            ident=it['id']+'_remix';wav=dest/(ident+'.wav');sf.write(wav,mix,SR,subtype='PCM_16')
            x=features(wav);target,outside=note_targets(it['notes'],len(x))
            cache=root/'cache'/(ident+'.npz');np.savez_compressed(cache,x=x,y=target)
            extra.append({**it,'id':ident,'dataset':'remix','audio':str(wav.resolve()),'cache':str(cache.resolve()),
                          'backing':str(bg.resolve()),'source_dataset':it['dataset'],'snr_db':snr})
    # Validation/test are identical in A0/A1; only training mixtures differ.
    _json(root/'manifest_clean.json',base+[x for x in extra if x['split']!='train'])
    _json(root/'manifest_robust.json',base+extra)
    print('Derived mixtures:',len(extra),'A0/A1 manifests ready')
