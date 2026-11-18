"""Pinned file-level Slakh mirror subset; skip every previously used source UUID.

Mirror hashes verify the mirror snapshot, not the official 105GB tarball.
Official rendering metadata and per-stem MIDI are retained for provenance.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,urllib.request,time,re,shutil
import yaml,pretty_midi

REPO='J1mmymm/MIMuT_Data_v2';REV='3a62b3a1c8bb263a6a6ac37559070339ddb81080'
API=f'https://huggingface.co/api/datasets/{REPO}'
ROOT=Path('data/raw/slakh_temporal');REPORT=Path('reports/temporal_v1')

def request(url):
    for attempt in range(4):
        try:return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Band2Tab-research'}),timeout=90)
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)

def listing(path,recursive=False):
    url=f'{API}/tree/{REV}/{path}?limit=1000'+('&recursive=true' if recursive else '')
    out=[]
    while url:
        with request(url) as r:
            out+=json.load(r);link=r.headers.get('Link','')
        nxt=re.search(r'<([^>]+)>;\s*rel="next"',link);url=nxt.group(1) if nxt else None
    return out

def download(spec,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    def valid(p):
        if not p.exists() or p.stat().st_size!=spec['size']:return False
        data=p.read_bytes()
        if spec.get('lfs'):return hashlib.sha256(data).hexdigest()==spec['lfs']['oid']
        return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==spec['oid']
    if not valid(dest):
        with request(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{spec["path"]}') as r:
            temp=dest.with_suffix(dest.suffix+'.part')
            with temp.open('wb') as f:shutil.copyfileobj(r,f,1024**2)
        if not valid(temp):raise ValueError('Mirror file hash/length mismatch: '+spec['path'])
        temp.replace(dest)
    return {'path':str(dest.resolve()),'mirror_path':spec['path'],'bytes':spec['size'],
            'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'mirror_hash_verified':True}

def main(root=ROOT,report=REPORT,caps=None,exclude_acquisitions=(),excluded_plugins=(),sort_salt='',budget_gib=6):
    import soundfile as sf
    root=Path(root);report=Path(report)
    root.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    ledger_path=report/'source_acquisition.json'
    if ledger_path.exists():raise RuntimeError('Source subset already frozen')
    previous=json.loads(Path('data/manifest_duration.json').read_text())
    used={r['group'].removeprefix('slakh_') for r in previous if r['group'].startswith('slakh_')}
    # Include the historical Baby test groups too (not in duration train/val).
    used|={r['group'].removeprefix('slakh_') for r in json.loads(Path('data/manifest_mix_diagnostic.json').read_text())}
    for path in exclude_acquisitions:
        used|={r['group'].removeprefix('slakh_') for r in json.loads(Path(path).read_text())['selected']}
    caps=caps or {'train':32,'validation':8,'test':8};selected=[];rejected=[];seen=set(used);planned_bytes=0
    for split,cap in caps.items():
        dirs=[p for p in listing('data/Slakh2100_redux/'+split) if p['type']=='directory' and Path(p['path']).name.startswith('Track')]
        dirs.sort(key=lambda p:hashlib.sha256((sort_salt+p['path']).encode()).hexdigest())
        accepted=0
        for d in dirs:
            specs={Path(p['path']).relative_to(d['path']).as_posix():p for p in listing(d['path'],True) if p['type']=='file'}
            if 'metadata.yaml' not in specs:continue
            folder=root/split/Path(d['path']).name
            meta_record=download(specs['metadata.yaml'],folder/'metadata.yaml');meta=yaml.safe_load((folder/'metadata.yaml').read_text());uuid=str(meta['UUID'])
            if uuid in seen:continue
            if any(v.get('plugin_name') in excluded_plugins for v in meta['stems'].values()):
                rejected.append({'track':d['path'],'reason':'training_discovered_ambiguous_plugin'});continue
            targets={p:[] for p in ('guitar','bass')};backing=[];records=[meta_record];all_valid=True
            for stem,info in meta['stems'].items():
                cls=info.get('inst_class','').lower();part=cls if cls in targets else None
                audio='stems/'+stem+'.flac';midi='MIDI/'+stem+'.mid'
                if part:
                    if audio not in specs or midi not in specs:all_valid=False;break
                    records.append(download(specs[midi],folder/midi))
                    notes=[n for ins in pretty_midi.PrettyMIDI(str(folder/midi)).instruments if not ins.is_drum for n in ins.notes]
                    if part=='bass' and not info.get('plugin_name'):all_valid=False;break
                    shift=-12 if part=='bass' and info['plugin_name']!='scarbee_jay_bass_slap_both.nkm' else 0
                    if any(not 23<=n.pitch+shift<=88 for n in notes):all_valid=False;break
                    targets[part].append(audio)
                elif audio in specs and (cls=='piano' or info.get('is_drum',False)):
                    if not any(c==cls for c,_ in backing):backing.append((cls,audio))
            if not all_valid or not all(targets.values()) or 'mix.flac' not in specs:
                rejected.append({'track':d['path'],'reason':'missing_target_or_out_of_vocabulary'});continue
            wanted=sorted({'mix.flac',*[a for values in targets.values() for a in values],*[a for _,a in backing]})
            cost=sum(specs[k]['size'] for k in wanted)
            if planned_bytes+cost>budget_gib*1024**3:raise RuntimeError('Source byte budget reached; do not silently change frozen counts')
            if shutil.disk_usage(root).free<cost+4*1024**3:raise RuntimeError('Insufficient source disk space')
            with ThreadPoolExecutor(max_workers=4) as pool:
                records+=list(pool.map(lambda k:download(specs[k],folder/k),wanted))
            # Decode all selected audio, rather than accepting a header alone.
            for key in wanted:
                with sf.SoundFile(folder/key) as f:
                    if f.frames<=0:raise ValueError('Empty audio')
                    for block in f.blocks(blocksize=44100*30,dtype='float32'):
                        import numpy as np
                        if not np.isfinite(block).all():raise ValueError('Non-finite decoded audio')
            planned_bytes+=cost;seen.add(uuid);accepted+=1
            selected.append({'track':Path(d['path']).name,'group':'slakh_'+uuid,'official_split':split,
                'split':'val' if split=='validation' else split,'folder':str(folder.resolve()),'records':records})
            print('NEW_SOURCE',split,accepted,'/',cap,Path(d['path']).name,round(planned_bytes/1024**3,3),'GiB',flush=True)
            # Incremental acquisition state survives loss of an Azure session.
            (report/'acquisition_progress.json').write_text(json.dumps({'selected':selected,'rejected':rejected},indent=2))
            if accepted==cap:break
        if accepted!=cap:raise RuntimeError('Not enough independent eligible sources')
    ledger={'mirror_repo':REPO,'revision':REV,'official_source':'https://zenodo.org/records/4599666',
        'verification':'Pinned mirror Git/LFS hashes + full audio decode + original per-stem MIDI. Official archive hash not verified.',
        'skipped_previous_uuids':sorted(used),'selected':selected,'rejected':rejected,'audio_bytes':planned_bytes,
        'counts':caps,'excluded_plugins':list(excluded_plugins),'sort_salt':sort_salt,'test_used_for_selection':False}
    ledger_path.write_text(json.dumps(ledger,indent=2));print('NEW_SOURCES_FROZEN',caps,flush=True)

if __name__=='__main__':main()
