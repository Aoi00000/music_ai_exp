"""Official MUSDB real-bass stems for teacher retention, not invented labels."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile,zlib,struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np,soundfile as sf,yaml
from band2tab.techs import RangeReader
from band2tab.separation import write_input
from scripts.a_candidate_data import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('backups/azure'));a=p.parse_args();root=a.root.resolve()
    folder=root/'data/bass_retention_v1';report=root/'reports/bass_retention_v1'
    record=json.load(urllib.request.urlopen('https://zenodo.org/api/records/1256064',timeout=60))
    spec=next(f for f in record['files'] if f['key']=='MUSDB18-7-WAV.zip')
    setup='https://raw.githubusercontent.com/sigsep/sigsep-mus-db/v0.4.0/musdb/configs/mus.yaml'
    setup_bytes=urllib.request.urlopen(setup,timeout=60).read()
    if hashlib.sha256(setup_bytes).hexdigest()!='8fda708c77ab9d207467b5484277dfda981a0e5cdb6deacca9cc59fe4f07cd5e':raise ValueError('Official split config changed')
    validation=set(yaml.safe_load(setup_bytes)['validation_tracks']);rows=[];ledger=[]
    reader=RangeReader(spec['links']['self'],spec['size'])
    with zipfile.ZipFile(reader) as z:
        pools={'train':[],'val':[]}
        for info in z.infolist():
            path=Path(info.filename)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe ZIP member')
            if len(path.parts)==3 and path.parts[0]=='train' and path.parts[-1]=='bass.wav':
                pools['val' if path.parts[1] in validation else 'train'].append(info)
        selected=[]
        for split,count in (('train',32),('val',8)):
            chosen=sorted(pools[split],key=lambda x:hashlib.sha256(('band2tab-bass-retention-v1-'+x.filename).encode()).hexdigest())[:count]
            if len(chosen)!=count:raise ValueError('Missing bass split')
            selected.extend((split,i) for i in chosen)
        policy={'record':1256064,'archive':spec['key'],'archive_bytes':spec['size'],'archive_md5_verified':False,
            'archive_checksum_expected':spec['checksum'],'declared_record_license':record['metadata'].get('license'),
            'selection':[{'split':s,'member':i.filename} for s,i in selected],
            'official_split_sha256':hashlib.sha256(setup_bytes).hexdigest(),'test_accessed':False,
            'labels_available':False,'purpose':'Unlabelled real electric-bass retention; teacher predictions are not ground truth'}
        path=report/'acquisition.json'
        if path.exists() and json.load(open(path))!=policy:raise ValueError('Frozen retention selection changed')
        save(path,policy)
        if sum(i.file_size for s,i in selected)>80*1024**2:raise ValueError('Subset cap exceeded')
        def fetch_member(item):
            split,info=item
            raw=root/'data/raw/musdb_retention_7s'/info.filename;raw.parent.mkdir(parents=True,exist_ok=True)
            if raw.exists():data=raw.read_bytes()
            else:
                header=reader._fetch(info.header_offset,info.header_offset+29);fields=struct.unpack('<4s5H3L2H',header)
                if fields[0]!=b'PK\x03\x04' or fields[2]&1 or fields[3] not in (0,8):raise ValueError('Unsupported bass ZIP entry')
                start=info.header_offset+30+fields[-2]+fields[-1]
                payload=reader._fetch(start,start+info.compress_size-1)
                data=zlib.decompress(payload,-15) if fields[3]==8 else payload
            if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise ValueError('Bass member checksum failed')
            if not raw.exists():raw.write_bytes(data)
            ident='musdb_bass_'+hashlib.sha256(info.filename.encode()).hexdigest()[:12];audio=folder/'audio'/(ident+'.wav')
            y,sr=sf.read(raw,dtype='float32',always_2d=True);write_input(audio,y,sr)
            row={'id':ident,'group':'/'.join(Path(info.filename).parts[:2]),'split':split,'kind':'bass_stem',
                'audio':str(audio),'duration':len(y)/sr,'audio_sha256':sha(audio),'labels_available':False,
                'ac_rms':float(np.std(y.mean(1))),'raw_source':str(raw)}
            member={'member':info.filename,'sha256':sha(raw),'bytes':len(data),'zip_crc32_verified':True}
            return row,member
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index,(row,member) in enumerate(pool.map(fetch_member,selected)):
                rows.append(row);ledger.append(member);save(report/'members.json',ledger)
                print('RETENTION_BASS_READY',index+1,row['split'],row['id'],flush=True)
    save(folder/'manifest.json',rows)
    print('RETENTION_DATA_READY',len(rows),flush=True)


if __name__=='__main__':main()
