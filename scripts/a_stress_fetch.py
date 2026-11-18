"""Small official MUSDB 7-second vocal/drum subset for controlled interference.

These stems have no note labels. They are only non-target interference, never
invented guitar/bass ground truth. Song-level splits are fixed before inference.
"""
from pathlib import Path
import argparse,json,urllib.request,zipfile,hashlib,zlib
import yaml
from band2tab.techs import RangeReader
from scripts.a_candidate_data import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('backups/azure'));a=p.parse_args()
    report=a.root/'reports/band_stress_v1';root=a.root/'data/raw/musdb_7s'
    record=json.load(urllib.request.urlopen('https://zenodo.org/api/records/1256064',timeout=60))
    spec=next(f for f in record['files'] if f['key']=='MUSDB18-7-WAV.zip')
    setup_url='https://raw.githubusercontent.com/sigsep/sigsep-mus-db/v0.4.0/musdb/configs/mus.yaml'
    setup_bytes=urllib.request.urlopen(setup_url,timeout=60).read();valid=set(yaml.safe_load(setup_bytes)['validation_tracks'])
    with zipfile.ZipFile(RangeReader(spec['links']['self'],spec['size'])) as z:
        entries=z.infolist();by={}
        for i in entries:
            q=Path(i.filename)
            if q.is_absolute() or '..' in q.parts:raise ValueError('Unsafe background archive member')
            if len(q.parts)==3 and q.parts[-1] in ('vocals.wav','drums.wav') and q.parts[0] in ('train','test'):
                by.setdefault((q.parts[0],q.parts[1]),[]).append(i)
        pools={'train':[],'val':[],'test':[]}
        for (split,name),items in by.items():
            if len(items)!=2:continue
            target='test' if split=='test' else 'val' if name in valid else 'train'
            pools[target].append({'group':split+'/'+name,'split':target,'members':[i.filename for i in items]})
        selected=[]
        for split,count in (('train',6),('val',2),('test',2)):
            pool=sorted(pools[split],key=lambda r:hashlib.sha256(('band2tab-stress-v1-'+r['group']).encode()).hexdigest())
            if len(pool)<count:raise ValueError('Missing official background split')
            selected+=pool[:count]
        policy={'record':1256064,'archive':spec['key'],'archive_bytes':spec['size'],'archive_checksum_expected':spec['checksum'],
            'archive_md5_verified':False,'declared_record_license':record['metadata'].get('license'),
            'official_split_setup':setup_url,'setup_sha256':hashlib.sha256(setup_bytes).hexdigest(),'selected':selected,
            'selection':'Before inference: official train/validation/test, hash-pick 6/2/2 songs; vocals and drums only',
            'target_note_labels_provided':False,'test_used_for_selection':False}
        policy_path=report/'acquisition.json'
        if policy_path.exists() and json.loads(policy_path.read_text())!=policy:raise ValueError('Background selection changed')
        save(policy_path,policy);wanted={m for r in selected for m in r['members']}
        picked=[i for i in entries if i.filename in wanted]
        if sum(i.file_size for i in picked)>50*1024**2:raise ValueError('Background extraction cap exceeded')
        ledger=[]
        for i in picked:
            out=root/i.filename;out.parent.mkdir(parents=True,exist_ok=True)
            data=out.read_bytes() if out.exists() else z.read(i)
            if len(data)!=i.file_size or zlib.crc32(data)!=i.CRC:raise ValueError('Background member corrupted')
            if not out.exists():out.write_bytes(data)
            ledger.append({'member':i.filename,'bytes':len(data),'zip_crc32_verified':True,'sha256':hashlib.sha256(data).hexdigest()})
            save(report/'members.json',ledger);print('STRESS_STEM_VERIFIED',i.filename,len(data),flush=True)
        print('STRESS_STEMS_READY',len(ledger),sum(r['bytes'] for r in ledger),flush=True)

if __name__=='__main__':main()
