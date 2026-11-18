"""Inspect the approved official GOAT archive without persisting access tokens.

The credential is read from a private file outside tracked source. Reports
retain public record identifiers, archive metadata and member integrity only.
"""
from pathlib import Path
import argparse,json,urllib.request,urllib.parse,zipfile,hashlib,os,zlib,struct
from concurrent.futures import ThreadPoolExecutor
from band2tab.techs import RangeReader

REPORT=Path('reports/goat');ROOT=Path('data/raw/goat')

def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','metadata','selected'])
    p.add_argument('--token-file',type=Path,default=Path('vendor/private/goat_access_token'))
    p.add_argument('--selection',type=Path);a=p.parse_args()
    token=a.token_file.read_text().strip()
    if not token or any(c.isspace() for c in token):raise ValueError('Invalid private token file')
    query=urllib.parse.urlencode({'token':token})
    req=urllib.request.Request('https://zenodo.org/api/records/15690894?'+query,headers={'User-Agent':'Band2Tab-university-research/1.0'})
    with urllib.request.urlopen(req,timeout=60) as response:record=json.load(response)
    files=[{'name':f['key'],'size':f['size'],'checksum':f['checksum']} for f in record.get('files',[])]
    if not files:raise ValueError('Approved access did not expose archive metadata')
    public={'record':15690894,'title':record['metadata']['title'],'license':record['metadata'].get('license'),
        'access_verified':True,'files':files,'personal_url_persisted':False}
    save(REPORT/'record.json',public)
    spec=next(f for f in files if f['name']=='GOAT.zip')
    url='https://zenodo.org/records/15690894/files/GOAT.zip?download=1&'+query
    with zipfile.ZipFile(RangeReader(url,spec['size'])) as z:
        entries=z.infolist()
        for i in entries:
            path=Path(i.filename)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe ZIP member')
        catalog=[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':i.CRC} for i in entries if not i.is_dir()]
        save(REPORT/'catalog.json',{'archive':spec,'members':catalog,'archive_md5_verified':False})
        from collections import Counter
        print('GOAT_ACCESS_VERIFIED',spec['size'],'bytes',len(catalog),'members',flush=True)
        print('GOAT_EXTENSIONS',dict(Counter(Path(i['name']).suffix for i in catalog)),flush=True)
        if a.command=='inspect':
            for i in catalog[:35]:print(i['name'],i['bytes'],flush=True)
            return
        if a.command=='metadata':
            selected=[i for i in entries if not i.is_dir() and i.file_size<=10*1024**2 and Path(i.filename).suffix.lower() not in ('.wav','.flac','.mp3','.aif','.aiff')]
        else:
            if a.selection is None:raise ValueError('--selection is required')
            wanted=set(json.loads(a.selection.read_text())['members']);selected=[i for i in entries if i.filename in wanted]
            if len(selected)!=len(wanted):raise ValueError('Requested ZIP members not found')
        ledger=[];old=ROOT/'selected_members.json'
        if old.exists():ledger=json.loads(old.read_text())
        known={i['member']:i for i in ledger}
        selected=[i for i in selected if not i.filename.startswith('__MACOSX/') and not Path(i.filename).name.startswith('._')]
        if sum(i.file_size for i in selected)>3*1024**3:raise ValueError('Selection exceeds 3 GiB extraction cap')
        def fetch_member(i):
            out=ROOT/i.filename;out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():
                data=out.read_bytes()
                if len(data)!=i.file_size or zlib.crc32(data)!=i.CRC:raise ValueError('Existing GOAT member corrupted')
            else:
                reader=RangeReader(url,spec['size']);header=reader._fetch(i.header_offset,i.header_offset+29)
                fields=struct.unpack('<4s5H3L2H',header);name_len,extra_len=fields[-2:]
                if fields[0]!=b'PK\x03\x04' or fields[2]&1 or fields[3] not in (0,8):raise ValueError('Unsupported member header')
                start=i.header_offset+30+name_len+extra_len
                payload=reader._fetch(start,start+i.compress_size-1) if i.compress_size else b''
                data=zlib.decompress(payload,-15) if fields[3]==8 else payload
                if len(data)!=i.file_size or zlib.crc32(data)!=i.CRC:raise ValueError('GOAT member integrity mismatch')
                temp=out.with_suffix(out.suffix+'.part');temp.write_bytes(data);temp.replace(out)
            return {'member':i.filename,'bytes':len(data),'zip_crc32_verified':True,
                'sha256':hashlib.sha256(data).hexdigest(),'archive_checksum_expected':spec['checksum'],'archive_md5_verified':False}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for item in pool.map(fetch_member,selected):
                known[item['member']]=item;save(old,list(known.values()));print('GOAT_MEMBER_VERIFIED',item['member'],item['bytes'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        # HTTP exceptions can contain the authenticated URL: never print them.
        raise SystemExit('GOAT acquisition failed: '+type(error).__name__+' status='+str(getattr(error,'code','not-http')))
