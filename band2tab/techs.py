"""Official Guitar-TECHS DI/MIDI subset; ZIP members retain their CRC and SHA256."""
from pathlib import Path
import io,json,urllib.request,time,zipfile,hashlib,argparse
from concurrent.futures import ThreadPoolExecutor

class RangeReader(io.BufferedIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.position=0
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset+(0 if whence==0 else self.position if whence==1 else self.size)
        if self.position<0:raise ValueError('Negative seek')
        return self.position
    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size)
        if n<=0:return b''
        start=self.position;end=start+n-1
        if n>=16*1024**2:
            chunk=(n+3)//4;bounds=[(a,min(end,a+chunk-1)) for a in range(start,end+1,chunk)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                data=b''.join(pool.map(lambda pair:self._fetch(*pair),bounds))
        else:data=self._fetch(start,end)
        if len(data)!=n:raise ValueError('Truncated ZIP range')
        self.position+=n;return data
    def _fetch(self,start,end):
        for attempt in range(3):
            try:
                req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('Server range response mismatch')
                    data=r.read()
                if len(data)!=end-start+1:raise ValueError('Truncated ZIP range')
                return data
            except Exception:
                if attempt==2:raise
                time.sleep(2)

def fetch(record='reports/guitar_techs_record.json',names=('P1_chords.zip',),root='data/raw/guitar_techs'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);record=Path(record)
    if not record.exists():
        full=json.load(urllib.request.urlopen('https://zenodo.org/api/records/14963133',timeout=30))
        metadata={'record':14963133,'title':full['metadata']['title'],'license':full['metadata'].get('license'),'files':[{'name':f['key'],'size':f['size'],'checksum':f['checksum'],'url':f['links']['self']} for f in full['files']]}
        record.parent.mkdir(parents=True,exist_ok=True);record.write_text(json.dumps(metadata,indent=2))
    metadata=json.loads(record.read_text());ledger=[]
    for spec in metadata['files']:
        if spec['name'] not in names:continue
        reader=RangeReader(spec['url'],spec['size'])
        with zipfile.ZipFile(reader) as z:
            for info in z.infolist():
                p=Path(info.filename)
                if '..' in p.parts or p.is_absolute():raise ValueError('Unsafe ZIP member')
                if p.name.startswith('._') or '__MACOSX' in p.parts:continue
                if not (('/audio/directinput/' in info.filename and p.suffix=='.wav') or ('/midi/' in info.filename and p.suffix=='.mid')):continue
                dest=root/p;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    import zlib
                    data=dest.read_bytes()
                    if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise ValueError('Existing member corrupted')
                else:
                    data=z.read(info)  # zipfile checks the published member's CRC32.
                    tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(dest)
                ledger.append({'archive':spec['name'],'archive_md5_expected':spec['checksum'],'archive_md5_verified':False,'member':info.filename,'bytes':len(data),'zip_crc32_verified':True,'sha256':hashlib.sha256(data).hexdigest()})
                print('Verified ZIP member',info.filename,len(data),flush=True)
    ledger_path=root/'selected_members.json'
    old=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    combined={(row['archive'],row['member']):row for row in old+ledger}
    ledger_path.write_text(json.dumps(list(combined.values()),indent=2));return ledger

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args();fetch(names=a.names)
