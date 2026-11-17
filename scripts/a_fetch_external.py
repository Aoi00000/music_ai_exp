from pathlib import Path
import zipfile
from band2tab.data import fetch_checked
p=Path('data/raw/idmt_bass_tracks/IDMT-SMT-BASS-SINGLE-TRACKS.zip');p.parent.mkdir(parents=True,exist_ok=True)
fetch_checked('https://zenodo.org/records/7544099/files/IDMT-SMT-BASS-SINGLE-TRACKS.zip?download=1',p,'c85540c664e8b13badd86880bacf11fa')
with zipfile.ZipFile(p) as z:
    for f in z.infolist():
        member=Path(f.filename)
        if member.is_absolute() or '..' in member.parts:raise ValueError('Unsafe archive member')
        if not f.filename.startswith('__MACOSX'):z.extract(f,p.parent)
    print(z.namelist()[:50],flush=True)
# Only the five musical excerpts from dataset3 are held out; none is trained on.
import json,hashlib
from band2tab.techs import RangeReader
import urllib.request
index_path=Path('reports/idmt_guitar_file_index.json')
if not index_path.exists():
    record=json.load(urllib.request.urlopen('https://zenodo.org/api/records/7544110',timeout=30))
    spec=next(f for f in record['files'] if f['key']=='IDMT-SMT-GUITAR_V2.zip')
    index_path.parent.mkdir(exist_ok=True);index_path.write_text(json.dumps({'file':spec},indent=2))
index=json.loads(index_path.read_text());spec=index['file'];dest=Path('data/raw/idmt_guitar')
ledger=[]
with zipfile.ZipFile(RangeReader(spec['links']['self'],spec['size'])) as z:
    for f in z.infolist():
        if '/dataset3/' not in f.filename or Path(f.filename).suffix.lower() not in ('.wav','.xml'):continue
        if '..' in Path(f.filename).parts:raise ValueError('Unsafe archive member')
        data=z.read(f);p=dest/f.filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        ledger.append({'member':f.filename,'crc32_verified':True,'sha256':hashlib.sha256(data).hexdigest(),'archive_md5_verified':False});print('Verified external guitar',f.filename,len(data),flush=True)
(dest/'selected_members.json').write_text(json.dumps(ledger,indent=2))
