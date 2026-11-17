"""Official FiloBass distribution: checksum-verified archive and safe extraction."""
from pathlib import Path
import json,urllib.request,zipfile
from band2tab.data import fetch_checked

record=json.load(urllib.request.urlopen('https://zenodo.org/api/records/10069709',timeout=30))
Path('reports/filobass_record.json').write_text(json.dumps(record,indent=2))
spec=next(f for f in record['files'] if f['key']=='FiloBass_v1.0.0.zip')
root=Path('data/raw/filobass');root.mkdir(parents=True,exist_ok=True);archive=root/spec['key']
fetch_checked(spec['links']['self'],archive,spec['checksum'].split(':')[-1])
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        p=Path(info.filename)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe FiloBass ZIP path')
        if '__MACOSX' in p.parts or p.name.startswith('._'):continue
        z.extract(info,root)
print('FILOBASS_VERIFIED',len(list(root.rglob('*'))),'files and directories',flush=True)
