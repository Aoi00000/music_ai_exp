"""Verify candidate/YourMT3 experiments and create portable Mac manifests."""
from pathlib import Path
import argparse,json,hashlib

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

DEST=Path('reports/candidate_v2/audit_files.json')
FOLDERS=('data/candidate_v2','data/raw/slakh_candidate_v2','data/yourmt3_lora_extra','data/goat_v1','data/goat_v2','data/raw/goat','data/band_stress_v1','data/band_training_v1','data/raw/musdb_7s',
    'reports/candidate_v2','reports/yourmt3','reports/yourmt3_lora','reports/yourmt3_lora_expanded',
    'reports/yourmt3_lora_goat','reports/yourmt3_lora_goat_amp_v2','reports/yourmt3_lora_goat_band_v1','reports/yourmt3_final','reports/goat','reports/band_stress_v1','reports/band_training_v1',
    'runs/candidate_v2','runs/yourmt3_lora','vendor/yourmt3')

def create():
    paths={p for d in FOLDERS for p in Path(d).rglob('*') if p.is_file() and
        p.suffix not in ('.tmp','.part','.pyc') and '__pycache__' not in p.parts}
    paths.update(p for p in Path('models').glob('*') if p.is_file() and p.name.startswith(('muscriptor_candidate','yourmt3_')))
    manifests=[Path('data/candidate_v2/manifest.json'),Path('data/yourmt3_lora_extra/manifest.json'),Path('data/goat_v1/manifest.json'),Path('data/goat_v2/manifest.json'),Path('data/band_stress_v1/manifest.json'),Path('data/band_training_v1/manifest.json')]
    for manifest in manifests:
        if not manifest.exists():continue
        for r in json.loads(manifest.read_text()):
            for key in ('audio','reuse_cache'):
                if r.get(key) and Path(r[key]).is_file():paths.add(Path(r[key]).resolve().relative_to(Path.cwd()))
            for path in r.get('inputs',{}).values():
                if Path(path).is_file():paths.add(Path(path).resolve().relative_to(Path.cwd()))
    paths.discard(DEST)
    files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)]
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(json.dumps({'root':str(Path.cwd()),'files':files,
        'scope':'Candidate and YourMT3 artifacts, raw new Slakh, selected source and derived data. Earlier raw-data audits remain separate.'},indent=2))
    print('CANDIDATE_AUDIT_CREATED',len(files),sum(r['bytes'] for r in files),flush=True)

def verify(root):
    root=Path(root).resolve();ledger=json.loads((root/DEST).read_text())
    for record in ledger['files']:
        p=root/record['path']
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
            raise ValueError('Backup mismatch '+str(p))
    prefix=ledger['root']+'/'
    def rebase(value):
        if isinstance(value,dict):return {k:rebase(v) for k,v in value.items()}
        if isinstance(value,list):return [rebase(v) for v in value]
        if isinstance(value,str) and value.startswith(prefix):return str(root/value.removeprefix(prefix))
        return value
    for folder in ('candidate_v2','yourmt3_lora_extra','goat_v1','goat_v2','band_stress_v1','band_training_v1'):
        p=root/'data'/folder/'manifest.json'
        if p.exists():p.with_name('manifest_mac.json').write_text(json.dumps(rebase(json.loads(p.read_text())),indent=2))
    result={'all_sha256_matched':True,'verified_files':len(ledger['files']),
        'verified_bytes':sum(r['bytes'] for r in ledger['files']),'scope':ledger['scope']}
    (root/'reports/candidate_v2/mac_audit.json').write_text(json.dumps(result,indent=2))
    print('CANDIDATE_MAC_VERIFIED',result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['create','verify']);p.add_argument('--root',default='backups/azure');a=p.parse_args()
    if a.command=='create':create()
    else:verify(a.root)
