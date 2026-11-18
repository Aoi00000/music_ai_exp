"""Audit new source audio, derived data, selected weights and portable Mac paths."""
from pathlib import Path
import json,hashlib,argparse
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def create():
    folders=('data/temporal_v1','data/raw/slakh_temporal','reports/temporal_v1','runs/temporal_v1','models/torch')
    paths=sorted({p for folder in folders for p in Path(folder).rglob('*') if p.is_file() and p.suffix not in ('.tmp','.part','.pyc')})
    paths+=list(Path('models').glob('muscriptor_temporal*'));dest=Path('reports/temporal_v1/audit_files.json')
    records=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths if p!=dest]
    dest.write_text(json.dumps({'root':str(Path.cwd()),'files':records},indent=2));print('AUDITED',len(records))
def verify(root):
    root=Path(root).resolve();ledger=json.loads((root/'reports/temporal_v1/audit_files.json').read_text())
    for r in ledger['files']:
        p=root/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Backup mismatch '+str(p))
    prefix=ledger['root']+'/'
    def rebase(obj):
        if isinstance(obj,dict):return {k:rebase(v) for k,v in obj.items()}
        if isinstance(obj,list):return [rebase(v) for v in obj]
        if isinstance(obj,str) and obj.startswith(prefix):return str(root/obj.removeprefix(prefix))
        return obj
    p=root/'data/temporal_v1/manifest.json';p.with_name('manifest_mac.json').write_text(json.dumps(rebase(json.loads(p.read_text())),indent=2))
    report={'verified_files':len(ledger['files']),'all_sha256_matched':True}
    (root/'reports/temporal_v1/mac_audit.json').write_text(json.dumps(report));print('MAC_VERIFIED',report)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['create','verify']);p.add_argument('--root',default='backups/azure');a=p.parse_args()
    if a.command=='create':create()
    else:verify(a.root)
