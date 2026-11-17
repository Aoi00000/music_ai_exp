"""Create/verify checksums for this experiment, and rebase its nested Mac paths."""
from pathlib import Path
import argparse,hashlib,json

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def create():
    folders=['data/separation_v1','reports/separation_v1','runs/separation_v1','data/raw/guitar_techs/P3_music/audio/micamp']
    paths=sorted({p for folder in folders for p in Path(folder).rglob('*') if p.is_file() and p.suffix not in ('.tmp','.pyc')})
    paths += list(Path('models').glob('muscriptor_separation_*'))
    ledger=Path('reports/separation_v1/audit_files.json')
    records=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths if p!=ledger]
    ledger.write_text(json.dumps({'files':records,'root':str(Path.cwd())},indent=2));print('CHECKSUMS_CREATED',len(records))

def verify(root):
    root=Path(root).resolve();ledger=json.loads((root/'reports/separation_v1/audit_files.json').read_text())
    for record in ledger['files']:
        p=root/record['path']
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('Backup mismatch: '+str(p))
    remote=ledger['root']
    def rebase(obj):
        if isinstance(obj,dict):return {k:rebase(v) for k,v in obj.items()}
        if isinstance(obj,list):return [rebase(v) for v in obj]
        if isinstance(obj,str) and obj.startswith(remote+'/'):return str(root/obj.removeprefix(remote+'/'))
        return obj
    for name in ('manifest','manifest_amp_condition'):
        path=root/'data/separation_v1'/(name+'.json')
        if path.exists():path.with_name(name+'_mac.json').write_text(json.dumps(rebase(json.loads(path.read_text())),indent=2))
    report={'verified_files':len(ledger['files']),'all_sha256_matched':True,'local_root':str(root)}
    (root/'reports/separation_v1/mac_audit.json').write_text(json.dumps(report,indent=2));print('BACKUP_VERIFIED',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['create','verify']);p.add_argument('--root',default='backups/azure');a=p.parse_args()
    if a.command=='create':create()
    else:verify(a.root)
