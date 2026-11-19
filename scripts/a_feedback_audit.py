"""Verify every new feedback-cycle artifact recovered from Azure."""
from pathlib import Path
import argparse,json
from scripts.a_candidate_data import save,sha
FOLDERS=[f'{kind}/{name}' for name in ('keep_cycle_v1','keep_amp_v3','keep_agreement_v4','sound_keep_v6') for kind in ('data','reports','runs')]
LEDGER=Path('reports/keep_feedback_20261119/audit.json')
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['create','verify']);p.add_argument('--root',type=Path,default=Path('backups/azure'));a=p.parse_args()
    if a.command=='create':
        paths=sorted(p for folder in FOLDERS for p in Path(folder).rglob('*') if p.is_file() and p.suffix not in ('.tmp','.part','.pyc'))
        save(LEDGER,dict(files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]))
        print('FEEDBACK_AUDIT_CREATED',len(paths),flush=True)
    else:
        ledger=json.load(open(a.root/LEDGER))
        for r in ledger['files']:
            path=a.root/r['path']
            if not path.is_file() or path.stat().st_size!=r['bytes'] or sha(path)!=r['sha256']:raise ValueError('Backup mismatch: '+str(path))
        result=dict(files=len(ledger['files']),bytes=sum(r['bytes'] for r in ledger['files']),all_sha256_matched=True)
        save(a.root/'reports/keep_feedback_20261119/mac_audit.json',result);print('FEEDBACK_BACKUP_VERIFIED',result,flush=True)
if __name__=='__main__':main()
