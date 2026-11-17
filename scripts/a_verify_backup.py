"""Mac: verify all remote checkpoints, then rebase cached manifests locally."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',default='aiexp');p.add_argument('--backup',default='backups/azure');a=p.parse_args()
root=Path(a.backup).resolve()
remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',a.host,"cd ~/music_ai_exp && find runs -type f -name '*.pt' -exec sha256sum {} \\;"],capture_output=True,text=True,check=True)
checked=[]
for row in remote.stdout.splitlines():
    expected,path=row.split(maxsplit=1);local=root/path
    if not local.exists() or hashlib.sha256(local.read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Backup mismatch: {path}')
    checked.append({'file':path,'sha256':expected})
if not checked:raise RuntimeError('No checkpoints verified')
for manifest in ('manifest_clean','manifest_robust'):
    items=json.loads((root/'data'/(manifest+'.json')).read_text())
    for it in items:
        for key in ('audio','cache'):
            relative=it[key].split('/music_ai_exp/data/',1)[-1]
            local=root/'data'/relative
            if not local.is_file():raise FileNotFoundError(local)
            it[key]=str(local)
    (root/'data'/(manifest+'_mac.json')).write_text(json.dumps(items,indent=2))
(root/'reports'/'backup_verification.json').write_text(json.dumps({'all_checkpoints_sha256_match':True,'files':checked,'portable_manifests_created':True},indent=2))
print('Verified',len(checked),'checkpoints; portable Mac manifests ready')
