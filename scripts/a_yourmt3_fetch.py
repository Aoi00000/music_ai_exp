"""Fetch only official inference source and one pinned YourMT3+ checkpoint.

Skip examples, auth/cache files, apps, datasets and unrelated checkpoints.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,urllib.request,shutil

REV='5e66c1ea173a8186e0d20432b841d3180cc015b5'
REPO='mimbres/YourMT3'
CHECKPOINT='amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt'

def main():
    root=Path('vendor/yourmt3');report=Path('reports/yourmt3');report.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/api/spaces/{REPO}/tree/{REV}?recursive=true&limit=1000'
    with urllib.request.urlopen(url,timeout=60) as r:
        entries=json.load(r)
        if r.headers.get('Link'):raise ValueError('Unexpected pagination; do not accept a truncated source tree')
    selected=[r for r in entries if r['type']=='file' and
        (r['path'] in (CHECKPOINT,'model_helper.py','README.md') or
         (r['path'].endswith('.py') and r['path'].startswith(('amt/src/model/','amt/src/config/','amt/src/utils/')) and '/auth' not in r['path']))]
    def fetch(r):
        p=root/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
        def valid(path):
            if not path.exists() or path.stat().st_size!=r['size']:return False
            h=hashlib.sha256() if r.get('lfs') else hashlib.sha1()
            if not r.get('lfs'):h.update(b'blob '+str(r['size']).encode()+b'\0')
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024**2),b''):h.update(block)
            return h.hexdigest()==(r['lfs']['oid'] if r.get('lfs') else r['oid'])
        if not valid(p):
            temp=p.with_suffix(p.suffix+'.part')
            with urllib.request.urlopen(f'https://huggingface.co/spaces/{REPO}/resolve/{REV}/{r["path"]}',timeout=120) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,1024**2)
            if not valid(temp):raise ValueError('Hash mismatch '+r['path'])
            temp.replace(p)
        return {'path':str(p),'bytes':p.stat().st_size,'published_hash_verified':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,selected))
    (report/'acquisition.json').write_text(json.dumps({'repo':REPO,'revision':REV,'checkpoint':CHECKPOINT,'files':records},indent=2))
    print('YOURMT3_FETCHED',len(records),'files',sum(r['bytes'] for r in records),flush=True)

if __name__=='__main__':main()
