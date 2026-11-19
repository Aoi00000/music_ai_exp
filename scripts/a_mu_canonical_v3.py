"""Matched canonical-program control for native v2: fixed GradScaler512."""
from pathlib import Path
import argparse,copy,json
from scripts.a_candidate_data import save,sha
DATA=Path('data/mu_canonical_v3');REPORT=Path('reports/mu_canonical_v3');RUN=Path('runs/mu_canonical_v3')

def prepare():
    if (DATA/'protocol.json').exists():raise ValueError('Control protocol frozen')
    rows=json.load(open('data/mu_adapt_v1/manifest.json'));source=json.load(open('data/mu_native_v2/manifest.json'));assert [r['id'] for r in rows]==[r['id'] for r in source]
    for r in rows:
        dest=DATA/'cache'/(r['id']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(r['cache']).read_bytes());r['cache']=str(dest);r['cache_sha256']=sha(dest)
    save(DATA/'manifest.json',rows)
    for file in ['monitor.json','review_manifest.json']:save(DATA/file,json.load(open(Path('data/mu_native_v2')/file)))
    p=copy.deepcopy(json.load(open('data/mu_native_v2/protocol.json')));p.update(hypothesis='Final audit found initial canonical pilot used GradScaler default65536, whereas native v2 used512. Re-run the canonical controls with512, matching native v2; preserve initial pilot as non-isolated history.',
      precision='Frozen base float32; autocast float16, float32 adapter gradients; GradScaler init512; abort on nonfinite updates',
      teacher='Canonical program26/33 as initial pilot; other labels, audio, draws, code and optimizer identical to native v2.',
      manifest_sha256=sha(DATA/'manifest.json'),monitor_sha256=sha(DATA/'monitor.json'),review_manifest_sha256=sha(DATA/'review_manifest.json'),
      native_protocol_sha256=sha('data/mu_native_v2/protocol.json'),implementation_sha256={f:sha(f) for f in ['band2tab/mu_adapter.py','scripts/a_mu_adapt.py','scripts/a_mu_canonical_v3.py','scripts/a_mu_canonical_eval.py','band2tab/augment.py']})
    save(DATA/'protocol.json',p);print('MATCHED_CANONICAL_CONTROL_REGISTERED',len(rows),flush=True)

def train():
    import scripts.a_mu_adapt as runner
    runner.DATA=DATA;runner.REPORT=REPORT;runner.RUN=RUN;runner.train()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);a=p.parse_args();globals()[a.command]()
