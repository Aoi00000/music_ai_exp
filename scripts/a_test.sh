#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python - <<'PY'
import json,subprocess,sys,hashlib
from pathlib import Path
p=json.loads(Path('reports/test_protocol.json').read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
if sha(p['manifest'])!=p['manifest_sha256']:raise RuntimeError('Frozen manifest changed')
for name,run in p['runs'].items():
    if sha(run['checkpoint'])!=run['checkpoint_sha256']:raise RuntimeError('Frozen checkpoint changed')
    th=run['thresholds']
    subprocess.run([sys.executable,'-m','band2tab','evaluate','--checkpoint',run['checkpoint'],'--manifest',p['manifest'],'--split','test','--threshold',str(th['threshold']),'--frame-threshold',str(th['frame_threshold']),'--offset-threshold',str(th['offset_threshold']),'--device','cuda','--out',f'reports/{name}_test.json'],check=True)
PY
printf 'FIXED_TESTS_COMPLETED\n'
