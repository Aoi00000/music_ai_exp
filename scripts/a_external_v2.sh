#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venv/bin/activate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/a_freeze_v2.py
python - <<'PY'
from pathlib import Path
import json,subprocess,sys,hashlib
p=json.loads(Path('reports/v2_external_protocol.json').read_text())
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
if sha(p['external_manifest'])!=p['external_manifest_sha256']:raise ValueError('External reference changed')
for name,r in p['runs'].items():
 if sha(r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('Frozen weights changed')
 t=r['thresholds']
 subprocess.run([sys.executable,'-m','band2tab','evaluate','--checkpoint',r['checkpoint'],'--manifest',p['external_manifest'],'--split','test','--threshold',str(t['threshold']),'--frame-threshold',str(t['frame_threshold']),'--offset-threshold',str(t['offset_threshold']),'--device','cuda','--out',f'reports/{name}_external.json'],check=True)
PY
python -m band2tab.report baseline --manifest data/manifest_external.json --predictions reports/basic_pitch_external_predictions --split test --out reports/basic_pitch_external.json
python -m band2tab.report baseline --manifest data/manifest_external.json --predictions reports/muscriptor_external_predictions --split test --out reports/muscriptor_external.json
python -m band2tab.report summarize reports/a6_sounding_external.json reports/a7_sounding_techs_external.json reports/a8_sounding_full_external.json reports/basic_pitch_external.json reports/muscriptor_external.json --out reports/presentation_v2_external
printf 'EXTERNAL_V2_COMPARISON_COMPLETED\n'
