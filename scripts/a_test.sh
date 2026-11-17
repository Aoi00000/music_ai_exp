#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python - <<'PY'
import json,subprocess,sys
for run in ('a0','a1'):
    result=json.load(open(f'reports/{run}_val.json'))
    if result['macro_note_f1']<=.01:
        raise RuntimeError(f'{run}: validation note F1 is too low; diagnose before looking at test')
    th=result['threshold']
    subprocess.run([sys.executable,'-m','band2tab','evaluate','--checkpoint',f'runs/{run}/best.pt','--manifest','data/manifest_clean.json','--split','test','--threshold',str(th),'--frame-threshold',str(result['frame_threshold']),'--offset-threshold',str(result['offset_threshold']),'--device','cuda','--out',f'reports/{run}_test.json'],check=True)
PY
printf 'FIXED_TESTS_COMPLETED\n'
