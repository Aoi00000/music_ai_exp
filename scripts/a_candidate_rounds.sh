#!/bin/bash
# Validation-only learning rounds. No test inference and no model adoption.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4
PY="$HOME/venv/bin/python"
"$PY" -u - <<'PY'
from pathlib import Path
import json,time
rows=[r for r in json.load(open('data/candidate_v2/manifest.json')) if r['split'] in ('train','val')]
deadline=time.monotonic()+1200
while True:
    missing=[r['id'] for r in rows if not all((Path('reports/candidate_v2/midi')/r['id']/(k+'.json')).is_file() for k in ('mix','guitar','bass'))]
    if not missing:break
    if time.monotonic()>deadline:raise TimeoutError('Candidate inference incomplete: '+str(missing[:5]))
    print('WAITING_FOR_CANDIDATES',len(missing),flush=True);time.sleep(10)
print('ALL_DEVELOPMENT_MIDI_READY',len(rows),flush=True)
PY
"$PY" -u scripts/a_candidate_data.py cache
run() {
  local name="$1"
  shift
  if [ ! -f "runs/candidate_v2/$name/completed.json" ]; then
    "$PY" -u scripts/a_candidate_train.py train --name "$name" --steps 3000 "$@"
  fi
}
run expanded_mlp --data expanded
run expanded_context --data expanded --context
run expanded_unique_context --data expanded --context --unique-targets
run balanced_unique_context --data balanced --context --unique-targets
run balanced_unique_context_neg2 --data balanced --context --unique-targets --negative-weight 2
run hard_unique_context_neg2 --data hard --context --unique-targets --negative-weight 2
