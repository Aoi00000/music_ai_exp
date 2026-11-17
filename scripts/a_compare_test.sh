#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venv/bin/activate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/a_freeze.py
.venv-bp/bin/python -u scripts/a_baseline_predict.py --manifest data/manifest_clean.json \
  --split test --out reports/basic_pitch_test_predictions > logs/basic_pitch_test.log 2>&1 &
baseline_pid=$!
trap 'kill "$baseline_pid" 2>/dev/null || true' EXIT
bash scripts/a_test.sh
wait "$baseline_pid"
trap - EXIT
python -m band2tab.report baseline --manifest data/manifest_clean.json \
  --predictions reports/basic_pitch_test_predictions --split test --out reports/basic_pitch_test.json
python -m band2tab.report summarize reports/a0_test.json reports/a1_test.json reports/basic_pitch_test.json --out reports/presentation_test
printf 'COMPARISON_COMPLETED\n'
