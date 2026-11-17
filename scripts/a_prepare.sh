#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p reports logs runs
python -u -m band2tab fetch guitarset
python -u -m band2tab fetch baby
python -u -m band2tab prepare
python -u -m band2tab.a_experiment audit --out reports/audit_clean.json
python -u -m band2tab remix --limit 30 --snr 0
python -u -m band2tab.a_experiment audit --manifest data/manifest_robust.json --out reports/audit_robust.json
python -u -m band2tab.a_experiment smoke --manifest data/manifest_clean.json --out reports/learnability_check.json
printf 'PREPARATION_AND_LEARNABILITY_CHECK_PASSED\n'
