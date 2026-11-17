#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venv/bin/activate"
export BAND2TAB_CACHE_ITEMS=512 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m band2tab.correct
python -u -m band2tab.a_experiment audit --manifest data/manifest_v2_techs_full.json --out reports/v2_full_data_audit.json
python -u -m band2tab train --manifest data/manifest_v2_techs_full.json --out runs/a8_sounding_full \
  --epochs 30 --steps 100 --batch 4 --lr .0003 --pitch-augment --device cuda --init runs/a2_long/selected_10000.pt
python -u -m band2tab evaluate --checkpoint runs/a8_sounding_full/best.pt --manifest data/manifest_v2_clean.json \
  --split val --tune --device cuda --out reports/a8_sounding_full_val.json
printf 'FULL_SOUNDING_EXPERIMENT_COMPLETED\n'
