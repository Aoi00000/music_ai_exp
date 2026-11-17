#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u -m band2tab train --manifest data/manifest_clean.json --out runs/a0 --epochs 5 --steps 100 --batch 4 --device cuda
python -u -m band2tab train --manifest data/manifest_robust.json --out runs/a1 --epochs 5 --steps 100 --batch 4 --device cuda
python -u -m band2tab evaluate --checkpoint runs/a0/best.pt --manifest data/manifest_clean.json --split val --tune --device cuda --out reports/a0_val.json
python -u -m band2tab evaluate --checkpoint runs/a1/best.pt --manifest data/manifest_clean.json --split val --tune --device cuda --out reports/a1_val.json
printf 'TRAINING_AND_VALIDATION_COMPLETED\n'
