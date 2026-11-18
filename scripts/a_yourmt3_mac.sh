#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/vendor/yourmt3/amt/src:$PWD/vendor/yourmt3:$PWD"
export HF_HOME="$PWD/vendor/hf_cache_mac"
export MPLCONFIGDIR="$PWD/vendor/mpl_cache_mac"
export XDG_CACHE_HOME="$PWD/vendor/cache_mac"
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
exec .venv-yourmt3/bin/python scripts/a_yourmt3_infer.py "$@"
