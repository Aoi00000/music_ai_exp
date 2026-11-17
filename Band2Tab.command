#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python || ! -x .venv-mu/bin/muscriptor ]]; then
  print '初回準備が必要です。このフォルダで bash setup_mac.sh を実行してください。'
  read '?Enterで閉じます。'
  exit 1
fi
export HF_HOME="$PWD/models/huggingface"
export TORCH_HOME="$PWD/models/torch"
exec .venv/bin/python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless false
