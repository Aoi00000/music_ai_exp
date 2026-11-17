#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${BAND2TAB_PYTHON:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [ -x $HOME/.pyenv/versions/3.12.2/bin/python3 ]; then
    PYTHON_BIN=$HOME/.pyenv/versions/3.12.2/bin/python3
  else
    echo 'Python 3.12 の場所を BAND2TAB_PYTHON に指定してください。'
    exit 1
  fi
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
ARCH=$("$PYTHON_BIN" -c 'import platform; print(platform.machine())')
TORCH_VERSION=2.9.1
if [ "$ARCH" = x86_64 ]; then TORCH_VERSION=2.2.2; fi
if [ ! -x .venv/bin/python ]; then "$PYTHON_BIN" -m venv .venv; fi
.venv/bin/python -m pip install "torch==$TORCH_VERSION" -r requirements.txt uv
if [ ! -x .venv-mu/bin/python ]; then
  .venv/bin/uv venv --python .venv/bin/python .venv-mu
fi
.venv/bin/uv pip install --python .venv-mu/bin/python \
  "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" muscriptor==0.4.0
mkdir -p models/huggingface models/torch reports
chmod +x Band2Tab.command
export HF_HOME="$PWD/models/huggingface"
export TORCH_HOME="$PWD/models/torch"
.venv/bin/python -m chordchart.setup bootstrap
.venv/bin/python -m chordchart.setup doctor
.venv/bin/python -m unittest discover -s tests -v
if [ -f models/muscriptor-medium/model.safetensors ]; then
  printf '\n準備完了。取得済みのローカルMuScriptor mediumを使用します。\n'
else
  printf '\n準備完了。モデルの利用条件に同意した後、次を実行します。\n'
  printf 'export HF_HOME="$PWD/models/huggingface"\n.venv-mu/bin/hf auth login\n'
fi
printf '普段の起動は Band2Tab.command をダブルクリックします。\n'
