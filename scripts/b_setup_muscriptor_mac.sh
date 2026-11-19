#!/bin/bash
# Prepare only B's transcription runtime. Never install into the editor's .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'この手順はApple SiliconのmacOS用です。ここで停止します。' >&2
  exit 1
fi
EDITOR_PYTHON="$PWD/.venv/bin/python"
MU_PYTHON="$PWD/.venv-mu-b/bin/python"
if [[ ! -x "$EDITOR_PYTHON" ]]; then
  echo '編集画面用の .venv/bin/python がありません。既存環境を確認してください。' >&2
  exit 1
fi
"$EDITOR_PYTHON" -c 'import sys,platform; assert sys.version_info[:2] == (3,12); assert platform.machine() == "arm64"'
# Keep a check of the working editor dependencies, without writing pip tokens/URLs.
editor_versions() {
  "$EDITOR_PYTHON" - <<'PY'
import importlib.metadata as m,json
print(json.dumps({n:m.version(n) for n in ('numpy','pretty_midi','streamlit','pyarrow')},sort_keys=True))
PY
}
EDITOR_BEFORE=$(editor_versions)
UV_BIN="${BAND2TAB_UV:-}"
if [[ -z "$UV_BIN" ]]; then
  for candidate in "$PWD/.venv/bin/uv" "$PWD/../band2tab/.venv-bootstrap/bin/uv"; do
    if [[ -x "$candidate" ]]; then UV_BIN="$candidate"; break; fi
  done
fi
if [[ -z "$UV_BIN" ]] && command -v uv >/dev/null 2>&1; then UV_BIN=$(command -v uv); fi
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo 'uvが見つかりません。BAND2TAB_UVに既存のuvの絶対パスを指定してください。' >&2
  exit 1
fi
"$EDITOR_PYTHON" - <<'PY'
import shutil
print(f'空き容量: {shutil.disk_usage(".").free / 1024**3:.1f} GiB')
PY
if [[ -L .venv-mu-b ]]; then
  echo ".venv-mu-b はリンクです。別環境への書込みを避けるため停止します。" >&2
  exit 1
fi
if [[ -e .venv-mu-b && ! -x "$MU_PYTHON" ]]; then
  echo '.venv-mu-b が不完全です。自動削除せず停止します。' >&2
  exit 1
fi
if [[ ! -x "$MU_PYTHON" ]]; then "$UV_BIN" venv --python "$EDITOR_PYTHON" .venv-mu-b; fi
"$MU_PYTHON" -c 'import sys,platform; assert sys.version_info[:2] == (3,12); assert platform.machine() == "arm64"'
# Avoid a second copy of the large downloaded wheels in uv's cache.
"$UV_BIN" --no-cache pip install --python "$MU_PYTHON" -r requirements-mu-mac-b.txt
"$UV_BIN" pip check --python "$MU_PYTHON"
"$MU_PYTHON" - <<'PY'
import importlib.metadata as m
import muscriptor, torch, torchaudio, numpy, soundfile, imageio_ffmpeg
assert m.version('muscriptor') == '0.4.0'
assert torch.__version__.split('+')[0] == '2.9.1'
assert torchaudio.__version__.split('+')[0] == '2.9.1'
assert int(numpy.__version__.split('.')[0]) >= 2
print('MuScriptor:',m.version('muscriptor'))
print('PyTorch:',torch.__version__,'NumPy:',numpy.__version__)
print('MPS利用可能:',torch.backends.mps.is_available())
print('FFmpeg:',imageio_ffmpeg.get_ffmpeg_exe())
PY
.venv-mu-b/bin/muscriptor transcribe --help >/dev/null
EDITOR_AFTER=$(editor_versions)
if [[ "$EDITOR_BEFORE" != "$EDITOR_AFTER" ]]; then
  echo '編集画面側の依存バージョンが変わっています。ここで停止します。' >&2
  exit 1
fi
"$EDITOR_PYTHON" -m unittest discover -s tests -p test_b_workflow.py -v
"$EDITOR_PYTHON" scripts/b_make_test_audio.py
printf '\n環境準備OK。編集画面の依存バージョンを保持しました。\n'
printf 'モデル重みの取得と採譜はまだ実行していません。\n'
printf '次に、本人がMuScriptor smallの条件を確認・同意してHugging Faceへログインします。\n'
