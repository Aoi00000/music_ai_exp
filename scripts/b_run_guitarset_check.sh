#!/bin/bash
# One small real-recording integration check; do not train or select a model.
set -euo pipefail
cd "$(dirname "$0")/.."
test -x .venv/bin/python
test -x .venv-mu-b/bin/muscriptor
export HF_HOME="$PWD/models/huggingface"
export TORCH_HOME="$PWD/models/torch"
mkdir -p results
RUN_DIR=$(mktemp -d "$PWD/results/b_real_guitar_XXXXXX")
echo "保存先: $RUN_DIR"
.venv/bin/python scripts/b_prepare_guitarset_sample.py --out "$RUN_DIR"
.venv-mu-b/bin/muscriptor transcribe "$RUN_DIR/input.wav" \
  --model small --device cpu --dtype float32 --detect-tempo false \
  --output "$RUN_DIR/raw.mid" 2>&1 | tee "$RUN_DIR/transcribe.log"
.venv/bin/python - "$RUN_DIR" <<'PY'
from pathlib import Path
import json, sys
import pretty_midi
from scripts.b_inspect_muscriptor_midi import remove_duplicate_metadata
folder=Path(sys.argv[1])
source=folder/'raw.mid'
normalized,removed=remove_duplicate_metadata(source.read_bytes())
copy=folder/'normalized.mid'
copy.write_bytes(normalized)
midi=pretty_midi.PrettyMIDI(str(copy))
reference=json.loads((folder/'reference.json').read_text())
rows=[]
for part in midi.instruments:
    name='Drums' if part.is_drum else pretty_midi.program_to_instrument_name(part.program)
    print(f'{name}: {len(part.notes)}音')
    rows.append(dict(program=int(part.program),name=name,is_drum=bool(part.is_drum),notes=len(part.notes)))
count=sum(len(part.notes) for part in midi.instruments)
target=sum(len(part.notes) for part in midi.instruments if not part.is_drum and 24<=part.program<=39)
report=dict(instruments=rows,notes=count,guitar_bass_notes=target,
            reference_notes=sum(map(len,reference['parts'].values())),
            removed_duplicate_metadata=removed,purpose='integration_check',independent_unknown_test=False)
(folder/'instrument_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('検出音符:',count,'/ ギター・ベース判定:',target)
print('重複メタ情報の確認・除去:',sum(removed.values()),'件（別コピー）')
print('確認用音源:',folder/'input.wav')
print('検査結果:',folder/'instrument_report.json')
if count==0:
    raise SystemExit('音符が0件です。採譜結果の確認が必要です。')
if target==0:
    print('ギター・ベース判定が0音です。楽器の取り違えを確認してから画面へ接続します。')
else:
    print('次にモデルが出した音高・時刻・楽器を確認して編集画面へ接続します。')
PY
