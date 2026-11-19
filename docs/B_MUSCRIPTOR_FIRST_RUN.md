# B担当のMac：MuScriptorの初回準備

対象はHaruyaのApple Silicon Mac（メモリ16 GiB、準備前の空き容量36.9 GiB）。
最初はsmallで12秒の人工音源を使い、入出力の接続を確認する。
Aが比較しているmediumや追加学習済みYourMT3+の代替として採用したという意味ではない。
モデル本体の学習や、Aの学習データ取得は実行しない。

## 環境を用意する

`setup_mac.sh` の通常版はコード認識の準備まで行うため、この段階では使わない。
B用の準備スクリプトは、編集画面の `.venv` と別の `.venv-mu-b` に採譜用の依存を入れる。
既存の `.venv-mu` や重みは変更しない。
NumPyは編集側1.26、MuScriptor側2系が必要なので、環境を混ぜない。

Macの新しいターミナルで実行する。起動中の画面があれば、終了・再起動時に未保存の編集を失わないよう、先に必要なJSONを保存する。

```bash
bash <<'BASH'
set -euo pipefail
cd /Users/haruya/Downloads/music_ai_exp_B
if [ "$(git branch --show-current)" != codex/b-mac-editor-20261117 ]; then
  echo 'B専用ブランチではありません。自動で切り替えず停止します。'
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  git status --short
  echo 'ローカル変更があります。上書きせず停止します。'
  exit 1
fi
git pull --ff-only origin codex/b-mac-editor-20261117
BAND2TAB_UV=/Users/haruya/Downloads/band2tab/.venv-bootstrap/bin/uv \
  bash scripts/b_setup_muscriptor_mac.sh
BASH
```

成功条件：依存の整合性チェック、MuScriptorのインポートとCLIヘルプ、既存B用17件のテストが成功し、最後に「環境準備OK」と出る。
`MPS利用可能: True` は利用可否の検出であり、MPSで採譜が成功した意味ではない。
空き容量36.9 GiBだけでは必要容量を確定できないため、インストールの実際の結果を確認する。

同時に `results/b_audio_fixture_.../input.wav` と生成元 `reference.json` を作る。
毎回別の保存先を作るので、過去の結果は保持する。
音源は54音・12秒・16 kHz・モノラルの合成波形。楽器音色を正確に再現するものではない。
`reference.json` を画面へ読み込んだ場合は生成元の音符を表示するだけで、採譜モデルが正解したことにはならない。

## 本人が利用条件を確認してログインする

1. [MuScriptor smallの公式モデルページ](https://huggingface.co/MuScriptor/muscriptor-small)を開く。
2. 自分のHugging Faceアカウントで、モデル重みのCC BY-NC 4.0および追加条件を読み、同意する場合に申請する。
3. 同じMacで以下を実行する。トークンはターミナルの入力欄へ入れる。チャットやGitへ貼らない。Git認証情報への追加は不要なので、聞かれたら `n`。

```bash
cd /Users/haruya/Downloads/music_ai_exp_B
export HF_HOME="$PWD/models/huggingface"
export TORCH_HOME="$PWD/models/torch"
.venv-mu-b/bin/hf auth login
.venv-mu-b/bin/hf auth whoami
```

初回準備スクリプトは利用条件への同意や認証を代行せず、モデル重みも取得しない。
認証キャッシュはGit除外済みの `models/` 以下に置く。

## 次の採譜で確認すること

環境の実行ログと本人の同意・認証完了を確認してから、生成したWAVをsmallで解析する。
最初はCPUで互換性を確認し、その後必要ならMPSで同じ入力を比較する。
初回は拍検出を無効にし、追加モデルの取得を抑えて生MIDIを保存する。
人工音源は生成時に120 BPM・4/4と確定しているが、任意の実音源にこの値を流用しない。
音符が0件でも「成功した採譜」と判定せず、生MIDI・楽器・音数・時刻を確認する。
実バンド音源での精度確認は、Aからの10〜30秒の原音・完全な音符JSON・確認済みBPMを受け取って別に行う。

## クラウドで確認した範囲

macOS arm64 / Python 3.12向けの依存解決（バイナリwheelのみ）とシェル構文を確認。
LinuxではMac用セットアップがインストール前に停止することも確認。
人工音源の長さ・音数・波形を確認。Mac上の実インストール・モデル取得・採譜は利用者の実行結果で確認する。

## Macでの初回実行ログと次の検査（11/19）

利用者のMacで環境準備・B用17件のテストが成功した。
Hugging Faceの本人による認証後、small重みへのHEADアクセスが成功した。
CPU / float32 / 拍検出なしで人工音源を採譜し、MIDIの読み込みまで実行できた。
生成元54音に対し推定42音、楽器はすべてAcoustic Grand Pianoだった。
この件数だけで一致数・取りこぼし数は決められない。人工波形のため実楽器の精度は示さない。

pretty_midiの「non-zero tracks」の警告について、muscriptor 0.4.0の
`tokenizer/notes.py` の `note_events_to_midi` は、MuseScore対応のため
先頭トラックに加えて楽器トラックにもテンポ列を反復している。
警告を一律に隠さず、実際のMIDIで同時刻・同内容か確認する。

```bash
cd /Users/haruya/Downloads/music_ai_exp_B
.venv/bin/python scripts/b_inspect_muscriptor_midi.py \
  results/b_mu_small_cLPv7o/raw.mid \
  --reference results/b_audio_fixture_obvhwo_x/reference.json
```

検査は追加のモデル実行・ダウンロードを行わない。
元MIDIのSHAとイベントを確認し、別の `inspection_.../normalized.mid` に
先頭トラックと一致した重複メタ情報だけを除いたコピーを作る。
音符、プログラム、制御情報、他イベントの絶対tickを保持する。
先頭トラックと一致しない情報があれば推測せず停止する。

生成元と推定の音符を、楽器を区別せず同じ音高・開始±50msで一対一照合する。
終了時刻の差は別に記録する。これを楽器込み・音長込みの正解率や実曲の精度とは呼ばない。
推定のピアノ42音をギター／ベースに付け替えたり、生成元の音符で埋めたりしない。
このピアノのみのMIDIは、既存のギター／ベース抽出では対象音が0になるため、
実際のギター・ベースの録音かAからの受け渡しを使って次の接続確認を行う。

クラウドでは重複テンポの除去前後の音符時刻、矛盾時の停止、ピアノ判定と元ファイルの保持を
3件の回帰テストで検査した。利用者のMIDIそのものの内容は、このコマンドの結果で確認する。
