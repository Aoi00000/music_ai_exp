# 担当B：Macで音符編集とMusicXMLを確認する

## 変更範囲

担当Aの `codex/a-training-20261117` の `43bdec6263d3647aa52601515d3ba88664a468f5` を土台にする。
フォルダ配置はAと同じリポジトリ直下。Aのデータ・モデル・学習・評価コードと既存テストは変更しない。
既存のMuScriptor、取得済みローカル重み、Basic Pitch、MIDI読込み、拍検出、コード認識への接続を保持する。

- 既知音符と音符JSONからモデルなしで起動できる入口を追加。
- 編集中も、他パート、出典、音響スコア等の付加情報を保持。
- 音高または秒時刻を変更した行は、古い音響スコア・運指を無効にして要確認とする。
- 不正な音高／秒時刻／強さ／拍列を拒否し、書出しの成功後に保存対象を切り替える。
- 古い書出しもresults/ui内に保持する。失敗した編集で前の保存済み譜面を置き換えない。
- 未割当音をTABテキストにも記載し、再割当後の古い警告を除去。
- NumPy 1.26と互換のPyArrowを固定。
- `setup_mac.sh --score-only` と、MuScriptorなしで編集画面を起動できるランチャー。
- コード解析が失敗しても、成功した音符の結果を表示する。

## 確認済みのMac側の状態

元の開始セットは `/Users/haruya/Downloads/band2tab`。
Downloads権限を許可した後、専用のPython 3.12.16 / arm64をuv 0.13.8で準備した。
既存のシステムPython 3.9.6は保持している。
既知音符9音、issues 0、JSON／MIDI／MusicXML／TABの4ファイル生成の実行ログを確認した。
MuseScoreを導入し、MusicXMLを開けたとの回答を受けた。
その後、MacでB用11件のテスト成功、9音・issues 0、編集画面で52→53の反映成功との回答を確認した。
保存済みのJSON・MIDI・MusicXMLの読戻しと、譜面の全項目の目視確認は次の手順で行う。

## MacでBブランチを取得する

元のband2tabと仮想環境は保存し、別のGit作業フォルダを使う。
下記は初回専用で、保存先が存在すれば停止する。

```bash
bash <<'BASH'
set -euo pipefail
SOURCE=/Users/haruya/Downloads/band2tab
TARGET=/Users/haruya/Downloads/music_ai_exp_B
if [ -e "$TARGET" ]; then
  echo 'music_ai_exp_B は既にあります。上書きせず停止します。'
  exit 1
fi
test -x "$SOURCE/.venv-bootstrap/bin/uv"
git clone --branch codex/b-mac-editor-20261117 --single-branch \
  https://github.com/Aoi00000/music_ai_exp.git "$TARGET"
cd "$TARGET"
git status --short --branch
"$SOURCE/.venv-bootstrap/bin/uv" venv --python 3.12 --seed .venv
BAND2TAB_PYTHON="$PWD/.venv/bin/python" bash setup_mac.sh --score-only
./Band2Tab.command
BASH
```

期待値：B用テスト11件がOK、既知音符9音・issues 0、ターミナルにローカルURLが表示される。
Chromeで `http://127.0.0.1:8501` を開く。起動時の任意のメール登録は表示しない。
ポート8501が他のアプリで使用中なら `BAND2TAB_PORT=8502 ./Band2Tab.command`。
終了は起動したターミナルでCtrl+C。

## 編集の受入れ確認

1. 「既知音符の例を開く」。ギターとベースのTAB、時刻別の音名が出る。
2. 編集パートguitarを選び、最初のMIDI音高52を53に変える。
3. 「修正を反映する」。時刻0のギター音名がE3からF3へ変わり、ベースE1は保持される。
4. notes.jsonとscore.musicxmlを保存する。JSONのpitch 53と、ベースpitch 28を確認。
5. MusicXMLをMuseScoreで開く。6線／4線、音高、休符、2回の同音発音を確認。
6. 終了時刻を開始時刻より前にする等の不正な編集が拒否され、以前の保存対象が残ることを確認。

issues=0は認識精度や見栄えの保証ではない。未割当・量子化衝突がある譜面を完成版としない。
原音の秒時刻と量子化した拍位置は別に保存する。

アプリを動かしたまま別のターミナルを開き、編集後の保存ファイルを照合する。

```bash
cd /Users/haruya/Downloads/music_ai_exp_B
.venv/bin/python scripts/b_check_saved.py --example-edit --open-score
```

最新の編集結果を読み、JSONとMIDIの音高・秒時刻、弦の競合、52→53以外の音符の保持、
既知音符例のMusicXMLの9音を検査する。検査後の譜面をMuseScoreで開く。
複数の結果がある場合は `--notes results/ui/.../notes.json` で対象を明示できる。

## モデルを使う段階

モデルなしの確認後、同じプロジェクト直下で次を使う。

```bash
BAND2TAB_PYTHON="$PWD/.venv/bin/python" bash setup_mac.sh
export HF_HOME="$PWD/models/huggingface"
export TORCH_HOME="$PWD/models/torch"
.venv-mu/bin/hf auth login
```

モデル利用条件を本人が確認・同意して認証し、30秒の入力で解析を試す。
Aの重み・評価結果は必要なものだけ別途取り込む。公開モデルや学習重みをGitへ追加しない。
MacでのMPS実行、実音源精度、拍の妥当性、オフラインでの新規解析は未確認。

## 統合後のクラウド検証

- Python 3.12.14 / Linux x86_64 / PyTorch 2.9.1+cpu。
- Aの既存57件＋Bの11件、計68件が成功。
- PyTorch・librosaなしの別環境でもB用11件が成功。
- bash／zsh構文とPythonコンパイルが成功。
- 開始セットの既知音符MusicXMLは4.0公式スキーマに適合することを確認済み。
- Mac上の修正版の導入・操作は利用者と確認してから完了扱いにする。
