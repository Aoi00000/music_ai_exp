# Mac統合用：分離・採譜と追加学習の接続

担当Aは学習・評価・認識処理、担当Bはファイル選択・進捗表示・音一覧・編集・譜面表示を担当する。最終的にはこのMacの同じプロジェクトを使う。既存画面の既定処理は今回変更していない。

## 最優先：小さな結果セットで結合確認

11/19の追加指示を反映し、モデル一式の移送より先に12秒の同じ入力を渡す。
`output/handoff/band2tab_AB_integration_20261119.zip`（約3.38MB）はGit対象外で、原音・推定G/Bステム・基準224音・補正案・コード結果・未確認の拍情報・重みハッシュを含む。
全18ファイルのCRC/SHA256照合済み。今回は補正が追加0・削除0で基準と同一であり、精度改善のデモにはしない。
既使用Slakh合成バンドの操作確認用で、未知曲の成績には追加しない。

Bへそのまま渡す依頼文は [B_AGENT_PROMPT_20261119.md](B_AGENT_PROMPT_20261119.md)。
公開可能な受け渡し監査は [B_HANDOFF_20261119.json](measurements/B_HANDOFF_20261119.json)。
モデル・音源を公開Gitへ入れず、ZIPは別途渡す。Bの画面での読込・編集・再保存・再生同期・画面からAの認識を呼ぶ受入確認は未完了。

## 1. 実行場所

以下はすべてMacのターミナルで実行する。

```bash
cd '/Users/moment/Documents/EEIC_3A/0_後期実験/人工知能演習/music/music_ai_exp'
```

`.venv-mu`にはMuScriptor 0.4.0・Demucs 4.0.1・FFmpegを導入済み。`.venv-yourmt3`はYourMT3の学習・評価用で、NumPyの互換性のため環境を分けている。モデルとデータはこのMacに回収済みで、GitHubには含めない。

## 2. 分離してから既存モデルで認識する

`/absolute/path/song.mp4`を取り込むファイルの絶対パスへ置き換える。MP3・WAV・MP4・MOV等をFFmpegで音声へ変換する。出力には毎回、新しいフォルダ名を指定する。

```bash
PYTHONPATH="$PWD" .venv-mu/bin/python -m scripts.a_band_pipeline \
  '/absolute/path/song.mp4' --out results/song_baseline
```

処理は「元ファイル→44.1kHzステレオ→6パートDemucs→推定ギター／ベースを個別にMuScriptor→秒単位の音符JSON」。分離前に16kHzへ落とさない。採譜器へ渡す分離音は共通の16kHzモノラル。鍵盤・歌・ドラムの音符は表示対象にしないが、分離音への漏れは残り得る。

`notes.json`の`parts.guitar`／`parts.bass`をB側で読み込めばよい。各音は`pitch`（MIDI番号）、`onset`／`offset`（秒）、`velocity`を持つ。`onset <= t < offset`で時刻`t`に鳴る音を列挙できる。推定音高から周波数を示すなら`440 * 2 ** ((pitch - 69) / 12)`だが、これは平均律の換算値で、実測した連続ピッチではない。

## 3. 学習済み補正を比較用に試す

```bash
PYTHONPATH="$PWD" .venv-mu/bin/python -m scripts.a_band_pipeline \
  '/absolute/path/song.mp4' --out results/song_learned_review \
  --corrector models/stem_note_corrector_development.pt
```

`parts`に補正案、`raw_parts`に同じ分離音・同じMuScriptorの補正前結果を保存する。`raw_guitar.mid`／`raw_bass.mid`も保持する。音高を新たに捏造する補正ではない。モデルと設定は固定した最終評価に使ったもの。時刻ごとの音集合は改善した一方、音符開始は悪化したため、自動採用は承認していない。Bの画面では比較・修正用として接続する。

最初の動作確認には`--seconds 10`を付ける。今回、Mac CPUで3秒の入力から分離・採譜・補正・JSON保存まで約36秒で完了した。これは処理時間の例で、任意の曲の速度や認識精度を保証しない。

歪み増強で追加学習したYourMT3の比較用重みも、`models/yourmt3_lora_augmentation_review.pt`へ保存した。以前の`yourmt3_lora_development.pt`は上書きしていない。YourMT3専用コマンドで`--adapter`へ指定できるが、Mu＋分離器を全面的に上回ったという意味ではない。

## 4. 譜面との接続

### 新しい補完案を比較する

元Muの音を保持したまま、学習で選別した未検出ギター候補を別のJSONへ出す処理を追加した。終了補正モデルは全音一致の保護基準に届かず不採用。新しい候補も開発・回帰確認段階なので、元の出力を自動的に置き換えない。

次は前節で作った`results/song_baseline`に対して実行する。YourMT3専用環境を使うため`PYTHONPATH`も含めて指定する。

```bash
PYTHONPATH="$PWD/vendor/yourmt3_deps:$PWD/vendor/yourmt3/amt/src:$PWD/vendor/yourmt3:$PWD" \
  HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled OMP_NUM_THREADS=2 \
  .venv-yourmt3/bin/python -m scripts.a_addition_apply results/song_baseline \
  --out results/song_addition_review
```

`notes.json`の`parts`と`raw_parts`は元Mu、`suggested_parts`が学習した追加候補。Bは両者を切り替えて確認できる。同じコマンドに`--octave-filter models/octave_filter_review.pt`を付けると、近接するオクターブ音の学習した除外案も組み合わせる。原結果は残り、ベースの音高・時刻は動かさない。この指定は自動採用の承認を意味しない。3秒の保存済み入力で補完のみの処理を約7.7秒で確認したが、任意の曲の時間・精度を保証する値ではない。

生の秒単位音符を保存し、量子化・運指割当で元の結果を上書きしない。モデルがMIDIに書く120 BPMは仮の値であり、曲のテンポではない。利用者が確認したBPMを指定した場合だけ、この処理は譜面用`score_input.json`を作る。

```bash
PYTHONPATH="$PWD" .venv-mu/bin/python -m scripts.a_band_pipeline \
  '/absolute/path/song.mp4' --out results/song_tempo_confirmed --bpm 96
PYTHONPATH="$PWD" .venv-yourmt3/bin/python -m band2tab score-json \
  results/song_tempo_confirmed/score_input.json --out results/song_score
```

`96`は例なので実際のテンポへ置き換える。4拍子等の譜面設定も確認する。変拍子・テンポ変化・3連符・奏法記号・複数ギターのパート分けは、この認識評価で精度検証していない。弦が足りない音を消したり別音高へ移したりせず、要確認として残す。

## 5. 曲のコード譜も書き出す

コード認識の現行実装はCQT＋CNN/双方向LSTMの公開5モデル。BTCは採用していない。MuScriptorの音符認識と別経路で、原ミックスからコード名を推定する。今回の音符F1はコード認識の精度ではない。

このMacには固定版lv-chordiaの重み5個を配置・読込確認済み。別のMacでは一度だけ準備する。

```bash
PYTHONPATH="$PWD" .venv-yourmt3/bin/python -m chordchart.setup bootstrap
OMP_NUM_THREADS=2 PYTHONPATH="$PWD" .venv-yourmt3/bin/python \
  -m chordchart.export '/absolute/path/song.mp4' --out results/song_chords
```

`chord_chart.html`をブラウザで開く。印刷からPDF保存できる。`chords.csv`と`chords.json`も保存する。小節頭が不明な場合は秒表示で、存在しない拍子や小節線を作らない。一定テンポと小節頭を確認できた場合だけ、次のように指定する。96、1.2、4はいずれも例。

```bash
OMP_NUM_THREADS=2 PYTHONPATH="$PWD" .venv-yourmt3/bin/python \
  -m chordchart.export '/absolute/path/song.mp4' --out results/song_chords_bars \
  --bpm 96 --first-downbeat 1.2 --beats-per-bar 4
```

既存の解析JSONから、再推論せず整形するには入力をそのJSONへ変え、`--json`を付ける。Bは`chordchart.export.write_chart`を呼び、画面のコード進行と同じJSONからダウンロードを作れる。既存の`band2tab.harmony.annotate_score`はTABのMusicXMLにコード記号を重ねる。複雑なコード・分数コードを簡単な三和音へ黙って置き換えない。

今回、新しいpower用12行だけを学習したコードモデルを`models/chord_head_review.pt`へ保存した。比較用として明示する場合だけ、上のコマンドへ`--checkpoint models/chord_head_review.pt`を追加する。既存画面の詳細設定にも、この重みがあるMacだけ「追加学習した候補（比較用）」を追加した。標準の5モデル経路は変更していない。18開発録音では小幅な改善だが、複雑なコードやC5の高精度な認識は未達である。

現在のHTMLは表記・ファイル入出力を確認した下書きで、任意の楽曲で正しいコード名を保証するものではない。時間ごとの音符一覧も別に残し、コードの誤認識を確認する材料にする。

## 6. Azureの安全な終了

Azure CLIは本人の許可を得て接続済み。認証情報はGit対象外の`vendor/private/azure_cli`に保存する。この場所を共有・コミットしない。

```bash
.venv-yourmt3/bin/python scripts/a_azure.py status
.venv-yourmt3/bin/python scripts/a_azure.py deallocate
```

終了確認は`PowerState/deallocated`。VM内部の`shutdown`だけで済ませない。

次のGPU実験では、具体的なSSH実行命令を次の終了処理で包む。

```bash
caffeinate -i .venv-yourmt3/bin/python scripts/a_gpu_session.py -- \
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes aiexp \
  'cd music_ai_exp && 実験用に確認した実行命令'
```

最後の行は説明用で、そのまま実行しない。VMの開始と学習命令を準備してから使う。実験の成功・失敗にかかわらず、成果回収→チェックポイント照合→割り当て解除を試み、成否を`reports/azure_session_cleanup.json`へ残す。通常の中断も処理するが、Macの電源断・ネットワーク断・強制終了まで保証する仕組みではない。毎日の自動シャットダウン設定は変更していない。終了後は必ず状態を確認する。

## ファイルの役割

| ファイル | 役割 |
|---|---|
| `scripts/a_band_pipeline.py` | Bが呼ぶローカル認識処理。音声・動画→音符JSON |
| `band2tab/separation.py` | 学習済み分離器への入出力・サンプルレート統一 |
| `band2tab/stem_refiner.py` | 音響特徴・教師・補正ネットワーク・補正案生成 |
| `scripts/a_stem_refiner_data.py` / `a_stem_refiner_predict.py` | 学習音源作成／固定モデルの予測作成 |
| `scripts/a_stem_refiner_cache.py` / `a_stem_refiner_train.py` | 特徴量キャッシュ／教師あり学習 |
| `scripts/a_stem_refiner_select.py` | 検証データだけで設定を固定 |
| `scripts/a_stem_refiner_holdout_*.py` | 固定後の別演奏で比較。結果を使った再選択はしない |
| `scripts/a_bass_augmentation_data.py` | 正解を維持した歪み・短い減衰音の増強 |
| `scripts/a_bass_retention_fetch.py` / `a_bass_retention_cache.py` | 追加の実ベース音源取得／元モデルの疑似教師生成 |
| `scripts/a_azure.py` / `a_gpu_session.py` | 特定VMの状態確認・終了処理 |

過去のYourMT3専用経路は[以前の引継ぎ](YOURMT3_MAC_HANDOFF.md)に残す。現在の判断と実験結果は[2026-11-19の記録](EXPERIMENT_20261119.md)を優先する。
