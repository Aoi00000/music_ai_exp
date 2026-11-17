# MuScriptorを土台にした学習補正（2026-11-17）

## 方針の変更

主採譜器はMuScriptor mediumに固定します。「独自モデルと対決して勝ったものに置換する」ことを主方針にはしません。音声→MuScriptor→学習した補正候補→既存の編集・TAB化という構成にします。

MuScriptorの約3億パラメータは今回更新しません。まず、原音とMuScriptorの音符を入力にした小型の**音符単位の補正器**を学習します。大規模モデルを直接追加学習する方針とは区別します。既存のA8は比較・確認支援用に保持します。

本体の微調整も将来の選択肢ですが、今回の11/25までの期間と少量の正解データでは、既存モデルの能力を崩す危険と学習・検証の実装負担があります。先に追加部分の効果を単独で測れる構成を実験します。これは本体の微調整では効果が出ないと証明した意味ではありません。

## 何を学習するか

`band2tab/refine.py` のEventRefinerは5,507パラメータの小型ニューラルネットです。

- 入力：MuScriptorの音高・音長・楽器指定、原音CQTの局所・倍音・全体の特徴（計82特徴）。
- 正解：音高と開始時刻±50msで既存の正解MIDIと照合し、ギター／ベース／対応する対象音なしの3クラス。
- 出力：各クラスのスコア。MuScriptorの元の楽器指定を初期の優先値として残し、学習した追加出力で補正します。
- 学習：AdamW、学習率0.001、重み減衰0.001、seed17、150回の全訓練候補更新。録音ごとの総重みを揃えます。開始時のモデルと本体は別物です。

MuScriptorが「ピアノ」などと出力した音符も候補に残すため、ギター・ベースへの楽器の取り違えを回収できる可能性があります。余計なギター音を対象外と提案することもあります。

**この初版では、候補にない音の生成、音高変更、開始・終了の修正はしません。** 同じ音高を両楽器が同時に鳴らす場合の完全な分離も保証しません。「すべての誤りが直る」モデルではありません。まず何が改善するかを測ります。

## データと評価

今回は取得済みデータを再利用し、MuScriptor自身の誤りを実際に生成して教師にします。正解MIDIを適当に壊したデータだけで本番の誤りを再現したことにはしません。

- 訓練：BabySlakhのtrain元曲14曲のバンドミックス＋GuitarSet 12録音＋Guitar-TECHS 8録音。それぞれ固定12秒。
- 検証：別のBabySlakh元曲3曲＋GuitarSet 4録音。それぞれ固定12秒。
- 元曲群をtrain/valで分け、録音の長さから切り出し位置を決めます。正解やモデルの得点を見て区間を選びません。
- 切出し境界の偽オンセットを避けるため、訓練・検証とも最初の0.25秒の候補・正解を採点対象から除外します。音長の終端は切出し端で打ち切りです。
- 特徴量の平均・標準偏差は訓練部分だけから計算します。
- epoch25/50/100/150、信頼閾値0.6/0.75/0.9、元判定との差0.2/0.4を検証だけで選びます。何もしない条件も候補に含めます。
- 選択指標はギター＋ベースのNote F1。ミックスと単独音源を同じ重みで平均します。ミックスのFrame F1、楽器別、変更内容も別途保存します。
- IDMT22録音と従来のミックス3区間は学習・選択に使いません。ただし既に観察済みなので、再比較しても「初めての未知外部テスト」とは呼びません。公開モデルの事前学習との非重複も未保証です。

小さい検証セットで多数の条件を選ぶ試作です。検証で上がっただけで実バンドへの一般化を主張しません。音を削っただけで良く見えないよう、欠落・余計な音・音集合・楽器別の値を確認します。

## 初回学習の実測結果

34訓練区間から3,206候補（対象外1,551・ギター1,389・ベース266）を学習し、7検証区間でepoch150・閾値0.6・元判定との差0.2を選びました。各区間は12秒です。

|比較条件|MuScriptor単体|学習補正の提案|
|---|---:|---:|
|検証：合成ミックス3区間・音集合Frame F1|0.3312|0.4262|
|既観察のIDMT22録音・楽器込みNote F1|0.8331|0.8382|
|同22録音・楽器込みFrame F1|0.7679|0.7481|
|同22録音・楽器込みOffset F1|0.5516|0.5667|
|従来の合成ミックス3区間・音集合Frame F1|0.3848|0.3904|

「楽器込み」は、正解の楽器だけを取り出して採点せず、存在しない側のギター／ベースの誤検出も数える指標です。各録音のTP/FP/FNを楽器別照合後に合算してF1を求め、録音間で平均しました。Frameも楽器と音高の組を照合しています。

前の資料と同じ「正解楽器だけ」のIDMT Note F1も0.8361→0.8434でしたが、アプリで余計な別楽器が出る問題を隠さないよう、上表の厳しい数え方を優先します。ギター5録音では改善した例・悪化した例があり、ベースの正解楽器限定Note F1は0.8607→0.8502と低下しました。

**全面的な精度向上は確認できませんでした。** Note F1に小さな改善があってもFrame F1は低下し、ベースも一部悪化しています。したがって学習補正をアプリの自動処理にはまだ採用せず、今回の重みは確認候補の提示用に保存します。元のMuScriptorを常に利用できます。

新規の未知データでの優位性を示した結果ではありません。次は、単に±50msで音符が合うかだけでなく、持続中の音を消していないかも正解に反映した学習と、元曲・音色を増やした検証を優先します。現在のIDMT22録音で閾値を調整し直してはいません。

数値の原記録はMacの `backups/azure/reports/refiner_validation.json`、`refiner_existing_comparisons.json`、`refiner_strict_comparison.json` です。音符の変更履歴と全検証条件を残しています。

## 担当Bへの影響

**画面、TAB描画、音符編集、MusicXMLのデータ形式を作り直す必要はありません。** `parts -> guitar/bass -> pitch/onset/offset` を維持します。MuScriptorの生MIDIは保存し、補正候補・変更理由を別に保存します。

既存の `read_midi` と `write_outputs` の間に以下を追加できます。初期状態はreviewで、元の音符をそのまま渡します。

```python
from band2tab.refine import refine

r = refine(audio, midi, "models/muscriptor_refiner.pt", mode="review")
parts = r["parts"]                 # 元のMuScriptorのギター/ベース
proposals = r["suggested_parts"]   # 学習した補正候補（別に確認）
changes = r["refinement_changes"]
write_outputs(parts, out, bpm, meter, grid, beats=beats)
```

重みがないときは `refine(audio, midi, checkpoint=None)` で補正なしです。モデルの対象音域外や短い元音符もreview出力から削りません。Bの既存実装は今回変更していません。

補正候補を見せる場合は `changes` の from/to/score と音高・時刻を追加表示します。**スコアは校正済みの正解確率ではありません。** 自動適用は別の未使用データと実バンドで受入確認後に限定します。今回の重みは `automatic_application_approved=false` なので、applyを指定すると停止します。

## 現在のMacで試すコマンド

Macのターミナルで、以下を実行します。アプリ環境がまだなら先に `bash setup_mac.sh`。音声とMIDIは同じ音源・秒単位の時刻に対応させ、量子化前のMIDIを使用します。

```bash
cd "$HOME/Documents/EEIC_3A/0_後期実験/人工知能演習/music/music_ai_exp"
.venv/bin/python -m band2tab.refine \
  --audio inputs/song.wav --midi results/song.mid \
  --checkpoint models/muscriptor_refiner.pt \
  --mode review --out results/song_refinement.json
```

担当Aの今回の確認環境は `/private/tmp/chord-v5-qa-env/bin/python` です。通常は上記の`.venv`を使用します。Azureの学習を再実行する場合は、既存の固定プロトコルや結果を上書きせず、別の出力名と新しい評価設計を用意してください。

## 研究としての位置づけ

採譜結果を学習で補正する考え方自体には先行研究があります。今回の価値は新しい一般原理の発明ではなく、ギター・ベースに用途を限定し、原音の証拠・既存モデルの候補・変更履歴を合わせ、MuScriptor単体との差を評価することです。編集時間の短縮や実バンドでの有効性は今後測る項目です。

- MuScriptor公式： https://github.com/muscriptor/muscriptor
- 重み・モデル仕様： https://huggingface.co/MuScriptor/muscriptor-medium
- McLeod et al., *The MIDI Degradation Toolkit: Symbolic Music Augmentation and Correction*, ISMIR 2020： https://archives.ismir.net/ismir2020/paper/000182.pdf （誤り検出・補正タスクの参考。今回、同手法をそのまま実装したわけではありません。）
- Hiramatsu et al., *Statistical Correction of Transcribed Melody Notes…*, ICASSP 2021： https://eita-nakamura.github.io/articles/Hiramatsu_etal_TranscriptionCorrection_ICASSP2021.pdf （歌唱旋律の後段補正で、今回の多声音源・楽器割当とは条件が異なります。）
