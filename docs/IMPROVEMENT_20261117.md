# 学習改善・実音補正・外部評価（2026-11-17 改訂版）

**その後の方針変更と初回実験：[MuScriptorを固定し、学習補正を追加](MUSCRIPTOR_REFINEMENT.md)。本書の独自モデルA8の比較は過去の実験として保持します。**

**現在の成果発表には、この改訂版の値を使ってください。** 旧資料はSlakhベースのオクターブ補正前です。補正前のベース成績と総合平均は、実際に鳴っている高さの精度としては撤回します。旧データ・旧結果は追跡用に保存しています。

## 結果

新しい外部評価は、我々の学習・閾値選択に使っていないIDMTのギター5録音（4曲群）とベース17録音です。各セットの奏者・楽器は1種類で、自然なバンドミックスではありません。Note F1はオクターブ込みの音高と開始時刻±50msを照合した、曲別F1の平均（0〜1）です。

|方法|ギター5録音|ベース17録音|全22録音平均|
|---|---:|---:|---:|
|Basic Pitch 標準設定|0.743|0.543|0.588|
|Basic Pitch 最短音符30ms|0.609|0.395|0.443|
|独自CRNN 採用A8|0.714|0.758|0.748|
|MuScriptor medium・拍への丸めなし|0.753|0.861|0.836|

Basic Pitchは標準の最短音符127.7msと、他方式に揃えた30msの両方を示しています。閾値はいずれも公開標準値です。標準設定の追加比較は、最初の外部採点後に透明性のため追加したもので、独自モデルの選択や調整には使っていません。各方式の出力音域は共通のB0〜E6です。公開モデルの事前学習にIDMTが含まれないことまでは保証できません。

独自モデルは、同じ検証用112パートで、初期実装を実音基準で再採点した0.095から0.718へ改善しました。これは学習量・ラベル修正・音高拡張を含む実装全体の改善であり、データ追加だけの効果ではありません。検証値と外部テスト値を混ぜないでください。

## 何を修正・学習したか

1. **正解のオクターブ**：SlakhのベースMIDIは多くの音源設定で実音より1オクターブ高く、YourMT3の著者実装と同じ補正を適用しました。`scarbee_jay_bass_slap_both.nkm` は例外として保持します。GuitarSet、IDMT、推定結果全般を一律に移調する処理ではありません。
2. **キャッシュの整合性**：旧ラベルのキャッシュを使い回さず `cache_v2` を分離。正解から再生成したターゲットとキャッシュの完全一致を検査します。旧Slakhベースのmanifestによる学習・評価はエラーで停止させます。
3. **学習の継続**：初回500更新から最大10,000更新まで継続。読込みの繰返し解凍を避けて計算を効率化しました。その独自チェックポイントを初期値とし、実音ラベルで3,000更新の追加学習を行いました。1 epochは100更新であり、全曲を1巡する意味ではありません。
4. **音高拡張**：CQTの周波数軸と音高ラベルを同じ半音数だけ平行移動。ギターは-2〜+2半音、ベースは-12〜+2半音で、対象音がB0〜E6から外れない範囲に制限します。元音声を再録音したり、実際のアンプ音色を再現したりする拡張ではありません。
5. **実録音の追加**：Guitar-TECHSからDI音声・弦別MIDIを取得し、同期検査に通った32録音・6,846音を追加しました。P1の同一奏者の録音で、32人の演奏者・32個の完全に独立した楽曲という意味ではありません。
6. **既存モデルとの比較**：MuScriptor mediumを取得し、T4で動作確認。モデルは固定し、我々はこの基盤モデルを追加学習していません。公式画面に示されたSHA256と取得ファイルを照合しました。

例として実音G2（約98Hz）に対して、旧MIDIがG3（約196Hz）だったケースがあります。倍音は基音の整数倍に現れるため、最大のスペクトルピークが必ず実際の音高とは限りません。正解ラベルと実際の波形の対応を確認する必要があります。

## 機械学習の具体的な内容

`band2tab/model.py` のNoteCRNNを教師あり学習しています。約68万パラメータの小型モデルです。

- CQT：周波数を音階に沿って並べる。36 bin/オクターブ、22,050Hz、hop256（計算間隔約11.6ms）。低音の実効時間分解能が11.6msという意味ではありません。
- CNN：倍音や音色の局所的な形を学習。
- 双方向GRU：前後約4秒窓の流れから、持続音や発音し直しを学習。
- 出力：B0〜E6の66音高それぞれについて開始・持続・終了を同時に予測。単音だけでなく複数音を同時出力できます。
- 入力する楽器指定はguitar/bass。楽器自体を自動識別するモデルではありません。単独・分離済みパートで使うモデルです。

AdamW、追加学習率0.0003、batch4、seed17。開始/持続/終了の重み3:1:1のBCEを使用し、正例の不均衡を補う重みは上限20です。閾値は検証用データで選択しました。初期モデルは我々のランダム初期化からの学習で、Basic Pitch等の追加学習ではありません。

## データ追加の有無を分けた比較

同じ初期重み、seed、追加3,000更新、学習率、音高拡張で比較しました。

|条件|追加実録音|検証Note F1|外部Note F1|外部Offset F1|
|---|---:|---:|---:|---:|
|A6 対照|0|0.7165|0.7436|0.4204|
|A7 取得途中サブセット|9|0.7083|0.7512|0.4246|
|A8 同期検査済み全取得分|32|0.7181|0.7476|0.3823|

A8は検証用の得点だけで採用しています。外部評価でA7の値が上でも、採用モデルを差し替えていません。9録音は取得途中の同期済みサブセットで、無作為な枚数比較ではありません。

32録音の追加によるNote F1の差は対照比約+0.004と小さく、Offset F1は低下しました。データを増やせば常に改善するとは言えません。少ない元曲群・奏者、seed1種類の実験なので、統計的優位性は主張しません。

## 同期とデータ品質

Guitar-TECHSは最大100ms程度の位置ずれを含むとの配布元の注意があります。訓練用音声のアタックと訓練用MIDIから録音単位の中央値のずれを求め、中央値からの残差が40ms以内等の条件を通る録音を採用しました。取得したP1の41録音のうち32録音を使用し、9録音を同期不確実・不正な時間等の理由で除外しています。人手で全注釈を校閲したデータという意味ではありません。

外部IDMTの正解時刻はこの方法で補正していません。ギターの `promenade_mono.xml` が存在しない `promenade_poly.wav` を指す1件は、同梱の同名WAVと対応させ、長さを確認しました。原XMLは変更していません。

GuitarSetの既知の3件の注釈誤り、BabySlakhの無効な配布フラグ、分割の重複、出力音域外は以前から記録・検査しています。ラベル付きMIDIが存在するだけでなく、実音との意味上の対応も確認するよう修正しました。

## まだ達成していないこと

- **バンド入力の精度は未達です。** 歴史的なBabySlakhテスト3曲の先頭30秒ずつを固定して診断したところ、ギター＋ベースの実音集合のFrame F1は、MuScriptor直接解析0.3848、固定HTDemucs-6s→A8で0.3286でした。完全一致率はそれぞれ0.2165、0.2548です。分離を加えれば必ず改善するとは言えません。この区間にはギター無音の区間も含み、自然な実バンドの新規テストではありません。公開モデルのSlakh事前学習との重複も否定できません。単独IDMTのNote F1との直接比較はできません。
- A8の外部Offset F1は0.382、MuScriptorは0.554。音の長さの認識はまだ課題です。
- 単独楽器の「鳴っている音集合」の完全一致率は、A8でギター約0.529・ベース約0.489、MuScriptorで約0.564・約0.697です。Note F1が高くても全時刻で全音が正しいわけではありません。
- 自然な実バンド録音、強い歪み、ボーカルの干渉、奏法、コード名、原演奏と同じ運指、編集時間の短縮は別評価が必要です。
- TABは弾けない候補を明示する下書きです。音高から弦・フレットは一意に決まりません。今回のMac出力例では28音を保持し、要確認5件を残しました。譜面の完全性の評価ではありません。

## 研究・プロダクトとしての価値

独自小型モデルは外部評価でBasic Pitchの標準設定を上回りましたが、MuScriptorには届きませんでした。事前学習規模・学習曲・パラメータ数は等しくないため、手法単体の優越やSOTAとは主張しません。今回の32録音追加の効果も限定的でした。

成果として示せるのは、(1) 小型モデルの学習と実音のラベル品質を検証できる再現可能な比較、(2) コード名とは独立したギター/ベースの実音を保持する処理、(3) 音を削除せず修正・TAB下書きに渡すMac用基盤です。主採譜の比較候補はMuScriptor、独自モデルは小型の単独パート解析・確認支援候補とします。ただし、バンド入力で実用精度に達した構成はまだ選定できていません。確認支援による編集時間短縮も未評価です。

## 現在のMacで使う場所・コマンド

Macのターミナル：

```bash
cd "$HOME/Documents/EEIC_3A/0_後期実験/人工知能演習"
cd music/music_ai_exp
```

現在の成果物：

- `models/acoustic_review.pt`：検証データで採用したA8。実音ラベル・閾値を同梱。
- `models/acoustic_review.json`：SHA256と出所・閾値。
- `models/muscriptor-medium/model.safetensors` と `config.json`：取得・照合済みMuScriptor。アプリはこのローカル重みを優先します。
- `backups/azure/`：音声・正解・CQT・チェックポイント・評価・ログ。
- `backups/azure/data/manifest_v2_*_mac.json`：Macの保存先に合わせたmanifest。

最終アプリ用の環境を作る場合：

```bash
bash setup_mac.sh
```

これは担当Bと統合するときに実行します。今回のMac確認は担当Aの検証環境で行っており、GUI全体の受入試験が完了した意味ではありません。

環境構築後、単独パートの独自モデルを使う例：

```bash
.venv/bin/python -m band2tab notes inputs/guitar.wav \
  --checkpoint models/acoustic_review.pt --part guitar \
  --out results/guitar
```

拍・小節の正解を検査する例ではありません。指定しなければ譜面用テンポは120です。raw MIDI/JSONの秒単位時刻は保持します。入力音源に合わせて `--bpm` を指定してください。

MuScriptorを取得済みの重みで使う例：

```bash
.venv-mu/bin/muscriptor transcribe inputs/song.mp3 \
  --model "$PWD/models/muscriptor-medium/model.safetensors" \
  --output results/song.mid
.venv/bin/python -m band2tab tab results/song.mid \
  --out results/song
```

最終的な使用場所は1台のMacです。Azureへの接続や認証トークンのチャット送信は推論に不要です。

## Azureで次の追加学習をする場合

Azure Portalで割当VMを起動してから、Macで `ssh aiexp`。以下はAzureのターミナルです。

```bash
cd "$HOME/music_ai_exp"
source "$HOME/venv/bin/activate"
export BAND2TAB_CACHE_ITEMS=512 OMP_NUM_THREADS=2
python -m band2tab train \
  --manifest data/manifest_v2_techs_full.json \
  --init models/acoustic_review.pt --out runs/next \
  --epochs 30 --steps 100 --batch 4 --lr .0003 \
  --pitch-augment --device cuda
python -m band2tab evaluate \
  --checkpoint runs/next/best.pt \
  --manifest data/manifest_v2_clean.json --split val \
  --tune --device cuda --out reports/next_val.json
```

今回の外部22録音は今後の閾値調整に使いません。次の最終比較は別の未使用の元曲・録音を確保してください。未使用のGuitar-TECHS P3音楽12録音は取得済みですが、評価に使う前に同期・正解の品質を確認する必要があります。

Macへ戻って回収：

```bash
bash scripts/a_backup.sh
python3 scripts/a_verify_backup.py
```

新しい音声・特徴量も増やした場合は、空き容量を確認して `data/` を追加回収します。今回のデータ一式は回収済みです。最後にAzure Portalで停止し、必ず **停止済み（割り当て解除）** を確認します。

## 11/25までの担当

- A（11/18〜21）：失敗例・音の長さ・未知音色を検査。追加学習をするなら新しい留保群を先に固定。モデル・閾値・JSON仕様をBへ渡す。
- B（並行）：1台のMacで入力→主採譜→ギター/ベース一覧→編集→TAB/MusicXMLを接続。取得済みローカル重みを使用する。
- 2人（11/22）：MP3/WAV/MP4/MOVの入力、長い曲、無音、弾けない運指、コードと実音の不一致を確認。
- 2人（11/23〜24）：図表・失敗例・デモを固定。認識精度、譜面化、使いやすさの評価を区別する。
- 11/25：発表。今回の学習、データ追加の限定的効果、実音補正、既存モデルとの比較を説明する。

## 一次資料

- Slakhベース補正の著者コード：https://huggingface.co/spaces/mimbres/YourMT3/blob/main/amt/src/utils/preprocess/preprocess_slakh.py
- GuitarSet：https://github.com/marl/GuitarSet
- BabySlakh：https://zenodo.org/records/4603870
- Guitar-TECHS：https://zenodo.org/records/14963133 （CC BY 4.0）
- IDMT-SMT-Guitar：https://zenodo.org/records/7544110
- IDMT-SMT-Bass-Single-Track：https://zenodo.org/records/7544099 （IDMTは今回外部評価のみに使用）
- Basic Pitch：https://github.com/spotify/basic-pitch
- MuScriptor：https://github.com/muscriptor/muscriptor
- モデル配布：https://huggingface.co/MuScriptor/muscriptor-medium
- 音源分離の著者実装：https://github.com/facebookresearch/demucs （今回は固定htdemucs_6s、shifts=1、segment=7秒、overlap=0.25。学習なし）
