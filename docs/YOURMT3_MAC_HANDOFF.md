# AからBへ：Macで追加学習済みモデルを試す

> 2026-11-19更新：現在の実行・分離処理は[Mac統合手順](MAC_BAND_HANDOFF.md)、学習結果と評価用データの開封状況は[最新実験記録](EXPERIMENT_20261119.md)を優先してください。以下は以前の段階の手順・記録を含みます。

これは音符認識の研究用経路。画面側の既定モデルを変更せず、同じ入力で比較する。配布重み・データ・環境はGitHubに含まれないため、AのMacに回収済みのものを使う。

## 実行場所と入力

Macのターミナルで次を実行する。

```bash
cd '/Users/moment/Documents/EEIC_3A/0_後期実験/人工知能演習/music/music_ai_exp'
mkdir -p runs/manual
```

アプリで使っている音声取り出し処理を共用する。手動なら、次の`/absolute/path/input.mp4`を自分の音声／動画ファイルの絶対パスへ置き換える。

```bash
ffmpeg -nostdin -n -i '/absolute/path/input.mp4' \
  -vn -ac 1 -ar 16000 -c:a pcm_f32le runs/manual/input.wav
```

`-n`は既存出力を上書きしない指定。元ファイルは変更しない。入力をmp3・movなどへ変えてもよい。下記の採譜コマンドには取り出したWAVを渡す。

## 同じファイルで比較

公開モデルそのまま：

```bash
bash scripts/a_yourmt3_mac.sh runs/manual/input.wav \
  --out runs/manual/base.mid --device cpu
```

開発用に選んだ追加学習済み重み：

```bash
bash scripts/a_yourmt3_mac.sh runs/manual/input.wav \
  --out runs/manual/learned.mid --device cpu \
  --adapter models/yourmt3_lora_development.pt
```

この重みは全候補の同一Mac検証が終わってから書き出す。まだ存在しない場合、学習途中の任意の重みを同じ名前へ置き換えて採用しない。モデルの由来と比較結果は、同名の`models/yourmt3_lora_development.json`に記録する。`.venv-yourmt3`は既存モデル用の環境から分離している。現時点で確認した実行先はCPUで、MPSの速度や同等性は未確認。

`learned.mid`に加えて`learned.json`ができる。JSONの`parts.guitar`と`parts.bass`には、次の情報が入る。

```json
{"pitch": 60, "onset": 1.2, "offset": 1.7, "velocity": 100, "confidence": null}
```

`pitch=60`はC4。`onset`・`offset`は秒。`confidence=null`は未計算であり、信頼度0や100%を意味しない。ボーカル等の音はJSON表示対象から除くが、モデルの楽器取り違えまでなくなる保証ではない。

## 表示・譜面化で守ること

- 任意時刻`t`で`onset <= t < offset`の音を集め、ギター・ベース別の音名とオクターブを表示する。同じ音高を両楽器が出した場合、楽器情報を残す。
- 生の秒単位音符を保存し、拍への整列や運指の割り当てで上書きしない。コード予測に含まれない旋律音も残す。
- 生MIDIの120 BPMは仮の値。原音から拍／小節を推定するか、利用者が確認したBPMを譜面化へ渡す。
- ギター6弦／ベース4弦を超える重なりは、複数奏者の可能性もある。弦が足りない音を黙って消したり、別の音高へ移したりせず、修正候補として残す。
- ベンド・スライドの奏法記号や、元の奏者の運指の復元は今回の音符F1で検証していない。TABは修正可能な下書きとして扱う。

6パート分離後の個別解析は追加の検証経路。現状のMac用コマンドは分離前のWAVを1つ解析するもの。Azure上での分離結果の改善を、そのままMacアプリの動作確認済みと読み替えない。

混合教材で学習した別の重みは`models/yourmt3_lora_band_development.pt`。評価で0.6766だったのは、この重みと6パート分離・個別解析を組み合わせた条件であり、上の1ファイル用コマンドで全バンド曲が同じ精度になる意味ではない。Bへ渡す形式は同じだが、Mac上の音源分離経路の確認を追加で行う。

一般向けに選んだ重みでも、外部の実エレキベース17件では音符F1が0.7649→0.7398、ミュート奏法4件では0.8748→0.4685と悪化した。既存エンジンを一律に置き換える段階ではない。まず同じ音源の原モデル／追加学習版を比較する機能として接続し、結果の違いを原音で確認できるようにする。

評価用データの`reports/idmt_bass_lines/manifest.json`には、原XMLの`raw_string`（1が最低音弦）と`raw_tuning_low_to_high`を保持している。このデータは4弦E-A-D-G。譜面側で1を最高音弦として扱う場合、弦番号は`5 - raw_string`へ変換する。ハーモニクス等はフレットだけから鳴る音高を求められないため、原ラベルの奏法も確認する。学習用へは混ぜない。
