# 担当Bのエージェントにそのまま渡す更新プロンプト

あなたはBand2Tabの担当Bです。先に渡した `docs/B_AGENT_PROMPT_20261119.md` の結合確認を継続してください。この文書は担当Aの追加学習後の差分です。Bの既存作業・人の編集・拍設定を保護し、app.pyをA版で置き換えないでください。

## A側で新たに完了したこと

Aのブランチは `codex/a-training-20261117`、リポジトリは https://github.com/Aoi00000/music_ai_exp です。新たに27回の教師あり学習を実行し、音を残す補正器とフルバンド教材を比較しました。Mu本体や分離器の再学習、強化学習は今回行っていません。

反復16条件の音符F1は、基準81.842%、改善幅を優先した案82.634%、変更を抑える案82.214%。既知14条件では67.061%、67.684%、67.132%。未知曲全体での優越やTAB精度を証明した値ではありません。追加案は既定で自動適用しません。

重要：改善幅を優先したベース補正は、先に渡した12秒フルバンドで11音を削除し、音符F1を上げてもFrame F1を35.061%から34.403%へ下げました。この案を採用済み・実用的と扱わないでください。そこから再学習した変更を抑える案を、今回の任意の結合確認対象にします。

## 受け取る追加物（元のZIPはそのまま必要）

- 元の `band2tab_AB_integration_20261119.zip` は変更していません。
- 新しい追加ZIP：`band2tab_review_addendum_20261119.zip`、272,540 bytes。
- SHA256：`a840d86e4e689afe221a31011d2c42181af2b59bd86f10dabf623811b287a747`。
- 音源やMu本体は含まず、任意比較のJSON、ベース補正の3-seed重み、従来のギター・オクターブ補正重み、ハッシュと説明を含みます。私的な受渡し用でGitには置いていません。ユーザーから別途受け取ってください。
- Gitの検証記録：`docs/measurements/B_REVIEW_ADDENDUM_20261119.json`。
- 結果と計画：`docs/FEEDBACK_CYCLE_20261119.md` と8ページの追補PDF。

今回の12秒の追加JSONは **ギター90音・ベース134音で、基準と同一** です。無理に変更や改善を作らないでください。BPM・拍子・最初の小節頭も引き続き未確認です。結合確認用の既に使った訓練由来音源であり、独立精度テストではありません。

## まず行う作業：再推論は不要

1. Bの未コミット変更を確認・保護し、別の統合ブランチで必要なコードだけ取り込んでください。追加学習によってUIのJSON契約は変わっていません。
2. 元ZIPと追加ZIPのSHA256、内部 `checksums.json` を照合してください。
3. 元原音・基準JSONに、追加ZIPの `review/notes.json` を別の比較案として読み込みます。`parts` / `raw_parts` が基準、`suggested_parts` が任意案です。`parts`だけ読む画面には `review/suggested_notes.json` を利用者の明示選択で読み込めます。基準と案を混同して上書きしないでください。
4. 原音を聴いて確認した誤りを1か所編集し、元結果・他パート・拍情報を保持して再保存・再読込してください。今回の補正は同一結果なので、編集箇所をAI改善として説明しません。
5. 再生位置、ギター・ベースの実音一覧、TAB、コード表示を同時刻で確認。コード外音も残します。拍未確認のまま確定した小節線にしないでください。
6. ファイル選択→解析→比較→編集→MusicXML保存の受入確認へ進めます。Aが確認したのはコマンドとデータの一致で、B画面の結合完了ではありません。

## 任意：Macから比較案を再生成する場合

作業場所は **BのMacのmusic_ai_expリポジトリルート**。以下はAコードとMu medium環境を用意した後です。Aの仮想環境をコピーせず、必要な重みだけ追加します。追加ZIP展開先を実際の場所へ置き換えてください。

```bash
mkdir -p models
cp '/実際の追加ZIP展開先/weights/bass_sound_keep_review.pt' models/
cp '/実際の追加ZIP展開先/weights/octave_filter_review.pt' models/
PYTHONPATH="$PWD" .venv-mu/bin/python -m scripts.a_bass_keep_apply \
  results/b_integration_baseline --out results/b_integration_sound_review \
  --checkpoint models/bass_sound_keep_review.pt \
  --octave-filter models/octave_filter_review.pt
```

前提：`results/b_integration_baseline` は同じMu **medium**（SHA256 `ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec`）でAの `a_band_pipeline` が出した未編集結果です。元のmix/guitar/bass WAV、raw_guitar.mid、raw_bass.midと完全なnotes.jsonが必要です。smallや編集後JSONには使わず、拒否されたら無理に検証を外さないでください。元の小さなZIPだけで再生成に必要な全ファイルが揃うとは限らないため、最初はJSON読込だけで進めます。補正推論にはYourMT3モデルは不要です。既存アプリの採譜器を勝手に変更しません。

出力の `automatic_application_approved` はfalse、元の音は保存されます。新しいフォルダを使ってください。Aでは本番Mac環境と評価環境の音符一致、全92テスト合格を確認しました。Bの16 GiB Macの所要時間・メモリは別途測定が必要です。

完了報告：コミット、実機・モデルサイズ、基準224音の保持、編集前後の差分、原音時刻との同期、拍情報、MusicXML再読込、時間・メモリ・空き容量、未完了事項。11/20〜21結合確認、11/22設定固定、11/23デモ固定、11/24練習、11/25発表。AzureはAが成果回収後に割り当て解除済みで、Bの結合確認に起動は不要です。
