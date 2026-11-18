# music_ai_exp / Band2Tab

最新の実測と判断は[11月19日の学習記録](docs/EXPERIMENT_20261119.md)、実行・担当Bへの接続は[Mac統合手順](docs/MAC_BAND_HANDOFF.md)を参照してください。

[発表用PDF：学習結果と残りの計画](output/pdf/learning_progress_20261119.pdf)に、比較図・学習内容・2人の分担・Macの実行コマンドをまとめています。

**現在の判断**：歪み・短い減衰音の増強で、既知の実ベース17件の再検査におけるYourMT3の音符F1が0.7649→0.8962へ改善しました。音源分離＋MuScriptorを強い基準にして、独自の教師あり補正も比較しました。設定固定後の別演奏ではFrame F1が0.7299→0.7360へ改善しましたが、音符F1は0.6706→0.6578へ悪化したため、補正の全自動採用は保留です。Mac内でファイル取り込み→分離→採譜→補正前後JSONまで実行確認済み。既存UIの既定処理は変更していません。Azureは割り当て解除済みです。

[公開集計](docs/measurements/learning_20261119.json)・[比較図](docs/measurements/learning_20261119.png)。以前の最終評価32条件のうち、GOATの7演奏に対応する21条件は元演奏が今回の評価で開封済みです。未使用のまま残るSlakhは11条件です。

過去の経緯：[GOAT・混合音LoRA](docs/GOAT_BAND_PROGRESS_20261118.md)、[時系列補正](docs/TEMPORAL_RESULTS_20261118.md)、[分離・候補追加](docs/SEPARATION_RESULTS_20261118.md)、[持続時間補正](docs/DURATION_REFINEMENT_20261118.md)、[初回補正](docs/MUSCRIPTOR_REFINEMENT.md)。古い数値・未実装の記述は当時の段階の記録です。

ギター全パートとベース全パートの音を認識し、コード・時刻別音一覧・TAB下書きを提示する人工知能演習プロジェクトです。発表日は **2026-11-25**。

- 学習・比較実験はAzure。最終利用は1台のMac内で完結させます。
- アプリの標準経路はMuScriptor medium。研究ではYourMT3+へのLoRA追加学習も進めています。以前のCNN+BiGRU、候補補正MLP、分離経路選択、時系列補正との比較を記録し、採用候補の選択と本番での自動適用を区別します。
- 指定時刻の音一覧はギター＋ベースのみ。鍵盤・歌声・ドラム等は含めません。
- コード予測と音符を独立に保持し、コード構成音以外の音も表示します。
- コード・設定・モデル・評価を回収後、Azureを「停止済み（割り当て解除）」へ戻します。

## 担当A

[担当Aの実行手順](docs/A_TRAINING.md)、[学習・評価・発表の指針](docs/EVALUATION.md)。データ取得、正解検査、学習、評価、バックアップを担当します。[初回実験結果](docs/EXPERIMENT_20261117.md)に実測値と限界を記録しています。

## 担当B / 最終Mac

Python 3.12を用意して `bash setup_mac.sh` を実行します。現在のMacには同意・取得済みのMuScriptor mediumがあり、アプリはローカル重みを優先します。別のMacへの導入時は、重みの利用条件を確認してください。普段は `Band2Tab.command` をダブルクリックします。GUI全体の受入試験は担当Bとの統合作業として残っています。

学習を待たずに `.venv/bin/python -m band2tab score-json examples/notes.json --out results/example` で譜面出力を確認できます。

公開モデル・音源・自作の学習済み重み・認証情報はGitに登録しません。現在の精度達成状況は担当Aの実行記録で確認し、動作テストと実曲での認識精度を区別します。

再利用したコードのライセンスは [THIRD_PARTY.md](THIRD_PARTY.md) を参照してください。
