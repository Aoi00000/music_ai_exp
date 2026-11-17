# 担当A：実行・確認・回収

制作は2026-11-17〜11-24、発表11-25。学習はAzure、最終推論は1台のMac。端末で実行する場所を混同しないでください。

## 1. Macから接続

Macのターミナルで `ssh aiexp`。SSHエイリアスは授業の接続先を指すローカル設定です。鍵がパスフレーズ付きの場合は、Macで `ssh-add ~/.ssh/id_rsa_aiexp` を実行してご自身で入力します。鍵やパスフレーズをGitやチャットへ載せません。

Azure Portalで割当VMを起動し、実行中を確認してから接続します。新規VMや外部サービスは不要です。

## 2. Azure：プロジェクト直下

この実行では `~/music_ai_exp` を使用しています。既存の授業環境を利用し、GPU版PyTorchは置き換えません。

```bash
cd "$HOME/music_ai_exp"
source "$HOME/venv/bin/activate"
python /opt/aiexp/check-gpu.py
python -m pip install -r requirements-a.txt
python -m pip check
python -m unittest discover -s tests -v
```

CPUではなく実際のCUDAデバイスで、モデルの前向き計算と勾配計算まで検査します。

## 3. Azure：データ取得・検査

```bash
bash scripts/a_prepare.sh
```

順序はGuitarSet→BabySlakh→CQT作成→監査→伴奏付きデータ→監査→少量実データの学習確認です。公式チェックサムで検証し、ダウンロードは中断再開可能です。

GuitarSetの既知の注釈誤り `04_BN3-154-E_comp`、`04_Jazz1-200-B_comp`、`02_Funk2-119-G_comp` は記録して除外します。原データは改変しません。

- https://github.com/marl/GuitarSet/issues/5
- https://github.com/marl/GuitarSet/issues/4

学習確認では同じ少量データを学習・確認に使い、配線と学習可能性を確認します。その得点はテスト精度として使いません。

## 4. Azure：A0/A1を学習

```bash
bash scripts/a_train.sh
```

同じ構造・初期化・学習量で、訓練データだけを変更します。このスクリプトは訓練と検証までです。検証データで設定を決めてから `bash scripts/a_test.sh` を実行します。テスト結果を見ながら設定を変更しません。

- `runs/a0/best.pt`, `runs/a1/best.pt`: 推論用の最良重み。
- `runs/*/epoch_01.pt`等: 各エポックの重みを保持。
- `runs/*/last.pt`: 最終完了エポックの重み・Adam状態・乱数状態。途中再開用。
- `reports/*_val.json`, `reports/*_test.json`: 曲別の精度。訓練スコアと区別。

各エポックで一時ファイルから置き換えるため、書込み途中の重みをバックアップしません。

A0が2エポックまで終わって中断した場合、残りを実行する例：

```bash
python -m band2tab train --manifest data/manifest_clean.json \
  --out runs/a0 --epochs 5 --steps 100 --batch 4 --device cuda \
  --resume runs/a0/last.pt
```

データの論理的な指紋と学習設定を検査します。`--init best.pt` は重みを受け継ぐ別の学習であり、途中再開とは異なります。

## 5. Mac：作業中と終了前に回収

Macのローカルリポジトリ直下で実行します。

```bash
bash scripts/a_backup.sh
```

`backups/azure`にモデル・評価・ログ・manifestを回収します。Git登録対象ではありません。モデルのSHA256を照合し、最終版はMacでも読み込んで確認します。特徴量キャッシュと公式データのアーカイブも容量を確認して回収します。

Azureが利用不能になった場合は、回収したキャッシュとmanifestのパスをMacの保存先に合わせ、CPU/MPSで再開できます。公開アーカイブを再取得するときは、公式URLとチェックサムから再構成します。

## 6. 終了処理は必須

1. 学習・転送を完了するか、安全なチェックポイントで中断する。
2. モデル・設定・分割・評価・ログをMacへ回収し検証する。
3. Azure Portalで停止する。
4. **「停止済み（割り当て解除）」になったことを確認する。**

SSHを終了しただけ、学習プロセスを止めただけではVM課金は止まりません。認証・ネットワーク・学習エラーで中断するときも、同じ終了処理を行います。
