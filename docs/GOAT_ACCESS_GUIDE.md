# GOATの利用申請

2026-11-18に公式ページの表示を確認。申請はまだ送信していない。

公式ページ: https://zenodo.org/records/15690894

1. Files内のRequest accessへ進む。現在はログインしていなくてもフォームが表示される。
2. Your email addressに承認通知を受け取るメールアドレス、Your full nameに氏名を入力する。大学の所属を記載する場合は、正しい名称を自分で確認して研究目的に追記する。
3. Request messageに下記の英文を入れ、実際の研究目的と一致することを確認する。Chromeには下書きを入力済み。
4. 氏名とメールアドレスがレコード所有者へ共有される説明を確認し、同意する場合にチェックする。
5. Request accessを押す。ログインやメール確認を求められた場合は、自分で画面に従う。パスワードや認証コードをチャットに送る必要はない。
6. 送信後の受付表示や通知メールを保管する。承認通知を受け取ってから、ダウンロード権限と利用条件を確認する。申請しただけでは利用可能とは扱わない。

公式説明では非商用の研究・オープンソース用途が対象で、商用利用は内部テストも不可とされる。レコードのライセンス表示はCC BY-NC 4.0。承認日や11/25までの利用開始は保証されない。

## Request message（コピー用）

We are two students working on a non-commercial university machine-learning course project. We are developing a local research prototype that transcribes guitar and bass notes from audio and produces editable tablature. We would like to use GOAT's electric-guitar recordings and aligned MIDI/tablature annotations to adapt pretrained transcription models and evaluate note accuracy and playable fingering. Training will run on our university Azure GPU environment, and inference will run locally on a Mac. We will comply with the non-commercial terms, cite GOAT and the accompanying paper, and will not redistribute the dataset files. Could you please grant access for this educational research use?

## 承認された後

- データ所有者から指定された方法で取得する。承認メールに含まれる個人用リンクやトークンをGitHubに入れない。
- 配布ファイルのハッシュと構成を保存し、音声と対応するfine-aligned MIDIを確認する。量子化済みMIDIと混同しない。
- 同じ曲・元演奏のDI、アンプ違い、複数ギターパートを別の分割に散らさず、元曲で訓練・検証・評価に分ける。
- まず少数の訓練側音源で音高・同期・奏法ラベルを確認する。ベンドの途中の音高と押弦位置は区別する。
- 独立した新評価を確保してから追加学習する。既に見た今回の評価を新しい最終評価として再利用しない。

論文: https://arxiv.org/html/2509.22655v1

作者コード: https://github.com/JackJamesLoth/GOAT-Dataset
