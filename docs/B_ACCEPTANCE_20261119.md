# B担当：Macでの接続確認結果とAへの受渡し条件

2026年11月19日。利用者がMacで実行し、チャットに貼ったログと回答で確認した記録。
69音の実音源接続を確認したコードはBブランチ `cd4a01f`。
担当Aの最新ブランチは読み取りのみで確認し、Bの画面へ一括マージしていない。

## 確認できた範囲

- Apple Silicon / メモリ16 GiB / Python 3.12.16。作業開始前の空き容量36.9 GiB。
- 編集画面の `.venv` とMuScriptor用 `.venv-mu-b` を分離。編集側の依存バージョンを保持。
- 利用者本人がMuScriptor smallの利用条件を確認し、認証・モデルへのアクセスを完了。
- 人工波形の入出力を経て、実際のGuitarSet単独ギター録音の先頭12秒を解析。
- small / CPU / float32 / 拍検出なし。ログ上の採譜処理は3.13秒。モデル読込み・取得を含む総所要時間ではない。
- 69音をAcoustic Guitar (nylon)として推定。
- JSONにモデルの69音の音高と生の秒時刻を保持。
- 書出しMIDIを読み戻し、音高一致、秒時刻の差が1 MIDI tick以内であることを確認。
- 元の注釈にある129 BPM・4/4・拍列で記譜。MIDIの仮120 BPMを推定テンポとして使用しない。
- 運指の音高整合・弦競合0・MusicXMLの音価と小節長・issues 0を検査。
- MuseScoreで警告なく開けたと利用者が回答。
- Band2TabのChrome上の音符表へJSONと比較用原音を読み込めたと回答。
- 表を変更せず「修正を反映する」を実行し、再保存した69音と拍設定が元のJSONと一致。
- 再保存後のJSON・MIDI・運指・MusicXMLの検査もすべてPASS。

## Macにある成果物

基準フォルダ：`/Users/haruya/Downloads/music_ai_exp_B`

|内容|基準フォルダからのパス|
|---|---|
|12秒の実ギター原音|`results/b_real_guitar_HvAk62/input.wav`|
|モデルの元MIDI|`results/b_real_guitar_HvAk62/raw.mid`|
|注釈と拍設定|`results/b_real_guitar_HvAk62/reference.json`|
|最初の譜面・JSON・照合記録|`results/b_real_guitar_HvAk62/handoff_r9uvi1ks/`|
|編集画面での再保存結果|`results/ui/band2tab_2vh5115m/edited_f479et2o/`|

これらの実行結果・音声・重み・認証情報はGitには含めない。
元音源の出典・CC BY 4.0の帰属表示は音源フォルダの `ATTRIBUTION.txt` に保持。

## 認識について分かったことと、未確認のこと

注釈74音に対し、同じ音高・開始±50msの一対一照合で58組が一致。
推定側未対応11音、注釈側未対応16音。楽器・終了時刻を含む完全一致の正解率ではない。
元のGuitarSetはAも利用している既存データセットなので、未知実曲での精度として公表しない。

この結果は「実ギター1件を編集・譜面保存へ通せる」ことの確認。
任意のバンド録音でのギター／ベース分離、演奏技法、元奏者の運指、コード認識の正しさ、
原音を聴いて行う実際の音符修正、Aの重みを含むHaruyaのMacへの導入は別途確認が必要。

## Aから最初に受け取るもの

最初は重み一式より、次の小さなセットで結合を確認する。

1. ギター・ベースを含む10〜30秒の原音。JSONと同じ時刻原点を保つ。
2. 対応する完全な `notes.json`。`parts.guitar` / `parts.bass` に秒単位の `pitch/onset/offset/velocity`。
3. 原音で確認したBPM・拍子、または拍列。未確認ならその旨を残す。MIDIの120 BPMを根拠にしない。
4. 採譜経路・モデル版・重みのハッシュ。補正を含む場合は補正前の `raw_parts` と提案も保持。

Aの2026-11-19の最新手順は
[MAC_BAND_HANDOFF.md](https://github.com/Aoi00000/music_ai_exp/blob/codex/a-training-20261117/docs/MAC_BAND_HANDOFF.md)。
記載の「このMacにモデル回収済み」はAのMacを指し、HaruyaのMacへの回収完了ではない。

Bの取込み方針：

- `parts` が補正後で `corrector_applied=true` の形式は、既定で `raw_parts` を表示する。
- 最新の `parts/raw_parts` が元結果、`suggested_parts` が追加候補の形式にも対応する。
- 元結果が既定。比較案は利用者が明示的に選び、`automatic_application_approved=false` を保持する。
- BPMがないJSONは、確認済みの値を画面で指定してから記譜する。
- 元の音符・提案・出典・手動修正を保存し、音符を黙って削除・移調しない。

別環境でのモデル導入はAの固定した必要ファイルとサイズを確認してから進める。
最初の受渡しJSONの表示にはモデル一式や分離器のダウンロードは不要。

## 最新の互換性修正の検証範囲

`suggested_parts` の既定表示・明示選択、テンポ未確認時の停止、
再保存後の元音符・編集・提案・出典の保持をクラウドで回帰検査した。
この互換性修正を加えたB用テストは計26件。
上記69音のMacでの受入結果と、追加形式のクラウド検査は区別して記録する。
