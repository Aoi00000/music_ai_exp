"""Presentation-ready Japanese progress report, built from measured results."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,PageBreak,NextPageTemplate,Image,Preformatted


def main():
    data=json.load(open('docs/measurements/learning_20261119.json'));font='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
    pdfmetrics.registerFont(TTFont('JP',font));navy=colors.HexColor('#21435d');grey=colors.HexColor('#edf2f5')
    body=ParagraphStyle('body',fontName='JP',fontSize=9.5,leading=15,wordWrap='CJK',spaceAfter=8,textColor=colors.HexColor('#243746'))
    title=ParagraphStyle('title',parent=body,fontSize=22,leading=31,spaceAfter=15,textColor=navy)
    sub=ParagraphStyle('sub',parent=body,fontSize=13,leading=20,spaceBefore=10,spaceAfter=9,textColor=navy)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=12,spaceAfter=6)
    code_style=ParagraphStyle('code',parent=body,fontSize=8,leading=12,backColor=grey,borderPadding=8,spaceBefore=5,spaceAfter=10)
    cell=ParagraphStyle('cell',parent=body,fontSize=8.5,leading=13,spaceAfter=0)
    out=Path('output/pdf/learning_progress_20261119.pdf');out.parent.mkdir(parents=True,exist_ok=True)
    doc=BaseDocTemplate(str(out),pagesize=A4,title='Band2Tab 学習結果と11月25日までの計画',author='Band2Tab Project',leftMargin=42,rightMargin=42,topMargin=53,bottomMargin=40)
    def decorate(canvas,document):
        w,h=canvas._pagesize;canvas.setStrokeColor(navy);canvas.line(42,h-37,w-42,h-37);canvas.setFont('JP',8);canvas.setFillColor(navy)
        canvas.drawString(42,h-28,'Band2Tab  |  人工知能演習  |  2026-11-19');canvas.drawRightString(w-42,23,str(document.page))
    for name,size in [('Portrait',A4),('Landscape',landscape(A4))]:
        w,h=size;doc.addPageTemplates(PageTemplate(id=name,pagesize=size,frames=[Frame(42,40,w-84,h-93,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=decorate))
    flow=[]
    def p(text,style=body):flow.append(Paragraph(text,style))
    def h(text):p(text,sub)
    def table(rows,widths):
        t=Table([[Paragraph(escape(str(v)).replace('\n','<br/>'),cell) for v in row] for row in rows],colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),grey),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,navy),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d6e0e6')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));flow.extend([t,Spacer(1,10)])
    def code(text):flow.append(Preformatted(text,code_style))
    width=A4[0]-84
    p('学習で何が改善し、<br/>どこがまだ難しいか',title)
    p('音声・動画からギターとベースの音を認識し、修正できるコード譜・TABへつなぐ。発表日：2026年11月25日。担当A＝学習・評価、担当B＝画面・編集・譜面。最終利用は1台のMac。')
    h('今回の実施：追加学習6条件＋固定評価')
    p('YourMT3へのLoRAを2条件、MuScriptor出力への教師あり補正を4条件学習した。歪み・短い減衰音を正解付きで増やし、実ベースの以前の失敗を改善した。一方、強い既存モデルへの上乗せは、持続中の音を改善しても音符開始を壊すことが分かった。')
    table([['比較の条件','既存 → 学習後','判断'],['既知の実ベース17件\nYourMT3・GM分類も採点','音符F1  0.7649 → 0.8962\nFrame F1  0.6671 → 0.8295','大きく改善。ただし再検査であり、未使用テストではない。'],['別ギター演奏7組の固定評価\n同じ分離音・同じMuScriptor','音符F1  0.6706 → 0.6578\nFrame F1  0.7299 → 0.7360','持続音は改善。音符開始は悪化し、自動適用を保留。']],[width*.40,width*.32,width*.28])
    h('既存モデルを超えたと言える範囲')
    p('ベースと分かる同一条件では、実ベース17件のFrame F1は増強後YourMT3が0.8326、MuScriptorが0.8210。一方、音符F1はMuの0.9154が増強後YourMT3の0.8972を上回る。特定の指標・条件の改善と、全体の優位を区別する。')
    h('現在使えるもの')
    p('Macでファイル取り込み→6パート分離→ギター／ベース個別採譜→補正前後JSON保存まで確認済み。既存画面の既定処理は変更していない。BはこのJSONを音一覧・編集へ接続する。')
    p('Azureは停止済み（割り当て解除）。11,450ファイルと449チェックポイントの回収・照合を確認。',small)
    flow.append(PageBreak())
    p('01  機械学習と教材',title)
    h('何を学習し、何を固定したか')
    table([['段階','方法と自分たちの学習'],['音源分離','学習済みHTDemucs 6sを固定。ドラム・ボーカル・鍵盤等を分け、推定ギター／ベースを使う。固定の周波数カットだけでは分けない。'],['音符認識','YourMT3本体約4,708万パラメータを固定し、LoRA約235万パラメータを更新。既存ラベルの音高・楽器・開始・終了を教師にする。'],['Muへの補正','MuScriptorを固定し、350次元の音響特徴から音符を残す確率と時刻補正を学習。最終候補は約2.5万パラメータの小型NN。'],['譜面化','生の秒単位音符を保存し、拍・音価・弦とフレットを後段で整える。元の運指・奏法の復元精度は未検証。']],[100,width-100])
    p('これは教師あり深層学習。強化学習は行っていない。損失関数は「正解音符との違い」「音を消しすぎる誤り」「開始・終了のずれ」を数値化したもの。学習はその損失を小さくする。')
    p('LoRAの学習と分離・予測生成はAzure GPU。回収後の小型補正器4条件はMac CPUで学習し、GPUを止めたまま比較した。最終評価もMac CPUで実施。',small)
    h('今回追加した教材')
    table([['用途','量と制約'],['歪み・短い減衰音','Slakhの正解付きベースから204学習／28検証条件。元曲は30学習／8検証。歪みは時刻を維持し、減衰は終了ラベルも変更。重複音の曖昧な例は除外。'],['実ベース保持','MUSDBの7秒音源を32学習／8検証。正解音符はない。元YourMT3の疑似教師を使う保持学習は、増強のみより悪化した。'],['補正用混合音','GOAT学習演奏47件のclean/noisy対94条件＋増強ベース54条件。最終補正器は混合94条件だけで学習。検証16条件、固定後の別演奏14条件。']],[100,width-100])
    p('データ増強は新しい奏者や弾き方を増やしたことにはならない。FiloBassは主にコントラバス、Slakhは合成音。歪んだ実エレキベースとの領域差が残る。CQTは音階に沿う特徴量であり、正解を出すAIそのものではない。',small)
    flow.extend([NextPageTemplate('Landscape'),PageBreak()])
    p('02  同一条件の比較図',sub)
    flow.append(Image('docs/measurements/learning_20261119.png',width=680,height=680*1360/2040))
    flow.extend([NextPageTemplate('Portrait'),PageBreak()])
    p('03  評価・採否と残りの計画',title)
    h('開発用の共通67条件')
    rows=[['モデル','音符F1','Frame F1','全音一致']]
    for r in data['common_development_validation'].values():
        s=r['summary'];rows.append([r['label'],f"{s['note_f1']:.4f}",f"{s['frame_f1']:.4f}",f"{s['active_exact_match']:.4f}"])
    table(rows,[width*.49,width*.17,width*.17,width*.17])
    p('67条件は9データセット群の平均。繰り返し使用する開発検証であり、最終テストではない。増強＋追加更新の一式を変えたため、増強だけの因果効果とは断定しない。')
    h('固定評価からのセルフレビュー')
    p('音符F1は音高・楽器・開始（50ms以内）を確認し、Frame F1は20msごとの発音を確認する。補正が正しかった開始を動かすと、持続時間は改善しても音符F1が下がる。補正器は新しい音高の取りこぼしを補えない。')
    p('固定評価は人工ミックス14条件＝ギター7組の対。自然なバンド録音ではない。ベースは学習外奏者の別時刻だが、同じ録音の他区間は確認済み。7組のFrame差の95%区間は約+0.0009～+0.0138。少数標本から実バンド全般の優越を主張しない。')
    h('11月25日までの分担')
    table([['日程','担当A','担当B／共同'],['11/19-20','開始時刻の保護、ギターの未検出音を補う比較を別プロトコルで準備。','Mac処理を画面から呼ぶ。原結果と補正案を切り替えて表示。'],['11/21-22','設定を固定して別データで評価。今回開封した評価を再び未使用と呼ばない。','音一覧・編集・拍と運指の確認。元の音符を保持。'],['11/23-24','図表・成功／失敗例・学習の説明を確定。','デモ固定、発表リハーサル。'],['11/25','発表：意図→学習→比較→限界を示す。','一台のMacでデモ。']],[65,(width-65)/2,(width-65)/2])
    p('次の優先順位：開始を保護した終了のみの学習、ギター候補の補完、複数音色・分離ノイズへの頑健性。学習回数やデータ量を増やすこと自体を目的にしない。',small)
    flow.append(PageBreak())
    p('04  Macでの操作と終了確認',title)
    p('実行場所：/Users/moment/Documents/EEIC_3A/0_後期実験/人工知能演習/music/music_ai_exp',small)
    code("cd '/Users/moment/Documents/EEIC_3A/'\\\n'0_後期実験/人工知能演習/music/music_ai_exp'")
    h('ファイルを取り込む：まず強い基準で確認')
    code('PYTHONPATH="$PWD" .venv-mu/bin/python \\\n  -m scripts.a_band_pipeline \\\n  \'/absolute/path/song.mp4\' --out results/song_baseline')
    p('入力を自分のファイルの絶対パスへ置き換える。MP3・WAV・MP4・MOV等に対応。出力は毎回新しいフォルダへ。notes.jsonのparts.guitar／parts.bassをBの画面へ渡す。onset &lt;= t &lt; offsetで時刻tの全音を列挙できる。',small)
    h('学習済み補正は比較用に指定')
    code('PYTHONPATH="$PWD" .venv-mu/bin/python \\\n  -m scripts.a_band_pipeline \\\n  \'/absolute/path/song.mp4\' --out results/song_review \\\n  --corrector models/stem_note_corrector_development.pt')
    p('raw_partsとrawのMIDIを保持する。--seconds 10で短く試せる。テンポが確認できた場合だけ--bpm 96等を加え、作られるscore_input.jsonを既存の譜面処理へ渡す。96は例。生MIDIの120 BPMを検出テンポと扱わない。',small)
    h('Azureは割り当て解除まで確認')
    code('.venv-yourmt3/bin/python scripts/a_azure.py deallocate\n.venv-yourmt3/bin/python scripts/a_azure.py status')
    p('PowerState/deallocatedが終了状態。本人の許可でCLIを接続済み。認証情報は非公開のvendor/private内に保持。次の実験はa_gpu_session.pyで包み、実験失敗時も回収・照合・停止を試みる。電源断・通信断・強制終了まで保証するものではない。',small)
    h('詳細・再現と参考資料')
    p('詳しい実行手順：docs/MAC_BAND_HANDOFF.md<br/>実験記録：docs/EXPERIMENT_20261119.md<br/>公開集計・図：docs/measurements/learning_20261119.*<br/>コード：https://github.com/Aoi00000/music_ai_exp<br/>今回の作業ブランチ：codex/a-training-20261117',small)
    p('先行実装：MuScriptor（github.com/muscriptor/muscriptor）、YourMT3+（github.com/mimbres/YourMT3）、Demucs（github.com/facebookresearch/demucs）。データ：GOAT（arxiv.org/abs/2509.22655）、Slakh（slakh.com）、MUSDB（sigsep.github.io/datasets/musdb.html）。数値は本プロジェクトの独自評価で、各論文のベンチマーク値ではない。',small)
    doc.build(flow)
    from pypdf import PdfReader
    reader=PdfReader(out)
    if len(reader.pages)!=5:raise ValueError('Expected five deliberate pages, got '+str(len(reader.pages)))
    if any(len(page.extract_text().strip())<20 for page in reader.pages):raise ValueError('Empty PDF page')
    print('LEARNING_PDF_READY',out,len(reader.pages),flush=True)


if __name__=='__main__':main()
