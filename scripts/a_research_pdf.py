"""Readable Japanese history and measured-review PDF; source Markdown retained."""
from pathlib import Path
import re,json,textwrap,argparse
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Preformatted,Image,CondPageBreak,KeepTogether


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');a=parser.parse_args()
    status=json.load(open('docs/measurements/review_20261119.json'))
    if not a.draft and ('tone' not in status or not status['azure_cleanup'].get('deallocated')):raise ValueError('Final PDF requires completed GPU results and verified Azure deallocation')
    pdfmetrics.registerFont(TTFont('JP','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    navy=colors.HexColor('#203f58');grey=colors.HexColor('#eef3f7');width=A4[0]-88
    body=ParagraphStyle('body',fontName='JP',fontSize=9.4,leading=15.2,wordWrap='CJK',spaceAfter=9,textColor=colors.HexColor('#243744'))
    h1=ParagraphStyle('h1',parent=body,fontSize=22,leading=30,spaceAfter=18,textColor=navy,keepWithNext=True)
    h2=ParagraphStyle('h2',parent=body,fontSize=13,leading=20,spaceBefore=16,spaceAfter=10,textColor=navy,keepWithNext=True)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=12,spaceAfter=8)
    cell=ParagraphStyle('cell',parent=small,fontSize=7.9,leading=12,spaceAfter=0)
    code=ParagraphStyle('code',parent=body,fontSize=8.2,leading=12.5,backColor=grey,borderPadding=7,spaceAfter=10)
    flow=[]
    def p(text,style=body):flow.append(Paragraph(escape(text),style))
    p('Band2Tab',h1);p('学習で何が改善し、何が残ったか',h1)
    p('人工知能演習 / 2026年11月19日 / 発表：11月25日')
    p('音声・動画からギターとベースの実音を認識し、コード譜とTABの下書きを作る。強い既存モデルに対して、独自学習・データ加工・誤り補完の有効性を測る研究記録。')
    p('読み方',h2);p('第1部は最新の比較結果。第2部は独自CRNNから、音源分離、Mu補正、時系列モデル、候補選別、YourMT3へのLoRA、GOAT・増強へ至る記録。失敗・不採用・未検証事項も含む。')
    p('評価範囲',h2);p('開発検証と、以前見た録音の回帰確認を区別する。別々の表の値は直接比較しない。F1が高いことは、全音・全小節・原演奏の運指が完全に正しいという意味ではない。')
    p('機械学習の位置',h2);p('我々が更新したのは、正解音符による教師あり学習と、既存採譜器へのLoRA・候補真偽判定。MuScriptor開発元の強化学習とは区別する。音源分離器とMuScriptor本体は固定して使う。')
    p('詳細な再現資料：docs/RESEARCH_REVIEW_20261119.md、docs/LATEST_RESULTS_20261119.md、docs/measurements/review_20261119.json。',small)
    flow.append(PageBreak())
    def add_markdown(path,part):
        p(part,h1);lines=Path(path).read_text().splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            if line.startswith('# '):i+=1;continue
            if line.startswith('## '):
                flow.append(CondPageBreak(200));p(line[3:],h2);i+=1;continue
            if line.startswith('```'):
                block=[];i+=1
                while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
                # Keep literal commands and Japanese comments; no tiny overflowing lines.
                wrapped=[]
                for row in block:wrapped.extend(textwrap.wrap(row,100,replace_whitespace=False,drop_whitespace=False) or [''])
                flow.append(Preformatted('\n'.join(wrapped),code));i+=1;continue
            if line.startswith('|'):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    values=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                    if not all(re.fullmatch(r'[:\- ]+',v) for v in values):rows.append(values)
                    i+=1
                n=len(rows[0]);fractions=([.40]+[.60/(n-1)]*(n-1)) if n in (4,5) else ([.18,.40,.42] if n==3 else [1/n]*n)
                t=Table([[Paragraph(escape(v),cell) for v in row] for row in rows],colWidths=[width*f for f in fractions],repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),grey),('LINEBELOW',(0,0),(-1,0),.6,navy),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#cedae3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));flow.extend([KeepTogether([t]) if len(rows)<=8 else t,Spacer(1,10)]);continue
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')):paragraph.append(lines[i].strip());i+=1
            p(' '.join(paragraph))
    add_markdown('docs/LATEST_RESULTS_20261119.md','第1部  最新実験の結果')
    figure=Path('docs/measurements/review_20261119.png')
    if figure.exists():
        flow.append(PageBreak());p('同じ条件の中で比較する',h1);flow.append(Image(str(figure),width=width,height=width*1.05));p('各パネル内だけで比較する。上段は開発検証、下段の既知回帰は未使用テストではない。複雑なコード名やTAB運指の精度とは別の音符評価。',small)
    flow.append(PageBreak());add_markdown('docs/RESEARCH_REVIEW_20261119.md','第2部  目的・全体像・試行の記録')
    def decorate(canvas,doc):
        canvas.setStrokeColor(navy);canvas.line(44,A4[1]-38,A4[0]-44,A4[1]-38);canvas.setFont('JP',8);canvas.setFillColor(navy)
        canvas.drawString(44,A4[1]-28,'Band2Tab | 実験と分析の記録 | 2026-11-19');canvas.drawRightString(A4[0]-44,24,str(doc.page))
    out=Path('output/pdf/research_review_20261119.pdf');out.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=54,bottomMargin=44,title='Band2Tab 学習・評価の全体記録',author='Band2Tab Project')
    doc.build(flow,onFirstPage=decorate,onLaterPages=decorate)
    from pypdf import PdfReader
    r=PdfReader(out)
    if any(len(p.extract_text().strip())<25 for p in r.pages):raise ValueError('Unexpected empty page')
    print('RESEARCH_PDF_READY',out,len(r.pages))


if __name__=='__main__':main()
