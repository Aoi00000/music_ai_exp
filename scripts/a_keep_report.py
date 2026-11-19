"""Publish aggregate feedback evidence and render its Japanese PDF addendum.

Run with the bundled PDF Python. Does not read private audio, notes, or tokens.
"""
from pathlib import Path
import argparse, hashlib, json, re
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def evidence():
    sources=['reports/keep_feedback_20261119/comparison.json','reports/bass_keep_v5/composition.json',
             'reports/sound_keep_conservative_v7/comparison.json','reports/keep_amp_v3/fixed_bass_policy.json',
             'reports/keep_feedback_20261119/fixture_metrics.json','reports/conservative_cli_replay_20261119.json',
             'backups/azure/reports/keep_feedback_20261119/mac_audit.json','reports/azure_session_cleanup.json',
             'reports/azure_keep_deallocated_20261119.json']
    out={'date':'2026-11-19','scope':'Reused development16 and known regression14, not fresh tests',
         'new_supervised_runs':27,'steps_per_run':800,'updates':21600,'reinforcement_learning':False,
         'mu_and_separator_frozen':True,'remaining_unopened_slakh_sources':11,
         'cycle_runs':{'C1':9,'C2':0,'C3':6,'C4':6,'C5':0,'C6':6,'C7':0},
         'seeds':[17,29,43],'tests_passed':92,'automatic_application_approved':False,
         'sources':{p:sha(p) for p in sources},
         'cycles_1_to_4':read(sources[0])['summary'],
         'improvement_priority':{k:read(sources[1])[k] for k in ('summary','bootstrap')},
         'smaller_change':{k:read(sources[2])[k] for k in ('summary','bootstrap')},
         'amp_same_policy':read(sources[3]),'fixture':read(sources[4]),'production_replay':read(sources[5]),
         'azure_backup':read(sources[6]),'azure_cleanup':read(sources[7]),'azure_direct_status':read(sources[8]),
         'checkpoint_sha256':{p:sha(p) for p in ('models/bass_keep_review.pt','models/bass_sound_keep_review.pt','models/octave_filter_review.pt')}}
    for name,src in [('improvement_priority',sources[1]),('smaller_change',sources[2])]:
        tracks=read(src)['tracks'];counts={}
        for scope,variants in tracks.items():
            counts[scope]={v:{key:sum(t['metrics'][key] for t in rows) for key in ('note_tp','note_fp','note_fn','frame_tp','frame_fp','frame_fn')} for v,rows in variants.items()}
        out[name]['pooled_counts']=counts
    path=ROOT/'docs/measurements/keep_feedback_20261119.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return path

def pdf():
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,KeepTogether
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    pdfmetrics.registerFont(TTFont('JP','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    styles={
      'body':ParagraphStyle('body',fontName='JP',fontSize=10.3,leading=16.2,spaceAfter=10,wordWrap='CJK'),
      'title':ParagraphStyle('title',fontName='JP',fontSize=19,leading=27,spaceAfter=17,textColor=colors.HexColor('#122b44'),wordWrap='CJK'),
      'head':ParagraphStyle('head',fontName='JP',fontSize=16,leading=23,spaceAfter=16,textColor=colors.HexColor('#122b44'),wordWrap='CJK'),
      'cell':ParagraphStyle('cell',fontName='JP',fontSize=9,leading=14,wordWrap='CJK'),
      'code':ParagraphStyle('code',fontName='JP',fontSize=8.4,leading=13,spaceAfter=3,wordWrap='CJK',backColor=colors.HexColor('#edf3f7'))}
    def p(s,style='body'):
        s=escape(s);s=re.sub(r'`([^`]+)`',r'<font color="#216b74">\1</font>',s)
        return Paragraph(s,styles[style])
    story=[];pages=(ROOT/'docs/FEEDBACK_CYCLE_20261119.md').read_text().split('<!-- page -->')
    width=A4[0]-84
    for ix,part in enumerate(pages):
        if ix:story.append(PageBreak())
        lines=part.strip().splitlines();i=0;code=False
        while i<len(lines):
            line=lines[i].strip();i+=1
            if not line:continue
            if line.startswith('```'):code=not code;continue
            if code:story.append(p(line,'code'));continue
            if line.startswith('|'):
                rows=[line]
                while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
                cells=[[c.strip() for c in r.strip('|').split('|')] for r in rows if not re.fullmatch(r'[| :\-]+',r)]
                n=len(cells[0]);widths=([width*.29]+[width*.71/(n-1)]*(n-1)) if n>3 else ([width*.17,width*.43,width*.40] if n==3 else [width/n]*n)
                t=Table([[p(c,'cell') for c in row] for row in cells],colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#43808a')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe4'))]))
                story.extend([t,Spacer(1,13)]);continue
            if line.startswith('# '):story.append(p(line[2:],'title'))
            elif line.startswith('## '):story.append(p(line[3:],'head'))
            else:story.append(p(line))
    dest=ROOT/'output/pdf/band2tab_feedback_plan_20261119.pdf'
    def decoration(c,doc):
        c.setFillColor(colors.HexColor('#122b44'));c.rect(0,A4[1]-14,A4[0],14,fill=1,stroke=0)
        c.setFont('JP',8);c.setFillColor(colors.HexColor('#607789'));c.drawString(42,26,'Band2Tab / Feedback & Plan / 2026-11-19');c.drawRightString(A4[0]-42,26,str(doc.page))
    SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=39,bottomMargin=44,title='Band2Tab 追加学習の結果と11/25までの計画',author='Band2Tab 担当A').build(story,onFirstPage=decoration,onLaterPages=decoration)
    print(dest)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pdf',action='store_true');args=a.parse_args()
    print(evidence())
    if args.pdf:pdf()
