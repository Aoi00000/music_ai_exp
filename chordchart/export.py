"""Readable, printable chord charts; preserve every raw timed segment."""
from pathlib import Path
import argparse,csv,html,json,math
from .core import display,write_json


def write_chart(result,out,title='コード譜',bpm=None,first_downbeat=None,beats_per_bar=4):
    if (bpm is None)!=(first_downbeat is None):raise ValueError('小節表示にはBPMと最初の小節頭（秒）の両方を指定してください')
    if bpm is not None and (not math.isfinite(bpm) or bpm<=0 or not math.isfinite(first_downbeat) or first_downbeat<0):raise ValueError('Invalid tempo or downbeat')
    if beats_per_bar not in (2,3,4,5,6,7):raise ValueError('Unsupported meter')
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for s in result['segments']:
        start,end=float(s['start']),float(s['end'])
        if not (math.isfinite(start) and math.isfinite(end) and 0<=start<end):raise ValueError('Invalid chord interval')
        label=s.get('label',s.get('chord','N'));rows.append(dict(start=start,end=end,label=label,symbol=display(label),needs_review=bool(s.get('needs_review',True))))
    rows.sort(key=lambda s:s['start'])
    write_json(dest/'chords.json',result)
    with (dest/'chords.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['start','end','label','symbol','needs_review']);w.writeheader();w.writerows(rows)
    esc=lambda s:html.escape(str(s));cards=[]
    if bpm is None:
        for s in rows:
            cards.append('<div class="cell"><small>'+f"{s['start']:.2f} - {s['end']:.2f} 秒"+'</small><strong>'+esc(s['symbol'])+('</strong><em>要確認</em>' if s['needs_review'] else '</strong>')+'</div>')
        explanation='秒単位のコード進行。拍・小節は未確認です。区間を省略せず表示しています。'
    else:
        duration=float(result['duration']);bar_seconds=60/bpm*beats_per_bar
        bounds=([0.] if first_downbeat>0 else [])+[first_downbeat+i*bar_seconds for i in range(max(0,math.ceil((duration-first_downbeat)/bar_seconds)))]+[duration]
        bounds=sorted(set(x for x in bounds if 0<=x<=duration))
        for left,right in zip(bounds,bounds[1:]):
            pickup=left<first_downbeat;number=0 if pickup else round((left-first_downbeat)/bar_seconds)+1
            content=[]
            for s in rows:
                if s['start']>=right or s['end']<=left:continue
                beat=1+(max(left,s['start'])-left)*bpm/60
                content.append('<span><b>'+esc(s['symbol'])+'</b><small>'+f'{beat:.2f}拍'+(' / 要確認' if s['needs_review'] else '')+'</small></span>')
            cards.append('<div class="cell"><small>'+('前奏・弱起' if pickup else str(number)+'小節')+f' / {left:.2f}秒</small>'+(''.join(content) or '<span>未解析</span>')+'</div>')
        explanation=f'確認済み一定テンポ {bpm:g} BPM / {beats_per_bar}拍子 / 最初の小節頭 {first_downbeat:g}秒。小節内の変化位置を拍で表示。変拍子・テンポ変化には秒表示を使ってください。'
    text='<!doctype html><html lang="ja"><meta charset="utf-8"><title>'+esc(title)+'</title><style>body{font:16px system-ui,sans-serif;color:#19364a;margin:32px;line-height:1.6}h1{font-size:28px}p{max-width:75em}.chart{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.cell{border:1px solid #a9bcc8;padding:12px;break-inside:avoid}.cell small{display:block;font-size:12px;color:#526672}.cell strong{display:block;font-size:25px}.cell span{display:inline-block;margin:5px 10px 0 0}.cell em{font-size:12px;color:#934900}footer{margin-top:24px;font-size:12px}@media(max-width:650px){.chart{grid-template-columns:repeat(2,minmax(0,1fr))}}@media print{body{margin:0}.cell{break-inside:avoid}h1{font-size:22px}@page{size:A4;margin:15mm}}</style><h1>'+esc(title)+'</h1><p>'+esc(explanation)+'</p><div class="chart">'+''.join(cards)+'</div><footer>自動推定の下書き。N.C.＝和音なし、?＝不明。要確認はモデル候補間の差による目印で、正解確率ではありません。元の時刻・ラベルはJSONとCSVに保持しています。ブラウザの印刷からPDF保存できます。</footer></html>'
    (dest/'chord_chart.html').write_text(text,encoding='utf-8');return dest/'chord_chart.html'


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--json',action='store_true',help='Use existing chord inference JSON');p.add_argument('--checkpoint',type=Path,help='Explicit project review model; default is published ensemble');p.add_argument('--title',default='コード譜');p.add_argument('--bpm',type=float);p.add_argument('--first-downbeat',type=float);p.add_argument('--beats-per-bar',type=int,default=4);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new output folder')
    if a.json:result=json.loads(a.input.read_text())
    else:
        from .infer import infer
        result=infer(a.input,checkpoint=a.checkpoint,baseline=a.checkpoint is None,device='cpu')
    print(write_chart(result,a.out,a.title,a.bpm,a.first_downbeat,a.beats_per_bar))


if __name__=='__main__':main()
