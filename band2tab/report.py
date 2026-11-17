"""Presentation tables: paired tracks, source-group bootstrap, no test tuning."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .learn import note_metrics

def baseline(manifest,predictions,split,out):
    items=[i for i in json.loads(Path(manifest).read_text()) if i['split']==split];rows=[]
    for it in items:
        p=Path(predictions)/(it['id']+'.json')
        if not p.exists():raise FileNotFoundError(f'Missing prediction: {p}')
        obj=json.loads(p.read_text());est=obj['notes']
        rows.append({k:it[k] for k in ('id','dataset','part','group')}|{'reference_notes':len(it['notes']),'estimated_notes':len(est),**note_metrics(it['notes'],est),'offset_f1':note_metrics(it['notes'],est,True)['f1'],'inference_seconds':obj['seconds']})
    result={'engine':'basic-pitch','config':obj['config'],'split':split,'tracks':rows,'macro_note_f1':float(np.mean([r['f1'] for r in rows])),'macro_offset_f1':float(np.mean([r['offset_f1'] for r in rows]))}
    Path(out).write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='tracks'})

METRICS=['precision','recall','f1','offset_f1','frame_f1','active_exact_match']
def summarize(files,out):
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True);experiments={Path(f).stem:json.loads(Path(f).read_text()) for f in files}
    # IDs and source groups must match within each split: no selectively missing failures.
    for split in {r['split'] for r in experiments.values()}:
        sets=[{t['id'] for t in r['tracks']} for r in experiments.values() if r['split']==split]
        if any(s!=sets[0] for s in sets):raise ValueError('Unpaired evaluation tracks')
    rows=[]
    for label,result in experiments.items():
        for part in ('guitar','bass'):
            for condition in ('clean','mixture'):
                tracks=[r for r in result['tracks'] if r['part']==part and (r['dataset']=='remix')==(condition=='mixture')]
                if not tracks:continue
                row={'experiment':label,'split':result['split'],'part':part,'condition':condition,'tracks':len(tracks),'source_groups':len({r['group'] for r in tracks})}
                row.update({m:float(np.mean([r[m] for r in tracks])) for m in METRICS})
                for poly in ('single','double','poly3plus'):
                    buckets=[r['by_reference_polyphony'][poly] for r in tracks]
                    tp=sum(b['tp'] for b in buckets);fp=sum(b['fp'] for b in buckets);fn=sum(b['fn'] for b in buckets)
                    row[poly+'_frame_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
                    row[poly+'_frames']=sum(b['frames'] for b in buckets)
                rows.append(row)
    with (dest/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Clustered, paired bootstrap: same resampled source groups for both runs.
    diffs=[];rng=np.random.default_rng(17)
    for split in ('val','test'):
        a=experiments.get('a0_'+split);b=experiments.get('a1_'+split)
        if not a or not b:continue
        for part in ('guitar','bass'):
            for condition in ('clean','mixture'):
                left={r['id']:r for r in a['tracks'] if r['part']==part and (r['dataset']=='remix')==(condition=='mixture')};right={r['id']:r for r in b['tracks']}
                if not left:continue
                groups=sorted({r['group'] for r in left.values()});bygroup={g:[right[k]['f1']-r['f1'] for k,r in left.items() if r['group']==g] for g in groups}
                samples=[np.mean([v for g in rng.choice(groups,len(groups),replace=True) for v in bygroup[g]]) for _ in range(1000)]
                diffs.append({'split':split,'part':part,'condition':condition,'delta_a1_minus_a0':float(np.mean([v for vs in bygroup.values() for v in vs])),'bootstrap_95_percent_interval':np.quantile(samples,[.025,.975]).tolist(),'independent_groups':len(groups),'caution':'small group count; one seed only; exploratory, not proof of superiority'})
    (dest/'comparison.json').write_text(json.dumps({'rows':rows,'paired_differences':diffs},indent=2))
    text=['# 学習・既存モデルの比較','', 'Note F1は音程（オクターブ込み）と発音時刻±50msの一致。値は0〜1です。Offset F1は終了時刻も評価します。Frame F1は20msごとの鳴っている音の一致です。完全一致率は、正解または予測に音がある時刻で音集合がすべて合う割合です。','', '|実験|分割|楽器|条件|曲/群|Note F1|Offset F1|Frame F1|音集合完全一致|','|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f'|{r["experiment"]}|{r["split"]}|{r["part"]}|{r["condition"]}|{r["tracks"]}/{r["source_groups"]}|{r["f1"]:.3f}|{r["offset_f1"]:.3f}|{r["frame_f1"]:.3f}|{r["active_exact_match"]:.3f}|')
    text+=['','## 解釈上の制限','','A0とA1は同一構造・初期値・更新数で訓練データを比較します。混合音は同じ分割内の他楽器を0dB RMSで重ねた制御実験で、音源分離後のアーティファクトや自然なバンド全体の評価ではありません。BabySlakhは合成音源20曲の試作用セットで、独立群が少なく、実録音への一般化は未確認です。','', 'Basic Pitchは公開済み学習を利用します。独自CRNNはゼロからの学習です。事前学習量・モデル規模は等しくありません。共通の出力音域B0〜E6、最短音符30msを使います。CRNNの閾値は検証データで選択し、Basic Pitchは公開の標準閾値です。学習データの重複について公開モデルの事前学習全体との非重複は保証できません。','', '同時音数別のFrame F1は正解の異なるMIDI音高数（1、2、3以上）で集計します。同音ユニゾンの弦本数や奏法は識別しません。TAB運指の正解率・コード認識精度・実曲全体の完成度は、この音符評価では検証していません。','', 'A1がA0を下回った条件もそのまま報告してください。差の区間は元曲群単位の対応付きブートストラップ（1000回）です。独立群が少なく乱数種も1つなので、統計的優位性の主張には使えません。']
    (dest/'evaluation.md').write_text('\n'.join(text)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(experiments);fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,part in zip(axes,('guitar','bass')):
        use=[r for r in rows if r['part']==part and r['split']==('test' if any(v['split']=='test' for v in experiments.values()) else 'val')]
        for j,condition in enumerate(('clean','mixture')):
            vals=[next((r['f1'] for r in use if r['experiment']==l and r['condition']==condition),np.nan) for l in labels]
            ax.bar(np.arange(len(labels))+(j-.5)*.35,vals,width=.35,label=condition)
        ax.set(xticks=np.arange(len(labels)),xticklabels=labels,ylim=(0,1),ylabel='Note F1',title=part);ax.tick_params(axis='x',rotation=20);ax.legend()
    fig.savefig(dest/'note_f1.png',dpi=160);plt.close(fig)
    print('Saved',dest)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('baseline');a.add_argument('--manifest',required=True);a.add_argument('--predictions',required=True);a.add_argument('--split',required=True);a.add_argument('--out',required=True)
    s=sub.add_parser('summarize');s.add_argument('files',nargs='+');s.add_argument('--out',default='reports/presentation')
    a=p.parse_args()
    if a.cmd=='baseline':baseline(a.manifest,a.predictions,a.split,a.out)
    else:summarize(a.files,a.out)
if __name__=='__main__':main()
