"""Presentation tables: paired tracks, source-group bootstrap, no test tuning."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .learn import note_metrics,require_sounding_labels

def baseline(manifest,predictions,split,out):
    items=[i for i in json.loads(Path(manifest).read_text()) if i['split']==split];rows=[]
    require_sounding_labels(items)
    for it in items:
        p=Path(predictions)/(it['id']+'.json')
        if not p.exists():raise FileNotFoundError(f'Missing prediction: {p}')
        obj=json.loads(p.read_text());est=obj['notes'];config=obj.get('config',{'engine':'NoteCRNN legacy prediction'})
        rows.append({k:it[k] for k in ('id','dataset','part','group')}|{'reference_notes':len(it['notes']),'estimated_notes':len(est),**note_metrics(it['notes'],est),'offset_f1':note_metrics(it['notes'],est,True)['f1'],'inference_seconds':obj.get('seconds')})
    result={'engine':config.get('engine','unknown'),'config':config,'split':split,'tracks':rows,'macro_note_f1':float(np.mean([r['f1'] for r in rows])),'macro_offset_f1':float(np.mean([r['offset_f1'] for r in rows]))}
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
    pairs=[('a0_val','a1_val'),('a0_test','a1_test'),('a6_sounding_external','a8_sounding_full_external'),('a6_sounding_val','a8_sounding_full_val')]
    for left_name,right_name in pairs:
        a=experiments.get(left_name);b=experiments.get(right_name)
        split=a['split'] if a else None
        if not a or not b:continue
        for part in ('guitar','bass'):
            for condition in ('clean','mixture'):
                left={r['id']:r for r in a['tracks'] if r['part']==part and (r['dataset']=='remix')==(condition=='mixture')};right={r['id']:r for r in b['tracks']}
                if not left:continue
                groups=sorted({r['group'] for r in left.values()});bygroup={g:[right[k]['f1']-r['f1'] for k,r in left.items() if r['group']==g] for g in groups}
                samples=[np.mean([v for g in rng.choice(groups,len(groups),replace=True) for v in bygroup[g]]) for _ in range(1000)]
                diffs.append({'split':split,'part':part,'condition':condition,'left_experiment':left_name,'right_experiment':right_name,'delta_right_minus_left':float(np.mean([v for vs in bygroup.values() for v in vs])),'bootstrap_95_percent_interval':np.quantile(samples,[.025,.975]).tolist(),'independent_groups':len(groups),'caution':'small group count; one seed only; exploratory, not proof of superiority'})
    (dest/'comparison.json').write_text(json.dumps({'rows':rows,'paired_differences':diffs},indent=2))
    text=['# 学習・既存モデルの比較','', 'Note F1は音程（オクターブ込み）と発音時刻±50msの一致。値は0〜1です。Offset F1は終了時刻も評価します。Frame F1は20msごとの鳴っている音の一致です。完全一致率は、正解または予測に音がある時刻で音集合がすべて合う割合です。','', '|実験|分割|楽器|条件|曲/群|Note F1|Offset F1|Frame F1|音集合完全一致|','|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f'|{r["experiment"]}|{r["split"]}|{r["part"]}|{r["condition"]}|{r["tracks"]}/{r["source_groups"]}|{r["f1"]:.3f}|{r["offset_f1"]:.3f}|{r["frame_f1"]:.3f}|{r["active_exact_match"]:.3f}|')
    external=all(r['dataset'].startswith('idmt_') for exp in experiments.values() for r in exp['tracks'])
    text+=['','## 解釈上の制限','']
    if external:
        text+=['外部評価はIDMTのベース17録音とギター5録音（4曲群）です。我々の学習・閾値選択には使っていません。単独楽器の短い録音で、自然なバンドミックスの評価ではありません。奏者・楽器は各セット1種類であり、22人の演奏者がいる意味ではありません。','']
    else:
        text+=['混合音は同じ分割内の他楽器を0dB RMSで重ねた制御実験で、自然なバンド全体や音源分離後のアーティファクトの評価ではありません。BabySlakhは20曲の合成音源で、独立群が少ない試作用セットです。','']
    text+=['Basic PitchとMuScriptorは公開の学習済み重み、独自CRNNは本プロジェクトで学習した重みです。事前学習量・モデル規模は等しくなく、公開モデルの事前学習全体との非重複は保証できません。共通の出力音域B0〜E6、最短音符30msを使います。独自モデルの閾値は検証用データで選び、外部評価では変更しません。','',
           'Slakhのベースは実音に合わせて1オクターブ補正し、例外の音源設定は保持しています。補正前の評価を正しい音高の成績として比較しないでください。損失曲線の上下は認識精度を意味しません。','',
           '同時音数別Frame F1は正解の異なるMIDI音高数（1、2、3以上）で集計します。同音ユニゾンの弦数や奏法は識別しません。TABの弾きやすさ、複雑コード、編集時間、実バンドの完成度は別の評価が必要です。','',
           '追加データで改善しない条件もそのまま記載します。差の区間は元曲群単位の対応付きブートストラップ1000回です。奏者と群が少なくseedも1つなので、統計的優位性の証明には使えません。']
    (dest/'evaluation.md').write_text('\n'.join(text)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(experiments);fig,axes=plt.subplots(1,2,figsize=(14,4.5),layout='constrained')
    for ax,part in zip(axes,('guitar','bass')):
        use=[r for r in rows if r['part']==part and r['split']==('test' if any(v['split']=='test' for v in experiments.values()) else 'val')]
        for j,condition in enumerate(('clean','mixture')):
            vals=[next((r['f1'] for r in use if r['experiment']==l and r['condition']==condition),np.nan) for l in labels]
            if not any(np.isfinite(v) for v in vals):continue
            ax.bar(np.arange(len(labels))+(j-.5)*.35,vals,width=.35,label='solo' if external and condition=='clean' else condition)
        display_labels=[('CRNN control' if l.startswith('a6_') else 'CRNN +9 real' if l.startswith('a7_') else 'CRNN +32 real' if l.startswith('a8_') else 'BP default' if l.startswith('basic_pitch_defaults') else 'BP 30ms' if l.startswith('basic_pitch') else 'MuScriptor' if l.startswith('muscriptor') else l) for l in labels]
        ax.set(xticks=np.arange(len(labels)),xticklabels=display_labels,ylim=(0,1),ylabel='Note F1',title=part);ax.tick_params(axis='x',rotation=25);ax.legend()
    fig.savefig(dest/'note_f1.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');found=False
    curve_runs=[l.removesuffix('_external').removesuffix('_val') for l in labels if l.startswith(('a0','a1','a6','a7','a8'))]
    for run in dict.fromkeys(curve_runs):
        history=Path(files[0]).parent.parent/'runs'/run/'history.json'
        if history.exists():
            values=json.loads(history.read_text());found=True
            for key,style in [('train_bce','-'),('val_bce','--')]:
                ax.plot([r['epoch']*100 for r in values],[r[key] for r in values],style,label=run+' '+key)
    if found:
        ax.set(xlabel='Optimizer updates (100 per reported epoch)',ylabel='Weighted BCE',title='Learning curves (loss, not transcription accuracy)');ax.legend()
        fig.savefig(dest/'learning_curves.png',dpi=160)
    plt.close(fig)
    print('Saved',dest)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('baseline');a.add_argument('--manifest',required=True);a.add_argument('--predictions',required=True);a.add_argument('--split',required=True);a.add_argument('--out',required=True)
    s=sub.add_parser('summarize');s.add_argument('files',nargs='+');s.add_argument('--out',default='reports/presentation')
    a=p.parse_args()
    if a.cmd=='baseline':baseline(a.manifest,a.predictions,a.split,a.out)
    else:summarize(a.files,a.out)
if __name__=='__main__':main()
