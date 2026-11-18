"""Frozen temporal results and diagnostic figures; never selects models."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',default='backups/azure');p.add_argument('--out',default='reports/presentation_temporal');a=p.parse_args()
root=Path(a.root);rpt=root/'reports/temporal_v1';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
report=json.loads((rpt/'test_comparison.json').read_text());selection=report['selection']
keys=('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')
with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','cases','method',*keys])
    for ds,e in report['summary'].items():
        for name,m in e['methods'].items():w.writerow([ds,e['cases'],name,*[m[k] for k in keys]])
with (out/'errors.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','method','part','reference_notes','missed_notes','extra_notes','missed_per_100','extra_per_100','wrong_role_overlap','octave_overlap','timing_overlap','piano_overlap','matched_onset_median_ms','known_matched_offset_median_ms'])
    for ds in report['summary']:
        cases=[r for r in report['tracks'] if r['dataset']==ds]
        for method in report['summary'][ds]['methods']:
            for part in ('guitar','bass'):
                d=[r['diagnostics'][method][part] for r in cases];n=sum(r['reference_notes'] for r in d)
                med=lambda key:float(np.median([v for r in d for v in r[key]])) if any(r[key] for r in d) else ''
                missed=sum(r['missed_notes'] for r in d);extra=sum(r['extra_notes'] for r in d)
                w.writerow([ds,method,part,n,missed,extra,100*missed/n if n else '',100*extra/n if n else '',
                    *[sum(r[k] for r in d) for k in ('wrong_role_overlap','octave_error_overlap','late_or_early_same_pitch','piano_overlap')],
                    med('onset_abs_error_ms'),med('offset_abs_error_ms')])
with (out/'polyphony.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','method','part','polyphony','frames','tp','fp','fn','pooled_frame_f1'])
    for ds in report['summary']:
        use=[r for r in report['tracks'] if r['dataset']==ds]
        for method in report['summary'][ds]['methods']:
            for part in ('union','guitar','bass'):
                for level in ('single','double','poly3plus','silence'):
                    values=[r['metrics'][method]['polyphony'][part][level] for r in use];counts=[sum(v[k] for v in values) for k in ('frames','tp','fp','fn')]
                    frames,tp,fp,fn=counts;den=2*tp+fp+fn;w.writerow([ds,method,part,level,*counts,2*tp/den if den else ''])
names={'Mu_audio_bounded':'MuScriptor, audio bounded','Mu_6s_audio_bounded':'6-source stems + MuScriptor, bounded',
       'Mu_ft_audio_bounded':'FT remove drums/vocals + MuScriptor, bounded','A8_on_estimated_stems':'Frozen A8 on estimated stems',
       'temporal_standard':'Temporal, standard data','temporal_hard':'Temporal, hard data',
       'temporal_calibrated':'Temporal, calibrated correction strength'}
datasets=[d for d in ('slakh_new','hard_filobass_guitarset_mix','filobass_guitarset_mix','guitar_techs_amp_new_condition') if d in report['summary']]
labels={'slakh_new':'New Slakh source songs','hard_filobass_guitarset_mix':'Constructed hard real-instrument mixes',
        'filobass_guitarset_mix':'Constructed guitar+bass mixes','guitar_techs_amp_new_condition':'Guitar amp recordings (regression)'}
fig,axes=plt.subplots(4,1,figsize=(14,13),sharex=True);x=np.arange(len(datasets));width=.78/len(names)
for ax,key,title in zip(axes,('note_f1','frame_f1','union_frame_f1','offset_f1'),('Note F1: instrument + pitch + onset','Frame F1: instrument + sounding pitch','Frame F1: guitar+bass pitch union','Offset F1: known reference releases only')):
    for i,(name,label) in enumerate(names.items()):
        ax.bar(x+(i-(len(names)-1)/2)*width,[report['summary'][d]['methods'][name][key] for d in datasets],width,label=label)
    ax.set_ylim(0,1);ax.set_ylabel('F1');ax.set_title(title,loc='left',fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[-1].set_xticks(x,[labels[d]+'\n(n='+str(report['summary'][d]['cases'])+')' for d in datasets],fontsize=9)
handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,ncol=3,loc='upper center',bbox_to_anchor=(.5,.997),frameon=False,fontsize=9)
fig.suptitle('Temporal residual learning: additional songs, constructed mixtures and separators',y=1.02,fontsize=15)
fig.text(.5,.015,'New Slakh songs are new to our own experiments; overlap with public pretraining is unknown.\nOther sets reuse performers/recordings. No claim of natural vocal-band generalization.',ha='center',fontsize=9)
fig.subplots_adjust(top=.90,bottom=.12,hspace=.32);fig.savefig(out/'comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for condition in ('standard','hard'):
    history=json.loads((root/'runs/temporal_v1'/condition/'history.json').read_text())
    ax.plot([r['step'] for r in history],[r['loss'] for r in history],label=condition)
ax.set(xlabel='Training update',ylabel='Mean loss over 100 minibatches',title='Supervised temporal training loss (not accuracy)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training_loss.png',dpi=170);plt.close(fig)
lines=['# 時系列補正・難しい混合音・分離器比較の結果','',
       'MuScriptorと分離器は固定し、A8由来の音響特徴とMuScriptorの音符を使うCNN＋BiGRUの時系列補正を教師あり学習しました。強化学習ではありません。','',
       '検証で選んだ研究比較用モデル：`'+selection['research_selected']+'`。採用候補：`'+selection['adoption_candidate']+'`。自動適用は無効です。','',
       '|対象|件数|方法|Note F1|楽器込みFrame F1|音高集合F1|Offset F1|','|---|---:|---|---:|---:|---:|---:|']
for ds,e in report['summary'].items():
    for method in names:
        m=e['methods'][method];lines.append('|'+ds+'|'+str(e['cases'])+'|'+names[method]+'|'+f"{m['note_f1']:.4f}|{m['frame_f1']:.4f}|{m['union_frame_f1']:.4f}|{m['offset_f1']:.4f}|")
cal=report['calibration']['selected']
lines+=['','検証のみで追加調整した確認用パッケージ：'+cal['condition']+'、補正強度 '+str(cal['residual_scale'])+'。検証の非悪化条件を満たしておらず、自動適用は無効です。',
        'errors.csvは曲をまとめた音符数、比較表は曲ごとのF1の平均です。両者の集計方法は異なります。各カテゴリは重複し、鍵盤と音が一致しても混入が原因と断定できません。開始・終了の誤差中央値は一致した音符だけの値です。',
        '切出し端で実際の終了が不明な音は、Offset F1と終了誤差から除きました。旧方式の切出し端を含むOffset F1も元JSONに保存しています。',
        'polyphony.csvは単音・2音・3音以上を分けたフレーム集計です。曲平均の比較表とは集計方法が異なります。',
        '音高集合の認識だけで、原演奏の運指や弾けるTABの完成度が確認された意味ではありません。自然な歌入りバンド音源と実際の編集時間は別評価が必要です。']
(out/'evaluation.md').write_text('\n'.join(lines)+'\n');print('EXPORTED',out)
