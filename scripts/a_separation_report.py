"""Export frozen separation/fusion results; no model or threshold selection."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',default='backups/azure');p.add_argument('--out',default='reports/presentation_separation');a=p.parse_args()
root=Path(a.root);folder=root/'reports/separation_v1';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
result_file=folder/'router_conservative_test.json'
if not result_file.exists():result_file=folder/'fusion_test.json'
test=json.loads(result_file.read_text());training=json.loads((folder/'training.json').read_text())
selection=test['selection'];keys=('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')
with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','cases','method',*keys])
    for dataset,entry in test['summary'].items():
        for method,score in entry['methods'].items():w.writerow([dataset,entry['n'],method,*[score[k] for k in keys]])
with (out/'polyphony.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','method','part','reference_polyphony','frames','tp','fp','fn','pooled_frame_f1'])
    for dataset in test['summary']:
        use=[r for r in test['tracks'] if r['dataset']==dataset]
        for method in test['summary'][dataset]['methods']:
            for part in ('union','guitar','bass'):
                for level in ('single','double','poly3plus','silence'):
                    scores=[r['metrics'][method]['polyphony'][part][level] for r in use]
                    counts={k:sum(s[k] for s in scores) for k in ('frames','tp','fp','fn')}
                    den=2*counts['tp']+counts['fp']+counts['fn'];f1=2*counts['tp']/den if den else ''
                    w.writerow([dataset,method,part,level,*[counts[k] for k in ('frames','tp','fp','fn')],f1])
labels={'baby':'Synthetic full mixes','filobass_guitarset_mix':'Constructed guitar+bass mixes',
        'filobass':'Double-bass solo','p3':'Guitar direct input',
        'guitar_techs_amp_new_condition':'Guitar amp microphone'}
names={'M0_original':'Original MuScriptor','M1_no_drums_vocals':'Remove drums/vocals','M2_separated_forced':'Separate + assign stem role',
       'M3_naive_union':'Original + separated, no learning','selected':'Validation-selected event fusion'}
if 'router_selection' in test:names['combined_selected']='Validation-selected routing'
else:names['fixed_learned_additions']='Learned additions (fixed)'
methods=list(names);datasets=list(test['summary']);x=np.arange(len(datasets));width=.78/len(methods)
fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
for ax,key,title in zip(axes,('note_f1','frame_f1','union_frame_f1'),('Note F1: instrument + pitch + onset','Frame F1: instrument + sounding pitch','Frame F1: sounding pitches, guitar+bass union')):
    for i,method in enumerate(methods):
        ax.bar(x+(i-(len(methods)-1)/2)*width,[test['summary'][d]['methods'][method][key] for d in datasets],width,label=names[method])
    ax.set_ylim(0,1);ax.set_ylabel('F1');ax.set_title(title,loc='left',fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[-1].set_xticks(x,[labels.get(d,d)+'\n(n='+str(test['summary'][d]['n'])+')' for d in datasets],fontsize=9)
handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',ncol=3,bbox_to_anchor=(.5,.995),frameon=False)
fig.suptitle('Frozen MuScriptor: separation and learned event fusion',y=1.025,fontsize=15)
fig.text(.5,.015,'Repeated regression sets. Amp microphone is a new audio condition of the same P3 performances.\nNo natural vocal-band generalization claim. Test results did not select the system.',ha='center',fontsize=9)
fig.subplots_adjust(top=.91,bottom=.12,hspace=.3);fig.savefig(out/'comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
curve=training['curve'];fig,ax=plt.subplots(figsize=(7,3));ax.plot([r['epoch'] for r in curve],[r['loss'] for r in curve])
if selection['selected'].get('epoch'):
    ax.axvline(selection['selected']['epoch'],color='tab:orange',linestyle='--',label='Validation-selected checkpoint');ax.legend()
ax.set(xlabel='Full-batch update',ylabel='Weighted binary cross entropy',title='Event fusion training loss (not recognition accuracy)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training_loss.png',dpi=170);plt.close(fig)
lines=['# 音源分離・補正学習の評価結果','',
    '採用判断は検証用データだけで固定しました。テストの結果を見た再選択は行っていません。',
    '既存の評価セットは再評価です。アンプ録音は同じP3演奏の未使用音声条件であり、新しい独立曲／奏者ではありません。','',
    '候補補正の選択：`'+selection['selected']['method']+'`。自動適用は無効。',
    '自作EventFusion：'+str(training['parameters'])+'パラメータ、'+str(training['train_records'])+'学習区間、'+str(training['train_candidates'])+'候補。MuScriptorと分離器の重みは固定。','',
    '|評価対象|区間数|方法|Note F1|楽器込みFrame F1|音高集合Frame F1|','|---|---:|---|---:|---:|---:|']
for d,e in test['summary'].items():
    for method in methods:
        m=e['methods'][method];lines.append('|'+d+'|'+str(e['n'])+'|'+names[method]+'|'+f"{m['note_f1']:.4f}|{m['frame_f1']:.4f}|{m['union_frame_f1']:.4f}|" )
lines+=['','音高はオクターブを区別します。音高集合のF1は、楽器名だけを変える補正と、鳴っている音そのものの改善を区別するための指標です。',
    '単音・2音・3音以上のフレーム集計はpolyphony.csvです。曲をまとめたTP/FP/FNから算出しており、comparison.csvの曲平均とは集計が異なります。',
    'Oracle（正解ステム）の結果は元JSON／CSVにあり、実運用の成績ではありません。コード名・運指・TAB編集時間は別評価です。']
if 'router_selection' in test:
    routing=test['router_selection'];lines+=['','処理経路の選択：`'+routing['combined_selection']+'`。決定木は音響・認識特徴から経路ごとの改善量を学習します。',
        '最初の平均改善量を学ぶ条件は検証で採用できず、Note／楽器込みFrame／音高集合Frameの最小改善量を学ぶ条件を追加しました。いずれも評価結果を見る前に固定しています。',
        '```text',routing['selected'].get('readable_tree','Identity'),'```']
(out/'evaluation.md').write_text('\n'.join(lines)+'\n')
print('EXPORTED',out)
