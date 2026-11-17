"""CSV and shareable plots from frozen JSON; no model selection or fitting."""
from pathlib import Path
import argparse,csv,json

p=argparse.ArgumentParser();p.add_argument('--root',default='backups/azure');p.add_argument('--out',default='reports/presentation_duration');a=p.parse_args()
root=Path(a.root);dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
report=json.loads((root/'reports/duration_test_comparison_full_output.json').read_text())
selection=json.loads((root/'reports/duration_selection.json').read_text())
rows=[]
for evaluation,methods in report['summary'].items():
    tracks=[r for r in report['tracks'] if r['evaluation']==evaluation]
    for method,scores in methods.items():
        rows.append({'evaluation':evaluation,'method':method,'recordings':len(tracks),'source_groups':len({r['group'] for r in tracks}),
                     'used_to_fit_current_weights_or_thresholds':False,'previously_inspected':evaluation.startswith('previously_'),**scores})
with (dest/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
text=['# Duration-aware refinement comparison','',
      'Canonical full-output scoring: hallucinations after full recordings are counted. Only fixed FiloBass excerpts have 0.25-second edge guards.',
      'Held-out players: one P3 guitarist and two FiloBass bassists. Constructed mixes are not natural bands. Public-model pretraining overlap is unknown.','',
      '|Evaluation|Method|Recordings/groups|Note F1|Frame F1 (instrument)|Frame F1 (pitch union)|',
      '|---|---|---:|---:|---:|---:|']
for r in rows:text.append(f"|{r['evaluation']}|{r['method']}|{r['recordings']}/{r['source_groups']}|{r['note_f1']:.4f}|{r['frame_f1']:.4f}|{r['union_frame_f1']:.4f}|")
text+=['','Selected on validation only: '+report['selected_by_validation'],
       'Automatic application remains disabled. Small guitar and historical synthetic-mix regressions remain.',
       'Both new conditions use 150 full-batch updates; expanded data also increases example presentations and compute. One seed; no statistical superiority claim.']
(dest/'evaluation.md').write_text('\n'.join(text)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
methods=('muscriptor','v1_onset','duration_small','duration_expanded')
labels=('MuScriptor','Old onset refiner','Duration / small','Duration / expanded')
colors=('#687883','#bc5555','#92b7ac','#087f8c')
domains=('new_p3_player','new_filobass_player_solo','new_filobass_player_constructed_mix')
names=('Guitar P3\nn=10, 1 player','Bass solo\nn=7, 2 players','Constructed mix\nn=7, 2 bassists')
fig,axes=plt.subplots(1,3,figsize=(13.8,5.4))
for ax,metric,title in zip(axes,('note_f1','frame_f1','union_frame_f1'),('Note F1 (instrument + pitch)','Frame F1 (instrument + pitch)','Frame F1 (pitch union)')):
    for j,(method,label,color) in enumerate(zip(methods,labels,colors)):
        values=[report['summary'][d][method][metric] for d in domains]
        ax.bar(np.arange(3)+(j-1.5)*.18,values,width=.18,label=label,color=color)
    ax.set(xticks=np.arange(3),xticklabels=names,ylim=(0,1),title=title,ylabel='F1');ax.tick_params(axis='x',labelsize=8)
fig.subplots_adjust(left=.05,right=.995,bottom=.23,top=.84,wspace=.22)
fig.legend(*axes[0].get_legend_handles_labels(),fontsize=9,loc='lower center',bbox_to_anchor=(.5,.065),ncol=4,frameon=False)
fig.suptitle('MuScriptor + learned refinement: players held out from refiner training',fontsize=13,y=.98)
fig.text(.5,.025,'No new natural-band recordings. Public-model pretraining overlap is unknown.',ha='center',fontsize=9,color='#46545e')
fig.savefig(dest/'heldout_comparison.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for condition,color in [('duration_small','#687883'),('duration_expanded','#087f8c')]:
    history=json.loads((root/'runs'/condition/'history.json').read_text())
    ax.plot([r['epoch'] for r in history],[r['loss'] for r in history],label=condition,color=color)
ax.set(xlabel='Full-batch optimizer updates',ylabel='Training soft-target cross entropy',title='Training fit; loss is not transcription accuracy')
ax.legend();fig.savefig(dest/'training_loss.png',dpi=170);plt.close(fig)
print('Saved frozen result table and plots to',dest)
