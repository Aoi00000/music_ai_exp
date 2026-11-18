"""Rebuild presentation metrics and figures from verified, complete evaluations."""
from pathlib import Path
from collections import defaultdict
import json,os
import numpy as np
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate
from band2tab.score import read_midi


def main():
    tags={'native_base_goat':'YourMT3 base','amp5_preserve_cpu':'Previous selected LoRA','band_mixture_cpu':'Band-trained parent',
          'augmentation_cpu':'Augmentation LoRA','augmentation_retention_cpu':'Augmentation + retention'}
    common={};ids=None
    for tag,name in tags.items():
        path=Path('reports/yourmt3/mac_validation')/tag/'comparison.json';record=json.load(open(path))
        if record['cases']!=67 or not record['validation_only']:raise ValueError('Complete common validation required')
        if ids is not None and record['case_ids']!=ids:raise ValueError('Different validation cases')
        ids=record['case_ids'];common[tag]=dict(label=name,summary=record['summary'],report_sha256=sha(path),adapter_sha256=record['adapter_sha256'])
    held_path=Path('reports/stem_refiner_holdout_v1/comparison.json');held=json.load(open(held_path));selection=json.load(open('reports/stem_refiner_v1/selection.json'))
    mu=json.load(open('reports/band_hybrid_v1/mu_mac/comparison.json'))
    bass_rows=json.load(open('reports/idmt_bass_lines/manifest.json'));regression={}
    for tag,label in {'raw_base':'YourMT3 base','selected_learned':'Previous LoRA','augmentation_regression':'Augmentation LoRA','augmentation_retention_regression':'Augmentation + retention'}.items():
        folder=Path('reports/idmt_bass_lines')/tag;record=json.load(open(folder/'comparison.json'));known=[];by=defaultdict(list)
        for row in bass_rows:
            parts=read_midi(folder/'midi'/(row['id']+'.mid'),force_part='bass');score=metrics(row,parts);known.append((row,score))
            technique='+'.join(sorted({n['plucking'] for n in row['parts']['bass']}));by[technique].append((row,score))
        regression[tag]=dict(label=label,strict=record['summary'],source_known=aggregate(known),by_plucking_source_known={k:aggregate(v) for k,v in by.items()})
    mc=json.load(open('reports/idmt_bass_lines/mu_control/comparison.json'));by=defaultdict(list)
    lookup={r['id']:r for r in bass_rows}
    for t in mc['tracks']:
        row=lookup[t['id']];technique='+'.join(sorted({n['plucking'] for n in row['parts']['bass']}));by[technique].append((row,t['source_known']['metrics']))
    regression['mu_control']=dict(label='MuScriptor medium',strict=mc['summary']['strict'],source_known=mc['summary']['source_known'],by_plucking_source_known={k:aggregate(v) for k,v in by.items()})
    by_part={p:{method:{k:float(np.mean([r[method]['by_part'][p][k] for r in held['tracks']])) for k in ('note_f1','frame_f1')} for method in ('baseline','learned')} for p in ('guitar','bass')}
    payload=dict(date='2026-11-19',common_development_validation=common,
        mu_development_controls=mu['summary'],stem_development=dict(control=selection['control'],learned=selection['best']['summary'],
            run=selection['best']['run'],policy=selection['best']['policy'],policy_choices=selection['policy_choices_evaluated']),
        frozen_holdout={k:held[k] for k in ('baseline','learned','delta','paired_group_bootstrap','groups','conditions','final_policy','source_limitations','natural_band_recording')},
        holdout_by_part_descriptive=by_part,holdout_report_sha256=sha(held_path),automatic_application_approved=False,
        seen_bass_regression=regression,seen_bass_regression_used_to_choose_models=False,
        training='Supervised deep learning; LoRA plus frozen-Mu residual correction; no reinforcement learning',
        limitations=['Repeated 67-case/16-case development validation','Seven held-out guitar groups, constructed mixtures; bass recordings seen at other times',
                    'Frame improvement accompanies note-onset degradation','No universal superiority or practical score accuracy demonstrated'])
    dest=Path('docs/measurements/learning_20261119.json');save(dest,payload)
    table=['| 同一67条件・開発検証 | 音符F1 | Frame F1 | 全音一致 | 終了込みF1 |','|---|---:|---:|---:|---:|']
    for c in common.values():
        s=c['summary'];table.append('| '+c['label']+' | '+' | '.join(f'{s[k]:.4f}' for k in ('note_f1','frame_f1','active_exact_match','offset_f1'))+' |')
    doc=Path('docs/EXPERIMENT_20261119.md');text=doc.read_text();start='<!-- COMMON_TABLE_START -->';end='<!-- COMMON_TABLE_END -->'
    before,rest=text.split(start);_,after=rest.split(end);doc.write_text(before+start+'\n'+'\n'.join(table)+'\n'+end+after)
    table=['| 既知の実ベース17件・再検査 | GM分類のまま音符F1 | ベース既知の音符F1 | ベース既知のFrame F1 | ミュート4件・ベース既知音符F1 |','|---|---:|---:|---:|---:|']
    for c in regression.values():table.append(f"| {c['label']} | {c['strict']['note_f1']:.4f} | {c['source_known']['note_f1']:.4f} | {c['source_known']['frame_f1']:.4f} | {c['by_plucking_source_known']['MU']['note_f1']:.4f} |")
    text=doc.read_text();start='<!-- REGRESSION_TABLE_START -->';end='<!-- REGRESSION_TABLE_END -->';before,rest=text.split(start);_,after=rest.split(end);doc.write_text(before+start+'\n'+'\n'.join(table)+'\n'+end+after)
    os.environ.setdefault('MPLCONFIGDIR',str(Path('vendor/matplotlib_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'band2tab-20261119'})
    fig,grid=plt.subplots(2,2,figsize=(12,8));axes=grid.flatten();colors=['#496a8a','#d28442'];x=np.arange(2)
    for ax,d,title in ((axes[0],selection,'Development: 16 reused cases'),(axes[1],held,'Frozen holdout: 7 paired groups')):
        a=d.get('control',d.get('baseline'));b=d.get('best',{}).get('summary',d.get('learned'))
        keys=('note_f1','frame_f1');ax.bar(x-.18,[a[k] for k in keys],.35,color=colors[0],label='Mu + separation')
        ax.bar(x+.18,[b[k] for k in keys],.35,color=colors[1],label='+ learned correction')
        for i,k in enumerate(keys):
            ax.text(i-.18,a[k]+.007,f'{a[k]:.4f}',ha='center',fontsize=9);ax.text(i+.18,b[k]+.007,f'{b[k]:.4f}',ha='center',fontsize=9)
        ax.set(xticks=x,xticklabels=['Note F1','Frame F1'],ylim=(0,1),title=title,ylabel='F1');ax.grid(axis='y',alpha=.15)
    deltas=held['paired_group_bootstrap']['all'];keys=('note_f1','frame_f1','union_frame_f1');values=[100*deltas[k]['mean_delta'] for k in keys]
    lo=[values[i]-100*deltas[k]['percentile_95'][0] for i,k in enumerate(keys)];hi=[100*deltas[k]['percentile_95'][1]-values[i] for i,k in enumerate(keys)]
    axes[2].errorbar(values,np.arange(3),xerr=[lo,hi],fmt='o',color=colors[1],capsize=5);axes[2].axvline(0,color='#666',linestyle='--')
    axes[2].set(yticks=np.arange(3),yticklabels=['Note','Frame','Pitch-set'],xlabel='Learned − baseline (percentage points)',title='Holdout: paired bootstrap 95% interval')
    axes[2].invert_yaxis();axes[0].legend(loc='lower left',fontsize=8)
    order=('raw_base','selected_learned','augmentation_regression','augmentation_retention_regression','mu_control')
    values=[regression[k]['source_known']['note_f1'] for k in order]
    axes[3].bar(np.arange(5),values,color=[colors[0],colors[0],colors[1],colors[1],'#609474'])
    axes[3].set(xticks=np.arange(5),xticklabels=['Your base','Old LoRA','Aug. LoRA','Aug. + KD','Mu medium'],ylim=(0,1.05),ylabel='Note F1',title='Seen bass regression (17 clips): known bass role')
    axes[3].tick_params(axis='x',labelsize=9)
    for i,v in enumerate(values):axes[3].text(i,v+.02,f'{v:.3f}',ha='center',fontsize=9)
    fig.suptitle('Learning experiments: bass augmentation and frozen-Mu correction',fontsize=14)
    fig.text(.02,.02,'Top/left: constructed guitar + bass mixtures; same separator and Mu predictions. Bottom/right: previously inspected solo bass regression.',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(dest.with_suffix('.png'),dpi=170);fig.savefig(dest.with_suffix('.svg'),metadata={'Date':None});plt.close(fig)
    p=dest.with_suffix('.svg');p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    print('LEARNING_REPORT_READY',str(dest),flush=True)


if __name__=='__main__':main()
