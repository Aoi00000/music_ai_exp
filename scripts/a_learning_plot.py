"""Export a static presentation figure from the frozen development comparison."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=Path('docs/measurements/goat_band_validation_20261118.json'));args=parser.parse_args()
    selection=json.load(open(args.input))
    labels={'native_base_goat':'Pretrained', 'expanded_bass_cpu_goat':'Bass data',
            'expanded_wide_cpu_goat':'Wider LoRA', 'goat_wide_cpu':'GOAT',
            'goat_absent_cpu':'GOAT +\nabsence loss', 'amp5_preserve_cpu':'5 amp views\n+ preservation',
            'band_mixture_cpu':'+ band\nmixtures'}
    rows=selection['compared'] if 'compared' in selection else [{'tag':tag,'summary':summary} for tag,summary in selection['mac_validation'].items()]
    x=np.arange(len(rows));width=.24
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,ax=plt.subplots(figsize=(12,5.7),layout='constrained')
    for offset,key,name,color in ((-width,'note_f1','Note F1 (50 ms)','#2f66a3'),
                                  (0,'frame_f1','Sounding-note F1 (20 ms)','#26a69a'),
                                  (width,'active_exact_match','Exact note set (fraction)','#c89739')):
        values=[r['summary'][key] for r in rows]
        bars=ax.bar(x+offset,values,width,label=name,color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    ax.set_xticks(x,[labels[r['tag']] for r in rows]);ax.set_ylim(0,1.08)
    ax.set_ylabel('F1 / exact-match fraction');ax.set_title('Guitar and bass recognition: development validation',loc='left',fontweight='bold',pad=17)
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.yaxis.grid(True,alpha=.18)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False)
    fig.text(.01,-.03,'Same 67 conditions, Mac CPU float32, dataset-macro mean. Not a final-test result.',fontsize=10)
    folder=Path('reports/figures');folder.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(folder/('goat_band_validation_20261118.'+suffix),dpi=180,bbox_inches='tight')
    svg=folder/'goat_band_validation_20261118.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print('LEARNING_FIGURE_EXPORTED',flush=True)


if __name__=='__main__':main()
