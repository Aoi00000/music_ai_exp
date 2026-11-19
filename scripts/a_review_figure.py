"""Standalone measured figures; never join different datasets into one curve."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('vendor/mpl_cache_mac').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    data=json.load(open('docs/measurements/review_20261119.json'));fig,axes=plt.subplots(2,2,figsize=(8,8.4),layout='constrained')
    panels=[('Development band\n16 conditions',data['octave_regression']['development']),
        ('Native Mu configuration\nSame 16 conditions',data['conditioning']),
        ('Targeted data example\n6 jazz-guitar recordings',{'Mu+condition':data['jazz_mu']['val_conditioned'],'No extra data':data['addition_jazz_v3_control']['selected']['summary']['datasets']['jazz_guitarset'],'+12 recordings':data['addition_jazz_v3']['selected']['summary']['datasets']['jazz_guitarset']}),
        ('Previously inspected regression\n14 conditions',data['octave_regression']['known_regression'])]
    labels={'baseline':'Mu','octave_only':'Octave','addition_only':'Addition','combined':'Both','unconditioned':'None','conditioned_cfg1':'CFG 1','conditioned_cfg2':'CFG 2'}
    for ax,(title,rows) in zip(axes.flat,panels):
        x=np.arange(len(rows));width=.25
        for j,(metric,name,col) in enumerate([('note_f1','Note F1','#285878'),('frame_f1','Frame F1','#439f9a'),('active_exact_match','Exact set','#dca55b')]):
            bars=ax.bar(x+(j-1)*width,[v[metric] for v in rows.values()],width,label=name,color=col)
            ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3,rotation=90)
        ax.set_xticks(x,[labels.get(k,k) for k in rows],rotation=20,ha='right',fontsize=8);ax.set_ylim(0,1.06);ax.set_ylabel('Score (higher is better)',fontsize=9);ax.tick_params(axis='y',labelsize=8);ax.set_title(title,fontsize=10,pad=12);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    axes[0,0].legend(loc='upper left',fontsize=7,ncol=3);fig.suptitle('Supervised learning on frozen transcription models\nDevelopment / known regression, not a new final test',fontsize=11)
    for ext in ('png','svg'):fig.savefig('docs/measurements/review_20261119.'+ext,dpi=180)
    svg=Path('docs/measurements/review_20261119.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig);print('REVIEW_FIGURE_READY')


if __name__=='__main__':main()
