"""Adapt the validation-selected pre-GOAT model with new real electric guitar."""
from pathlib import Path
import argparse,json,subprocess,sys
from scripts.a_candidate_data import save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--absent-weight',type=float,default=0.)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--lr',type=float,default=.00005);a=p.parse_args()
    if not Path('data/goat_v1/manifest.json').exists():raise ValueError('Inspect and freeze GOAT data first')
    trials=[json.loads(p.read_text()) for p in Path('runs/yourmt3_lora').glob('*/completed.json')]
    old=[t for t in trials if not t['config'].get('goat',False)]
    if len(old)!=7:raise ValueError('Complete all seven pre-GOAT comparisons first')
    best=max((t['best'] for t in old),key=lambda x:x['validation_monitor']['score'])
    if sha(best['checkpoint'])!=best['checkpoint_sha256']:raise ValueError('Warm-start adapter changed')
    path=Path('reports/goat/warmstart_selection.json');record={'selected':best,'old_learning_runs':7,
        'selection':'Old 14-condition validation only; GOAT final test never opened','test_used_for_selection':False}
    if path.exists() and json.loads(path.read_text())!=record:raise ValueError('Warm-start selection changed')
    save(path,record)
    cmd=[sys.executable,'scripts/a_yourmt3_lora.py','train','--name',a.name,'--scope','both','--targets','wide',
        '--steps',str(a.steps),'--lr',str(a.lr),'--anchor-kl','.25','--solo-balanced','--extra-bass','--goat',
        '--init-adapter',best['checkpoint'],'--absent-weight',str(a.absent_weight)]
    print('GOAT_LEARNING_START',a.name,'warmstart',best['run'],best['step'],'other-channel weight',a.absent_weight,flush=True)
    subprocess.run(cmd,check=True)

if __name__=='__main__':main()
