"""Bounded Mu LoRA pilot, only through a_gpu_session.py cleanup wrapper."""
from pathlib import Path
import json,subprocess,time
from scripts.a_candidate_data import sha

def main():
    root=Path(__file__).resolve().parents[1];p=json.load(open('data/mu_adapt_v1/protocol.json'))
    for f,h in p['implementation_sha256'].items():
        if sha(f)!=h:raise ValueError('Frozen Mu implementation changed')
    subprocess.run([str(root/'.venv-yourmt3/bin/python'),'scripts/a_azure.py','start'],check=True)
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','aiexp']
    for _ in range(24):
        try:
            if subprocess.run([*ssh,'true'],capture_output=True,timeout=12).returncode==0:break
        except subprocess.TimeoutExpired:pass
        time.sleep(5)
    else:raise RuntimeError('SSH unavailable')
    shell='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10'
    for f in ['band2tab/mu_adapter.py','band2tab/augment.py','scripts/a_mu_adapt.py','scripts/a_mu_adapt_eval.py']:
        subprocess.run(['rsync','-az','-e',shell,f,'aiexp:music_ai_exp/'+f],check=True)
    subprocess.run([*ssh,'mkdir -p music_ai_exp/data/mu_adapt_v1'],check=True)
    subprocess.run(['rsync','-az','-e',shell,'data/mu_adapt_v1/','aiexp:music_ai_exp/data/mu_adapt_v1/'],check=True)
    prefix='cd music_ai_exp && export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 && '
    for cmd,label in [('scripts.a_mu_adapt train','train'),('scripts.a_mu_adapt_eval monitor','monitor'),('scripts.a_mu_adapt_eval review','review')]:
        subprocess.run([*ssh,prefix+'.venv-mus/bin/python -u -m '+cmd+' > logs/mu_adapt_'+label+'_20261120.log 2>&1'],check=True,timeout=1800);print('MU_ADAPT_STAGE_COMPLETE',label,flush=True)
if __name__=='__main__':main()
