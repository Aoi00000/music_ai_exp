"""Bounded Azure batch; always invoke through the independent cleanup runner."""
from pathlib import Path
import subprocess,time,json
from scripts.a_candidate_data import sha

def main():
    root=Path(__file__).resolve().parents[1]
    protocol=json.load(open('data/keep_cycle_v1/protocol.json'))
    for p,h in protocol['implementation_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen implementation changed')
    subprocess.run([str(root/'.venv-yourmt3/bin/python'),'scripts/a_azure.py','start'],check=True)
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','aiexp']
    for _ in range(24):
        try:
            if subprocess.run([*ssh,'true'],capture_output=True,timeout=12).returncode==0:break
        except subprocess.TimeoutExpired:pass
        time.sleep(5)
    else:raise RuntimeError('SSH unavailable after VM start')
    files=['band2tab/keep.py','band2tab/addition.py','band2tab/stem_refiner.py','scripts/a_keep_cycle.py']
    for f in files:subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10',f,'aiexp:music_ai_exp/'+f],check=True)
    subprocess.run([*ssh,'mkdir -p music_ai_exp/data/keep_cycle_v1'],check=True)
    subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10','data/keep_cycle_v1/','aiexp:music_ai_exp/data/keep_cycle_v1/'],check=True)
    prefix='cd music_ai_exp && export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 && '
    command=prefix+'.venv-mus/bin/python -u -m scripts.a_keep_cycle train --device cuda > logs/keep_cycle_v1_train.log 2>&1 && .venv-mus/bin/python -u -m scripts.a_keep_cycle select > logs/keep_cycle_v1_select.log 2>&1'
    subprocess.run([*ssh,command],check=True,timeout=1800)
    print('KEEP_AZURE_BATCH_COMPLETE',flush=True)

if __name__=='__main__':main()
