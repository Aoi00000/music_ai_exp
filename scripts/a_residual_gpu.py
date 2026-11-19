"""Bounded residual training. Invoke ONLY inside a_gpu_session cleanup wrapper."""
from pathlib import Path
import json,subprocess,time
from scripts.a_candidate_data import sha

def main():
    root=Path(__file__).resolve().parents[1];p=json.load(open('data/residual_v1/protocol.json'))
    for f,h in p['implementation_sha256'].items():
        if sha(f)!=h:raise ValueError('Frozen implementation differs')
    if sha('data/residual_v1/manifest.json')!=p['cache_manifest_sha256']:raise ValueError('Frozen manifest differs')
    subprocess.run([str(root/'.venv-yourmt3/bin/python'),'scripts/a_azure.py','start'],check=True)
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','aiexp']
    for _ in range(24):
        try:
            if subprocess.run([*ssh,'true'],capture_output=True,timeout=12).returncode==0:break
        except subprocess.TimeoutExpired:pass
        time.sleep(5)
    else:raise RuntimeError('SSH not ready')
    shell='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10'
    for f in ['band2tab/residual.py','scripts/a_residual_cycle.py']:
        subprocess.run(['rsync','-az','-e',shell,f,'aiexp:music_ai_exp/'+f],check=True)
    subprocess.run([*ssh,'mkdir -p music_ai_exp/data/residual_v1'],check=True)
    subprocess.run(['rsync','-az','-e',shell,'data/residual_v1/','aiexp:music_ai_exp/data/residual_v1/'],check=True)
    prefix='cd music_ai_exp && export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 && '
    check=".venv-mus/bin/python -c 'import json; from scripts.a_candidate_data import sha; p=json.load(open(\"data/residual_v1/protocol.json\")); assert all(sha(f)==h for f,h in p[\"cache_sha256\"].items()); print(\"RESIDUAL_STAGED_HASHES_OK\")'"
    subprocess.run([*ssh,prefix+check],check=True,timeout=180)
    subprocess.run([*ssh,prefix+'.venv-mus/bin/python -u -m scripts.a_residual_cycle train --device cuda > logs/residual_train_20261120.log 2>&1'],check=True,timeout=1800)
    print('RESIDUAL_GPU_COMPLETE',flush=True)
if __name__=='__main__':main()
