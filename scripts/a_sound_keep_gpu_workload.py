"""Azure matched full-mixture extension, inside a_gpu_session cleanup wrapper."""
from pathlib import Path
import subprocess,time,json
from scripts.a_candidate_data import sha

def main():
    root=Path(__file__).resolve().parents[1];protocol=json.load(open('data/sound_keep_v6/protocol.json'))
    for f,h in protocol['implementation_sha256'].items():
        if sha(f)!=h:raise ValueError('Frozen implementation changed')
    subprocess.run([str(root/'.venv-yourmt3/bin/python'),'scripts/a_azure.py','start'],check=True)
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','aiexp']
    for _ in range(24):
        try:
            if subprocess.run([*ssh,'true'],capture_output=True,timeout=12).returncode==0:break
        except subprocess.TimeoutExpired:pass
        time.sleep(5)
    else:raise RuntimeError('SSH unavailable after VM start')
    shell='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10'
    for f in ['band2tab/bass_keep.py','band2tab/keep_agreement.py','band2tab/keep.py','scripts/a_sound_keep_cycle.py','scripts/a_feedback_audit.py']:
        subprocess.run(['rsync','-az','-e',shell,f,'aiexp:music_ai_exp/'+f],check=True)
    subprocess.run([*ssh,'mkdir -p music_ai_exp/data/sound_keep_v6'],check=True)
    subprocess.run(['rsync','-az','-e',shell,'data/sound_keep_v6/','aiexp:music_ai_exp/data/sound_keep_v6/'],check=True)
    prefix='cd music_ai_exp && export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 && '
    subprocess.run([*ssh,prefix+".venv-mus/bin/python -c 'import scripts.a_sound_keep_cycle'"],check=True)
    for command,log in [('scripts.a_sound_keep_cycle train','sound_keep_train'),('scripts.a_sound_keep_cycle select','sound_keep_select'),('scripts.a_feedback_audit create','sound_keep_audit')]:
        subprocess.run([*ssh,prefix+'.venv-mus/bin/python -u -m '+command+' > logs/'+log+'_20261119.log 2>&1'],check=True,timeout=1800)
        print('SOUND_KEEP_STAGE_COMPLETE',log,flush=True)
    print('SOUND_KEEP_GPU_BATCH_COMPLETE',flush=True)
if __name__=='__main__':main()
