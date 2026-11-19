"""Execute paired augmentation and corroboration experiments, then clean up."""
from pathlib import Path
import subprocess,time,json
from scripts.a_candidate_data import sha

def main():
    root=Path(__file__).resolve().parents[1]
    files=['band2tab/keep.py','band2tab/addition.py','band2tab/stem_refiner.py','band2tab/keep_agreement.py',
        'scripts/a_keep_cycle.py','scripts/a_keep_policy_cycle.py','scripts/a_error_taxonomy.py',
        'scripts/a_keep_amp_cycle.py','scripts/a_keep_agreement_cycle.py','scripts/a_feedback_audit.py']
    for folder in ('keep_amp_v3','keep_agreement_v4'):
        protocol=json.load(open(Path('data')/folder/'protocol.json'))
        for f,h in protocol['implementation_sha256'].items():
            if sha(f)!=h:raise ValueError('Frozen implementation changed')
    subprocess.run([str(root/'.venv-yourmt3/bin/python'),'scripts/a_azure.py','start'],check=True)
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','aiexp']
    for _ in range(24):
        try:
            if subprocess.run([*ssh,'true'],capture_output=True,timeout=12).returncode==0:break
        except subprocess.TimeoutExpired:pass
        time.sleep(5)
    else:raise RuntimeError('SSH unavailable after start')
    remote_shell='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10'
    for f in files:subprocess.run(['rsync','-az','-e',remote_shell,f,'aiexp:music_ai_exp/'+f],check=True)
    for folder in ('keep_amp_v3','keep_agreement_v4'):
        subprocess.run([*ssh,'mkdir -p music_ai_exp/data/'+folder],check=True)
        subprocess.run(['rsync','-az','-e',remote_shell,'data/'+folder+'/','aiexp:music_ai_exp/data/'+folder+'/'],check=True)
    prefix='cd music_ai_exp && export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 HF_HUB_DISABLE_TELEMETRY=1 && '
    # Preflight imports before starting an expensive pass.
    subprocess.run([*ssh,prefix+".venv-mus/bin/python -c 'import muscriptor,demucs,librosa; import scripts.a_keep_amp_cycle,scripts.a_keep_agreement_cycle'"],check=True)
    for command,log in [('scripts.a_keep_amp_cycle infer','amp_infer'),('scripts.a_keep_amp_cycle cache','amp_cache'),
        ('scripts.a_keep_amp_cycle train','amp_train'),('scripts.a_keep_amp_cycle select','amp_select'),
        ('scripts.a_keep_agreement_cycle train','agreement_train'),('scripts.a_keep_agreement_cycle select','agreement_select'),
        ('scripts.a_feedback_audit create','feedback_audit')]:
        subprocess.run([*ssh,prefix+'.venv-mus/bin/python -u -m '+command+' > logs/keep_'+log+'_20261119.log 2>&1'],check=True,timeout=1800)
        print('KEEP_FEEDBACK_STAGE_COMPLETE',log,flush=True)
    print('KEEP_FEEDBACK_GPU_BATCH_COMPLETE',flush=True)
if __name__=='__main__':main()
