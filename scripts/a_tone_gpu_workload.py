"""Explicit foreground GPU batch; invoke only through a_gpu_session.py."""
from pathlib import Path
import subprocess,time,json
from scripts.a_candidate_data import save,sha


def main():
    root=Path(__file__).resolve().parents[1]
    ready=json.load(open('reports/tone_v1/protocol.json'))
    if any(sha(path)!=digest for path,digest in ready['implementation_sha256'].items()):raise ValueError('Frozen tone implementation changed')
    subprocess.run([str(root/'.venv-yourmt3/bin/python'),'scripts/a_azure.py','start'],check=True)
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5','aiexp']
    for _ in range(30):
        try:
            check=subprocess.run([*ssh,'true'],capture_output=True,timeout=12)
            if check.returncode==0:break
        except subprocess.TimeoutExpired:pass
        time.sleep(5)
    else:raise RuntimeError('VM started but SSH did not become ready')
    for file in ('band2tab/augment.py','scripts/a_yourmt3_lora.py','scripts/a_tone_evaluate.py','scripts/a_candidate_audit.py'):
        subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes',file,'aiexp:music_ai_exp/'+file],check=True)
    subprocess.run([*ssh,'mkdir -p music_ai_exp/reports/tone_v1'],check=True)
    subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes','reports/tone_v1/protocol.json','aiexp:music_ai_exp/reports/tone_v1/protocol.json'],check=True)
    prefix='cd music_ai_exp && export PYTHONPATH="$PWD/vendor/yourmt3_deps:$PWD/vendor/yourmt3/amt/src:$PWD/vendor/yourmt3:$PWD" OMP_NUM_THREADS=2 WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1 && '
    args=' --scope both --targets wide --rank 8 --lr 0.00002 --steps 600 --seed 17 --anchor-kl 1 --solo-balanced --extra-bass --goat --goat-manifest data/goat_v2/manifest.json --band-manifest data/band_training_v1/manifest.json --augmentation-manifest data/bass_augmentation_v1/manifest.json --init-adapter runs/yourmt3_lora/aug_wide_r8/step_1200.pt --monitor-steps 200 600'
    for name,flag in [('tone_control_r8_600',''),('tone_colour_r8_600',' --tone-augment')]:
        cmd=prefix+'.venv-mus/bin/python -u -m scripts.a_yourmt3_lora train --name '+name+args+flag+' > logs/'+name+'_20261119.log 2>&1'
        subprocess.run([*ssh,cmd],check=True)
    subprocess.run([*ssh,prefix+'.venv-mus/bin/python -u -m scripts.a_tone_evaluate > logs/tone_paired_eval_20261119.log 2>&1'],check=True)
    subprocess.run([*ssh,'cd music_ai_exp && .venv-mus/bin/python scripts/a_candidate_audit.py create > logs/tone_audit_20261119.log 2>&1'],check=True)
    print('TONE_GPU_WORKLOAD_COMPLETE',flush=True)


if __name__=='__main__':main()
