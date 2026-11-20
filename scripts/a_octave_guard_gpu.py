"""Run only through a_gpu_session.py; stage source/protocol snapshots, not live edits."""
from pathlib import Path
import tempfile,json,subprocess,time,hashlib
from scripts.a_candidate_data import sha
DATA=Path('data/octave_guard_v1')

def main():
    root=Path(__file__).resolve().parents[1];protocol_bytes=(DATA/'protocol.json').read_bytes();p=json.loads(protocol_bytes);expected_protocol=hashlib.sha256(protocol_bytes).hexdigest()
    # Materialize the exact execution input before starting the VM. Later local
    # authoring cannot change the staged implementation or executed protocol.
    with tempfile.TemporaryDirectory(prefix='band2tab-octave-stage-') as stage:
        stage=Path(stage)
        for file,h in p['implementation_sha256'].items():
            source=DATA/'frozen_sources'/file
            if sha(source)!=h or sha(file)!=h:raise ValueError('Frozen source mismatch before VM start: '+file)
            target=stage/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        (stage/'protocol.json').write_bytes(protocol_bytes)
        subprocess.run([str(root/'.venv-yourmt3/bin/python'),'scripts/a_azure.py','start'],check=True)
        ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','aiexp']
        for _ in range(24):
            try:
                if subprocess.run([*ssh,'true'],capture_output=True,timeout=12).returncode==0:break
            except subprocess.TimeoutExpired:pass
            time.sleep(5)
        else:raise RuntimeError('SSH not ready')
        shell='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10'
        for file in p['implementation_sha256']:
            subprocess.run(['rsync','-az','-e',shell,str(stage/file),'aiexp:music_ai_exp/'+file],check=True)
        subprocess.run([*ssh,'mkdir -p music_ai_exp/data/octave_guard_v1'],check=True)
        subprocess.run(['rsync','-az','-e',shell,str(DATA)+'/', 'aiexp:music_ai_exp/data/octave_guard_v1/'],check=True)
        # Override the copied protocol with the immutable pre-start bytes.
        subprocess.run(['rsync','-az','-e',shell,str(stage/'protocol.json'),'aiexp:music_ai_exp/data/octave_guard_v1/protocol.json'],check=True)
        prefix='cd music_ai_exp && export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 && '
        check=".venv-mus/bin/python -c 'import json; from scripts.a_candidate_data import sha; f=\"data/octave_guard_v1/protocol.json\"; assert sha(f)==\""+expected_protocol+"\"; p=json.load(open(f)); assert all(sha(f)==h for f,h in p[\"implementation_sha256\"].items()); assert all(sha(f)==h for f,h in p[\"cache_sha256\"].items()); assert sha(\"data/octave_guard_v1/synthetic.npz\")==p[\"synthetic_sha256\"]; print(\"IMMUTABLE_STAGE_VERIFIED\")'"
        subprocess.run([*ssh,prefix+check],check=True,timeout=180)
        subprocess.run([*ssh,prefix+'.venv-mus/bin/python -u -m scripts.a_octave_guard_cycle train > logs/octave_guard_train_20261120.log 2>&1'],check=True,timeout=1800)
        print('OCTAVE_GUARD_GPU_COMPLETE',expected_protocol,flush=True)
if __name__=='__main__':main()
