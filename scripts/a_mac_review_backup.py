"""Local verified archive of this review; excludes credentials and large base models."""
from pathlib import Path
import argparse,subprocess,tarfile,json,hashlib
from scripts.a_candidate_data import sha,save

FOLDERS=('runs/release_v2','runs/addition_v1','runs/addition_balanced_v2','runs/addition_jazz_v3_control','runs/addition_jazz_v3','runs/octave_v1','runs/chord_head_v1','runs/chord_head_v2',
    'data/addition_v1','data/addition_jazz_v3','data/jazz_guitar_v1','data/chord_head_v1','data/chord_head_v2',
    'reports/release_v2','reports/release_gate_v3','reports/addition_v1','reports/addition_balanced_v2','reports/addition_jazz_v3_control','reports/addition_jazz_v3','reports/addition_regression_v1','reports/addition_jazz_regression_v1',
    'reports/octave_v1','reports/octave_regression_v1','reports/error_taxonomy_v1','reports/mu_conditioning_v1','reports/jazz_guitar_v1','reports/chord_diagnostic_v1','reports/chord_head_v1','reports/chord_head_v2','reports/tone_v1',
    'reports/idmt_bass_lines/tone_control_regression','reports/idmt_bass_lines/tone_colour_regression','vendor/lv-chordia/weights',
    'results/chord_chart_jazz_demo_20261119','results/chord_chart_learned_demo_20261119','results/native_addition_smoke_20261119','results/native_combined_smoke_20261119')


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='mac_review_20261119');a=p.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('Use a simple archive name')
    root=Path.cwd().resolve();archive=Path('backups')/(a.name+'.tar.gz');ledger=Path('reports')/(a.name+'.json')
    if archive.exists() or ledger.exists():raise ValueError('Choose a new backup name; never overwrite')
    source=[]
    for args in (['git','ls-files','-z'],['git','ls-files','--others','--exclude-standard','-z']):source.extend(Path(x) for x in subprocess.check_output(args).decode().split('\0') if x)
    source.extend(p for folder in FOLDERS for p in Path(folder).rglob('*') if p.is_file())
    source.extend(p for p in Path('models').glob('*') if p.is_file() and p.name.startswith(('guitar_addition','addition_jazz','octave_filter','chord_head')))
    source.extend(Path('reports')/x for x in ('session_state.json','azure_session_cleanup.json','azure_tone_deallocated_20261119.json','chord_model_source.json'))
    files=sorted({p for p in source if p.is_file() and p.resolve().is_relative_to(root) and p.suffix not in ('.pyc','.tmp','.part') and '__pycache__' not in p.parts and 'private' not in p.parts})
    records=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files];archive.parent.mkdir(exist_ok=True)
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p),recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        for r in records:
            with tar.extractfile(r['path']) as f:
                if hashlib.sha256(f.read()).hexdigest()!=r['sha256']:raise ValueError('Archive verification mismatch: '+r['path'])
    save(ledger,dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,verified_files=len(records),all_member_sha256_verified=True,
        scope='This review code, reports, small learned weights and features. Azure raw data/base models are separately verified in backups/azure. No credentials.',files=records))
    print('MAC_REVIEW_BACKUP_VERIFIED',len(records),archive.stat().st_size,flush=True)


if __name__=='__main__':main()
