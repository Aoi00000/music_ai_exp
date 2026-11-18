"""Freeze validation-selected LoRA and only then unlock the final evaluation."""
from pathlib import Path
import argparse,json,torch
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import aggregate

REPORT=Path('reports/yourmt3_lora')

def select(expected_runs):
    dest=REPORT/'monitor_selection.json'
    if dest.exists():raise ValueError('Monitor selection already frozen')
    trials=[json.loads(p.read_text()) for p in sorted(Path('runs/yourmt3_lora').glob('*/completed.json'))]
    if expected_runs<2 or len(trials)!=expected_runs:raise ValueError('Complete all planned learning runs before selection')
    if any(not (p.parent/'completed.json').exists() for p in Path('runs/yourmt3_lora').glob('*/config.json')):
        raise ValueError('A learning run remains incomplete')
    best=max((t['best'] for t in trials),key=lambda x:x['validation_monitor']['score'])
    protocol=json.loads((REPORT/'protocol.json').read_text())
    base=json.loads(Path('reports/yourmt3/validation_comparison.json').read_text())
    monitor=[r for r in base['tracks'] if r['id'] in protocol['monitor_ids']]
    baseline=aggregate([(r,r['metrics']['YourMT3']) for r in monitor])
    save(dest,{'research_selected':best,'base_monitor':baseline,'completed_runs':len(trials),
        'trials':[t['best'] for t in trials],'selection':'Monitor dataset macro score only',
        'test_used_for_selection':False,'full_validation_required':True})
    print('LORA_MONITOR_SELECTED',best['run'],best['step'],flush=True)

def freeze():
    dest=REPORT/'selection.json'
    if dest.exists():raise ValueError('Final selection already frozen')
    monitor=json.loads((REPORT/'monitor_selection.json').read_text());best=monitor['research_selected']
    full=json.loads((Path('runs/yourmt3_lora')/best['run']/'full_validation/comparison.json').read_text())
    if full['selected']!=best:raise ValueError('Full validation used a different adapter')
    baseline=json.loads(Path('reports/yourmt3/validation_comparison.json').read_text())['summary']['YourMT3']
    if sha(best['checkpoint'])!=best['checkpoint_sha256']:raise ValueError('Adapter changed')
    package=Path('models/yourmt3_lora_review.pt');package.parent.mkdir(exist_ok=True)
    # Preserve exact serialized adapter bytes for a portable hash identity.
    package.write_bytes(Path(best['checkpoint']).read_bytes())
    state=torch.load(package,map_location='cpu',weights_only=True)
    gates=json.loads(Path('reports/candidate_v2/protocol.json').read_text())['practical_targets']
    practical=all(v[k]>=limit for v in full['summary']['datasets'].values() for k,limit in gates.items())
    info={'base_sha256':state['base_sha256'],'adapter_sha256':sha(package),'parameters':state['config']['trainable_parameters'],
        'selected':best,'base_validation':baseline,'learned_validation':full['summary'],
        'validation_winner':'LoRA' if full['summary']['score']>baseline['score'] else 'YourMT3',
        'provisional_practical_validation_passed':practical,'automatic_application_approved':False,
        'test_used_for_selection':False,'test_manifest_sha256':sha('data/candidate_v2/manifest.json')}
    save(package.with_suffix('.json'),info);save(dest,info)
    lock={'test_access_enabled':True,'reason':'All candidate and YourMT3 learning/model selection frozen before final scoring',
        'selection_sha256':sha(dest),'candidate_selection_sha256':sha('reports/candidate_v2/selection.json'),
        'test_source_manifest':'data/candidate_v2/manifest.json','test_used_for_selection':False}
    save('reports/global_development_lock.json',lock)
    print('LORA_FINAL_SELECTION_FROZEN',info['validation_winner'],'practical',practical,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','freeze']);p.add_argument('--expected-runs',type=int);a=p.parse_args()
    if a.command=='select':
        if a.expected_runs is None:p.error('--expected-runs is required when selecting')
        select(a.expected_runs)
    else:freeze()
