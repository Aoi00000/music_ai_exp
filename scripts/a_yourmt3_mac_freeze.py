"""Choose on common full Mac validation, then unlock one fixed final test pair."""
from pathlib import Path
import argparse,json,torch
from scripts.a_candidate_data import save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--tags',nargs='+',required=True);p.add_argument('--expected-runs',type=int,required=True)
    p.add_argument('--development-only',action='store_true',help='Export a review adapter without unlocking final tests');a=p.parse_args()
    destination=Path('reports/yourmt3_lora')/('development_selection.json' if a.development_only else 'selection.json')
    if destination.exists():raise ValueError('Selection already frozen; do not select from final test')
    backup=Path('backups/azure');runs=list((backup/'runs/yourmt3_lora').glob('*/completed.json'))
    if len(runs)!=a.expected_runs:raise ValueError('Back up all planned completed learning runs first')
    if any(not (p.parent/'completed.json').exists() for p in (backup/'runs/yourmt3_lora').glob('*/config.json')):
        raise ValueError('A learning run is incomplete')
    manifest=backup/'data/candidate_v2/manifest.json';goat=backup/'data/goat_v1/manifest.json'
    expected=[r['id'] for source in (manifest,goat) for r in json.loads(source.read_text()) if r['split']=='val']
    results=[]
    for tag in a.tags:
        path=Path('reports/yourmt3/mac_validation')/tag/'comparison.json';r=json.loads(path.read_text())
        if not r['validation_only'] or r['device']!='cpu' or r['dtype']!='float32' or r['case_ids']!=expected:
            raise ValueError('Full common target-device validation required: '+tag)
        if r['manifest_sha256']!=sha(manifest) or r['goat_manifest_sha256']!=sha(goat):raise ValueError('Validation source changed')
        results.append({'tag':tag,'summary':r['summary'],'adapter_sha256':r['adapter_sha256'],
            'report_sha256':sha(path),'base_sha256':r['base']['checkpoint_sha256']})
    if len({r['base_sha256'] for r in results})!=1:raise ValueError('Different frozen bases')
    baseline=[r for r in results if r['adapter_sha256'] is None]
    if len(baseline)!=1:raise ValueError('Exactly one raw-base control required')
    learned=max((r for r in results if r['adapter_sha256']),key=lambda r:r['summary']['score'])
    checkpoint=None
    for report in runs:
        record=json.loads(report.read_text())['best'];path=backup/record['checkpoint']
        if record['checkpoint_sha256']==learned['adapter_sha256']:
            if sha(path)!=learned['adapter_sha256']:raise ValueError('Adapter backup changed')
            checkpoint=path;break
    if checkpoint is None:raise ValueError('Validation adapter is not a completed-run selected checkpoint')
    target=Path('models')/('yourmt3_lora_development.pt' if a.development_only else 'yourmt3_lora_review.pt');target.parent.mkdir(exist_ok=True);target.write_bytes(checkpoint.read_bytes())
    state=torch.load(target,map_location='cpu',weights_only=True)
    gates=json.loads((backup/'reports/candidate_v2/protocol.json').read_text())['practical_targets']
    practical=all(v[k]>=limit for v in learned['summary']['datasets'].values() for k,limit in gates.items())
    winner=max(results,key=lambda r:r['summary']['score'])
    report={'selection_device':'Mac CPU float32','validation_winner':winner['tag'],'research_selected_tag':learned['tag'],
        'base_sha256':learned['base_sha256'],'adapter_sha256':sha(target),'parameters':state['config']['trainable_parameters'],
        'source_run':state['config']['name'],'source_step':state['step'],'compared':results,
        'provisional_practical_validation_passed':practical,'automatic_application_approved':False,
        'local_final_test_enabled':not a.development_only,'development_only':a.development_only,'test_manifest_sha256':sha(manifest),'goat_manifest_sha256':sha(goat),
        'test_used_for_selection':False,'completed_learning_runs':len(runs),
        'scope':'GOAT item split preserves author test but cannot prove original-composition disjointness. New Slakh own-training UUID holdout does not exclude upstream pretraining overlap. Natural band, rich chord accuracy and editing time remain unvalidated.'}
    save(destination,report);save(target.with_suffix('.json'),report)
    print('MAC_DEVELOPMENT_PACKAGED' if a.development_only else 'MAC_SELECTION_FROZEN',winner['tag'],'research adapter',state['config']['name'],state['step'],'practical',practical,flush=True)

if __name__=='__main__':main()
