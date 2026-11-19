"""Verify executed protocols, source versions, recovered data, and learned LoRA."""
from pathlib import Path
import json,torch
from scripts.a_candidate_data import sha,save
FAMILIES=('residual_v1','mu_adapt_v1','mu_native_v2','mu_canonical_v3')
def main():
    recovered=Path('backups/azure');records=[];source_checks=[];run_checks=[];adapters={}
    for family in FAMILIES:
        protocol=Path('data',family,'protocol.json');d=json.load(open(protocol))
        for file,h in d['implementation_sha256'].items():
            actual=Path(file)
            if family=='mu_adapt_v1' and file=='scripts/a_mu_adapt.py':actual=Path('experiments/frozen_20261120/a_mu_adapt_initial.py')
            if sha(actual)!=h:raise ValueError('Executed source unavailable or changed: '+family+' '+file)
            source_checks.append(dict(family=family,declared=file,recovered_source=str(actual),sha256=h))
        for cfg in Path('runs',family).rglob('config.json'):
            c=json.load(open(cfg))
            if c['protocol_sha256']!=sha(protocol):raise ValueError('Run does not match EXECUTED protocol: '+str(cfg))
            run_checks.append(str(cfg))
        for kind in ('data','runs','reports'):
            for p in (recovered/kind/family).rglob('*'):
                if not p.is_file() or p.suffix not in ('.npz','.json','.pt','.mid'):continue
                relative=p.relative_to(recovered)
                if not Path(relative).is_file() or sha(p)!=sha(relative):raise ValueError('Recovered artifact differs: '+str(relative))
                records.append(dict(path=str(relative),bytes=p.stat().st_size,sha256=sha(p)))
        if family.startswith('mu_'):
            hs=[]
            for arm in ('gold','gold_tone'):
                path=Path('runs',family,arm,'final.pt');s=torch.load(path,weights_only=True,map_location='cpu');done=json.load(open(path.parent/'completed.json'))
                if not done['frozen_base_unchanged'] or done['successful_updates']!=120:raise ValueError('Mu update or frozen-base invariant failed')
                learned=[name for name,m in s['adapters'].items() if torch.isfinite(m['b']).all() and float(m['b'].abs().sum())>0]
                if len(learned)!=8:raise ValueError('Mu layers did not learn')
                adapters[family+'/'+arm]=dict(learned_layers=learned,parameters=sum(m['a'].numel()+m['b'].numel() for m in s['adapters'].values()),checkpoint_sha256=sha(path));hs.append(done['draw_sha256'])
            if len(set(hs))!=1:raise ValueError('Tone pair draws differ')
    # The isolated program comparison additionally uses the exact same draw IDs.
    for arm in ('gold','gold_tone'):
        a=json.load(open('runs/mu_native_v2/'+arm+'/completed.json'));b=json.load(open('runs/mu_canonical_v3/'+arm+'/completed.json'))
        if a['draw_sha256']!=b['draw_sha256']:raise ValueError('Matched program comparison draws differ')
    result=dict(files=len(records),bytes=sum(r['bytes'] for r in records),all_sha256_matched=True,executed_sources=source_checks,executed_run_configs=run_checks,mu_adapters=adapters,records=records)
    save('reports/artifact_audit_20261120.json',result);print('FINAL_FEEDBACK_AUDIT',result['files'],result['bytes'],'bytes; executed protocols/sources and six Mu adapters verified')
if __name__=='__main__':main()
