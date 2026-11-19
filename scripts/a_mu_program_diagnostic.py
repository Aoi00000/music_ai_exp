"""Reproduce the training-only program diagnostic used before native Mu learning.

Extracted from the original preparation command; existing results must match.
"""
from pathlib import Path
from collections import Counter
import json
from band2tab.stem_refiner import read_candidates

def main():
    rows={r['id']:r for r in json.load(open('data/residual_v1/manifest.json'))};old=json.load(open('data/mu_adapt_v1/manifest.json'));changes=[];counts=Counter()
    for a in old:
        ident=a['id'].rsplit('_',1)[0];r=rows[ident];part=a['part']
        if r.get('base_cache'):
            pool=json.load(open(Path(r['base_cache']).with_suffix('.json')))['pool'];pool=[n for n in pool if n['part']==part]
        else:pool=read_candidates(Path('backups/azure/reports/candidate_v2/midi')/ident/(part+'.mid'),part,r['duration'])
        allowed=range(24,32) if part=='guitar' else (*range(32,40),43);by=Counter()
        for n in pool:
            if n.get('program') in allowed:by[n['program']]+=n['offset']-n['onset']
        native=by.most_common(1)[0][0] if by else a['program'];counts[(part,a['program'],native)]+=1
        changes.append(dict(id=a['id'],native_program=native,canonical_program=a['program'],changed=native!=a['program'],evidence='Duration-weighted dominant frozen Mu program in own source, restricted to guitar24..31 or bass32..39/43; fallback canonical. Not selected by evaluation accuracy.'))
    result={'counts':{str(k):v for k,v in counts.items()},'records':changes};path=Path('reports/mu_adapt_v1/program_diagnostic.json')
    if path.exists():
        if json.load(open(path))!=result:raise ValueError('Diagnostic changed; do not overwrite the historical teacher')
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print('PROGRAM_DIAGNOSTIC_REPRODUCED',dict(counts))
if __name__=='__main__':main()
