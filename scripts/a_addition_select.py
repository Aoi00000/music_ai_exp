"""Freeze guitar addition policy against the strong same-input Mu baseline."""
from pathlib import Path
import json,torch,argparse
from scripts.a_candidate_data import save,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',choices=['addition_v1','addition_balanced_v2','addition_jazz_v3_control','addition_jazz_v3'],default='addition_v1');a=parser.parse_args()
    out=Path('reports')/a.experiment/'selection.json'
    if out.exists():raise ValueError('Addition selection already frozen')
    records=[];controls=None;union=None
    for kind in (('linear',) if a.experiment.startswith('addition_jazz_') else ('linear','mlp')):
        r=json.load(open(Path('runs')/a.experiment/kind/'completed.json'));controls=r['control'];union=r['unfiltered_union'];base=controls['summary']
        for h in r['history']:
            for c in h['choices']:
                s=c['summary'];eligible=s['active_exact_match']>=base['active_exact_match']-1e-12 and all(s['datasets'][d][k]>=v[k]-1e-12 for d,v in base['datasets'].items() for k in ('note_f1','frame_f1'))
                records.append(dict(kind=kind,checkpoint=h['checkpoint'],checkpoint_sha256=h['sha256'],eligible=eligible,**c))
    allowed=[r for r in records if r['eligible'] and r['summary']['addition_score']>controls['summary']['addition_score']+1e-8]
    best=max(allowed,key=lambda r:r['summary']['addition_score']) if allowed else None
    result=dict(control=controls,unfiltered_union=union,best=best,choices=[{k:v for k,v in r.items() if k!='tracks'} for r in records],
        test_used_for_selection=False,automatic_application_approved=False,identity_selected=best is None,
        manifest_sha256=sha('data/addition_jazz_v3/manifest.json' if a.experiment.startswith('addition_jazz_') else 'data/addition_v1/manifest.json'))
    if best:
        state=torch.load(best['checkpoint'],map_location='cpu',weights_only=True);state['threshold']=best['threshold'];state['automatic_application_approved']=False
        filename='guitar_addition_development.pt' if a.experiment=='addition_v1' else 'guitar_addition_balanced_review.pt' if a.experiment=='addition_balanced_v2' else a.experiment+'_review.pt'
        target=Path('models')/filename;torch.save(state,target);result.update(export=str(target),export_sha256=sha(target))
    save(out,result);print('GUITAR_ADDITION_FROZEN',None if best is None else (best['kind'],best['threshold'],best['summary']),flush=True)


if __name__=='__main__':main()
