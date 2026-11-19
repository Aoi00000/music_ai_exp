"""Freeze development selection before any held-out band inference."""
from pathlib import Path
import itertools,json,shutil
import numpy as np,torch
from band2tab.stem_refiner import predict_corrections,suggest
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate


def main():
    target=Path('reports/stem_refiner_v1/selection.json')
    if target.exists():raise ValueError('Selection already frozen')
    root=Path('backups/azure');manifest=root/'data/stem_refiner_v1/manifest.json'
    rows=[r for r in json.load(open(manifest)) if r['split']=='val'];cache={};metadata={}
    for r in rows:
        path=root/'data/stem_refiner_v1/cache'/(r['id']+'.npz');cache[r['id']]=np.load(path)['x'];metadata[r['id']]=json.load(open(path.with_suffix('.json')))
    control=aggregate([(r,metrics(r,metadata[r['id']]['baseline_parts'])) for r in rows]);choices=[];torch.set_num_threads(2)
    runs=('supervised_v1','conservative_v1','band_recall_v2','band_recall_replica_v2')
    for name in runs:
        folder=Path('runs/stem_refiner_v1')/name
        for path in sorted(folder.glob('step_*.pt')):
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state['config']['manifest_sha256']!=sha(manifest):raise ValueError('Manifest changed')
            pred={r['id']:predict_corrections(state,cache[r['id']]) for r in rows}
            for threshold,retime in itertools.product((0.,.01,.05,.1,.2,.35,.5,.65,.8),(False,True)):
                policy=dict(threshold=threshold,retime=retime);tracks=[];scores=[]
                for r in rows:
                    meta=metadata[r['id']];prob,delta=pred[r['id']]
                    parts=suggest(meta['pool'],prob,delta,r['duration'],unmodelled=meta['unmodelled'],**policy);score=metrics(r,parts);scores.append((r,score))
                    tracks.append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],metrics=score,parts=parts))
                summary=aggregate(scores)
                # A global increase must not hide a drop in noisy-band notes.
                eligible=all(summary['datasets'][d]['note_f1']>=v['note_f1']-1e-9 for d,v in control['datasets'].items())
                choices.append(dict(run=name,checkpoint=str(path),checkpoint_sha256=sha(path),policy=policy,summary=summary,eligible=eligible,tracks=tracks))
    best=max((c for c in choices if c['eligible']),key=lambda c:c['summary']['score'])
    state=torch.load(best['checkpoint'],map_location='cpu',weights_only=True);state['policy']=best['policy']
    export=Path('models/stem_note_corrector_development.pt');export.parent.mkdir(exist_ok=True);torch.save(state,export)
    selection=dict(best=best,control=control,export=str(export),export_sha256=sha(export),manifest_sha256=sha(manifest),
        hypotheses=['Remove false notes conservatively from identical separated Mu predictions','Optionally correct attack/release timing'],
        development_runs=list(runs),policy_choices_evaluated=len(choices),test_used_for_selection=False,
        repeated_validation_warning='Repeated validation is optimistic; only subsequent frozen held-out comparison estimates generalization.',
        evaluation_protocol='Same native 44.1k source mixtures, frozen six-source separation, Mu medium source-known non-drum policy; no tempo quantization.',
        final_holdout='Seven GOAT official test performances, held-out FiloBass players, MUSDB test vocal/drum backgrounds; clean/noisy pairs. Seven paired groups, not fourteen independent songs.',
        final_test_permitted=True,automatic_application_approved=False)
    save(target,selection);save(target.parent/'validation_choices.json',[{k:v for k,v in c.items() if k!='tracks'} for c in choices])
    print('STEM_SELECTION_FROZEN',best['run'],best['checkpoint'],best['policy'],best['summary']['score'],control['score'],flush=True)


if __name__=='__main__':main()
