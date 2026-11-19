"""Second hypothesis: correct releases only on confidently matched notes."""
from pathlib import Path
import json,numpy as np,torch
from band2tab.release import predict_release,apply_release
from band2tab.stem_refiner import predict_corrections
from scripts.a_release_train import load,evaluate
from scripts.a_candidate_data import save,sha


def main():
    report=Path('reports/release_gate_v3');selection=report/'selection.json'
    if selection.exists():raise ValueError('Confidence-gate selection already frozen')
    classifier=Path('models/stem_note_corrector_development.pt');cls=torch.load(classifier,map_location='cpu',weights_only=True)
    protocol=dict(hypothesis='Release-only inference was applied to unmatched false notes despite matched-note training. Gate release changes using the previously trained note-validity network.',
        classifier_sha256=sha(classifier),source_release_protocol_sha256=sha('reports/release_v2/protocol.json'),
        confidence_thresholds=[.9,.98],scale=.5,models=4,checkpoints_per_model=2,
        criteria='Same release score and active-exact gate as v2; preserve every onset/role/pitch/note; zero correction fallback',
        validation='Only original 16 development conditions; no new test access',test_used=False)
    save(report/'protocol.json',protocol);manifest,rows,cache,metadata=load();validation=[r for r in rows if r['split']=='val'];torch.set_num_threads(2)
    confidence={r['id']:predict_corrections(cls,cache[r['id']]['x'])[0] for r in validation}
    control=evaluate(validation,metadata,{r['id']:np.zeros(len(cache[r['id']]['x'])) for r in validation},0.);records=[]
    for completed in sorted(Path('runs/release_v2').glob('*/completed.json')):
        for checkpoint in json.load(open(completed))['history']:
            state=torch.load(checkpoint['checkpoint'],map_location='cpu',weights_only=True)
            pred={r['id']:predict_release(state,cache[r['id']]['x']) for r in validation}
            for threshold in protocol['confidence_thresholds']:
                masked={r['id']:pred[r['id']]*(confidence[r['id']]>=threshold) for r in validation}
                evaluated=evaluate(validation,metadata,masked,.5);s=evaluated['summary']
                records.append(dict(checkpoint=checkpoint['checkpoint'],checkpoint_sha256=sha(checkpoint['checkpoint']),threshold=threshold,
                    eligible=s['active_exact_match']>=control['summary']['active_exact_match']-.005,**evaluated))
    allowed=[r for r in records if r['eligible'] and r['summary']['release_score']>control['summary']['release_score']]
    best=max(allowed,key=lambda r:r['summary']['release_score']) if allowed else None
    save(selection,dict(protocol_sha256=sha(report/'protocol.json'),classifier=str(classifier),classifier_sha256=sha(classifier),
        manifest_sha256=sha(manifest),control=control,best=best,choices=[{k:v for k,v in r.items() if k!='tracks'} for r in records],
        automatic_application_approved=False,new_test_opened=False,identity_selected=best is None))
    print('RELEASE_GATE_FROZEN',None if best is None else (best['checkpoint'],best['threshold'],best['summary']),flush=True)


if __name__=='__main__':main()
