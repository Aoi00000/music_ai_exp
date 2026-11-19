"""Calibration-only final gate: tied scores adopt explicit identity, not a threshold.

Preserves the raw selection audit and all trained checkpoints. The ambiguity was
found by checking internal calibration scores, before reading review outcomes.
"""
from pathlib import Path
import json,copy,torch
from scripts.a_candidate_data import save,sha

def main():
    source=Path('reports/residual_v1/selection.json');d=json.load(open(source));out=copy.deepcopy(d)
    b=d['baseline'];baseline_score=.5*b['note_f1']+.3*b['frame_f1']+.2*b['active_exact_match']
    out.update(raw_selection_sha256=sha(source),reason='Every NN arm tied baseline on calibration. No strict gain means explicit identity; do not apply a low threshold that happened to change zero calibration notes.',reused_review_outcomes_used_for_gate=False)
    for a in out['arms']:
        a['calibration_strict_gain']=a['best']['score']>baseline_score+1e-9
        if not a['calibration_strict_gain']:
            a['best']['thresholds']=[1.1,1.1];a['best']['explicit_identity']=True
        bundle=torch.load(a['export'],weights_only=True,map_location='cpu');bundle['thresholds']=a['best']['thresholds'];p=Path('runs/residual_identity_v1')/(a['arm']+'.pt');p.parent.mkdir(parents=True,exist_ok=True);torch.save(bundle,p);a.update(export=str(p),sha256=sha(p))
    save('reports/residual_identity_v1/selection.json',out);print('EXPLICIT_IDENTITY_GATE',[a['calibration_strict_gain'] for a in out['arms']],flush=True)
if __name__=='__main__':main()
