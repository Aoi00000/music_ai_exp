"""Validation-only residual-strength calibration, after observing validation harm.

This is an explicit second development phase. It never reads test predictions.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from band2tab.temporal import TemporalFusion,predict_scales,decode_parts
from scripts.a_temporal_train import ROOT,REPORT,load_cache,metrics,aggregate,eligible
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):Path(p).write_text(json.dumps(o,indent=2))

def main():
    out=REPORT/'calibration.json'
    if out.exists() or (REPORT/'test_comparison.json').exists():raise RuntimeError('Calibrate once before test')
    selection=json.loads((REPORT/'selection.json').read_text());baseline=selection['controls']['audio_bounded']
    rows=[r for r in json.loads((ROOT/'manifest.json').read_text()) if r['split']=='val'];cache={r['id']:load_cache(r) for r in rows}
    policies=[{'onset':a,'frame':b,'offset':off,'min_duration':.03} for a in (.3,.5,.7) for b in (.3,.5,.7) for off in (.5,1.1)]
    torch.set_num_threads(4);comparisons=[];best=None;results={}
    for condition,record in selection['results'].items():
        if sha(record['checkpoint'])!=record['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        state=torch.load(record['checkpoint'],weights_only=True,map_location='cpu');model=TemporalFusion(include_ft=True).to('cuda');model.load_state_dict(state['model']);model.eval()
        probabilities={r['id']:predict_scales(model,cache[r['id']]['e'],cache[r['id']]['m'],(.25,.5,.75,1.),'cuda') for r in rows}
        condition_best=None
        for strength in (.25,.5,.75,1.):
            for policy in policies:
                score=aggregate([(r,metrics(r,decode_parts(probabilities[r['id']][strength],policy))) for r in rows])
                item={'condition':condition,'residual_scale':strength,'policy':policy,'validation':score,
                      'eligible':eligible(score,baseline),'checkpoint':record['checkpoint'],'checkpoint_sha256':record['checkpoint_sha256']}
                comparisons.append(item)
                if best is None or score['selection_score']>best['validation']['selection_score']:best=item
                if condition_best is None or score['selection_score']>condition_best['validation']['selection_score']:condition_best=item
        results[condition]=condition_best
        print('CALIBRATED',condition,condition_best['residual_scale'],condition_best['validation']['selection_score'],flush=True)
    package=torch.load(best['checkpoint'],weights_only=True,map_location='cpu');package.update(policy=best['policy'],residual_scale=best['residual_scale'],automatic_application_approved=False)
    path=Path('models/muscriptor_temporal_calibrated_review.pt');torch.save(package,path)
    meta={'schema':package['schema'],'sha256':sha(path),'source_sha256':best['checkpoint_sha256'],
          'source_checkpoint':best['checkpoint'],'residual_scale':best['residual_scale'],'policy':best['policy'],
          'trainable_parameters':package['trainable_parameters'],'automatic_application_approved':False}
    save(path.with_suffix('.json'),meta)
    save(out,{'selected':best,'results':results,'comparisons':comparisons,'test_used_for_selection':False,
        'phase':'Additional validation-only calibration prompted by validation solo-bass degradation',
        'base_selection_sha256':sha(REPORT/'selection.json'),'packaged_sha256':sha(path),'automatic_application_approved':False})
    print('CALIBRATION_FROZEN',best['condition'],best['residual_scale'],best['policy'],'eligible',best['eligible'],flush=True)

if __name__=='__main__':main()
