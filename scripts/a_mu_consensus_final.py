"""Transfer only agreeing learned note releases, protecting base attacks/counts."""
from pathlib import Path
import argparse,copy,itertools,json
from band2tab.addition import _assign
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate
ROOT=Path('reports/mu_consensus_final_v4')

def transfer(base,learned,scales,tolerance=.02):
    parts=copy.deepcopy(base)
    for part,scale in zip(('guitar','bass'),scales):
        for i,j in _assign(base[part],learned[part]):
            a=base[part][i];b=learned[part][j]
            if abs(a['onset']-b['onset'])>tolerance:continue
            delta=b['offset']-a['offset'];limit=min(.15,(a['offset']-a['onset'])*.5)
            if abs(delta)>limit+1e-9:continue
            parts[part][i]['offset']=a['offset']+float(delta*scale)
    return parts

def register():
    if (ROOT/'protocol.json').exists():raise ValueError('Protocol frozen')
    save(ROOT/'protocol.json',dict(reason='Direct gold Mu adaptation reduced noteF1 while slightly improving FrameF1 on reused16. Test learned release transfer without moving attacks or deleting notes.',
      arms=['canonical_gold','canonical_tone','native_gold','native_tone'],scales=[0.,.25,.5,1.],onset_agreement_seconds=.02,
      maximum_offset_change='min150ms, half baseline duration; ignore larger disagreements instead of clipping',
      objective='.5 Frame+.5 offset; all4 per dataset nondecreasing vs same-device base; strict gain else identity',
      diagnosis_used_repeated16=True,calibration='Same5 internal conditions as direct Mu pilot; tiny, reused, not new test.',test_used=False,implementation_sha256=sha(__file__)))

def evaluate(rows,base,learned,scales):
    items=[];tracks=[]
    for r in rows:
        p=transfer(base[r['id']],learned[r['id']],scales);s=metrics(r,p);items.append((r,s));tracks.append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=p,metrics=s))
    return dict(summary=aggregate(items),tracks=tracks)

def select():
    if (ROOT/'selection.json').exists():raise ValueError('Consensus selection frozen')
    rows=json.load(open('data/mu_canonical_v3/monitor.json'));first=json.load(open('reports/mu_canonical_v3/selection.json'));second=json.load(open('reports/mu_native_v2/selection.json'));base={r['id']:r['parts'] for r in first['tracks']['baseline']}
    if base!={r['id']:r['parts'] for r in second['tracks']['baseline']}:raise ValueError('Matched baselines differ between Mu pilots')
    controls=evaluate(rows,base,base,(0.,0.));b=controls['summary'];choices=[]
    for name,source,arm in [('canonical_gold',first,'gold'),('canonical_tone',first,'gold_tone'),('native_gold',second,'gold'),('native_tone',second,'gold_tone')]:
        learn={r['id']:r['parts'] for r in source['tracks'][arm]}
        for scales in itertools.product((0.,.25,.5,1.),repeat=2):
            v=evaluate(rows,base,learn,scales);s=v['summary'];eligible=all(s['datasets'][d][k]>=ref[k]-1e-10 for d,ref in b['datasets'].items() for k in ('note_f1','frame_f1','active_exact_match','offset_f1'));score=.5*s['frame_f1']+.5*s['offset_f1'];choices.append(dict(name=name,scales=scales,summary=s,eligible=eligible,score=score))
    allowed=[v for v in choices if v['eligible'] and v['score']>.5*b['frame_f1']+.5*b['offset_f1']+1e-9];best=max(allowed,key=lambda v:v['score']) if allowed else None
    save(ROOT/'selection.json',dict(best=best,choices=choices,baseline=b,identity_selected=best is None,source_sha256={p:sha(p) for p in ['reports/mu_canonical_v3/selection.json','reports/mu_native_v2/selection.json']},test_used=False));print('MU_CONSENSUS_SELECTED',best,flush=True)

def review():
    if (ROOT/'review.json').exists():raise ValueError('Consensus review frozen')
    selection=json.load(open(ROOT/'selection.json'));rows=json.load(open('data/mu_canonical_v3/review_manifest.json'));a=json.load(open('reports/mu_canonical_v3/review.json'));b=json.load(open('reports/mu_native_v2/review.json'));base={r['id']:r['parts'] for r in a['results']['baseline']['tracks']};best=selection['best']
    if base!={r['id']:r['parts'] for r in b['results']['baseline']['tracks']}:raise ValueError('Review baselines differ')
    if best:
        source=a if best['name'].startswith('canonical') else b;arm='gold_tone' if best['name'].endswith('tone') else 'gold';learn={r['id']:r['parts'] for r in source['results'][arm]['tracks']};scales=best['scales']
    else:learn=base;scales=(0.,0.)
    v=evaluate(rows,base,learn,scales);save(ROOT/'review.json',dict(baseline=a['results']['baseline']['summary'],selected=v,best=best,test_used=False));print('MU_CONSENSUS_REVIEW',v['summary'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','select','review']);a=p.parse_args();globals()[a.command]()
