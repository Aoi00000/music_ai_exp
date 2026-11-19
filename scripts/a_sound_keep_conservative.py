"""Fixed conservative composition: old octave filter + new sounding-aware bass."""
from pathlib import Path
import json,copy
import numpy as np,torch
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,KEYS
from scripts.a_keep_feedback_review import bootstrap
from scripts.a_release_train import load
from band2tab.bass_keep import apply

REPORT=Path('reports/sound_keep_conservative_v7')

def main():
    if (REPORT/'comparison.json').exists():raise ValueError('Comparison frozen')
    source=Path('runs/sound_keep_v6/full_band/selected_bundle.pt')
    save(REPORT/'protocol.json',dict(reason='Aggressive v5 reduced note FP but worsened sounding frames on the separate application fixture. Use sounding-aware bass plus previous octave-only guitar; omit additions that previously reduced offset F1 on known14.',
        source_sha256=sha(source),guitar_sha256=sha('models/octave_filter_review.pt'),no_new_threshold_tuning=True,
        thresholds_fixed_on_internal24=True,scope='Repeated16, known14, already used training fixture. Not a fresh test.',automatic_application_approved=False))
    bundle=torch.load(source,map_location='cpu',weights_only=True);_,rows,cc,mm=load();ds=[r for r in rows if r['split']=='val'];rs=json.load(open('data/keep_feedback_regression_v1/manifest.json'));rc={};rm={}
    for r in rs:
        with np.load(r['cache']) as f:rc[r['id']]={k:f[k].copy() for k in f.files}
        rm[r['id']]=json.load(open(Path(r['cache']).with_suffix('.json')))
    old=json.load(open('reports/octave_regression_v1/comparison.json'))['tracks'];all_s={};all_t={};ci={};torch.set_num_threads(2)
    for scope,rows,arrays,meta in [('development',ds,cc,mm),('known_regression',rs,rc,rm)]:
        previous={r['id']:r for r in old[scope]};items={k:[] for k in ('baseline','octave_only','sound_bass_only','combined')};tracks={k:[] for k in items}
        for r in rows:
            m=meta[r['id']];bass=apply(bundle,arrays[r['id']]['x'],m['pool'],m['unmodelled'],r['duration'],r['guard_seconds']);g=previous[r['id']]['variants']['octave_only']['parts']
            for name,p in [('baseline',m['baseline_parts']),('octave_only',g),('sound_bass_only',bass),('combined',dict(guitar=g['guitar'],bass=bass['bass']))]:
                s=metrics(r,p);items[name].append((r,s));tracks[name].append(dict(id=r['id'],group=r['group'],dataset=r['dataset'],parts=p,metrics=s))
        all_s[scope]={k:aggregate(v) for k,v in items.items()};all_t[scope]=tracks;ci[scope]=bootstrap(rows,tracks['baseline'],tracks['combined'])
    export=Path('models/bass_sound_keep_review.pt');torch.save(bundle,export)
    save(REPORT/'comparison.json',dict(summary=all_s,tracks=all_t,bootstrap=ci,export=str(export),export_sha256=sha(export),automatic_application_approved=False,test_used=False))
    print('CONSERVATIVE_COMPOSITION',all_s,flush=True)
if __name__=='__main__':main()
