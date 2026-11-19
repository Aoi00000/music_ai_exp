"""Fixed-policy final comparison; never select or retrain using these results."""
from pathlib import Path
import json
from collections import defaultdict
import numpy as np,soundfile as sf,torch
from band2tab.audio import features,PITCH_MIN,PITCH_MAX
from band2tab.stem_refiner import read_candidates,rms_envelope,feature_matrix,predict_corrections,suggest
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import metrics,aggregate,KEYS


def main():
    manifest=Path('data/stem_refiner_holdout_v1/manifest.json');selection=Path('reports/stem_refiner_v1/selection.json');protocol=json.load(open(manifest.parent/'protocol.json'));selected=json.load(open(selection))
    if protocol['manifest_sha256']!=sha(manifest) or protocol['selection_sha256']!=sha(selection):raise ValueError('Frozen protocol changed')
    if sha(selected['export'])!=selected['export_sha256']:raise ValueError('Frozen model changed')
    state=torch.load(selected['export'],map_location='cpu',weights_only=True);rows=json.load(open(manifest));results=[];torch.set_num_threads(2)
    folder=Path('reports/stem_refiner_holdout_v1');baseline_scores=[];learned_scores=[]
    for r in rows:
        original={};outside={p:[] for p in ('guitar','bass')};pool=[]
        for part in ('guitar','bass'):
            path=folder/'midi'/r['id']/(part+'.mid');meta=json.load(open(path.with_suffix('.json')))
            if meta['fingerprint']['selection_sha256']!=sha(selection) or meta['fingerprint']['audio_sha256']!=sha(r['inputs'][part]):raise ValueError('Stale held-out inference')
            notes=read_candidates(path,part,r['duration']);original[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes]
            pool += [n for n in notes if PITCH_MIN<=n['pitch']<=PITCH_MAX]
            outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
        views={};env={}
        for part in ('mix','guitar','bass'):
            views[part]=features(r['inputs'][part]);y,sr=sf.read(r['inputs'][part],dtype='float32',always_2d=True);env[part]=rms_envelope(y,sr)
        x=feature_matrix(views,env,pool,r['duration']);prob,delta=predict_corrections(state,x)
        proposed=suggest(pool,prob,delta,r['duration'],unmodelled=outside,**state['policy'])
        base=metrics(r,original);learned=metrics(r,proposed);baseline_scores.append((r,base));learned_scores.append((r,learned))
        record=dict(id=r['id'],group=r['group'],dataset=r['dataset'],baseline=base,learned=learned,raw_parts=original,parts=proposed,
                    changes=dict(removed=len(pool)-sum(float(p)>=state['policy']['threshold'] for p in prob),retime=state['policy']['retime']))
        save(folder/'tracks'/(r['id']+'.json'),record);results.append(record);print('FROZEN_HELDOUT_EVALUATED',r['id'],base['note_f1'],learned['note_f1'],flush=True)
    base=aggregate(baseline_scores);learned=aggregate(learned_scores);rng=np.random.default_rng(19);by=defaultdict(list)
    for r in results:by[r['group']].append(r)
    groups=sorted(by);ci={}
    for dataset in ('all','held_band_clean','held_band_noisy'):
        # Bootstrap whole guitar groups; paired clean/noisy are not independent.
        values=np.array([[np.mean([r['learned'][k]-r['baseline'][k] for r in by[g] if dataset=='all' or r['dataset']==dataset]) for k in KEYS] for g in groups])
        samples=values[rng.integers(0,len(groups),size=(10000,len(groups)))].mean(1)
        ci[dataset]={k:dict(mean_delta=float(values[:,j].mean()),percentile_95=[float(v) for v in np.quantile(samples[:,j],[.025,.975])]) for j,k in enumerate(KEYS)}
    save(folder/'comparison.json',dict(baseline=base,learned=learned,delta={k:learned[k]-base[k] for k in (*KEYS,'score')},
        paired_group_bootstrap=ci,groups=len(groups),conditions=len(rows),tracks=results,selection_sha256=sha(selection),model_sha256=selected['export_sha256'],
        final_policy=state['policy'],test_used_for_selection=False,further_tuning_on_this_holdout_allowed=False,source_limitations=protocol['source_limitations'],
        natural_band_recording=False,baseline_description='Frozen MuScriptor medium + six-source Demucs, same stems and source-role parser',native_mac_cpu=True))
    save('reports/holdout_registry.json',dict(opened='stem_refiner_holdout_v1',guitar_source_groups=groups,
        consumed_legacy_goat_test_conditions=21,remaining_unopened_legacy_slakh_conditions=11,
        note='All three GOAT views of each opened performance must now be considered seen; never claim all previous 32 conditions remain unseen.'))
    print('FROZEN_HELDOUT_COMPLETE',base['note_f1'],learned['note_f1'],flush=True)


if __name__=='__main__':main()
