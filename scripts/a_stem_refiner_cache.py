"""Audio-only note features and separate training labels; no held-out test."""
from pathlib import Path
import argparse,json,numpy as np,soundfile as sf
from band2tab.audio import features,PITCH_MIN,PITCH_MAX
from band2tab.stem_refiner import read_candidates,rms_envelope,feature_matrix,supervised_targets
from scripts.a_candidate_data import save,sha
from scripts.a_stress_data import local


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--validation-midi-root',type=Path);a=p.parse_args();root=a.root.resolve()
    manifest=root/'data/stem_refiner_v1/manifest.json';rows=json.load(open(manifest))
    for row in rows:
        path=root/'data/stem_refiner_v1/cache'/(row['id']+'.npz');meta=path.with_suffix('.json')
        if row['split']=='train':midis={p:root/'reports/stem_refiner_v1/midi'/row['id']/(p+'.mid') for p in ('guitar','bass')}
        else:midis={p:(a.validation_midi_root or root/'reports/band_hybrid_v1/mu/six_parts')/(row['id']+'_'+p+'.mid') for p in ('guitar','bass')}
        inputs={p:local(root,path) for p,path in row['inputs'].items()}
        fingerprint={'audio_sha256':{p:sha(inputs[p]) for p in ('mix','guitar','bass')},'manifest_sha256':sha(manifest),
            'midi_sha256':{p:sha(path) for p,path in midis.items() if path.exists()},'feature_schema':'three_CQT_and_energy_350_v1',
            'target_schema':'source_isolated_unique_assignment_v2'}
        if path.exists() and meta.exists():
            if json.load(open(meta))['fingerprint']!=fingerprint:raise ValueError('Stale corrector features')
            continue
        pool=[];outside={p:[] for p in ('guitar','bass')};baseline={p:[] for p in ('guitar','bass')}
        for part,midi in midis.items():
            if row.get('known_part') and part!=row['known_part']:continue
            notes=read_candidates(midi,part,row['duration'])
            baseline[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes]
            pool+=[n for n in notes if PITCH_MIN<=n['pitch']<=PITCH_MAX]
            outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
        views={p:features(inputs[p]) for p in ('mix','guitar','bass')};envelopes={}
        for p in views:
            audio,sr=sf.read(inputs[p],dtype='float32',always_2d=True);envelopes[p]=rms_envelope(audio,sr)
        x=feature_matrix(views,envelopes,pool,row['duration']);targets=supervised_targets(pool,row['parts'],row['guard_seconds'],row['duration'])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,x=x,**targets)
        save(meta,{'fingerprint':fingerprint,'pool':pool,'unmodelled':outside,'baseline_parts':baseline})
        print('STEM_REFINER_FEATURES',row['id'],len(pool),flush=True)
    print('STEM_REFINER_CACHE_COMPLETE',flush=True)


if __name__=='__main__':main()
