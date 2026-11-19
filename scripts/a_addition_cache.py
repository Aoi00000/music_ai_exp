"""Source-disjoint guitar proposal features; truth only in separate targets."""
from pathlib import Path
import json,time,argparse,numpy as np,soundfile as sf
from band2tab.audio import features
from band2tab.stem_refiner import read_candidates,rms_envelope
from band2tab.addition import proposals,addition_features,targets
from scripts.a_candidate_data import save,sha
from scripts.a_stress_data import local


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');a=parser.parse_args();deadline=time.monotonic()+1800
    root=Path('backups/azure').resolve();manifest=root/'data/stem_refiner_v1/manifest.json';rows=[r for r in json.load(open(manifest)) if r['dataset'].startswith(('stem_band_','band_stress_'))]
    output=[]
    for r in rows:
        cache=Path('data/addition_v1/cache')/(r['id']+'.npz');meta=cache.with_suffix('.json')
        old=json.load(open(root/'data/stem_refiner_v1/cache'/(r['id']+'.json')));midi=Path('reports/addition_v1/midi')/(r['id']+'.mid')
        while a.wait and not midi.with_suffix('.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Proposal generation did not complete')
            time.sleep(5)
        fp=dict(proposal_midi_sha256=sha(midi),mu_feature_fingerprint=old['fingerprint'],schema='guitar_addition_358_v1',source_manifest_sha256=sha(manifest))
        if cache.exists() and meta.exists():
            if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale addition features')
        else:
            baseline=old['baseline_parts'];pool=proposals(read_candidates(midi,'guitar',r['duration']),baseline['guitar']);views={};env={}
            for p,path in r['inputs'].items():
                path=local(root,path);views[p]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);env[p]=rms_envelope(y,sr)
            x=addition_features(views,env,pool,baseline['guitar'],r['duration']);truth=targets(pool,baseline['guitar'],r['parts']['guitar'],r['guard_seconds'],r['duration'])
            if x.shape!=(len(pool),358):raise ValueError('Unexpected addition feature shape')
            cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,x=x,**truth);save(meta,dict(fingerprint=fp,pool=pool,baseline=baseline))
        output.append(r);print('ADDITION_FEATURES_READY',r['id'],flush=True)
    save('data/addition_v1/manifest.json',output);print('ADDITION_FEATURES_COMPLETE',len(output),flush=True)


if __name__=='__main__':main()
