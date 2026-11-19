"""Append targeted jazz candidates with a fixed native-conditioned Mu baseline."""
from pathlib import Path
import json,time,argparse
import numpy as np,soundfile as sf
from band2tab.audio import features
from band2tab.stem_refiner import read_candidates,rms_envelope
from band2tab.addition import proposals,addition_features,targets
from scripts.a_candidate_data import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();deadline=time.monotonic()+1800
    rows=json.load(open('data/jazz_guitar_v1/manifest.json'));all_rows=json.load(open('data/addition_v1/manifest.json'))
    for r in rows:
        mu=Path('reports/jazz_guitar_v1/midi/conditioned')/(r['id']+'.mid');your=Path('reports/jazz_guitar_v1/midi/your')/(r['id']+'.mid')
        while a.wait and not all(p.with_suffix('.json').exists() for p in (mu,your)):
            if time.monotonic()>deadline:raise TimeoutError('Jazz predictions did not finish')
            time.sleep(5)
        cache=Path('data/addition_jazz_v3/cache')/(r['id']+'.npz');meta=cache.with_suffix('.json');fp=dict(mu_sha256=sha(mu),proposal_sha256=sha(your),manifest_sha256=sha('data/jazz_guitar_v1/manifest.json'))
        if meta.exists():
            if json.load(open(meta))['fingerprint']!=fp:raise ValueError('Stale jazz feature cache')
        else:
            ns=read_candidates(mu,'guitar',r['duration']);baseline={'guitar':[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in ns],'bass':[]}
            pool=proposals(read_candidates(your,'guitar',r['duration']),baseline['guitar']);views={};env={}
            for part,path in r['inputs'].items():views[part]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);env[part]=rms_envelope(y,sr)
            x=addition_features(views,env,pool,baseline['guitar'],r['duration']);truth=targets(pool,baseline['guitar'],r['parts']['guitar'],r['guard_seconds'],r['duration'])
            cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,x=x,**truth);save(meta,dict(fingerprint=fp,pool=pool,baseline=baseline))
        all_rows.append({**r,'addition_cache':str(cache)})
        print('JAZZ_ADDITION_CACHE',r['id'],flush=True)
    save('data/addition_jazz_v3/manifest.json',all_rows);print('JAZZ_ADDITION_DATA_COMPLETE',len(all_rows),flush=True)


if __name__=='__main__':main()
