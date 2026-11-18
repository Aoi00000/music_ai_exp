"""Once-only, source-disjoint held-out mixtures after model selection is frozen."""
from pathlib import Path
import json,hashlib
import numpy as np,soundfile as sf
from scripts.a_candidate_data import save,sha
from scripts.a_stress_data import read_stereo,crop_again,local
from band2tab.separation import write_input
from band2tab.filobass import records


def main():
    selection=Path('reports/stem_refiner_v1/selection.json');s=json.load(open(selection))
    if not s['final_test_permitted'] or s['test_used_for_selection']:raise ValueError('Selection not frozen')
    root=Path('backups/azure').resolve();target=Path('data/stem_refiner_holdout_v1').resolve();manifest=target/'manifest.json'
    if manifest.exists():raise ValueError('Holdout manifest already frozen')
    guitars=[r for r in json.load(open(root/'data/goat_v1/manifest.json')) if r['split']=='test' and r['view']=='amp5']
    # Earlier FiloBass test excerpts were already inspected as regressions.
    # Use fresh, non-overlapping times from the same training-held-out players;
    # disclose that their recordings have been seen in other excerpts.
    previous=[r for r in json.load(open(root/'data/candidate_v2/manifest.json')) if r['dataset']=='previous_filobass']
    basses=[]
    for r in records(root/'data/raw/filobass/FiloBass ISMIR Publication'):
        if r['split']!='test':continue
        duration=7.;start=max(0.,(sf.info(r['audio']).duration-duration)*.1)
        old=next(x for x in previous if x['id']=='duration_'+r['id'])
        if start<old['excerpt_start']+old['duration'] and start+duration>old['excerpt_start']:raise ValueError('Previously inspected bass time overlaps')
        basses.append({**r,'duration':duration,'excerpt_start':start,'parts':{'bass':crop_again(r['notes'],start,duration),'guitar':[]}})
    backgrounds=[r for r in json.load(open(root/'reports/band_stress_v1/acquisition.json'))['selected'] if r['split']=='test']
    if len(guitars)!=7 or not basses or len(backgrounds)!=2:raise ValueError('Unexpected frozen held-out source counts')
    development=json.load(open(root/'data/stem_refiner_v1/manifest.json'))
    if {r['group'] for r in guitars}&{r['group'] for r in development}:raise ValueError('Guitar source overlap')
    key=lambda text:hashlib.sha256(('band2tab-stem-holdout-v1-'+text).encode()).hexdigest();rows=[]
    for guitar in guitars:
        bass=min(basses,key=lambda r:key(guitar['id']+r['id']));background=min(backgrounds,key=lambda r:key(guitar['id']+r['group']))
        duration=min(7.,guitar['duration'],bass['duration']);g_start=(guitar['duration']-duration)*.35;b_start=(bass['duration']-duration)*.35
        item=guitar['group'].removeprefix('goat_');gp=root/'data/raw/goat/GOAT/data'/item/(item+'_amp_5.wav')
        ga=read_stereo(gp,guitar['excerpt_start']+g_start,duration);ba=read_stereo(local(root,bass['audio']),bass['excerpt_start']+b_start,duration)
        h=int(key(guitar['id'])[:8],16);bass_db=(-6.,0.,6.)[h%3];voice_db=(-6.,0.,6.)[(h//3)%3]
        rms=lambda y:max(1e-7,float(np.sqrt(np.mean(y*y))))
        clean=.1*ga/rms(ga)+.1*ba/rms(ba)*10**(bass_db/20)
        paths={Path(m).name:root/'data/raw/musdb_7s'/m for m in background['members']}
        va=read_stereo(paths['vocals.wav'],0.,duration);da=read_stereo(paths['drums.wav'],0.,duration)
        noisy=clean+va/rms(va)*rms(clean)*10**(voice_db/20)+da/rms(da)*rms(clean)*10**((voice_db-3)/20)
        truth=dict(guitar=crop_again(guitar['parts']['guitar'],g_start,duration),bass=crop_again(bass['parts']['bass'],b_start,duration))
        for kind,y in (('clean',clean),('noisy',noisy)):
            ident='held_'+item+'_'+kind;folder=target/'audio'/ident;folder.mkdir(parents=True,exist_ok=True)
            native=folder/'native.wav';y=y*min(1.,.95/max(1e-7,float(np.abs(y).max())));sf.write(native,y,44100,subtype='FLOAT');write_input(folder/'mix.wav',y,44100)
            rows.append(dict(id=ident,group=guitar['group'],split='test',kind='mixture',dataset='held_band_'+kind,duration=duration,guard_seconds=.25,
                audio=str(native),inputs={p:str(folder/(p+'.wav')) for p in ('mix','guitar','bass')},parts=truth,audio_sha256=sha(native),
                components=dict(guitar=guitar['id'],bass=bass['id'],background=background['group']),bass_excerpt_start=bass['excerpt_start']+b_start,
                mixing=dict(bass_to_guitar_rms_db=bass_db,vocals_to_target_rms_db=voice_db,drums_to_target_rms_db=voice_db-3),constructed_unrelated_song_components=True))
    save(manifest,rows);save(target/'protocol.json',dict(selection_sha256=sha(selection),manifest_sha256=sha(manifest),conditions=len(rows),paired_guitar_groups=len(guitars),
        setting='Single frozen comparison; no further tuning on this holdout',natural_band_recording=False,
        source_limitations='GOAT test performances are disjoint recordings, not guaranteed unseen performers or compositions. FiloBass uses fresh times of training-held-out players, but other excerpts from the same recordings were previously inspected. Pretrained model training overlap cannot be ruled out.'))
    print('STEM_HOLDOUT_READY',len(rows),flush=True)


if __name__=='__main__':main()
