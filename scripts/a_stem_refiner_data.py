"""Native-bandwidth, source-disjoint training for a learned Mu note corrector."""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib
import numpy as np,soundfile as sf
from band2tab.separation import write_input
from scripts.a_stress_data import read_stereo
from scripts.a_candidate_data import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args();root=a.root.resolve()
    dest=root/'data/stem_refiner_v1';manifest=dest/'manifest.json'
    if manifest.exists():raise ValueError('Stem-refiner manifest already frozen')
    band=json.load(open(root/'data/band_training_v1/manifest.json'));goat={r['id']:r for r in json.load(open(root/'data/goat_v2/manifest.json'))}
    bass={r['id']:r for r in json.load(open(root/'data/yourmt3_lora_extra/manifest.json'))}
    backgrounds={r['group']:r for r in json.load(open(root/'reports/band_stress_v1/acquisition.json'))['selected']}
    key=lambda s:hashlib.sha256(('stem-refiner-v1-'+s).encode()).hexdigest();groups=defaultdict(list)
    for r in band:groups[r['group']].append(r)
    rows=[]
    for group,items in sorted(groups.items()):
        guitar_id=min({r['components']['guitar'] for r in items},key=key)
        for old in items:
            if old['components']['guitar']!=guitar_id:continue
            g=goat[guitar_id];b=bass[old['components']['bass']];bg=backgrounds[old['components']['background']]
            if any(r['split']!='train' for r in (old,g,b,bg)):raise ValueError('Non-training component')
            item=g['group'].removeprefix('goat_');suffix='' if g['view']=='di' else '_amp_'+g['view'].removeprefix('amp')
            gp=root/'data/raw/goat/GOAT/data'/item/(item+suffix+'.wav')
            title=b['id'].removeprefix('full_filobass_').rsplit('_',1)[0]
            bp=root/'data/raw/filobass/FiloBass ISMIR Publication/audio_bass_stems'/(title+'.mp3')
            duration=old['duration'];ga=read_stereo(gp,g['excerpt_start']+(g['duration']-duration)*.35,duration)
            ba=read_stereo(bp,b['excerpt_start']+(b['duration']-duration)*.35,duration)
            rms=lambda y:max(1e-7,float(np.sqrt(np.mean(y*y))))
            levels=old['mixing'];clean=.1*ga/rms(ga)+.1*ba/rms(ba)*10**(levels['bass_to_guitar_rms_db']/20);y=clean
            if old['dataset'].endswith('noisy'):
                paths={Path(m).name:root/'data/raw/musdb_7s'/m for m in bg['members']}
                v=read_stereo(paths['vocals.wav'],0,duration);d=read_stereo(paths['drums.wav'],0,duration)
                y=clean+v/rms(v)*rms(clean)*10**(levels['vocals_to_target_rms_db']/20)+d/rms(d)*rms(clean)*10**(levels['drums_to_target_rms_db']/20)
            ident='stem_'+old['id'];folder=dest/'audio'/ident;folder.mkdir(parents=True,exist_ok=True)
            native=folder/'native.wav';y=y*min(1.,.95/max(1e-7,float(np.abs(y).max())));sf.write(native,y,44100,subtype='FLOAT');write_input(folder/'mix.wav',y,44100)
            rows.append({**old,'id':ident,'audio':str(native),'inputs':{k:str(folder/(k+'.wav')) for k in ('mix','guitar','bass')},'audio_sha256':sha(native),
                'native_rendered_from_original_sources':True,'dataset':'stem_band_'+old['dataset'].rsplit('_',1)[-1]})
    augmented=json.load(open(root/'data/bass_augmentation_v1/manifest.json'));by=defaultdict(list)
    for r in augmented:
        if r['split']=='train' and r['dataset'] in ('bass_aug_drive','bass_aug_damped_drive'):by[(r['group'],r['dataset'])].append(r)
    for candidates in by.values():
        old=min(candidates,key=lambda r:key(r['id']));ident='stem_'+old['id'];folder=dest/'audio'/ident;folder.mkdir(parents=True,exist_ok=True)
        y,sr=sf.read(old['audio'],dtype='float32');write_input(folder/'mix.wav',y,sr);write_input(folder/'bass.wav',y,sr);write_input(folder/'guitar.wav',np.zeros_like(y),sr)
        rows.append({**old,'id':ident,'dataset':'stem_aug_bass','known_part':'bass','audio':str(folder/'mix.wav'),
            'inputs':{k:str(folder/(k+'.wav')) for k in ('mix','guitar','bass')},'audio_sha256':sha(folder/'mix.wav')})
    for old in json.load(open(root/'data/band_stress_v1/manifest.json')):
        rows.append({**old,'inputs':{'mix':str(root/old['inputs']['mix']),**{part:str(root/'data/band_stress_v1/separated'/('six_'+part)/(old['id']+'.wav')) for part in ('guitar','bass')}},'audio':str(root/old['audio'])})
    if {r['group'] for r in rows if r['split']=='train'}&{r['group'] for r in rows if r['split']=='val'}:raise ValueError('Refiner guitar source leakage')
    save(manifest,rows);save(root/'reports/stem_refiner_v1/protocol.json',{'manifest_sha256':sha(manifest),'train_conditions':sum(r['split']=='train' for r in rows),
        'validation_conditions':sum(r['split']=='val' for r in rows),'training_guitar_performances':len(groups),'test_used':False,
        'training':'One fixed excerpt/amp per GOAT training performance, clean/noisy pairs; driven/short-decay gold bass stems',
        'inference':'Frozen six-source Demucs then frozen MuScriptor on each stem; learned note keep/onset/release correction',
        'selection':'Development clean/noisy dataset-macro score; same original Mu stem baseline, no IDMT selection',
        'natural_band_recording':False})
    print('STEM_REFINER_DATA_READY',len(rows),flush=True)


if __name__=='__main__':main()
