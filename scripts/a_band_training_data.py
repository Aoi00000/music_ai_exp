"""Training-only controlled mixtures from approved existing labelled sources."""
from pathlib import Path
from collections import defaultdict
import argparse, hashlib, json
import numpy as np
from scripts.a_stress_data import local, read_stereo, crop_again
from scripts.a_candidate_data import save, sha
from band2tab.separation import write_input


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('backups/azure'));a=p.parse_args();root=a.root.resolve()
    manifest=root/'data/band_training_v1/manifest.json'
    if manifest.exists():raise ValueError('Training mixtures already frozen')
    goat=json.load(open(root/'data/goat_v2/manifest.json'))
    basses=json.load(open(root/'data/yourmt3_lora_extra/manifest.json'))
    backgrounds=[r for r in json.load(open(root/'reports/band_stress_v1/acquisition.json'))['selected'] if r['split']=='train']
    if any(r['split']!='train' for r in basses) or len(backgrounds)!=6:raise ValueError('Training sources required')
    forbidden_g={r['group'] for r in goat if r['split']!='train'}
    forbidden_b={r['group'] for r in json.load(open(root/'data/candidate_v2/manifest.json')) if r['split']!='train'}
    if any(r['group'] in forbidden_b for r in basses):raise ValueError('Bass source leakage')
    key=lambda s:hashlib.sha256(('band2tab-band-training-v1-'+s).encode()).hexdigest()
    by=defaultdict(list)
    for r in goat:
        if r['split']=='train':by[r['group']].append(r)
    if set(by)&forbidden_g or len(by)!=47:raise ValueError('Unexpected guitar source split')
    rows=[]
    for group,items in sorted(by.items()):
        # Two fixed timbres/excerpts per performance, not two new performances.
        for guitar in sorted(items,key=lambda r:key(r['id']))[:2]:
            bass=min(basses,key=lambda r:key(guitar['id']+r['id']))
            background=min(backgrounds,key=lambda r:key(guitar['id']+r['group']))
            duration=7.;g_start=(guitar['duration']-duration)*.35;b_start=(bass['duration']-duration)*.35
            if min(g_start,b_start)<0:raise ValueError('Short training source')
            g=read_stereo(local(root,guitar['inputs']['mix']),g_start,duration,16000)
            b=read_stereo(local(root,bass['inputs']['mix']),b_start,duration,16000)
            h=int(key(guitar['id'])[:8],16);bass_db=(-6.,0.,6.)[h%3];voice_db=(-6.,0.,6.)[(h//3)%3]
            rms=lambda y:max(1e-7,float(np.sqrt(np.mean(y*y))))
            clean=.1*g/rms(g)+.1*b/rms(b)*10**(bass_db/20)
            paths={Path(m).name:root/'data/raw/musdb_7s'/m for m in background['members']}
            v=read_stereo(paths['vocals.wav'],0,duration,16000);d=read_stereo(paths['drums.wav'],0,duration,16000)
            noisy=clean+v/rms(v)*rms(clean)*10**(voice_db/20)+d/rms(d)*rms(clean)*10**((voice_db-3)/20)
            truth={'guitar':crop_again(guitar['parts']['guitar'],g_start,duration),'bass':crop_again(bass['parts']['bass'],b_start,duration)}
            for condition,y in (('clean',clean),('noisy',noisy)):
                ident='band_'+guitar['id']+'_'+condition;path=root/'data/band_training_v1/audio'/(ident+'.wav')
                y=y*min(1.,.95/max(1e-7,float(np.abs(y).max())));write_input(path,y,16000)
                rows.append({'id':ident,'group':group,'split':'train','kind':'mixture','dataset':'band_training_'+condition,
                    'duration':duration,'guard_seconds':.25,'inputs':{'mix':str(path)},'audio':str(path),'parts':truth,
                    'components':{'guitar':guitar['id'],'bass':bass['id'],'background':background['group']},
                    'component_groups':{'guitar':group,'bass':bass['group'],'background':background['group']},
                    'mixing':{'bass_to_guitar_rms_db':bass_db,'vocals_to_target_rms_db':voice_db,'drums_to_target_rms_db':voice_db-3},
                    'constructed_unrelated_song_components':True,'audio_sha256':sha(path)})
    save(manifest,rows)
    save(root/'reports/band_training_v1/protocol.json',{'manifest_sha256':sha(manifest),'train_conditions':len(rows),
        'train_seconds':sum(r['duration'] for r in rows),'guitar_performances':len(by),'background_songs':len(backgrounds),
        'all_components_train_only':True,'natural_band_recording':False,'test_used_for_selection':False,
        'source_sha256':{p:sha(root/p) for p in ('data/goat_v2/manifest.json','data/yourmt3_lora_extra/manifest.json','reports/band_stress_v1/acquisition.json')},
        'labels':'Retain aligned guitar/bass notes and boundary censoring. No MIDI is inferred from the unlabelled MUSDB stems.'})
    print('BAND_TRAINING_READY',len(rows),flush=True)


if __name__=='__main__':main()
