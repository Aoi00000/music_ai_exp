"""Controlled validation: known guitar+bass, then add real vocals and drums.

This is a constructed interference test, not a recording of a natural band.
MUSDB bass/other stems are never added without their unknown note labels.
"""
from pathlib import Path
from math import gcd
import argparse,json,hashlib
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from scripts.a_candidate_data import save,sha
from band2tab.separation import write_input

def local(root,path):
    text=str(path)
    if '/music_ai_exp/data/' in text:return root/'data'/text.split('/music_ai_exp/data/',1)[1]
    return Path(text) if Path(text).is_absolute() else root/text

def read_stereo(path,start,duration,sr=44100):
    with sf.SoundFile(path) as f:
        rate=f.samplerate;f.seek(round(start*rate));y=f.read(round(duration*rate),dtype='float32',always_2d=True)
    if y.shape[1]==1:y=np.repeat(y,2,axis=1)
    if y.shape[1]!=2:raise ValueError('Expected mono or stereo source')
    d=gcd(rate,sr);y=resample_poly(y,sr//d,rate//d,axis=0);length=round(duration*sr)
    return np.pad(y[:length],((0,max(0,length-len(y))),(0,0)))

def crop_again(notes,start,duration):
    return [{**n,'onset':max(0.,n['onset']-start),'offset':min(start+duration,n['offset'])-start,
        'censored_onset':bool(n.get('censored_onset') or n['onset']<start),
        'censored_offset':bool(n.get('censored_offset') or n['offset']>start+duration)}
        for n in notes if n['onset']<start+duration and n['offset']>start]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('backups/azure'));a=p.parse_args();root=a.root.resolve()
    target=root/'data/band_stress_v1';manifest=target/'manifest.json'
    if manifest.exists():raise ValueError('Stress validation manifest is frozen')
    guitars=[r for r in json.load(open(root/'data/goat_v1/manifest.json')) if r['split']=='val' and r['view']=='amp5']
    basses=[r for r in json.load(open(root/'data/candidate_v2/manifest.json')) if r['split']=='val' and r['dataset']=='filobass']
    backgrounds=[r for r in json.load(open(root/'reports/band_stress_v1/acquisition.json'))['selected'] if r['split']=='val']
    if len(guitars)!=8 or len(basses)!=7 or len(backgrounds)!=2:raise ValueError('Unexpected frozen validation source counts')
    key=lambda text:hashlib.sha256(('band2tab-band-stress-v1-'+text).encode()).hexdigest()
    rows=[]
    for guitar in guitars:
        bass=min(basses,key=lambda r:key(guitar['id']+r['id']));background=min(backgrounds,key=lambda r:key(guitar['id']+r['group']))
        duration=min(7.,guitar['duration'],bass['duration']);g_start=(guitar['duration']-duration)*.35;b_start=(bass['duration']-duration)*.35
        item=guitar['group'].removeprefix('goat_');g_file=root/'data/raw/goat/GOAT/data'/item/(item+'_amp_5.wav')
        b_file=local(root,bass['audio']);g=read_stereo(g_file,guitar['excerpt_start']+g_start,duration);b=read_stereo(b_file,b_start,duration)
        h=int(key(guitar['id'])[:8],16);bass_db=(-6.,0.,6.)[h%3];voice_db=(-6.,0.,6.)[(h//3)%3]
        rms=lambda y:max(1e-7,float(np.sqrt(np.mean(y*y))))
        clean=.1*g/rms(g)+.1*b/rms(b)*10**(bass_db/20)
        sources={Path(m).name:root/'data/raw/musdb_7s'/m for m in background['members']}
        v=read_stereo(sources['vocals.wav'],0.,duration);d=read_stereo(sources['drums.wav'],0.,duration)
        noisy=clean+v/rms(v)*rms(clean)*10**(voice_db/20)+d/rms(d)*rms(clean)*10**((voice_db-3)/20)
        truth={'guitar':crop_again(guitar['parts']['guitar'],g_start,duration),
               'bass':crop_again(bass['parts']['bass'],b_start,duration)}
        for name,y in (('clean',clean),('noisy',noisy)):
            ident='stress_'+item+'_'+name;folder=Path('data/band_stress_v1/audio')/ident
            native=root/folder/'native.wav';native.parent.mkdir(parents=True,exist_ok=True)
            y=y*min(1.,.95/max(1e-7,float(np.abs(y).max())));sf.write(native,y,44100,subtype='FLOAT')
            canonical=folder/'mix.wav';write_input(root/canonical,y,44100)
            rows.append({'id':ident,'group':guitar['group'],'split':'val','kind':'mixture','dataset':'band_stress_'+name,
                'duration':duration,'guard_seconds':.25,'audio':str(folder/'native.wav'),'inputs':{'mix':str(canonical)},
                'parts':truth,'audio_sha256':sha(native),'components':{'guitar':guitar['id'],'bass':bass['id'],'background':background['group']},
                'mixing':{'bass_to_guitar_rms_db':bass_db,'vocals_to_target_rms_db':voice_db,'drums_to_target_rms_db':voice_db-3},
                'constructed_unrelated_song_components':True})
    save(manifest,rows);save(root/'reports/band_stress_v1/protocol.json',{'manifest_sha256':sha(manifest),'conditions':len(rows),
        'unique_guitar_validation_items':len(guitars),'test_used_for_selection':False,'validation_only':True,
        'purpose':'Compare clean known-target mix with real vocal/drum interference and separation routes',
        'natural_band_recording':False,'target_note_labels':'Only GOAT and FiloBass performance-aligned MIDI; MUSDB supplies non-target audio only',
        'original_timing_and_pitch_labels_preserved':True})
    print('BAND_STRESS_READY',len(rows),flush=True)

if __name__=='__main__':main()
