"""Targeted real jazz-guitar clips from existing note-labelled GuitarSet."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,numpy as np,soundfile as sf
from band2tab.separation import write_input
from scripts.a_stress_data import local,crop_again
from scripts.a_candidate_data import save,sha


def main():
    root=Path('backups/azure').resolve();dest=Path('data/jazz_guitar_v1');manifest=dest/'manifest.json'
    if manifest.exists():raise ValueError('Jazz data protocol is frozen')
    originals=json.load(open(root/'data/manifest_v2_techs_full.json'));groups=defaultdict(list)
    for r in originals:
        if r.get('dataset')=='guitarset' and '_Jazz' in r['id'] and r['id'].endswith('_comp'):
            player=r['id'].split('_')[1];groups[(r['group'],player)].append(r)
    rows=[]
    for (group,player),items in sorted(groups.items()):
        r=min(items,key=lambda n:hashlib.sha256(('jazz-data-v1-'+n['id']).encode()).hexdigest())
        split='val' if group=='guitarset_Jazz3' else 'train';audio=local(root,r['audio']);info=sf.info(audio);duration=7.;start=max(0,(info.duration-duration)*.35)
        with sf.SoundFile(audio) as f:f.seek(round(start*info.samplerate));y=f.read(round(duration*info.samplerate),dtype='float32',always_2d=True)
        ident='jazz_'+r['id'];folder=dest/'audio'/ident;folder.mkdir(parents=True,exist_ok=True);native=folder/'native.wav';sf.write(native,y,info.samplerate,subtype='FLOAT')
        write_input(folder/'mix.wav',y,info.samplerate);write_input(folder/'guitar.wav',y,info.samplerate);write_input(folder/'bass.wav',np.zeros_like(y),info.samplerate)
        rows.append(dict(id=ident,group=group,player=player,split=split,dataset='jazz_guitarset',kind='solo',audio=str(native.resolve()),duration=duration,guard_seconds=.25,
            excerpt_start=start,source_id=r['id'],source_audio_sha256=sha(audio),inputs={p:str((folder/(p+'.wav')).resolve()) for p in ('mix','guitar','bass')},
            parts={'guitar':crop_again(r['notes'],start,duration),'bass':[]}))
    if len(rows)!=18 or sum(r['split']=='train' for r in rows)!=12:raise ValueError('Expected 12 training / 6 validation jazz recordings')
    save(manifest,rows);save('reports/jazz_guitar_v1/protocol.json',dict(manifest_sha256=sha(manifest),source_manifest_sha256=sha(root/'data/manifest_v2_techs_full.json'),
        train_progressions=['Jazz1','Jazz2'],validation_progression='Jazz3',performers_disjoint=False,
        scope='New targeted dataset for the correction experiment, not new independent recordings. Original project classified all Jazz groups as train; upstream models/older model families may have seen them. Do not call this a wholly unseen project test.',
        hypothesis='Diagnose jazz-guitar note transcription and dense simultaneous pitch sets; chord-name recognition is a separate task.',test_used=False))
    print('JAZZ_TARGETED_DATA_READY',len(rows),flush=True)


if __name__=='__main__':main()
