"""Gold Slakh bass stems: drive/short-decay augmentation with matched labels."""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib
import numpy as np,soundfile as sf
from band2tab.data import slakh_items
from band2tab.augment import soft_drive,damp_isolated
from band2tab.separation import write_input
from scripts.a_temporal_data import clipped_truth,read_clip
from scripts.a_candidate_data import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args();root=a.root.resolve()
    dest=root/'data/bass_augmentation_v1';manifest=dest/'manifest.json'
    if manifest.exists():raise ValueError('Augmentation manifest is frozen')
    old=json.load(open(root/'data/candidate_v2/manifest.json'));allowed={r['group']:r['split'] for r in old if r['dataset']=='slakh_candidate_new' and r['split'] in ('train','val')}
    by=defaultdict(list)
    for directory in ('train','validation'):
        for source in slakh_items(root/'data/raw/slakh_candidate_v2'/directory,'bass_aug_source'):
            if source['part']=='bass' and source['group'] in allowed:by[source['group']].append(source)
    key=lambda s:hashlib.sha256(('bass-aug-v1-'+s).encode()).hexdigest();rows=[];skipped=[]
    for group,sources in sorted(by.items()):
        source=min(sources,key=lambda r:key(r['id']));split=allowed[group]
        full=sf.info(source['audio']).duration;length=min(12.,full)
        for index,fraction in enumerate((.2,.6) if split=='train' else (.35,)):
            start=(full-length)*fraction;y=read_clip(source['audio'],start,length)
            if float(np.std(y))<1e-5:skipped.append({'source':source['id'],'clip':index,'reason':'No varying audio'});continue
            h=int(key(source['id']+str(index))[:8],16);max_seconds=(.10,.12,.16)[h%3];decay=(.025,.035,.045)[(h//3)%3]
            for variant in ('dry','drive','damped','damped_drive'):
                x=y.copy();notes=source['notes']
                if variant.startswith('damped'):
                    try:x,notes=damp_isolated(x,16000,notes,start,max_seconds,decay)
                    except ValueError:
                        skipped.append({'source':source['id'],'clip':index,'reason':'Polyphonic damping ambiguous','variant':variant});continue
                if 'drive' in variant:x=soft_drive(x,drive=(4.,8.,12.)[(h//9)%3])
                ident='aug_'+source['id']+'_'+str(index)+'_'+variant;path=dest/'audio'/(ident+'.wav');write_input(path,x,16000)
                rows.append({'id':ident,'group':group,'split':split,'dataset':'bass_aug_'+variant,'kind':'solo',
                    'audio':str(path),'inputs':{'mix':str(path)},'duration':len(x)/16000,'guard_seconds':.25,
                    'parts':clipped_truth({'guitar':[],'bass':notes},start,len(x)/16000),
                    'source':source['id'],'source_audio_sha256':sha(source['audio']),'audio_sha256':sha(path),
                    'augmentation':{'variant':variant,'synthetic_short_decay_not_real_mute':variant.startswith('damped'),
                        'note_offsets_updated':variant.startswith('damped'),'max_note_seconds':max_seconds,'decay_seconds':decay},
                    'bass_octave_semitones':source['bass_octave_semitones']})
    if {r['group'] for r in rows if r['split']=='train'}&{r['group'] for r in rows if r['split']=='val'}:raise ValueError('Augmentation source leakage')
    save(manifest,rows);save(root/'reports/bass_augmentation_v1/protocol.json',{'manifest_sha256':sha(manifest),
        'counts':{s:sum(r['split']==s for r in rows) for s in ('train','val')},'groups':{s:len({r['group'] for r in rows if r['split']==s}) for s in ('train','val')},
        'source_manifest_sha256':sha(root/'data/candidate_v2/manifest.json'),'test_used':False,'skipped_before_inference':skipped,
        'semantics':'Original gold isolated bass stems, not estimated Demucs bass. Saturation keeps note timing; short-decay synthesis explicitly shortens note releases. Original data unchanged.'})
    print('BASS_AUGMENTATION_READY',len(rows),flush=True)


if __name__=='__main__':main()
