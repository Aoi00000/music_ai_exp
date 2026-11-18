"""Use more of already acquired, artist-disjoint TRAIN bass performances.

Fixed three 20-second excerpts per training recording; no inference-dependent
selection, no validation/test recordings and no pitch-label correction.
"""
from pathlib import Path
import json
from band2tab.filobass import records
from band2tab.separation import write_input
from scripts.a_candidate_data import save,sha
from scripts.a_temporal_data import read_clip,clipped_truth

def main():
    root=Path('data/yourmt3_lora_extra');dest=root/'manifest.json'
    if dest.exists():raise ValueError('Expanded manifest already frozen')
    sources=records('data/raw/filobass/FiloBass ISMIR Publication')
    old=json.loads(Path('data/candidate_v2/manifest.json').read_text())
    forbidden={r['group'] for r in old if r['split']!='train'}
    rows=[];ledger=[]
    import soundfile as sf
    for source in sources:
        if source['split']!='train':continue
        if source['group'] in forbidden:raise ValueError('Performer overlap with held-out data')
        duration=sf.info(source['audio']).duration;length=min(20.,duration)
        ledger.append({'id':source['id'],'group':source['group'],'audio_sha256':sha(source['audio']),
            'midi_sha256':sha(source['midi']),'label_semantics':source['label_semantics']})
        for i,fraction in enumerate((.15,.45,.75)):
            start=max(0.,(duration-length)*fraction);ident='full_'+source['id']+'_'+str(i)
            path=root/'audio'/(ident+'.wav');write_input(path,read_clip(source['audio'],start,length),16000)
            actual=sf.info(path).duration
            rows.append({'id':ident,'group':source['group'],'split':'train','dataset':'filobass','kind':'solo',
                'duration':actual,'excerpt_start':start,'guard_seconds':.25,'audio':str(path.resolve()),
                'inputs':{'mix':str(path.resolve())},'parts':clipped_truth({'bass':source['notes']},start,actual),
                'audio_sha256':sha(path)})
    if len(ledger)!=34 or len(rows)!=102:raise ValueError('Unexpected official training split size')
    save(dest,rows);save(root/'source_ledger.json',{'sources':ledger,'clips':len(rows),
        'total_seconds':sum(r['duration'] for r in rows),'selection':'Fixed fractions of training performers only',
        'test_used_for_selection':False})
    print('EXTRA_TRAIN_BASS',len(rows),sum(r['duration'] for r in rows),flush=True)

if __name__=='__main__':main()
