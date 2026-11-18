"""Add only training amp2--5 views; leave all validation/test cases unchanged."""
from pathlib import Path
import json,collections,argparse
import soundfile as sf,pretty_midi
from band2tab.separation import write_input
from scripts.a_candidate_data import sha,save
from scripts.a_temporal_data import read_clip,clipped_truth

def main():
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');a=p.parse_args()
    if a.select:
        protocol=json.loads(Path('reports/goat/data_protocol.json').read_text())
        catalog={r['name']:r for r in json.loads(Path('reports/goat/catalog.json').read_text())['members']}
        members=['GOAT/data/'+r['item']+'/'+r['item']+'_amp_'+str(i)+'.wav'
            for r in protocol['sources'] if r['split']=='train' for i in (2,3,4,5)]
        record={'members':members,'source_protocol':'reports/goat/data_protocol.json','source_items':47,'split':'train_only',
            'purpose':'Additional amplifier variants of the same 47 valid training performances; no new independent performances',
            'extracted_bytes':sum(catalog[m]['bytes'] for m in members),'compressed_bytes':sum(catalog[m]['compressed_bytes'] for m in members),'test_used_for_selection':False}
        dest=Path('reports/goat/amp_extra_selection.json')
        if dest.exists():
            if json.loads(dest.read_text())!=record:raise ValueError('Existing amp selection differs')
        else:save(dest,record)
        print('GOAT_EXTRA_AMP_SELECTION',len(members),record['extracted_bytes'],flush=True);return
    root=Path('data/goat_v2');dest=root/'manifest.json'
    if dest.exists():raise ValueError('Expanded amp manifest is frozen')
    original=Path('data/goat_v1/manifest.json');rows=json.loads(original.read_text())
    protocol=json.loads(Path('reports/goat/data_protocol.json').read_text());added=[]
    for source in protocol['sources']:
        if source['split']!='train':continue
        item=source['item'];folder=Path('data/raw/goat/GOAT/data')/item
        notes=[{'pitch':int(n.pitch),'onset':float(n.start),'offset':float(n.end)}
            for i in pretty_midi.PrettyMIDI(str(folder/(item+'_fine_aligned.mid'))).instruments if not i.is_drum for n in i.notes]
        templates=[r for r in rows if r['split']=='train' and r['group']=='goat_'+item and r['view']=='di']
        for index in (2,3,4,5):
            path=folder/(item+'_amp_'+str(index)+'.wav')
            if abs(sf.info(path).duration-source['duration'])>.05:raise ValueError('New amp view duration differs '+item)
            for template in templates:
                ident=template['id'].replace('_di_','_amp'+str(index)+'_');out=root/'audio'/(ident+'.wav')
                write_input(out,read_clip(path,template['excerpt_start'],template['duration']),16000);actual=sf.info(out).duration
                if abs(actual-template['duration'])>.0002:raise ValueError('Amp excerpt timing differs')
                added.append({**template,'id':ident,'dataset':'goat_amp','view':'amp'+str(index),
                    'audio':str(out.resolve()),'inputs':{'mix':str(out.resolve())},'duration':actual,
                    'parts':clipped_truth({'guitar':notes},template['excerpt_start'],actual),'audio_sha256':sha(out)})
    combined=rows+added
    if any(r['split']!='train' for r in added):raise ValueError('Only training variants may be added')
    save(dest,combined);save('reports/goat/expanded_amp_protocol.json',{'base_manifest_sha256':sha(original),
        'manifest_sha256':sha(dest),'added_training_clips':len(added),'training_seconds':sum(r['duration'] for r in combined if r['split']=='train'),
        'counts':dict(collections.Counter(r['split'] for r in combined)),
        'validation_and_test_unchanged':True,'unique_valid_training_performances':47,'amp_versions':5,'test_used_for_selection':False})
    print('GOAT_AMP_EXPANDED',len(added),collections.Counter(r['split'] for r in combined),flush=True)

if __name__=='__main__':main()
