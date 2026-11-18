"""Raw IDMT bass-line acquisition and a fixed development-snapshot evaluation.

The publisher describes this as an evaluation dataset; never add it to training.
"""
from pathlib import Path
import argparse,hashlib,json,time,urllib.request,zipfile
import xml.etree.ElementTree as ET
import soundfile as sf
from scripts.a_candidate_data import save,sha

RECORD='https://zenodo.org/records/7544099'
ARCHIVE='IDMT-SMT-BASS-SINGLE-TRACKS.zip'
MD5='c85540c664e8b13badd86880bacf11fa'
ROOT=Path('backups/azure/data/raw/idmt_bass_lines')
REPORT=Path('reports/idmt_bass_lines')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);archive=ROOT/ARCHIVE
    if not archive.exists():
        temp=archive.with_suffix('.part')
        urllib.request.urlretrieve(RECORD+'/files/'+ARCHIVE+'?download=1',temp)
        temp.replace(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest()!=MD5:raise ValueError('Official archive MD5 mismatch')
    rows=[];files=[]
    with zipfile.ZipFile(archive) as z:
        for number in range(1,18):
            ident=f'{number:03d}'
            for member in ('audio/'+ident+'.wav','annotation/'+ident+'.xml'):
                path=ROOT/member;raw=z.read(member)  # zipfile verifies the member CRC.
                if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing raw source changed')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                files.append({'path':str(path),'bytes':len(raw),'sha256':sha(path)})
            xml=ET.parse(ROOT/'annotation'/(ident+'.xml')).getroot();notes=[]
            tuning=[int(x) for x in xml.findtext('globalParameter/instrumentTuning').split(',')]
            duration=sf.info(ROOT/'audio'/(ident+'.wav')).duration
            for event in xml.findall('transcription/event'):
                n={'pitch':int(event.findtext('pitch')),'onset':float(event.findtext('onsetSec')),
                   'offset':float(event.findtext('offsetSec')),'raw_string':int(event.findtext('stringNumber')),
                   'fret':int(event.findtext('fretNumber')),'technique':event.findtext('expressionStyle'),
                   'plucking':event.findtext('excitationStyle')}
                if not 0<=n['pitch']<=127 or not 0<=n['onset']<n['offset']<=duration+.05:raise ValueError('Invalid raw event '+ident)
                if n['technique']=='NO' and n['pitch']!=tuning[n['raw_string']-1]+n['fret']:raise ValueError('Normal-note string/pitch convention mismatch')
                notes.append(n)
            rows.append({'id':'idmt_bass_line_'+ident,'group':'idmt_bass_line_'+ident,'split':'external_snapshot',
                'dataset':'idmt_bass_lines','kind':'solo','audio':str(ROOT/'audio'/(ident+'.wav')),
                'duration':duration,'guard_seconds':0.,'parts':{'guitar':[],'bass':notes},'raw_tuning_low_to_high':tuning})
    manifest=REPORT/'manifest.json'
    if manifest.exists() and json.load(open(manifest))!=rows:raise ValueError('External manifest changed')
    save(manifest,rows);save(REPORT/'acquisition.json',{'source':RECORD,'publisher':'Fraunhofer IDMT',
        'publisher_purpose':'Evaluation only in this project, not training','declared_license':'CC BY-NC-ND 4.0',
        'official_license_source':'https://www.idmt.fraunhofer.de/en/publications/datasets/bass_lines.html',
        'archive_md5':MD5,'archive_sha256':sha(archive),'all_member_crc_checked':True,'files':files,
        'manifest_sha256':sha(manifest),'recordings':len(rows),'audio_seconds':sum(r['duration'] for r in rows),
        'pretraining_overlap':'Not excluded; this is an external source, not guaranteed unseen by the pretrained base'})
    print('IDMT_BASS_PREPARED',len(rows),flush=True)


def evaluate(tag,adapter):
    import torch
    from band2tab.lora import restore
    from band2tab.score import read_midi
    from scripts.a_candidate_train import metrics,aggregate,bound,KEYS
    from scripts.a_candidate_review import errors,matching
    from scripts.a_yourmt3_predict import load_model,transcribe
    selection=json.load(open('reports/yourmt3_lora/development_selection.json'))
    if adapter and sha(adapter)!=selection['adapter_sha256']:raise ValueError('Only the preselected development adapter may be evaluated')
    rows=json.load(open(REPORT/'manifest.json'));manifest_sha=sha(REPORT/'manifest.json')
    torch.set_num_threads(4);model,info=load_model('cpu');adapter_sha=None
    if adapter:
        state=torch.load(adapter,map_location='cpu',weights_only=True)
        if state['base_sha256']!=info['checkpoint_sha256']:raise ValueError('Adapter/base mismatch')
        restore(model,state);model.eval();adapter_sha=sha(adapter)
    folder=REPORT/tag;items=[];tracks=[]
    for row in rows:
        audio=Path(row['audio']);out=folder/'midi'/(row['id']+'.mid');meta=out.with_suffix('.json')
        fingerprint={'audio_sha256':sha(audio),'manifest_sha256':manifest_sha,'base_sha256':info['checkpoint_sha256'],'adapter_sha256':adapter_sha}
        if out.exists() and meta.exists():
            record=json.load(open(meta))
            if record['fingerprint']!=fingerprint:raise ValueError('Stale external evaluation')
        else:
            start=time.monotonic();decode=transcribe(model,audio,out,'cpu');raw=read_midi(out)
            parts=bound({p:raw.get(p,[]) for p in ('guitar','bass')},row['duration']);scores=metrics(row,parts)
            matched={i for i,j in matching(row['parts']['bass'],parts['bass'])};techniques={}
            for technique in sorted({n['technique'] for n in row['parts']['bass']}):
                ids=[i for i,n in enumerate(row['parts']['bass']) if n['technique']==technique]
                techniques[technique]={'reference_notes':len(ids),'matched':sum(i in matched for i in ids)}
            record={'id':row['id'],'metrics':{k:scores[k] for k in KEYS},'parts':parts,'errors':errors(row,parts),
                'technique_onset_recall_counts':techniques,'fingerprint':fingerprint,'seconds':time.monotonic()-start,'decoding_errors':decode}
            save(meta,record)
        items.append((row,record['metrics']));tracks.append(record)
        print('IDMT_EXTERNAL',tag,row['id'],round(record['metrics']['note_f1'],4),flush=True)
    save(folder/'comparison.json',{'summary':aggregate(items),'tracks':tracks,'adapter_sha256':adapter_sha,
        'selection_sha256':sha('reports/yourmt3_lora/development_selection.json'),'manifest_sha256':manifest_sha,
        'external_snapshot_only':True,'natural_band_recording':False,'test_used_for_selection':False,
        'limitations':['17 short electric-bass lines, not 17 independent performers or full band songs.',
                       'Pretraining overlap is unknown. Do not tune the selected model using these scores.',
                       'All original events retained, including dead notes and pitch-modulated techniques; fixed-note scoring has limitations.']})
    print('IDMT_EXTERNAL_COMPLETE',tag,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','evaluate']);p.add_argument('--tag');p.add_argument('--adapter',type=Path);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:
        if not a.tag:p.error('--tag required')
        evaluate(a.tag,a.adapter)
