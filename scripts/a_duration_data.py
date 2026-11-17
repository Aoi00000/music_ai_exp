"""Freeze expanded training and a new P3-player holdout before model inference."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,zipfile
import soundfile as sf
import pretty_midi
from band2tab.data import slakh_items
from band2tab.adapt import alignment
from band2tab.filobass import records as filobass_records

p=argparse.ArgumentParser();p.add_argument('--base-only',action='store_true');a=p.parse_args()
root=Path('data');out=root/('manifest_duration_preparation.json' if a.base_only else 'manifest_duration.json')
if out.exists():raise RuntimeError('Duration protocol already frozen; do not overwrite')
folder=root/'duration_clips';folder.mkdir(exist_ok=True)
base=json.loads((root/'manifest_v2_techs_full.json').read_text())
sources={it['id']:it for it in base if it['dataset'] in ('guitarset','guitar_techs')}
mixtures=defaultdict(list)
for it in slakh_items(root/'raw/baby','baby'):mixtures[it['group']].append(it)
for group,items in mixtures.items():
    audio=Path(items[0]['audio']).parent.parent/'mix.wav'
    sources['mix_'+audio.parent.name]={'id':'mix_'+audio.parent.name,'group':group,'audio':str(audio),'split':items[0]['split'],'parts':{part:[n for it in items if it['part']==part for n in it['notes']] for part in ('guitar','bass')}}
rows=[];seen=set();quality=[]
def clip(source,ident,fraction=.35,reuse=None,small=False):
    audio=Path(source['audio']);info=sf.info(audio);start=max(0,(info.duration-12)*fraction)
    if reuse:
        path=Path(reuse['audio']);start=reuse['excerpt_start'];duration=reuse['duration']
    else:
        path=folder/(ident+'.wav')
        with sf.SoundFile(audio) as f:
            f.seek(round(start*info.samplerate));y=f.read(round(12*info.samplerate),dtype='float32')
        sf.write(path,y,info.samplerate,subtype='PCM_16');duration=len(y)/info.samplerate
    end=start+duration;parts=source.get('parts',{source.get('part','guitar'):source.get('notes',[])})
    truth={part:[{**n,'onset':max(0,n['onset']-start),'offset':min(end,n['offset'])-start,
                  'censored_onset':bool(n['onset']<start),'censored_offset':bool(n['offset']>end)}
                 for n in parts.get(part,[]) if n['onset']<end and n['offset']>start and 23<=n['pitch']<=88] for part in ('guitar','bass')}
    row={'id':ident,'audio':str(path.resolve()),'group':source['group'],'split':source['split'],
         'kind':'mixture' if 'parts' in source else 'solo','parts':truth,'duration':duration,'excerpt_start':start,
         'small_training':small,'source_dataset':source.get('dataset','baby'),'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if reuse:row['reuse_prediction']=str((Path('reports/refiner_base')/(reuse['id']+'.mid')).resolve())
    rows.append(row);seen.add(source['id'])
old=json.loads((root/'manifest_refiner.json').read_text())
for it in old:
    key=it['id'].removeprefix('ref_');source=sources[key]
    clip(source,it['id'],reuse=it,small=it['split']=='train')
for key,source in sorted(sources.items()):
    if not key.startswith('mix_') or source['split']=='test':continue
    for fraction in (.1,.7):clip(source,'duration_'+key+'_'+str(int(fraction*100)),fraction)
for dataset,count in [('guitarset',80),('guitar_techs',10000)]:
    pool=[it for it in base if it['dataset']==dataset and it['split']=='train']
    pool.sort(key=lambda it:hashlib.sha256(it['id'].encode()).hexdigest())
    for it in pool[:count]:
        if it['id'] not in seen:clip(it,'duration_'+it['id'])
if a.base_only:
    out.write_text(json.dumps(rows,indent=2));print('BASE_AUDIO_PREPARATION',len(rows),flush=True);raise SystemExit(0)
techs=root/'raw/guitar_techs'
def prepare_real(prefix,split):
    accepted=[]
    for audio in sorted(techs.glob(prefix+'*/audio/directinput/*.wav')):
        group=audio.relative_to(techs).parts[0];midi=techs/group/'midi'/audio.name.replace('directinput_','midi_').replace('.wav','.mid')
        if not midi.exists():quality.append({'audio':str(audio),'accepted':False,'reason':'missing_midi'});continue
        notes=[{'pitch':int(n.pitch),'onset':float(n.start),'offset':float(n.end)} for ins in pretty_midi.PrettyMIDI(str(midi)).instruments if not ins.is_drum for n in ins.notes]
        lag=alignment(audio,notes);duration=sf.info(audio).duration
        if not lag['accepted'] or not notes or any(not 23<=n['pitch']<=88 for n in notes):
            quality.append({'audio':str(audio),'accepted':False,'reason':'alignment_or_range',**lag});continue
        raw_notes=[dict(n) for n in notes]
        for n in notes:n['onset']=max(0,n['onset']+lag['shift_seconds']);n['offset']=min(duration,n['offset']+lag['shift_seconds'])
        if any(n['onset']>=n['offset'] for n in notes):quality.append({'audio':str(audio),'accepted':False,'reason':'invalid_time',**lag});continue
        it={'id':'duration_'+group+'_'+audio.stem,'audio':str(audio.resolve()),'group':'techs_'+group,'split':split,'notes':notes,'raw_notes':raw_notes,'alignment':lag}
        quality.append({'audio':str(audio),'accepted':True,'notes':len(notes),**lag});accepted.append(it)
    return accepted
for it in prepare_real('P2_', 'train'):clip(it,it['id'])
# Local whole-archive checksum + member CRCs before annotation QA. No model output.
archive=techs/'P3_music.zip';metadata=json.loads(Path('reports/guitar_techs_record.json').read_text());spec=next(f for f in metadata['files'] if f['name']==archive.name)
assert hashlib.md5(archive.read_bytes()).hexdigest()==spec['checksum'].split(':')[-1]
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        p=Path(info.filename)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe ZIP member')
        if info.is_dir() or p.name.startswith('._') or '__MACOSX' in p.parts:continue
        if ('/audio/directinput/' in info.filename and p.suffix=='.wav') or ('/midi/' in info.filename and p.suffix=='.mid'):
            dest=techs/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(info))
for it in prepare_real('P3_', 'test'):
    # Full recordings, no artificial onset/offset boundaries in the new test.
    path=Path(it['audio']);rows.append({'id':it['id'],'audio':it['audio'],'group':it['group'],'split':'test','kind':'solo',
        'parts':{'guitar':it['notes'],'bass':[]},'unshifted_parts':{'guitar':it['raw_notes'],'bass':[]},'annotation_alignment':it['alignment'],
        'duration':sf.info(path).duration,'excerpt_start':0.,'small_training':False,
        'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
# Real double-bass stems: artist-disjoint split, no inference used to pick tracks.
filo=filobass_records(root/'raw/filobass/FiloBass ISMIR Publication')
filo_ledger=[]
for it in filo:
    clip(it,'duration_'+it['id'])
    solo=rows[-1]
    if it['split']=='test':solo['evaluation']='new_filobass_player_solo'
    # Constructed real-instrument mixes, NOT recordings of a natural band.
    pool=[g for g in base if g['dataset']=='guitarset' and g['split']==it['split']]
    pool.sort(key=lambda g:hashlib.sha256((it['id']+g['id']).encode()).hexdigest())
    guitar=pool[0];clip(guitar,'duration_filo_partner_'+it['id'])
    partner=rows.pop()
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd
    def mono16(path):
        y,sr=sf.read(path,dtype='float32',always_2d=True);y=y.mean(1);d=gcd(sr,16000)
        return resample_poly(y,16000//d,sr//d)
    bass=mono16(solo['audio']);gt=mono16(partner['audio']);size=min(len(bass),len(gt));bass=bass[:size];gt=gt[:size]
    level=(-6,0,6)[int(hashlib.sha256(it['id'].encode()).hexdigest()[:8],16)%3]
    rms=lambda y:max(1e-7,float(np.sqrt(np.mean(y*y))))
    mix=bass/rms(bass)*10**(level/20)+gt/rms(gt);mix=.9*mix/max(1.,float(np.abs(mix).max()))
    path=folder/('duration_filo_mix_'+it['id']+'.wav');sf.write(path,mix,16000,subtype='PCM_16')
    duration=size/16000;truth={}
    for part,source in [('guitar',partner),('bass',solo)]:
        truth[part]=[{**n,'offset':min(duration,n['offset'])} for n in source['parts'][part] if n['onset']<duration]
    rows.append({'id':'duration_filo_mix_'+it['id'],'audio':str(path.resolve()),'group':it['group'],'split':it['split'],'kind':'mixture',
                 'parts':truth,'duration':duration,'excerpt_start':0.,'small_training':False,'source_dataset':'filobass_guitarset_mix',
                 'guitar_group':guitar['group'],'bass_to_guitar_rms_db':level,'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 **({'evaluation':'new_filobass_player_constructed_mix'} if it['split']=='test' else {})})
    filo_ledger.append({'id':it['id'],'player':it['player'],'split':it['split'],'source_programs':it['source_programs'],'midi_variant':'midi_fully_aligned','semitone_shift':0})
groups=defaultdict(set)
for it in rows:
    groups[it['group']].add(it['split'])
    if 'guitar_group' in it:groups[it['guitar_group']].add(it['split'])
assert all(len(v)==1 for v in groups.values())
assert sum(it['split']=='test' for it in rows)>0
out.write_text(json.dumps(rows,indent=2))
protocol={'created_date_jst':'2026-11-18','manifest_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'counts':{str(k):v for k,v in Counter((r['split'],r['kind']) for r in rows).items()},
          'quality':quality,'filobass':filo_ledger,'new_test':'Guitar-TECHS P3 full recordings and FiloBass artist-disjoint 12s solo/constructed mix excerpts; not natural full band',
          'annotation_qa':'fixed onset alignment using supplied MIDI, before any test model inference; no prediction-based exclusion',
          'training_conditions':['duration_small','duration_expanded'],'occupancy_weight':.8,'seed':17,
          'epochs':[25,50,100,150],'thresholds':[.75,.9,.97],'margins':[.2,.4],'allow_removal':[False,True],
          'selection':'validation only; maximize mean strict Note+Frame F1; no domain or instrument Note/Frame degradation vs identity; identity allowed',
          'automatic_application_approved':False,'test_used_for_selection':False}
Path('reports/duration_protocol.json').write_text(json.dumps(protocol,indent=2));print('DURATION_DATA_FROZEN',protocol['counts'],flush=True)
