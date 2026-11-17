"""Additional real data: training-only TECHS DI; untouched IDMT external test."""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from .audio import features,note_targets

def alignment(audio,notes):
    import soundfile as sf
    y,sr=sf.read(audio,dtype='float32');y=y.mean(1) if y.ndim==2 else y
    step=max(1,round(sr*.005));n=len(y)//step
    rms=np.sqrt((y[:n*step].reshape(n,step)**2).mean(1));dt=step/sr
    # Locate attacks near annotated events. Only training annotation cleanup uses GT.
    strength=np.maximum(0,rms-np.r_[np.zeros(2),rms[:-2]])
    anchors=[]
    for onset in sorted(n['onset'] for n in notes):
        if not anchors or onset-anchors[-1]>.15:anchors.append(onset)
    lags=[]
    for t in anchors:
        a=max(0,round((t-.15)/dt));b=min(len(strength),round((t+.15)/dt)+1)
        if t<.2 or a>=b:continue
        k=a+int(np.argmax(strength[a:b]));lags.append(k*dt-t)
    median=float(np.median(lags)) if lags else 0.;mad=float(np.median(np.abs(np.array(lags)-median))) if lags else 1.
    return {'shift_seconds':median,'median_absolute_residual_seconds':mad,'anchors':len(lags),'accepted':len(lags)>=8 and mad<=.04 and abs(median)<=.12}

def prepare_techs(root='data',manifest_name='manifest_techs.json'):
    import pretty_midi,soundfile as sf
    root=Path(root);folder=root/'raw/guitar_techs';added=[];excluded=[]
    for audio in sorted(folder.rglob('directinput_*.wav')):
        if not audio.relative_to(folder).parts[0].startswith('P1_'):continue
        group=audio.relative_to(folder).parts[0];midi=folder/group/'midi'/audio.name.replace('directinput_','midi_').replace('.wav','.mid')
        if not midi.exists():excluded.append({'audio':str(audio),'reason':'missing_midi'});continue
        m=pretty_midi.PrettyMIDI(str(midi));notes=[{'pitch':n.pitch,'onset':n.start,'offset':n.end} for ins in m.instruments if not ins.is_drum for n in ins.notes]
        if not notes or any(not 23<=n['pitch']<=88 for n in notes):excluded.append({'audio':str(audio),'reason':'empty_or_out_of_range'});continue
        lag=alignment(audio,notes)
        if not lag['accepted']:excluded.append({'audio':str(audio),'reason':'uncertain_alignment',**lag});continue
        duration=sf.info(audio).duration
        for n in notes:
            n['onset']=max(0,n['onset']+lag['shift_seconds']);n['offset']=min(duration,n['offset']+lag['shift_seconds'])
        if any(n['offset']<=n['onset'] for n in notes):excluded.append({'audio':str(audio),'reason':'invalid_time'});continue
        ident='techs_'+group+'_'+audio.stem;cache=root/'cache'/(ident+'.npz')
        if not cache.exists():
            x=features(audio);y,outside=note_targets(notes,len(x));assert not outside
            tmp=cache.with_suffix('.tmp.npz');np.savez_compressed(tmp,x=x,y=y);tmp.replace(cache)
        added.append({'id':ident,'dataset':'guitar_techs','part':'guitar','split':'train','group':'techs_'+group,'audio':str(audio.resolve()),'cache':str(cache.resolve()),'notes':notes,'alignment':lag,'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest()})
        print('Prepared',ident,len(notes),lag,flush=True)
    if not added:raise ValueError('No additional training examples passed alignment checks')
    base=json.loads((root/'manifest_clean.json').read_text());weight=sum(it['part']=='guitar' and it['split']=='train' for it in base)/(3*len(added))
    for it in added:it['sampling_weight']=weight
    (root/manifest_name).write_text(json.dumps(base+added,indent=2))
    (root/(Path(manifest_name).stem+'_preparation.json')).write_text(json.dumps({'added':len(added),'excluded':excluded,'sampling_guitar_domain_probability':.25,'new_data_training_only':True,'alignment_uses_training_annotations_only':True},indent=2))
    print('TECHS_READY',len(added),flush=True)

def prepare_idmt(root='data'):
    import soundfile as sf
    root=Path(root);folder=root/'raw/idmt_bass_tracks';items=[]
    for xml in sorted((folder/'annotation').glob('*.xml')):
        tree=ET.parse(xml);audio=folder/'audio'/tree.findtext('.//audioFileName');duration=sf.info(audio).duration
        notes=[{'pitch':int(e.findtext('pitch')),'onset':float(e.findtext('onsetSec')),'offset':float(e.findtext('offsetSec')),'expression':e.findtext('expressionStyle')} for e in tree.findall('.//transcription/event')]
        if any(not(23<=n['pitch']<=88 and 0<=n['onset']<n['offset']<=duration+.1) for n in notes):raise ValueError(f'Invalid IDMT labels: {xml}')
        ident='idmt_bass_'+xml.stem;cache=root/'cache'/(ident+'.npz')
        if not cache.exists():
            x=features(audio);y,outside=note_targets(notes,len(x));assert not outside;np.savez_compressed(cache,x=x,y=y)
        items.append({'id':ident,'dataset':'idmt_bass_tracks','part':'bass','split':'test','group':'idmt_bass_'+xml.stem,'audio':str(audio.resolve()),'cache':str(cache.resolve()),'notes':notes})
    if len(items)!=17:raise ValueError('Expected all 17 external bass tracks')
    (root/'manifest_external_idmt.json').write_text(json.dumps(items,indent=2));print('IDMT_READY',len(items),flush=True)

def prepare_external(root='data'):
    import soundfile as sf
    root=Path(root);items=json.loads((root/'manifest_external_idmt.json').read_text());folder=root/'raw/idmt_guitar/IDMT-SMT-GUITAR_V2/dataset3'
    for xml in sorted((folder/'annotation').glob('*.xml')):
        tree=ET.parse(xml);audio=folder/'audio'/tree.findtext('.//audioFileName')
        # Published XML typo: promenade_mono.xml names a nonexistent poly WAV.
        if xml.stem=='promenade_mono' and audio.name=='promenade_poly.wav':audio=folder/'audio'/'promenade_mono.wav'
        duration=sf.info(audio).duration
        notes=[{'pitch':int(e.findtext('pitch')),'onset':float(e.findtext('onsetSec')),'offset':float(e.findtext('offsetSec'))} for e in tree.findall('.//transcription/event')]
        if any(not(23<=n['pitch']<=88 and 0<=n['onset']<n['offset']<=duration+.1) for n in notes):raise ValueError(f'Invalid guitar labels: {xml}')
        ident='idmt_guitar_'+xml.stem;cache=root/'cache'/(ident+'.npz');x=features(audio);y,outside=note_targets(notes,len(x));assert not outside;np.savez_compressed(cache,x=x,y=y)
        group=xml.stem.replace('_mono','').replace('_poly','')
        items.append({'id':ident,'dataset':'idmt_guitar','part':'guitar','split':'test','group':'idmt_guitar_'+group,'audio':str(audio.resolve()),'cache':str(cache.resolve()),'notes':notes})
    if len(items)!=22:raise ValueError('Expected 17 bass + 5 guitar recordings')
    (root/'manifest_external.json').write_text(json.dumps(items,indent=2));print('EXTERNAL_READY',len(items),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['techs','techs-full','idmt','external']);a=p.parse_args()
    {'techs':prepare_techs,'techs-full':lambda:prepare_techs(manifest_name='manifest_techs_full.json'),'idmt':prepare_idmt,'external':prepare_external}[a.command]()
