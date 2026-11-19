"""Chord-name diagnostics separate from AMT; existing validation data only."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np,soundfile as sf,torch,mir_eval
from chordchart.infer import infer
from chordchart.core import signature,REV
from scripts.a_candidate_data import save,sha
from scripts.a_stress_data import local


def measure(ref,est,start,end):
    bounds=sorted({start,end,*[max(start,min(end,float(s[k]))) for seq in (ref,est) for s in seq for k in ('start','end')]});scores=defaultdict(float);seconds=0.;weights=defaultdict(float)
    for a,b in zip(bounds,bounds[1:]):
        t=(a+b)/2;rl=next((s['label'] for s in ref if s['start']<=t<s['end']),'X');el=next((s['label'] for s in est if s['start']<=t<s['end']),'X');rs,es=signature(rl),signature(el)
        if rs is None:continue
        seconds+=b-a;scores['structure_exact']+=(b-a)*int(rs==es)
        for key,fun in [('root',mir_eval.chord.root),('triads',mir_eval.chord.triads),('tetrads_inv',mir_eval.chord.tetrads_inv)]:
            value=float(fun([rl],[el])[0])
            if value>=0:scores[key]+=(b-a)*value;weights[key]+=b-a
    if seconds<=0:raise ValueError('No valid performed-chord reference')
    return {k:v/(seconds if k=='structure_exact' else weights[k]) for k,v in scores.items() if k=='structure_exact' or weights[k]>0}


def main():
    torch.set_num_threads(2);root=Path('backups/azure').resolve();folder=Path('reports/chord_diagnostic_v1');manifest=root/'data/manifest_v2_techs_full.json';groups=defaultdict(list)
    for r in json.load(open(manifest)):
        if r['dataset']=='guitarset' and r['split']=='val' and r['id'].endswith('_comp'):groups[(r['group'],r['id'].split('_')[1])].append(r)
    selected=[min(v,key=lambda r:hashlib.sha256(('chord-diagnostic-v1-'+r['id']).encode()).hexdigest()) for k,v in sorted(groups.items())]
    if len(selected)!=18:raise ValueError('Expected six performers across three validation progression/style groups')
    protocol=dict(selected_ids=[r['id'] for r in selected],selection='One tempo per group x player, fixed SHA256 ordering before prediction',reference='GuitarSet performed chords, annotation version0.9, not instructed lead sheet',
        excerpt_seconds=7.,excerpt_start_fraction=.35,guard_seconds=.25,upstream_revision=REV,model_weights=json.load(open('reports/chord_model_source.json'))['weights_sha256'],
        learning=False,btc_used=False,cqt_crnn_ensemble=True,scope='Repeated project validation data; solo guitar, three styles but shared progression2 template. No natural-band or unseen-composition claim.')
    if (folder/'protocol.json').exists() and json.load(open(folder/'protocol.json'))!=protocol:raise ValueError('Frozen chord diagnostic changed')
    save(folder/'protocol.json',protocol);tracks=[]
    for r in selected:
        source=local(root,r['audio']);audio=folder/'audio'/(r['id']+'.wav');info=sf.info(source);start=max(0,(info.duration-7.)*.35)
        if not audio.exists():
            with sf.SoundFile(source) as f:f.seek(round(start*info.samplerate));y=f.read(round(7*info.samplerate),dtype='float32',always_2d=True)
            audio.parent.mkdir(parents=True,exist_ok=True);sf.write(audio,y,info.samplerate,subtype='FLOAT')
        path=folder/'predictions'/(r['id']+'.json')
        if path.exists():record=json.load(open(path))
        else:record=infer(audio,baseline=True,device='cpu');save(path,record)
        jam=root/'data/raw/guitarset'/(r['id'].removeprefix('guitarset_')+'.jams');annotations=json.load(open(jam))['annotations'];performed=[a for a in annotations if a['namespace']=='chord' and a['annotation_metadata'].get('version')==.9]
        if len(performed)!=1:raise ValueError('No unambiguous performed-chord annotation')
        ref=[dict(start=x['time']-start,end=x['time']+x['duration']-start,label=x['value']) for x in performed[0]['data']]
        score=measure(ref,record['segments'],.25,6.75);tracks.append(dict(id=r['id'],group=r['group'],metrics=score,audio_sha256=sha(audio)));print('CHORD_DIAGNOSTIC_MEASURED',r['id'],score,flush=True)
    grouped=defaultdict(list)
    for r in tracks:grouped[r['group']].append(r['metrics'])
    by={g:{k:float(np.mean([r[k] for r in rs])) for k in tracks[0]['metrics']} for g,rs in grouped.items()};overall={k:float(np.mean([r[k] for r in by.values()])) for k in tracks[0]['metrics']}
    result=dict(summary=overall,groups=by,recordings=len(tracks),scored_seconds=18*6.5,tracks=tracks,protocol_sha256=sha(folder/'protocol.json'),warning=protocol['scope'],learning=False)
    save(folder/'comparison.json',result);save('docs/measurements/chord_diagnostic_20261119.json',{k:v for k,v in result.items() if k!='tracks'});print('CHORD_DIAGNOSTIC_COMPLETE',overall,flush=True)


if __name__=='__main__':main()
