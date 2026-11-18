"""Frozen new-source evaluation, timing/error diagnostics and separator controls."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
import torch,pretty_midi,soundfile as sf,mir_eval
from band2tab.temporal import TemporalFusion,predict,decode_parts,activity_parts,predict_scales
from band2tab.refine import candidates_from_midi
from scripts.a_temporal_data import ROOT,REPORT,rows_for
from scripts.a_temporal_train import original,clipped,metrics,load_cache
from scripts.a_separation_fusion import polyphony

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2));tmp.replace(p)
def read_part(r,kind,forced=None):
    notes=candidates_from_midi(REPORT/'midi'/r['id']/(kind+'.mid'))
    if forced:return [{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes]
    return {p:[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes if n['baseline_class']==c] for c,p in ((1,'guitar'),(2,'bass'))}

def diagnostic(r,parts,piano):
    g=r['guard_seconds'];end=r['duration']-g if g else float('inf');result={}
    for part in ('guitar','bass'):
        ref=[n for n in r['parts'].get(part,[]) if g<=n['onset']<end and not n.get('censored_onset')]
        est=[n for n in parts.get(part,[]) if g<=n['onset']<end]
        intervals=lambda ns:np.array([[n['onset'],n['offset']] for n in ns],float).reshape(-1,2)
        pitches=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
        pairs=mir_eval.transcription.match_notes(intervals(ref),pitches(ref),intervals(est),pitches(est),onset_tolerance=.05,offset_ratio=None) if ref and est else []
        used={e for _,e in pairs};unmatched=[n for i,n in enumerate(est) if i not in used]
        other=r['parts'].get('bass' if part=='guitar' else 'guitar',[])
        result[part]={'reference_notes':len(ref),'predicted_notes':len(est),'matched_notes':len(pairs),
            'missed_notes':len(ref)-len(pairs),'extra_notes':len(est)-len(pairs),
            'wrong_role_overlap':sum(any(a['pitch']==n['pitch'] and abs(a['onset']-n['onset'])<=.05 for a in other) for n in unmatched),
            'octave_error_overlap':sum(any(abs(a['pitch']-n['pitch'])==12 and abs(a['onset']-n['onset'])<=.05 for a in ref) for n in unmatched),
            'late_or_early_same_pitch':sum(any(a['pitch']==n['pitch'] and .05<abs(a['onset']-n['onset'])<=.25 for a in ref) for n in unmatched),
            'piano_overlap':sum(any(a['pitch']==n['pitch'] and abs(a['onset']-n['onset'])<=.05 for a in piano) for n in unmatched),
            'onset_abs_error_ms':[1000*abs(ref[a]['onset']-est[b]['onset']) for a,b in pairs],
            'offset_abs_error_ms':[1000*abs(ref[a]['offset']-est[b]['offset']) for a,b in pairs if not ref[a].get('censored_offset')]}
    return result

@torch.no_grad()
def a8_probabilities(encoder,data,device):
    e=data['e'];total=np.zeros_like(data['m'],np.float32);count=np.zeros(len(e),np.float32)
    for start in range(0,len(e),258):
        stop=min(len(e),start+344)
        for j in (0,1):
            z=torch.tensor(e[start:stop,j+1],device=device,dtype=torch.float32)[None]
            embedding=encoder.inst(torch.tensor([j],device=device))[:,None].expand(-1,z.shape[1],-1)
            z,_=encoder.gru(torch.cat([z,embedding],-1));p=encoder.head(z).reshape(1,len(z[0]),3,66).sigmoid()[0].cpu().numpy()
            total[start:stop,j]+=p
        count[start:stop]+=1
    return total/np.maximum(count[:,None,None,None],1)

def main():
    from band2tab.model import NoteCRNN
    out=REPORT/'test_comparison.json'
    if out.exists():raise RuntimeError('Do not reselect after test')
    torch.set_num_threads(4);selection=json.loads((REPORT/'selection.json').read_text())
    if selection['protocol_sha256']!=sha(REPORT/'protocol.json'):raise ValueError('Protocol changed')
    calibration=json.loads((REPORT/'calibration.json').read_text());calibrated=calibration['selected']
    if calibration['base_selection_sha256']!=sha(REPORT/'selection.json'):raise ValueError('Base selection changed')
    models={}
    for name,record in selection['results'].items():
        if sha(record['checkpoint'])!=record['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        state=torch.load(record['checkpoint'],map_location='cpu',weights_only=True);model=TemporalFusion(include_ft=state['include_ft']).to('cuda');model.load_state_dict(state['model']);model.eval();models[name]=(model,state,record)
    encoder=NoteCRNN().to('cuda');encoder.load_state_dict(next(iter(models.values()))[1]['encoder']);encoder.eval()
    other=defaultdict(list)
    for record in json.loads((REPORT/'other_annotation_acquisition.json').read_text()):
        other[record['group']]+=[{'pitch':n.pitch,'onset':n.start,'offset':n.end} for ins in pretty_midi.PrettyMIDI(record['path']).instruments if not ins.is_drum for n in ins.notes]
    results=[];policy={'onset':.5,'frame':.5,'offset':.5,'min_duration':.03}
    for r in rows_for(True):
        data=load_cache(r);parts={'Mu_raw':original(r),'Mu_ft_clean':read_part(r,'ft_clean'),
            'Mu_6s_labels':{p:read_part(r,p)[p] for p in ('guitar','bass')},
            'Mu_6s_forced':{p:read_part(r,p,p) for p in ('guitar','bass')}}
        parts['Mu_audio_bounded']=clipped(parts['Mu_raw'],r['duration'])
        parts['Mu_ft_audio_bounded']=clipped(parts['Mu_ft_clean'],r['duration'])
        parts['Mu_6s_audio_bounded']=clipped(parts['Mu_6s_forced'],r['duration'])
        parts['Mu_roll_identity']=decode_parts(1/(1+np.exp(-((data['m'].astype(float)*2-1)*2))),policy)
        parts['A8_on_estimated_stems']=decode_parts(a8_probabilities(encoder,data,'cuda'),{'onset':.7,'frame':.7,'offset':.5,'min_duration':.03})
        frame_heads={}
        for name,(model,state,record) in models.items():
            scales=(1.,calibrated['residual_scale']) if name==calibrated['condition'] else (1.,)
            probabilities=predict_scales(model,data['e'],data['m'],tuple(set(scales)),'cuda');probability=probabilities[1.]
            parts['temporal_'+name]=decode_parts(probability,record['policy'])
            frame_heads[name]=metrics(r,activity_parts(probability,record['policy']['frame']))
            if name==calibrated['condition']:
                probability=probabilities[calibrated['residual_scale']]
                parts['temporal_calibrated']=decode_parts(probability,calibrated['policy'])
                frame_heads['calibrated']=metrics(r,activity_parts(probability,calibrated['policy']['frame']))
        parts['validation_adoption_candidate']=parts['Mu_raw' if selection['adoption_candidate']=='identity' else 'temporal_'+selection['adoption_candidate']]
        if r['fresh_source']:
            parts['oracle_Mu']={p:read_part(r,'oracle_'+p,p) for p in ('guitar','bass')}
        piano=[{**n,'onset':n['onset']-r['excerpt_start'],'offset':n['offset']-r['excerpt_start']} for n in other[r['group']]]
        scores={name:{**metrics(r,ns),'polyphony':polyphony(r,ns)} for name,ns in parts.items()}
        errors={name:diagnostic(r,ns,piano) for name,ns in parts.items()}
        save(REPORT/'predictions'/(r['id']+'.json'),{'parts':parts,'metrics':scores,'diagnostics':errors,'frame_head_diagnostic':frame_heads})
        results.append({'id':r['id'],'group':r['group'],'dataset':r['dataset'],'fresh_source':r['fresh_source'],'metrics':scores,'diagnostics':errors,'frame_head_diagnostic':frame_heads})
        print('TEST',r['id'],{name:round(scores[name]['frame_f1'],4) for name in ('Mu_raw','Mu_6s_forced','Mu_ft_clean','temporal_standard','temporal_hard')},flush=True)
    summary={}
    for ds in sorted({r['dataset'] for r in results}):
        use=[r for r in results if r['dataset']==ds];names=set.intersection(*(set(r['metrics']) for r in use))
        summary[ds]={'cases':len(use),'methods':{name:{k:float(np.mean([r['metrics'][name][k] for r in use])) for k in ('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')} for name in sorted(names)}}
    save(out,{'selection':selection,'calibration':calibration,'summary':summary,'tracks':results,
        'scope':'New Slakh source UUIDs are new to our own training/evaluation, not guaranteed new to pretrained models. Other sources are regression or new acoustic conditions of known performances.',
        'diagnostic_scope':'Error categories may overlap. Matching a piano note does not prove it caused the estimated note.',
        'test_used_for_selection':False,'automatic_application_approved':False})
    print('TEMPORAL_TEST_COMPLETE',flush=True)

if __name__=='__main__':main()
