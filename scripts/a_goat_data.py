"""GOAT seconds-aligned audio/note clips with publisher test held out.

MIDI releases are retained as provided; no inference-based label correction.
"""
from pathlib import Path
import json,collections,hashlib
import numpy as np,soundfile as sf,pretty_midi
from band2tab.separation import write_input
from scripts.a_candidate_data import sha,save
from scripts.a_temporal_data import read_clip,clipped_truth

def main():
    root=Path('data/goat_v1');dest=root/'manifest.json'
    if dest.exists():raise ValueError('GOAT manifest frozen')
    selection=json.loads(Path('reports/goat/selection.json').read_text());rows=[];sources=[];canonical={};excluded=[]
    for source in selection['sources']:
        item=source['item'];folder=Path('data/raw/goat/GOAT/data')/item;mid=folder/(item+'_fine_aligned.mid')
        m=pretty_midi.PrettyMIDI(str(mid));notes=[{'pitch':int(n.pitch),'onset':float(n.start),'offset':float(n.end)} for ins in m.instruments if not ins.is_drum for n in ins.notes]
        if not notes or any(not (0<=n['pitch']<=127 and 0<=n['onset']<n['offset']) for n in notes):raise ValueError('Invalid GOAT note labels '+item)
        fingerprint=hashlib.sha256(json.dumps(sorted((n['pitch'],round(n['onset'],3),round(n['offset'],3)) for n in notes)).encode()).hexdigest()
        if fingerprint in canonical and canonical[fingerprint]['split']!=source['split']:raise ValueError('Exact normalized MIDI crosses splits')
        canonical[fingerprint]=source
        views=('di','amp1') if source['split']=='train' else ('di','amp1','amp5')
        paths={v:folder/(item+{'di':'','amp1':'_amp_1','amp5':'_amp_5'}[v]+'.wav') for v in views}
        infos={v:sf.info(p) for v,p in paths.items()};duration=min(i.duration for i in infos.values())
        # Do not silently turn an alignment/file mismatch into truncated truth.
        if max(i.duration for i in infos.values())-duration>.05 or max(n['offset'] for n in notes)>duration+.1:
            excluded.append({'item':item,'split':source['split'],'reason':'audio_or_label_length_mismatch',
                'audio_seconds':{v:i.duration for v,i in infos.items()},'last_label_offset':max(n['offset'] for n in notes),
                'midi_sha256':sha(mid),'action':'Quarantine all views; do not rescale labels'})
            continue
        if duration<3.:raise ValueError('GOAT source too short '+item)
        sources.append({'item':item,'split':source['split'],'player':source['player'],'group':'goat_'+item,
            'midi_sha256':sha(mid),'midi_note_fingerprint':fingerprint,'notes':len(notes),'duration':duration,
            'audio_sha256':{v:sha(p) for v,p in paths.items()},'alignment_f_measure_fine':source['alignment_f_measure_fine'],
            'pitch_bend_events':sum(len(ins.pitch_bends) for ins in m.instruments),
            'notes_outside_prior_B0_E6_range':sum(not 23<=n['pitch']<=88 for n in notes)})
        length=min(duration,20. if source['split']=='train' else 30. if source['split']=='val' else 40.)
        fractions=(.15,.45,.75) if source['split']=='train' else (.35,)
        for j,fraction in enumerate(fractions):
            start=max(0.,(duration-length)*fraction)
            for view,path in paths.items():
                ident='goat_'+item+'_'+view+'_'+str(j);out=root/'audio'/(ident+'.wav')
                write_input(out,read_clip(path,start,length),16000);actual=sf.info(out).duration
                y,_=sf.read(out,dtype='float32');ac=float(np.sqrt(np.mean((y-y.mean())**2)))
                truth=clipped_truth({'guitar':notes},start,actual)
                if ac<1e-7 and truth['guitar']:raise ValueError('GOAT labelled excerpt has no AC signal '+ident)
                rows.append({'id':ident,'group':'goat_'+item,'player':source['player'],'split':source['split'],'kind':'solo',
                    'dataset':'goat_di' if view=='di' else 'goat_amp','view':view,'duration':actual,'excerpt_start':start,
                    'guard_seconds':.25,'audio':str(out.resolve()),'inputs':{'mix':str(out.resolve())},
                    'parts':truth,'audio_sha256':sha(out)})
    save(dest,rows);save('reports/goat/data_protocol.json',{'manifest_sha256':sha(dest),'selection_sha256':sha('reports/goat/selection.json'),
        'counts':dict(collections.Counter(r['split'] for r in rows)),'sources':sources,
        'excluded':excluded,'prediction_blind_qc':{'view_length_tolerance_seconds':.05,'label_release_overrun_tolerance_seconds':.1,'labelled_clip_min_ac_rms':1e-7},
        'split':'Author test preserved; player2 excluded from training. All render variants grouped by source item. Anonymous metadata cannot establish original composition disjointness.',
        'test_used_for_selection':False,'automatic_application_approved':False})
    print('GOAT_DATA_READY',collections.Counter(r['split'] for r in rows),flush=True)

if __name__=='__main__':main()
