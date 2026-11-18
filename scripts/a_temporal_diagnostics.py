"""Describe frozen test errors; never fit, select or overwrite any model.

All-program coverage is recall of target notes among ALL non-drum MuScriptor
outputs. It is a diagnostic, not a usable guitar/bass prediction or precision.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import pretty_midi,mir_eval
from band2tab.audio import PITCH_MIN,PITCH_MAX
from band2tab.temporal import known_offset_score
from band2tab.refine_duration import instrument_metrics

def match(ref,est):
    if not ref or not est:return []
    intervals=lambda ns:np.array([[n['onset'],n['offset']] for n in ns],float).reshape(-1,2)
    pitches=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
    return mir_eval.transcription.match_notes(intervals(ref),pitches(ref),intervals(est),pitches(est),onset_tolerance=.05,offset_ratio=None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='backups/azure');p.add_argument('--out',default='reports/presentation_temporal');a=p.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'reports/temporal_v1/test_comparison.json').read_text())
    manifest={r['id']:r for r in json.loads((root/'data/temporal_v1/manifest.json').read_text())}
    coverage=[];splitting=[];frame_head=[];controls=[]
    for t in report['tracks']:
        r=manifest[t['id']];g=r['guard_seconds'];end=r['duration']-g if g else float('inf')
        pred=json.loads((root/'reports/temporal_v1/predictions'/(r['id']+'.json')).read_text())
        # Post-test diagnostic controls, not additional candidates for selection.
        raw=pred['parts']['Mu_raw']
        bounded=pred['parts']['Mu_audio_bounded']
        range_control={part:[n for n in ns if PITCH_MIN<=n['pitch']<=PITCH_MAX and n['offset']-n['onset']>=.03] for part,ns in bounded.items()}
        wrapper={part:list(pred['parts']['temporal_calibrated'][part])+[
            n for n in raw.get(part,[]) if not PITCH_MIN<=n['pitch']<=PITCH_MAX or n['offset']-n['onset']<.03]
            for part in ('guitar','bass')}
        for method,parts in (('Mu_range_duration_control',range_control),('calibrated_review_wrapper',wrapper)):
            scores=instrument_metrics(r['parts'],parts,start=g,end=end if g else None)
            scores.update(known_offset_score(r['parts'],parts,g,end))
            controls.append({'dataset':r['dataset'],'id':r['id'],'method':method,
                **{k:scores[k] for k in ('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')}})
        for method,m in t['frame_head_diagnostic'].items():
            decoded=t['metrics']['temporal_'+method]
            frame_head.append({'dataset':t['dataset'],'id':r['id'],'condition':method,
                **{'head_'+k:m[k] for k in ('frame_f1','union_frame_f1','active_exact_match')},
                **{'decoded_'+k:decoded[k] for k in ('frame_f1','union_frame_f1','active_exact_match')}})
        if r['fresh_source']:
            midi=pretty_midi.PrettyMIDI(str(root/'reports/temporal_v1/midi'/r['id']/'mix.mid'))
            allnotes=[{'pitch':n.pitch,'onset':n.start,'offset':n.end} for i in midi.instruments if not i.is_drum for n in i.notes if g<=n.start<end]
            rec={'id':r['id'],'programs':[{'program':int(i.program),'notes':len(i.notes)} for i in midi.instruments if not i.is_drum]}
            for part in ('guitar','bass'):
                refs=[n for n in r['parts'].get(part,[]) if g<=n['onset']<end and not n.get('censored_onset')]
                target=[n for n in pred['parts']['Mu_audio_bounded'][part] if g<=n['onset']<end]
                rec[part]={'reference_notes':len(refs),'correct_part_matches':len(match(refs,target)),
                    'all_program_matches':len(match(refs,allnotes))}
            coverage.append(rec)
        for method in ('Mu_audio_bounded','Mu_6s_audio_bounded','temporal_standard','temporal_hard','temporal_calibrated'):
            for part in ('guitar','bass'):
                refs=[n for n in r['parts'].get(part,[]) if g<=n['onset']<end and not n.get('censored_onset')]
                est=[n for n in pred['parts'][method][part] if g<=n['onset']<end]
                used={b for _,b in match(refs,est)}
                extra=[n for j,n in enumerate(est) if j not in used]
                suspected=sum(any(n['pitch']==b['pitch'] and b['onset']+.05<n['onset']<b['offset'] for b in refs) for n in extra)
                splitting.append({'dataset':r['dataset'],'id':r['id'],'method':method,'part':part,
                    'extra_notes':len(extra),'same_pitch_inside_reference_sustain':suspected})
    intervals=[];use=[t for t in report['tracks'] if t['dataset']=='slakh_new']
    # Each of the eight new UUIDs contributes exactly one excerpt here.
    assert len(use)==len({t['group'] for t in use})
    rng=np.random.default_rng(17);indices=rng.integers(0,len(use),(10000,len(use)))
    for method,baseline in (('temporal_standard','Mu_audio_bounded'),('temporal_standard','Mu_6s_audio_bounded'),('temporal_calibrated','Mu_6s_audio_bounded')):
        for metric in ('note_f1','frame_f1','union_frame_f1'):
            delta=np.array([t['metrics'][method][metric]-t['metrics'][baseline][metric] for t in use])
            low,high=np.quantile(delta[indices].mean(1),[.025,.975])
            intervals.append({'dataset':'slakh_new','source_songs':len(use),'method':method,'baseline':baseline,
                'metric':metric,'mean_delta':float(delta.mean()),'percentile_95_low':float(low),'percentile_95_high':float(high)})
    for name,rows in (('frame_head',frame_head),('suspected_splitting',splitting),('paired_intervals',intervals),('post_test_controls',controls)):
        with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'diagnostics.json').write_text(json.dumps({'all_program_target_coverage':coverage,'paired_intervals':intervals,
        'limits':['All-program coverage may count piano notes that coincide with targets; it is not proof of mislabelling.',
                  'A target in guitar and bass may be retrieved from the same Mu note. Coverage is not combined polyphonic accuracy.',
                  'Sustain coincidences suggest extra attacks or timing errors, not proven causes.',
                  'Intervals are descriptive paired bootstrap of eight excerpts, 10000 draws, seed17; no multiple-comparison correction or population guarantee.',
                  'Other regression datasets are not independent new songs; do not pool them as independent trials.']},indent=2))
    print('DIAGNOSTICS_EXPORTED',out)

if __name__=='__main__':main()
