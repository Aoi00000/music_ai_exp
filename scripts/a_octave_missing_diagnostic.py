"""Distinguish octave-apart FP with/without an already recovered true attack."""
from pathlib import Path
from collections import Counter
import json,numpy as np,mir_eval

def main():
    rows=[r for r in json.load(open('backups/azure/data/stem_refiner_v1/manifest.json')) if r['split']=='val'];counts=Counter()
    for r in rows:
        base=json.load(open('backups/azure/data/stem_refiner_v1/cache/'+r['id']+'.json'))['baseline_parts'];g=r['guard_seconds'];end=r['duration']-g
        refs=[n for n in r['parts']['guitar'] if g<=n['onset']<end and not n.get('censored_onset')];notes=[n for n in base['guitar'] if g<=n['onset']<end]
        interval=lambda ns:np.array([[n['onset'],n['offset']] for n in ns],float).reshape(-1,2)
        pitch=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
        pairs=mir_eval.transcription.match_notes(interval(refs),pitch(refs),interval(notes),pitch(notes),onset_tolerance=.05,offset_ratio=None) if refs and notes else []
        covered={i for i,j in pairs};used={j for i,j in pairs}
        for j,n in enumerate(notes):
            if j in used:continue
            if any(t['pitch']==n['pitch'] and t['onset']<=n['onset']<t['offset'] for t in r['parts']['guitar']):continue
            if any(t['pitch']==n['pitch'] and abs(t['onset']-n['onset'])<=.05 for t in r['parts']['bass']):continue
            if not any(abs(t['pitch']-n['pitch'])==12 and abs(t['onset']-n['onset'])<=.05 for t in r['parts']['guitar']):continue
            matched=any(i in covered and abs(t['pitch']-n['pitch'])==12 and abs(t['onset']-n['onset'])<=.05 for i,t in enumerate(refs))
            counts['octave_apart_false_notes']+=1;counts['corresponding_true_attack_already_detected' if matched else 'no_matched_corresponding_true_attack']+=1
    out=dict(counts=counts,scope='Same reused16, same mutually exclusive precedence as error_taxonomy_v1. Corresponding pitch may already be detected; not proof of acoustic harmonic cause.');p=Path('reports/octave_missing_diagnostic_20261120.json')
    if p.exists() and json.load(open(p))!=out:raise ValueError('Diagnostic changed')
    p.write_text(json.dumps(out,indent=2));print('OCTAVE_MISSING_DIAGNOSTIC',dict(counts))
if __name__=='__main__':main()
