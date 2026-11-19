"""Descriptive development-set error categories, never final-test selection."""
import json
from pathlib import Path
from collections import Counter
import numpy as np,mir_eval
from scripts.a_candidate_data import save


def classify(row,parts):
    counts=Counter();delta=[];guard=row['guard_seconds'];end=row['duration']-guard
    for part in ('guitar','bass'):
        refs=[n for n in row['parts'][part] if guard<=n['onset']<end and not n.get('censored_onset')];notes=[n for n in parts[part] if guard<=n['onset']<end]
        interval=lambda ns:np.array([[n['onset'],n['offset']] for n in ns],float).reshape(-1,2)
        pitch=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
        pairs=mir_eval.transcription.match_notes(interval(refs),pitch(refs),interval(notes),pitch(notes),onset_tolerance=.05,offset_ratio=None) if refs and notes else []
        counts[part+'_tp']+=len(pairs);counts[part+'_fn']+=len(refs)-len(pairs);counts[part+'_fp']+=len(notes)-len(pairs)
        for i,j in pairs:
            if not refs[i].get('censored_offset'):delta.append(notes[j]['offset']-refs[i]['offset'])
        used={j for i,j in pairs};other='bass' if part=='guitar' else 'guitar'
        for j,n in enumerate(notes):
            if j in used:continue
            if any(r['pitch']==n['pitch'] and r['onset']<=n['onset']<r['offset'] for r in row['parts'][part]):kind='same_pitch_inside_existing_note'
            elif any(r['pitch']==n['pitch'] and abs(r['onset']-n['onset'])<=.05 for r in row['parts'][other]):kind='other_part_same_pitch_onset'
            elif any(abs(r['pitch']-n['pitch'])==12 and abs(r['onset']-n['onset'])<=.05 for r in row['parts'][part]):kind='octave_apart_onset'
            else:kind='other_false_note'
            counts[part+'_'+kind]+=1
    return dict(counts),delta


def main():
    manifest=json.load(open('backups/azure/data/stem_refiner_v1/manifest.json'));tracks=[];counts=Counter();offset=[]
    for r in manifest:
        if r['split']!='val':continue
        baseline=json.load(open(Path('backups/azure/data/stem_refiner_v1/cache')/(r['id']+'.json')))['baseline_parts'];c,d=classify(r,baseline);counts.update(c);offset.extend(d);tracks.append(dict(id=r['id'],dataset=r['dataset'],counts=c))
    s=np.asarray(offset);result=dict(conditions=len(tracks),note_counts=dict(counts),matched_known_offset_notes=len(s),
        offset_error_seconds=dict(median=float(np.median(s)),median_absolute=float(np.median(abs(s))),p90_absolute=float(np.quantile(abs(s),.9)),shorter_by_over150ms=int((s<-.15).sum()),longer_by_over150ms=int((s>.15).sum())),
        caveat='Descriptive mutually exclusive categories with stated precedence, not proof of acoustic cause. Repeated 16-condition development set. Does not imply jazz-specific weakness.',tracks=tracks)
    save('reports/error_taxonomy_v1/diagnostic.json',result);public={k:v for k,v in result.items() if k!='tracks'};save('docs/measurements/error_taxonomy_20261119.json',public);print('ERROR_TAXONOMY_READY',public,flush=True)


if __name__=='__main__':main()
