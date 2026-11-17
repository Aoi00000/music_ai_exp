"""Count false notes in the absent instrument too; replay frozen changes only."""
from pathlib import Path
import json
import numpy as np
from band2tab.refine import candidates_from_midi,PARTS
from band2tab.learn import note_metrics,pitch_set_metrics

report=json.loads(Path('reports/refiner_existing_comparisons.json').read_text())
truths={it['id']:it for it in json.loads(Path('data/manifest_external.json').read_text())}
def strict(ref,parts,offset=False):
    tp=fp=fn=0;frame_tp=frame_fp=frame_fn=0
    for part in ('guitar','bass'):
        r=ref.get(part,[]);e=parts.get(part,[]);metric=note_metrics(r,e,offset)
        t=round(metric['precision']*len(e));tp+=t;fp+=len(e)-t;fn+=len(r)-t
        for bucket in metric['by_reference_polyphony'].values():
            frame_tp+=bucket['tp'];frame_fp+=bucket['fp'];frame_fn+=bucket['fn']
    return {'note_f1':2*tp/max(1,2*tp+fp+fn),'note_tp':tp,'note_fp':fp,'note_fn':fn,
            'frame_f1':2*frame_tp/max(1,2*frame_tp+frame_fp+frame_fn)}
rows=[]
for record in report['solo_tracks']:
    it=truths[record['id']];midi=Path('reports/muscriptor_external_predictions')/(it['id']+'.mid')
    candidates=candidates_from_midi(midi);changes={c['candidate']:c for c in record['changes']}
    base={'guitar':[],'bass':[]};proposed={'guitar':[],'bass':[]}
    for i,n in enumerate(candidates):
        old=PARTS[n['baseline_class']];new=changes[i]['to'] if i in changes else old
        if old!='other':base[old].append(n)
        if new!='other':proposed[new].append(n)
    ref={it['part']:it['notes']}
    rows.append({'id':it['id'],'part':it['part'],
                 'baseline':strict(ref,base),'proposed':strict(ref,proposed),
                 'baseline_offset_f1':strict(ref,base,True)['note_f1'],
                 'proposed_offset_f1':strict(ref,proposed,True)['note_f1'],
                 'baseline_parts':{p:len(ns) for p,ns in base.items()},'proposed_parts':{p:len(ns) for p,ns in proposed.items()}})
summary={key:{metric:float(np.mean([r[key][metric] for r in rows])) for metric in ('note_f1','frame_f1')} for key in ('baseline','proposed')}
summary['scope']='same previously inspected solo22, all predicted guitar AND bass counted including the absent part; no refitting'
summary['tracks']=rows
Path('reports/refiner_strict_comparison.json').write_text(json.dumps(summary,indent=2));print('STRICT_COMPARISON',summary['baseline'],summary['proposed'],flush=True)
