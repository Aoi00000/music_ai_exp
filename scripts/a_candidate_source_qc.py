"""Prediction-blind source integrity screen, motivated by training DC-only audio.

The fixed criterion tests every acquired split before any new-test inference.
It does not transpose notes or filter on model accuracy.
"""
from pathlib import Path
import json,shutil
import numpy as np,soundfile as sf
from band2tab.data import slakh_items
from scripts.a_candidate_data import ROOT,REPORT,save,sha

def main():
    if list(Path('runs/candidate_v2').glob('*/config.json')):raise RuntimeError('Source policy must be fixed before learning')
    if (REPORT/'source_quality.json').exists():raise RuntimeError('Source QC already fixed')
    acquisition=json.loads((REPORT/'source_acquisition.json').read_text());allowed={r['group'] for r in acquisition['selected']}
    records=[];excluded=set()
    for r in slakh_items('data/raw/slakh_candidate_v2','qc'):
        if r['group'] not in allowed:continue
        notes=[n for n in r['notes'] if n['offset']-n['onset']>=.03]
        indices=np.unique(np.linspace(0,len(notes)-1,min(32,len(notes))).astype(int)) if notes else []
        ac=[];means=[];info=sf.info(r['audio'])
        with sf.SoundFile(r['audio']) as f:
            for i in indices:
                n=notes[i];start=round(n['onset']*info.samplerate)
                if start>=len(f):continue
                f.seek(start);y=f.read(round(min(.2,n['offset']-n['onset'])*info.samplerate),dtype='float32',always_2d=True).mean(1)
                if not len(y):continue
                mean=float(y.mean());ac.append(float(np.sqrt(np.mean((y-mean)**2))));means.append(mean)
        silent=sum(v<1e-7 for v in ac);bad=len(ac)>=4 and silent/len(ac)>=.5
        records.append({'id':r['id'],'group':r['group'],'part':r['part'],'plugin':r['plugin_name'],
            'checked_attacks':len(ac),'constant_or_silent_attacks':silent,'ac_rms':ac,'dc_mean':means,'excluded':bad})
        if bad:excluded.add(r['group'])
    old=json.loads((ROOT/'manifest.json').read_text());rows=[r for r in old if not (r['fresh_source'] and r['group'] in excluded)]
    shutil.copy2(ROOT/'manifest.json',ROOT/'manifest_before_source_qc.json');shutil.copy2(REPORT/'protocol.json',REPORT/'protocol_before_source_qc.json')
    save(REPORT/'source_quality.json',{'motivation':'Training Track00357 bass waveform becomes constant DC despite MIDI attacks',
        'criterion':'At least 4 attacks checked, at least half with AC RMS <1e-7 during first min(0.2s,note duration); sample up to32 evenly spaced notes per target stem',
        'scope':'Prediction-blind rendering validity check applied uniformly to acquired train/val/test before training and before new-test inference',
        'excluded_groups':sorted(excluded),'excluded_records':[r['id'] for r in old if r not in rows],'records':records,
        'limits':'Does not establish perceptual audibility, fundamental-pitch correctness or all-note label correctness. No MIDI edits.'})
    save(ROOT/'manifest.json',rows);protocol=json.loads((REPORT/'protocol.json').read_text())
    from collections import Counter
    protocol.update(manifest_sha256=sha(ROOT/'manifest.json'),source_quality_sha256=sha(REPORT/'source_quality.json'),counts=dict(Counter(r['split'] for r in rows)))
    save(REPORT/'protocol.json',protocol);print('SOURCE_QC_FROZEN','excluded_groups',len(excluded),'counts',protocol['counts'],flush=True)

if __name__=='__main__':main()
