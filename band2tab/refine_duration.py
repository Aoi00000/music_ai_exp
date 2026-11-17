"""Duration-aware targets and instrument-aware metrics for frozen-base refinement."""
import math
import numpy as np
from .audio import PITCH_MIN,PITCH_MAX
from .learn import note_metrics

SCHEMA='muscriptor_event_refiner_duration_v2'

def duration_targets(notes,truth,occupancy_weight=.8):
    """Soft class targets: exact sounding-time overlap plus onset correspondence.

    Boundary-spanning references contribute occupancy but no artificial onset.
    Unison in two instruments keeps an already valid assignment; an 'other'
    candidate receives equal support for the two instruments.
    """
    import mir_eval
    lookup={}
    for label,part in enumerate(('guitar','bass'),1):
        for n in truth.get(part,[]):lookup.setdefault((label,n['pitch']),[]).append(n)
    onset=np.zeros((len(notes),3),np.float32);onset[:,0]=1
    matched=[set(),set()]
    for j,part in enumerate(('guitar','bass')):
        ref=[n for n in truth.get(part,[]) if not n.get('censored_onset')]
        if not ref or not notes:continue
        intervals=lambda ns:np.array([[n['onset'],n['offset']] for n in ns],float)
        pitches=lambda ns:440*2**((np.array([n['pitch'] for n in ns])-69)/12)
        pairs=mir_eval.transcription.match_notes(intervals(ref),pitches(ref),intervals(notes),pitches(notes),onset_tolerance=.05,offset_ratio=None)
        matched[j]={e for _,e in pairs}
    occupancy=np.zeros_like(onset)
    for i,n in enumerate(notes):
        a,b=n['onset'],n['offset'];d=b-a
        if d<=0:raise ValueError('Non-positive candidate duration')
        intervals={c:[(max(a,r['onset']),min(b,r['offset'])) for r in lookup.get((c,n['pitch']),[]) if r['offset']>a and r['onset']<b] for c in (1,2)}
        boundaries=sorted({a,b,*[t for pairs in intervals.values() for pair in pairs for t in pair]})
        for lo,hi in zip(boundaries,boundaries[1:]):
            mid=(lo+hi)/2;active=[c for c,pairs in intervals.items() if any(x<=mid<y for x,y in pairs)]
            valid=[n['baseline_class']] if n['baseline_class'] in active else active or [0]
            for c in valid:occupancy[i,c]+=(hi-lo)/(d*len(valid))
        active=[j+1 for j,m in enumerate(matched) if i in m]
        if active:
            onset[i]=0
            valid=[n['baseline_class']] if n['baseline_class'] in active else active
            for c in valid:onset[i,c]=1/len(valid)
    result=occupancy_weight*occupancy+(1-occupancy_weight)*onset
    if not np.allclose(result.sum(1),1,atol=1e-5):raise ValueError('Invalid target distribution')
    return result

def instrument_metrics(truth,parts,start=0.,end=None,hop=.02):
    """Do not give an empty, correctly silent instrument an artificial Note F1=1."""
    if end is None:end=max([n['offset'] for ns in list(truth.values())+list(parts.values()) for n in ns]+[start+hop])
    frames=max(1,math.ceil((end-start)/hop));rolls=[];counts={}
    for source in (truth,parts):
        roll=np.zeros((frames,2,PITCH_MAX-PITCH_MIN+1),bool)
        for j,part in enumerate(('guitar','bass')):
            for n in source.get(part,[]):
                if not PITCH_MIN<=n['pitch']<=PITCH_MAX:continue
                a=max(0,math.ceil((n['onset']-start)/hop-1e-9));b=min(frames,math.ceil((n['offset']-start)/hop-1e-9))
                if a<b:roll[a:b,j,n['pitch']-PITCH_MIN]=True
        rolls.append(roll)
    r,e=rolls;tp=int((r&e).sum());fp=int((~r&e).sum());fn=int((r&~e).sum());active=(r|e).any(axis=(1,2))
    by_part={}
    for offset in (False,True):
        ntp=nfp=nfn=0
        for part in ('guitar','bass'):
            ref=[n for n in truth.get(part,[]) if start<=n['onset']<end and not n.get('censored_onset')]
            est=[n for n in parts.get(part,[]) if start<=n['onset']<end]
            score=note_metrics(ref,est,offset,include_frames=False);t=round(score['precision']*len(est))
            ntp+=t;nfp+=len(est)-t;nfn+=len(ref)-t
            if not offset:
                j=0 if part=='guitar' else 1;rr,ee=r[:,j],e[:,j]
                pt=int((rr&ee).sum());pf=int((~rr&ee).sum());pn=int((rr&~ee).sum())
                by_part[part]={'reference_notes':len(ref),'reference_frames':int(rr.sum()),
                               'note_f1':2*t/max(1,len(est)+len(ref)),
                               'frame_f1':2*pt/max(1,2*pt+pf+pn)}
        name='offset_f1' if offset else 'note_f1';counts[name]=2*ntp/max(1,2*ntp+nfp+nfn)
        if not offset:counts.update(note_tp=ntp,note_fp=nfp,note_fn=nfn)
    union_r=r.any(1);union_e=e.any(1);utp=int((union_r&union_e).sum());ufp=int((~union_r&union_e).sum());ufn=int((union_r&~union_e).sum())
    return {**counts,'by_part':by_part,'frame_f1':2*tp/max(1,2*tp+fp+fn),'frame_tp':tp,'frame_fp':fp,'frame_fn':fn,
            'union_frame_f1':2*utp/max(1,2*utp+ufp+ufn),
            'active_exact_match':float((r[active]==e[active]).all(axis=(1,2)).mean()) if active.any() else 1.}

def summarize_records(rows):
    metrics=('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')
    kinds=sorted({r['kind'] for r in rows})
    domains={kind:{k:float(np.mean([r[k] for r in rows if r['kind']==kind])) for k in metrics} for kind in kinds}
    mean={k:float(np.mean([v[k] for v in domains.values()])) for k in metrics}
    by_instrument={}
    for part in ('guitar','bass'):
        use=[r['by_part'][part] for r in rows if r['by_part'][part]['reference_frames']>0]
        if use:by_instrument[part]={k:float(np.mean([r[k] for r in use])) for k in ('note_f1','frame_f1')}
    return {**mean,'by_domain':domains,'by_instrument':by_instrument,'selection_score':.5*(mean['note_f1']+mean['frame_f1']),'tracks':rows}
