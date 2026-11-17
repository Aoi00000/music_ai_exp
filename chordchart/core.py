"""CQT, structural labels, decoding and portable paths.
The upstream package is read from a pinned source checkout; its pip metadata
is not used. The project's CRNN port uses stable PyTorch operations.
"""
import json, sys, re, hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/lv-chordia'
REV = 'd8ab9ac99bce081bb79065d717f96e2c3b2099c8'
RWC_REV = '0a1a6c31dbe73a7f5d44f7caef8cd0999402a4c2'
SR, HOP, DT = 22050, 512, 512 / 22050
SIZES = (85, 13, 4, 4, 3, 3)  # N + 7 triad types x 12, bass, 7/9/11/13

@lru_cache(maxsize=1)
def upstream():
    if not VENDOR.exists(): raise FileNotFoundError('Run python -m chordchart.setup bootstrap first')
    import importlib.util
    spec=importlib.util.spec_from_file_location('_chordchart_parser',VENDOR/'lv_chordia/complex_chord.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Chord

def write_json(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def resolve(s): return Path(s) if Path(s).is_absolute() else ROOT / s

@lru_cache(maxsize=20000)
def target(label):
    """Retain partial supervision. Never map an unknown annotation to N."""
    import mir_eval
    try:
        mir_eval.chord.encode(label,reduce_extended_chords=True)  # syntax validation, original string preserved
        a=upstream()(label).to_numpy().astype(np.int64)
        a[1]+=1
        for k,n in enumerate(SIZES):
            if not 0<=a[k]<n: a[k]=-100
        if a[0]==0: a[2:]=-100  # no extensions to learn for no-chord
        return a
    except Exception:
        return np.full(6,-100,dtype=np.int64)

@lru_cache(maxsize=20000)
def signature(label, bass=True):
    """Pitch-class equality; enharmonic spellings compare equally."""
    import mir_eval
    if label=='X': return None
    try:
        root,bits,b=mir_eval.chord.encode(label,reduce_extended_chords=True)
        if root<0: return ('N',)
        return (int(root),tuple(int(v) for v in bits),int(b)) if bass else (int(root),tuple(int(v) for v in bits))
    except Exception: return None

@lru_cache(maxsize=20000)
def family(label):
    if label in ('N','X'):return label
    s=signature(label,False)
    if s is None:return 'X'
    for q in ['maj','min','dim','aug','sus2','sus4','5','maj7','7','min7','minmaj7','maj6','min6','hdim7','dim7','maj(9)','9','maj9','min9','11','min11','13','maj13','min13','sus4(b7)','sus2(b7)','7(b9)','7(#9)','7(13)','7(9,13)','aug(b7)']:
        if signature('C:'+q,False)[1]==s[1]:return 'add9' if q=='maj(9)' else q
    return 'other'

def quality(label):
    if label in ('N','X'):return label
    return label.split(':',1)[1].split('/')[0] if ':' in label else 'maj'

def cqt(path):
    import librosa
    y,_=librosa.load(path,sr=SR,mono=True)
    if len(y)<SR//2:raise ValueError('Audio must be at least 0.5 seconds')
    x=np.abs(librosa.hybrid_cqt(y,sr=SR,hop_length=HOP,fmin=librosa.note_to_hz('F#0'),
                              n_bins=288,bins_per_octave=36,tuning=None)).T.astype(np.float32)
    return x,len(y)/SR

def transpose_label(label, shift):
    if label in ('N','X'):return label
    notes=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    m=re.match(r'^([A-G])([#b]*)(.*)$',label)
    if not m:return label
    pc={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[m[1]]+m[2].count('#')-m[2].count('b')
    return notes[(pc+shift)%12]+m[3]  # Harte relative slash degree stays relative

def display(label, shift=0):
    label=transpose_label(label,shift)
    if label=='N':return 'N.C.'
    if label=='X':return '?'
    import mir_eval
    try:
        r,q,adds,b=mir_eval.chord.split(label)
        suffix={'maj':'','min':'m','maj7':'maj7','min7':'m7','minmaj7':'m(maj7)',
                'hdim7':'m7b5','maj6':'6','min6':'m6','min9':'m9','maj9':'maj9',
                'min11':'m11','min13':'m13'}.get(q,q)
        # Common add9 spelling, without throwing away its absence of a seventh
        if q=='maj' and set(adds) in ({'9'},{'2'}):suffix='add9';adds=[]
        if adds:suffix+='('+','.join(sorted(adds))+')'
        name=r+suffix
        root,_,bass=mir_eval.chord.encode(label,reduce_extended_chords=True)
        if b!='1':
            notes=['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
            name+='/'+notes[(root+bass)%12]
        return name
    except Exception:return label

@lru_cache(maxsize=1)
def dictionary():
    """Explicit named vocabulary. Structural expressivity is not label accuracy."""
    upstream()
    path=VENDOR/'lv_chordia/data/full_chord_list.txt'
    items=[s.strip() for s in path.read_text().splitlines() if s.strip()]
    items+=['C:5','C:maj(9)','C:7','C:maj7','C:min7','C:minmaj7','C:dim7','C:hdim7',
            'C:aug','C:sus2','C:sus4','C:9','C:maj9','C:min9','C:11','C:min11','C:13','C:maj13','C:min13']
    # All inversions for the primary explicit qualities.
    base=items[-18:]
    for s in base:
        for degree in ['b2','2','b3','3','4','b5','5','#5','6','b7','7']:
            items.append(s.split('/')[0]+'/'+degree)
    result=['N'];seen={signature('N')}
    for s in items:
        if s=='N' or not s.startswith('C:'):continue
        t=target(s)
        if (t<0).any():continue
        for k in range(12):
            lab=transpose_label(s,k);sig=signature(lab)
            if sig and sig not in seen:result.append(lab);seen.add(sig)
    return result

def decode(probs, penalty=2.0):
    """O(T*K) Viterbi with uniform switch cost; every frame may change.
    Does not force changes onto beats or discard short segments.
    """
    labs=[s for s in dictionary() if all(t<probs[j].shape[1] for j,t in enumerate(target(s)))];tags=np.stack([target(s) for s in labs])
    scores=np.zeros((len(probs[0]),len(labs)),np.float32)
    for j,p in enumerate(probs):
        mask=tags[:,j]>=0
        scores[:,mask]+=np.log(np.maximum(p[:,tags[mask,j]],1e-8))
    prev=scores[0].copy();back=np.zeros(scores.shape,dtype=np.int32)
    for t in range(1,len(scores)):
        k=int(prev.argmax());switch=prev[k]-penalty
        stay=prev>=switch
        back[t]=np.where(stay,np.arange(len(labs)),k)
        prev=np.where(stay,prev,switch)+scores[t]
    ids=np.zeros(len(scores),np.int32);ids[-1]=prev.argmax()
    for t in range(len(ids)-1,0,-1):ids[t-1]=back[t,ids[t]]
    # Candidate scores are uncalibrated: use a relative margin, not a probability.
    segments=[];start=0
    for t in range(1,len(ids)+1):
        if t==len(ids) or ids[t]!=ids[start]:
            avg=scores[start:t].mean(0);best=np.argsort(avg)[-3:][::-1]
            chosen=int(ids[start]);other=best[best!=chosen]
            margin=float(avg[chosen]-avg[other[0]])
            segments.append({'start':start*DT,'end':t*DT,'label':labs[ids[start]],
                             'candidates':[labs[k] for k in best],
                             'margin':margin,
                             'needs_review':bool(margin<0.5)})
            start=t
    return segments


def symbol_to_label(symbol):
    """Accept common musician-facing spellings for manual corrections."""
    s=str(symbol).strip()
    if s.upper() in ('N','N.C.','NC'):return 'N'
    if s=='?':return 'X'
    if ':' in s:
        import mir_eval
        mir_eval.chord.encode(s,reduce_extended_chords=True);return s
    m=re.match(r'^([A-G][#b]?)(.*)$',s)
    if not m:raise ValueError('Cmaj7、Cm7、C/Eなどの形式で入力してください')
    root,rest=m.groups();bass=None
    if '/' in rest:rest,bass=rest.rsplit('/',1)
    aliases={'':'maj','m':'min','dim':'dim','aug':'aug','+':'aug','5':'5','sus2':'sus2','sus4':'sus4',
             'M7':'maj7','maj7':'maj7','7':'7','m7':'min7','mM7':'minmaj7','m(maj7)':'minmaj7',
             '6':'maj6','m6':'min6','m7b5':'hdim7','dim7':'dim7','add9':'maj(9)',
             '9':'9','m9':'min9','maj9':'maj9','11':'11','m11':'min11','13':'13','m13':'min13','maj13':'maj13'}
    q=aliases.get(rest)
    if q is None and '(' in rest:
        base,mods=rest.split('(',1);q=aliases.get(base,base)+'('+mods
    if q is None:raise ValueError('コード表記を確認してください。例：C7(b9)、Cadd9、C/E')
    label=root+':'+q
    if bass:
        def pc(n):
            z=re.fullmatch(r'([A-G])([#b]?)',n)
            if not z:raise ValueError('ベース音はCやEbなどの音名で入力してください')
            return ({'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[z[1]]+z[2].count('#')-z[2].count('b'))%12
        degree=['1','b2','2','b3','3','4','b5','5','#5','6','b7','7'][(pc(bass)-pc(root))%12]
        label+='/'+degree
    import mir_eval
    mir_eval.chord.encode(label,reduce_extended_chords=True)
    return label
