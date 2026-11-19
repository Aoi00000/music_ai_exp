"""Pitch-preserving, sustain-aware fingering and minimal MusicXML export."""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

TUNINGS = {'guitar': [64, 59, 55, 50, 45, 40], 'bass': [43, 38, 33, 28]}
NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def pitch_name(p): return NAMES[int(p)%12]+str(int(p)//12-1)
def clean_notes(notes):
    out=[]
    for n in notes:
        n=dict(n); raw_pitch=n['pitch']; p=int(raw_pitch); a=float(n['onset']); b=float(n['offset'])
        if isinstance(raw_pitch,bool) or p!=raw_pitch:
            raise ValueError('pitch must be an integer MIDI pitch')
        if not (0<=p<=127 and math.isfinite(a) and math.isfinite(b) and 0<=a<b):
            raise ValueError(f'Invalid note: {n}')
        n.update(pitch=p,onset=a,offset=b,name=pitch_name(p),id=len(out))
        out.append(n)
    return sorted(out,key=lambda n:(n['onset'],n['pitch'],n['id']))

def assign_frets(notes, part='guitar', tuning=None, max_fret=24, beam=64, span=5):
    """Beam search over active notes. Never truncate notes to manufacture feasibility.

    Span and at most four distinct pressed frets are conservative constraints,
    not a complete hand biomechanics model. Strings, pitch
    equality, fret range and overlapping-note exclusion are hard constraints.
    Unassignable notes remain in the JSON and MIDI, with an explicit reason.
    """
    notes=clean_notes(notes); tuning=list(tuning or TUNINGS[part])
    # cost, active (offset,string,fret), hand_position, assignments
    states=[(0.,(),0.,{})]
    for n in notes:
        candidates=[(s+1,n['pitch']-op) for s,op in enumerate(tuning)
                    if 0<=n['pitch']-op<=max_fret]
        nxt=[]
        for cost,active,pos,path in states:
            active=tuple(x for x in active if x[0]>n['onset']+1e-7)
            occupied={x[1] for x in active}
            for string,fret in candidates:
                if string in occupied: continue
                held=[x[2] for x in active if x[2]>0]+([fret] if fret else [])
                newpos=sum(held)/len(held) if held else pos
                stretch=max(held)-min(held) if held else 0
                if stretch>span or len(set(held))>4: continue
                movement=abs(newpos-pos)
                local=.12*fret+.6*movement+3*max(0,stretch-span)**2
                if fret==0: local-=.2
                nxt.append((cost+local,active+((n['offset'],string,fret),),newpos,
                            {**path,n['id']:(string,fret)}))
            # Rejecting a note is explicit and costly, not a pitch substitution.
            nxt.append((cost+40+10*float(n.get('confidence',.5) or .5),active,pos,
                        {**path,n['id']:None}))
        states=sorted(nxt,key=lambda x:x[0])[:beam]
    best=states[0][3] if states else {}
    for n in notes:
        n.pop('review_reason',None)
        sf=best.get(n['id']); n['string']=sf[0] if sf else None
        n['fret']=sf[1] if sf else None
        if not sf: n['review_reason']='unplayable_or_out_of_range'
    return notes

def beat_position(t, beats):
    import numpy as np
    beats=np.asarray(beats,float)
    if beats.ndim!=1 or len(beats)<2 or not np.all(np.isfinite(beats)) or not np.all(np.diff(beats)>0):
        raise ValueError('At least two finite, strictly increasing beats required')
    if t<beats[0]: return (t-beats[0])/(beats[1]-beats[0])
    if t>beats[-1]: return len(beats)-1+(t-beats[-1])/(beats[-1]-beats[-2])
    return float(np.interp(t,beats,np.arange(len(beats))))

def quantize(notes,bpm=120.,beats_per_bar=4,grid=4,first_downbeat=0.,beats=None):
    """Keep raw times. Actual grid conflicts must be reviewed before final engraving."""
    if not math.isfinite(bpm) or bpm<=0 or beats_per_bar not in (3,4) or grid not in (3,4):
        raise ValueError('bpm>0, beats_per_bar=3/4, grid=3/4 required')
    out=[]
    for n in notes:
        n=dict(n)
        def to_tick(t):
            b=beat_position(t,beats) if beats is not None else (t-first_downbeat)*bpm/60
            return max(0,round(b*grid))
        n['q_start']=to_tick(n['onset']); n['q_end']=max(n['q_start']+1,to_tick(n['offset']))
        out.append(n)
    return out

def _add(parent,tag,text=None,**attrs):
    e=ET.SubElement(parent,tag,attrs)
    if text is not None:e.text=str(text)
    return e

def _rhythm_chunks(length, grid, start=0):
    """Spell grid durations with standard note types, dots and 3:2 tuplets."""
    from fractions import Fraction
    choices = {}
    for kind, beats in [('whole', Fraction(4)), ('half', Fraction(2)),
                        ('quarter', Fraction(1)), ('eighth', Fraction(1, 2)),
                        ('16th', Fraction(1, 4))]:
        for dots in (0, 1):
            ticks = beats * grid * (Fraction(3, 2) if dots else 1)
            if ticks.denominator != 1:
                continue
            ticks = int(ticks)
            candidate = (dots, kind)
            if ticks not in choices or dots < choices[ticks][0]:
                choices[ticks] = candidate
    while length:
        if grid == 3 and (start % grid or length < grid):
            # Complete each beat's three eighth-note tuplet slots. Mixed
            # half/eighth tuplets can make MuseScore invent a closing rest.
            size, kind, dots, triplet = 1, 'eighth', 0, True
        else:
            size = max(size for size in choices if size <= length)
            dots, kind = choices[size]
            triplet = False
        yield size, kind, dots, triplet
        length -= size
        start += size


def _note_value(element, kind, dots, triplet):
    _add(element, 'type', kind)
    for _ in range(dots):
        _add(element, 'dot')
    if triplet:
        ratio = _add(element, 'time-modification')
        _add(ratio, 'actual-notes', 3)
        _add(ratio, 'normal-notes', 2)


def musicxml(parts, dest, bpm=120., beats_per_bar=4, grid=4, tunings=None, beats=None):
    """One TAB staff per instrument; time slices form chords in one voice.

    Independent sustains are tied across slice and measure boundaries. This
    avoids a separate voice for every string (MuseScore supports four voices
    per staff). Every duration is explicitly spelled, rather than asking an
    importer to round arbitrary lengths such as a 2.5-quarter-note rest.
    Quantization collisions remain explicit issues; raw JSON/MIDI keep them.
    """
    tunings = tunings or TUNINGS
    score = ET.Element('score-partwise', version='4.0')
    work = _add(score, 'work'); _add(work, 'work-title', 'Band2Tab - review draft')
    plist = _add(score, 'part-list'); issues = []
    divisions = 12; tick = divisions // grid; bar = beats_per_bar * grid
    maxend = max((n['q_end'] for notes in parts.values() for n in notes), default=bar)
    count = max(1, math.ceil(maxend / bar))
    for pi, (part, notes) in enumerate(parts.items(), 1):
        pid = f'P{pi}'
        sp = _add(plist, 'score-part', id=pid); _add(sp, 'part-name', part.title())
        tuning = tunings[part]; p = _add(score, 'part', id=pid)
        by_string = defaultdict(list)
        accepted = []
        for n in sorted(notes, key=lambda n: (n['q_start'], n['q_end'])):
            if n.get('string') is None:
                continue
            string_notes = by_string[n['string']]
            if string_notes and n['q_start'] < string_notes[-1]['q_end']:
                issues.append({'part': part, 'note_id': n['id'], 'reason': 'quantization_collision'})
                continue
            string_notes.append(n); accepted.append(n)
        for bi in range(count):
            m = _add(p, 'measure', number=str(bi + 1)); lo = bi * bar; hi = lo + bar
            if bi == 0:
                at = _add(m, 'attributes'); _add(at, 'divisions', divisions)
                tm = _add(at, 'time'); _add(tm, 'beats', beats_per_bar); _add(tm, 'beat-type', 4)
                clef = _add(at, 'clef'); _add(clef, 'sign', 'TAB'); _add(clef, 'line', 5 if len(tuning) == 6 else 3)
                details = _add(at, 'staff-details'); _add(details, 'staff-lines', len(tuning))
                for line, op in enumerate(reversed(tuning), 1):
                    st = _add(details, 'staff-tuning', line=str(line)); name = NAMES[op % 12]
                    _add(st, 'tuning-step', name[0])
                    if len(name) > 1:
                        _add(st, 'tuning-alter', 1)
                    _add(st, 'tuning-octave', op // 12 - 1)
                dr = _add(m, 'direction'); dt = _add(dr, 'direction-type'); met = _add(dt, 'metronome')
                _add(met, 'beat-unit', 'quarter'); _add(met, 'per-minute', bpm)
                _add(dr, 'sound', tempo=str(bpm))
            if beats is not None and bi > 0 and bi * beats_per_bar + 1 < len(beats):
                k = bi * beats_per_bar
                tempo = round(60 / max(1e-6, beats[k + 1] - beats[k]), 3)
                dr = _add(m, 'direction'); dt = _add(dr, 'direction-type'); met = _add(dt, 'metronome')
                _add(met, 'beat-unit', 'quarter'); _add(met, 'per-minute', tempo)
                _add(dr, 'sound', tempo=str(tempo))
            measure_notes = [n for n in accepted if n['q_start'] < hi and n['q_end'] > lo]
            boundaries = sorted({lo, hi, *(max(lo, n['q_start']) for n in measure_notes),
                                 *(min(hi, n['q_end']) for n in measure_notes)})
            for start, end in zip(boundaries, boundaries[1:]):
                active = sorted([n for n in measure_notes if n['q_start'] <= start < n['q_end']],
                                key=lambda n: (n['pitch'], n['id']))
                cursor = start
                for length, kind, dots, triplet in _rhythm_chunks(end - start, grid, start):
                    chunk_end = cursor + length
                    if not active:
                        no = _add(m, 'note'); _add(no, 'rest'); _add(no, 'duration', length * tick)
                        _add(no, 'voice', 1); _note_value(no, kind, dots, triplet)
                    for index, n in enumerate(active):
                        no = _add(m, 'note')
                        if index:
                            _add(no, 'chord')
                        pitch = _add(no, 'pitch'); name = NAMES[n['pitch'] % 12]
                        _add(pitch, 'step', name[0])
                        if len(name) > 1:
                            _add(pitch, 'alter', 1)
                        _add(pitch, 'octave', n['pitch'] // 12 - 1)
                        _add(no, 'duration', length * tick)
                        stop = n['q_start'] < cursor; onward = n['q_end'] > chunk_end
                        if stop:
                            _add(no, 'tie', type='stop')
                        if onward:
                            _add(no, 'tie', type='start')
                        _add(no, 'voice', 1); _note_value(no, kind, dots, triplet)
                        nt = _add(no, 'notations')
                        if stop:
                            _add(nt, 'tied', type='stop')
                        if onward:
                            _add(nt, 'tied', type='start')
                        tech = _add(nt, 'technical'); _add(tech, 'string', n['string']); _add(tech, 'fret', n['fret'])
                    cursor = chunk_end
    ET.indent(score, space='  ')
    Path(dest).write_bytes(ET.tostring(score, encoding='utf-8', xml_declaration=True))
    from .notation import validate_musicxml
    validate_musicxml(dest)
    return issues

def write_midi(parts,dest,bpm=120.):
    import pretty_midi
    midi=pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for part,notes in parts.items():
        ins=pretty_midi.Instrument(program=25 if part=='guitar' else 33,name=part)
        for n in notes:
            ins.notes.append(pretty_midi.Note(int(n.get('velocity',80)),int(n['pitch']),n['onset'],n['offset']))
        midi.instruments.append(ins)
    midi.write(str(dest))

def read_midi(path,force_part=None):
    import pretty_midi
    parts=defaultdict(list)
    for ins in pretty_midi.PrettyMIDI(str(path)).instruments:
        if ins.is_drum:continue
        part=force_part or ('guitar' if 24<=ins.program<=31 else 'bass' if 32<=ins.program<=39 else None)
        if part is None:continue
        parts[part].extend({'pitch':n.pitch,'onset':n.start,'offset':n.end,
                           'velocity':n.velocity,'confidence':None} for n in ins.notes)
    return dict(parts)

def write_outputs(parts,out,bpm=120.,beats_per_bar=4,grid=4,tunings=None,beats=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);assigned={};issues=[]
    for part,notes in parts.items():
        tuning=(tunings or TUNINGS)[part]
        a=assign_frets(notes,part,tuning=tuning);q=quantize(a,bpm,beats_per_bar,grid,beats=beats)
        assigned[part]=q
        issues.extend({'part':part,'note_id':n['id'],'reason':n['review_reason']} for n in a if n.get('review_reason'))
    if not any(assigned.values()): raise ValueError('No guitar/bass notes found. Inspect the MIDI or explicitly choose --part.')
    issues+=musicxml(assigned,out/'score.musicxml',bpm,beats_per_bar,grid,tunings,beats)
    write_midi(assigned,out/'raw.mid',bpm)
    (out/'notes.json').write_text(json.dumps({'parts':assigned,'issues':issues,
        'bpm':bpm,'beats_per_bar':beats_per_bar,'grid':grid,'tuning':tunings or TUNINGS,
        'beat_times':beats,'status':'review_draft'},ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['Band2Tab / REVIEW DRAFT / x = unassigned / raw times retained','']
    bar=beats_per_bar*grid
    for part,notes in assigned.items():
        lines.append(part.upper())
        for n in notes:
            if n['string'] is None:
                lines.append(f"x / note {n['id']} / {n['name']} / {n['onset']:.3f}-{n['offset']:.3f}s / unassigned")
        for start in range(0,max((n['q_end'] for n in notes),default=bar),bar):
            for s in range(1,len((tunings or TUNINGS)[part])+1):
                cells=['---']*bar
                for n in notes:
                    if n['string']==s and start<=n['q_start']<start+bar:
                        cells[n['q_start']-start]=f'{n["fret"]:02d}-'
                lines.append(f'{s}|'+''.join(cells)+'|')
            lines.append('')
    (out/'tab.txt').write_text('\n'.join(lines),encoding='utf-8')
    return {'notes':sum(map(len,assigned.values())),'issues':len(issues),'out':str(out)}

def midi_timing(path):
    import pretty_midi
    import numpy as np
    m=pretty_midi.PrettyMIDI(str(path))
    times,tempi=m.get_tempo_changes()
    beats=m.get_beats().tolist()
    meter=m.time_signature_changes[0].numerator if m.time_signature_changes else 4
    if meter not in (3,4): raise ValueError('Starter notation supports 3/4 or 4/4 only; use native notation for this meter')
    if len(beats)<2: beats=[0.,60/float(tempi[0])]
    while beats[-1]<m.get_end_time()+2:beats.append(beats[-1]+60/float(tempi[-1]))
    return float(np.median(tempi)),meter,beats

def active_at(parts,time,chords=None):
    """Point-in-time membership uses raw seconds: onset <= time < offset.

    Only guitar/bass parts are inspected. Chord labels never select which notes survive. Chord membership, frequency
    and instrument names here are model estimates, not measurements of truth.
    """
    rows=[]
    for part,notes in parts.items():
        if part not in ('guitar','bass'):continue
        for n in notes:
            if float(n['onset'])<=time<float(n['offset']):
                p=int(n['pitch']);rows.append({'part':part,'pitch':p,'name':pitch_name(p),
                    'pitch_class':NAMES[p%12],'fundamental_hz':round(440*2**((p-69)/12),2),
                    'onset':n['onset'],'offset':n['offset']})
    rows.sort(key=lambda n:(n['pitch'],n['part']))
    chord=next((s.get('label',s.get('chord')) for s in (chords or {}).get('segments',[])
                if float(s['start'])<=time<float(s['end'])),None)
    return {'time':float(time),'chord_prediction':chord,'active_notes':rows,
            'pitch_classes':[NAMES[p] for p in sorted({n['pitch']%12 for n in rows})],
            'notes_with_octave':[pitch_name(p) for p in sorted({n['pitch'] for n in rows})]}
