"""FiloBass: use performance-aligned MIDI, not score/beat-aligned MIDI."""
from pathlib import Path
import csv,hashlib,re
import pretty_midi
import soundfile as sf

def records(root):
    root=Path(root);normalize=lambda s:re.sub('[^a-z0-9]','',s.lower())
    players={}
    with (root/'notebooks_and_scripts/note_data.csv').open(newline='') as f:
        for row in csv.DictReader(f):players.setdefault(normalize(row['title']),set()).add(row['player'])
    if any(len(p)!=1 for p in players.values()):raise ValueError('Ambiguous FiloBass performer metadata')
    names=sorted({next(iter(v)) for v in players.values()},key=lambda n:hashlib.sha256(n.encode()).hexdigest())
    split={name:('test' if i<2 else 'val' if i<4 else 'train') for i,name in enumerate(names)}
    midis={normalize(p.stem):p for p in (root/'midi_fully_aligned').glob('*.mid')}
    result=[]
    for audio in sorted((root/'audio_bass_stems').glob('*.mp3')):
        key=normalize(audio.stem)
        if key not in players or key not in midis:raise ValueError('Unmatched FiloBass track: '+audio.name)
        midi=pretty_midi.PrettyMIDI(str(midis[key]));duration=sf.info(audio).duration
        notes=[{'pitch':int(n.pitch),'onset':float(n.start),'offset':float(n.end)} for ins in midi.instruments if not ins.is_drum for n in ins.notes]
        if not notes or any(not(23<=n['pitch']<=88 and 0<=n['onset']<n['offset']<=duration+.1) for n in notes):raise ValueError('Invalid FiloBass concert-pitch or time range: '+audio.name)
        player=next(iter(players[key]))
        # MIDI program 26 is present in the author's bass MIDI. The known role
        # comes from this bass-stem dataset, not the MIDI's GM program number.
        result.append({'id':'filobass_'+audio.stem,'audio':str(audio.resolve()),'notes':notes,'part':'bass','dataset':'filobass',
                       'group':'filobass_player_'+player,'player':player,'split':split[player],
                       'midi':str(midis[key].resolve()),'source_programs':sorted({int(i.program) for i in midi.instruments}),
                       'label_semantics':'performance_aligned_concert_pitch_no_transposition'})
    if len(result)!=48 or len(names)!=12:raise ValueError('Expected the 48-track, 12-player official FiloBass release')
    return result
