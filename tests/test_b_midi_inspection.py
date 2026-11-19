"""Regression tests for the duplicated MuScriptor tempo warning."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import mido
import pretty_midi
from scripts.b_inspect_muscriptor_midi import remove_duplicate_metadata, inspect


def midi_bytes(duplicate=True, conflict=False):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage('set_tempo', tempo=500000),
        mido.MetaMessage('set_tempo', tempo=1000000, time=480),
        mido.MetaMessage('end_of_track'),
    ]))
    track = mido.MidiTrack([mido.Message('program_change', program=0)])
    if duplicate:
        track.append(mido.MetaMessage('set_tempo', tempo=600000 if conflict else 500000))
    track.extend([mido.Message('note_on', note=60, velocity=80),
                  mido.MetaMessage('set_tempo', tempo=1000000, time=480),
                  mido.Message('note_off', note=60, time=480),
                  mido.MetaMessage('end_of_track')])
    midi.tracks.append(track)
    data = io.BytesIO(); midi.save(file=data)
    return data.getvalue()


class MidiInspection(unittest.TestCase):
    def test_duplicate_tempo_removed_without_changing_note_time(self):
        raw = midi_bytes()
        clean, removed = remove_duplicate_metadata(raw)
        self.assertEqual(removed, {'set_tempo':2})
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            midi = pretty_midi.PrettyMIDI(io.BytesIO(clean))
        note = midi.instruments[0].notes[0]
        self.assertEqual(note.pitch,60)
        self.assertEqual((note.start,note.end),(0.,1.5))
        self.assertEqual(midi.instruments[0].program,0)
        self.assertEqual(remove_duplicate_metadata(clean),(clean,{}))

    def test_conflicting_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError,'先頭トラックと一致しません'):
            remove_duplicate_metadata(midi_bytes(conflict=True))

    def test_report_keeps_piano_and_source_bytes_and_uses_actual_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); source=folder/'raw.mid'; source.write_bytes(midi_bytes())
            reference=folder/'reference.json'
            reference.write_text(json.dumps(dict(synthetic=True,accuracy_benchmark=False,
                parts={'guitar':[dict(pitch=60,onset=0.,offset=1.5),dict(pitch=62,onset=2.,offset=2.5)]})))
            original=source.read_bytes()
            report=inspect(source,reference)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(report['guitar_bass_notes_by_model'],0)
            self.assertEqual(report['instruments'][0]['program'],0)
            self.assertEqual(report['matching']['matched'],1)
            self.assertEqual(report['matching']['unmatched_reference'],1)
            self.assertEqual(report['matching']['unmatched_estimates'],0)
            self.assertEqual(len(list(folder.glob('inspection_*/normalized.mid'))),1)
