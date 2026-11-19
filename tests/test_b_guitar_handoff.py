"""Ensure the real-audio handoff never substitutes annotation notes for estimates."""
import json
from pathlib import Path
import tempfile
import unittest
import pretty_midi
from scripts.b_export_guitar_handoff import export_handoff


class GuitarHandoff(unittest.TestCase):
    def fixture(self,folder,program):
        midi=pretty_midi.PrettyMIDI(initial_tempo=120)
        part=pretty_midi.Instrument(program=program)
        part.notes.append(pretty_midi.Note(80,60,.1,.8));midi.instruments.append(part)
        source=folder/'raw.mid';midi.write(str(source))
        reference=folder/'reference.json'
        reference.write_text(json.dumps(dict(dataset='GuitarSet',recording_id='00_BN1-129-Eb_comp',
            bpm=129.,beats_per_bar=4,beat_times=[i*60/129 for i in range(30)],timing_source='test dataset annotation',
            parts={'guitar':[dict(pitch=62,onset=.1,offset=.8),dict(pitch=64,onset=1.,offset=1.5)]})))
        return source,reference

    def test_model_pitch_time_and_source_kept_with_annotation_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            source,reference=self.fixture(Path(temp),24)
            original=source.read_bytes(); ref_original=reference.read_bytes()
            folder,report=export_handoff(source,reference)
            result=json.loads((folder/'notes.json').read_text())
            self.assertEqual(len(result['parts']['guitar']),1)
            self.assertEqual(result['parts']['guitar'][0]['pitch'],60)
            self.assertAlmostEqual(result['parts']['guitar'][0]['onset'],.1)
            self.assertAlmostEqual(result['parts']['guitar'][0]['offset'],.8)
            self.assertEqual(result['bpm'],129.)
            self.assertEqual(result['notation_tempo_source'],'dataset_annotation')
            self.assertEqual(report['comparison']['matched'],0)
            self.assertEqual(report['comparison']['reference_notes'],2)
            self.assertEqual((folder/'source.mid').read_bytes(),original)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(reference.read_bytes(),ref_original)

    def test_piano_does_not_become_guitar(self):
        with tempfile.TemporaryDirectory() as temp:
            source,reference=self.fixture(Path(temp),0)
            with self.assertRaisesRegex(ValueError,'楽器を自動で付け替えず'):
                export_handoff(source,reference)
            self.assertFalse(list(Path(temp).glob('handoff_*')))
