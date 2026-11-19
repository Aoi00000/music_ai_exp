"""B's score/editor acceptance tests; no acoustic models or network required."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import pretty_midi

from band2tab.score import assign_frets, quantize, TUNINGS, active_at
from band2tab.workflow import load_document, validate_document, prepare_handoff, editor_rows, replace_part, export_document

ROOT = Path(__file__).resolve().parents[1]


class ScoreWorkflow(unittest.TestCase):
    def setUp(self):
        self.document = load_document((ROOT / 'examples/notes.json').read_text(encoding='utf-8'))

    def test_example_preserves_all_nine_raw_notes_and_has_no_issues(self):
        with tempfile.TemporaryDirectory() as d:
            summary = export_document(self.document, d)
            self.assertEqual(summary['notes'], 9)
            self.assertEqual(summary['issues'], 0)
            saved = json.loads((Path(d) / 'notes.json').read_text())
            midi = pretty_midi.PrettyMIDI(str(Path(d) / 'raw.mid'))
            self.assertEqual(sum(len(p.notes) for p in midi.instruments), 9)
            for part, notes in self.document['parts'].items():
                actual = saved['parts'][part]
                self.assertEqual([(n['pitch'], n['onset'], n['offset']) for n in notes],
                                 [(n['pitch'], n['onset'], n['offset']) for n in actual])
                for n in actual:
                    self.assertEqual(TUNINGS[part][n['string'] - 1] + n['fret'], n['pitch'])

    def test_editor_preserves_other_part_and_metadata_but_invalidates_changed_score(self):
        self.document['parts']['guitar'][0].update(acoustic_score=.9, source='A', review_reason='old')
        rows = editor_rows(self.document, 'guitar')
        rows[0]['pitch'] = 53
        edited = replace_part(self.document, 'guitar', rows)
        self.assertEqual(edited['parts']['bass'], self.document['parts']['bass'])
        self.assertEqual(edited['parts']['guitar'][0]['source'], 'A')
        self.assertNotIn('acoustic_score', edited['parts']['guitar'][0])
        self.assertNotIn('review_reason', edited['parts']['guitar'][0])
        self.assertTrue(edited['parts']['guitar'][0]['needs_review'])
        self.assertEqual(self.document['parts']['guitar'][0]['pitch'], 52)
        rows = editor_rows(edited, 'guitar')
        rows.pop(1)
        rows.append(dict(_row=None, pitch=60, onset=2., offset=2.5, velocity=80))
        self.assertEqual(len(replace_part(edited, 'guitar', rows)['parts']['guitar']), 7)

    def test_invalid_edits_are_rejected_without_mutating_document(self):
        before = json.dumps(self.document, sort_keys=True)
        for field, bad in [('pitch', 52.5), ('onset', -.1), ('offset', 0.),
                           ('offset', float('nan')), ('velocity', 0), ('pitch', 128)]:
            with self.subTest(field=field, value=bad):
                rows = editor_rows(self.document, 'guitar')
                rows[0][field] = bad
                with self.assertRaises(ValueError):
                    replace_part(self.document, 'guitar', rows)
        self.assertEqual(json.dumps(self.document, sort_keys=True), before)

    def test_incorrect_timing_maps_are_rejected(self):
        for beats in [[0., 0.], [1., 0.], [0.], [0., float('inf')]]:
            with self.subTest(beats=beats), self.assertRaises(ValueError):
                validate_document(dict(self.document, beat_times=beats))
        for bpm in [0, -120, float('nan')]:
            with self.subTest(bpm=bpm), self.assertRaises(ValueError):
                validate_document(dict(self.document, bpm=bpm))

    def test_variable_tempo_map_is_kept_after_editor_and_export(self):
        doc = dict(self.document, beat_times=[0., .5, 1.5, 2.5],analysis_scope='whole_mix')
        edited = replace_part(doc, 'guitar', editor_rows(doc, 'guitar'))
        with tempfile.TemporaryDirectory() as d:
            export_document(edited, d)
            saved = json.loads((Path(d) / 'notes.json').read_text())
            self.assertEqual(saved['beat_times'], doc['beat_times'])
            self.assertEqual(saved['analysis_scope'], 'whole_mix')
            self.assertEqual(saved['parts']['guitar'][5]['q_start'], 6)

    def test_barline_sustain_has_ties_and_every_voice_fills_measure(self):
        doc = dict(parts={'guitar': [dict(pitch=64, onset=0., offset=4.)]},
                   bpm=120., beats_per_bar=3, grid=3)
        with tempfile.TemporaryDirectory() as d:
            export_document(doc, d)
            root = ET.parse(Path(d) / 'score.musicxml')
            self.assertTrue(root.findall('.//tie[@type="start"]'))
            self.assertTrue(root.findall('.//tie[@type="stop"]'))
            from band2tab.notation import validate_musicxml
            self.assertEqual(validate_musicxml(Path(d)/'score.musicxml'), 3)
            for measure in root.findall('.//measure'):
                self.assertEqual(sum(int(n.findtext('duration')) for n in measure.findall('note')
                                     if n.find('chord') is None), 36)

    def test_example_has_explicit_values_and_complete_measures(self):
        from band2tab.notation import validate_musicxml
        with tempfile.TemporaryDirectory() as d:
            export_document(self.document, d)
            path = Path(d)/'score.musicxml'
            self.assertEqual(validate_musicxml(path), 2)
            root = ET.parse(path)
            self.assertEqual(len(root.findall('.//technical/string')),9)
            self.assertTrue(root.findall('.//chord'))
            # The original exporter omitted types: this must no longer pass.
            note=root.find('.//note');note.remove(note.find('type'));root.write(path)
            with self.assertRaisesRegex(ValueError,'音価'):
                validate_musicxml(path)

    def test_six_string_chord_and_independent_sustain_fit_one_voice(self):
        from band2tab.notation import validate_musicxml
        doc=dict(parts={'guitar':[dict(pitch=p,onset=0.,offset=1.25 if i==0 else .5)
                                 for i,p in enumerate([40,45,50,55,59,64])]},bpm=120.)
        with tempfile.TemporaryDirectory() as d:
            result=export_document(doc,d)
            self.assertEqual(result['issues'],0)
            path=Path(d)/'score.musicxml';validate_musicxml(path)
            root=ET.parse(path)
            self.assertEqual({n.text for n in root.findall('.//voice')},{'1'})
            self.assertEqual(len(root.findall('.//chord')),5)
            self.assertTrue(root.findall('.//tie[@type="start"]'))
            self.assertEqual(sum(1 for n in root.findall('.//note') if n.find('pitch') is not None
                                 and n.find('tie[@type="stop"]') is None),6)

    def test_triplets_and_nonstandard_lengths_preserve_quantized_intervals(self):
        from fractions import Fraction
        from band2tab.notation import validate_musicxml
        for grid in (3,4):
            with self.subTest(grid=grid),tempfile.TemporaryDirectory() as d:
                doc=dict(parts={'guitar':[dict(pitch=64,onset=0.,offset=2.5/grid)]},
                         bpm=120.,grid=grid,beats_per_bar=4)
                export_document(doc,d)
                path=Path(d)/'score.musicxml';validate_musicxml(path)
                root=ET.parse(path)
                duration=sum(int(n.findtext('duration')) for n in root.findall('.//note') if n.find('pitch') is not None)
                self.assertEqual(Fraction(duration,12),Fraction(5,grid))
                if grid==3:self.assertTrue(root.findall('.//time-modification'))

    def test_unassigned_and_quantization_collision_are_visible_but_raw_midi_keeps_notes(self):
        doc = dict(parts={'guitar': [dict(pitch=0, onset=0., offset=1.),
                                    dict(pitch=40, onset=.01, offset=.04),
                                    dict(pitch=40, onset=.05, offset=.09)]}, bpm=120.)
        with tempfile.TemporaryDirectory() as d:
            summary = export_document(doc, d)
            saved = json.loads((Path(d) / 'notes.json').read_text())
            self.assertEqual(summary['notes'], 3)
            self.assertEqual({x['reason'] for x in saved['issues']},
                             {'unplayable_or_out_of_range', 'quantization_collision'})
            self.assertIn('C-1', (Path(d) / 'tab.txt').read_text())
            self.assertEqual(sum(len(i.notes) for i in pretty_midi.PrettyMIDI(str(Path(d) / 'raw.mid')).instruments), 3)

    def test_stale_unassigned_reason_is_cleared_when_feasible(self):
        notes = assign_frets([dict(pitch=40, onset=0., offset=1., review_reason='unplayable_or_out_of_range')])
        self.assertIsNotNone(notes[0]['string'])
        self.assertNotIn('review_reason', notes[0])

    def test_snapshot_chords_do_not_remove_extra_tones_and_use_half_open_times(self):
        doc = {'guitar': [dict(pitch=p, onset=0., offset=1.) for p in (48, 52, 55, 69)],
               'bass': [dict(pitch=28, onset=0., offset=1.)]}
        chords = dict(segments=[dict(start=0., end=1., label='C:maj')])
        self.assertEqual(active_at(doc, .5, chords)['notes_with_octave'], ['E1', 'C3', 'E3', 'G3', 'A4'])
        self.assertFalse(active_at(doc, 1., chords)['active_notes'])

    def test_chord_annotation_does_not_change_notes(self):
        with tempfile.TemporaryDirectory() as d:
            export_document(self.document, d, dict(segments=[dict(start=0., end=2., label='C:maj7/3')]))
            root = ET.parse(Path(d) / 'score.musicxml')
            self.assertTrue(root.findall('.//harmony'))
            self.assertEqual(len(root.findall('.//technical/string')), 9)
            self.assertEqual(root.findtext('.//harmony/bass/bass-step'), 'E')

    def test_initial_page_and_example_export(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == '既知音符の例を開く').click().run()
        self.assertFalse(app.exception)
        # Keep AppTest's artificial edits out of the user's latest-saved lookup.
        self.addCleanup(shutil.rmtree,Path(app.session_state.run),ignore_errors=True)
        self.assertTrue(app.success)
        folder = Path(app.session_state.output_dir)
        self.assertEqual(len(json.loads((folder / 'notes.json').read_text())['parts']['guitar']), 7)
        # Applying edits switches downloads to a new, successfully generated export.
        next(b for b in app.button if b.label == '修正を反映する').click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(Path(app.session_state.output_dir),folder)
        self.assertTrue((folder/'notes.json').exists())

    def test_a_handoff_defaults_to_raw_and_retains_the_correction_proposal(self):
        source=dict(parts={'guitar':[dict(pitch=64,onset=0.,offset=.5)]},
                    raw_parts={'guitar':[dict(pitch=60,onset=.1,offset=.6)]},
                    corrector_applied=True,midi_tempo_is_placeholder=True,source_sha256='recording-A')
        with self.assertRaisesRegex(ValueError,'BPM'):
            prepare_handoff(source)
        baseline=prepare_handoff(source,bpm=96.,meter=3,grid=4)
        proposed=prepare_handoff(source,note_set='corrected',bpm=96.)
        self.assertEqual(baseline['parts'],source['raw_parts'])
        self.assertEqual(baseline['proposed_parts'],source['parts'])
        self.assertFalse(baseline['corrector_applied'])
        self.assertTrue(baseline['source_corrector_applied'])
        self.assertEqual(proposed['parts'],source['parts'])
        self.assertTrue(proposed['corrector_applied'])
        self.assertEqual(baseline['source_sha256'],'recording-A')
        self.assertEqual(source['parts']['guitar'][0]['pitch'],64)
        with tempfile.TemporaryDirectory() as d:
            export_document(baseline,d)
            saved=json.loads((Path(d)/'notes.json').read_text())
            self.assertEqual(saved['raw_parts'],source['raw_parts'])
            self.assertEqual(saved['proposed_parts'],source['parts'])
            self.assertEqual(saved['bpm'],96.)
            self.assertEqual(prepare_handoff(saved,note_set='corrected')['parts'],source['parts'])
            rows=editor_rows(saved,'guitar');rows[0]['pitch']=61
            edited=replace_part(saved,'guitar',rows)
            self.assertEqual(prepare_handoff(edited)['parts']['guitar'][0]['pitch'],61)

    def test_a_handoff_does_not_assume_midi_placeholder_is_the_song_tempo(self):
        document=dict(self.document,midi_tempo_is_placeholder=True)
        with self.assertRaisesRegex(ValueError,'BPM'):
            prepare_handoff(document)
        confirmed=prepare_handoff(document,bpm=100.)
        self.assertEqual(prepare_handoff(confirmed)['bpm'],100.)
        self.assertEqual(prepare_handoff(self.document)['bpm'],120.)
        with self.assertRaisesRegex(ValueError,'補正案'):
            prepare_handoff(self.document,note_set='corrected')
        with self.assertRaisesRegex(ValueError,'raw_parts'):
            prepare_handoff(dict(self.document,corrector_applied=True))

    def test_manual_handoff_tempo_override_keeps_the_source_beat_map(self):
        document=dict(self.document,beat_times=[0.,.5,1.5,2.5])
        unchanged=prepare_handoff(document)
        self.assertEqual(unchanged['beat_times'],document['beat_times'])
        fixed=prepare_handoff(document,bpm=96.,meter=3,grid=3)
        self.assertNotIn('beat_times',fixed)
        self.assertEqual(fixed['source_beat_times'],document['beat_times'])
        self.assertEqual((fixed['bpm'],fixed['beats_per_bar'],fixed['grid']),(96.,3,3))


class SeparateSuggestionHandoff(unittest.TestCase):
    def fixture(self):
        raw = {'guitar':[dict(pitch=60,onset=0.,offset=.5)],
               'bass':[dict(pitch=28,onset=0.,offset=1.)]}
        suggestion = {'guitar':raw['guitar']+[dict(pitch=64,onset=.5,offset=1.)],
                      'bass':raw['bass']}
        return dict(parts=raw,raw_parts=raw,suggested_parts=suggestion,
                    requires_review=True,automatic_application_approved=False,
                    ranker_sha256='fixture-not-a-real-model')

    def test_separate_suggestions_require_tempo_and_explicit_selection(self):
        source = self.fixture()
        with self.assertRaisesRegex(ValueError,'BPM'):
            prepare_handoff(source)
        raw = prepare_handoff(source,bpm=96.)
        proposal = prepare_handoff(source,'corrected',bpm=96.)
        self.assertEqual(raw['parts'],source['parts'])
        self.assertEqual(proposal['parts'],source['suggested_parts'])
        self.assertEqual(raw['proposed_parts'],source['suggested_parts'])
        self.assertFalse(raw['corrector_applied'])
        self.assertFalse(proposal['automatic_application_approved'])
        self.assertNotIn('bpm',source)
        incomplete = self.fixture();incomplete.pop('raw_parts')
        with self.assertRaisesRegex(ValueError,'raw_parts'):
            prepare_handoff(incomplete,bpm=96.)

    def test_suggestion_export_reimport_keeps_raw_edits_and_proposal(self):
        source = self.fixture()
        raw = prepare_handoff(source,bpm=96.)
        rows = editor_rows(raw,'guitar');rows[0]['pitch']=61
        edited = replace_part(raw,'guitar',rows)
        with tempfile.TemporaryDirectory() as folder:
            export_document(edited,folder)
            saved = json.loads((Path(folder)/'notes.json').read_text())
        self.assertEqual(prepare_handoff(saved)['parts']['guitar'][0]['pitch'],61)
        self.assertEqual(prepare_handoff(saved,'corrected')['parts'],source['suggested_parts'])
        self.assertEqual(saved['raw_parts'],source['raw_parts'])
        self.assertEqual(saved['ranker_sha256'],'fixture-not-a-real-model')
        self.assertFalse(saved['automatic_application_approved'])


if __name__ == '__main__':
    unittest.main()
