"""Seconds-mode integrity and timing gates, independent of inference or audio models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from band2tab.integration import validate_review,select_view,apply_one_edit,save_review,confirm_timing,engrave,difference,restore_review
from band2tab.score import active_at


class ABReview(unittest.TestCase):
    def fixture(self):
        raw={'guitar':[dict(pitch=60,onset=.1,offset=.4,velocity=80),dict(pitch=64,onset=.5,offset=1.,velocity=80)],
             'bass':[dict(pitch=28,onset=0.,offset=1.,velocity=80)]}
        return dict(parts=deepcopy(raw),raw_parts=deepcopy(raw),suggested_parts=deepcopy(raw),source_baseline=dict(parts=deepcopy(raw)),
                    source_bundle_checksums={},source_bundle_root='fixture',source_bundle='test-only',
                    timing=dict(bpm=None,meter=None,first_downbeat_seconds=None),bpm=None,beats_per_bar=None,first_downbeat_seconds=None,
                    notation_timing_confirmed=False,duration=1.,human_edit=False,selected_view='baseline')

    def rows(self,d):
        return [dict(n) for n in d['parts']['guitar']]

    def test_unknown_timing_stays_unknown_and_cannot_engrave(self):
        d=validate_review(self.fixture())
        self.assertIsNone(d['bpm']);self.assertIsNone(d['beats_per_bar'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'拍未確認'):engrave(d,tmp,{'segments':[]})
            self.assertEqual(list(Path(tmp).iterdir()),[])
        with self.assertRaises(ValueError):confirm_timing(d,120.,4,None,True)
        with self.assertRaises(ValueError):confirm_timing(d,120.,4,0.,False)

    def test_one_edit_preserves_other_notes_baseline_proposal_and_timing(self):
        d=self.fixture();rows=self.rows(d);rows[0]['pitch']=61
        with self.assertRaisesRegex(ValueError,'原音'):apply_one_edit(d,'guitar',rows,'test edit',False)
        edited=apply_one_edit(d,'guitar',rows,'synthetic test, not a listened correction',True)
        self.assertEqual(len(difference(d['parts'],edited['parts'])),1)
        for field in ['raw_parts','suggested_parts','source_baseline','timing','bpm','beats_per_bar']:
            self.assertEqual(d[field],edited[field])
        self.assertEqual(edited['parts']['bass'],d['parts']['bass'])
        self.assertEqual(edited['parts']['guitar'][1],d['parts']['guitar'][1])
        with tempfile.TemporaryDirectory() as tmp:
            out,report=save_review(edited,tmp,{'segments':[]})
            saved=json.loads((out/'notes.json').read_text())
            self.assertEqual(saved,edited)
            self.assertTrue(report['suggestion_is_identity'])
            self.assertEqual(len(report['edit_diff']),1)
            back=select_view(select_view(saved,'baseline'),'human')
            self.assertEqual(back['parts'],edited['parts'])

    def test_restore_rejects_tampered_baseline_or_filled_unconfirmed_timing(self):
        d=self.fixture()
        self.assertEqual(restore_review(d,d),d)
        corrupt=deepcopy(d);corrupt['raw_parts']['bass'][0]['pitch']=29
        with self.assertRaisesRegex(ValueError,'元結果'):restore_review(corrupt,d)
        corrupt=deepcopy(d);corrupt['bpm']=120.
        with self.assertRaisesRegex(ValueError,'未確認'):restore_review(corrupt,d)

    def test_two_edits_rejected_without_mutation(self):
        d=self.fixture();before=deepcopy(d);rows=self.rows(d)
        rows[0]['pitch']=61;rows[1]['pitch']=65
        with self.assertRaisesRegex(ValueError,'2行'):apply_one_edit(d,'guitar',rows,'test',True)
        self.assertEqual(d,before)

    def test_seconds_membership_is_half_open_and_retains_non_chord_tone(self):
        d=self.fixture();chords={'segments':[dict(start=0,end=.5,label='C:maj'),dict(start=.5,end=1.,label='D:maj')]}
        first=active_at(d['parts'],.4,chords)
        self.assertNotIn('C4',first['notes_with_octave'])
        boundary=active_at(d['parts'],.5,chords)
        self.assertEqual(boundary['chord_prediction'],'D:maj')
        self.assertIn('E4',boundary['notes_with_octave'])
        self.assertIn('E1',boundary['notes_with_octave'])
        self.assertEqual(active_at(d['parts'],1.,chords)['active_notes'],[])

    def test_confirmed_pickup_uses_explicit_notation_preroll_keeps_raw_times(self):
        d=confirm_timing(self.fixture(),120.,4,.5,True)
        with tempfile.TemporaryDirectory() as tmp:
            out,_=engrave(d,tmp,{'segments':[]})
            saved=json.loads((out/'notes.json').read_text())
            self.assertEqual(difference(d['parts'],saved['parts']),[])
            self.assertAlmostEqual(saved['notation_preroll_seconds'],1.5)
            self.assertAlmostEqual(saved['beat_times'][4],.5)

    def test_inference_missing_dependencies_fails_before_creating_job(self):
        from band2tab.inference_job import launch
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'必要な'):
                launch('fixture.wav',Path(tmp)/'jobs','separated',Path(tmp)/'missing-python','medium')
            self.assertFalse((Path(tmp)/'jobs').exists())


class InferenceContract(unittest.TestCase):
    def test_virtualenv_path_is_not_resolved_into_system_python(self):
        import sys
        from band2tab.inference_job import launch
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);python=p/'venv/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
            source=p/'input.wav';source.write_bytes(b'fixture')
            with patch('band2tab.inference_job.requirements',return_value=[]),patch('band2tab.inference_job.subprocess.Popen') as process:
                job=launch(source,p/'jobs','separated',python,'medium','cpu')
            saved=json.loads(job.read_text())
            self.assertEqual(saved['python'],str(python))
            self.assertEqual(saved['seconds'],12)
            self.assertEqual((saved['model'],saved['device']),('medium','cpu'))
            self.assertTrue(process.call_args.kwargs['start_new_session'])

    def test_separated_worker_command_and_output_validation(self):
        from types import SimpleNamespace
        from band2tab.inference_job import run
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);job=folder/'job.json'
            job.write_text(json.dumps(dict(source=str(folder/'original.wav'),python='fixture-python',engine='separated',seconds=12,model='medium',device='cpu',dtype='float32')))
            commands=[]
            def fake_run(command,**kwargs):
                commands.append(command)
                if 'scripts.a_band_pipeline' in command:
                    out=Path(command[command.index('--out')+1]);out.mkdir()
                    (out/'notes.json').write_text(json.dumps(dict(parts={'guitar':[dict(pitch=60,onset=0.,offset=1.)]},midi_tempo_is_placeholder=True)))
                return SimpleNamespace(stdout='{}',returncode=0)
            with patch('band2tab.inference_job.subprocess.run',side_effect=fake_run):run(job)
            saved=json.loads(job.read_text())
            self.assertEqual(saved['status'],'complete')
            self.assertEqual(saved['note_counts'],{'guitar':1})
            self.assertIn('--seconds',commands[0]);self.assertNotIn('--bpm',commands[0])
            self.assertIn('max_child_rss_bytes',saved)
