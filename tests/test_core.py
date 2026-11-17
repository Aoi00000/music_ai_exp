import json,tempfile,unittest
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
from band2tab.score import assign_frets,write_outputs,musicxml,quantize,TUNINGS
from band2tab.audio import decode,note_targets,N_PITCHES

class Core(unittest.TestCase):
    def test_double_stop_and_sustain(self):
        notes=[{'pitch':60,'onset':0.,'offset':2.},{'pitch':64,'onset':0.,'offset':1.},{'pitch':62,'onset':1.,'offset':1.8}]
        out=assign_frets(notes)
        self.assertEqual(len(out),3);self.assertTrue(all(n['string'] for n in out))
        for n in out:self.assertEqual(TUNINGS['guitar'][n['string']-1]+n['fret'],n['pitch'])
        for i,a in enumerate(out):
            for b in out[i+1:]:
                if max(a['onset'],b['onset'])<min(a['offset'],b['offset']):self.assertNotEqual(a['string'],b['string'])
    def test_unplayable_is_not_dropped(self):
        out=assign_frets([{'pitch':40+i,'onset':0,'offset':1} for i in range(8)])
        self.assertEqual(len(out),8);self.assertTrue(any(n['string'] is None for n in out))
    def test_pitch_and_raw_times_retained_after_export(self):
        with tempfile.TemporaryDirectory() as d:
            notes=[{'pitch':28,'onset':.123,'offset':1.234},{'pitch':35,'onset':.123,'offset':.8}]
            write_outputs({'bass':notes},d,bpm=100);j=json.loads((Path(d)/'notes.json').read_text())
            self.assertEqual(j['parts']['bass'][0]['onset'],.123)
            self.assertEqual(sorted(n['pitch'] for n in j['parts']['bass']),[28,35])
            root=E.parse(Path(d)/'score.musicxml')
            self.assertEqual(len(root.findall('.//technical/string')),2);self.assertEqual(root.findtext('.//staff-lines'),'4')
    def test_rearticulation_and_multilabel(self):
        p=np.zeros((20,3,N_PITCHES),np.float32);p[1:18,1,40]=.9;p[[1,9],0,40]=.9
        p[1:10,1,44]=.9;p[1,0,44]=.9;notes=decode(p)
        self.assertEqual(sum(n['pitch']==63 for n in notes),2);self.assertEqual(sum(n['pitch']==67 for n in notes),1)
    def test_quantization_collision_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            write_outputs({'guitar':[{'pitch':40,'onset':.01,'offset':.04},{'pitch':40,'onset':.05,'offset':.09}]},d)
            j=json.loads((Path(d)/'notes.json').read_text());self.assertEqual(len(j['parts']['guitar']),2)
            self.assertTrue(any(i['reason']=='quantization_collision' for i in j['issues']))
    def test_note_label_and_group_splits(self):
        from band2tab.data import group_split
        y,n=note_targets([{'pitch':40,'onset':.1,'offset':.3},{'pitch':44,'onset':.1,'offset':.3}],50)
        self.assertEqual(n,0);self.assertEqual(int(y[:,0].sum()),2)
        self.assertEqual(group_split('guitarset_BN1'),group_split('guitarset_BN1'))
    def test_snapshot_keeps_non_chord_tone_and_octave(self):
        from band2tab.score import active_at
        parts={'guitar':[{'pitch':48,'onset':0.,'offset':1.},{'pitch':52,'onset':0.,'offset':1.},{'pitch':55,'onset':0.,'offset':1.},{'pitch':69,'onset':.2,'offset':.8}],
               'Piano':[{'pitch':78,'onset':.2,'offset':.8}]}
        chords={'segments':[{'start':0.,'end':1.,'label':'C:maj'}]}
        j=active_at(parts,.5,chords)
        self.assertEqual(j['chord_prediction'],'C:maj')
        self.assertEqual(j['pitch_classes'],['C','E','G','A'])
        self.assertEqual(j['notes_with_octave'],['C3','E3','G3','A4'])
        self.assertEqual(len(active_at(parts,.8,chords)['active_notes']),3)
        self.assertEqual(len(active_at(parts,1.,chords)['active_notes']),0)
    def test_active_set_metric_penalizes_missing_extra_tone(self):
        from band2tab.learn import pitch_set_metrics
        ref=[{'pitch':p,'onset':0.,'offset':1.} for p in (48,52,55,69)]
        result=pitch_set_metrics(ref,ref[:3])
        self.assertAlmostEqual(result['frame_recall'],.75)
        self.assertAlmostEqual(result['frame_precision'],1.)
        self.assertEqual(result['active_exact_match'],0.)
    def test_snapshot_includes_bass_and_excludes_other_instruments(self):
        from band2tab.score import active_at
        parts={'drums':[{'pitch':36,'onset':.5,'offset':.6}],
               'Piano':[{'pitch':69,'onset':.2,'offset':.8}],
               'bass':[{'pitch':36,'onset':.2,'offset':.8}],
               'guitar':[{'pitch':48,'onset':.2,'offset':.8}]}
        snap=active_at(parts,.52)
        self.assertEqual(snap['pitch_classes'],['C'])
        self.assertEqual(snap['notes_with_octave'],['C2','C3'])
        self.assertEqual({n['part'] for n in snap['active_notes']},{'guitar','bass'})
    def test_baby_song_split_keeps_all_three_sets(self):
        from band2tab.data import slakh_split_map
        from collections import Counter
        with tempfile.TemporaryDirectory() as d:
            for i in range(20):
                p=Path(d)/f'Track{i:05d}';p.mkdir();(p/'metadata.yaml').write_text(f'UUID: source-{i}\n')
            split=slakh_split_map(d)
            self.assertEqual(Counter(split.values()),{'train':14,'val':3,'test':3})
            self.assertEqual(split,slakh_split_map(d))
    def test_decoder_matches_reference_state_machine(self):
        from band2tab.audio import HOP,SR,PITCH_MIN
        rng=np.random.default_rng(4)
        for length in (0,1,20,100):
            prob=np.round(rng.random((length,3,5)),1).astype('float32')
            expected=[];dt=HOP/SR
            for pi in range(5):
                start=None
                for t in range(length):
                    peak=prob[t,0,pi]>=.5 and (t==0 or prob[t,0,pi]>prob[t-1,0,pi]) and (t+1==length or prob[t,0,pi]>=prob[t+1,0,pi])
                    if start is not None and (peak or prob[t,1,pi]<.3 or prob[t,2,pi]>=.8):
                        if (t-start)*dt>=.03:expected.append((pi+PITCH_MIN,start*dt,t*dt))
                        start=None
                    if peak:start=t
                if start is not None and (length-start)*dt>=.03:expected.append((pi+PITCH_MIN,start*dt,length*dt))
            actual=decode(prob,onset=.5,frame=.3,offset=.8)
            self.assertEqual(sorted(expected),sorted((n['pitch'],n['onset'],n['offset']) for n in actual))
    def test_evaluation_includes_every_track(self):
        from unittest.mock import patch
        from band2tab.learn import evaluate
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);np.savez(d/'x.npz',x=np.ones((8,264),np.float32))
            items=[{'id':str(i),'dataset':'test','part':'guitar','group':str(i),'split':'val','notes':[], 'cache':str(d/'x.npz')} for i in range(3)]
            (d/'manifest.json').write_text(json.dumps(items));(d/'model.pt').write_bytes(b'fake')
            with patch('band2tab.learn.load',return_value=(None,None)),patch('band2tab.learn.predict',return_value=np.zeros((8,3,66),np.float32)):
                result=evaluate(d/'model.pt',d/'manifest.json',out=d/'eval.json',tune=True)
            self.assertEqual(len(result['tracks']),3)
            self.assertEqual({r['id'] for r in result['tracks']},{'0','1','2'})
    def test_false_rendered_flag_does_not_discard_existing_pairs(self):
        import pretty_midi
        from band2tab.data import slakh_items
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                folder=Path(d)/f'Track{i:05d}';(folder/'stems').mkdir(parents=True);(folder/'MIDI').mkdir()
                (folder/'metadata.yaml').write_text(f'UUID: song-{i}\nstems:\n  S00:\n    inst_class: Guitar\n    audio_rendered: false\n    is_drum: false\n')
                (folder/'stems'/'S00.wav').write_bytes(b'existing audio')
                midi=pretty_midi.PrettyMIDI();ins=pretty_midi.Instrument(24);ins.notes.append(pretty_midi.Note(80,60,0.,1.));midi.instruments.append(ins);midi.write(str(folder/'MIDI'/'S00.mid'))
            items=list(slakh_items(d,'baby'))
            self.assertEqual(len(items),3);self.assertEqual({i['split'] for i in items},{'train','val','test'})
            self.assertTrue(all(i['metadata_audio_rendered'] is False and len(i['notes'])==1 for i in items))
    def test_model_shape_and_gradient(self):
        import torch
        from band2tab.model import NoteCRNN
        m=NoteCRNN();y=m(torch.rand(2,24,264),torch.tensor([0,1]))
        self.assertEqual(tuple(y.shape),(2,24,3,66));y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
if __name__=='__main__':unittest.main()
