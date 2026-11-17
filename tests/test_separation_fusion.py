import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from band2tab.fusion import candidate_pool,propose
from band2tab.separation import write_input
from band2tab.route import predict,route_features

def note(pitch=60,start=0.,end=1.,cls=1):
    return {'pitch':pitch,'onset':start,'offset':end,'velocity':90,'program':24 if cls==1 else 32,'baseline_class':cls}

class SeparationFusionTests(unittest.TestCase):
    def test_identity_preserves_both_instruments_at_unison(self):
        original=[note(cls=1),note(cls=2)]
        pools=candidate_pool(original,{'guitar':[note()], 'bass':[note(cls=2)]})
        predicted,changes=propose(pools,np.ones(4))
        self.assertEqual([n['pitch'] for n in predicted['guitar']],[60])
        self.assertEqual([n['pitch'] for n in predicted['bass']],[60])
        self.assertEqual(changes,[])

    def test_stem_support_cannot_move_original_note(self):
        original=[note()];before=copy.deepcopy(original)
        pools=candidate_pool(original,{'guitar':[note(start=.02,end=1.2)]})
        self.assertEqual(len(pools['guitar']),1)
        self.assertEqual(pools['guitar'][0]['onset'],0.)
        self.assertEqual(pools['guitar'][0]['offset'],1.)
        self.assertEqual(pools['guitar'][0]['support'],[1.,1.,0.])
        self.assertEqual(original,before)

    def test_missing_pitch_can_be_proposed_with_provenance(self):
        pools=candidate_pool([note()],{'guitar':[note(pitch=64)]})
        # Guitar pool is [original C, new E], bass pool contains original C.
        predicted,changes=propose(pools,[.9,.95,.1],add_threshold=.9)
        self.assertEqual([n['pitch'] for n in predicted['guitar']],[60,64])
        self.assertEqual(predicted['bass'],[])
        self.assertEqual(changes[0]['origin'],'separated_guitar')
        self.assertEqual(changes[0]['action'],'add')

    def test_separate_repeated_notes_are_not_merged(self):
        pools=candidate_pool([note(end=.2)],{'guitar':[note(start=.4,end=.6)]})
        self.assertEqual(len(pools['guitar']),2)

    def test_canonical_audio_retains_level_and_duration(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'input.wav';sr=32000
            y=np.repeat(np.full((sr,1),.12,np.float32),2,axis=1)
            write_input(path,y,sr);out,rate=sf.read(path)
            self.assertEqual(rate,16000);self.assertEqual(len(out),16000)
            self.assertAlmostEqual(float(np.median(out)),.12,places=5)
            self.assertEqual(sf.info(path).subtype,'FLOAT')

    def test_route_tree_uses_gain_guard_and_exact_boundary(self):
        tree={'children_left':[1,-1,-1],'children_right':[2,-1,-1],
              'feature':[0,-2,-2],'threshold':[.1,-2.,-2.],
              'value':[[0.,0.,0.,0.],[0.,-.1,.03,.01],[0.,-.1,.2,.05]]}
        self.assertEqual(predict(tree,[.1],.05)[0],'M0_original')
        name,gain=predict(tree,[.11],.05)
        self.assertEqual(name,'M2_separated_forced');self.assertAlmostEqual(gain,.2)

    def test_route_resampling_rounding_allows_only_two_samples(self):
        import pretty_midi
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);midi=td/'mix.mid';pretty_midi.PrettyMIDI().write(str(midi));inputs={}
            for k in ('mix','no_drums_vocals','guitar','bass'):
                path=td/(k+'.wav');sf.write(path,np.zeros(16000 if k=='mix' else 16001,np.float32),16000,subtype='FLOAT');inputs[k]=path
            self.assertTrue(np.isfinite(route_features(inputs,midi)).all())
            sf.write(inputs['bass'],np.zeros(16010,np.float32),16000,subtype='FLOAT')
            with self.assertRaises(ValueError):route_features(inputs,midi)

if __name__=='__main__':unittest.main()
