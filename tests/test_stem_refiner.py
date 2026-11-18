import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from band2tab.stem_refiner import supervised_targets,feature_matrix,suggest,read_candidates,StemNoteCorrector,correction_loss


def note(part='guitar',a=.2,b=.8):
    return dict(pitch=48,onset=a,offset=b,velocity=80,program=25,
                baseline_class=1,part=part,baseline=[float(part=='guitar'),float(part=='bass')])


class StemRefinerTests(unittest.TestCase):
    def test_assignment_is_per_source_and_unique(self):
        pool=[note(),note(a=.23),note('bass',a=.2)]
        y=supervised_targets(pool,{'guitar':[note(a=.23)],'bass':[note('bass')]},0,1.)
        np.testing.assert_array_equal(y['y'],[0,1,1])
        self.assertEqual(y['mask'].sum(),3)

    def test_censored_edges_guard_and_empty(self):
        n=note();n['censored_offset']=True
        y=supervised_targets([note(),note(a=0)],{'guitar':[n]},.05,1.)
        self.assertEqual(y['timing_mask'][0,1],0)
        self.assertEqual(y['mask'][1],0)
        self.assertEqual(supervised_targets([],{},0,1.)['delta'].shape,(0,2))

    def test_features_audio_only_finite_and_empty(self):
        views={p:np.ones((100,264),np.float32) for p in ('mix','guitar','bass')}
        env={p:np.ones(100,np.float32) for p in views}
        x=feature_matrix(views,env,[note(),note('bass')],1.)
        self.assertEqual(x.shape,(2,350));self.assertTrue(np.isfinite(x).all())
        self.assertEqual(feature_matrix(views,env,[],1.).shape,(0,350))

    def test_suggest_preserves_roles_and_outside_and_valid_timing(self):
        pool=[note(),note('bass')]
        parts=suggest(pool,[.9,.1],np.array([[1,-1],[0,0]]),1.,unmodelled={'bass':[dict(pitch=20,onset=0,offset=.5)]})
        self.assertEqual(len(parts['guitar']),1);self.assertEqual(parts['bass'][0]['pitch'],20)
        self.assertGreater(parts['guitar'][0]['offset'],parts['guitar'][0]['onset'])
        with self.assertRaises(ValueError):suggest(pool,[.9],np.zeros((2,2)),1.)
        with self.assertRaises(ValueError):suggest(pool,[.9,.1],np.full((2,2),np.nan),1.)

    def test_midi_candidates_serializable(self):
        import pretty_midi
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x.mid';m=pretty_midi.PrettyMIDI();i=pretty_midi.Instrument(25)
            i.notes=[pretty_midi.Note(80,48,.2,.8)];m.instruments=[i];m.write(str(path))
            json.dumps(read_candidates(path,'guitar',1.))

    def test_masked_targets_do_not_change_gradient(self):
        model=StemNoteCorrector(64).eval();output=model(torch.zeros(2,350))
        t=dict(y=torch.tensor([1.,0.]),mask=torch.tensor([1.,0.]),delta=torch.zeros(2,2),timing_mask=torch.tensor([[1.,0.],[0.,0.]]))
        first=correction_loss(output,t);changed={k:v.clone() for k,v in t.items()};changed['y'][1]=1;changed['delta'][1]=100
        self.assertTrue(torch.equal(first,correction_loss(output,changed)));first.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)
