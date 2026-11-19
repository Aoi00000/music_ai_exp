import unittest
import numpy as np,torch
from band2tab.release import ReleaseCorrector,apply_release,release_loss


class ReleaseTests(unittest.TestCase):
    def test_every_attack_role_pitch_and_duplicate_survives(self):
        pool=[dict(pitch=48,part=p,onset=.2,offset=.4,velocity=80) for p in ('guitar','guitar','bass')]
        parts=apply_release(pool,np.array([-20,20,1]),1.)
        self.assertEqual([len(parts[p]) for p in ('guitar','bass')],[2,1])
        for ns in parts.values():
            for n in ns:self.assertEqual((n['pitch'],n['onset']),(48,.2));self.assertLessEqual(abs(n['offset']-.4),.1000001)
        self.assertEqual(apply_release([],np.array([]),1.),{'guitar':[],'bass':[]})

    def test_zero_scale_and_outside_notes_are_unchanged(self):
        n=dict(pitch=48,part='guitar',onset=.2,offset=.4,velocity=80)
        outside=dict(pitch=20,onset=0,offset=.8)
        r=apply_release([n],np.array([1]),1.,0.,{'bass':[outside]})
        self.assertEqual(r['guitar'][0]['offset'],.4);self.assertEqual(r['bass'],[outside])

    def test_unreliable_attack_and_censored_end_do_not_teach(self):
        m=ReleaseCorrector();pred=m(torch.ones(3,350));t=dict(y=torch.ones(3),timing_mask=torch.tensor([[1.,1.],[1.,0.],[1.,1.]]),delta=torch.tensor([[0.,.1],[0.,1.],[1.,1.]]))
        loss=release_loss(pred,t);loss.backward();self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.parameters()),0)
        changed={k:v.clone() for k,v in t.items()};changed['delta'][1,1]=-100;changed['delta'][2,1]=100
        self.assertTrue(torch.equal(loss,release_loss(pred,changed)))
