import unittest
import numpy as np
from band2tab.octave import octave_context,prune_octaves


class OctaveTests(unittest.TestCase):
    def test_genuine_octaves_are_not_removed_just_for_existing(self):
        pool=[dict(part='guitar',pitch=p,onset=0.,offset=1.,velocity=80) for p in (48,60)]
        x,use=octave_context(np.zeros((2,350),np.float32),pool);self.assertEqual(x.shape,(2,704));self.assertTrue(use.all())
        out=prune_octaves(pool,np.ones(2),use,.2);self.assertEqual(len(out['guitar']),2)
    def test_pruning_never_retimes_or_changes_bass(self):
        pool=[dict(part=p,pitch=k,onset=.125,offset=1.25,velocity=80) for p,k in [('guitar',48),('guitar',60),('bass',36)]]
        _,use=octave_context(np.zeros((3,350),np.float32),pool);out=prune_octaves(pool,np.array([.9,.01,0]),use,.2)
        self.assertEqual([n['pitch'] for n in out['guitar']],[48]);self.assertEqual(out['bass'],[{k:v for k,v in pool[2].items() if k!='part'}]);self.assertEqual(out['guitar'][0]['onset'],.125)


if __name__=='__main__':unittest.main()
