import unittest
import numpy as np
from band2tab.augment import soft_drive,damp_isolated


class AugmentTests(unittest.TestCase):
    def test_drive_preserves_sample_positions_and_silence(self):
        x=np.zeros(1000,np.float32);x[300:400]=.2
        out=soft_drive(x)
        self.assertEqual(out.shape,x.shape)
        self.assertTrue(np.array_equal(out!=0,x!=0))
        self.assertTrue(np.isfinite(out).all())

    def test_damping_changes_release_label_and_handles_crop_ties(self):
        notes=[{'pitch':40,'onset':.4,'offset':.9},{'pitch':43,'onset':1.,'offset':1.5}]
        x=np.ones(1000,np.float32)
        out,labels=damp_isolated(x,1000,notes,.5,.12,.04)
        self.assertAlmostEqual(labels[0]['offset'],.52)
        self.assertAlmostEqual(labels[1]['offset'],1.12)
        self.assertGreater(out[0],0)
        self.assertTrue((out[20:500]==0).all())
        self.assertTrue((out[620:]==0).all())
        self.assertEqual(notes[0]['offset'],.9)

    def test_polyphonic_source_is_rejected_instead_of_inventing_labels(self):
        with self.assertRaisesRegex(ValueError,'monophonic'):
            damp_isolated(np.ones(1000),1000,[{'onset':0,'offset':.8},{'onset':.4,'offset':.9}],0)


if __name__=='__main__':unittest.main()
