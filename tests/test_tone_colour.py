import unittest,random
import numpy as np
from band2tab.augment import tone_colour


class ToneColourTests(unittest.TestCase):
    def test_impulse_no_delay_and_sample_clock_unchanged(self):
        x=np.zeros(32767,np.float32);x[16000]=.8
        for seed in range(12):
            y,meta=tone_colour(x,16000,seed,False)
            self.assertEqual(len(y),len(x));self.assertEqual(int(np.argmax(np.abs(y))),16000)
            self.assertTrue(np.isfinite(y).all())

    def test_deterministic_and_does_not_consume_sampler_rng(self):
        x=np.sin(np.arange(32767)*.1).astype(np.float32);random.seed(17);before=random.getstate();np.random.seed(17);nbefore=np.random.get_state()
        a,ma=tone_colour(x,16000,3,True);b,mb=tone_colour(x,16000,3,True)
        np.testing.assert_array_equal(a,b);self.assertEqual(ma,mb);self.assertEqual(before,random.getstate());np.testing.assert_array_equal(nbefore[1],np.random.get_state()[1])

    def test_silence_remains_silent(self):
        y,m=tone_colour(np.zeros(1000),16000,3,True);self.assertFalse(y.any());self.assertTrue(m['identity'])
