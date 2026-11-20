import unittest
import numpy as np
from band2tab.octave_six import summarize, probabilities, SCHEMA


class SixHarmonics(unittest.TestCase):
    def test_excluded_pitch_and_out_of_band_partials_cannot_change_features(self):
        rng = np.random.default_rng(15)
        x = rng.random((4, 24, 12)).astype(np.float32)
        meta = rng.random((4, 4)).astype(np.float32)
        expected = summarize(x, meta)
        x[:, :, 6:] = 1e6
        meta[:, 0] = -12
        np.testing.assert_array_equal(expected, summarize(x, meta))
        self.assertEqual(expected.shape, (4, 35))

    def test_silence_empty_and_exported_logistic(self):
        x = np.zeros((2, 24, 12), np.float32)
        meta = np.zeros((2, 4), np.float32)
        self.assertTrue(np.isfinite(summarize(x, meta)).all())
        self.assertEqual(summarize(x[:0], meta[:0]).shape, (0, 35))
        state = dict(schema=SCHEMA, mean=[0.] * 35, std=[1.] * 35,
                     coef=[0.] * 35, intercept=0.)
        np.testing.assert_array_equal(probabilities(state, x, meta), [.5, .5])


if __name__ == '__main__':
    unittest.main()
