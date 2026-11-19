import unittest
import numpy as np
from band2tab.addition import proposals,targets,add_notes


def n(p=48,a=.2):return dict(pitch=p,onset=a,offset=a+.3,velocity=80,part='guitar')


class AdditionTests(unittest.TestCase):
    def test_duplicate_filter_keeps_new_reattacks(self):
        p=proposals([n(a=.21),n(a=.5),n(a=.51),n(p=52)], [n()])
        self.assertEqual([(v['pitch'],v['onset']) for v in p],[(48,.5),(52,.2)])

    def test_already_covered_reference_is_not_positive(self):
        y=targets([n(a=.23),n(a=.5),n(a=.51)],[n()],[n(),n(a=.5)],0,1.)
        np.testing.assert_array_equal(y['y'],[0,1,0])

    def test_additions_preserve_original_bass_and_guitar(self):
        base={'guitar':[n()],'bass':[n(p=36)]};out=add_notes(base,[n(p=52)],[.9],.8)
        self.assertEqual(out['bass'],base['bass']);self.assertIn(base['guitar'][0],out['guitar']);self.assertEqual(len(base['guitar']),1)

    def test_crop_guard_matches_note_metric_coverage(self):
        y=targets([n(a=.23),n(a=.27)],[n(a=.23)],[n(a=.26)],.25,1.)
        np.testing.assert_array_equal(y['y'],[0,1]);np.testing.assert_array_equal(y['mask'],[0,1])
