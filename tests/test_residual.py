import copy, unittest
from unittest.mock import patch
import numpy as np
from band2tab.residual import candidate_pool,teacher,apply,residual_features
from band2tab.refine_duration import instrument_metrics

def note(p,a,b):return dict(pitch=p,onset=a,offset=b,velocity=100)

class ResidualLearning(unittest.TestCase):
    def test_octave_and_support_candidates_use_no_labels_and_preserve_time(self):
        base={'guitar':[note(60,.5,1.)],'bass':[note(40,.6,1.2)]};original=copy.deepcopy(base)
        pool=candidate_pool(base,{'guitar':[note(72,.51,.9)],'bass':[]},2.)
        self.assertEqual(base,original);self.assertEqual([n['part'] for n in pool],['guitar']*2+['bass']*2)
        matched=next(n for n in pool if n['part']=='guitar' and n['pitch']==72)
        self.assertEqual(matched['onset'],.51);self.assertEqual(matched['offset'],.9);self.assertEqual(matched['origin'][:3],[1.,0.,1.])
        out=apply(base,pool,np.ones(len(pool)),(.5,.5));self.assertEqual(base,original)
        for part in base:
            for n in base[part]:self.assertIn(n,out[part])
        with self.assertRaises(ValueError):apply(base,pool,np.full(len(pool),np.nan))

    def test_missing_note_targets_are_unique_and_frame_gain_matches_real_scoring(self):
        base={'guitar':[note(60,.5,1.)],'bass':[]};truth={'guitar':[note(60,.5,1.),note(72,.5,1.)],'bass':[]}
        pool=candidate_pool(base,{},2.);targets=teacher(pool,base,truth,2.,.25)
        before=instrument_metrics(truth,base,start=.25,end=1.75)['frame_f1']
        for i,n in enumerate(pool):
            added=copy.deepcopy(base);added[n['part']].append(n)
            after=instrument_metrics(truth,added,start=.25,end=1.75)['frame_f1']
            self.assertAlmostEqual(after-before,float(targets['frame_gain'][i]),places=6)
            self.assertEqual(targets['y'][i],float(n['pitch']==72))

    def test_feature_rows_stay_aligned_with_role_order(self):
        base={'guitar':[note(60,.5,1.)],'bass':[note(40,.5,1.)]};pool=candidate_pool(base,{},2.)
        def fake(v,e,ns,b,d):return np.full((len(ns),358),1 if ns[0]['part']=='guitar' else 2,np.float32)
        with patch('band2tab.residual.addition_features',side_effect=fake):x=residual_features({}, {},pool,base,2.)
        self.assertEqual(x.shape,(len(pool),366))
        for i,n in enumerate(pool):self.assertEqual(x[i,0],1 if n['part']=='guitar' else 2)

if __name__=='__main__':unittest.main()
