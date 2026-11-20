import copy,unittest
import numpy as np,torch
from band2tab.octave_guard import peers,patches,prune,OctaveGuard,probabilities,SCHEMA,keep_labels

class UpperOctaveGuard(unittest.TestCase):
    def test_only_upper_same_source_and_boundary_eligible(self):
        n=lambda p,t,part:dict(pitch=p,onset=t,offset=t+.3,part=part,velocity=100)
        pool=[n(48,.5,'guitar'),n(60,.51,'guitar'),n(72,.1,'guitar'),n(60,.5,'bass')]
        self.assertEqual(peers(pool,2.),[(1,0)])

    def test_harmonic_patch_orientation_and_finite_silence(self):
        pool=[dict(part='guitar',pitch=48,onset=.5,offset=1.,velocity=100),dict(part='guitar',pitch=60,onset=.5,offset=1.,velocity=100)]
        cqt=np.zeros((200,264),np.float32);base=3*(48-23);cqt[:,base-1:base+2]=1.;cqt[:,base+35:base+38]=3.
        x,m=patches(cqt,pool,[(1,0)]);self.assertEqual(x.shape,(1,24,12));self.assertTrue((x[0,:,1]>x[0,:,0]).all())
        z,_=patches(np.zeros_like(cqt),pool,[(1,0)]);self.assertTrue(np.isfinite(z).all());self.assertEqual(float(z.sum()),0.)
        model=OctaveGuard().eval();state=dict(schema=SCHEMA,members=[model.state_dict()]);q=probabilities(state,x,m);self.assertAlmostEqual(float(q[0]),float(torch.sigmoid(torch.tensor(2.))),places=6)

    def test_teacher_protects_one_true_frame_even_without_onset_match(self):
        pool=[dict(part='guitar',pitch=48,onset=.5,offset=1.,velocity=100),dict(part='guitar',pitch=60,onset=.5,offset=1.,velocity=100)]
        truth={'guitar':[dict(pitch=60,onset=.701,offset=.719)],'bass':[]}
        y,mask=keep_labels(pool,[(1,0)],truth,2.)
        self.assertEqual(y.tolist(),[1.]);self.assertEqual(mask.tolist(),[1.])
        truth['guitar'][0]['offset']=.709
        y,_=keep_labels(pool,[(1,0)],truth,2.);self.assertEqual(y.tolist(),[0.])

    def test_pruning_cap_keeps_lower_notes_and_bass_unchanged(self):
        pool=[]
        for i in range(10):
            pool += [dict(part='guitar',pitch=48,onset=.5+i*.1,offset=.6+i*.1,velocity=100),dict(part='guitar',pitch=60,onset=.5+i*.1,offset=.6+i*.1,velocity=100)]
        pool.append(dict(part='bass',pitch=36,onset=.5,offset=2.,velocity=100));baseline={p:[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in pool if n['part']==p] for p in ('guitar','bass')};before=copy.deepcopy(baseline)
        pairs=[(2*i+1,2*i) for i in range(10)];out=prune(baseline,pool,pairs,np.zeros(10),.5)
        self.assertEqual(baseline,before);self.assertEqual(len(out['guitar']),18);self.assertEqual(sum(n['pitch']==48 for n in out['guitar']),10);self.assertEqual(out['bass'],baseline['bass'])
        with self.assertRaises(ValueError):prune(baseline,pool,pairs,np.full(10,np.nan),.5)

if __name__=='__main__':unittest.main()
