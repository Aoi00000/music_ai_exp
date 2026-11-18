import unittest
import numpy as np,torch
from band2tab.candidate import candidate_pool,candidate_features,targets,unique_targets,propose,CandidateRanker,loss

def note(p=48,a=.2,b=.8,c=0):
    return {'pitch':p,'onset':a,'offset':b,'velocity':80,'program':0,'baseline_class':c}

class CandidateTests(unittest.TestCase):
    def test_unique_supervision_does_not_reward_duplicate_routes_or_lose_unison(self):
        pool=candidate_pool({'mix':[note(a=.2,c=1)],'guitar':[note(a=.25,b=.85)],'bass':[]},1.)
        truth={'guitar':[note(a=.2)],'bass':[note(a=.2)]}
        ordinary=targets(pool,truth);unique=unique_targets(pool,truth)
        np.testing.assert_array_equal(ordinary['y'].sum(0),[2,2])
        np.testing.assert_array_equal(unique['y'].sum(0),[1,1])
        self.assertEqual(unique['y'][0,0],1.)
    def test_other_program_is_retained_and_sources_merge_without_retiming(self):
        pool=candidate_pool({'mix':[note()], 'guitar':[note(a=.21,b=.81,c=1)],'bass':[]},1.)
        self.assertEqual(len(pool),1);self.assertEqual(pool[0]['support'],[1.,1.,0.])
        self.assertEqual(pool[0]['onset'],.2);self.assertEqual(pool[0]['baseline'],[0.,0.])

    def test_unison_roles_and_real_reattacks_survive_duplicate_suppression(self):
        pool=candidate_pool({'mix':[note(),note(a=.22,b=.79),note(a=.4,b=.9)],'guitar':[],'bass':[]},1.)
        pred={'probability':np.full((len(pool),2),.9),'quality':np.ones((len(pool),2)), 'delta':np.zeros((len(pool),2,2))}
        result=propose(pool,pred,1.,nms=.05)
        self.assertEqual([len(result[p]) for p in ('guitar','bass')],[2,2])
        self.assertEqual([n['onset'] for n in result['guitar']],[.2,.4])

    def test_censored_attack_is_not_false_negative_and_unison_has_two_targets(self):
        pool=candidate_pool({'mix':[note(a=0,b=.5)],'guitar':[],'bass':[]},1.)
        y=targets(pool,{'guitar':[{'pitch':48,'onset':0.,'offset':.5,'censored_onset':True}],
                        'bass':[{'pitch':48,'onset':.05,'offset':.5,'censored_offset':True}]})
        self.assertEqual(y['mask'][0,0],0);self.assertEqual(y['y'][0,1],1)
        self.assertEqual(y['timing_mask'][0,1,1],0)
        y=targets(pool,{'guitar':[note(a=.01,b=.5)],'bass':[note(a=.01,b=.5)]})
        np.testing.assert_array_equal(y['y'],[[1,1]])

    def test_feature_dimensions_and_empty_pool(self):
        views=[np.ones((100,264),np.float32) for _ in range(3)];e=np.ones((100,3,128),np.float16)
        pool=candidate_pool({'mix':[note()], 'guitar':[],'bass':[]},1.)
        x,p=candidate_features(views,e,pool,1.);self.assertEqual(x.shape,(1,1030));self.assertEqual(p.shape,(1,2))
        self.assertEqual(candidate_features(views,e,[],1.)[0].shape,(0,1030))

    def test_padding_labels_do_not_change_loss(self):
        model=CandidateRanker().eval();x=torch.zeros(1,3,1030);prior=torch.zeros(1,3,2)
        output=model(x,prior);t={k:torch.zeros(1,3,2) for k in ('y','mask','quality','keep_correct')}
        t.update(delta=torch.zeros(1,3,2,2),timing_mask=torch.zeros(1,3,2,2));t['mask'][:,:2]=1
        before=loss(output,t);t['y'][:,2]=1;t['quality'][:,2]=1
        self.assertEqual(float(before),float(loss(output,t)))

    def test_timing_adjustment_is_bounded_and_keeps_valid_intervals(self):
        pool=candidate_pool({'mix':[note(a=.02,b=.5)],'guitar':[],'bass':[]},.6)
        pred={'probability':np.ones((1,2)),'quality':np.ones((1,2)),'delta':np.array([[[-1.,1.],[-1.,1.]]])}
        result=propose(pool,pred,.6,retime=True)
        self.assertEqual(result['guitar'][0]['onset'],0.);self.assertEqual(result['guitar'][0]['offset'],.6)

if __name__=='__main__':unittest.main()
