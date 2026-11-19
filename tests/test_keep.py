import copy,unittest
import numpy as np
from band2tab.keep import keep_targets,prune
from band2tab.refine_duration import instrument_metrics

class KeepProtection(unittest.TestCase):
    def test_frame_teacher_agrees_with_actual_deletion(self):
        def note(p,a,b):return dict(pitch=p,onset=a,offset=b,velocity=100,part='guitar')
        pool=[note(60,.2,1),note(60,.2,.6),note(64,.3,1.2),note(67,1,1.5)]
        truth={'guitar':[note(60,.2,1),note(64,.2,1.2)],'bass':[]}
        baseline={'guitar':pool,'bass':[]};target=keep_targets(pool,truth,baseline,0.,2.)
        before=instrument_metrics(truth,baseline,end=2.)['frame_f1']
        for i in range(len(pool)):
            after=instrument_metrics(truth,{'guitar':[n for j,n in enumerate(pool) if j!=i],'bass':[]},end=2.)['frame_f1']
            self.assertAlmostEqual(after-before,float(target['delete_frame_gain'][i]),places=6)
        self.assertEqual(target['onset'].tolist(),[1,0,0,0])
        # The late but useful E4 is protected; duplicate and extra G4 need not be.
        self.assertEqual(target['joint'].tolist(),[1,0,1,0])

    def test_pruning_cannot_retime_or_mutate_original(self):
        pool=[dict(part='guitar',pitch=60,onset=.1,offset=.8,velocity=100),
              dict(part='bass',pitch=40,onset=.5,offset=1.,velocity=100)]
        original=copy.deepcopy(pool)
        out=prune(pool,np.array([0.,0.]),.5,{'guitar':[],'bass':[]},2.,guard=.25)
        self.assertEqual(pool,original)
        self.assertEqual(out['guitar'],[{k:pool[0][k] for k in ('pitch','onset','offset','velocity')}])
        self.assertEqual(out['bass'],[])
        with self.assertRaises(ValueError):prune(pool,np.array([np.nan,0]),.5,{},2.)

    def test_censored_start_is_not_a_false_note_teacher(self):
        pool=[dict(part='bass',pitch=40,onset=.5,offset=1.,velocity=100)]
        truth={'guitar':[],'bass':[dict(pitch=40,onset=0.,offset=1.2,censored_onset=True)]}
        target=keep_targets(pool,truth,{'guitar':[],'bass':pool},.25,2.)
        self.assertEqual(target['mask'].tolist(),[0.])

    def test_role_policy_cap_and_corroboration_guard(self):
        import torch
        from scripts.a_keep_policy_cycle import apply
        from band2tab.keep_agreement import agreement_features,AgreementKeepNet,probabilities,SCHEMA
        pool=[dict(part='guitar',pitch=60,onset=.5+i*.1,offset=.55+i*.1,velocity=100) for i in range(20)]
        out=apply(pool,np.zeros(20),(.8,.8),{'guitar':[],'bass':[]},4.)
        self.assertEqual(len(out['guitar']),18)
        support=[dict(pitch=60,onset=.51,offset=.55)]
        extra=agreement_features(pool,support);self.assertEqual(extra[0,0],1.)
        model=AgreementKeepNet()
        with torch.no_grad():model.net[-1].bias.fill_(-10.)
        state=dict(schema=SCHEMA,use_agreement=True,agreement_guard=True,members=[model.state_dict()],mean=[0.]*358,std=[1.]*358)
        q=probabilities(state,np.concatenate([np.zeros((20,350),np.float32),extra],axis=1))
        self.assertEqual(q[0],1.);self.assertLess(q[-1],.001)

    def test_hard_sounding_teacher_preserves_a_short_true_fragment(self):
        from scripts.a_sound_keep_cycle import sound_targets
        pool=[dict(part='bass',pitch=p,onset=0.,offset=2.,velocity=100) for p in (40,43)]
        truth={'guitar':[],'bass':[dict(pitch=40,onset=0.,offset=2.),dict(pitch=43,onset=.2,offset=.3)]}
        base={'guitar':[],'bass':pool};target=sound_targets(pool,truth,base,0.,2.)
        # Removing the long43 raises F1 overall but loses its genuine short sound.
        self.assertEqual(target['joint'][1],0.)
        self.assertEqual(target['sound'][1],1.)

if __name__=='__main__':unittest.main()
