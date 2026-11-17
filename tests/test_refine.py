import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from band2tab.refine import EventRefiner,event_features,proposed_parts,refine,SCHEMA,candidates_from_midi

class Refinement(unittest.TestCase):
    def note(self,**kwargs):
        return dict(pitch=60,onset=.1,offset=.4,velocity=100,program=27,baseline_class=1)|kwargs
    def test_frozen_baseline_prior_and_gradient(self):
        notes=[self.note(),self.note(pitch=40,baseline_class=2,program=33)]
        x=event_features(np.ones((50,264),np.float32),notes)
        self.assertEqual(x.shape,(2,82));self.assertTrue(np.isfinite(x).all())
        model=EventRefiner();z=model(torch.tensor(x),torch.tensor([1,2]))
        self.assertEqual(z.argmax(-1).tolist(),[1,2])
        torch.nn.functional.cross_entropy(z,torch.tensor([0,1])).backward()
        self.assertGreater(float(model.net[-1].weight.grad.abs().sum()),0)
    def test_real_midi_candidate_and_changes_are_json_serializable(self):
        import pretty_midi
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'notes.mid';midi=pretty_midi.PrettyMIDI();ins=pretty_midi.Instrument(0)
            ins.notes.append(pretty_midi.Note(100,60,.1,.4));midi.instruments.append(ins);midi.write(str(path))
            notes=candidates_from_midi(path)
            parts,changes=proposed_parts(notes,np.array([[.01,.98,.01]]),.9,.4)
            decoded=json.loads(json.dumps({'parts':parts,'changes':changes}))
            self.assertEqual(decoded['changes'][0]['to'],'guitar')
    def test_gated_reassignment_and_removal_preserve_pitch_and_time(self):
        notes=[self.note(),self.note(pitch=40),self.note(pitch=69,baseline_class=0)]
        snapshot=copy.deepcopy(notes)
        p=np.array([[.98,.01,.01],[.01,.01,.98],[.01,.98,.01]])
        parts,changes=proposed_parts(notes,p,.9,.4)
        self.assertEqual([(c['from'],c['to']) for c in changes],[('guitar','other'),('guitar','bass'),('other','guitar')])
        self.assertEqual(parts['bass'][0]['pitch'],40);self.assertEqual(parts['guitar'][0]['pitch'],69)
        self.assertEqual(parts['bass'][0]['onset'],.1);self.assertEqual(notes,snapshot)
        unchanged,changes=proposed_parts(notes,p,1.,1.)
        self.assertFalse(changes);self.assertEqual(len(unchanged['guitar']),2)
        _,changes=proposed_parts(notes,np.eye(3)[[0,2,1]],1.,1.)
        self.assertFalse(changes)  # Exact float32 saturation still means identity.
    def test_review_preserves_out_of_vocabulary_and_short_notes(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);model=EventRefiner()
            state={'schema':SCHEMA,'model':model.state_dict(),'mean':[0.]*82,'std':[1.]*82,
                   'policy':{'threshold':.6,'margin':.2},'automatic_application_approved':False}
            checkpoint=d/'refiner.pt';torch.save(state,checkpoint)
            original={'guitar':[{'pitch':100,'onset':0.,'offset':.01,'confidence':None}],
                      'bass':[{'pitch':40,'onset':.1,'offset':.4,'confidence':None}]}
            candidate=self.note(pitch=40,program=33,baseline_class=2)
            with patch('band2tab.score.read_midi',return_value=original),patch('band2tab.refine.candidates_from_midi',return_value=[candidate]),patch('band2tab.refine.features',return_value=np.ones((50,264),np.float32)):
                result=refine('audio.wav','base.mid',checkpoint)
                self.assertEqual(result['parts'],original)
                self.assertEqual(result['suggested_parts']['guitar'][0]['pitch'],100)
                with self.assertRaisesRegex(ValueError,'not validated'):refine('audio.wav','base.mid',checkpoint,mode='apply')
            with patch('band2tab.score.read_midi',return_value=original):
                self.assertEqual(refine(None,'base.mid')['parts'],original)

if __name__=='__main__':unittest.main()
