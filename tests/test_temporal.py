import unittest
import numpy as np
import torch
import tempfile,types,sys,struct
from pathlib import Path
from unittest.mock import patch
import soundfile as sf
from band2tab.temporal import TemporalFusion,piano_roll,masked_loss,decode_parts,known_offset_score,predict_scales
from band2tab.audio import HOP,SR
from band2tab.separation import write_input,separate_inputs

class TemporalTests(unittest.TestCase):
    def test_zero_correction_strength_preserves_prior_with_nonzero_network(self):
        model=TemporalFusion().eval()
        with torch.no_grad():model.head.bias.fill_(1.)
        e=np.zeros((12,4,128),np.float16);prior=np.zeros((12,2,3,66),np.float32)
        out=predict_scales(model,e,prior,(0.,1.))
        np.testing.assert_allclose(out[0.],float(torch.sigmoid(torch.tensor(-2.))),rtol=1e-6)
        self.assertTrue(np.all(out[1.]>out[0.]))

    def test_excerpt_cut_is_not_scored_as_known_note_release(self):
        truth={'guitar':[{'pitch':60,'onset':.2,'offset':1.,'censored_offset':True},
                          {'pitch':64,'onset':.3,'offset':.5}]}
        estimate={'guitar':[{'pitch':60,'onset':.2,'offset':.4},{'pitch':64,'onset':.3,'offset':.5}]}
        score=known_offset_score(truth,estimate)
        self.assertEqual(score['known_offset_reference_notes'],1);self.assertEqual(score['offset_f1'],1.)
        estimate['guitar'].append({'pitch':70,'onset':1.2,'offset':1.5})
        self.assertLess(known_offset_score(truth,estimate)['offset_f1'],1.)

    def test_sustained_boundary_note_and_unison_are_not_fake_attacks(self):
        parts={'guitar':[{'pitch':60,'onset':0.,'offset':.5,'censored_onset':True}],
               'bass':[{'pitch':60,'onset':.1,'offset':.5}]}
        y=piano_roll(parts,100,reference=True);p=60-23
        self.assertEqual(y[:,0,0,p].sum(),0.)
        self.assertGreater(y[:,0,1,p].sum(),0.)
        self.assertEqual(y[:,1,0,p].sum(),1.)
        self.assertTrue((y[:,0,1,p]*y[:,1,1,p]).any())

    def test_hallucinated_note_after_audio_is_not_moved_to_last_frame(self):
        y=piano_roll({'guitar':[{'pitch':60,'onset':50.,'offset':51.}]},100)
        self.assertEqual(y.sum(),0.)

    def test_residual_initialization_reproduces_note_prior_for_both_parts(self):
        model=TemporalFusion().eval();e=torch.randn(2,12,4,128);prior=torch.randint(0,2,(2,12,3,66)).float()
        with torch.no_grad():out=model(e,prior,torch.tensor([0,1]))
        self.assertTrue(torch.equal(out,(prior*2-1)*2))

    def test_padding_cannot_change_training_loss(self):
        logits=torch.zeros(1,5,3,66);target=torch.zeros_like(logits);mask=torch.tensor([[1.,1.,1.,0.,0.]])
        a=masked_loss(logits,target,mask);target[:,3:]=1.;b=masked_loss(logits,target,mask)
        self.assertAlmostEqual(float(a),float(b),places=6)

    def test_temporal_output_can_generate_a_note_absent_from_prior(self):
        p=np.full((100,2,3,66),.01,np.float32);index=64-23
        p[20,0,0,index]=.99;p[20:60,0,1,index]=.99;p[59,0,2,index]=.99
        notes=decode_parts(p,{'onset':.5,'frame':.5,'offset':.5,'min_duration':.03})
        self.assertEqual(len(notes['guitar']),1);self.assertEqual(notes['guitar'][0]['pitch'],64)
        self.assertAlmostEqual(notes['guitar'][0]['onset'],20*HOP/SR)
        self.assertEqual(notes['bass'],[])

    def test_separation_keeps_canonical_input_bytes_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);path=td/'mix.wav';write_input(path,np.full(16000,.1,np.float32),16000)
            payload=bytearray(path.read_bytes());i=payload.find(b'PEAK')
            if i>=0:payload[i+12:i+16]=struct.pack('<I',0)
            path.write_bytes(payload);before=path.read_bytes()
            fake=types.ModuleType('demucs.apply');fake.apply_model=lambda model,wav,**kw:torch.zeros(1,6,2,wav.shape[-1])
            model=types.SimpleNamespace(samplerate=16000,sources=['drums','bass','other','vocals','guitar','piano'])
            with patch.dict(sys.modules,{'demucs':types.ModuleType('demucs'),'demucs.apply':fake}):
                separate_inputs(path,td,model,device='cpu')
            self.assertEqual(before,path.read_bytes())

if __name__=='__main__':unittest.main()
