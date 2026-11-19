import importlib.util,unittest
import torch
from torch import nn
from band2tab.mu_adapter import inject_mu,encode_window,roundtrip

class MuAdaptation(unittest.TestCase):
    def test_zero_adapter_and_frozen_original_gradients(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__();self.self_attn=nn.Module();self.self_attn.out_proj=nn.Linear(4,4,bias=False)
        model=nn.Module();model.transformer=nn.Module();model.transformer.layers=nn.ModuleList([Layer() for _ in range(24)])
        original=list(model.parameters());x=torch.randn(2,4);base=model.transformer.layers[23].self_attn.out_proj(x).detach().clone()
        adapters=inject_mu(model);model.eval();y=model.transformer.layers[23].self_attn.out_proj(x)
        self.assertTrue(torch.equal(base,y));y.square().sum().backward()
        self.assertTrue(all(p.grad is None for p in original));self.assertGreater(float(adapters['transformer.layers.23.self_attn.out_proj'].b.grad.abs().sum()),0.)

    @unittest.skipUnless(importlib.util.find_spec('muscriptor'),'Run codec check in Mu environment')
    def test_installed_mu_codec_roundtrip_with_simultaneous_and_boundary_notes(self):
        from muscriptor.tokenizer.mt3 import MT3Tokenizer
        tok=MT3Tokenizer('MT3_FULL_PLUS');notes=[dict(pitch=60,onset=.3,offset=1.1),dict(pitch=64,onset=.3,offset=.8),dict(pitch=67,onset=-1.,offset=.4),dict(pitch=72,onset=4.8,offset=6.)]
        tokens,refs=encode_window(tok,notes,0.,5.,26);q=roundtrip(tok,tokens,refs)
        self.assertEqual(q['contained'],2);self.assertEqual(q['matched'],2);self.assertLessEqual(q['max_error_seconds'],.0051)
        self.assertEqual(tokens[-1],tok.eos_id)

if __name__=='__main__':unittest.main()
