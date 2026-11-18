import copy,unittest
import torch
from torch import nn
from band2tab.lora import LoRALinear,inject,disabled,adapter_state,restore,SCHEMA

class LoRATests(unittest.TestCase):
    def test_zero_adapter_matches_base_and_training_keeps_base_fixed(self):
        torch.manual_seed(17);base=nn.Linear(7,5);original=copy.deepcopy(base.state_dict());x=torch.randn(4,7)
        m=LoRALinear(base,rank=2,alpha=4,dropout=0).eval()
        self.assertTrue(torch.equal(m(x),base(x)))
        opt=torch.optim.AdamW([m.a,m.b],lr=.1);m(x).square().mean().backward();opt.step()
        for k,v in original.items():self.assertTrue(torch.equal(base.state_dict()[k],v))
        self.assertFalse(torch.equal(m(x),base(x)))
        with disabled({'q':m}):self.assertTrue(torch.equal(m(x),base(x)))
        self.assertTrue(m.enabled)

    def test_scope_and_portable_adapter_restore(self):
        model=nn.Module();model.encoder=nn.ModuleDict({'q':nn.Linear(5,5)});model.decoder=nn.ModuleDict({'v':nn.Linear(5,5)})
        untouched=copy.deepcopy(model);adapters=inject(model,'decoder',rank=2,alpha=4,dropout=0)
        self.assertEqual(set(adapters),{'decoder.v'})
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        with torch.no_grad():adapters['decoder.v'].b.fill_(.1)
        state={'schema':SCHEMA,'config':{'scope':'decoder','rank':2,'alpha':4,'dropout':0},'adapters':adapter_state(adapters)}
        restore(untouched,state);model.eval();untouched.eval();x=torch.randn(3,5)
        torch.testing.assert_close(model.decoder['v'](x),untouched.decoder['v'](x))

    def test_requested_encoder_scope_cannot_silently_be_ignored(self):
        model=nn.Module();model.decoder=nn.ModuleDict({'v':nn.Linear(5,5)})
        with self.assertRaisesRegex(ValueError,'encoder'):
            inject(model,'both')

    def test_wider_adapters_include_features_but_preserve_small_expert_router(self):
        model=nn.Module();model.encoder=nn.ModuleDict({'dense':nn.Linear(64,128),'router':nn.Linear(64,8)})
        model.decoder=nn.ModuleDict({'o':nn.Linear(64,64)})
        adapters=inject(model,'both',rank=2,targets='wide')
        self.assertEqual(set(adapters),{'encoder.dense','decoder.o'})
        self.assertIsInstance(model.encoder['router'],nn.Linear)
        self.assertFalse(model.encoder['router'].weight.requires_grad)

if __name__=='__main__':unittest.main()
