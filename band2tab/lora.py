"""Small supervised adapters on frozen transcription attention projections."""
from contextlib import contextmanager
import math,hashlib
import torch
from torch import nn

SCHEMA='yourmt3_attention_lora_v1'

class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16.,dropout=.05):
        super().__init__()
        if rank<1:raise ValueError('rank must be positive')
        self.base=base;self.rank=rank;self.alpha=float(alpha);self.enabled=True
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5));self.dropout=nn.Dropout(dropout)
        for p in self.base.parameters():p.requires_grad_(False)
    @property
    def weight(self):return self.base.weight
    @property
    def bias(self):return self.base.bias
    @property
    def in_features(self):return self.base.in_features
    @property
    def out_features(self):return self.base.out_features
    def forward(self,x):
        value=self.base(x)
        if self.enabled:
            update=torch.nn.functional.linear(torch.nn.functional.linear(self.dropout(x),self.a),self.b)
            value=value+update*(self.alpha/self.rank)
        return value

def inject(model,scope='decoder',rank=8,alpha=16.,dropout=.05,targets='attention'):
    if scope not in ('decoder','both'):raise ValueError('Unknown attention scope')
    if targets not in ('attention','wide'):raise ValueError('Unknown adapter targets')
    for p in model.parameters():p.requires_grad_(False)
    prefixes=('decoder.',) if scope=='decoder' else ('decoder.','encoder.')
    candidates=[(name,module) for name,module in model.named_modules()
        if isinstance(module,nn.Linear) and name.startswith(prefixes) and
        ((targets=='attention' and name.rsplit('.',1)[-1] in ('q','v','query','value')) or
         (targets=='wide' and module.out_features>=32))]
    if not candidates:raise ValueError('No compatible attention projections found')
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name,_ in candidates):
            raise ValueError('Missing requested attention scope: '+prefix)
    result={}
    for name,module in candidates:
        parent,leaf=name.rsplit('.',1);adapter=LoRALinear(module,rank,alpha,dropout)
        setattr(model.get_submodule(parent),leaf,adapter);result[name]=adapter
    return result

@contextmanager
def disabled(adapters):
    previous={name:m.enabled for name,m in adapters.items()}
    try:
        for m in adapters.values():m.enabled=False
        yield
    finally:
        for name,m in adapters.items():m.enabled=previous[name]

def adapter_state(adapters):
    return {name:{'a':m.a.detach().cpu().clone(),'b':m.b.detach().cpu().clone(),
                  'rank':m.rank,'alpha':m.alpha,'dropout':m.dropout.p} for name,m in adapters.items()}

def restore(model,state):
    if state['schema']!=SCHEMA:raise ValueError('Unknown adapter schema')
    config=state['config'];adapters=inject(model,config['scope'],config['rank'],config['alpha'],config['dropout'],config.get('targets','attention'))
    if set(adapters)!=set(state['adapters']):raise ValueError('Adapter names differ from frozen base')
    with torch.no_grad():
        for name,m in adapters.items():
            m.a.copy_(state['adapters'][name]['a']);m.b.copy_(state['adapters'][name]['b'])
    return adapters

def frozen_parameter_digest(parameters):
    h=hashlib.sha256()
    for p in parameters:
        h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
