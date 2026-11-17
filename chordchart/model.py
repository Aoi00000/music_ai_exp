"""Compatible CRNN port of Jiang et al. ISMIR2019 / lv-chordia.
CNN architecture adapted under MIT; see THIRD_PARTY.md and vendor/LICENSE.
Keeps the checkpoint's state keys. Adds 12 power-triad rows to the first head.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .core import VENDOR, SIZES

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        channels=[('1a',1,16),('1b',16,16),('1c',16,16),('2a',16,32),('2b',32,32),
                  ('2c',32,32),('3a',32,64),('3b',64,64),('4a',64,80),('4b',80,80)]
        for name,a,b in channels:
            setattr(self,'conv'+name,nn.Conv2d(a,b,(3,3),padding=(1,0) if name.startswith('4') else (1,1)))
            setattr(self,'norm'+name,nn.InstanceNorm2d(b))
        self.pool1=nn.MaxPool2d((1,3));self.pool2=nn.MaxPool2d((1,3));self.pool3=nn.MaxPool2d((1,4))
    def forward(self,x):
        b,t,_=x.shape;x=x.reshape(b,1,t,252)
        for group,names in [('1',['1a','1b','1c']),('2',['2a','2b','2c']),('3',['3a','3b']),('4',['4a','4b'])]:
            for k in names:x=F.selu(getattr(self,'norm'+k)(getattr(self,'conv'+k)(x)))
            if group!='4':x=getattr(self,'pool'+group)(x)
        return x.transpose(1,2).contiguous().reshape(b,t,240)

class CRNN(nn.Module):
    def __init__(self, power=True):
        super().__init__();self.power=power
        self.audio_feature_block=CNNFeatureExtractor()
        self.condition_linear=nn.Linear(240+12+6+12,128)  # unused but required by original state
        self.lstm1=nn.LSTM(240,96,num_layers=1,bidirectional=True,batch_first=True)
        self.final_fc1=nn.Linear(192,112 if power else 100)
    def forward(self,x):
        x=self.audio_feature_block(x);x=self.lstm1(x)[0];x=self.final_fc1(x)
        return torch.split(x,SIZES if self.power else (73,13,4,4,3,3),dim=-1)
    def freeze(self, mode):
        for p in self.parameters():p.requires_grad=False
        modules=[self.final_fc1] + ([self.lstm1] if mode=='partial' else [])
        for module in modules:
            for p in module.parameters():p.requires_grad=True
        self.eval()  # InstanceNorm has no running stats; no stochastic dropout here
        return [p for p in self.parameters() if p.requires_grad]

def original_path(fold=0):
    return VENDOR/'weights'/f'joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{fold}.best.sdict'

def load_original(fold=0, power=True, device='cpu'):
    state=torch.load(original_path(fold),map_location='cpu',weights_only=False)['net']
    m=CRNN(power=False);m.load_state_dict(state,strict=True)
    if power:
        old=m.final_fc1;new=nn.Linear(192,112)
        with torch.no_grad():
            new.weight[:73]=old.weight[:73];new.bias[:73]=old.bias[:73]
            new.weight[85:]=old.weight[73:];new.bias[85:]=old.bias[73:]
            new.weight[73:85]=old.weight[1:13];new.bias[73:85]=old.bias[1:13]-2.0
        m.final_fc1=new;m.power=True
    return m.to(device)

def load_project(path,device='cpu'):
    c=torch.load(path,map_location='cpu',weights_only=False)
    if c.get('model_type')!='CRNN7_v1':raise ValueError('Expected CRNN7_v1 project checkpoint')
    m=CRNN();m.load_state_dict(c['model'],strict=True)
    return m.to(device),c

@torch.no_grad()
def predict(m,features,device='cpu',window=256,overlap=64):
    import numpy as np
    x=features[:,18:270].astype(np.float32)
    n=len(x);stride=window-overlap
    if stride<=0:raise ValueError('overlap must be < window')
    sizes=SIZES if m.power else (73,13,4,4,3,3)
    sums=[np.zeros((n,k),np.float32) for k in sizes];cnt=np.zeros(n,np.float32)
    starts=list(range(0,max(1,n-window+1),stride));starts.append(max(0,n-window));starts=sorted(set(starts))
    m.eval()
    for a in starts:
        k=min(window,n-a);chunk=x[a:a+k]
        if k<window:chunk=np.pad(chunk,((0,window-k),(0,0)))
        outputs=m(torch.from_numpy(chunk[None]).to(device))
        w=np.maximum(np.hanning(window),0.05)[:k].astype(np.float32)
        for i,logits in enumerate(outputs):sums[i][a:a+k]+=logits.softmax(-1)[0,:k].cpu().numpy()*w[:,None]
        cnt[a:a+k]+=w
    if (cnt==0).any():raise RuntimeError('Uncovered inference frames')
    return [v/cnt[:,None] for v in sums]
