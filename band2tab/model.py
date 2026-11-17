"""Our trainable CRNN; no public pretrained note weights are claimed."""
import torch
from torch import nn
from .audio import N_PITCHES

class NoteCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(1,16,(3,5),padding=(1,2)),nn.ReLU(),
            nn.MaxPool2d((1,3)),nn.Conv2d(16,32,(3,5),padding=(1,2)),nn.ReLU(),
            nn.MaxPool2d((1,3)))
        self.proj=nn.Linear(32*29,128)
        self.inst=nn.Embedding(2,8)
        self.gru=nn.GRU(136,128,num_layers=2,batch_first=True,bidirectional=True,dropout=.1)
        self.head=nn.Linear(256,3*N_PITCHES)
    def forward(self,x,part):
        z=self.cnn(x[:,None]);z=z.permute(0,2,1,3).flatten(2)
        z=torch.relu(self.proj(z));e=self.inst(part)[:,None].expand(-1,z.shape[1],-1)
        z,_=self.gru(torch.cat([z,e],dim=-1))
        return self.head(z).reshape(x.shape[0],x.shape[1],3,N_PITCHES)

def load(path,device='cpu'):
    model=NoteCRNN().to(device)
    state=torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict(state['model']);model.eval()
    return model,state

@torch.no_grad()
def predict(model,x,part=0,device='cpu',window=344,overlap=86):
    # Average probabilities in overlapping windows, then decode once globally.
    import numpy as np
    total=np.zeros((len(x),3,N_PITCHES),np.float32);count=np.zeros(len(x),np.float32)
    for start in range(0,len(x),window-overlap):
        z=torch.from_numpy(x[start:start+window]).to(device)[None]
        p=model(z,torch.tensor([part],device=device)).sigmoid()[0].cpu().numpy()
        total[start:start+len(p)]+=p;count[start:start+len(p)]+=1
    return total/np.maximum(count[:,None,None],1)
