"""Flag acoustic inconsistency without deleting baseline notes."""
from pathlib import Path
import json,numpy as np
from .audio import features,SR,HOP,PITCH_MIN,PITCH_MAX
from .model import load,predict

def verify(audio,checkpoint,notes_json,part,threshold=.3,device='cpu'):
    model,_=load(checkpoint,device);p=predict(model,features(audio),int(part=='bass'),device)
    j=json.loads(Path(notes_json).read_text())
    for n in j['parts'].get(part,[]):
        pi=int(n['pitch'])-PITCH_MIN
        if not 0<=pi<=PITCH_MAX-PITCH_MIN:
            n['acoustic_score']=None;n['needs_review']=True;continue
        a=max(0,int(n['onset']*SR/HOP));b=min(len(p),max(a+1,int(n['offset']*SR/HOP)))
        if a>=len(p):n['acoustic_score']=None;n['needs_review']=True;continue
        onset=float(p[max(0,a-4):min(len(p),a+5),0,pi].max())
        frame=float(p[a:b,1,pi].mean());score=float(np.sqrt(onset*frame))
        n['acoustic_score']=score;n['needs_review']=score<threshold
    j['review_note']='Acoustic scores are uncalibrated; they do not prove correctness or detect every omitted note.'
    Path(notes_json).write_text(json.dumps(j,ensure_ascii=False,indent=2))
    return j
