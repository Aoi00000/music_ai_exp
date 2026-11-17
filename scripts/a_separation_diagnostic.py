"""Fixed HTDemucs-6s -> A8 diagnostic; no threshold/model fitting."""
from pathlib import Path
import argparse,json,os
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

p=argparse.ArgumentParser();p.add_argument('command',choices=['separate','score']);a=p.parse_args()
rows=json.loads(Path('data/manifest_mix_diagnostic.json').read_text())
folder=Path('data/mix_separated');folder.mkdir(parents=True,exist_ok=True)
if a.command=='separate':
    os.environ.setdefault('TORCH_HOME',str(Path('models/torch').resolve()))
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    model=get_model('htdemucs_6s').to('cuda').eval()
    for audio in sorted({it['audio'] for it in rows}):
        name=Path(audio).stem
        if all((folder/(name+'_'+part+'.wav')).exists() for part in ('guitar','bass')):continue
        y,sr=sf.read(audio,dtype='float32',always_2d=True)
        if y.shape[1]==1:y=np.repeat(y,2,axis=1)
        from math import gcd
        d=gcd(sr,model.samplerate);y=resample_poly(y,model.samplerate//d,sr//d,axis=0)
        wav=torch.from_numpy(y.T.copy());ref=wav.mean(0);mean=ref.mean();std=ref.std().clamp_min(1e-8)
        with torch.no_grad():
            sources=apply_model(model,((wav-mean)/std)[None],device='cuda',shifts=1,split=True,overlap=.25,segment=7)[0].cpu()*std+mean
        for part in ('guitar','bass'):
            sf.write(folder/(name+'_'+part+'.wav'),sources[model.sources.index(part)].numpy().T,model.samplerate,subtype='PCM_16')
        print('SEPARATED',name,model.sources,flush=True)
else:
    from band2tab.model import load,predict
    from band2tab.audio import features,decode
    from band2tab.learn import note_metrics,pitch_set_metrics
    from collections import defaultdict
    model,state=load('models/acoustic_review.pt','cuda');cfg=state['decoding'];groups=defaultdict(list)
    out=Path('reports/separation_predictions');out.mkdir(exist_ok=True)
    for it in rows:
        audio=folder/(Path(it['audio']).stem+'_'+it['part']+'.wav')
        prob=predict(model,features(audio),int(it['part']=='bass'),'cuda')
        notes=decode(prob,onset=cfg['onset'],frame=cfg['frame'],offset=cfg['offset'])
        (out/(it['id']+'.json')).write_text(json.dumps({'notes':notes,'audio':str(audio),'decoding':cfg}))
        groups[it['group']].append((it,notes))
    results=[]
    for group,pairs in groups.items():
        results.append({'source_group':group,'parts':{it['part']:{'reference_notes':len(it['notes']),'predicted_notes':len(pred),'metrics':note_metrics(it['notes'],pred)} for it,pred in pairs},'guitar_bass_union':pitch_set_metrics([n for it,_ in pairs for n in it['notes']],[n for _,pred in pairs for n in pred])})
    obj={'scope':'same 3 historical synthetic 30-second mix excerpts; fixed HTDemucs6s and validation-selected A8, no tuning; not real-band generalization','tracks':results,'mean_union_frame_f1':float(np.mean([r['guitar_bass_union']['frame_f1'] for r in results])),'mean_union_active_exact_match':float(np.mean([r['guitar_bass_union']['active_exact_match'] for r in results]))}
    Path('reports/separation_diagnostic.json').write_text(json.dumps(obj,indent=2));print(obj['mean_union_frame_f1'],obj['mean_union_active_exact_match'],flush=True)
