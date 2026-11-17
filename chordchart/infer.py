import argparse,subprocess,tempfile,shutil,json
from pathlib import Path
import numpy as np
import torch
from .core import ROOT,resolve,cqt,decode,write_json,read_json,DT,SR
from .model import load_original,load_project,predict

def ffmpeg_exe():
    if shutil.which('ffmpeg'):return shutil.which('ffmpeg')
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def extract_audio(src,dst):
    r=subprocess.run([ffmpeg_exe(),'-nostdin','-y','-v','error','-i',str(Path(src).resolve()),'-map','0:a:0','-vn',
                      '-af','aresample=async=1:first_pts=0','-ac','1','-ar',str(SR),'-c:a','pcm_s16le',str(dst)],capture_output=True,text=True)
    if r.returncode:raise ValueError('Audio conversion failed: '+r.stderr[-1000:])

def infer(path,checkpoint=None,baseline=False,penalty=None,device='cpu'):
    meta_path=resolve(checkpoint).with_suffix('.json') if checkpoint else None
    meta=read_json(meta_path) if meta_path and meta_path.exists() else {}
    if penalty is None:penalty=meta.get('penalty',2.)
    window=int(meta.get('window',256));overlap=min(64,window//4)
    with tempfile.TemporaryDirectory(prefix='chordchart_') as d:
        wav=Path(d)/'audio.wav';extract_audio(path,wav);features,duration=cqt(wav)
        if baseline:
            members=[predict(load_original(i,power=False,device=device),features,device,window=window,overlap=overlap) for i in range(5)]
            probs=[np.mean([m[i] for m in members],axis=0) for i in range(6)]
        else:
            if not checkpoint:raise ValueError('Specify a project checkpoint or baseline')
            m,_=load_project(resolve(checkpoint),device);probs=predict(m,features,device,window=window,overlap=overlap)
        segments=decode(probs,penalty)
        for s in segments:
            s['end']=min(s['end'],duration);s['needs_review']=s['margin']<meta.get('review_margin',.5)
        segments=[s for s in segments if s['end']>s['start']]
        import librosa
        y,_=librosa.load(wav,sr=SR)
        try:
            tempo,beats=librosa.beat.beat_track(y=y,sr=SR,hop_length=512,units='time')
            tempo=float(np.asarray(tempo).ravel()[0])
        except Exception:tempo=0.;beats=np.array([])
        return {'model':'original_ensemble' if baseline else str(checkpoint),'duration':duration,
                'segments':segments,'beats':beats.tolist(),'tempo':tempo,
                'beat_note':'Beat positions are estimates; downbeat/meter must be confirmed.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--checkpoint');p.add_argument('--baseline',action='store_true');p.add_argument('--penalty',type=float,default=None);p.add_argument('--device',default='cpu');p.add_argument('--out',default='outputs/result.json');a=p.parse_args()
    if '://' in a.input:p.error('Local files only')
    r=infer(a.input,a.checkpoint,a.baseline,a.penalty,a.device);write_json(resolve(a.out),r);print('Saved',a.out)
if __name__=='__main__':main()
