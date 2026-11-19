"""Local files -> six-source separation -> Mu -> optional learned corrections.

Keeps raw MIDI and original seconds. Engraving remains a separate reviewed step.
No network inference, beat quantization or guessed performance techniques.
"""
from pathlib import Path
import argparse,json,os,time,random,gc
import torch,numpy as np,soundfile as sf
from band2tab.audio import convert,features,PITCH_MIN,PITCH_MAX
from band2tab.separation import separate_inputs
from band2tab.stem_refiner import read_candidates,rms_envelope,feature_matrix,predict_corrections,suggest
from scripts.a_candidate_data import sha,save


def main():
    p=argparse.ArgumentParser();p.add_argument('file',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--corrector',type=Path);p.add_argument('--seconds',type=float);p.add_argument('--bpm',type=float);a=p.parse_args()
    if not a.file.is_file():raise FileNotFoundError(a.file)
    if a.out.exists():raise ValueError('Choose a new output folder; previous results are preserved')
    if a.seconds is not None and a.seconds<=0:raise ValueError('seconds must be positive')
    if a.bpm is not None and a.bpm<=0:raise ValueError('bpm must be positive')
    torch.set_num_threads(4);a.out.mkdir(parents=True);started=time.monotonic();native=a.out/'native.wav';convert(a.file,native,a.seconds)
    if sf.info(native).frames==0:raise ValueError('Input has no audio samples')
    os.environ['TORCH_HOME']=str(Path('models/torch').resolve())
    from demucs.pretrained import get_model
    separator=get_model('htdemucs_6s').eval();seed=int(sha(native)[:8],16)%2**31;torch.manual_seed(seed);random.seed(seed)
    separation=separate_inputs(native,a.out/'stems',separator,'cpu');del separator;gc.collect()
    from muscriptor import TranscriptionModel
    weight=Path('models/muscriptor-medium/model.safetensors');model=TranscriptionModel.load_model(weight,device='cpu',dtype='float32');duration=sf.info(native).duration
    raw={};pool=[];outside={p:[] for p in ('guitar','bass')}
    for part in ('guitar','bass'):
        midi,_=model.transcribe_and_postprocess(a.out/'stems'/(part+'.wav'),detect_tempo=False);path=a.out/('raw_'+part+'.mid');path.write_bytes(midi)
        notes=read_candidates(path,part,duration);raw[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes]
        pool += [n for n in notes if PITCH_MIN<=n['pitch']<=PITCH_MAX]
        outside[part]=[{k:n[k] for k in ('pitch','onset','offset','velocity')} for n in notes if not PITCH_MIN<=n['pitch']<=PITCH_MAX]
    result=dict(parts=raw,raw_parts=raw,duration=duration,source_sha256=sha(a.file),muscriptor_sha256=sha(weight),
        separation=separation,source_role_policy='All non-drum notes in estimated guitar/bass stem assigned to that source; leakage may remain',
        detected_tempo=None,raw_time_unit='seconds',midi_tempo_is_placeholder=True,corrector_applied=False)
    if a.corrector:
        state=torch.load(a.corrector,map_location='cpu',weights_only=True);views={};env={}
        for part in ('mix','guitar','bass'):
            path=a.out/'stems'/(part+'.wav');views[part]=features(path);y,sr=sf.read(path,dtype='float32',always_2d=True);env[part]=rms_envelope(y,sr)
        x=feature_matrix(views,env,pool,duration);prob,delta=predict_corrections(state,x)
        result.update(parts=suggest(pool,prob,delta,duration,unmodelled=outside,**state['policy']),corrector_applied=True,
            corrector_sha256=sha(a.corrector),corrector_policy=state['policy'],requires_review=True)
    result['seconds']=time.monotonic()-started;save(a.out/'notes.json',result)
    # Engraving requires explicitly supplied tempo; MIDI's 120 is not evidence.
    if a.bpm is not None:save(a.out/'score_input.json',{'parts':result['parts'],'bpm':a.bpm})
    print('BAND_PIPELINE_COMPLETE',str(a.out/'notes.json'),flush=True)


if __name__=='__main__':main()
