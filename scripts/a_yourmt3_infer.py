"""Local file inference with a pinned YourMT3 base and optional learned LoRA.

JSON exposes only guitar/bass notes. The raw MIDI is preserved for inspection.
Dependencies and model source must be installed separately, as documented.
"""
from pathlib import Path
import argparse,json,time,torch
from band2tab.score import read_midi
from band2tab.lora import restore
from scripts.a_candidate_data import sha,save
from scripts.a_yourmt3_predict import load_model,transcribe

def main():
    p=argparse.ArgumentParser();p.add_argument('audio',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');a=p.parse_args()
    if not a.audio.is_file():raise FileNotFoundError(a.audio)
    torch.set_num_threads(4);started=time.monotonic();model,info=load_model(a.device);adapter_sha=None
    if a.adapter:
        state=torch.load(a.adapter,map_location='cpu',weights_only=True)
        if state['base_sha256']!=info['checkpoint_sha256']:raise ValueError('Adapter/base mismatch')
        restore(model,state);model.eval();adapter_sha=sha(a.adapter)
    loaded=time.monotonic();decoding=transcribe(model,a.audio,a.out,device=a.device);raw=read_midi(a.out)
    save(a.out.with_suffix('.json'),{'audio_sha256':sha(a.audio),'base':info,'adapter_sha256':adapter_sha,
        'device':a.device,'load_seconds':loaded-started,'inference_seconds':time.monotonic()-loaded,
        'parts':{p:raw.get(p,[]) for p in ('guitar','bass')},'decoding_errors':decoding,
        'timing_source':'Note times in seconds; raw MIDI default 120 BPM is not an estimated beat grid',
        'automatic_application_approved':False,'purpose':'Review before TAB integration'})
    print('YOURMT3_FILE_COMPLETE',a.out,a.out.with_suffix('.json'),flush=True)

if __name__=='__main__':main()
