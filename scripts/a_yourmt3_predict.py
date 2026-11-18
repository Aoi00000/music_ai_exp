"""Local, pinned YourMT3+ inference for validation before any test access.

No hosted inference, uploads, YouTube or online experiment logging is used.
Requires the separately installed vendor/yourmt3_deps on PYTHONPATH.
"""
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
from unittest.mock import patch
import argparse,json,os,time
import numpy as np,soundfile as sf,torch
from band2tab.separation import resample
from scripts.a_candidate_data import rows_for,sha,save
from scripts.a_yourmt3_fetch import CHECKPOINT

class InertMetadata:
    """Unused publisher training metadata; no publisher reconstruction code."""
    def __new__(cls,*args,**kwargs):return object.__new__(cls)
    def __init__(self,*args,**kwargs):pass

def tensor_checkpoint(path,loader):
    inert_names=('utils.note_event_dataclasses.Event','utils.note_event_dataclasses.EventRange',
        'utils.task_manager.TaskManager','utils.tokenizer.NoteEventTokenizer','utils.event_codec.FastCodec')
    dtype_classes={type(np.dtype(k)) for k in ('bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64','complex64','complex128','O','S','U')}
    allowed=[(InertMetadata,name) for name in inert_names]+[np.ndarray,np.dtype,np.core.multiarray._reconstruct,np.core.multiarray.scalar,*dtype_classes]
    with torch.serialization.safe_globals(allowed):checkpoint=loader(path,map_location='cpu',weights_only=True)
    state=checkpoint['state_dict']
    if not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):raise ValueError('Expected a tensor-only state dictionary')
    return {'state_dict':state}

@contextmanager
def working_directory(path):
    previous=Path.cwd();os.chdir(path)
    try:yield
    finally:os.chdir(previous)

def load_model(device='cuda'):
    import setuptools  # Register distutils compatibility for upstream WandB on Python 3.12.
    from model_helper import load_model_checkpoint
    from model.ymt3 import YourMT3
    root=Path('vendor/yourmt3').resolve();acq=json.loads(Path('reports/yourmt3/acquisition.json').read_text())
    record=next(r for r in acq['files'] if r['path'].endswith(CHECKPOINT))
    if sha(root/CHECKPOINT)!=record['sha256']:raise ValueError('Checkpoint changed')
    safe_load=torch.load;load_state=YourMT3.load_state_dict;checks=[]
    def only_weights(*args,**kwargs):
        return tensor_checkpoint(args[0],safe_load)
    def checked_state(model,state_dict,strict=False,**kwargs):
        status=load_state(model,state_dict,strict=strict,**kwargs)
        if status.missing_keys or status.unexpected_keys:raise ValueError('YourMT3 state mismatch '+str(status))
        checks.append({'missing':list(status.missing_keys),'unexpected':list(status.unexpected_keys)});return status
    name='mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt'
    args=[name,'-p','2024','-tk','mc13_full_plus_256','-dec','multi-t5','-nl','26','-enc','perceiver-tf',
          '-sqr','1','-ff','moe','-wf','4','-nmoe','8','-kmoe','2','-act','silu','-epe','rope','-rp','1',
          '-ac','spec','-hop','300','-atc','1','-pr','32','-wb','disabled']
    with working_directory(root),patch('torch.load',only_weights),patch.object(YourMT3,'load_state_dict',checked_state):
        model=load_model_checkpoint(args=args,device='cpu')
    if not checks:raise ValueError('State-dictionary load was not verified')
    return model.to(device).eval(),{'revision':acq['revision'],'checkpoint_sha256':record['sha256'],
        'state_checks':checks,'parameters':sum(p.numel() for p in model.parameters()),'autocast':'float16 on CUDA','batch_size':1}

@torch.inference_mode()
def transcribe(model,audio,out,device='cuda'):
    from utils.audio import slice_padded_array
    from utils.note2event import mix_notes
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.utils import write_model_output_as_midi
    y,sr=sf.read(audio,dtype='float32',always_2d=True);y=resample(y.mean(1),sr,model.audio_cfg['sample_rate'])
    segments=slice_padded_array(y[None],model.audio_cfg['input_frames'],model.audio_cfg['input_frames'])
    segments=torch.from_numpy(segments.astype(np.float32)).to(device).unsqueeze(1)
    with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
        tokens,_=model.inference_file(bsz=1,audio_segments=segments)
    starts=[model.audio_cfg['input_frames']*i/model.audio_cfg['sample_rate'] for i in range(len(segments))]
    channels=[];errors=Counter()
    for channel in range(model.task_manager.num_decoding_channels):
        batches=[b[:,channel,:] for b in tokens]
        zipped,_,counter=model.task_manager.detokenize_list_batches(batches,starts,return_events=True)
        notes,counter2=merge_zipped_note_events_and_ties_to_notes(zipped);channels.append(notes);errors+=counter;errors+=counter2
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    write_model_output_as_midi(mix_notes(channels),str(out.parent),out.stem,model.midi_output_inverse_vocab)
    generated=out.parent/'model_output'/(out.stem+'.mid')
    if not generated.exists():raise ValueError('Expected YourMT3 MIDI was not generated')
    generated.replace(out);return dict(errors)

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val','test'],default='val');p.add_argument('--limit',type=int);p.add_argument('--id');a=p.parse_args()
    os.environ['WANDB_MODE']='disabled';os.environ['WANDB_SILENT']='true';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';torch.set_num_threads(2)
    # Test access remains controlled by the candidate experiment's frozen selection.
    rows=[r for r in rows_for(a.split=='test') if r['split']==a.split and (a.split!='train' or not r['hard_training'])]
    if a.id:rows=[r for r in rows if r['id']==a.id]
    if a.limit:rows=rows[:a.limit]
    if not rows:raise ValueError('No requested rows')
    model=None;model_info=None
    for r in rows:
        out=Path('reports/yourmt3/midi')/(r['id']+'.mid');meta=out.with_suffix('.json');source_sha=sha(r['inputs']['mix'])
        if out.exists() and meta.exists():
            if json.loads(meta.read_text())['audio_sha256']!=source_sha:raise ValueError('Stale YourMT3 MIDI')
            continue
        if model is None:model,model_info=load_model();save('reports/yourmt3/runtime.json',model_info)
        started=time.monotonic();errors=transcribe(model,r['inputs']['mix'],out)
        save(meta,{'audio_sha256':source_sha,'model':model_info,'seconds':time.monotonic()-started,'decoding_errors':errors,'split':a.split})
        print('YOURMT3_TRANSCRIBED',r['id'],round(time.monotonic()-started,2),flush=True)

if __name__=='__main__':main()
