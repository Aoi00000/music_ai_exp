"""Cache frozen-base soft targets on unlabelled real bass; no GT claims."""
from pathlib import Path
import json,torch,numpy as np,soundfile as sf
from scripts.a_candidate_data import save,sha
from scripts.a_yourmt3_predict import load_model
from scripts.a_yourmt3_lora import role_channels
from utils.event2note import merge_zipped_note_events_and_ties_to_notes


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3)
    manifest=Path('data/bass_retention_v1/manifest.json');rows=json.load(open(manifest));folder=Path('data/bass_retention_v1/teacher_cache')
    if any(r['split'] not in ('train','val') or r['labels_available'] for r in rows):raise ValueError('Only unlabelled train/val bass is allowed')
    model,info=load_model();model.eval();channel=role_channels(model)[1];records=[]
    for row in rows:
        if row['ac_rms']<1e-5:continue
        audio,sr=sf.read(row['audio'],dtype='float32')
        if sr!=16000 or audio.ndim!=1:raise ValueError('Retention audio format changed')
        size=model.audio_cfg['input_frames']
        for index,fraction in enumerate((0.,.5,1.)):
            start=round(.25*sr+(len(audio)-size-.5*sr)*fraction);ident=row['id']+'_'+str(index);path=folder/(ident+'.pt')
            fp={'base_sha256':info['checkpoint_sha256'],'source_sha256':sha(row['audio']),'start_sample':start,'frames':size}
            if path.exists():
                cached=torch.load(path,map_location='cpu',weights_only=True)
                if cached['fingerprint']!=fp:raise ValueError('Retention cache source changed')
            else:
                x=torch.from_numpy(audio[start:start+size].copy())[None,None].to('cuda')
                if x.shape[-1]!=size:raise ValueError('Short retention window')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    tokens=model.inference(x);sequence=tokens[0,channel];eos=torch.nonzero(sequence==1)
                    length=int(eos[0])+1 if len(eos) else len(sequence)
                    if not len(eos):raise ValueError('Teacher bass generation was truncated')
                    labels=torch.zeros((1,model.task_manager.num_decoding_channels,length),dtype=torch.long,device='cuda');labels[:,:,0]=1
                    labels[:,channel]=sequence[:length];logits=model(x,labels)['logits'][:,channel:channel+1]
                    mask=labels[:,channel:channel+1]!=0;agreement=float(((logits.argmax(-1)==labels[:,channel:channel+1])*mask).sum()/mask.sum())
                if agreement<.9:raise ValueError('Generation/teacher-forcing token alignment failed')
                zipped,_,_=model.task_manager.detokenize_list_batches([labels[:,channel].cpu().numpy()],[0.],return_events=True)
                notes,_=merge_zipped_note_events_and_ties_to_notes(zipped)
                cached={'audio':x.cpu(),'labels':labels.cpu(),'teacher_logits':logits.cpu(),'teacher_bass_notes':len(notes),
                    'fingerprint':fp,'teacher_forcing_argmax_agreement':agreement,'pseudo_labels_not_ground_truth':True}
                path.parent.mkdir(parents=True,exist_ok=True);torch.save(cached,path)
            records.append({'id':ident,'group':row['group'],'split':row['split'],'cache':str(path.resolve()),
                'cache_sha256':sha(path),'teacher_bass_notes':cached['teacher_bass_notes'],'teacher_forcing_argmax_agreement':cached['teacher_forcing_argmax_agreement']})
            print('RETENTION_TEACHER_CACHED',ident,row['split'],cached['teacher_bass_notes'],flush=True)
    save(folder/'index.json',{'records':records,'manifest_sha256':sha(manifest),'base_sha256':info['checkpoint_sha256'],
        'supervision':'Frozen-model pseudo targets for retention only, not annotated note truth','target_channel':channel,'test_used':False})
    print('RETENTION_TEACHER_READY',len(records),flush=True)


if __name__=='__main__':main()
