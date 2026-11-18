"""Check every GOAT training clip's note/token roundtrip without loading audio AI."""
import json,random,argparse
from types import SimpleNamespace
import soundfile as sf
from scripts.a_candidate_data import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/goat_v1/manifest.json');p.add_argument('--out',default='reports/goat/token_preflight.json');a=p.parse_args()
    import setuptools
    from utils.task_manager import TaskManager
    from scripts.a_yourmt3_lora import sample,token_roundtrip
    model=SimpleNamespace(audio_cfg={'sample_rate':16000,'input_frames':32767},task_manager=TaskManager('mc13_full_plus_256'))
    rows=[r for r in json.load(open(a.manifest)) if r['split']=='train']
    random.seed(17);records=[];boundary=[]
    for r in rows:
        y,_=sf.read(r['inputs']['mix'],dtype='float32')
        for attempt in range(2):
            x,t,tr=sample(r,y,model,False,True);ref=tr.pop('teacher_reference')
            try:result=token_roundtrip(model,t,ref)
            except ValueError as e:
                if str(e)!='Preflight needs a contained reference note':raise
                boundary.append({'id':r['id'],'attempt':attempt,'teacher_notes':len(ref),'tokens':tr['tokens']});continue
            records.append({'id':r['id'],'attempt':attempt,'tokens':tr['tokens'],**result})
    if len({r['id'] for r in records})<len(rows)*.8:raise ValueError('Too few clips with observable attacks/releases to validate tokenizer')
    save(a.out,{'checked_training_clips':len(rows),'checked_windows':len(records)+len(boundary),
        'verified_contained_windows':len(records),'boundary_or_silent_windows':boundary,'records':records,
        'max_tokens_per_channel':model.task_manager.max_note_token_length_per_ch,'all_contained_roundtrips_passed':True,
        'maximum_tokens_observed':max(r['tokens'] for r in records),'test_used_for_selection':False})
    print('GOAT_TOKENS_OK',len(rows),len(records),'max',max(r['tokens'] for r in records),model.task_manager.max_note_token_length_per_ch,flush=True)

if __name__=='__main__':main()
