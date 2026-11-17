"""Explicit subcommands; external models stay in separate environments."""
import argparse,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Band2Tab research starter (not a validated high-accuracy model)')
    sub=p.add_subparsers(dest='cmd',required=True)
    f=sub.add_parser('fetch');f.add_argument('dataset',choices=['guitarset','baby','slakh-subset']);f.add_argument('--root',default='data')
    f.add_argument('--train',type=int,default=100);f.add_argument('--val',type=int,default=20);f.add_argument('--test',type=int,default=20)
    d=sub.add_parser('prepare');d.add_argument('--root',default='data');d.add_argument('--dataset',default='all',choices=['all','guitarset','baby','slakh'])
    r=sub.add_parser('remix');r.add_argument('--root',default='data');r.add_argument('--limit',type=int,default=30);r.add_argument('--snr',type=float,default=0.)
    c=sub.add_parser('convert');c.add_argument('input');c.add_argument('--out',default='inputs/demo.wav');c.add_argument('--seconds',type=float)
    t=sub.add_parser('train');t.add_argument('--manifest',default='data/manifest.json');t.add_argument('--out',default='runs/clean')
    t.add_argument('--epochs',type=int,default=5);t.add_argument('--steps',type=int,default=100);t.add_argument('--batch',type=int,default=4)
    t.add_argument('--lr',type=float,default=.001);t.add_argument('--device',default='cuda');t.add_argument('--init');t.add_argument('--resume');t.add_argument('--augment',action='store_true')
    t.add_argument('--pitch-augment',action='store_true');t.add_argument('--dataset',default='all',choices=['all','guitarset','baby','slakh'])
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint',required=True);e.add_argument('--manifest',default='data/manifest.json')
    e.add_argument('--split',default='val',choices=['val','test']);e.add_argument('--out',default='reports/eval.json');e.add_argument('--device',default='cpu')
    e.add_argument('--frame-threshold',type=float);e.add_argument('--offset-threshold',type=float);e.add_argument('--threshold',type=float,default=.5);e.add_argument('--tune',action='store_true')
    b=sub.add_parser('benchmark');b.add_argument('--engine',choices=['muscriptor','basic-pitch'],required=True);b.add_argument('--binary',required=True)
    b.add_argument('--manifest',default='data/manifest_clean.json');b.add_argument('--split',choices=['val','test'],default='val')
    b.add_argument('--out',required=True);b.add_argument('--model',default='medium');b.add_argument('--device',default='cuda');b.add_argument('--separator');b.add_argument('--limit',type=int,default=0)
    m=sub.add_parser('tab');m.add_argument('midi');m.add_argument('--out',default='results/tab');m.add_argument('--part',choices=['guitar','bass'])
    for a in [m]:
        a.add_argument('--bpm',type=float,default=None);a.add_argument('--meter',type=int,choices=[3,4],default=None);a.add_argument('--grid',type=int,choices=[3,4],default=4)
    n=sub.add_parser('notes');n.add_argument('audio');n.add_argument('--checkpoint',required=True);n.add_argument('--part',choices=['guitar','bass'],required=True)
    n.add_argument('--out',default='results/crnn');n.add_argument('--device',default='cpu');n.add_argument('--frame-threshold',type=float);n.add_argument('--offset-threshold',type=float);n.add_argument('--threshold',type=float);n.add_argument('--bpm',type=float,default=120.)
    q=sub.add_parser('snapshot');q.add_argument('input');q.add_argument('--time',type=float,required=True);q.add_argument('--chords')
    v=sub.add_parser('verify');v.add_argument('audio');v.add_argument('--checkpoint',required=True);v.add_argument('--notes-json',required=True);v.add_argument('--part',required=True,choices=['guitar','bass']);v.add_argument('--threshold',type=float,default=.3);v.add_argument('--device',default='cpu')
    s=sub.add_parser('score-json');s.add_argument('input');s.add_argument('--out',default='results/edited')
    a=p.parse_args()
    if a.cmd=='fetch':
        from .data import download,download_slakh_subset
        if a.dataset=='slakh-subset':download_slakh_subset(a.root,a.train,a.val,a.test)
        else:download(a.dataset,a.root)
    elif a.cmd=='prepare':
        from .data import prepare
        prepare(a.root,a.dataset)
    elif a.cmd=='remix':
        from .data import remix
        remix(a.root,a.limit,a.snr)
    elif a.cmd=='convert':
        from .audio import convert
        print(convert(a.input,a.out,a.seconds))
    elif a.cmd=='train':
        from .learn import train
        print(train(a.manifest,a.out,a.epochs,a.steps,a.batch,a.lr,a.device,a.init,a.augment,a.dataset,a.resume,a.pitch_augment))
    elif a.cmd=='evaluate':
        from .learn import evaluate
        evaluate(a.checkpoint,a.manifest,a.split,a.out,a.device,a.threshold,a.tune,a.frame_threshold,a.offset_threshold)
    elif a.cmd=='benchmark':
        from .learn import benchmark
        benchmark(a.engine,a.binary,a.manifest,a.split,a.out,a.model,a.device,a.separator,a.limit)
    elif a.cmd=='tab':
        from .score import read_midi,write_outputs,midi_timing
        bpm,meter,beats=midi_timing(a.midi)
        print(write_outputs(read_midi(a.midi,a.part),a.out,a.bpm or bpm,a.meter or meter,a.grid,beats=beats if a.bpm is None else None))
    elif a.cmd=='notes':
        from .model import load,predict
        from .audio import features,decode
        from .score import write_outputs
        model,state=load(a.checkpoint,a.device);prob=predict(model,features(a.audio),int(a.part=='bass'),a.device)
        config=state.get('decoding',{})
        onset=a.threshold if a.threshold is not None else config.get('onset',.5)
        frame=a.frame_threshold if a.frame_threshold is not None else config.get('frame',onset)
        offset=a.offset_threshold if a.offset_threshold is not None else config.get('offset',onset)
        notes=decode(prob,onset=onset,frame=frame,offset=offset)
        print(write_outputs({a.part:notes},a.out,a.bpm))
    elif a.cmd=='snapshot':
        from .score import active_at
        j=json.loads(Path(a.input).read_text());ch=json.loads(Path(a.chords).read_text()) if a.chords else None
        print(json.dumps(active_at(j.get('parts',j),a.time,ch),ensure_ascii=False,indent=2))
    elif a.cmd=='verify':
        from .review import verify
        verify(a.audio,a.checkpoint,a.notes_json,a.part,a.threshold,a.device)
    elif a.cmd=='score-json':
        from .score import write_outputs
        j=json.loads(Path(a.input).read_text());print(write_outputs(j['parts'],a.out,j.get('bpm',120),j.get('beats_per_bar',4),j.get('grid',4),j.get('tuning'),j.get('beat_times')))
if __name__=='__main__':main()
