"""Fixed two-arm hypothesis after measured octave-apart false-note errors."""
from pathlib import Path
import json,random
import numpy as np,torch
from band2tab.addition import _assign
from band2tab.octave import octave_context,prune_octaves
from scripts.a_candidate_data import save,sha
from scripts.a_candidate_train import aggregate,metrics
from scripts.a_release_train import load


def main():
    dest=Path('reports/octave_v1');run=Path('runs/octave_v1')
    if dest.exists() or run.exists():raise ValueError('Do not overwrite or extend frozen octave protocol')
    manifest,rows,cache,meta=load();rows=[r for r in rows if r['split']=='val' or r['dataset'].startswith('stem_band_')];train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    protocol=dict(arms=['local350','octave_context704'],steps=500,monitors=[100,500],thresholds=[0.,.05,.1,.2,.35],seed=23,
        purpose='Measured guitar FP include octave-apart attacks. Compare local features against nearest octave-peer differences; do not force true octaves to disappear.',
        candidates='Only guitar notes with another same-source note +/-12 semitones within 80ms. Bass and all other notes unchanged.',
        objective='Binary keep probability, positive weight 3. Exact-pitch unique onset50ms target; censor/guard masked.',
        gate='Each clean/noisy Note and Frame >= baseline, aggregate active exact >= baseline. Choose by .5 Note + .3 Frame + .2 Union.',
        selection_scope='Repeated 16 development conditions; diagnosis-driven experiment, not unseen evidence',manifest_sha256=sha(manifest),test_used=False)
    save(dest/'protocol.json',protocol)
    for r in rows:
        ident=r['id'];c=cache[ident];m=meta[ident];c['context'],c['eligible']=octave_context(c['x'],m['pool']);c['keep']=np.zeros(len(m['pool']),np.float32);c['valid']=np.zeros(len(m['pool']),bool)
        indices=[i for i,n in enumerate(m['pool']) if n['part']=='guitar' and r['guard_seconds']<=n['onset']<r['duration']-r['guard_seconds']]
        refs=[n for n in r['parts']['guitar'] if not n.get('censored_onset') and r['guard_seconds']<=n['onset']<r['duration']-r['guard_seconds']]
        for i,j in _assign(refs,[m['pool'][i] for i in indices]):c['keep'][indices[j]]=1.
        for i in indices:
            n=m['pool'][i];c['valid'][i]=c['eligible'][i] and not any(t.get('censored_onset') and t['pitch']==n['pitch'] and t['onset']<=n['onset']<t['offset'] for t in r['parts']['guitar'])
    def evaluate(probs,threshold):
        items=[];tracks=[]
        for r in val:
            m=meta[r['id']];parts=prune_octaves(m['pool'],probs[r['id']],cache[r['id']]['eligible'],threshold,m['unmodelled']);s=metrics(r,parts);items.append((r,s));tracks.append(dict(id=r['id'],dataset=r['dataset'],metrics=s,parts=parts))
        s=aggregate(items);s['octave_score']=.5*s['note_f1']+.3*s['frame_f1']+.2*s['union_frame_f1'];return dict(summary=s,tracks=tracks)
    baseline=evaluate({r['id']:np.ones(len(cache[r['id']]['x'])) for r in val},0.);results=[]
    torch.set_num_threads(2)
    for arm,key in [('local350','x'),('octave_context704','context')]:
        torch.manual_seed(23);np.random.seed(23);rng=random.Random(23)
        x=np.concatenate([cache[r['id']][key][cache[r['id']]['valid']] for r in train]);y=np.concatenate([cache[r['id']]['keep'][cache[r['id']]['valid']] for r in train]);mean=x.mean(0);std=x.std(0).clip(.05);tx=torch.from_numpy(np.clip((x-mean)/std,-8,8));ty=torch.from_numpy(y)
        model=torch.nn.Linear(x.shape[1],1);torch.nn.init.zeros_(model.weight);torch.nn.init.constant_(model.bias,2.);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1)
        config=dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),training_candidates=len(x),training_correct=int(y.sum()),protocol_sha256=sha(dest/'protocol.json'));save(run/arm/'config.json',config)
        for step in range(1,501):
            ix=np.random.randint(0,len(x),256);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tx[ix]).squeeze(-1),ty[ix],pos_weight=torch.tensor(3.));loss.backward();opt.step()
            if step not in protocol['monitors']:continue
            with torch.no_grad():probs={r['id']:model(torch.from_numpy(np.clip((cache[r['id']][key]-mean)/std,-8,8).astype(np.float32))).squeeze(-1).sigmoid().numpy() for r in val}
            ckpt=run/arm/f'step_{step}.pt';torch.save(dict(model=model.state_dict(),mean=mean.tolist(),std=std.tolist(),config=config,step=step),ckpt)
            for threshold in protocol['thresholds']:
                score=evaluate(probs,threshold);s=score['summary'];b=baseline['summary'];eligible=s['active_exact_match']>=b['active_exact_match']-1e-12 and all(s['datasets'][d][k]>=v[k]-1e-12 for d,v in b['datasets'].items() for k in ('note_f1','frame_f1'))
                results.append(dict(arm=arm,step=step,threshold=threshold,checkpoint=str(ckpt),eligible=eligible,**score))
            print('OCTAVE_VALIDATED',arm,step,flush=True)
        save(run/arm/'completed.json',dict(config=config,steps=500))
    allowed=[r for r in results if r['eligible'] and r['summary']['octave_score']>baseline['summary']['octave_score']+1e-8];best=max(allowed,key=lambda r:r['summary']['octave_score']) if allowed else None
    save(dest/'selection.json',dict(baseline=baseline,best=best,choices=results,identity_selected=best is None,automatic_application_approved=False,test_used=False));print('OCTAVE_SELECTION',None if best is None else (best['arm'],best['step'],best['threshold'],best['summary']),flush=True)


if __name__=='__main__':main()
