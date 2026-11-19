"""Feedback experiment: retain source Mu timbre program, supervise gold pitches."""
from pathlib import Path
import argparse,copy,json
import numpy as np
from band2tab.mu_adapter import encode_window,roundtrip
from scripts.a_candidate_data import save,sha
DATA=Path('data/mu_native_v2');REPORT=Path('reports/mu_native_v2');RUN=Path('runs/mu_native_v2')

def prepare():
    from muscriptor.tokenizer.mt3 import MT3Tokenizer
    if (DATA/'protocol.json').exists():raise ValueError('Native-program protocol frozen')
    old=json.load(open('data/mu_adapt_v1/manifest.json'));rows={r['id']:r for r in json.load(open('data/residual_v1/manifest.json'))};programs={r['id']:r for r in json.load(open('reports/mu_adapt_v1/program_diagnostic.json'))['records']};tok=MT3Tokenizer('MT3_FULL_PLUS');out=[];qc=[]
    for a in old:
        r=rows[a['id'].rsplit('_',1)[0]];native=programs[a['id']]['native_program'];tokens,refs=encode_window(tok,r['parts'][a['part']],a['source_start'],5.,native);q=roundtrip(tok,tokens,refs)
        with np.load(a['cache']) as z:audio=z['audio'].copy();before=z['tokens'].copy()
        after=np.array(tokens,np.int64)
        if len(before)!=len(after):raise ValueError('Program-only target change altered token count')
        changed=np.flatnonzero(before!=after);program_ids={i for (kind,value),i in tok._token_index.items() if kind=='program'}
        if any(int(before[i]) not in program_ids or int(after[i]) not in program_ids for i in changed):raise ValueError('Non-program training token changed')
        dest=DATA/'cache'/(a['id']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,audio=audio,tokens=after)
        out.append({**a,'cache':str(dest),'cache_sha256':sha(dest),'program':native,'canonical_program':a['program']});qc.append(dict(id=a['id'],changed_program_tokens=len(changed),**q))
    save(DATA/'manifest.json',out);save(DATA/'monitor.json',json.load(open('data/mu_adapt_v1/monitor.json')));save(DATA/'review_manifest.json',json.load(open('data/mu_adapt_v1/review_manifest.json')))
    save(REPORT/'token_preflight.json',dict(records=qc,contained_total=sum(r['contained'] for r in qc),all_contained_roundtrips_passed=True,only_program_tokens_changed=True))
    protocol=copy.deepcopy(json.load(open('data/mu_adapt_v1/protocol.json')));protocol.update(hypothesis='Canonical program26 disagreed with dominant frozen Mu timbre program in80/108 guitar windows. Compare native timbre program targets with otherwise identical gold audio, pitches, timing, RNG,120 steps and optional tone processing.',
       timbre_teacher='Duration-weighted dominant own-role frozen Mu program; onlyG24..31 orB32..39/43; fallback canonical. Pseudo subtype, not reference timbre annotation.',
       manifest_sha256=sha(DATA/'manifest.json'),monitor_sha256=sha(DATA/'monitor.json'),review_manifest_sha256=sha(DATA/'review_manifest.json'),
       canonical_protocol_sha256=sha('data/mu_adapt_v1/protocol.json'),implementation_sha256={p:sha(p) for p in ['band2tab/mu_adapter.py','scripts/a_mu_adapt.py','scripts/a_mu_native_cycle.py','scripts/a_mu_native_eval.py','band2tab/augment.py']})
    save(DATA/'protocol.json',protocol);print('MU_NATIVE_PREPARED',len(out),'program_only_tokens_changed',sum(r['changed_program_tokens'] for r in qc),flush=True)

def train():
    import scripts.a_mu_adapt as runner
    runner.DATA=DATA;runner.REPORT=REPORT;runner.RUN=RUN;runner.train()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);a=p.parse_args();globals()[a.command]()
