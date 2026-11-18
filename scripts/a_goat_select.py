"""Pre-inference selection from publisher metadata; all views stay together."""
from pathlib import Path
import json,csv,hashlib,collections

def main():
    root=Path('reports/goat');dest=root/'selection.json'
    catalog=json.loads((root/'catalog.json').read_text())
    mapping={r['name']:r for r in catalog['members'] if not r['name'].startswith('__MACOSX/')}
    rows=list(csv.DictReader(open('data/raw/goat/GOAT/metadata.csv')));eligible=[];missing=[]
    for r in rows:
        # Publisher CSV paths omit the actual ZIP's extra data/ component.
        members=['GOAT/data/'+r['item']+'/'+r['item']+s for s in ('.wav','_amp_1.wav','_amp_5.wav','_fine_aligned.mid','.txt','.gp5')]
        if not all(p in mapping for p in members):
            missing.append({'item':r['item'],'missing':[p for p in members if p not in mapping]});continue
        r['source_split']=r['split'];r['split']='test' if r['split']=='test' else 'val' if r['player']=='player2' else 'train'
        r['members']=members;eligible.append(r)
    key=lambda r:hashlib.sha256(('band2tab-goat-v1-'+r['item']).encode()).hexdigest()
    selected=[]
    for split,cap in (('train',48),('val',8),('test',100)):
        selected+=sorted([r for r in eligible if r['split']==split],key=key)[:cap]
    members=['GOAT/metadata.csv']
    for r in selected:members += [p for p in r['members'] if r['split']!='train' or not p.endswith('_amp_5.wav')]
    record={'members':members,'sources':selected,'excluded_missing_annotations_or_required_views':missing,
        'selection':'Before inference: author test preserved; author training player2 supplies validation. Hash-select 48 training and 8 validation items. No performer-independent test claim; original composition identifiers not supplied.',
        'extracted_bytes':sum(mapping[p]['bytes'] for p in members),'compressed_bytes':sum(mapping[p]['compressed_bytes'] for p in members),'test_used_for_selection':False}
    if dest.exists() and json.loads(dest.read_text())!=record:raise ValueError('Frozen GOAT selection differs')
    dest.write_text(json.dumps(record,indent=2))
    print('GOAT_SELECTION',collections.Counter(r['split'] for r in selected),record['extracted_bytes'],flush=True)

if __name__=='__main__':main()
