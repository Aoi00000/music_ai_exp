"""Additional original piano MIDI for error diagnosis, never model selection."""
from pathlib import Path
import json,yaml
from scripts.a_temporal_fetch import listing,download

root=Path('reports/temporal_v1');acquisition=json.loads((root/'source_acquisition.json').read_text());records=[]
for source in acquisition['selected']:
    folder=Path(source['folder']);metadata=yaml.safe_load((folder/'metadata.yaml').read_text())
    paths={f'MIDI/{stem}.mid' for stem,info in metadata['stems'].items() if info.get('inst_class','').lower()=='piano'}
    base='data/Slakh2100_redux/'+source['official_split']+'/'+source['track']
    specs={Path(s['path']).relative_to(base).as_posix():s for s in listing(base,True) if s['type']=='file'}
    for path in sorted(paths):
        if path not in specs:continue
        r=download(specs[path],folder/path);records.append({'group':source['group'],'part':'piano',**r})
    print('PIANO_LABELS',source['track'],flush=True)
(root/'other_annotation_acquisition.json').write_text(json.dumps(records,indent=2))
