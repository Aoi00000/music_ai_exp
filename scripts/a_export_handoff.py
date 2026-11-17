"""Export the validation-selected acoustic model, preserving its provenance."""
from pathlib import Path
import json,hashlib,torch
p=json.loads(Path('reports/v2_external_protocol.json').read_text());name=p['selected_by_validation'];run=p['runs'][name];source=Path(run['checkpoint']);source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha!=run['checkpoint_sha256']:raise ValueError('Frozen source mismatch')
state=torch.load(source,map_location='cpu',weights_only=True);t=run['thresholds']
state.update(label_semantics='sounding_pitch_v2',source_checkpoint_sha256=source_sha,decoding={'onset':t['threshold'],'frame':t['frame_threshold'],'offset':t['offset_threshold'],'minimum_note_seconds':.03},analysis_scope='single known guitar or bass part; not instrument identification',selected_on_validation=True)
folder=Path('models');folder.mkdir(exist_ok=True);out=folder/'acoustic_review.pt';torch.save(state,out)
info={'selected_experiment':name,'selected_on':'validation only','model_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_sha256':source_sha,'label_semantics':state['label_semantics'],'decoding':state['decoding'],'pitch_min':23,'pitch_max':88,'external_protocol_sha256':hashlib.sha256(Path('reports/v2_external_protocol.json').read_bytes()).hexdigest()}
(folder/'acoustic_review.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
