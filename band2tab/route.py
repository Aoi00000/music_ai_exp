"""Small supervised route selector, exported as auditable JSON decision trees."""
import numpy as np
import soundfile as sf
from .refine import candidates_from_midi

NAMES=('mix_log_rms','removed_to_mix_rms','guitar_to_mix_rms','bass_to_mix_rms',
       'retained_to_mix_rms','guitar_mix_correlation','bass_mix_correlation',
       'guitar_notes_per_second','bass_notes_per_second','other_notes_per_second',
       'guitar_mean_pitch','bass_mean_pitch')
METHODS=('M0_original','M1_no_drums_vocals','M2_separated_forced','M3_naive_union')

def route_features(inputs,mix_midi):
    ys={}
    for k in ('mix','no_drums_vocals','guitar','bass'):
        y,sr=sf.read(inputs[k],dtype='float32')
        if sr!=16000 or y.ndim!=1:raise ValueError('Expected canonical mono 16kHz')
        ys[k]=y
    lengths=[len(y) for y in ys.values()]
    # A two-stage 44.1k/16k resample can round up by one sample. Align only
    # feature comparisons; leave saved audio/model inputs unchanged.
    if max(lengths)-min(lengths)>2:raise ValueError('Route audio lengths differ by more than resampling rounding')
    ys={k:y[:min(lengths)] for k,y in ys.items()}
    rms=lambda y:max(1e-8,float(np.sqrt(np.mean(y*y))))
    scale=rms(ys['mix']);ratio=lambda y:min(8.,rms(y)/scale)
    def correlation(y):
        a=ys['mix']-ys['mix'].mean();b=y-y.mean()
        return float(np.dot(a,b)/max(1e-8,np.sqrt(np.dot(a,a)*np.dot(b,b))))
    notes=candidates_from_midi(mix_midi);duration=max(.01,len(ys['mix'])/16000)
    grouped={c:[n for n in notes if n['baseline_class']==c] for c in (0,1,2)}
    return np.array([np.log1p(scale),ratio(ys['mix']-ys['no_drums_vocals']),ratio(ys['guitar']),
        ratio(ys['bass']),ratio(ys['no_drums_vocals']),correlation(ys['guitar']),correlation(ys['bass']),
        *[len(grouped[c])/duration for c in (1,2,0)],
        *[float(np.mean([n['pitch'] for n in grouped[c]]))/127 if grouped[c] else 0. for c in (1,2)]],np.float32)

def predict(tree,x,minimum_gain=0.):
    index=0
    while tree['children_left'][index]!=-1:
        feature=tree['feature'][index];threshold=tree['threshold'][index]
        index=tree['children_left'][index] if x[feature]<=threshold else tree['children_right'][index]
    values=np.asarray(tree['value'][index]);best=int(np.argmax(values));gain=float(values[best])
    return (METHODS[best] if gain>minimum_gain else METHODS[0]),gain

def recommend(inputs,mix_midi,checkpoint):
    """Recommend an optional route; do not overwrite any recognized notes."""
    import json,hashlib
    from pathlib import Path
    path=Path(checkpoint);state=json.loads(path.read_text())
    if state['schema']!='separation_route_quality_tree_v1' or tuple(state['feature_names'])!=NAMES:
        raise ValueError('Unknown route model schema/features')
    selected=state['selected'];x=route_features(inputs,mix_midi)
    name,gain=predict(selected['tree'],x,selected['minimum_gain']) if selected['tree'] else (METHODS[0],0.)
    return {'recommended_route':name,'predicted_gain':gain,'features':dict(zip(NAMES,x.astype(float).tolist())),
            'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'automatic_application_approved':False,
            'limitations':'Validation-selected recommendation; predicted gain is not a calibrated probability.'}

def main():
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--mix-midi',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    inputs={k:Path(a.input_dir)/(k+'.wav') for k in ('mix','no_drums_vocals','guitar','bass')}
    result=recommend(inputs,a.mix_midi,a.checkpoint);path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print('Recommended',result['recommended_route'])

if __name__=='__main__':main()
