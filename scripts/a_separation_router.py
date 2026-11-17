"""Train route-quality prediction without touching test scores or labels."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from band2tab.route import NAMES,METHODS,route_features,predict
from scripts.a_separation_fusion import aggregate,eligible
ROOT=Path('data/separation_v1');REPORT=Path('reports/separation_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):Path(p).write_text(json.dumps(o,indent=2))
def inputs(r):return route_features(r['inputs'],REPORT/'midi'/r['id']/'mix.mid')
def export(model):
    t=model.tree_
    return {'children_left':t.children_left.tolist(),'children_right':t.children_right.tolist(),
            'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,:,0].tolist()}
def train(conservative=False):
    from sklearn.tree import DecisionTreeRegressor,export_text
    prefix='router_conservative' if conservative else 'router'
    path=REPORT/(prefix+'_selection.json')
    if path.exists() or (REPORT/'fusion_test.json').exists():raise RuntimeError('Freeze router before inspecting test scores')
    manifest=[r for r in json.loads((ROOT/'manifest.json').read_text()) if r['split']!='test']
    scored={r['id']:r for r in json.loads((REPORT/'comparison_development.json').read_text())['tracks']}
    tr=[r for r in manifest if r['split']=='train'];va=[r for r in manifest if r['split']=='val']
    x=np.stack([inputs(r) for r in tr]);vx=np.stack([inputs(r) for r in va])
    quality=lambda m:np.mean([m[k] for k in ('note_f1','frame_f1','union_frame_f1')])
    if conservative:
        # A gain in labels/onsets must not hide worse sounding-pitch recognition.
        y=np.array([[min(scored[r['id']]['metrics'][m][k]-scored[r['id']]['metrics']['M0_original'][k]
                         for k in ('note_f1','frame_f1','union_frame_f1')) for m in METHODS] for r in tr])
        datasets=[r.get('source_dataset','p3') for r in tr]
        weights=np.array([1/datasets.count(d) for d in datasets])
    else:
        y=np.array([[quality(scored[r['id']]['metrics'][m])-quality(scored[r['id']]['metrics']['M0_original']) for m in METHODS] for r in tr])
        weights=None
    baseline=aggregate([(r,scored[r['id']]['metrics']['M0_original']) for r in va]);selected=None;comparisons=[]
    for depth in (1,2,3):
        for leaf in (4,8):
            model=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=leaf,random_state=17).fit(x,y,sample_weight=weights)
            tree=export(model)
            # Confirm JSON traversal reproduces scikit-learn leaf values.
            for xx,values in zip(vx,model.predict(vx)):
                name,gain=predict(tree,xx,-1.)
                if abs(gain-float(values.max()))>1e-8:raise ValueError('Tree export mismatch')
            for minimum_gain in (0.,.02,.05,.1):
                routes=[predict(tree,xx,minimum_gain)[0] for xx in vx]
                score=aggregate([(r,scored[r['id']]['metrics'][route]) for r,route in zip(va,routes)])
                ok=eligible(score,baseline)
                record={'depth':depth,'min_samples_leaf':leaf,'minimum_gain':minimum_gain,'validation':score,'eligible':ok}
                comparisons.append(record)
                if ok and (selected is None or score['union_frame_f1']+score['note_f1']>selected['validation']['union_frame_f1']+selected['validation']['note_f1']+1e-8):
                    selected={**record,'tree':tree,'readable_tree':export_text(model,feature_names=list(NAMES)),'validation_routes':dict(zip([r['id'] for r in va],routes))}
    if selected is None:selected={'tree':None,'minimum_gain':0.,'validation':baseline,'validation_routes':{r['id']:'M0_original' for r in va}}
    previous=json.loads((REPORT/'selection.json').read_text())['selected']
    chosen='router' if selected['validation']['union_frame_f1']+selected['validation']['note_f1']>previous['validation']['union_frame_f1']+previous['validation']['note_f1']+1e-8 else 'previous_selection'
    report={'schema':'separation_route_quality_tree_v1','selected':selected,'combined_selection':chosen,
        'previous_selection':previous,'feature_names':NAMES,'route_names':METHODS,'comparisons':comparisons,
        'train_ids':[r['id'] for r in tr],'val_ids':[r['id'] for r in va],
        'target':('minimum of Note/instrument Frame/union Frame improvements; equal training-dataset weights' if conservative else 'mean(Note, instrument Frame, union Frame) improvement over original, training labels only'),
        'test_used_for_selection':False,'automatic_application_approved':False,'protocol_sha256':sha(REPORT/'protocol.json')}
    save(path,report);save(Path('models/muscriptor_separation_'+prefix+'_review.json'),report)
    print('ROUTER_FROZEN',chosen,{k:round(selected['validation'][k],4) for k in ('note_f1','frame_f1','union_frame_f1')},flush=True)
    print(selected.get('readable_tree','Identity'),flush=True)
def evaluate(conservative=False):
    prefix='router_conservative' if conservative else 'router'
    path=REPORT/(prefix+'_test.json')
    if path.exists():raise RuntimeError('Do not overwrite')
    selector=json.loads((REPORT/(prefix+'_selection.json')).read_text());base=json.loads((REPORT/'fusion_test.json').read_text())
    from scripts.a_separation_compare import get_rows
    mapping={r['id']:r for r in get_rows(True)};results=[]
    for row in base['tracks']:
        r=mapping[row['id']];selected=selector['selected']
        name,gain=predict(selected['tree'],inputs(r),selected['minimum_gain']) if selected['tree'] else ('M0_original',0.)
        metrics={**row['metrics'],'router':row['metrics'][name]}
        metrics['combined_selected']=metrics['router'] if selector['combined_selection']=='router' else metrics['selected']
        results.append({**row,'metrics':metrics,'route':name,'predicted_gain':gain})
    summary={}
    for ds in sorted({r['dataset'] for r in results}):
        use=[r for r in results if r['dataset']==ds];names=set.intersection(*(set(r['metrics']) for r in use))
        summary[ds]={'n':len(use),'methods':{name:{k:float(np.mean([r['metrics'][name][k] for r in use])) for k in ('note_f1','offset_f1','frame_f1','union_frame_f1','active_exact_match')} for name in sorted(names)}}
    save(path,{'selection':base['selection'],'router_selection':selector,'summary':summary,'tracks':results,
               'scope':base['scope']});print('ROUTER_TEST_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate','train_conservative','evaluate_conservative']);a=p.parse_args()
    if a.command.startswith('train'):train(a.command.endswith('conservative'))
    else:evaluate(a.command.endswith('conservative'))
