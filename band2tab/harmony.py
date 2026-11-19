"""Reuse the already pinned chord inference port; never derive notes from chords."""
import json,re
from pathlib import Path
import xml.etree.ElementTree as ET
from .score import beat_position,NAMES

def recognize(audio,out,device='cpu',checkpoint=None):
    from chordchart.infer import infer
    result=infer(audio,checkpoint=checkpoint,baseline=checkpoint is None,device=device)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result

def display(label):
    if label=='N':return 'N'
    names={'maj':'','min':'m','maj7':'maj7','min7':'m7','minmaj7':'m(maj7)',
           'maj6':'6','min6':'m6','hdim7':'m7(b5)','dim7':'dim7','maj(9)':'add9'}
    if ':' not in label:return label
    root,qual=label.split(':',1);q,*bass=qual.split('/')
    suffix=''
    if bass:
        try:
            import mir_eval
            ri,_,bi=mir_eval.chord.encode(label,reduce_extended_chords=True)
            suffix='/'+NAMES[(ri+bi)%12]
        except Exception:suffix='/'+bass[0]
    return root+names.get(q,q)+suffix

def annotate_score(xml_path,chords,score_json):
    """Chord symbols on the first staff; raw times and labels remain in JSON."""
    import mir_eval
    root=ET.parse(xml_path);j=score_json;grid=j['grid'];meter=j['beats_per_bar'];beats=j.get('beat_times')
    measures=root.findall('./part')[0].findall('measure');seen=set()
    for seg in chords['segments']:
        label=seg['label'] if 'label' in seg else seg.get('chord','N')
        if label in ('N','X'):continue
        try:
            ri,_,bass=mir_eval.chord.encode(label,reduce_extended_chords=True)
            if ri<0:continue
        except Exception:continue
        b=beat_position(seg['start'],beats) if beats is not None else seg['start']*j['bpm']/60
        q=max(0,round(b*grid));mi=q//(meter*grid)
        if mi>=len(measures) or (mi,q,label) in seen:continue
        seen.add((mi,q,label));h=ET.Element('harmony');r=ET.SubElement(h,'root');name=NAMES[ri]
        ET.SubElement(r,'root-step').text=name[0]
        if len(name)>1:ET.SubElement(r,'root-alter').text='1'
        # Printed text preserves the rich source symbol; no invented simpler quality.
        quality=display(label)[len(label.split(':',1)[0]):].split('/')[0]
        ET.SubElement(h,'kind',{'text':quality}).text='other'
        if bass>0:
            ba=ET.SubElement(h,'bass');bn=NAMES[(ri+bass)%12];ET.SubElement(ba,'bass-step').text=bn[0]
            if len(bn)>1:ET.SubElement(ba,'bass-alter').text='1'
        ET.SubElement(h,'offset').text=str((q%(meter*grid))*(12//grid))
        m=measures[mi];idx=next((i for i,e in enumerate(m) if e.tag=='note'),len(m));m.insert(idx,h)
    ET.indent(root,space='  ');root.write(xml_path,encoding='utf-8',xml_declaration=True)
