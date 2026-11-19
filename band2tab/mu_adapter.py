"""Supervised MuScriptor adaptation; codec mirrors its installed event decoder."""
import copy
import torch
from .lora import LoRALinear,adapter_state
SCHEMA='muscriptor_last8_attention_lora_v1'

def encode_window(tokenizer,notes,start,duration,program):
    from muscriptor.tokenizer.notes import Note,trim_overlapping_notes
    ns=trim_overlapping_notes([Note(False,program,float(n['onset']),float(n['offset']),int(n['pitch'])) for n in notes])
    end=start+duration;idx=tokenizer._token_index;active=sorted({n.pitch for n in ns if n.onset<start<n.offset});tokens=[];state_program=None
    if active:tokens.append(idx['program',program]);state_program=program;tokens.extend(idx['pitch',p] for p in active)
    tokens.append(idx['tie',0]);events=[];refs=[]
    for n in ns:
        if start<=n.onset<end:events.append((round((n.onset-start)*100),1,n.pitch))
        if start<n.offset<end and n.onset<end:events.append((round((n.offset-start)*100),0,n.pitch))
        if start<=n.onset<n.offset<end-.02:refs.append(dict(pitch=n.pitch,onset=n.onset-start,offset=n.offset-start,program=program))
    tick=0;velocity=None
    for t,v,p in sorted(events):
        if t>tick:tokens.append(idx['shift',t]);tick=t
        if state_program!=program:tokens.append(idx['program',program]);state_program=program
        if velocity!=v:tokens.append(idx['velocity',v]);velocity=v
        tokens.append(idx['pitch',p])
    tokens.append(tokenizer.eos_id)
    return tokens,refs

def roundtrip(tokenizer,tokens,refs):
    from muscriptor.events import decode_model_tokens,ChunkBoundary,NoteEndEvent
    events=list(decode_model_tokens(iter([ChunkBoundary(0.,None),*tokens[:-1]]),tokenizer._vocab,str,frame_rate=100));notes=[e for e in events if isinstance(e,NoteEndEvent)];used=set();errors=[]
    for n in refs:
        matches=[(j,e) for j,e in enumerate(notes) if j not in used and e.start_event.pitch==n['pitch'] and abs(e.start_event.start_time-n['onset'])<=.0051 and abs(e.end_time-n['offset'])<=.0051]
        if not matches:raise ValueError('Mu training token roundtrip lost or shifted a contained note')
        j,e=matches[0];used.add(j);errors.extend([abs(e.start_event.start_time-n['onset']),abs(e.end_time-n['offset'])])
    return dict(contained=len(refs),matched=len(used),max_error_seconds=max(errors,default=0.))

def inject_mu(model):
    for p in model.parameters():p.requires_grad_(False)
    adapters={}
    for i in range(16,24):
        layer=model.transformer.layers[i].self_attn
        module=LoRALinear(layer.out_proj,rank=8,alpha=16.,dropout=.05);layer.out_proj=module;adapters[f'transformer.layers.{i}.self_attn.out_proj']=module
    return adapters

def restore_mu(model,state):
    if state['schema']!=SCHEMA:raise ValueError('Wrong Mu adapter schema')
    adapters=inject_mu(model)
    if set(adapters)!=set(state['adapters']):raise ValueError('Mu adapter modules differ')
    with torch.no_grad():
        for name,m in adapters.items():m.a.copy_(state['adapters'][name]['a']);m.b.copy_(state['adapters'][name]['b'])
    return adapters
