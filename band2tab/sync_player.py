"""One audio clock drives notes, unquantized TAB and chord intervals in an iframe."""
import base64
import json
from pathlib import Path
from .score import assign_frets


def player_html(document,chords,audio_files):
    payload=dict(parts={p:assign_frets(notes,p) for p,notes in document['parts'].items()},
                 chords=chords.get('segments',[]),duration=document.get('duration',12.),
                 audio={name:'data:audio/wav;base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii') for name,path in audio_files.items()})
    serialized=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="ja"><meta charset="utf-8"><style>
body{font:15px system-ui;color:#172638;margin:8px}audio{width:100%}select,input,button{font:inherit;margin:6px;padding:6px}input[type=range]{width:65%}h3{margin:10px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:5px;border-bottom:1px solid #dce3ea}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pane{background:#f2f6fa;padding:10px;border-radius:8px}pre{white-space:pre-wrap;font:14px monospace}#clock{font-weight:bold}#chord{font-weight:bold;color:#126a8c}.note{color:#5a6879;font-size:13px}
</style><label>聴く音源 <select id="track"></select></label><audio id="audio" controls preload="metadata"></audio>
<div><input id="seek" aria-label="再生時刻" type="range" min="0" step="0.01"><output id="clock"></output></div>
<label>秒へ移動 <input id="jump" type="number" min="0" step="0.01" value="0" style="width:90px"></label><button id="go">移動</button>
<div id="chord"></div><div class="note">音声プレイヤーの時刻で音一覧・TAB・コードが連動します。音源切替時も同じ秒を保ちます。拍・小節は未確認です。</div>
<div class="grid"><section class="pane"><h3>ギター</h3><div id="guitar"></div><pre id="guitar-tab"></pre></section><section class="pane"><h3>ベース</h3><div id="bass"></div><pre id="bass-tab"></pre></section></div>
<script>const D=PAYLOAD;const audio=document.getElementById('audio'), track=document.getElementById('track'), seek=document.getElementById('seek'),clock=document.getElementById('clock');
const names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];function name(p){return names[p%12]+(Math.floor(p/12)-1)}
for(const label of Object.keys(D.audio)){const o=document.createElement('option');o.textContent=label;o.value=label;track.append(o)}audio.src=D.audio[track.value];seek.max=D.duration;document.getElementById('jump').max=D.duration;
function render(){let t=audio.currentTime||0;seek.value=t;clock.value=t.toFixed(3)+' 秒';clock.dataset.time=String(t);const chord=D.chords.find(s=>s.start<=t&&t<s.end);document.getElementById('chord').textContent='コード予測: '+(chord?(chord.label||chord.chord):'区間外');
for(const p of ['guitar','bass']){const active=(D.parts[p]||[]).filter(n=>n.onset<=t&&t<n.offset), box=document.getElementById(p);box.replaceChildren();box.dataset.active=JSON.stringify(active.map(n=>[n.pitch,n.onset,n.offset]));
if(!active.length){box.textContent='検出なし'}else{const table=document.createElement('table');for(const n of active){const row=table.insertRow();for(const v of [name(n.pitch),(440*Math.pow(2,(n.pitch-69)/12)).toFixed(2)+' Hz',n.onset.toFixed(3)+'–'+n.offset.toFixed(3)+'秒']){row.insertCell().textContent=v}}box.append(table)}
const lines=Array.from({length:p==='guitar'?6:4},(_,i)=>String(i+1)+' | '+active.filter(n=>n.string===i+1).map(n=>n.fret+' ('+name(n.pitch)+')').join(' / '));const unassigned=active.filter(n=>n.string==null);if(unassigned.length)lines.push('x | '+unassigned.map(n=>name(n.pitch)+' 未割当').join(' / '));document.getElementById(p+'-tab').textContent=lines.join('\\n');}}
function jump(t){audio.currentTime=Math.max(0,Math.min(D.duration,Number(t)));render()}
seek.addEventListener('input',()=>jump(seek.value));document.getElementById('go').onclick=()=>jump(document.getElementById('jump').value);audio.addEventListener('timeupdate',render);audio.addEventListener('seeked',render);audio.addEventListener('loadedmetadata',render);track.onchange=()=>{const t=audio.currentTime,playing=!audio.paused;audio.src=D.audio[track.value];audio.addEventListener('loadedmetadata',()=>{jump(t);if(playing)audio.play().catch(()=>{})},{once:true})};function animate(){if(!audio.paused)render();requestAnimationFrame(animate)}render();animate();
</script></html>'''.replace('PAYLOAD',serialized)
