"""Local UI. Existing transcribers run as explicit external commands."""
from pathlib import Path
import json, subprocess, tempfile, shutil
import streamlit as st
import pandas as pd
from band2tab.audio import convert
from band2tab.score import read_midi,write_outputs,midi_timing,active_at

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Band2Tab',layout='wide')
st.title('Band2Tab')
st.caption('音声・動画からギター／ベースのTAB下書きを作成し、音を確かめながら修正します。')
upload=st.file_uploader('音声・動画ファイル',type=['wav','mp3','flac','m4a','mp4','mov','mid','midi'])
with st.expander('詳細設定',expanded=False):
    engines=['MuScriptor（ミックス音源）']
    if (ROOT/'.venv-bp/bin/basic-pitch').exists():engines.append('Basic Pitch（分離済みパート）')
    engines.append('MIDIを読み込む')
    engine=st.selectbox('採譜方法',engines)
    binary=st.text_input('採譜プログラムの場所',str(ROOT/'.venv-mu/bin/muscriptor') if engine.startswith('Mu') else str(ROOT/'.venv-bp/bin/basic-pitch'),key=engine)
    model=st.selectbox('MuScriptorモデル',['medium','small','large'])
    device=st.selectbox('計算する場所',['auto','mps','cpu','cuda'])
    part=st.selectbox('Basic Pitch / 単一パートMIDIの楽器',['guitar','bass'])
    force=st.checkbox('MIDI内の全音符をこの楽器として扱う',value=False)
    grid=st.selectbox('1拍の分割数',[4,3],format_func=lambda x:'16分音符' if x==4 else '8分3連符')
    chords_enabled=st.checkbox('コード名も解析する（初回は事前準備が必要）',value=True)
    chord_checkpoint=None
    if (ROOT/'models/chord_head_review.pt').exists():
        chord_method=st.selectbox('コード認識',['標準','追加学習した候補（比較用）'])
        if chord_method.startswith('追加'):chord_checkpoint=ROOT/'models/chord_head_review.pt'
    manual=st.checkbox('テンポを手動指定する（一定テンポの場合）')
    bpm=st.number_input('手動BPM',min_value=30.,max_value=300.,value=120.)
    meter=st.selectbox('拍子',[4,3])
if upload and st.button('解析する',type='primary'):
    try:
        run=Path(tempfile.mkdtemp(prefix='band2tab_'));src=run/('input'+Path(upload.name).suffix.lower());src.write_bytes(upload.getvalue())
        with st.spinner('解析中です。初回はモデルの取得に時間がかかります。'):
            if src.suffix.lower() in ('.mid','.midi'):midi=src
            else:
                audio=convert(src,run/'input.wav')
                if engine.startswith('Mu'):
                    midi=run/'raw.mid'
                    local_model=ROOT/'models'/f'muscriptor-{model}'/'model.safetensors'
                    model_source=str(local_model) if local_model.exists() else model
                    cmd=[binary,'transcribe',str(audio),'--model',model_source,'--device',device,
                         '--detect-tempo','true','--output',str(midi)]
                    if device=='cuda':cmd+=['--dtype','float16']
                elif engine.startswith('Basic'):
                    (run/'bp').mkdir();cmd=[binary,str(run/'bp'),str(audio)]
                else:raise ValueError('MIDIファイルを選択してください。')
                result=subprocess.run(cmd,text=True,capture_output=True)
                if result.returncode:raise RuntimeError(result.stderr[-4000:])
                if engine.startswith('Basic'):midi=next((run/'bp').glob('*.mid'))
            forced=part if force or engine.startswith('Basic') else None
            parts=read_midi(midi,forced)
            auto_bpm,auto_meter,beats=midi_timing(midi)
            if not manual:meter=auto_meter
            out=run/'score';write_outputs(parts,out,bpm if manual else auto_bpm,meter,grid,
                                        beats=None if manual else beats)
            shutil.copyfile(midi,out/'source.mid')
            meta=json.loads((out/'notes.json').read_text())
            meta['analysis_scope']='input_midi' if src.suffix.lower() in ('.mid','.midi') else 'whole_mix' if engine.startswith('Mu') and not forced else 'single_part'
            (out/'notes.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
            if chords_enabled and (run/'input.wav').exists():
                from band2tab.harmony import recognize,annotate_score
                ch=recognize(run/'input.wav',out/'chords.json',device='cuda' if device=='cuda' else 'cpu',checkpoint=chord_checkpoint)
                annotate_score(out/'score.musicxml',ch,json.loads((out/'notes.json').read_text()))
            st.session_state['run']=str(run);st.session_state['score']=json.loads((out/'notes.json').read_text())
        st.success('下書きを作成しました。原音と照合して確認してください。')
    except Exception as e:st.error(str(e))
if 'score' in st.session_state:
    j=st.session_state['score'];run=Path(st.session_state['run']);out=run/'score'
    if (run/'input.wav').exists():st.audio(str(run/'input.wav'))
    if j['issues']:st.warning(f"要確認: {len(j['issues'])}件。未割当・量子化衝突はJSONに残しています。")
    st.subheader('ギター＋ベース：時刻を指定して音を確認')
    if j.get('analysis_scope')=='single_part':st.info('この入力は単一パートです。ギターとベースを合わせて確認する場合は、原音をMuScriptorで解析してください。')
    inspection=dict(j['parts'])
    end=max((n['offset'] for ns in inspection.values() for n in ns),default=1.)
    when=st.slider('確認する時刻（秒）',min_value=0.,max_value=max(.01,float(end)),value=0.,step=.01)
    chords_data=json.loads((out/'chords.json').read_text()) if (out/'chords.json').exists() else None
    snap=active_at(inspection,when,chords_data)
    from band2tab.harmony import display
    st.write('コード予測：'+(display(snap['chord_prediction']) if snap['chord_prediction'] else '未解析'))
    st.write('ギター＋ベースで鳴っている音の推定：'+('・'.join(snap['notes_with_octave']) or '検出なし'))
    st.caption('音名のみ：'+('・'.join(snap['pitch_classes']) or '検出なし')+'。コード構成音への絞り込みは行いません。')
    if snap['active_notes']:
        st.dataframe(pd.DataFrame(snap['active_notes']).rename(columns={'part':'楽器','name':'音','fundamental_hz':'推定基音Hz','onset':'開始秒','offset':'終了秒'}),use_container_width=True)
    for label,part_key in [('ギター','guitar'),('ベース','bass')]:
        names=[n['name'] for n in snap['active_notes'] if n['part']==part_key]
        st.write(label+'：'+('・'.join(dict.fromkeys(names)) or '検出なし'))
    bass_notes=[n for n in snap['active_notes'] if n['part']=='bass']
    if bass_notes:st.caption('ベースの推定最低音：'+min(bass_notes,key=lambda n:n['pitch'])['name']+'。コードのルート確認の参考になります。分数コードや経過音の場合もあります。')
    st.caption('音・楽器・コード名は推定です。表の音名にオクターブを付け、低音と高音を区別しています。')
    for part,notes in j['parts'].items():
        st.subheader(part)
        if not notes:continue
        df=pd.DataFrame(notes)
        st.dataframe(df[['onset','offset','name','string','fret']],use_container_width=True)
        st.scatter_chart(df,x='onset',y='pitch')
    st.code((out/'tab.txt').read_text(),language=None)
    if (out/'chords.json').exists():
        from band2tab.harmony import display
        ch=json.loads((out/'chords.json').read_text())
        st.subheader('コード進行')
        st.dataframe(pd.DataFrame([{'開始':s['start'],'終了':s['end'],'コード':display(s.get('label',s.get('chord','N'))),'要確認':s.get('needs_review',True)} for s in ch['segments']]),use_container_width=True)
        from chordchart.export import write_chart
        chart_dir=out/'chord_chart';write_chart(ch,chart_dir,title='曲のコード譜')
        chord_cols=st.columns(2)
        chord_cols[0].download_button('コード譜を保存（印刷用HTML）',(chart_dir/'chord_chart.html').read_bytes(),file_name='chord_chart.html',mime='text/html')
        chord_cols[1].download_button('コード進行を保存（CSV）',(chart_dir/'chords.csv').read_bytes(),file_name='chords.csv',mime='text/csv')
        st.caption('HTMLをブラウザで開くと印刷・PDF保存できます。小節頭の確認前は秒で表示します。')
    cols=st.columns(4)
    for col,name in zip(cols,['score.musicxml','raw.mid','notes.json','tab.txt']):
        col.download_button(name,(out/name).read_bytes(),file_name=name)
    st.caption('音高・長さの修正は下の音符表で行えます。運指は再計算します。MusicXMLはMuseScoreで開いて確認できます。')
    rows=[]
    for part,notes in j['parts'].items():
        for n in notes:rows.append({'part':part,'pitch':n['pitch'],'onset':n['onset'],'offset':n['offset']})
    edited=st.data_editor(pd.DataFrame(rows),num_rows='dynamic',use_container_width=True)
    if st.button('修正を反映する'):
        try:
            parts={}
            for r in edited.to_dict('records'):
                if r['part'] not in ('guitar','bass'):raise ValueError('partはguitarまたはbassを指定してください')
                parts.setdefault(r['part'],[]).append({k:r[k] for k in ['pitch','onset','offset']})
            write_outputs(parts,out,j['bpm'],j['beats_per_bar'],j['grid'],j.get('tuning'),j.get('beat_times'))
            if (out/'chords.json').exists():
                from band2tab.harmony import annotate_score
                annotate_score(out/'score.musicxml',json.loads((out/'chords.json').read_text()),json.loads((out/'notes.json').read_text()))
            st.session_state['score']=json.loads((out/'notes.json').read_text());st.rerun()
        except Exception as e:st.error(str(e))
