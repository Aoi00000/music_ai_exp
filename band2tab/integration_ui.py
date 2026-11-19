"""A/B integration UI layered beside the existing B score editor."""
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .integration import (ROOT,verify_extract,open_bundle,open_analysis,validate_review,select_view,apply_one_edit,
                          save_review,confirm_timing,engrave,difference,restore_review)
from .sync_player import player_html


def load_folder(folder,local=False):
    st.session_state.ab_document=open_analysis(folder) if local else open_bundle(folder)
    st.session_state.ab_folder=str(Path(folder).resolve())
    st.session_state.ab_revision=0
    st.session_state.ab_pending_view='baseline'


def render():
    last=ROOT/'results/ab_last_bundle.json'
    if 'ab_document' not in st.session_state and last.exists():
        try:load_folder(json.loads(last.read_text())['folder'])
        except Exception as error:st.warning('前回のセットを開けません: '+str(error))
    st.header('A/B結合確認：秒表示・比較・編集')
    st.caption('拍が未確認でも音符を表示できます。原音・音一覧・TAB・コードは同じ0秒を使います。')
    with st.expander('受け渡しZIPを開く',expanded='ab_document' not in st.session_state):
        upload=st.file_uploader('Aの受け渡しZIP',type=['zip'],key='ab_zip')
        if st.button('ZIPを検証して開く',disabled=upload is None):
            try:
                parent=ROOT/'inputs';parent.mkdir(exist_ok=True)
                temp=Path(tempfile.mkdtemp(prefix='ab_upload_',dir=parent))/'handoff.zip';temp.write_bytes(upload.getvalue())
                folder=verify_extract(temp,ROOT/'inputs');load_folder(folder);st.rerun()
            except Exception as error:st.error(str(error))
        folder_text=st.text_input('展開済みセットの場所',key='ab_folder_input')
        if st.button('展開済みセットを検証して開く',disabled=not folder_text):
            try:load_folder(folder_text);st.rerun()
            except Exception as error:st.error(str(error))
        saved_upload=st.file_uploader('保存した秒単位JSONを再読込',type=['json'],key='ab_saved')
        if st.button('保存した修正版を再読込',disabled=saved_upload is None):
            try:
                document=validate_review(json.loads(saved_upload.getvalue()))
                if 'ab_folder' not in st.session_state:raise ValueError('先に同じ音源セットを開いてください。')
                document=restore_review(document,st.session_state.ab_document)
                st.session_state.ab_document=document;st.session_state.ab_revision+=1;st.session_state.ab_pending_view=document.get('selected_view','baseline');st.rerun()
            except Exception as error:st.error(str(error))
    if 'ab_document' in st.session_state:
        document=st.session_state.ab_document;folder=Path(st.session_state.ab_folder)
        counts={p:len(ns) for p,ns in document['raw_parts'].items()}
        st.success('基準 '+ ' / '.join(f'{p}: {n}音' for p,n in counts.items())+f" / 計{sum(counts.values())}音")
        identity=document['raw_parts']==document['suggested_parts']
        st.info('補正案は基準と同一です（追加0・削除0）。改善を示す結果ではありません。' if identity else '補正案は比較用です。原音と確認して選択してください。')
        options=['baseline','suggested']+(['human'] if document.get('human_edit') else [])
        if 'ab_pending_view' in st.session_state:
            st.session_state.ab_view=st.session_state.pop('ab_pending_view')
        selected=st.radio('表示する音符',options,format_func=lambda v:{'baseline':'基準','suggested':'補正案（比較用）','human':'人の修正版'}[v],horizontal=True,key='ab_view')
        view=select_view(document,selected)
        chords_path=folder/'chords.json'
        chords=json.loads(chords_path.read_text()) if chords_path.exists() else {'segments':[],'duration':document.get('duration',12)}
        audio={}
        for label,path in [('原音',folder/'original.wav'),('原音',folder/'native.wav'),('ギターステム',folder/'stems/guitar.wav'),('ベースステム',folder/'stems/bass.wav')]:
            if path.is_file():audio[label]=path
        if audio:components.html(player_html(view,chords,audio),height=600,scrolling=True)
        st.caption('上のTABは秒時刻での運指候補です。未割当音はxで残します。拍・小節への量子化は行っていません。')
        with st.expander('コード進行（秒）'):
            st.dataframe(pd.DataFrame(chords.get('segments',[])),use_container_width=True)
            from chordchart.export import write_chart
            import hashlib
            chart=ROOT/'results/ab_charts'/hashlib.sha256(json.dumps(chords,sort_keys=True).encode()).hexdigest()[:16]
            chart.mkdir(parents=True,exist_ok=True)
            write_chart(chords,chart,title='A/B結合確認・コード予測')
            for name,label in [('chord_chart.html','印刷・PDF保存用HTML'),('chords.csv','コード進行CSV')]:
                st.download_button(label,(chart/name).read_bytes(),file_name=name)
        with st.expander('原音を聴いて確認した1行を修正する',expanded=False):
            part=st.selectbox('修正するパート',list(document['raw_parts']),key='ab_edit_part')
            edit_source=document.get('human_parts',document['raw_parts'])
            rows=[dict(_row=i,pitch=n['pitch'],onset=n['onset'],offset=n['offset'],velocity=n.get('velocity',80)) for i,n in enumerate(edit_source[part])]
            table=st.data_editor(pd.DataFrame(rows),num_rows='fixed',disabled=['_row'],key=f'ab_editor_{part}_{st.session_state.ab_revision}',use_container_width=True)
            listened=st.checkbox('原音またはステムを聴いて、この変更が必要だと確認した',key='ab_listened')
            reason=st.text_input('修正理由（聴こえた内容と時刻）',key='ab_reason')
            if st.button('確認した1行の修正を保存'):
                try:
                    edited=apply_one_edit(document,part,table.to_dict('records'),reason,listened)
                    out,report=save_review(edited,ROOT/'results/ab_review',chords)
                    st.session_state.ab_document=edited;st.session_state.ab_saved_path=str(out);st.session_state.ab_revision+=1;st.session_state.ab_pending_view='human'
                    st.success('1行の変更、他の音符とパートの保持、再読込を確認しました。')
                    st.json(report['edit_diff']);st.rerun()
                except Exception as error:st.error(str(error))
            if document.get('human_edit'):st.json(document['edit_diff'])
        if st.button('表示中の音符を秒単位JSONで保存・照合'):
            try:
                out,report=save_review(view,ROOT/'results/ab_review',chords)
                st.session_state.ab_saved_path=str(out);st.success(f"{report['notes']}音を保存・再読込しました。拍情報は未確認のまま保持できます。")
            except Exception as error:st.error(str(error))
        if st.session_state.get('ab_saved_path'):
            out=Path(st.session_state.ab_saved_path)
            st.write('保存先: '+str(out))
            st.download_button('秒単位の音符JSONをダウンロード',(out/'notes.json').read_bytes(),file_name='notes.json')
            st.download_button('保存検査レポート',(out/'verification.json').read_bytes(),file_name='verification.json')
        with st.expander('原音で拍を確認してからMusicXMLを作る'):
            st.caption('推定BPMやMIDIの120 BPMは採用しません。3点を確認できるまでは秒表示を使ってください。')
            bpm=st.number_input('確認したBPM',min_value=20.,max_value=400.,value=None,key='ab_bpm')
            meter=st.selectbox('確認した拍子',[3,4],index=None,format_func=lambda v:f'{v}/4',key='ab_meter')
            downbeat=st.number_input('確認した最初の小節頭（秒）',min_value=0.,value=None,key='ab_downbeat')
            confirmed=st.checkbox('BPM・拍子・小節頭の3点を原音で確認した',key='ab_timing_confirmed')
            grid=st.selectbox('記譜の分割',[4,3],format_func=lambda x:'16分音符' if x==4 else '8分3連符',key='ab_grid')
            if st.button('確認した拍設定でMusicXMLを保存',disabled=not (confirmed and bpm is not None and meter is not None and downbeat is not None)):
                try:
                    timed=confirm_timing(view,bpm,meter,downbeat,confirmed,grid)
                    out,summary=engrave(timed,ROOT/'results/ab_scores',chords)
                    st.session_state.ab_score_path=str(out);st.success(f"譜面保存: {summary['notes']}音 / 要確認{summary['issues']}件")
                    st.caption('音源が小節途中から始まる場合は、記譜用の前置休符を付けます。原音の秒時刻はJSON・MIDIに保持します。')
                except Exception as error:st.error(str(error))
            if st.session_state.get('ab_score_path'):
                out=Path(st.session_state.ab_score_path)
                for name in ['score.musicxml','notes.json','raw.mid','tab.txt']:
                    st.download_button(name,(out/name).read_bytes(),file_name=name,key='ab_score_'+name)
        with st.expander('参照ラベル（聴いた確認の補助）'):
            path=folder/'reference/notes.json'
            if path.exists():
                ref=json.loads(path.read_text());st.caption('参照ラベルをモデルの結果へ自動適用しません。')
                for part,notes in ref['parts'].items():st.write(part);st.dataframe(pd.DataFrame(notes),use_container_width=True)
    render_inference()


def render_inference():
    from .inference_job import launch,launch_aux
    with st.expander('新しい音声・動画を解析する（モデル準備後）'):
        upload=st.file_uploader('解析する音声・動画',type=['wav','mp3','m4a','flac','mp4','mov'],key='ab_infer_file')
        route=st.radio('認識経路',['direct','separated'],format_func=lambda v:'直接ミックス → MuScriptor（比較基準）' if v=='direct' else 'HTDemucs 6s → MuScriptor medium（Aの経路）',key='ab_route')
        model=st.selectbox('モデルサイズ',['small','medium'],index=1 if route=='separated' else 0,disabled=route=='separated',key='ab_model_'+route)
        st.caption('分離の効果を比較するときは、直接ミックス側もmedium・CPUにそろえてください。smallは別条件として記録します。')
        device=st.selectbox('計算場所',['cpu','mps'],disabled=route=='separated',key='ab_device_'+route)
        python=st.text_input('認識用Pythonの場所',str(ROOT/'.venv-mu-b/bin/python'),key='ab_python')
        st.caption('最初の12秒で実行します。分離経路はmedium / CPU固定。仮想環境はコピーせず、このMacで依存と必要な重みを準備します。')
        if st.button('この経路で12秒を解析',disabled=upload is None):
            try:
                parent=ROOT/'inputs';parent.mkdir(exist_ok=True)
                source=Path(tempfile.mkdtemp(prefix='audio_',dir=parent))/('input'+Path(upload.name).suffix.lower());source.write_bytes(upload.getvalue())
                job=launch(source,ROOT/'results/ab_jobs',route,python,model,'cpu' if route=='separated' else device)
                st.session_state.ab_job=str(job)
            except Exception as error:st.error(str(error))
        if st.session_state.get('ab_job'):
            path=Path(st.session_state.ab_job);job=json.loads(path.read_text())
            st.write('状態: '+job['status']+' / '+job.get('stage','開始準備'))
            if st.button('進捗・ログを更新'):st.rerun()
            log=path.parent/'transcribe.log'
            if log.exists():st.code(log.read_text(errors='replace')[-5000:])
            if job['status']=='failed':st.error(job.get('error','解析失敗'))
            if job['status'] in ('complete','failed'):
                st.json({k:job.get(k) for k in ['engine','model','device','dtype','elapsed_seconds','max_child_rss_bytes','free_before_bytes','free_after_bytes','memory_measurement','package_versions','git_commit','validated_weights','model_weight']})
            if job['status']=='complete' and job['engine']!='chords' and st.button('この解析結果を秒表示で開く'):
                load_folder(job['analysis'],local=True);st.rerun()
            if job['status']=='complete' and job['engine'] in ('direct','separated'):
                aux_python=st.text_input('補完・コード用Python',str(ROOT/'.venv-yourmt3/bin/python'),key='ab_aux_python')
                if job['engine']=='separated' and st.button('この結果から補完候補を別に作る（任意）'):
                    try:st.session_state.ab_job=str(launch_aux('addition',job['analysis'],ROOT/'results/ab_jobs',aux_python));st.rerun()
                    except Exception as error:st.error(str(error))
                if st.button('この原音のコードを別に解析する'):
                    try:st.session_state.ab_parent_analysis=job['analysis'];st.session_state.ab_job=str(launch_aux('chords',job['analysis'],ROOT/'results/ab_jobs',aux_python));st.rerun()
                    except Exception as error:st.error(str(error))
            if job['status']=='complete' and job['engine']=='chords':
                st.write('コード結果: '+job['analysis'])
                if st.button('同じ原音の音符結果とコードを開く'):
                    original=Path(st.session_state.ab_parent_analysis)
                    # Copy into a new result folder. Do not replace any previous analysis.
                    parent=ROOT/'results/ab_jobs';combined=Path(tempfile.mkdtemp(prefix='combined_',dir=parent))
                    shutil.copytree(original,combined/'analysis')
                    shutil.copy2(Path(job['analysis'])/'chords.json',combined/'analysis/chords.json')
                    load_folder(combined/'analysis',local=True);st.rerun()
