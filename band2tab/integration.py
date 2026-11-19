"""Seconds-first A/B review. No model imports, assumed tempo, or source-path access."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from .score import clean_notes, TUNINGS

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'band2tab_AB_integration_20261119/'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_extract(archive, destination):
    expected=json.loads((ROOT/'docs/measurements/B_HANDOFF_20261119.json').read_text())
    if digest(archive)!=expected['archive_sha256']:
        raise ValueError('ZIPのSHA256がAの記録と一致しません。')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or len(names)!=expected['members']:
            raise ValueError('ZIPの件数または重複が不正です。')
        sums=json.loads(z.read(PREFIX+'checksums.json'))
        if set(names)!={PREFIX+p for p in sums}|{PREFIX+'checksums.json'}:
            raise ValueError('ZIPに未記録のファイルがあります。')
        for name in names:
            if not name.startswith(PREFIX) or '..' in PurePosixPath(name).parts:
                raise ValueError('ZIPのパスが不正です。')
            if name!=PREFIX+'checksums.json' and hashlib.sha256(z.read(name)).hexdigest()!=sums[name[len(PREFIX):]]:
                raise ValueError('ZIP内のSHA256が一致しません: '+name)
        parent=Path(destination);parent.mkdir(parents=True,exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix='ab_bundle_',dir=parent))
        z.extractall(out)
    return out/PREFIX.rstrip('/')


def verify_directory(folder):
    folder=Path(folder).resolve()
    if digest(folder/'checksums.json')!='683b3bc43d073ae36bbf5f9677c0d8689571bb322b1aeaf3ea4dd49eae086286':
        raise ValueError('checksums.json 自体のハッシュが一致しません。')
    sums=json.loads((folder/'checksums.json').read_text())
    # The checksum list itself is authenticated by A's recorded ZIP digest on import.
    for rel,sha in sums.items():
        path=(folder/rel).resolve()
        if not path.is_relative_to(folder) or path.is_symlink() or digest(path)!=sha:
            raise ValueError('展開済みファイルの検証に失敗しました: '+rel)
    return sums


def validate_review(document):
    result=deepcopy(document)
    if not isinstance(result.get('parts'),dict) or not result['parts'] or set(result['parts'])-set(TUNINGS):
        raise ValueError('guitar / bass の音符が必要です。')
    if not any(result['parts'].values()):raise ValueError('音符がありません。')
    for notes in result['parts'].values():
        if not isinstance(notes,list):raise ValueError('音符は配列にしてください。')
        clean_notes(notes)
        for note in notes:
            velocity=note.get('velocity',80)
            if isinstance(velocity,bool) or not isinstance(velocity,(int,float)) or not math.isfinite(velocity) or int(velocity)!=velocity or not 1<=velocity<=127:
                raise ValueError('velocity は1〜127の整数にしてください。')
    # Intentionally do not insert bpm, meter, grid, or beat_times.
    return result


def open_bundle(folder):
    folder=Path(folder).resolve();sums=verify_directory(folder)
    baseline=json.loads((folder/'baseline/notes.json').read_text())
    review=json.loads((folder/'review/notes.json').read_text())
    if baseline['parts']!=review['parts'] or baseline['parts']!=review['raw_parts']:
        raise ValueError('基準と補正前の音符が一致しません。')
    suggested=review['suggested_parts']
    for part,count in [('guitar',90),('bass',134)]:
        if len(baseline['parts'].get(part,[]))!=count:raise ValueError('基準224音が一致しません。')
    document=deepcopy(baseline)
    document.update(raw_parts=deepcopy(baseline['parts']),suggested_parts=deepcopy(suggested),
                    source_baseline=deepcopy(baseline),source_review=deepcopy(review),
                    selected_view='baseline',human_edit=False,
                    timing=json.loads((folder/'timing.json').read_text()),
                    source_bundle_checksums=sums,source_bundle='band2tab_AB_integration_20261119',
                    source_bundle_root=str(folder),time_unit='seconds')
    document['bpm']=None;document['beats_per_bar']=None;document['first_downbeat_seconds']=None
    document['notation_timing_confirmed']=False
    return validate_review(document)


def select_view(document,view):
    if view not in ('baseline','suggested','human'):raise ValueError('表示の指定が不正です。')
    result=deepcopy(document)
    if view=='baseline':result['parts']=deepcopy(document['raw_parts'])
    elif view=='suggested':result['parts']=deepcopy(document['suggested_parts'])
    elif not document.get('human_edit'):raise ValueError('人の修正版はまだありません。')
    else:result['parts']=deepcopy(document['human_parts'])
    result['selected_view']=view
    return validate_review(result)


def difference(before,after):
    changes=[]
    if set(before)!=set(after):raise ValueError('今回の確認ではパートを増減できません。')
    for part,notes in before.items():
        if len(notes)!=len(after[part]):raise ValueError('今回の確認では音符数を変更せず、1行だけ修正してください。')
        for index,(a,b) in enumerate(zip(notes,after[part])):
            fields={k:dict(before=a.get(k,80 if k=='velocity' else None),after=b.get(k,80 if k=='velocity' else None))
                    for k in ('pitch','onset','offset','velocity') if a.get(k,80 if k=='velocity' else None)!=b.get(k,80 if k=='velocity' else None)}
            if fields:changes.append(dict(part=part,row=index,fields=fields))
    return changes


def apply_one_edit(document,part,rows,reason,audible_confirmed=False):
    if not audible_confirmed or not reason.strip():raise ValueError('原音を聴いた確認と修正理由を記入してください。')
    baseline=select_view(document,'baseline')
    if len(rows)!=len(baseline['parts'][part]):raise ValueError('音符数を変更できません。')
    edited=deepcopy(baseline)
    for index,row in enumerate(rows):
        for field in ('pitch','onset','offset','velocity'):
            value=row[field]
            edited['parts'][part][index][field]=int(value) if field in ('pitch','velocity') and float(value).is_integer() else value
    edited=validate_review(edited)
    changes=difference(baseline['parts'],edited['parts'])
    if len(changes)!=1:raise ValueError(f'今回の確認では1行だけ修正してください。変更は{len(changes)}行です。')
    edited.update(human_edit=True,selected_view='human',edit_reason=reason.strip(),
                  audible_confirmation=True,edit_diff=changes,human_parts=deepcopy(edited['parts']))
    return edited


def save_review(document,destination,chords):
    document=validate_review(document)
    parent=Path(destination);parent.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='review_',dir=parent))
    for name,data in [('notes.json',document),('baseline.json',document['source_baseline']),('suggested.json',dict(parts=document['suggested_parts'])),('chords.json',chords)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    saved=validate_review(json.loads((out/'notes.json').read_text()))
    if saved!=document:raise ValueError('再保存の読戻しが一致しません。')
    if document.get('human_edit') and len(difference(document['raw_parts'],document['human_parts']))!=1:
        raise ValueError('1行以外が変更されています。')
    report=dict(notes=sum(map(len,document['parts'].values())),
                baseline_counts={k:len(v) for k,v in document['raw_parts'].items()},
                suggestion_is_identity=document['raw_parts']==document['suggested_parts'],
                human_edit=bool(document.get('human_edit')),edit_diff=document.get('edit_diff',[]),
                timing_confirmed=bool(document.get('notation_timing_confirmed')),roundtrip_verified=True,
                source_bundle_checksums=document['source_bundle_checksums'])
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return out,report


def confirm_timing(document,bpm,meter,downbeat,confirmed=False,grid=4):
    if not confirmed or bpm is None or meter is None or downbeat is None:raise ValueError('BPM・拍子・小節頭の3点を原音で確認してください。')
    if not math.isfinite(bpm) or not 20<=bpm<=400 or meter not in (3,4) or not math.isfinite(downbeat) or downbeat<0 or grid not in (3,4):raise ValueError('拍設定が不正です。')
    result=deepcopy(document)
    result.update(bpm=float(bpm),beats_per_bar=meter,first_downbeat_seconds=float(downbeat),grid=grid,notation_timing_confirmed=True)
    return result


def engrave(document,destination,chords):
    if not document.get('notation_timing_confirmed'):raise ValueError('拍未確認のためMusicXMLはまだ保存できません。秒単位のJSONは保存できます。')
    from .workflow import export_document
    bpm=document['bpm'];meter=document['beats_per_bar'];downbeat=document['first_downbeat_seconds']
    beat=60/bpm;bar=beat*meter
    origin=downbeat-math.ceil(downbeat/bar)*bar
    if abs(origin)<1e-9:origin=0.
    duration=max(n['offset'] for ns in document['parts'].values() for n in ns)
    beats=[origin+i*beat for i in range(math.ceil((duration-origin)/beat)+meter+2)]
    notation=deepcopy(document);notation['beat_times']=beats
    notation['notation_preroll_seconds']=-origin
    notation['notation_tempo_source']='user_confirmed_fixed'
    notation['notation_tempo_confirmed']=True
    # Grid has an explicit notated preroll when audio begins inside a measure.
    parent=Path(destination);parent.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='score_',dir=parent))
    summary=export_document(notation,out,chords)
    if difference(document['parts'],json.loads((out/'notes.json').read_text())['parts']):raise ValueError('譜面化で原音時刻が変わりました。')
    return out,summary


def open_analysis(folder):
    """Open a new local job, never follow provenance paths embedded in its JSON."""
    folder=Path(folder).resolve()
    baseline=json.loads((folder/'notes.json').read_text())
    raw=baseline.get('raw_parts',baseline['parts'])
    suggested=baseline['parts'] if baseline.get('corrector_applied') else baseline.get('suggested_parts',raw)
    document=deepcopy(baseline)
    document.update(parts=deepcopy(raw),raw_parts=deepcopy(raw),suggested_parts=deepcopy(suggested),
        source_baseline=deepcopy(baseline),source_review={},source_bundle='local_inference',
        source_bundle_root=str(folder),source_bundle_checksums={'notes.json':digest(folder/'notes.json')},
        selected_view='baseline',human_edit=False,time_unit='seconds',
        timing={'bpm':{'value':None,'status':'unconfirmed'},'meter':{'numerator':None,'denominator':None,'status':'unconfirmed'},'first_downbeat_seconds':{'value':None,'status':'unconfirmed'}},
        bpm=None,beats_per_bar=None,first_downbeat_seconds=None,notation_timing_confirmed=False)
    return validate_review(document)


def restore_review(saved,source):
    saved=validate_review(saved)
    for key in ('source_bundle_checksums','raw_parts','suggested_parts','source_baseline','timing'):
        if saved.get(key)!=source.get(key):raise ValueError('保存JSONの元結果・提案・拍情報が音源セットと一致しません: '+key)
    selected=saved.get('selected_view')
    if selected not in ('baseline','suggested','human'):raise ValueError('保存JSONの表示指定が不正です。')
    if saved.get('human_edit'):
        if not saved.get('audible_confirmation') or not saved.get('edit_reason') or len(difference(saved['raw_parts'],saved['human_parts']))!=1:
            raise ValueError('保存JSONの聴取確認・1行修正を確認できません。')
    expected=saved['raw_parts'] if selected=='baseline' else saved['suggested_parts'] if selected=='suggested' else saved.get('human_parts')
    if saved['parts']!=expected:raise ValueError('保存JSONの表示音符と選択状態が一致しません。')
    if not saved.get('notation_timing_confirmed') and any(saved.get(k) is not None for k in ('bpm','beats_per_bar','first_downbeat_seconds')):
        raise ValueError('未確認の拍情報が埋まっています。')
    return saved
