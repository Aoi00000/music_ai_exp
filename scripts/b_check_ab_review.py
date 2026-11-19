"""Read back a saved 224-note integration review and its one human edit."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from band2tab.integration import validate_review,difference,verify_directory


def check(path,require_edit=False):
    path=Path(path)
    document=validate_review(json.loads(path.read_text()))
    source=Path(document['source_bundle_root'])
    verify_directory(source)
    base=json.loads((source/'baseline/notes.json').read_text())
    proposal=json.loads((source/'review/notes.json').read_text())['suggested_parts']
    if document['raw_parts']!=base['parts'] or document['suggested_parts']!=proposal:raise ValueError('基準または補正案が変更されています。')
    if {p:len(ns) for p,ns in document['parts'].items()}!={'guitar':90,'bass':134}:raise ValueError('224音が一致しません。')
    changes=difference(base['parts'],document['parts'])
    if require_edit and (document.get('selected_view')!='human' or len(changes)!=1 or not document.get('audible_confirmation') or not document.get('edit_reason')):
        raise ValueError('人が聴いて確認した1行の修正がまだありません。')
    if document.get('notation_timing_confirmed') is not True:
        if any(document.get(k) is not None for k in ('bpm','beats_per_bar','first_downbeat_seconds')):raise ValueError('拍未確認なのに拍設定が埋まっています。')
    if document['timing']!=json.loads((source/'timing.json').read_text()):raise ValueError('Aの未確認拍情報が変更されています。')
    if json.loads((path.parent/'chords.json').read_text())!=json.loads((source/'chords.json').read_text()):raise ValueError('コード情報が変わっています。')
    print('PASS: 基準90+134=224音・補正案・原音時刻・拍情報・コードJSONを保持')
    print('人の修正:',len(changes),'行 / 聴取確認:',bool(document.get('audible_confirmation')))
    if changes:print(json.dumps(changes,ensure_ascii=False,indent=2))
    print('保存先:',path.parent)
    return changes


def main():
    p=argparse.ArgumentParser();p.add_argument('--notes',type=Path);p.add_argument('--require-edit',action='store_true');a=p.parse_args()
    paths=list((ROOT/'results/ab_review').glob('review_*/notes.json'))
    path=a.notes or max(paths,key=lambda p:p.stat().st_mtime_ns)
    check(path,a.require_edit)


if __name__=='__main__':main()
