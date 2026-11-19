"""Export and check B's known-note rhythm fixtures without acoustic models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from band2tab.workflow import load_document, export_document
from scripts.b_check_saved import check


def triples(notes):
    return sorted((n['pitch'], n['onset'], n['offset']) for n in notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open-scores', action='store_true', help='4つの確認用譜面をMacのMuseScore 4で開く')
    args = parser.parse_args()
    if args.open_scores and sys.platform != 'darwin':
        parser.error('--open-scores はMacで使用してください。')
    fixtures = sorted((ROOT/'examples/notation').glob('*.json'))
    if not fixtures:
        raise ValueError('確認用のJSONがありません。Gitの更新状態を確認してください。')
    parent = ROOT/'results';parent.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='b_notation_', dir=parent))
    summary = []
    for fixture in fixtures:
        document = load_document(fixture.read_text(encoding='utf-8'))
        folder = run/fixture.stem
        result = export_document(document, folder)
        saved = json.loads((folder/'notes.json').read_text(encoding='utf-8'))
        for part, notes in document['parts'].items():
            if triples(notes) != triples(saved['parts'][part]):
                raise ValueError(f'{fixture.stem}: 元の音高・秒時刻が変わりました。')
        check(folder/'notes.json')
        # Give each score a distinct title and filename in MuseScore tabs.
        tree = ET.parse(folder/'score.musicxml')
        tree.find('./work/work-title').text = document['title']
        ET.indent(tree, space='  ')
        score = run/(fixture.stem+'.musicxml')
        tree.write(score, encoding='utf-8', xml_declaration=True)
        entry = dict(name=fixture.stem, description=document['description_ja'],
                     notes=result['notes'], issues=result['issues'],
                     score=str(score), notes_json=str(folder/'notes.json'))
        summary.append(entry)
        print(f"  {fixture.stem}: {result['notes']}音 / issues={result['issues']}")
    (run/'summary.json').write_text(json.dumps(dict(scope='notation_only_known_notes',cases=summary),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS: {len(summary)}ケース / 元の音符保持・MIDI・運指・MusicXMLの音価と拍数を確認')
    print('確認記録:', run/'summary.json')
    if args.open_scores:
        subprocess.run(['open','-a','MuseScore 4',*[entry['score'] for entry in summary]],check=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print('STOP:', error, file=sys.stderr)
        sys.exit(1)
