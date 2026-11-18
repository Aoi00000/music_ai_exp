"""Read back the Mac editor's saved JSON, MIDI and tablature MusicXML."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUNINGS = {'guitar': [64, 59, 55, 50, 45, 40], 'bass': [43, 38, 33, 28]}


def check(notes_path, example_edit=False):
    folder = notes_path.parent
    document = json.loads(notes_path.read_text(encoding='utf-8'))
    parts = document['parts']
    if document['issues']:
        raise ValueError(f"要確認が{len(document['issues'])}件あります。保存ファイルのissuesを確認してください。")
    midi = pretty_midi.PrettyMIDI(str(folder / 'raw.mid'))
    tolerance = max(60 / float(bpm) / midi.resolution for bpm in midi.get_tempo_changes()[1]) + 1e-7
    midi_parts = {}
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        name = 'guitar' if 24 <= instrument.program <= 31 else 'bass' if 32 <= instrument.program <= 39 else None
        if name:
            midi_parts.setdefault(name, []).extend(instrument.notes)
    if set(midi_parts) != {name for name, notes in parts.items() if notes}:
        raise ValueError('JSONとMIDIの楽器が一致しません。')
    tuning = document.get('tuning') or DEFAULT_TUNINGS
    count = 0
    for part, notes in parts.items():
        expected = sorted((int(n['pitch']), float(n['onset']), float(n['offset'])) for n in notes)
        actual = sorted((n.pitch, n.start, n.end) for n in midi_parts.get(part, []))
        if len(expected) != len(actual):
            raise ValueError(part + ': JSONとMIDIの音符数が一致しません。')
        for a, b in zip(expected, actual):
            if a[0] != b[0] or abs(a[1] - b[1]) > tolerance or abs(a[2] - b[2]) > tolerance:
                raise ValueError(part + ': JSONとMIDIの音高・秒時刻が一致しません。')
        for index, note in enumerate(notes):
            string, fret = note.get('string'), note.get('fret')
            if string is None or fret is None or not 1 <= string <= len(tuning[part]):
                raise ValueError(part + ': 未割当または不正な弦があります。')
            if not 0 <= fret <= 24 or tuning[part][string - 1] + fret != note['pitch']:
                raise ValueError(part + ': 運指と音高が一致しません。')
            for other in notes[index + 1:]:
                if other.get('string') == string and min(note['offset'], other['offset']) - max(note['onset'], other['onset']) > 1e-7:
                    raise ValueError(part + ': 同時に同じ弦へ2音が割り当てられています。')
        count += len(notes)
    xml = ET.parse(folder / 'score.musicxml')
    if xml.getroot().tag != 'score-partwise' or xml.getroot().get('version') != '4.0':
        raise ValueError('MusicXML 4.0の譜面ではありません。')
    if example_edit:
        expected_document = json.loads((ROOT / 'examples/notes.json').read_text(encoding='utf-8'))
        expected_document['parts']['guitar'][0]['pitch'] = 53
        for part, reference in expected_document['parts'].items():
            triples = lambda ns: sorted((int(n['pitch']), float(n['onset']), float(n['offset'])) for n in ns)
            if triples(reference) != triples(parts.get(part, [])):
                raise ValueError('既知音符の52→53以外に違いがあります: ' + part)
        xml_pitches = []
        for note in xml.findall('.//note'):
            pitch = note.find('pitch')
            if pitch is None:
                continue
            step = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[pitch.findtext('step')]
            xml_pitches.append(12 * (int(pitch.findtext('octave')) + 1) + step + int(pitch.findtext('alter', '0')))
        if Counter(xml_pitches) != Counter(n['pitch'] for notes in parts.values() for n in notes):
            raise ValueError('既知音符とMusicXMLの音高・音符数が一致しません。')
        if len(xml.findall('.//technical/string')) != count:
            raise ValueError('MusicXMLのTAB音符数が一致しません。')
    print(f'PASS: {count}音 / JSONとMIDIの音高・秒時刻が一致 / 運指が音高と一致 / 弦の競合0 / issues=0')
    if example_edit:
        print('PASS: ギター52→53、ベース28を含む他の音符の保持、MusicXMLの全9音を確認')
    print('保存先:', folder)
    return folder / 'score.musicxml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes', type=Path, help='省略時はresults/ui内の最新の編集結果を検査')
    parser.add_argument('--example-edit', action='store_true', help='既知音符の最初のギター音52を53に変えた例を照合')
    parser.add_argument('--open-score', action='store_true', help='検査成功後にMacのMuseScore 4で開く')
    args = parser.parse_args()
    notes = args.notes
    if notes is None:
        candidates = [p for p in (ROOT / 'results/ui').glob('*/edited_*/notes.json')]
        if not candidates:
            raise ValueError('保存済みの編集結果がありません。画面で「修正を反映する」を押してください。')
        notes = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    score = check(notes, args.example_edit)
    if args.open_score:
        if sys.platform != 'darwin':
            raise ValueError('--open-score はMacで使用してください。')
        subprocess.run(['open', '-a', 'MuseScore 4', str(score.resolve())], check=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print('STOP:', error, file=sys.stderr)
        sys.exit(1)
