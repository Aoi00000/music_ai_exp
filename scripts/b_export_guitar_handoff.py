"""Export model notes using known recording annotations only for notation timing."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pretty_midi
from mir_eval.transcription import match_notes
from band2tab.notation import validate_musicxml
from band2tab.score import read_midi
from band2tab.workflow import export_document, prepare_handoff
from scripts.b_inspect_muscriptor_midi import remove_duplicate_metadata


def note_comparison(reference, estimated):
    ref = [n for notes in reference.values() for n in notes]
    est = [n for notes in estimated.values() for n in notes]
    if ref and est:
        matches = match_notes(
            np.array([[n['onset'], n['offset']] for n in ref]),
            pretty_midi.note_number_to_hz(np.array([n['pitch'] for n in ref])),
            np.array([[n['onset'], n['offset']] for n in est]),
            pretty_midi.note_number_to_hz(np.array([n['pitch'] for n in est])),
            onset_tolerance=.05, pitch_tolerance=1., offset_ratio=None,
        )
    else:
        matches = []
    return dict(reference_notes=len(ref), estimated_notes=len(est), matched=len(matches),
                unmatched_reference=len(ref)-len(matches), unmatched_estimates=len(est)-len(matches),
                onset_tolerance_seconds=.05, pitch_tolerance_cents=1.,
                instrument_scored=False, offset_scored=False,
                matched_offset_absolute_error_seconds=[float(abs(ref[i]['offset']-est[j]['offset'])) for i,j in matches])


def export_handoff(source, reference_path):
    source, reference_path = Path(source), Path(reference_path)
    original = source.read_bytes()
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    if reference.get('dataset') != 'GuitarSet' or reference.get('recording_id') != '00_BN1-129-Eb_comp':
        raise ValueError('b_prepare_guitarset_sample.pyで生成した実ギターのreference.jsonを指定してください。')
    if reference.get('bpm') != 129. or reference.get('beats_per_bar') != 4 or not reference.get('timing_source'):
        raise ValueError('音源に対応した注釈の129 BPM・4/4がありません。')
    normalized, removed = remove_duplicate_metadata(original)
    midi = pretty_midi.PrettyMIDI(io.BytesIO(normalized))
    parts = {}
    instruments = []
    for instrument in midi.instruments:
        instruments.append(dict(program=int(instrument.program), is_drum=bool(instrument.is_drum), notes=len(instrument.notes)))
        name = None if instrument.is_drum else 'guitar' if 24<=instrument.program<=31 else 'bass' if 32<=instrument.program<=39 else None
        if name:
            parts.setdefault(name, []).extend(dict(pitch=int(n.pitch), onset=float(n.start), offset=float(n.end),
                                                   velocity=int(n.velocity), confidence=None) for n in instrument.notes)
    if not any(parts.values()):
        raise ValueError('モデルがギター／ベースと判定した音がありません。楽器を自動で付け替えず停止します。')
    comparison = note_comparison(reference['parts'], parts)
    document = dict(parts=parts, bpm=reference['bpm'], beats_per_bar=reference['beats_per_bar'],
                    beat_times=reference['beat_times'], grid=4,
                    title='GuitarSet / MuScriptor small / 実録音の接続確認',
                    analysis_scope='single_part', source_engine='MuScriptor', source_model='small',
                    source_muscriptor_version='0.4.0', source_model_midi_sha256=hashlib.sha256(original).hexdigest(),
                    source_reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                    source_midi_instruments=instruments, midi_tempo_is_placeholder=True,
                    notation_tempo_confirmed=True, notation_tempo_source='dataset_annotation',
                    notation_timing_source=reference['timing_source'],
                    dataset=reference['dataset'], recording_id=reference['recording_id'],
                    attribution=reference.get('attribution'), license=reference.get('license'),
                    integration_check=True, independent_unknown_test=False,
                    notes_origin='model_prediction', corrector_applied=False)
    document = prepare_handoff(document)
    folder = Path(tempfile.mkdtemp(prefix='handoff_', dir=source.parent))
    (folder / 'source.mid').write_bytes(original)
    (folder / 'normalized_source.mid').write_bytes(normalized)
    summary = export_document(document, folder)
    saved = json.loads((folder / 'notes.json').read_text())
    triples = lambda ns: Counter((n['pitch'], n['onset'], n['offset']) for n in ns)
    if set(saved['parts']) != set(parts) or any(triples(saved['parts'][p]) != triples(ns) for p,ns in parts.items()):
        raise ValueError('モデルの音高・秒時刻の保持を確認できません。')
    midi_parts = read_midi(folder / 'raw.mid')
    written_midi = pretty_midi.PrettyMIDI(str(folder / 'raw.mid'))
    tolerance = 60 / reference['bpm'] / written_midi.resolution + 1e-7
    if set(midi_parts) != set(parts):
        raise ValueError('書出しMIDIの楽器が一致しません。')
    for part, notes in parts.items():
        expected = sorted((n['pitch'],n['onset'],n['offset']) for n in notes)
        actual = sorted((n['pitch'],n['onset'],n['offset']) for n in midi_parts[part])
        if len(expected) != len(actual) or any(a[0]!=b[0] or abs(a[1]-b[1])>tolerance or abs(a[2]-b[2])>tolerance for a,b in zip(expected,actual)):
            raise ValueError('書出しMIDIの音高・秒時刻が一致しません。')
    validate_musicxml(folder / 'score.musicxml')
    if source.read_bytes()!=original or reference_path.read_bytes()!=reference_bytes:
        raise ValueError('処理中に入力ファイルが変更されました。')
    report = dict(purpose='integration_check', independent_unknown_test=False, comparison=comparison,
                  export=summary, removed_duplicate_metadata=removed,
                  preserved_model_notes=True, exported_midi_verified=True, notation_timing_source=reference['timing_source'])
    (folder / 'handoff_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('モデルの音符を保持:',summary['notes'],'音 / 音高・生の秒時刻が一致')
    print('注釈との音高・開始±50ms照合:',comparison['matched'],'組一致 / 推定側未対応',comparison['unmatched_estimates'],'音 / 注釈側未対応',comparison['unmatched_reference'],'音')
    print('書出しMIDIの音高・秒時刻の読戻し: OK（1 MIDI tick以内）')
    print('記譜用テンポ: 元データ注釈の129 BPM・4/4・拍列')
    print('MusicXMLの音価・小節長検査: OK')
    print('未割当・量子化などの要確認:',summary['issues'],'件')
    print('編集画面で開くJSON:',folder/'notes.json')
    print('譜面:',folder/'score.musicxml')
    print('比較用の原音:',source.parent/'input.wav')
    print('これは接続確認です。音数や楽器判定の一致だけでは採譜精度を判断しません。')
    return folder, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--open-score',action='store_true')
    args = parser.parse_args()
    # Match the known 12-second crop; do not attach another recording's timing.
    with wave.open(str(args.run/'input.wav'),'rb') as audio:
        if abs(audio.getnframes()/audio.getframerate()-12)>1e-6:
            raise ValueError('確認用音源が12秒ではありません。')
    folder, _ = export_handoff(args.run/'raw.mid',args.run/'reference.json')
    if args.open_score:
        if sys.platform!='darwin':
            raise ValueError('--open-scoreはMacで使用してください。')
        subprocess.run(['open','-a','MuseScore 4',str(folder/'score.musicxml')],check=True)
        subprocess.run(['open','-R',str(folder/'notes.json')],check=True)


if __name__=='__main__':
    main()
