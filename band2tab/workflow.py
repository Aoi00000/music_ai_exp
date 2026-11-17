"""Validated UI operations. No model downloads or inference at import time."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess

from .score import TUNINGS, clean_notes, quantize, read_midi, write_outputs


def validate_document(document):
    if not isinstance(document, dict) or not isinstance(document.get('parts'), dict):
        raise ValueError('parts を持つ音符JSONを指定してください。')
    result = deepcopy(document)
    parts = result['parts']
    if not parts or set(parts) - set(TUNINGS):
        raise ValueError('parts は guitar / bass のみに対応しています。')
    if not any(parts.values()):
        raise ValueError('音符がありません。')
    bpm = float(result.get('bpm', 120))
    meter = result.get('beats_per_bar', 4)
    grid = result.get('grid', 4)
    beats = result.get('beat_times')
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError('BPMは正の有限数を指定してください。')
    if beats is not None:
        if not isinstance(beats, list) or len(beats) < 2:
            raise ValueError('beat_times は2点以上必要です。')
        if any(not math.isfinite(float(b)) for b in beats):
            raise ValueError('beat_times に非有限値があります。')
        if any(float(b) <= float(a) for a, b in zip(beats, beats[1:])):
            raise ValueError('beat_times は厳密な昇順で指定してください。')
    tunings = result.get('tuning') or TUNINGS
    for part, notes in parts.items():
        if not isinstance(notes, list):
            raise ValueError('各パートの音符は配列にしてください。')
        tuning = tunings.get(part)
        if not isinstance(tuning, list) or len(tuning) != len(TUNINGS[part]):
            raise ValueError('標準6弦ギター / 4弦ベースのチューニングを指定してください。')
        if any(isinstance(p, bool) or not isinstance(p, int) or not 0 <= p <= 127 for p in tuning):
            raise ValueError('開放弦の音高は0〜127の整数にしてください。')
        for note in notes:
            if not isinstance(note, dict):
                raise ValueError('音符はオブジェクトで指定してください。')
            pitch = note.get('pitch')
            if isinstance(pitch, bool) or not isinstance(pitch, (int, float)) or not math.isfinite(pitch) or int(pitch) != pitch:
                raise ValueError('pitch は整数のMIDI音高を指定してください。')
            velocity = note.get('velocity', 80)
            if isinstance(velocity, bool) or not isinstance(velocity, (int, float)) or not math.isfinite(velocity) or int(velocity) != velocity or not 1 <= velocity <= 127:
                raise ValueError('velocity は1〜127の整数にしてください。')
        # Validate without discarding any original fields or reordering the editor rows.
        quantize(clean_notes(notes), bpm, meter, grid, beats=beats)
    result.update(bpm=bpm, beats_per_bar=meter, grid=grid)
    return result


def load_document(data):
    try:
        return validate_document(json.loads(data))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('音符JSONの必須項目・型を確認してください。') from exc


def editor_rows(document, part):
    return [dict(_row=i, pitch=n['pitch'], onset=n['onset'], offset=n['offset'],
                 velocity=n.get('velocity', 80)) for i, n in enumerate(document['parts'][part])]


def replace_part(document, part, rows):
    """Preserve metadata on surviving rows; invalidate acoustic scores after edits."""
    result = deepcopy(document)
    original = document['parts'][part]
    notes = []
    used = set()
    for row in rows:
        key = row.get('_row')
        existing = key is not None and not (isinstance(key, float) and math.isnan(key))
        if existing:
            if int(key) != key or not 0 <= int(key) < len(original) or int(key) in used:
                raise ValueError('編集行の識別子が不正です。再読み込みしてください。')
            key = int(key)
            used.add(key)
        note = deepcopy(original[key]) if existing else {}
        changed = not existing or any(note.get(k) != row[k] for k in ('pitch', 'onset', 'offset'))
        note.update({k: row[k] for k in ('pitch', 'onset', 'offset', 'velocity')})
        if changed:
            for field in ('acoustic_score', 'string', 'fret', 'q_start', 'q_end', 'review_reason'):
                note.pop(field, None)
            note.update(edited=True, needs_review=True)
        notes.append(note)
    result['parts'][part] = notes
    result.pop('issues', None)
    return validate_document(result)


def export_document(document, folder, chords=None):
    document = validate_document(document)
    summary = write_outputs(document['parts'], folder, document['bpm'],
                            document['beats_per_bar'], document['grid'],
                            document.get('tuning'), document.get('beat_times'))
    saved_path = Path(folder) / 'notes.json'
    saved = json.loads(saved_path.read_text(encoding='utf-8'))
    controlled = {'parts', 'issues', 'bpm', 'beats_per_bar', 'grid', 'tuning', 'beat_times', 'status'}
    saved.update({key: value for key, value in document.items() if key not in controlled})
    saved_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
    if chords:
        from .harmony import annotate_score
        saved = json.loads((Path(folder) / 'notes.json').read_text(encoding='utf-8'))
        annotate_score(Path(folder) / 'score.musicxml', chords, saved)
        (Path(folder) / 'chords.json').write_text(json.dumps(chords, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def transcribe_file(source, out, binary, model='medium', device='cpu', bpm=120., meter=4, grid=4):
    """Separate process/env for MuScriptor. No shell interpolation."""
    source, out, binary = Path(source), Path(out), Path(binary)
    if model not in ('small', 'medium') or device not in ('cpu', 'mps', 'auto'):
        raise ValueError('モデルまたはデバイスの指定が不正です。')
    if not binary.is_file():
        raise FileNotFoundError('MuScriptorがありません。通常の bash setup_mac.sh が必要です。')
    out.mkdir(parents=True, exist_ok=True)
    from .audio import convert
    wav = convert(source, out / 'input.wav')
    midi = out / 'transcription.mid'
    command = [str(binary.resolve()), 'transcribe', str(wav.resolve()), '--model', model,
               '--device', device, '--detect-tempo', 'false', '--output', str(midi.resolve())]
    with (out / 'transcribe.log').open('w', encoding='utf-8') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'採譜に失敗しました。ログ: {out / "transcribe.log"}')
    if not midi.is_file():
        raise RuntimeError('MuScriptorがMIDIを出力しませんでした。ログを確認してください。')
    return validate_document(dict(parts=read_midi(midi), bpm=bpm, beats_per_bar=meter, grid=grid))
