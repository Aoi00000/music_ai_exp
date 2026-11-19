"""Inspect a synthetic smoke test; remove only provably duplicate conductor events."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tempfile

import mido

GLOBAL_META = {'set_tempo', 'time_signature', 'key_signature'}


def event_key(message, tick):
    return tick, tuple(sorted((k, v) for k, v in message.dict().items() if k != 'time'))


def absolute_events(track):
    tick = 0
    for message in track:
        tick += message.time
        yield tick, message


def remove_duplicate_metadata(data):
    """Keep ticks/programs/notes/controls unchanged; reject ambiguous global timing."""
    midi = mido.MidiFile(file=io.BytesIO(data))
    if midi.type not in (0, 1) or not midi.tracks:
        raise ValueError('MIDI type 0/1の有効なトラックが必要です。')
    conductor = {event_key(m, t) for t, m in absolute_events(midi.tracks[0]) if m.type in GLOBAL_META}
    removed = Counter()
    original_events = []
    for index, track in enumerate(midi.tracks):
        original_events.append([event_key(m, t) for t, m in absolute_events(track) if m.type not in GLOBAL_META])
        pending = 0
        retained = mido.MidiTrack()
        for tick, message in absolute_events(track):
            pending += message.time
            if index and message.type in GLOBAL_META:
                if event_key(message, tick) not in conductor:
                    raise ValueError(f'トラック{index}・tick {tick}の{message.type}は先頭トラックと一致しません。時刻を推測せず停止します。')
                removed[message.type] += 1
                continue
            retained.append(message.copy(time=pending))
            pending = 0
        if pending:
            retained.append(mido.MetaMessage('end_of_track', time=pending))
        midi.tracks[index] = retained
    for index, track in enumerate(midi.tracks):
        actual = [event_key(m, t) for t, m in absolute_events(track) if m.type not in GLOBAL_META]
        if actual != original_events[index]:
            raise ValueError('MIDIイベントの時刻・内容の保持に失敗しました。')
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue(), dict(removed)


def inspect(source, reference):
    import numpy as np
    import pretty_midi
    from mir_eval.transcription import match_notes

    source, reference = Path(source), Path(reference)
    original = source.read_bytes()
    normalized, removed = remove_duplicate_metadata(original)
    midi = pretty_midi.PrettyMIDI(io.BytesIO(normalized))
    truth = json.loads(reference.read_text(encoding='utf-8'))
    if truth.get('synthetic') is not True or truth.get('accuracy_benchmark') is not False:
        raise ValueError('この照合にはb_make_test_audio.pyで生成した人工音源のreference.jsonを指定してください。')
    ref = [n for notes in truth['parts'].values() for n in notes]
    est = [n for p in midi.instruments if not p.is_drum for n in p.notes]
    if not ref:
        raise ValueError('生成元の音符がありません。')
    if est:
        matches = match_notes(
            np.array([[n['onset'], n['offset']] for n in ref]),
            pretty_midi.note_number_to_hz(np.array([n['pitch'] for n in ref])),
            np.array([[n.start, n.end] for n in est]),
            pretty_midi.note_number_to_hz(np.array([n.pitch for n in est])),
            onset_tolerance=.05, pitch_tolerance=1., offset_ratio=None,
        )
    else:
        matches = []
    instruments = [dict(program=int(p.program), name='Drums' if p.is_drum else pretty_midi.program_to_instrument_name(p.program),
                        is_drum=bool(p.is_drum), notes=len(p.notes)) for p in midi.instruments]
    guitar_bass_count = sum(len(p.notes) for p in midi.instruments if not p.is_drum and 24 <= p.program <= 39)
    report = dict(synthetic=True, accuracy_benchmark=False,
                  source_sha256=hashlib.sha256(original).hexdigest(),
                  removed_duplicate_metadata=removed, source_notes=len(ref), estimated_pitched_notes=len(est),
                  instruments=instruments, guitar_bass_notes_by_model=guitar_bass_count,
                  matching=dict(onset_tolerance_seconds=.05, pitch_tolerance_cents=1.,
                                instrument_scored=False, offset_scored=False,
                                matched=len(matches), unmatched_estimates=len(est)-len(matches),
                                unmatched_reference=len(ref)-len(matches)),
                  matched_offset_absolute_error_seconds=[float(abs(ref[i]['offset']-est[j].end)) for i,j in matches])
    if source.read_bytes() != original:
        raise ValueError('検査中に入力MIDIが変更されました。再実行してください。')
    folder = Path(tempfile.mkdtemp(prefix='inspection_', dir=source.parent))
    (folder / 'normalized.mid').write_bytes(normalized)
    (folder / 'inspection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('生成元:',len(ref),'音 / 推定:',len(est),'音（ドラムを除く）')
    for p in instruments:
        print(f"  {p['name']}: {p['notes']}音")
    print('音高・開始±50msのみの照合:',len(matches),'組一致 / 推定側の未対応',len(est)-len(matches),'音 / 生成元側の未対応',len(ref)-len(matches),'音')
    print('モデルがギター/ベースと判定した音:',guitar_bass_count)
    print('先頭トラックと同時刻・同内容の重複メタ情報:',sum(removed.values()),'件を別コピーから除去')
    print('元MIDIの保持・他イベントのtick保持: OK')
    print('検査結果:',folder / 'inspection.json')
    print('読込み用コピー:',folder / 'normalized.mid')
    print('人工音源の診断です。実楽器の認識精度や楽器分離の評価ではありません。')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('midi')
    parser.add_argument('--reference', required=True)
    args = parser.parse_args()
    inspect(args.midi, args.reference)


if __name__ == '__main__':
    main()
