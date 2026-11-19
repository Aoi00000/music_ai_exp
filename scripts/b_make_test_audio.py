"""Create a repeatable synthetic smoke-test fixture, not an accuracy benchmark."""
import json
from pathlib import Path
import tempfile
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    document = json.loads((ROOT / 'examples/notes.json').read_text(encoding='utf-8'))
    rate, repetitions, cycle = 16000, 6, 2.0
    signal = np.zeros(round(rate * repetitions * cycle), dtype=np.float64)
    parts = {part: [] for part in document['parts']}
    for repeat in range(repetitions):
        for part, notes in document['parts'].items():
            for note in notes:
                onset = note['onset'] + cycle * repeat
                offset = note['offset'] + cycle * repeat
                parts[part].append(dict(note, onset=onset, offset=offset))
                start, stop = round(onset * rate), round(offset * rate)
                time = np.arange(stop - start, dtype=np.float64) / rate
                duration = (stop - start) / rate
                frequency = 440 * 2 ** ((note['pitch'] - 69) / 12)
                tone = np.zeros(len(time), dtype=np.float64)
                for harmonic in range(1, 9):
                    if frequency * harmonic < rate / 2:
                        tone += np.sin(2 * np.pi * frequency * harmonic * time) / harmonic ** 1.7
                envelope = np.minimum(time / .008, 1) * np.minimum((duration - time) / .025, 1)
                envelope *= np.exp(-time * (3 if part == 'guitar' else 1.5))
                signal[start:stop] += tone * envelope * (.5 if part == 'bass' else .3)
    signal *= .8 / max(1e-12, float(np.max(np.abs(signal))))
    parent = ROOT / 'results'
    parent.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='b_audio_fixture_', dir=parent))
    with wave.open(str(folder / 'input.wav'), 'wb') as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes((signal * 32767).astype('<i2').tobytes())
    document.update(parts=parts, synthetic=True, accuracy_benchmark=False,
                    title='12秒の人工音源：入出力の動作確認用。実楽器の認識精度の評価には使わない。',
                    duration_seconds=repetitions * cycle, sample_rate=rate,
                    note_count=sum(map(len, parts.values())))
    (folder / 'reference.json').write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    print('人工音源:', folder / 'input.wav')
    print('生成元の音符:', folder / 'reference.json')
    print('12秒 / 16 kHz / モノラル / 元の音符54音。認識結果の音数は未確認です。')
    print('この人工音色での結果は、実楽器の認識精度を示しません。')


if __name__ == '__main__':
    main()
