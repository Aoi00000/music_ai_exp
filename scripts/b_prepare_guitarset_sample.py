"""Fetch one official CC-BY-4.0 GuitarSet recording using bounded HTTP ranges."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RECORD = 'https://zenodo.org/records/3371780'
ATTRIBUTION = 'Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, Juan P. Bello. GuitarSet v1.1.0 (2019). DOI:10.5281/zenodo.3371780. CC BY 4.0.'


class RangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek mode')
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if not 0 <= position <= self.size:
            raise ValueError('Archive seek out of bounds')
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if not size:
            return b''
        if size > 8 * 1024**2 or self.transferred + size > 12 * 1024**2:
            raise ValueError('取得量の上限を超えるため停止します。全アーカイブは取得しません。')
        start, end = self.position, self.position + size - 1
        request = urllib.request.Request(
            self.url + f'&band2tab_range={start}-{end}',
            headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'Band2Tab-course-check/1.0'},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('サーバーが指定した部分取得に応じません。全体取得へ切り替えず停止します。')
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError('取得サイズが一致しません。')
        self.position += size
        self.transferred += size
        return data


def prepare(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError('空の出力フォルダを指定してください。以前の結果は上書きしません。')
    sources = json.loads((ROOT / 'examples/sources/guitarset_one_recording.json').read_text())
    transferred = 0
    for source in sources:
        reader = RangeReader(RECORD + '/files/' + source['archive'] + '?download=1', source['archive_size'])
        with zipfile.ZipFile(reader) as archive:
            member = archive.getinfo(source['member'])
            if member.file_size != source['bytes'] or f'{member.CRC:08x}' != source['crc32']:
                raise ValueError('公開アーカイブのメンバー情報が変更されています。')
            data = archive.read(member)
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise ValueError('取得ファイルのSHA-256が一致しません。')
        (folder / Path(source['member']).name).write_bytes(data)
        transferred += reader.transferred
        print('取得・ハッシュ確認OK:', source['member'], file=sys.stderr)
    audio = folder / '00_BN1-129-Eb_comp_mic.wav'
    jams = json.loads((folder / '00_BN1-129-Eb_comp.jams').read_text())
    duration = 12.
    with wave.open(str(audio), 'rb') as original:
        if original.getnchannels() != 1 or original.getsampwidth() != 2 or original.getframerate() != 44100:
            raise ValueError('想定したPCM音声形式と異なります。')
        frames = original.readframes(round(duration * original.getframerate()))
        with wave.open(str(folder / 'input.wav'), 'wb') as cropped:
            cropped.setparams(original.getparams())
            cropped.writeframes(frames)
    notes, tempos, beats, meters = [], [], [], set()
    for annotation in jams['annotations']:
        for event in annotation['data']:
            if annotation['namespace'] == 'note_midi':
                start, stop = event['time'], event['time'] + event['duration']
                if 0 <= start < min(duration, stop):
                    notes.append(dict(pitch=round(event['value']), onset=start, offset=min(duration, stop)))
            elif annotation['namespace'] == 'tempo':
                tempos.append(float(event['value']))
            elif annotation['namespace'] == 'beat_position':
                beats.append(float(event['time']))
                meters.add((event['value']['num_beats'], event['value']['beat_units']))
    if set(tempos) != {129.} or meters != {(4, 4)} or not notes or len(beats) < 2 or any(b <= a for a,b in zip(beats,beats[1:])):
        raise ValueError('注釈の音符・テンポ・拍子・拍列を確認できません。')
    metadata = dict(dataset='GuitarSet', recording_id='00_BN1-129-Eb_comp',
                    source=RECORD, license='CC-BY-4.0', attribution=ATTRIBUTION,
                    crop_start_seconds=0., crop_duration_seconds=duration,
                    bpm=129., beats_per_bar=4, grid=4, beat_times=beats,
                    timing_source='GuitarSet beat_position and tempo annotations',
                    synthetic=False, purpose='integration_check',
                    independent_unknown_test=False, parts={'guitar': sorted(notes,key=lambda n:(n['onset'],n['pitch']))})
    (folder / 'reference.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder / 'ATTRIBUTION.txt').write_text(ATTRIBUTION + '\n' + RECORD + '\nhttps://creativecommons.org/licenses/by/4.0/\nChange: first 12 seconds cropped from original mono microphone recording; note offsets clipped at 12 seconds.\n', encoding='utf-8')
    print(f'部分取得: {transferred / 1024**2:.2f} MiB / 音源12秒 / 注釈{len(notes)}音 / 注釈上129 BPM・4/4', file=sys.stderr)
    print('学習には使いません。既存データセットでの接続確認です。', file=sys.stderr)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    prepare(args.out)
    print('確認用音源:', Path(args.out).resolve() / 'input.wav')


if __name__ == '__main__':
    main()
