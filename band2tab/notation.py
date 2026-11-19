"""Check the written rhythm, not only MusicXML duration integers."""
from collections import defaultdict
from fractions import Fraction
import xml.etree.ElementTree as ET

VALUES = {'whole': Fraction(4), 'half': Fraction(2), 'quarter': Fraction(1),
          'eighth': Fraction(1, 2), '16th': Fraction(1, 4), '32nd': Fraction(1, 8),
          '64th': Fraction(1, 16), '128th': Fraction(1, 32)}


def validate_musicxml(path):
    """Validate Band2Tab's explicit note values and complete per-voice measures.

    Optional MusicXML types are required here because MuseScore can round
    unspelled durations such as a 2.5-beat rest to a dotted half rest.
    """
    root = ET.parse(path).getroot()
    if root.tag != 'score-partwise' or root.get('version') != '4.0':
        raise ValueError('MusicXML 4.0の譜面ではありません。')
    total = 0
    for part in root.findall('part'):
        divisions = None; meter = None
        for measure in part.findall('measure'):
            attributes = measure.find('attributes')
            if attributes is not None:
                if attributes.find('divisions') is not None:
                    divisions = Fraction(attributes.findtext('divisions'))
                time = attributes.find('time')
                if time is not None:
                    meter = Fraction(time.findtext('beats')) * 4 / Fraction(time.findtext('beat-type'))
            if divisions is None or divisions <= 0 or meter is None:
                raise ValueError('MusicXMLの拍子またはdivisionsがありません。')
            sums = defaultdict(Fraction); previous = {}
            for note in measure.findall('note'):
                kind = note.findtext('type')
                if kind not in VALUES:
                    raise ValueError('MusicXMLに明示された音価がありません。--regenerate で再生成してください。')
                written = VALUES[kind] * sum((Fraction(1, 2 ** i) for i in range(len(note.findall('dot')) + 1)), Fraction())
                ratio = note.find('time-modification')
                if ratio is not None:
                    actual = int(ratio.findtext('actual-notes')); normal = int(ratio.findtext('normal-notes'))
                    if actual <= 0 or normal <= 0:
                        raise ValueError('連符の比率が不正です。')
                    written *= Fraction(normal, actual)
                duration = Fraction(note.findtext('duration')) / divisions
                if written != duration:
                    raise ValueError(f'MusicXMLの音価とdurationが一致しません: {written} / {duration}')
                voice = note.findtext('voice', '1')
                if note.find('chord') is None:
                    sums[voice] += written; previous[voice] = written
                elif previous.get(voice) != written:
                    raise ValueError('同時発音の音価が一致しません。')
            if not sums or len(sums) > 4:
                raise ValueError('MusicXMLの声部数がMuseScoreの対応範囲外です。')
            for voice, duration in sums.items():
                if duration != meter:
                    raise ValueError(f"小節の長さが不正です: {part.get('id')} / {measure.get('number')} / voice {voice}: {duration}拍、想定{meter}拍")
            total += 1
    if not total:
        raise ValueError('譜面に小節がありません。')
    return total
