import json,tempfile,unittest
from pathlib import Path
from chordchart.export import write_chart


class ChordExportTests(unittest.TestCase):
    def test_rich_chords_and_raw_seconds_preserved(self):
        r=dict(duration=3.,segments=[dict(start=0.,end=1.25,label='C:maj7',needs_review=False),dict(start=1.25,end=3.,label='A:min7',needs_review=True)])
        with tempfile.TemporaryDirectory() as d:
            out=write_chart(r,d,title='<test>');self.assertEqual(json.load(open(Path(d)/'chords.json')),r)
            text=out.read_text();self.assertIn('Cmaj7',text);self.assertIn('Am7',text);self.assertIn('1.25',text);self.assertIn('&lt;test&gt;',text);self.assertNotIn('<test>',text)
    def test_mid_bar_changes_are_not_dropped(self):
        r=dict(duration=4.,segments=[dict(start=0.,end=1.25,label='C:maj'),dict(start=1.25,end=4.,label='G:7')])
        with tempfile.TemporaryDirectory() as d:
            text=write_chart(r,d,bpm=120,first_downbeat=0).read_text();self.assertIn('3.50拍',text);self.assertIn('G7',text);self.assertIn('2小節',text)
    def test_bar_alignment_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):write_chart(dict(segments=[]),d,bpm=120)


if __name__=='__main__':unittest.main()
