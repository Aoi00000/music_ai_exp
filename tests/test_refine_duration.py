import io,unittest
from unittest.mock import patch
import numpy as np
from band2tab.refine_duration import duration_targets,instrument_metrics
from band2tab.refine import proposed_parts

class DurationRefinement(unittest.TestCase):
    def note(self,onset=0.,offset=1.,**kwargs):return {'pitch':60,'onset':onset,'offset':offset,'baseline_class':1,'velocity':100}|kwargs
    def test_late_onset_does_not_make_a_sounding_note_entirely_wrong(self):
        result=duration_targets([self.note(.1,1.1)],{'guitar':[self.note(.2,1.2)],'bass':[]})
        self.assertAlmostEqual(float(result[0,1]),.72,places=5)
        self.assertGreater(result[0,1],result[0,0]);self.assertAlmostEqual(float(result.sum()),1.)
    def test_censored_note_has_occupancy_without_an_artificial_onset(self):
        result=duration_targets([self.note()],{'guitar':[self.note(0.,.8,censored_onset=True)]})
        self.assertAlmostEqual(float(result[0,1]),.64,places=5)
    def test_unison_preserves_a_valid_instrument(self):
        truth={'guitar':[self.note()],'bass':[self.note()]}
        result=duration_targets([self.note(),self.note(baseline_class=0)],truth,occupancy_weight=1.)
        self.assertTrue(np.allclose(result[0],[0,1,0]));self.assertTrue(np.allclose(result[1],[0,.5,.5]))
    def test_false_absent_instrument_is_counted_even_for_the_same_pitch(self):
        n=self.note();score=instrument_metrics({'guitar':[n]},{'guitar':[n],'bass':[n]},end=1.)
        self.assertAlmostEqual(score['note_f1'],2/3);self.assertAlmostEqual(score['frame_f1'],2/3)
        self.assertAlmostEqual(score['union_frame_f1'],1.)
    def test_removal_guard_keeps_original_notes(self):
        notes=[self.note()];parts,changes=proposed_parts(notes,np.array([[1.,0,0]]),.75,.2,allow_removal=False)
        self.assertEqual(len(parts['guitar']),1);self.assertFalse(changes)
    def test_full_output_metric_counts_hallucinations_after_the_recording(self):
        ref={'guitar':[self.note()]};est={'guitar':[self.note(),self.note(1.1,1.3,pitch=64)]}
        full=instrument_metrics(ref,est)
        self.assertAlmostEqual(full['note_f1'],2/3);self.assertLess(full['frame_f1'],1.)
        window=instrument_metrics(ref,est,end=1.)
        self.assertEqual(window['note_f1'],1.)
    def test_parallel_range_reads_preserve_bytes_and_seek_position(self):
        from band2tab.techs import RangeReader
        data=bytes(range(256))*(17*1024**2//256)
        class Response(io.BytesIO):
            def __init__(self,a,b):
                super().__init__(data[a:b+1]);self.status=206;self.headers={'Content-Range':f'bytes {a}-{b}/{len(data)}'}
        def open_range(request,timeout):
            a,b=map(int,request.get_header('Range').split('=')[1].split('-'));return Response(a,b)
        reader=RangeReader('https://example.invalid/archive.zip',len(data))
        with patch('band2tab.techs.urllib.request.urlopen',side_effect=open_range):
            self.assertEqual(reader.read(),data);self.assertEqual(reader.tell(),len(data))
            reader.seek(37);self.assertEqual(reader.read(123),data[37:160]);self.assertEqual(reader.tell(),160)

if __name__=='__main__':unittest.main()
