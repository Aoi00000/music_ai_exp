"""Deallocation must be attempted even after workload or backup failure."""
import json,subprocess,unittest
from types import SimpleNamespace
from unittest.mock import patch
from scripts import a_gpu_session


class GpuSessionTests(unittest.TestCase):
    def run_case(self,fail_at=None):
        calls=[];reports=[]
        def run(cmd,**kwargs):
            calls.append(cmd)
            if fail_at=='workload' and cmd==['true']:raise RuntimeError('workload failed')
            if fail_at=='backup' and cmd[:2]==['bash','scripts/a_backup.sh']:raise subprocess.CalledProcessError(1,cmd)
            if fail_at=='stop' and cmd[-1]=='deallocate':raise subprocess.CalledProcessError(1,cmd)
            return SimpleNamespace(returncode=0)
        with patch('sys.argv',['a_gpu_session.py','--','true']),patch.object(a_gpu_session.subprocess,'run',side_effect=run),patch.object(a_gpu_session.signal,'signal'),patch.object(a_gpu_session.Path,'write_text',side_effect=lambda text:reports.append(json.loads(text))):
            try:a_gpu_session.main()
            except (RuntimeError,SystemExit):pass
        return calls,reports

    def test_stop_after_workload_failure(self):
        calls,reports=self.run_case('workload');self.assertEqual(calls[-1][-1],'deallocate');self.assertTrue(reports[-1]['deallocated'])

    def test_stop_after_backup_failure(self):
        calls,reports=self.run_case('backup');self.assertEqual(calls[-1][-1],'deallocate');self.assertFalse(reports[-1]['backup_checkpoints_verified']);self.assertTrue(reports[-1]['deallocated'])

    def test_failed_stop_is_not_reported_successful(self):
        _,reports=self.run_case('stop');self.assertFalse(reports[-1]['deallocated']);self.assertIsNotNone(reports[-1]['deallocation_failure'])
