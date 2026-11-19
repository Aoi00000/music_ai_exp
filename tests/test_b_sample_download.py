"""Ensure a sample fetch cannot silently become a whole archive download."""
import unittest
from unittest.mock import patch, MagicMock
from scripts.b_prepare_guitarset_sample import RangeReader


class SampleDownload(unittest.TestCase):
    def test_server_ignoring_range_is_rejected_before_body_read(self):
        response=MagicMock()
        response.__enter__.return_value=response
        response.status=200
        with patch('urllib.request.urlopen',return_value=response):
            with self.assertRaisesRegex(ValueError,'部分取得'):
                RangeReader('https://example.org/file?download=1',100000000).read(4)
        response.read.assert_not_called()

    def test_large_read_stops_without_network_and_wrong_range_stops(self):
        with patch('urllib.request.urlopen') as request:
            with self.assertRaisesRegex(ValueError,'上限'):
                RangeReader('https://example.org/file?download=1',100000000).read()
            request.assert_not_called()
        response=MagicMock()
        response.__enter__.return_value=response
        response.status=206
        response.headers={'Content-Range':'bytes 4-7/100'}
        with patch('urllib.request.urlopen',return_value=response):
            with self.assertRaisesRegex(ValueError,'部分取得'):
                RangeReader('https://example.org/file?download=1',100).read(4)
        response.read.assert_not_called()
