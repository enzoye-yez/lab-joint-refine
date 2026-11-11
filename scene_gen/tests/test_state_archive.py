import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_prep.config import Config
from asset_prep.errors import StaleEvidence, UncertainRequest
from asset_prep.storage import Store, encoded
from asset_prep.state_archive import read_state, write_state, ArchivedRecord


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def state(self):
        return {'calls':[], 'scene':{'stages':[{'kind':'scene','directory':'stages/one',
            'files':{},'report':{'checks':[{'id':'collision','status':'fail','details':['measured']*1000}]}}]}}

    def test_lossless_lazy_read_dedup_and_immutable_history(self):
        value=self.state(); expected=encoded(value)
        write_state(self.root/'state.json',value)
        write_state(self.root/'state.json',value)
        self.assertEqual(len(list((self.root/'state_reports').glob('*'))),1)
        restored=read_state(self.root/'state.json')
        stage=restored['scene']['stages'][0]
        self.assertIsInstance(stage,ArchivedRecord)
        with patch('asset_prep.state_archive._load',side_effect=AssertionError('metadata must not load reports')):
            self.assertEqual(stage['directory'],'stages/one')
            self.assertIn('report',stage)
            write_state(self.root/'state.json',restored)
        report=stage['report']; report['checks'].clear()
        self.assertEqual(encoded(restored),expected)

    def test_corrupt_archive_cannot_reuse_a_cached_report(self):
        value=self.state();write_state(self.root/'state.json',value)
        stage=value['scene']['stages'][0]
        self.assertTrue(stage['report']['checks'])
        path=self.root/stage.reference['path']
        path.write_bytes(gzip.compress(b'{"report":{"changed":true}}'))
        with self.assertRaises(StaleEvidence):stage['report']

    def test_unknown_request_reservation_survives_archive_resume(self):
        store=Store(self.root,{},Config())
        store.state['scene']=self.state()['scene']
        store.reserve('unknown','model',12345)
        totals=store.totals()
        restored=Store(self.root,{},Config(),resume=True)
        self.assertEqual(restored.totals(),totals)
        self.assertEqual(restored.state['calls'][0]['reserved_tokens'],12345)
        with self.assertRaises(UncertainRequest):restored.cached('unknown')

