import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_prep.config import Config
from asset_prep.storage import Store, atomic_json
from asset_generation.model import GenerationModel
from asset_generation.contracts import GenerationConfig
from generation_fixtures import FakeTransport


class LocalPersistenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = Config()

    def test_transient_replace_recovers_and_old_file_survives_until_success(self):
        path = self.root/'state.json'
        atomic_json(path, {'old':True})
        original = os.replace
        attempts = []
        def replace(source, target):
            attempts.append(source)
            if len(attempts) <= 3:
                self.assertEqual(json.loads(path.read_text()), {'old':True})
                raise PermissionError(errno.EACCES, 'synthetic sharing violation')
            original(source, target)
        with patch('asset_prep.storage.os.replace', side_effect=replace), patch('asset_prep.storage.time.sleep'):
            atomic_json(path, {'new':True})
        self.assertEqual(len(attempts),4)
        self.assertEqual(json.loads(path.read_text()), {'new':True})
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_permanent_failure_is_bounded_preserves_old_and_cleans_temp(self):
        path = self.root/'state.json'
        atomic_json(path, {'old':True})
        with patch('asset_prep.storage.os.replace',side_effect=PermissionError(errno.EACCES,'blocked')) as replace, \
             patch('asset_prep.storage.time.sleep'), self.assertRaises(PermissionError):
            atomic_json(path, {'new':True})
        self.assertEqual(replace.call_count,10)
        self.assertEqual(json.loads(path.read_text()), {'old':True})
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_disk_full_is_not_retried_as_sharing_violation(self):
        with patch('asset_prep.storage.os.replace',side_effect=OSError(errno.ENOSPC,'full')) as replace, \
             patch('asset_prep.storage.time.sleep') as sleep, self.assertRaises(OSError):
            atomic_json(self.root/'state.json',{})
        self.assertEqual(replace.call_count,1)
        sleep.assert_not_called()

    def test_received_response_recovers_without_resending_after_ledger_failure(self):
        store = Store(self.root, {}, self.config)
        transport = FakeTransport([{'answer':1}])
        model = GenerationModel(self.config,GenerationConfig(),{'HARNESS_MODEL_API_KEY':'fixture-only'},store,transport)
        original = store.save
        def save():
            if store.state['calls'] and store.state['calls'][-1]['status']=='done':
                raise PermissionError(errno.EACCES,'synthetic state save failure')
            original()
        with patch.object(store,'save',side_effect=save), self.assertRaises(PermissionError):
            model.ask('synthetic',{},[],'test')
        self.assertEqual(len(transport.requests),1)
        recovered = Store(self.root, {}, self.config,resume=True)
        no_requests = FakeTransport([])
        retry = GenerationModel(self.config,GenerationConfig(),{'HARNESS_MODEL_API_KEY':'fixture-only'},recovered,no_requests)
        value,_ = retry.ask('synthetic',{},[],'test')
        self.assertEqual(value,{'answer':1})
        self.assertEqual(no_requests.requests,[])
        self.assertEqual(recovered.state['calls'][0]['status'],'done')

    def test_reserved_failure_is_not_unknown_and_never_sends(self):
        store = Store(self.root, {}, self.config)
        transport = FakeTransport([])
        model = GenerationModel(self.config,GenerationConfig(),{'HARNESS_MODEL_API_KEY':'fixture-only'},store,transport)
        original = store.save
        def save():
            if store.state['calls'] and store.state['calls'][-1]['status']=='pending':
                raise PermissionError(errno.EACCES,'synthetic pre-send failure')
            original()
        with patch.object(store,'save',side_effect=save), self.assertRaises(PermissionError):
            model.ask('synthetic',{},[],'test')
        self.assertEqual(transport.requests,[])
        recovered = Store(self.root,{},self.config,resume=True)
        self.assertEqual(recovered.state['calls'][0]['status'],'not_sent')
