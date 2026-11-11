"""Synthetic offline migration risks; no provider calls."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import prepare_model_continuation as migration
from asset_prep.storage import atomic_json, file_hash
from asset_prep.errors import ValidationError, StaleEvidence


class ModelContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / 'runs/batches/source'
        self.run = self.project / 'runs/source_np001'
        self.run.mkdir(parents=True)
        self.provider = migration.batch.read(PROJECT / 'config/default.json')
        atomic_json(self.root / 'config/provider.json', self.provider)
        atomic_json(self.root / 'config/scene.json', {'max_model_calls': 400})
        atomic_json(self.run / 'records/00000.json', {'response': 'synthetic Astra asset'})
        atomic_json(self.run / 'stages/one/model.json', {'geometry': 'synthetic'})
        identity = {k:self.provider[k] for k in ('model','provider','base_url','output_token_parameter')}
        identity.update(implementation='same-code', binding={'model':self.provider['model']}, limits={'budget':400})
        self.state = dict(identity=identity, calls=[dict(status='done', file='records/00000.json',
                          sha256=file_hash(self.run/'records/00000.json'), requested_model=self.provider['model'], reserved_tokens=100)],
                          scene=dict(status='paused', jobs={}, stages=[dict(directory='stages/one', files=migration.snapshot(self.run/'stages/one'))]))
        atomic_json(self.run/'state.json', self.state)
        self.manifest = dict(batch='source', status='interrupted', budgets={'calls':400},
            frozen=migration.batch.hashes(list((self.root/'config').glob('*.json'))),
            entries=[dict(run='source_np001', protocol='NP-001', pdf='synthetic.pdf', package='synthetic.md', status='running'),
                     dict(run='source_np043', protocol='NP-043', pdf='synthetic.pdf', package='synthetic.md', status='queued')])
        atomic_json(self.root/'batch.json', self.manifest)
        self.addCleanup(patch.stopall)
        patch.object(migration.implementation, 'verify').start()
        patch.object(migration.implementation, 'from_identity', side_effect=lambda p,i: {'model':i['model']}).start()

    def prepare(self):
        return migration.prepare(self.project, 'source', 'sol', 'GPT-6.1-sol')

    def test_preserves_source_ledger_geometry_budget_and_records_model_boundary(self):
        before = migration.inventory(self.run)
        result = self.prepare()
        new = self.project/'runs/sol_np001'
        state = migration.batch.read(new/'state.json')
        self.assertEqual(migration.inventory(self.run), before)
        self.assertEqual(state['calls'], self.state['calls'])
        self.assertEqual(state['scene'], self.state['scene'])
        self.assertEqual(state['identity']['limits'], self.state['identity']['limits'])
        self.assertEqual(state['identity']['binding'], {'model':'GPT-6.1-sol'})
        self.assertEqual(state['model_history'][0]['previous_identity'], self.state['identity'])
        manifest = migration.batch.read(self.project/'runs/batches/sol/batch.json')
        migration.batch.verify(manifest, self.project)
        self.assertEqual(manifest['budgets'], self.manifest['budgets'])
        self.assertEqual(manifest['entries'][1]['status'], 'queued')
        self.assertEqual(result['model_calls_added'], 0)
        with self.assertRaises(ValidationError): self.prepare()

    def test_unknown_request_blocks_before_creating_target(self):
        self.state['calls'].append({'status':'pending'})
        atomic_json(self.run/'state.json', self.state)
        with self.assertRaises(ValidationError): self.prepare()
        self.assertFalse((self.project/'runs/batches/sol').exists())

    def test_explicit_unknown_preservation_keeps_source_and_budget(self):
        unknown = dict(status='pending', index=1, key='old-request', kind='model',
                       requested_model=self.provider['model'], reserved_tokens=4516356,
                       phase='scene_review', started_at=1)
        self.state['calls'].append(unknown)
        atomic_json(self.run/'state.json', self.state)
        before = migration.inventory(self.run)
        migration.prepare(self.project, 'source', 'sol', 'GPT-6.1-sol', preserve_unknown=True)
        target = migration.batch.read(self.project/'runs/sol_np001/state.json')
        self.assertEqual(migration.inventory(self.run), before)
        self.assertEqual(target['calls'][0], self.state['calls'][0])
        call = target['calls'][1]
        self.assertEqual(call['status'], 'unknown_not_replayed')
        self.assertTrue(call['outcome_unknown'])
        for key in ('key','index','phase','requested_model','reserved_tokens','started_at'):
            self.assertEqual(call[key], unknown[key])
        self.assertEqual(target['model_history'][-1]['unknown_calls_not_replayed'], [1])
        from asset_generation.model import request_summary
        self.assertEqual(request_summary([call])['unknown_attempts'], 1)
        from asset_prep.storage import Store
        ledger = Store.__new__(Store)
        ledger.state = {'calls':[call]}
        self.assertEqual(ledger.totals()['accounted_tokens'], unknown['reserved_tokens'])
        self.assertEqual(ledger.totals()['model_calls'], 1)
        self.assertFalse(any(c['status']=='pending' for c in target['calls']))

    def test_saved_response_is_not_abandoned_by_model_switch(self):
        self.state['calls'].append(dict(status='pending',index=1,key='saved',kind='model',
                                       requested_model=self.provider['model']))
        atomic_json(self.run/'state.json', self.state)
        atomic_json(self.run/'responses/00001.json', {'saved':'response'})
        with self.assertRaisesRegex(ValidationError, 'Recover saved response'):
            migration.prepare(self.project, 'source', 'sol', 'GPT-6.1-sol', preserve_unknown=True)
        self.assertFalse((self.project/'runs/batches/sol').exists())

    def test_unknown_flag_does_not_allow_reserved_or_unidentified_requests(self):
        for call in ({'status':'reserved'}, {'status':'pending'},
                     dict(status='pending',index=1,key='wrong',kind='model',requested_model='other')):
            with self.subTest(call=call):
                self.state['calls'] = [call]
                atomic_json(self.run/'state.json', self.state)
                with self.assertRaises(ValidationError):
                    migration.prepare(self.project, 'source', 'sol', 'GPT-6.1-sol', preserve_unknown=True)
                self.assertFalse((self.project/'runs/batches/sol').exists())

    def test_tampered_geometry_blocks(self):
        atomic_json(self.run/'stages/one/model.json', {'geometry':'changed'})
        with self.assertRaises(ValidationError): self.prepare()

    def test_frozen_configuration_tamper_blocks(self):
        atomic_json(self.root/'config/scene.json', {'max_model_calls':999})
        with self.assertRaises(ValueError): self.prepare()

    def test_pending_checkpoint_and_evidence_are_preserved(self):
        self.state['scene']['pending'] = dict(directory='stages/one', review_checkpoint=dict(
            files=migration.snapshot(self.run/'stages/one'), job_versions={}, kind='scene', record={'report':{'status':'pass'}}))
        atomic_json(self.run/'state.json', self.state)
        self.prepare()
        new = migration.batch.read(self.project/'runs/sol_np001/state.json')
        self.assertEqual(new['scene']['pending'], self.state['scene']['pending'])

    def test_checkpoint_dependency_mismatch_blocks(self):
        self.state['scene']['pending'] = dict(directory='stages/one', review_checkpoint=dict(
            files=migration.snapshot(self.run/'stages/one'), job_versions={'missing':'old'}))
        atomic_json(self.run/'state.json', self.state)
        with self.assertRaises(StaleEvidence): self.prepare()

    def test_user_paused_batch_retains_completed_entries_without_rerunning(self):
        completed=dict(run='earlier_np088',protocol='NP-088',pdf='done.pdf',package='done.md',status='completed',model_calls=22)
        self.manifest['status']='paused_user'
        self.manifest['entries'].insert(0,completed)
        atomic_json(self.root/'batch.json',self.manifest)
        self.prepare()
        target=migration.batch.read(self.project/'runs/batches/sol/batch.json')
        self.assertEqual(target['entries'][0],completed)
        self.assertFalse((self.project/'runs/sol_np088').exists())
        self.assertEqual(target['entries'][1]['run'],'sol_np001')

    def test_incomplete_author_workspace_is_preserved_for_normal_resume(self):
        self.state['scene']['pending']={'directory':'stages/one','kind':'asset'}
        atomic_json(self.run/'state.json',self.state)
        before=migration.inventory(self.run)
        self.prepare()
        target=migration.batch.read(self.project/'runs/sol_np001/state.json')
        self.assertEqual(target['scene']['pending'],self.state['scene']['pending'])
        self.assertEqual(migration.inventory(self.run),before)
        from scene_generation.checkpoint import recover_review_checkpoint
        from types import SimpleNamespace
        committed=[]
        probe=SimpleNamespace(root=self.project/'runs/sol_np001',state=target['scene'],
                              commit=lambda kind,record:committed.append((kind,record)))
        recover_review_checkpoint(probe)
        self.assertEqual(committed[0][0],'abandoned')

    def test_invalid_paths_and_running_batch_rejected(self):
        with self.assertRaises(ValidationError):
            migration.prepare(self.project, 'source', '../escape', 'GPT-6.1-sol')
        self.manifest['status']='running'
        atomic_json(self.root/'batch.json', self.manifest)
        with self.assertRaises(ValidationError): self.prepare()


if __name__ == '__main__': unittest.main()

