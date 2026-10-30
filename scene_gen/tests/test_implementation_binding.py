"""实现版本绑定与证据进展：开发者合成数据，不调用模型或正式实验。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_prep import implementation
from asset_prep.errors import StaleEvidence, ValidationError
from asset_prep.storage import Store
from asset_generation.sandbox import ExecutionBackend
from scene_generation.capabilities import REGISTRY
from scene_generation.progress import evidence_progress


class ManifestTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for package in implementation.PACKAGES:
            (self.root/'src'/package).mkdir(parents=True)
            (self.root/'src'/package/'__init__.py').write_text('', encoding='utf-8')
        (self.root/'src'/'asset_prep'/'alpha.py').write_text('A = 1\n', encoding='utf-8')
        (self.root/'src'/'scene_generation'/'beta.py').write_text('B = 1\n', encoding='utf-8')

    def test_manifest_keys_are_platform_independent_and_hash_every_source_file(self):
        manifest = implementation.source_manifest(self.root)
        self.assertEqual(sorted(manifest), ['asset_generation/__init__.py', 'asset_prep/__init__.py',
                                            'asset_prep/alpha.py', 'robot_operation/__init__.py', 'scene_generation/__init__.py',
                                            'scene_generation/beta.py'])
        # 反斜杠形式的历史清单必须归一到同一身份。
        legacy = {key.replace('/', '\\'): value for key, value in manifest.items()}
        self.assertEqual(implementation.implementation_digest(legacy),
                         implementation.implementation_digest(manifest))
        self.assertEqual(implementation.normalize_manifest({'src/asset_prep/alpha.py': 'x'}),
                         {'asset_prep/alpha.py': 'x'})

    def test_content_change_changes_identity(self):
        before = implementation.implementation_digest(implementation.source_manifest(self.root))
        (self.root/'src'/'asset_prep'/'alpha.py').write_text('A = 2\n', encoding='utf-8')
        after = implementation.implementation_digest(implementation.source_manifest(self.root))
        self.assertNotEqual(before, after)

    def test_verify_rejects_unbound_and_drifted_sources(self):
        identity = implementation.implementation_digest(implementation.source_manifest(self.root))
        self.assertEqual(implementation.verify(identity, base=self.root/'src'), identity)
        with self.assertRaises(ValidationError):
            implementation.verify(None, base=self.root/'src')
        (self.root/'src'/'asset_prep'/'alpha.py').write_text('A = 3\n', encoding='utf-8')
        with self.assertRaises(implementation.ImplementationDrift):
            implementation.verify(identity, base=self.root/'src')

    def test_missing_package_changes_identity_instead_of_making_an_empty_one(self):
        import shutil
        before = implementation.implementation_digest(implementation.source_manifest(self.root))
        shutil.rmtree(self.root/'src'/'scene_generation')
        after = implementation.implementation_digest(implementation.source_manifest(self.root))
        self.assertNotEqual(before, after)
        with self.assertRaises(implementation.ImplementationDrift):
            implementation.verify(before, base=self.root/'src')
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(empty, ignore_errors=True))
        for package in implementation.PACKAGES:
            (empty/'src'/package).mkdir(parents=True)
        with self.assertRaises(ValidationError):
            implementation.source_manifest(empty)

    def test_record_declares_what_each_identity_covers(self):
        record = implementation.runtime_record(self.root, inputs={'protocol': 'p'}, configuration={'limits': 1})
        self.assertEqual(record['implementation'],
                         implementation.implementation_digest(record['manifest']))
        self.assertEqual(record['inputs'], {'protocol': 'p'})
        self.assertEqual(record['configuration'], {'limits': 1})
        self.assertEqual(set(record['covers']), {'implementation', 'dependencies', 'inputs', 'configuration'})
        self.assertIn('字节码缓存', record['covers']['implementation'])


class BackendBindingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name)
        for package in implementation.PACKAGES:
            (self.project/'src'/package).mkdir(parents=True)
            (self.project/'src'/package/'__init__.py').write_text('', encoding='utf-8')
        self.backend = ExecutionBackend()
        self.backend.project = self.project
        self.backend.executable = None

    def test_job_identity_is_lazy_but_stable_and_explicitly_bindable(self):
        lazy = self.backend.job_identity()['implementation']
        self.assertEqual(self.backend.job_identity()['implementation'], lazy)
        self.backend.bind_implementation(lazy)
        self.assertEqual(self.backend.job_identity()['implementation'], lazy)
        (self.project/'src'/'asset_prep'/'__init__.py').write_text('X = 1\n', encoding='utf-8')
        # 已绑定后不随磁盘漂移；工作进程核对时才会拒绝。
        self.assertEqual(self.backend.job_identity()['implementation'], lazy)
        with self.assertRaises(implementation.ImplementationDrift):
            implementation.verify(lazy, base=self.project/'src', where='工作进程启动')

    def test_binding_rejects_missing_identity(self):
        with self.assertRaises(ValidationError):
            self.backend.bind_implementation(None)


class StoreBindingTests(unittest.TestCase):
    def test_resume_with_changed_implementation_requires_explicit_migration(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        config = type('C', (), {'max_model_calls': 1, 'max_search_calls': 0, 'max_image_downloads': 0,
                                'max_page_fetches': 0, 'max_reserved_tokens': 1})()
        base = {'catalogue': 'c', 'environment': {}, 'contract': {},
                'provider': 'p', 'model': 'm', 'base_url': 'u', 'output_token_parameter': 't',
                'generation': {}, 'limits': {}, 'robot_files': {}, 'backend': {}, 'provenance': 'x'}
        Store(root, {**base, 'implementation': 'a' * 64}, config)
        with self.assertRaises(StaleEvidence):
            Store(root, {**base, 'implementation': 'b' * 64}, config, resume=True)


class EvidenceProgressTests(unittest.TestCase):
    def previous(self):
        return {'status': 'partial', 'checks': [
            {'id': 'scene_collisions', 'status': 'partial', 'instances': ['a'],
             'coverage_gaps': [{'geom1': 'a__g', 'geom2': 'b__g'}]},
            {'id': 'support:a', 'status': 'pass', 'instances': ['a'], 'complete': True, 'coverage_gaps': []}],
            'recovery_failures': [{'id': 'f1', 'status': 'open', 'capability': 'c', 'kind': 'program',
                                   'remaining_checks': ['support:a']}]}

    def test_summary_change_alone_is_not_reported_as_resolved(self):
        previous = self.previous()
        updated = copy.deepcopy(previous)
        updated['checks'][0]['detail'] = '措辞改变但仍未知'
        progress = evidence_progress(previous, updated, REGISTRY['full_recheck_v2'],
                                     {'checks': ['scene_collisions']}, status='completed')
        self.assertFalse(progress['evidence_changed'])
        self.assertEqual(progress['remaining_gaps'], ['scene_collisions'])
        self.assertEqual(progress['resolved_gaps'], [])

    def test_real_gap_resolution_and_confirmed_defect_are_recorded(self):
        previous = self.previous()
        updated = copy.deepcopy(previous)
        updated['checks'][0].update(status='fail', coverage_gaps=[],
                                    collisions=[{'geom1': 'a__g', 'geom2': 'b__g', 'distance_m': -.002}])
        progress = evidence_progress(previous, updated, REGISTRY['collision_certificates_v1'],
                                     {'checks': ['scene_collisions']}, status='completed')
        self.assertTrue(progress['evidence_changed'])
        self.assertEqual(progress['resolved_gaps'], ['scene_collisions'])
        self.assertEqual(progress['confirmed_defects'], [{'check': 'scene_collisions', 'confirmed_defects': 1}])
        # 未被本动作覆盖的故障必须仍列在未解除集合中。
        self.assertEqual([f['fault'] for f in progress['undischarged_faults']], ['f1'])

    def test_failure_never_counts_as_progress(self):
        previous = self.previous()
        progress = evidence_progress(previous, previous, REGISTRY['full_recheck_v2'],
                                     {'checks': ['scene_collisions']}, status='failed',
                                     failure={'kind': 'program'})
        self.assertFalse(progress['evidence_changed'])
        self.assertEqual(progress['failure'], {'kind': 'program'})

    def test_local_result_does_not_claim_unrelated_checks(self):
        previous = self.previous()
        updated = copy.deepcopy(previous)
        updated['checks'][0].update(status='pass', coverage_gaps=[])
        progress = evidence_progress(previous, updated, REGISTRY['collision_certificates_v1'],
                                     {'checks': ['scene_collisions']}, status='completed')
        self.assertEqual(progress['requested_checks'], ['scene_collisions'])
        self.assertNotIn('support:a', progress['resolved_gaps'])
        self.assertNotIn('support:a', progress['remaining_gaps'])
