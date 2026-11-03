"""开发者合成故障与隔离源码副本；不访问网络，不调用正式模型。"""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from asset_prep import implementation as impl
from asset_prep.config import Config
from asset_prep.errors import StaleEvidence, ValidationError
from asset_prep.errors import ProviderError
from asset_prep.storage import Store, atomic_json, digest
from asset_generation.contracts import GenerationConfig
from asset_generation.local import LocalBackend
from scene_generation.capabilities import REGISTRY
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.pipeline import SceneRun
from scene_generation.progress import evidence_progress
from generation_fixtures import FakeTransport, PROVENANCE
import test_scene_pipeline as fixture
import test_scene_continuation as migration
from scene_fixtures import SceneSandbox, plan

PROJECT = Path(__file__).resolve().parents[1]


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root/'project'
        for package in impl.PACKAGES:
            for source in (PROJECT/'src'/package).rglob('*.py'):
                target = self.project/'src'/source.relative_to(PROJECT/'src')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        shutil.copyfile(PROJECT/'src/runtime_bootstrap.py', self.project/'src/runtime_bootstrap.py')

    def launch(self, script, *options):
        path = self.root/'probe.py'
        path.write_text(script, encoding='utf-8')
        return subprocess.run([sys.executable, '-B', '-P', str(self.project/'src/runtime_bootstrap.py'),
                               *options, '--script', str(path)], capture_output=True, text=True, timeout=30)

    def module(self):
        path = self.project/'src/scene_generation/review_probe.py'
        path.write_text('VALUE = 0\n', encoding='utf-8')
        return path

    def test_valid_but_stale_bytecode_is_not_executed(self):
        path = self.module()
        stat = path.stat()
        py_compile.compile(str(path), doraise=True)
        path.write_text('VALUE = 1\n', encoding='utf-8')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        output = self.launch('from scene_generation.review_probe import VALUE\nassert VALUE == 1\n')
        self.assertEqual(output.returncode, 0, output.stderr)

    def test_worker_bootstrap_accepts_container_style_source_root(self):
        app = self.root/'app'
        shutil.copytree(self.project/'src',app)
        script = self.root/'container_probe.py'
        script.write_text('import asset_prep.implementation\n',encoding='utf-8')
        expected = impl.implementation_digest(impl.package_manifest(app))
        output = subprocess.run([sys.executable,'-B','-P',str(app/'runtime_bootstrap.py'),
            '--expected-code',expected,'--script',str(script)],capture_output=True,text=True,timeout=30)
        self.assertEqual(output.returncode,0,output.stderr)

    def test_late_import_uses_pinned_bytes_then_drift_refuses_receipt(self):
        path = self.module()
        receipt = self.root/'verification.json'
        output = self.launch('from pathlib import Path\n'
            f'Path({str(path)!r}).write_text("VALUE = 9\\n", encoding="utf-8")\n'
            'from scene_generation.review_probe import VALUE\nassert VALUE == 0\nprint("pinned-source-used")\n',
            '--receipt', str(receipt))
        self.assertEqual(output.returncode, 78, output.stderr)
        self.assertIn('pinned-source-used', output.stdout)
        self.assertEqual(json.loads(receipt.read_text(encoding='utf-8'))['status'],'failed')

    def test_expected_version_mismatch_refuses_before_entrypoint(self):
        marker = self.root/'executed'
        output = self.launch(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n', '--expected-code', '0'*64)
        self.assertEqual(output.returncode, 78)
        self.assertFalse(marker.exists())

    def test_local_drift_is_stale_evidence_not_asset_failure(self):
        backend = LocalBackend(GenerationConfig(), self.project)
        backend.bind_implementation(impl.implementation_digest(impl.source_manifest(self.project)))
        self.module()
        script = self.root/'model.py'
        script.write_text('def build(output): pass\n', encoding='utf-8')
        with self.assertRaises(impl.ImplementationDrift):
            backend.build_and_check(script, self.root, ['A01'], {})
        self.assertFalse((self.root/'local-build.log').exists())

    def test_worker_mid_execution_drift_is_not_copied_as_valid_output(self):
        backend = LocalBackend(GenerationConfig(), self.project)
        script = self.root/'model.py'
        target = self.project/'src/scene_generation/pipeline.py'
        script.write_text('from pathlib import Path\ndef build(output):\n'
            f'    p=Path({str(target)!r})\n    p.write_bytes(p.read_bytes()+b"\\n# synthetic drift\\n")\n'
            '    (output/"model.xml").write_text("<mujoco/>")\n',encoding='utf-8')
        output = self.root/'output'
        output.mkdir()
        with self.assertRaises(impl.ImplementationDrift):
            backend._run('build',[(script,'/input/model.py')],output)
        self.assertFalse((output/'model.xml').exists())

    def make_run(self, transport, *, root=None):
        sample = fixture.ScenePipelineTests()
        sample.setUp()
        self.addCleanup(sample.doCleanups)
        provider = Config.load(PROJECT/'config/default.json')
        return SceneRun(sample.catalog, sample.env, numerical_contract({}), provider,
            GenerationConfig(), SceneConfig(validation_mode='static'), {provider.api_key_env:'fixture-only'},
            root or self.root/'run', transport, SceneSandbox(), sample.root/'robot', self.project, provenance=PROVENANCE)

    def test_migrated_scene_rechecks_saved_geometry_without_regeneration(self):
        replies = [plan(), {'model_py':fixture.ASSET_SOURCE},
                   {'model_py':fixture.TOOL_SOURCE}, fixture.REVIEW, fixture.REVIEW, fixture.REVIEW]
        old = self.make_run(FakeTransport(replies), root=self.project/'runs/old')
        old.run()
        old.state['status'] = 'needs_review'
        old.store.save()
        before_file = self.root/'before.json'
        atomic_json(before_file, impl.source_manifest(self.project))
        code = self.project/'src/scene_generation/pipeline.py'
        code.write_bytes(code.read_bytes()+b'\n# synthetic migration version\n')
        migration.script.prepare(self.project, 'old', 'new', before_file, 'implementation_binding')
        class RecheckBackend(SceneSandbox):
            rechecks = 0
            def recheck_asset(inner, artifact, destination, asset_ids, profile):
                inner.rechecks += 1
                renders = destination/'checked/renders'
                renders.mkdir(parents=True)
                shutil.copyfile(artifact.parent/'renders/fixture.png', renders/'fixture.png')
                return {'status':'pass','checks':[{'id':'developer_recheck','status':'pass'}],
                        'manifest':json.loads((artifact/'interfaces.json').read_text(encoding='utf-8')),
                        'renders':['fixture.png'],'coverage':['developer fixture']}, renders
        backend = RecheckBackend()
        transport = FakeTransport([fixture.REVIEW, fixture.REVIEW, fixture.REVIEW])
        resumed = SceneRun(old.catalogue,old.env,old.contract,old.model.config,old.generation,old.limits,
            old.model.credentials,self.project/'runs/new',transport,backend,old.robot_root,self.project,
            resume=True,provenance=PROVENANCE)
        resumed.run()
        self.assertEqual(backend.rechecks,2)
        self.assertEqual(backend.build_calls,0)
        self.assertEqual(len(transport.requests),3)
        self.assertEqual(resumed.store.state['calls'][:6], old.store.state['calls'])
        self.assertEqual(len(resumed.state['rounds']),2)
        # 新复查图像属于审查阶段，仍须绑定原几何，且允许再升级一次。
        resumed.state['status']='needs_review'
        resumed.store.save()
        atomic_json(before_file,impl.source_manifest(self.project))
        code.write_bytes(code.read_bytes()+b'\n# second synthetic migration\n')
        migration.script.prepare(self.project,'new','newer',before_file,'implementation_binding')

    def test_persistent_recheck_error_never_falls_through_to_asset_author(self):
        replies = [plan(), {'model_py':fixture.ASSET_SOURCE},
                   {'model_py':fixture.TOOL_SOURCE}, fixture.REVIEW, fixture.REVIEW, fixture.REVIEW]
        transport=FakeTransport(replies)
        run=self.make_run(transport)
        run.run()
        class BrokenRecheck(SceneSandbox):
            rechecks=0
            def recheck_asset(inner,*args):
                inner.rechecks+=1
                return {'status':'error','checks':[],'error_kind':'check_execution'},None
        backend=BrokenRecheck()
        run.sandbox=backend
        run.state['jobs']['instrument'].update(dirty=True,status='review_pending',needs_implementation_recheck=True)
        job=next(j for j in run.state['plan']['jobs'] if j['id']=='instrument')
        self.assertFalse(run.generate_job(job))
        self.assertFalse(run.generate_job(job))
        self.assertTrue(run.state['jobs']['instrument']['needs_implementation_recheck'])
        self.assertEqual(backend.rechecks,2)
        self.assertEqual(backend.build_calls,0)
        self.assertEqual(len(transport.requests),6)

    def test_main_boundary_refuses_without_spending_model_budget(self):
        transport = FakeTransport([plan()])
        run = self.make_run(transport)
        self.module()
        with self.assertRaises(impl.ImplementationDrift):
            run.initialize()
        self.assertEqual(transport.requests, [])
        self.assertEqual(run.store.state['calls'], [])

    def test_formal_programmatic_entry_cannot_bypass_pinned_version(self):
        transport = FakeTransport([plan()])
        sample = self.make_run(transport)
        with self.assertRaises(impl.ImplementationDrift):
            SceneRun(sample.catalogue,sample.env,sample.contract,sample.model.config,sample.generation,sample.limits,
                sample.model.credentials,self.root/'formal',transport,SceneSandbox(),sample.robot_root,self.project,
                expected_implementation=sample.store.state['identity']['implementation'])
        self.assertFalse((self.root/'formal/state.json').exists())
        self.assertEqual(transport.requests,[])

    def test_transport_retry_rechecks_version_before_spending_again(self):
        transport = FakeTransport([plan()])
        run = self.make_run(transport)
        def failing_post(*args):
            self.module()
            raise ProviderError('synthetic temporary failure',diagnostics={'http_status':503})
        with patch.object(transport,'post',side_effect=failing_post) as post, \
             patch('asset_generation.model.time.sleep'), self.assertRaises(impl.ImplementationDrift):
            run.initialize()
        self.assertEqual(post.call_count,1)
        self.assertEqual(len(run.store.state['calls']),1)
        self.assertEqual(run.store.state['calls'][0]['status'],'failed')

    def test_response_is_saved_before_mid_request_drift_stops_consumption(self):
        transport = FakeTransport([plan()])
        run = self.make_run(transport)
        original = transport.post
        def post(*args):
            response = original(*args)
            self.module()
            return response
        with patch.object(transport, 'post', side_effect=post), self.assertRaises(impl.ImplementationDrift):
            run.initialize()
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(run.store.state['calls'][0]['status'], 'done')
        self.assertNotIn('plan', run.state)
        # 恢复同一源码后重用已归档响应，不追加一次模拟付费调用。
        (self.project/'src/scene_generation/review_probe.py').unlink()
        run.initialize()
        self.assertEqual(len(transport.requests), 1)
        self.assertIn('plan', run.state)

    def test_replay_drift_cannot_publish_a_passing_receipt(self):
        spec = importlib.util.spec_from_file_location('binding_replay', PROJECT/'scripts/check_scene_archive.py')
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        stage = self.root/'archive/stages/00001_scene'
        (stage/'scene/artifact').mkdir(parents=True)
        atomic_json(stage/'scene_job.json', {'limits':{}, 'contract':{}})
        def checking(*args, **kwargs):
            self.module()
            return {'status':'pass','checks':[{'id':'synthetic','status':'pass'}]}
        output = self.root/'replay'
        with patch.object(replay, 'PROJECT', self.project), patch.object(replay, 'scene_checks', checking), \
             patch.object(sys, 'argv', ['replay','--stage',str(stage),'--output',str(output)]), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(impl.ImplementationDrift):
            replay.main()
        self.assertFalse((output/'receipt.json').exists())


class MigrationBindingTests(unittest.TestCase):
    def archive_case(self):
        case=migration.SceneContinuationTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.state['identity']['limits']={'validation_mode':'static'}
        case.state['scene']['plan']={'jobs':[{'id':'fixture','role':'fixture'}]}
        case.state['scene']['jobs']['fixture'].update(status='assembly_ready',dirty=False)
        atomic_json(case.root/'state.json',case.state)
        return case

    def test_archive_revalidation_records_missing_code_proof_and_preserves_budget(self):
        case=self.archive_case()
        original=(case.root/'state.json').read_bytes()
        migration.script.prepare(case.project,'old','new',kind='archive_revalidation')
        target=case.project/'runs/new'
        state=json.loads((target/'state.json').read_text(encoding='utf-8'))
        receipt=json.loads((target/'stage_review_upgrade.json').read_text(encoding='utf-8'))
        self.assertIsNone(receipt['before_sources'])
        self.assertIsNone(receipt['changed_files'])
        self.assertFalse(receipt['legacy_source_verified'])
        self.assertTrue(state['scene']['jobs']['fixture']['needs_implementation_recheck'])
        self.assertTrue(state['scene']['jobs']['fixture']['dirty'])
        self.assertEqual(state['calls'],case.state['calls'])
        self.assertEqual(json.loads((target/'result.json').read_text(encoding='utf-8'))['budget']['model_calls'],24)
        self.assertEqual((case.root/'state.json').read_bytes(),original)
        Store(target,state['identity'],SimpleNamespace(),resume=True)

    def test_archive_revalidation_rejects_unknown_requests_and_changed_artifacts(self):
        case=self.archive_case()
        case.state['calls']=[{'status':'pending'}]
        atomic_json(case.root/'state.json',case.state)
        with self.assertRaisesRegex(ValidationError,'requests with unknown outcome'):
            migration.script.prepare(case.project,'old','unknown',kind='archive_revalidation')
        self.assertFalse((case.project/'runs/unknown').exists())
        case.state['calls']=[]
        atomic_json(case.root/'state.json',case.state)
        (case.root/'stages/00001_asset/artifact/model.xml').write_text('<mujoco model="changed"/>',encoding='utf-8')
        with self.assertRaisesRegex(ValidationError,'archived evidence digest'):
            migration.script.prepare(case.project,'old','changed',kind='archive_revalidation')
        self.assertFalse((case.project/'runs/changed').exists())

    def test_code_upgrade_does_not_silently_fall_back_to_archive_revalidation(self):
        case=self.archive_case()
        with self.assertRaisesRegex(ValidationError,'a source upgrade requires'):
            migration.script.prepare(case.project,'old','new',kind='implementation_binding')
        self.assertFalse((case.project/'runs/new').exists())

    def test_migration_rebinds_legacy_and_new_identity_and_preserves_history(self):
        for has_binding in (False, True):
            with self.subTest(has_binding=has_binding):
                case = migration.SceneContinuationTests()
                case.setUp()
                try:
                    case.state['identity']['limits'] = {'validation_mode':'static'}
                    case.state['scene']['check_recovery'] = {'old':{'actions':[{'key':'completed-action','attempts':2}]}}
                    if has_binding:
                        case.state['identity']['binding'] = {'implementation':'old'}
                    atomic_json(case.root/'state.json',case.state)
                    old_state = (case.root/'state.json').read_bytes()
                    migration.script.prepare(case.project, 'old', 'new', case.before, 'implementation_binding')
                    target = case.project/'runs/new'
                    from asset_prep.state_archive import read_state
                    state = read_state(target/'state.json')
                    expected = copy.deepcopy(state['identity'])
                    expected['binding'] = impl.from_identity(case.project, expected)
                    Store(target, expected, SimpleNamespace(), resume=True)
                    self.assertEqual(state['identity']['binding']['implementation'], state['identity']['implementation'])
                    self.assertTrue(state['scene']['jobs']['fixture']['needs_implementation_recheck'])
                    self.assertEqual(state['calls'], case.state['calls'])
                    self.assertEqual(state['scene']['check_recovery_history'][0],case.state['scene']['check_recovery'])
                    self.assertEqual((case.root/'state.json').read_bytes(), old_state)
                finally:
                    case.doCleanups()


class ProgressDeltaTests(unittest.TestCase):
    def test_partial_pair_resolution_is_progress_and_unrelated_gap_is_kept(self):
        gaps = [{'geom1':'a','geom2':'b'}, {'geom1':'c','geom2':'d'}]
        before = {'checks':[{'id':'scene_collisions','status':'partial','coverage_gaps':gaps}]}
        after = copy.deepcopy(before)
        after['checks'][0]['coverage_gaps'] = gaps[1:]
        result = {'scope':'pairs','covered_pairs':[['a','b']], 'checks':after['checks']}
        value = evidence_progress(before, after, REGISTRY['collision_certificates_v1'],
            {'checks':['scene_collisions'],'pairs':[['a','b']]}, status='completed', result=result)
        self.assertTrue(value['evidence_changed'])
        self.assertEqual(value['resolved_pairs'], [{'check':'scene_collisions','pairs':[['a','b']]}])
        self.assertEqual(value['remaining_gaps'], ['scene_collisions'])

    def test_old_localization_and_failed_action_never_count_as_new_progress(self):
        report = {'checks':[{'id':'collision','status':'fail',
                  'localization':{'matched':{'a__g':'a'},'unresolved':[]}}]}
        for status in ('completed','failed'):
            value = evidence_progress(report, copy.deepcopy(report), REGISTRY['locate_defects_v1'],
                {'checks':['collision']}, status=status)
            self.assertFalse(value['evidence_changed'])
        changed = copy.deepcopy(report)
        changed['checks'][0]['localization']['matched']['b__g'] = 'b'
        value = evidence_progress(report, changed, REGISTRY['locate_defects_v1'],
                                  {'checks':['collision']}, status='failed', failure={'kind':'program'})
        self.assertFalse(value['evidence_changed'])
