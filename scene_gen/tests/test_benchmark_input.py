"""Developer-authored offline fixtures; no provider or robot simulation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from asset_prep.errors import ValidationError
from asset_prep.storage import digest, file_hash
from asset_generation.contracts import GenerationConfig
from scene_generation.benchmark_input import load_public, bind_environment, PUBLIC_PATH
from scene_generation.inputs import catalog, task_bundle

class BenchmarkInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.task=self.root/'NP-001';self.task.mkdir()
        self.package=self.task/'asset_package.md'
        self.package.write_text('# Developer test\n### A01 | Test instrument\nDeclared dimensions only.\n',encoding='utf-8')
        self.profile={'role':'public_generation_constraints_not_scoring','robot_count':1,'version':'test-only',
            'robot':{'name':'Developer fixture','arms':[],'shared_joints':[]},'resources':{'model.xml':'placeholder'},
            'workspace_design':{'tables':'variable'},'assistance':'stock allowed','delivery_stages':'static then execution','initial_state':'declared'}
        self.profile_path=self.root/'_evaluation/generation_environment.json';self.profile_path.parent.mkdir()
        self.save()
    def save(self):
        self.profile_path.write_text(json.dumps(self.profile),encoding='utf-8')
        self.manifest={'files':[{'path':'asset_package.md','sha256':file_hash(self.package)}],
            'shared_files':[{'path':PUBLIC_PATH,'sha256':file_hash(self.profile_path),'role':'public_generation_constraints'}]}
        self.save_manifest()
    def save_manifest(self):
        (self.task/'input_manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')
    def test_load_scoped_public_data(self):
        p=load_public(self.package);self.assertNotIn('resources',p['summary']);self.assertIn('public_sha256',p['identity'])
    def test_propagates_to_plan_author_and_identity(self):
        c=catalog(self.package,GenerationConfig());b=task_bundle(c,{'asset_ids':['A01'],'id':'J1'})
        self.assertIn('Public benchmark environment',c['source'])
        self.assertEqual(b['source']['benchmark_environment']['robot_count'],1)
        self.assertIn('benchmark_public',c['identity'])
        self.assertEqual(c['bundles']['A01']['identity']['source_digest'],digest(c['bundles']['A01']['source']))
    def test_standalone_asset_input_is_bound(self):
        from asset_generation.inputs import load_package
        from scene_generation.benchmark_input import attach_bundle
        bundle=attach_bundle(load_package(self.package,['A01'],GenerationConfig()),load_public(self.package))
        self.assertEqual(bundle['source']['benchmark_environment']['robot_count'],1)
        self.assertEqual(bundle['identity']['source_digest'],digest(bundle['source']))
        self.assertIn('manifest_sha256',bundle['identity']['benchmark_public'])

    def test_windows_image_path_matches_portable_manifest(self):
        from PIL import Image
        image=self.task/'images/ref.png';image.parent.mkdir();Image.new('RGB',(3,3)).save(image)
        self.package.write_text('### A01 | Test\n![test](images/ref.png)\n',encoding='utf-8');self.save()
        self.manifest['files'].append({'path':'images/ref.png','sha256':file_hash(image)});self.save_manifest()
        c=catalog(self.package,GenerationConfig());self.assertEqual(len(c['bundles']['A01']['images']),1)
    def test_unlisted_picture_rejected(self):
        from PIL import Image
        image=self.task/'images/ref.png';image.parent.mkdir();Image.new('RGB',(3,3)).save(image)
        self.package.write_text('### A01 | Test\n![test](images/ref.png)\n',encoding='utf-8');self.save()
        with self.assertRaises(ValidationError):catalog(self.package,GenerationConfig())
    def test_malformed_manifest_returns_validation_error(self):
        path=self.task/'input_manifest.json'
        for value in ([], {'shared_files':[None]}, {'shared_files':self.manifest['shared_files'],'files':[None]}):
            path.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(ValidationError):load_public(self.package)

    def test_changed_profile_rejected(self):
        self.profile_path.write_text('{}',encoding='utf-8')
        with self.assertRaises(ValidationError):load_public(self.package)
    def test_input_change_rejected(self):
        self.package.write_text('changed',encoding='utf-8')
        with self.assertRaises(ValidationError):load_public(self.package)
    def test_cannot_inject_rubric_or_arbitrary_parent(self):
        for path in ('../_evaluation/evaluation_policy.json','../../secrets.json'):
            self.manifest['shared_files'][0]['path']=path;self.save_manifest()
            with self.assertRaises(ValidationError):load_public(self.package)
    def test_no_scoring_file_in_local_allowlist(self):
        self.manifest['files'].append({'path':'rubric.json','sha256':'x'});self.save_manifest()
        with self.assertRaises(ValidationError):load_public(self.package)
    def test_duplicate_shared_rejected(self):
        self.manifest['shared_files']*=2;self.save_manifest()
        with self.assertRaises(ValidationError):load_public(self.package)
    def test_archive_only_protocol_excluded(self):
        with self.assertRaises(ValidationError):load_public(self.package,self.task/'source.pdf')
    def test_legacy_without_manifest(self):
        (self.task/'input_manifest.json').unlink();self.assertIsNone(load_public(self.package))
    def test_declared_public_cannot_silently_disappear(self):
        self.package.write_text(PUBLIC_PATH,encoding='utf-8');(self.task/'input_manifest.json').unlink()
        with self.assertRaises(ValidationError):load_public(self.package)
    def test_public_change_changes_recovery_identity(self):
        a=catalog(self.package,GenerationConfig())['identity']
        self.profile['assistance']='changed authorized rule';self.save()
        b=catalog(self.package,GenerationConfig())['identity'];self.assertNotEqual(digest(a),digest(b))
    def test_robot_binding_rejects_spec_and_resource_drift(self):
        robot=self.root/'robot';robot.mkdir();(robot/'model.xml').write_text('developer synthetic',encoding='utf-8')
        self.profile['resources']={'model.xml':file_hash(robot/'model.xml')};self.save()
        c={'benchmark_public':load_public(self.package)};env={'table':{},'robot':self.profile['robot']}
        with patch('scene_generation.contracts.environment',side_effect=lambda x:x):
            bind_environment(c,env,robot)
            changed=copy.deepcopy(env);changed['robot']['name']='different'
            with self.assertRaises(ValidationError):bind_environment(c,changed,robot)
            (robot/'model.xml').write_text('mutated',encoding='utf-8')
            with self.assertRaises(ValidationError):bind_environment(c,env,robot)
    def test_resource_path_escape(self):
        self.profile['resources']={'../outside':'x'};self.save()
        with patch('scene_generation.contracts.environment',side_effect=lambda x:x):
            with self.assertRaises(ValidationError):bind_environment({'benchmark_public':load_public(self.package)}, {'table':{},'robot':self.profile['robot']},self.root/'robot')
    def test_batch_freezes_shared_profile_and_omits_archive_pdf(self):
        from test_scene_batch import BatchTests, batch
        case=BatchTests();case.setUp();self.addCleanup(case.doCleanups)
        public=case.inputs/'_evaluation/generation_environment.json';public.parent.mkdir();public.write_text('{}')
        for source in case.inputs.glob('NP-*'):(source/'input_manifest.json').write_text('{}')
        with patch('scene_generation.benchmark_input.load_public',return_value={'identity':{'files':{'asset_package.md':'placeholder'}}}), patch('scene_generation.benchmark_input.bind_environment'):
            manifest=batch.prepare(case.project,case.root,case.args)
        self.assertTrue(all(e['pdf'] is None for e in manifest['entries']))
        self.assertIn(str(public.resolve()),manifest['frozen'])
        batch.verify(manifest,case.project)
        public.write_text('{"changed":true}')
        with self.assertRaises(ValueError):batch.verify(manifest,case.project)

if __name__=='__main__':unittest.main()
