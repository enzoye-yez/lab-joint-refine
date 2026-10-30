"""Archived NP-002 failures plus explicitly developer-authored corrections; offline only."""
import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from asset_generation.check_report import checked_report, validate_report
from asset_generation.checks import load_model, validate_manifest, validate_artifact
from asset_generation.contracts import GenerationConfig, check_profile, author_result
from asset_generation.local import LocalBackend
from asset_generation.model import GenerationModel
from asset_generation.pipeline import snapshot
from asset_prep.config import Config
from asset_prep.errors import ValidationError, BudgetExceeded, StaleEvidence
from asset_prep.storage import Store, digest, file_hash
from scene_generation.candidate import inspect_candidate, assembly_manifest, usable
from scene_generation.scene_first import collect, publish
from scene_generation.inputs import catalog
import test_scene_first as workflow
from scene_fixtures import plan

PROJECT=Path(__file__).resolve().parents[1]
CORPUS=PROJECT/'tests/data/np002_1027'
LOADABLE=('A01','A02','A08','A13','A15')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class ArchivedEvidenceTests(unittest.TestCase):
    def test_vendor_unnamed_geometries_have_traceable_names_without_physics_changes(self):
        import numpy as np
        from scene_generation.assembly import component
        root=PROJECT/'data/robots/ffw_sg2'
        # Use the same model selection as the archived run.
        state=read(PROJECT/'runs/np002_tatu_sol_1027_01_np002/state.json')
        from scene_generation.robot_contract import model_file
        filename=model_file(state['scene']['plan'],state['identity']['environment'])
        before=snapshot(root);original=load_model(root,model_file=filename)
        mapping={};derived=component(root,'robot__',filename,diagnostic_names=mapping).compile()
        self.assertTrue(mapping)
        self.assertTrue(all(derived.geom(g).name for g in range(derived.ngeom)))
        for field in ('geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_contype',
                      'geom_conaffinity','body_mass','body_inertia','jnt_range','actuator_ctrlrange'):
            np.testing.assert_array_equal(getattr(original,field),getattr(derived,field),err_msg=field)
        self.assertEqual(snapshot(root),before)

    def test_archived_corpus_is_hash_bound(self):
        for relative,record in read(CORPUS/'provenance.json')['files'].items():
            self.assertEqual(file_hash(CORPUS/relative),record['sha256'],relative)

    def test_five_real_models_admit_without_certifying_invalid_interfaces(self):
        for asset in LOADABLE:
            with self.subTest(asset=asset):
                root=CORPUS/asset/'artifact';before=snapshot(root)
                model=load_model(root);original=read(root/'interfaces.json')
                with self.assertRaises(ValidationError):validate_manifest(original,model,[asset])
                report=inspect_candidate(root,[asset],{'validation_mode':'static'})
                validate_report(report,domain='asset',mode='static')
                self.assertEqual(report['status'],'fail')
                self.assertTrue(report['assembly_readiness']['loadable'])
                self.assertFalse(report['assembly_readiness']['interfaces_valid'])
                self.assertFalse(report['assembly_readiness']['interaction_verified'])
                self.assertEqual(report['manifest'],original)
                projected,issues=assembly_manifest(original,model,[asset])
                self.assertTrue(issues)
                validate_manifest(projected,model,[asset])
                strict=validate_artifact(root,[asset],check_profile({'validation_mode':'static'}))
                self.assertEqual(strict['status'],'fail')
                self.assertEqual(snapshot(root),before)
                with self.assertRaises(ValidationError):
                    assembly_manifest(original,model,[asset],diagnostic=False)

    def test_a01_producer_is_fixed_consumer_stays_strict(self):
        raw=read(CORPUS/'A01/producer_report.json')
        self.assertTrue(raw['assembly_readiness']['loadable'])
        self.assertEqual(checked_report(raw,domain='asset',mode='static')['status'],'error')
        fresh=inspect_candidate(CORPUS/'A01/artifact',['A01'],{})
        self.assertEqual(checked_report(fresh,domain='asset',mode='static')['status'],'fail')
        self.assertTrue(any('机械事件' in r['detail'] for r in fresh['checks']))

    def test_geometry_reference_and_scope_remain_mandatory(self):
        root=CORPUS/'A02/artifact';model=load_model(root);mf=read(root/'interfaces.json')
        for field,value in [('body','missing_body'),('site','missing_site'),('asset_id','A99')]:
            bad=deepcopy(mf);bad['interfaces'][0][field]=value
            with self.assertRaises(ValidationError):assembly_manifest(bad,model,['A02'])
        with self.assertRaises(ValidationError):assembly_manifest(mf,model,['A99'])
        for interfaces in (None,{},[{'id':[]} ]):
            bad=deepcopy(mf);bad['interfaces']=interfaces
            with self.assertRaises(ValidationError):assembly_manifest(bad,model,['A02'])

    def test_real_compile_failures_remain_author_defects(self):
        for asset,reason in [('A03','axis'),('A05','torus'),('A07','cone'),('A12','range'),('A14','cone')]:
            with self.subTest(asset=asset),self.assertRaisesRegex((ValueError,ValidationError),reason):
                inspect_candidate(CORPUS/asset/'artifact',[asset],{})

    def test_artifact_mutation_invalidates_admission(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shutil.copytree(CORPUS/'A02/artifact',root/'artifact')
            report=inspect_candidate(root/'artifact',['A02'],{})
            record={'artifact':'artifact','assembly_readiness':report['assembly_readiness']}
            run=SimpleNamespace(root=root)
            self.assertTrue(usable(run,record))
            p=root/'artifact/interfaces.json';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaises(StaleEvidence):usable(run,record)


class SavedResponseTests(unittest.TestCase):
    def test_a16_response_consumes_once_and_keeps_actual_usage_over_request(self):
        response=read(CORPUS/'A16/response.json')
        class Archived:
            calls=0
            def post(inner,*args):
                inner.calls+=1
                self.assertEqual(inner.calls,1,'saved response must never be requested again')
                return deepcopy(response)
        with tempfile.TemporaryDirectory() as tmp:
            config=Config(max_reserved_tokens=70000);store=Store(Path(tmp),{'fixture':'A16'},config)
            transport=Archived();model=GenerationModel(config,GenerationConfig(),{'HARNESS_MODEL_API_KEY':'fixture'},store,transport)
            value,call=model.ask('offline archive replay',{},[],'asset:A16')
            source=author_result(value,None,500000)
            self.assertIn('def build(',source)
            self.assertEqual(call['usage_observation']['reported_completion_tokens'],23195)
            self.assertEqual(call['usage']['completion_tokens_details']['reasoning_tokens'],9840)
            self.assertEqual(store.totals()['reported_total_tokens'],78109)
            self.assertEqual(store.totals()['accounted_tokens'],78109)
            restored=Store(Path(tmp),{'fixture':'A16'},config,resume=True)
            model=GenerationModel(config,GenerationConfig(),{'HARNESS_MODEL_API_KEY':'fixture'},restored,transport)
            self.assertEqual(model.ask('offline archive replay',{},[],'asset:A16')[0],value)
            self.assertEqual(len(restored.state['calls']),1)
            self.assertEqual(len(restored.state['calls'][0]['validation_events']),1)
            with self.assertRaises(BudgetExceeded):model.ask('next paid request',{},[],'asset:A17')
            self.assertEqual(transport.calls,1)
            saved=Path(tmp)/restored.state['calls'][0]['file'];saved.write_bytes(saved.read_bytes()+b' ')
            with self.assertRaises(StaleEvidence):model.ask('offline archive replay',{},[],'asset:A16')

    def test_truncated_overlimit_response_is_never_executed_or_refetched(self):
        response=read(CORPUS/'A16/response.json');response['choices'][0]['finish_reason']='length'
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp),{},Config());transport=unittest.mock.Mock()
            transport.post.return_value=response
            model=GenerationModel(Config(),GenerationConfig(),{'HARNESS_MODEL_API_KEY':'fixture'},store,transport)
            for _ in range(2):
                with self.assertRaisesRegex(ValidationError,'截断'):model.ask('replay',{},[],'asset:A16')
            self.assertEqual(transport.post.call_count,1)


class RepairLoopTests(unittest.TestCase):
    setUp=workflow.SceneFirstTests.setUp
    make=workflow.SceneFirstTests.make

    def test_real_nameerror_feedback_repair_recheck_and_partial_scene(self):
        # The original author response is replayed; the correction is developer
        # authored, never counted as autonomous model repair success.
        source=(CORPUS/'A04/model.py').read_text(encoding='utf-8')
        old='"irreversible": false';self.assertEqual(source.count(old),1)
        correction={'edits':[{'old':old,'new':'"irreversible": False'}]}
        self.gen=replace(self.gen,max_iterations=3)
        src=self.root/'a04.md';src.write_text('### A04｜培养皿\n原失败脚本离线回放\n',encoding='utf-8')
        self.cat=catalog(src,self.gen)
        p=json.loads(json.dumps(plan()).replace('A01','A04'))
        backend=workflow.Backend();real=LocalBackend(self.gen,self.project);builds=[]
        def build(script,destination,ids,profile):
            report,renders=real.build_candidate(script,destination,ids,profile);builds.append(report)
            return report,renders
        backend.build_candidate=build
        site_correction={'edits':[{'old':'"site": "foil_grasp_site"','new':'"site": "foil_surface_site"'}]}
        r,t=self.make([p,{'model_py':source},correction,site_correction],backend)
        r.initialize()
        # Focus on a single asset task; the robot tool remains explicitly absent.
        r.state['plan']['jobs']=r.state['plan']['jobs'][:1]
        r.state['plan']['instances']=r.state['plan']['instances'][:1]
        r.state['order']=['instrument']
        missing=collect(r);published=publish(r,missing)
        self.assertEqual(missing,[])
        self.assertEqual(len(builds),3)
        self.assertIn("name 'false' is not defined",builds[0]['checks'][0]['detail'])
        self.assertTrue(builds[1]['assembly_readiness']['geometry_loadable'])
        self.assertFalse(builds[1]['assembly_readiness']['loadable'])
        self.assertTrue(builds[2]['assembly_readiness']['loadable'])
        self.assertTrue(any('foil_grasp_site' in row['detail'] for row in builds[1]['checks']))
        feedback=json.loads(t.requests[2]['messages'][1]['content'][0]['text'])
        self.assertIn("name 'false' is not defined",json.dumps(feedback['feedback']))
        self.assertEqual(feedback['current_model_py'],source)
        self.assertTrue((r.root/published['artifact']/'model.xml').is_file())
        self.assertFalse(r.result()['assembly']['quality_accepted'])
        candidates=[s for s in r.state['stages'] if s['kind']=='scene_candidate']
        self.assertTrue(candidates[0]['missing_jobs'])
        self.assertFalse(candidates[-1]['missing_jobs'])
        self.assertEqual(len(r.state['candidate_attempts']['instrument']),3)
        self.assertEqual(collect(r),[])
        self.assertEqual(len(t.requests),4)

    def test_repeated_real_failure_exhaustion_persists_across_collect(self):
        self.gen=replace(self.gen,max_iterations=2)
        p=plan()
        b=workflow.Backend()
        b.build_candidate=lambda *args:(read(CORPUS/'A03/original_checks.json'),None)
        r,t=self.make([p,workflow.ASSET,{'edits':[{'old':'    pass','new':'    return None'}]}],b)
        r.initialize()
        r.state['plan']['jobs']=r.state['plan']['jobs'][:1]
        r.state['plan']['instances']=r.state['plan']['instances'][:1]
        r.state['order']=['instrument']
        self.assertEqual(collect(r),['instrument'])
        self.assertEqual(r.state['blocked_jobs']['instrument']['kind'],'allocation_exhausted')
        self.assertTrue(r.state['published_scene']['missing_jobs'])
        self.assertEqual(collect(r),['instrument'])
        self.assertEqual(len(t.requests),3)
        self.assertEqual(len(r.state['candidate_attempts']['instrument']),2)

