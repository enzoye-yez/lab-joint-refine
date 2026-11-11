"""Real NP-002 report replay plus developer workflow/cache boundary checks; no API calls."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asset_generation.checks import measure_model
from asset_generation.pipeline import snapshot
from asset_prep.errors import StaleEvidence
from scene_generation.checkpoints import Checkpoints
from scene_generation.contracts import SceneConfig
from scene_generation.dispatch import recover
from scene_generation.recovery import decide, located_repairs
from scene_generation.recovery_geometry import load_scene
import test_scene_first as fixture
from test_parallel_authors import TwoAssetBackend


class RecordedRecoveryTests(unittest.TestCase):
    def test_real_defect_feedback_preserves_measurements_and_discloses_truncation(self):
        from scene_generation.progress import repair_tasks
        data=Path(__file__).parent/'data/np002_1028'
        report=json.loads((data/'measurement_excerpt.json').read_text(encoding='utf-8'))
        plan=json.loads((data/'plan.json').read_text(encoding='utf-8'))
        original=copy.deepcopy(report)
        tasks={t['check_id']:t for t in repair_tasks(report,None,plan,'recorded-input')}
        # One task per colliding instance pair; together they carry every recorded hit.
        pairs=[t for key,t in tasks.items() if key.startswith('scene_collisions:')]
        hits=[h for t in pairs for h in t['targets']['collisions']]
        key=lambda h:(h['geom1'],h['geom2'])
        self.assertEqual(sorted(hits,key=key),sorted(report['checks'][0]['collisions'],key=key))
        self.assertTrue(all(t['evidence_incomplete'] for t in pairs))
        self.assertTrue(all(t['evidence_source']=='feedback.checks[id=scene_collisions]' for t in pairs))
        self.assertNotIn('support:grinding_sample',tasks)
        material=tasks['material_integrity:filter_train']
        self.assertEqual(material['target_coverage']['collisions'],{'included':32,'recorded':58,
            'truncated':True,'source':'feedback.checks[id=material_integrity:filter_train].collisions'})
        self.assertEqual(material['targets']['collisions'],report['checks'][1]['collisions'][:32])
        self.assertIn('filter_train',material['shared_asset_instances'])
        self.assertEqual(report,original)

    def test_original_np002_defects_handoff_without_geometry_rechecks(self):
        data=Path(__file__).parent/'data/np002_1028'
        report=json.loads((data/'measurement_excerpt.json').read_text(encoding='utf-8'))
        plan=json.loads((data/'plan.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifact=root/'artifact';artifact.mkdir()
            (artifact/'model.xml').write_text('<mujoco/>')
            run=SimpleNamespace(root=root, state={'plan':plan}, sandbox=object(),
                limits=SceneConfig(validation_mode='static'),contract={},
                check_pause=lambda:None,store=SimpleNamespace(save=lambda:None))
            before=copy.deepcopy(report)
            # Even querying a recovery method is unnecessary when defects are localized.
            with patch('scene_generation.dispatch.REGISTRY',{}):
                result=recover(run,report,artifact,root/'stage')
            self.assertEqual(result['checks'],before['checks'])
            self.assertEqual(decide(result),'repair')
            session=next(iter(run.state['check_recovery'].values()))
            self.assertEqual(session['status'],'repair_ready')
            self.assertEqual(session['actions'],[])
            self.assertIn('support:grinding_sample',session['handoff']['remaining_checks'])
            self.assertIn('scene_collisions',session['handoff']['checks'])
            self.assertFalse(next(r for r in result['checks'] if r['id']=='scene_collisions')['complete'])

    def test_shared_checker_fault_must_not_be_bypassed(self):
        report={'status':'error','checks':[{'id':'collision','status':'fail','instances':['a']}],
                'recovery_failures':[{'status':'open','scope':'scene','kind':'program','remaining_checks':['collision']}]}
        self.assertEqual(located_repairs(report,{'instances':[{'id':'a'}]}),[])
        self.assertEqual(decide(report),'needs_environment')


class WorkflowTests(unittest.TestCase):
    setUp=fixture.SceneFirstTests.setUp
    make=fixture.SceneFirstTests.make

    def mobile_plan(self):
        from test_mobile_layout import mobile_fixture
        self.env,plan=mobile_fixture(self.root)
        # Two more synthetic assets (fixture role no longer exists) exercise batched publication.
        from scene_generation.inputs import catalog
        source=self.root/'assets3.md'
        source.write_text('# Test\n### A01｜Synthetic\n100 mm\n### A02｜Synthetic two\n100 mm\n### A03｜Synthetic three\n100 mm\n',encoding='utf-8')
        self.cat=catalog(source,self.gen)
        for index in range(2):
            key='fixture'+str(index)
            plan['jobs'].insert(-1,{'id':key,'asset_ids':['A0'+str(index+2)],'role':'asset',
                'depends_on':['instrument'],'requirements':'developer publication fixture'})
            plan['jobs'][-1]['depends_on'].append(key)
            plan['instances'].append({'id':key,'job':key,'placement':'table','table_id':'bench',
                'position':[-.5,index*.3,.8],'yaw':0})
        return plan

    def test_publication_batches_a_sweep_and_reuses_unchanged_scene(self):
        from scene_generation.scene_first import collect,publish
        plan=self.mobile_plan()
        backend=TwoAssetBackend()
        # The room is arranged (no model call) before the first non-asset (tool) job.
        run,_=self.make([plan,fixture.ASSET,fixture.ASSET,fixture.ASSET,fixture.TOOL],backend)
        run.initialize()
        self.assertEqual(collect(run),[])
        # The first asset is visible; all later assets are committed without full-scene loads.
        self.assertEqual(backend.scene_calls,1)
        self.assertEqual(set(run.state['published_scene']['job_versions']),{'instrument'})
        scene=publish(run,[])
        self.assertEqual(backend.scene_calls,2)
        self.assertEqual(len(scene['job_versions']),4)
        self.assertEqual(publish(run,[])['artifact_sha256'],scene['artifact_sha256'])
        self.assertEqual(backend.scene_calls,2)

    def test_pause_keeps_staged_assets_and_resume_assembles_without_regenerating(self):
        plan=self.mobile_plan()
        backend=TwoAssetBackend()
        run,_=self.make([plan,fixture.ASSET,fixture.ASSET,fixture.ASSET],backend)
        generate=run.generate_job
        def pause_before_tool(job,*args,**kwargs):
            if job['role']=='tool':raise KeyboardInterrupt
            return generate(job,*args,**kwargs)
        with patch.object(run,'generate_job',side_effect=pause_before_tool),self.assertRaises(KeyboardInterrupt):
            run.run()
        result=json.loads((run.root/'result.json').read_text(encoding='utf-8'))
        self.assertTrue(result['assembly']['available'])
        self.assertFalse(result['assembly']['complete'])
        self.assertEqual(set(result['assembly']['ready_unassembled_jobs']),{'fixture0','fixture1'})
        for artifact in result['assembly']['ready_unassembled_jobs'].values():
            self.assertTrue((run.root/artifact/'model.xml').is_file())
        before=result['budget']['model_calls']
        resumed,transport=self.make([fixture.TOOL]+[fixture.ACCEPT]*5,TwoAssetBackend(),resume=True)
        result=resumed.run()
        self.assertTrue(result['assembly']['complete'])
        self.assertEqual(result['assembly']['ready_unassembled_jobs'],{})
        self.assertEqual(resumed.sandbox.build_calls,1)
        self.assertEqual(result['budget']['model_calls'],before+len(transport.requests))

    def test_repair_of_reviewed_baseline_retains_full_comparison_scope(self):
        class Backend(fixture.Backend):
            def check_scene(inner,artifact,*args):
                report,renders=super().check_scene(artifact,*args)
                moved=json.loads((artifact/'scene.json').read_text())['plan']['instances'][0]['position'][0]>.5
                report.update(status='fail' if moved else 'pass',checks=[{'id':'placement',
                    'status':'fail' if moved else 'pass','instances':['station'],'complete':True}])
                return report,renders
        visual={'verdict':'repair','findings':['Developer fixture: move station'],
                'unverified':[],'affected_instances':['station']}
        action={'jobs':[],'placements':[{'id':'station','position':[.6,0,.8],'yaw':0}],'reason':'fixture visual repair'}
        # The tool has no asset review (its use is judged in the operation stage): the
        # instrument and scene reviews come before the repair, the scene review after it.
        run,_=self.make([fixture.plan(),fixture.ASSET,fixture.TOOL,fixture.ACCEPT,
                        visual,action,fixture.ACCEPT],Backend(),
                        limits=SceneConfig(validation_mode='static',max_iterations=2))
        run.run()
        scenes=[s for s in run.state['stages'] if s['kind']=='scene']
        self.assertEqual(len(scenes),2)
        self.assertTrue(all(s['assessment_scope']=='full_quality' and s['review'] for s in scenes))
        self.assertEqual(run.state['repair_history'][0]['selection'],'baseline_preserved')

    def test_confirmed_scene_defect_does_not_starve_asset_and_visual_reviews(self):
        class Backend(fixture.Backend):
            def check_scene(inner,artifact,*args):
                report,renders=super().check_scene(artifact,*args)
                meta=json.loads((artifact/'scene.json').read_text())
                report['checks']=[{'id':'placement','status':'pass','instances':['station'],'complete':True}]
                if meta['plan']['instances'][0]['position'][0]<.6:
                    report.update(status='fail',checks=[{'id':'placement','status':'fail','instances':['station'],
                        'complete':False,'coverage_gaps':[{'error_kind':'unsupported_check'}]}])
                return report,renders
        repair={'jobs':[],'placements':[{'id':'station','position':[.6,0,.8],'yaw':0}],'reason':'confirmed defect'}
        backend=Backend()
        run,transport=self.make([fixture.plan(),fixture.ASSET,fixture.TOOL,
                                fixture.ACCEPT,fixture.ACCEPT,repair,fixture.ACCEPT],backend)
        result=run.run()
        self.assertEqual(result['status'],'static_candidate_ready')
        self.assertEqual(result['scene_rounds'],2)
        self.assertEqual(len(transport.requests),7)
        # Assets are audited before the scene is measured; the scene defect still gets its review.
        self.assertLess(backend.events.index('asset_check'),backend.events.index('scene_check'))
        phases=[c.get('phase') for c in run.store.state['calls']]
        self.assertLess(phases.index('asset_review:instrument'),phases.index('scene_review'))
        first=next(s for s in run.state['stages'] if s['kind']=='scene')
        self.assertEqual(first['review']['verdict'],'accept')
        self.assertEqual(first['report']['checks'][0]['status'],'fail')
        self.assertFalse(first['report']['checks'][0]['complete'])

    def test_measurement_is_published_and_resumable_before_interrupted_recovery(self):
        class Backend(fixture.Backend):
            def check_scene(inner,*args):
                report,renders=super().check_scene(*args)
                report.update(status='partial',checks=[{'id':'support:station','status':'partial','complete':False}])
                return report,renders
        run,_=self.make([fixture.plan(),fixture.ASSET,fixture.TOOL,fixture.ACCEPT],Backend())
        with patch.object(run,'recover_checks',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
            run.run()
        self.assertTrue(any(s['kind']=='scene_measurement' for s in run.state['stages']))
        result=json.loads((run.root/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(result['status'],'paused')
        self.assertEqual(result['activity']['phase'],'scene_evidence')
        self.assertEqual(result['assembly']['issues'][0]['id'],'support:station')
        resumed,_=self.make([fixture.ACCEPT],Backend(),resume=True)  # Only the scene review remains.
        result=resumed.run()
        self.assertNotIn('scene_check',resumed.sandbox.events)
        self.assertFalse(result['assembly']['quality_accepted'])


class CompiledReuseTests(unittest.TestCase):
    def test_main_and_recovery_share_compile_but_not_check_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);art=root/'artifact';art.mkdir()
            (art/'model.xml').write_text('<mujoco><compiler angle="radian"/><worldbody><geom name="a__g" size=".1"/></worldbody></mujoco>')
            (art/'scene.json').write_text('{"home_qpos":[],"home_ctrl":[],"plan":{"instances":[]}}')
            limits=SceneConfig(validation_mode='static');before=snapshot(art)
            main=Checkpoints(root/'cache',art,limits,{},False)
            model,scale=main.load_model(lambda:measure_model(art,static=True))
            recovery=Checkpoints(root/'cache',art,limits,{},True)
            self.assertNotEqual(main.key,recovery.key)
            self.assertEqual(main.model_key,recovery.model_key)
            with patch('asset_generation.checks.measure_model',side_effect=AssertionError('recompiled')):
                cached,_,_=load_scene(art,limits,{},root/'cache')
            self.assertEqual(cached.ngeom,model.ngeom)
            self.assertEqual(snapshot(art),before)
            changed=Checkpoints(root/'cache',art,replace(limits,validation_mode='full'),{},False)
            self.assertNotEqual(main.model_key,changed.model_key)
            changed=Checkpoints(root/'cache',art,limits,{'changed':True},False)
            self.assertNotEqual(main.key,changed.key)  # Contract changes invalidate findings, not identical geometry compilation.
            self.assertEqual(main.model_key,changed.model_key)
            (art/'scene.json').write_text('{"changed":true}')
            with self.assertRaises(StaleEvidence):main.load_model(lambda:(model,scale))
            self.assertNotEqual(main.model_key,Checkpoints(root/'cache',art,limits,{},False).model_key)

    def test_corrupt_binary_is_recompiled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);art=root/'artifact';art.mkdir()
            (art/'model.xml').write_text('<mujoco><compiler angle="radian"/><worldbody><geom name="g" size=".1"/></worldbody></mujoco>')
            cache=Checkpoints(root/'cache',art,SceneConfig(validation_mode='static'),{},False)
            compute=lambda:measure_model(art,static=True)
            cache.load_model(compute)
            (cache.model_directory/'model.mjb').write_bytes(b'corrupt')
            with patch('mujoco.MjModel.from_binary_path',side_effect=AssertionError('unverified binary')):
                model,_=cache.load_model(compute)
            self.assertEqual(model.ngeom,1)
