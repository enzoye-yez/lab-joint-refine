"""Developer-authored workflow fixtures; never formal model output."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
from asset_generation.contracts import GenerationConfig
from asset_prep.config import Config
from asset_prep.storage import digest
from asset_generation.pipeline import snapshot
from asset_prep.errors import BudgetExceeded
from scene_generation.contracts import SceneConfig,numerical_contract
from scene_generation.inputs import catalog
from scene_generation.pipeline import SceneRun
from scene_generation.candidate import inspect_candidate
from scene_generation.preview import resolve_artifact
from generation_fixtures import FakeTransport,PROVENANCE
from scene_fixtures import SceneSandbox,environment,plan,manifest

ASSET={"model_py":"def build(output_dir):\n    pass\n"}
TOOL={"model_py":"def build(output_dir):\n    # TOOL_FIXTURE\n    pass\n"}
ACCEPT={"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":[]}

class Backend(SceneSandbox):
    def __init__(self,unknown=False,crash=False):
        super().__init__();self.unknown=unknown;self.crash=crash;self.events=[]
    def build_candidate(self,script,destination,ids,profile):
        self.events.append('generate')
        report,renders=super().build_and_check(script,destination,ids,profile)
        admission=inspect_candidate(destination/'artifact',ids,profile)
        admission['renders']=report['renders']
        return admission,renders
    def assemble_candidate(self,*args):
        self.events.append('assemble')
        report,renders=super().assemble_and_check(*args)
        report['assembly_readiness']={'loadable':True}
        return report,renders
    def check_scene(self,artifact,destination,limits,contract):
        self.events.append('scene_check')
        root=artifact.parents[3]
        result=json.loads((root/'result.json').read_text(encoding='utf-8'))
        assert result['scene_artifact'] and not result['assembly']['quality_accepted']
        if self.crash:raise OSError('synthetic checker interruption')
        renders=artifact.parent/'renders'
        return {'status':'pass','checks':[{'id':'geometry','status':'pass'}],'renders':['fixture.png']},renders
    def recheck_asset(self,artifact,destination,ids,profile):
        self.events.append('asset_check')
        return {'status':'partial' if self.unknown else 'pass','checks':[{'id':'material_integrity',
            'status':'partial' if self.unknown else 'pass','complete':not self.unknown}],
            'renders':['fixture.png'],'manifest':manifest()},artifact.parent/'renders'

class SceneFirstTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.project=Path(__file__).resolve().parents[1]
        src=self.root/'assets.md';src.write_text('# Test\n### A01｜Synthetic\n100 mm\n',encoding='utf-8')
        self.gen=GenerationConfig();self.cat=catalog(src,self.gen)
        self.env=environment(self.root/'robot')
    def make(self,replies,backend=None,resume=False,limits=None):
        t=FakeTransport(replies)
        r=SceneRun(self.cat,self.env,numerical_contract({}),Config(),
            self.gen,limits or SceneConfig(validation_mode='static'),{'HARNESS_MODEL_API_KEY':'fixture-only'},
            self.root/'run',t,backend or Backend(),self.root/'robot',self.project,resume=resume,provenance=PROVENANCE)
        return r,t
    def test_assembly_precedes_all_quality_and_unknown_keeps_scene(self):
        b=Backend(unknown=True);r,t=self.make([plan(),ASSET,TOOL,ACCEPT,ACCEPT,ACCEPT],b)
        result=r.run()
        self.assertEqual(b.events[:3],['generate','generate','assemble'])
        # An incomplete measurement is never waved through: the scene stops for checker
        # support with the gap listed, and is not a quality acceptance.
        self.assertEqual(result['status'],'needs_check_support')
        self.assertTrue(any('Main check incomplete' in u for u in result['unverified']))
        self.assertTrue(result['assembly']['complete']);self.assertFalse(result['assembly']['quality_accepted'])
        self.assertTrue(any(x['id']=='asset:instrument:material_integrity' and x['status']=='partial' for x in result['assembly']['issues']))
        self.assertTrue((r.root/result['scene_artifact']/'model.xml').is_file())
    def test_supplement_changes_authoritative_scene_decision_after_publication(self):
        class Supplement(Backend):
            def check_scene(inner,*args):
                report,renders=super().check_scene(*args)
                report.update(status="partial",checks=[{"id":"support:station","status":"partial","complete":False}])
                return report,renders
            def inspect_artifact(inner,artifact,destination,scope,requests):
                destination.mkdir(parents=True,exist_ok=True)
                if requests[0]["kind"]=="inventory":return {"status":"pass","inventory":{"bodies":["station__base"]}},None
                return {"status":"pass","results":[{"request":requests[0],"complete":True,"checks":[{
                    "id":"support:station","status":"pass","complete":True,"component_scope":"whole_instance"}]}]},None
        uncertain={"verdict":"uncertain","findings":[],"unverified":[],"blocking":["support unknown"]}
        r,t=self.make([plan(),ASSET,TOOL,ACCEPT,uncertain,{"requests":[{"kind":"support","instance":"station"}]},ACCEPT],Supplement())
        result=r.run()
        self.assertEqual(result["status"],"static_candidate_ready")
        row=next(r for r in result["assembly"]["issues"] if r["id"]=="support:station")
        self.assertEqual(row["status"],"pass")
        latest=r.state["latest"]
        self.assertEqual(latest["report"]["checks"][0]["status"],"pass")
        self.assertLess(r.sandbox.events.index("assemble"),r.sandbox.events.index("scene_check"))

    def test_quality_pass_still_requires_reviews(self):
        # The instrument and the scene are reviewed; the tool's use is deferred to the operation stage.
        r,t=self.make([plan(),ASSET,TOOL,ACCEPT,ACCEPT]);result=r.run()
        self.assertEqual(result['status'],'static_candidate_ready');self.assertEqual(len(t.requests),5)
    def test_passing_tool_skips_asset_review_and_defers_to_operation(self):
        b=Backend();r,t=self.make([plan(),ASSET,TOOL,ACCEPT,ACCEPT],b);result=r.run()
        self.assertEqual(result['status'],'static_candidate_ready')
        phases=[c.get('phase') for c in r.store.state['calls']]
        self.assertIn('asset_review:instrument',phases)
        self.assertNotIn('asset_review:tool',phases)
        self.assertEqual(result['jobs']['tool'],'candidate_ready')
        tool=r.state['jobs']['tool']
        self.assertEqual(tool['feedback']['visual']['source'],'operation_stage')
        self.assertIn('printability',[c['id'] for c in tool['feedback']['numerical']['checks']])
        rows=[x for x in result['deferred_checks'] if x['job']=='tool']
        self.assertEqual([(x['id'],x['stage'],x['status']) for x in rows],[('tool:tool_operation','operation','pending')])
    def test_checker_interruption_keeps_published_scene(self):
        r,t=self.make([plan(),ASSET,TOOL,ACCEPT],Backend(crash=True))
        with self.assertRaises(OSError):r.run()
        result=json.loads((r.root/'result.json').read_text(encoding='utf-8'))
        self.assertTrue(result['assembly']['available']);self.assertFalse(result['assembly']['quality_accepted'])
        self.assertTrue((r.root/result['scene_artifact']/'model.xml').exists())
    def test_review_budget_exhaustion_happens_after_publication(self):
        r,t=self.make([plan(),ASSET,TOOL],limits=SceneConfig(validation_mode='static',max_model_calls=4))
        self.assertEqual(r.run()['status'],'phase_budget_exhausted')
        result=json.loads((r.root/'result.json').read_text(encoding='utf-8'))
        self.assertTrue(result['assembly']['available']);self.assertFalse(result['assembly']['quality_accepted'])

    def test_resume_reuses_geometry_and_completed_scene_measurement(self):
        # Assets are reviewed before the scene is measured; interrupt before the scene review.
        b=Backend();r,t=self.make([plan(),ASSET,TOOL,ACCEPT],b)
        reserve=r.store.reserve
        def interrupt(*args,**kwargs):
            if len(r.store.state['calls'])==4:raise BudgetExceeded('developer interruption before sending')
            return reserve(*args,**kwargs)
        with patch.object(r.store,'reserve',side_effect=interrupt):
            with self.assertRaises(BudgetExceeded):r.run()
        original=r.result()['scene_sha256']
        self.assertIn('scene_check',b.events)
        b2=Backend();r2,t2=self.make([ACCEPT],b2,resume=True)
        result=r2.run()
        self.assertEqual(result['status'],'static_candidate_ready')
        self.assertEqual(result['scene_sha256'],original)
        self.assertNotIn('generate',b2.events)
        self.assertNotIn('assemble',b2.events)
        self.assertNotIn('scene_check',b2.events)
        self.assertEqual(len(t2.requests),1)

    def test_independent_job_runs_before_retry_and_partial_scene_is_labelled(self):
        from scene_fixtures import write_component
        class Broken(Backend):
            def build_candidate(inner,script,destination,ids,profile):
                inner.events.append('attempt:'+ids[0])
                if ids==['A01']:
                    return {'status':'fail','checks':[{'id':'assembly_compile','status':'fail','detail':'broken XML'}],
                            'renders':[],'manifest':None},None
                report,renders=SceneSandbox.build_and_check(inner,script,destination,ids,profile)
                mf=destination/'artifact'/'interfaces.json'
                data=json.loads(mf.read_text(encoding="utf-8"));data['asset_ids']=ids
                for item in data['interfaces']:item['asset_id']=ids[0]
                mf.write_text(json.dumps(data))
                admission=inspect_candidate(destination/'artifact',ids,profile);admission['renders']=report['renders']
                return admission,renders
        src=self.root/'two.md';src.write_text('### A01｜One\n100mm\n### A02｜Two\n100mm\n',encoding='utf-8')
        self.cat=catalog(src,self.gen);self.gen=replace(self.gen,max_iterations=2)
        p=plan();p['jobs'].insert(1,{'id':'other','asset_ids':['A02'],'role':'asset','depends_on':[],'requirements':'test'})
        p['jobs'][-1]['depends_on'].append('other');p['jobs'][-1]['asset_ids'].append('A02')
        p['instances'].append({'id':'other_station','job':'other','placement':'table','position':[-.5,0,.8],'yaw':0})
        deferred={"deferred":{"reason":"缺少依赖的装配接口","missing":["instrument接口"]}}
        b=Broken();r,t=self.make([p,ASSET,ASSET,{"edits":[{"old":"    pass","new":"    return None"}]},
            deferred,deferred,ACCEPT,ACCEPT],b);result=r.run()
        self.assertIn("asset_check",b.events)
        self.assertIn("scene_check",b.events)
        self.assertEqual([e for e in b.events if e.startswith('attempt')],['attempt:A01','attempt:A02','attempt:A01'])
        self.assertEqual(result['status'],'incomplete_scene')
        self.assertTrue(result['assembly']['available']);self.assertFalse(result['assembly']['complete'])
        self.assertEqual(result['assembly']['missing_jobs'],['instrument','tool'])

    def test_generation_budget_preserves_partial_scene_without_more_calls(self):
        r,t=self.make([plan(),ASSET],limits=SceneConfig(validation_mode='static',max_model_calls=3))
        self.assertEqual(r.run()['status'],'phase_budget_exhausted')
        result=r.result()
        self.assertTrue(result['assembly']['available']);self.assertFalse(result['assembly']['complete'])
        self.assertEqual(result['assembly']['missing_jobs'],['tool'])
        self.assertEqual(len(t.requests),2)

    def test_failed_replacement_preserves_last_loadable_asset_and_scene(self):
        from scene_generation.scene_first import collect,publish
        b=Backend();r,t=self.make([plan(),ASSET,TOOL],b)
        r.initialize();self.assertEqual(collect(r),[]);first=publish(r,[])
        old=r.state['jobs']['instrument'];old_hash=old['assembly_readiness']['artifact_sha256']
        old['regenerate']=True
        with patch.object(r,'generate_job') as generate:
            def broken(job,*args,**kwargs):
                r.state['jobs'][job['id']]={'stage':'failed_attempt','feedback':{'numerical':{'status':'fail'}},'assembly_readiness':{'loadable':False}}
                return False
            generate.side_effect=broken
            self.assertEqual(collect(r),[])
        self.assertEqual(r.state['jobs']['instrument']['assembly_readiness']['artifact_sha256'],old_hash)
        self.assertIn('replacement_failure',r.state['jobs']['instrument'])
        self.assertEqual(publish(r,[])['artifact_sha256'],first['artifact_sha256'])

    def test_first_assembly_failure_does_not_publish_uncompiled_directory(self):
        from scene_generation.scene_first import collect, publish
        r,t=self.make([plan(),ASSET,TOOL])
        r.initialize();self.assertEqual(collect(r),[])
        def broken(plan,env,artifacts,robot,destination,*args):
            artifact=destination/'scene'/'artifact'
            artifact.mkdir(parents=True)
            (artifact/'model.xml').write_text('<mujoco><invalid/></mujoco>')
            return {'status':'fail','checks':[{'id':'assembly_compile','status':'fail',
                'detail':'developer fixture: incomplete export'}],
                'assembly_readiness':{'loadable':False},'renders':[]},None
        with patch.object(r.sandbox,'assemble_candidate',side_effect=broken):
            self.assertIsNone(publish(r,[]))
        result=r.result()
        self.assertFalse(result['assembly']['available'])
        self.assertFalse(result['assembly']['complete'])
        self.assertIsNone(result['scene_artifact'])
        self.assertEqual(set(result['assembly']['missing_jobs']),{'instrument','tool'})
        self.assertEqual(result['assembly']['issues'][0]['id'],'assembly_compile')
        self.assertFalse((r.root/'planning_handoff.json').exists())

    def test_failed_scene_replacement_keeps_published_artifact_and_failure_evidence(self):
        from scene_generation.scene_first import collect, publish
        r,t=self.make([plan(),ASSET,TOOL])
        r.initialize();self.assertEqual(collect(r),[]);first=publish(r,[])
        r.state['plan']['instances'][0]['position'][0]+=.1
        failure={'status':'error','checks':[{'id':'assembly_compile','status':'error'}],
                 'assembly_readiness':{'loadable':False},'renders':[]}
        with patch.object(r.sandbox,'assemble_candidate',return_value=(failure,None)):
            self.assertIsNone(publish(r,[]))
        result=r.result()
        self.assertTrue(result['assembly']['available'])
        self.assertEqual(result['scene_sha256'],first['artifact_sha256'])
        self.assertFalse(result['assembly']['quality_accepted'])
        self.assertFalse(result['quality_progress']['evidence_current'])
        self.assertEqual(result['assembly_failure']['checks'],failure['checks'])
        self.assertIsNotNone(publish(r,[]))
        self.assertNotIn('assembly_failure',r.result())
        self.assertEqual(digest(snapshot(r.root/first['artifact'])),first['artifact_sha256'])

    def test_failed_assembly_repairs_cannot_refill_budget_on_resume(self):
        limits=SceneConfig(validation_mode='static',max_iterations=2)
        from scene_generation.scene_first import collect, publish
        r,t=self.make([plan(),ASSET,TOOL],limits=limits)
        r.initialize();collect(r)
        failure={'status':'fail','checks':[{'id':'scene_build','status':'fail',
                 'instances':['station'],'detail':'developer fixture: bad candidate'}],
                 'renders':[]}
        with patch.object(r.sandbox,'assemble_candidate',return_value=(failure,None)):
            self.assertIsNone(publish(r,[]))
        # Assembly failures do not create assessed scene rounds. Their applied
        # repair actions still consume the same finite allocation after resume.
        r.state['repair_history']=[{'status':'applied','cycle':str(i)} for i in range(2)]
        r.state['repair_pending']=True
        r.store.save()
        resumed,transport=self.make([],resume=True,limits=limits)
        self.assertFalse(resumed.route_repair())
        self.assertEqual(resumed.state['status'],'scene_repair_budget_exhausted')
        self.assertEqual(resumed.state['rounds'],[])
        self.assertEqual(len(resumed.state['repair_history']),2)
        self.assertEqual(transport.requests,[])

    def test_confirmed_layout_defect_repairs_after_publication_and_unknown_remains(self):
        class Layout(Backend):
            def check_scene(inner,artifact,destination,limits,contract):
                report,renders=super().check_scene(artifact,destination,limits,contract)
                meta=json.loads((artifact/'scene.json').read_text(encoding='utf-8'))
                failed=meta['plan']['instances'][0]['position'][0]<.6
                report['status']='fail' if failed else 'pass'
                report['checks']=[{'id':'placement','status':report['status'],'instances':['station'],
                                   'detail':'developer fixture: confirmed placement defect'}]
                return report,renders
        repair={'jobs':[],'placements':[{'id':'station','position':[.6,0,.8],'yaw':0}],'reason':'confirmed fixture placement defect'}
        b=Layout(unknown=True)
        r,t=self.make([plan(),ASSET,TOOL,ACCEPT,ACCEPT,ACCEPT,repair,ACCEPT,ACCEPT],b)
        result=r.run()
        self.assertEqual(result['status'],'needs_check_support')
        self.assertEqual(b.build_calls,2);self.assertEqual(b.scene_calls,2)
        self.assertEqual(result['scene_rounds'],2)  # Both candidates receive full quality review.
        self.assertEqual(r.state['repair_history'][0]['status'],'evaluated')
        candidates=[stage for stage in r.state['stages'] if stage['kind']=='scene_candidate']
        self.assertEqual(len(candidates),2)
        for stage in candidates:
            self.assertTrue((r.root/stage['artifact']/'model.xml').is_file())
        self.assertFalse(result['assembly']['quality_accepted'])
        self.assertTrue(any(row['id']=='asset:instrument:material_integrity' and row['status']=='partial' for row in result['assembly']['issues']))

    def test_large_scene_render_contains_geometry_beyond_default_capacity(self):
        import mujoco
        from asset_generation.rendering import renderer
        from asset_generation.static_state import geometry_state
        count=10020
        xml='<mujoco><worldbody><body>'+''.join(f'<geom name="g{i}" type="sphere" size=".001" pos="{i*.003} 0 0"/>' for i in range(count))+'</body></worldbody></mujoco>'
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);geometry_state(model,data)
        with renderer(model,width=64,height=64) as render:
            render.update_scene(data)
            ids={render.scene.geoms[i].objid for i in range(render.scene.ngeom) if render.scene.geoms[i].objtype==int(mujoco.mjtObj.mjOBJ_GEOM)}
        self.assertEqual(ids,set(range(count)))

if __name__=='__main__':unittest.main()
