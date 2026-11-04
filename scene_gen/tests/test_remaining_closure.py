"""Developer fixtures; no model calls. Regression evidence is not scene acceptance."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
import numpy as np
from asset_generation.spatial import BoxIndex
from asset_generation.distance import DistanceQuery
from asset_generation.material import visible_witness, surface_batch
from asset_prep.errors import ValidationError
from scene_generation.runtime import actor_groups
from scene_generation.checkpoints import Checkpoints
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.diagnostics import triage, localization
from scene_generation.progress import repair_comparison


class SpatialTests(unittest.TestCase):
    def test_random_boxes_exact_candidates_and_original_order(self):
        rng=np.random.default_rng(93)
        lo=rng.uniform(-10,10,(300,3));hi=lo+rng.uniform(.001,4,(300,3))
        index=BoxIndex(list(zip(lo,hi)))
        for i in range(300):
            expected=np.flatnonzero(np.all(hi+.002>=lo[i],axis=1)&np.all(hi[i]+.002>=lo,axis=1)).tolist()
            self.assertEqual(index.overlap(lo[i],hi[i],.002),expected)
        self.assertEqual(BoxIndex([]).overlap([0,0,0],[1,1,1]),[])

    def test_touching_and_large_coordinate_boxes(self):
        boxes=[([1e6,0,0],[1e6+1,1,1]),([1e6+1,0,0],[1e6+2,1,1])]
        self.assertEqual(BoxIndex(boxes).overlap([1e6+1,0,0],[1e6+1,0,0]),[0,1])

    def test_surface_batch_preserves_witness_and_invalidates_pose_size_visibility(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="a__root"><joint name="slide" type="slide" axis="1 0 0"/><geom name="a__g" type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d);q=DistanceQuery(m,d)
        with surface_batch(m,d,q):
            for _ in range(20):self.assertEqual(visible_witness(m,d,q,0,np.array([.1,0,0]))["geom"],"a__g")
        d.qpos[0]=1;mujoco.mj_forward(m,d)
        self.assertIsNone(visible_witness(m,d,q,0,np.array([.1,0,0])))
        self.assertIsNotNone(visible_witness(m,d,q,0,np.array([1.1,0,0])))
        m.geom_size[0,0]=.2
        self.assertIsNone(visible_witness(m,d,q,0,np.array([1.1,0,0])))
        self.assertIsNotNone(visible_witness(m,d,q,0,np.array([1.2,0,0])))
        m.geom_rgba[0,3]=0
        self.assertIsNone(visible_witness(m,d,q,0,np.array([1.2,0,0])))


class HandsTests(unittest.TestCase):
    def fixture(self,modes):
        bodies=[];arms=[];bindings={};items=[]
        for n,mode in enumerate(modes):
            side=("left","right")[n]
            stock=mode!="replace"
            fingers=f'<body name="robot__{side}_finger"><joint name="robot__{side}_finger_j"/><geom name="robot__{side}_finger_g" size=".02"/></body>' if stock else ''
            bodies.append(f'<body name="robot__{side}_arm"><geom name="robot__{side}_forearm" size=".05"/>{fingers}</body>')
            arms.append({"id":side,"stock":{"joints":[side+"_finger_j"]}})
            instance=None if mode=="stock" else side+"_tool"
            bindings[side]={"mode":mode,"instance":instance,"reason":"developer fixture"}
            if instance:
                items.append({"id":instance,"placement":"robot"})
                bodies.append(f'<body name="{instance}__base"><geom name="{instance}__g" size=".02"/></body>')
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+''.join(bodies)+'</worldbody></mujoco>')
        meta={"environment":{"robot":{"arms":arms,"model_variants":{"left":"l.xml","right":"r.xml","left,right":"lr.xml"}}},
              "plan":{"instances":items,"end_effectors":bindings}}
        return m,meta

    def test_stock_adapter_replace_and_both_custom(self):
        for modes in (("stock","stock"),("adapter","stock"),("replace","stock"),("replace","replace")):
            with self.subTest(modes=modes):
                m,meta=self.fixture(modes);groups=actor_groups(m,meta)
                self.assertEqual(set(groups),{"left","right"})
                for side,mode in zip(("left","right"),modes):
                    expected=([side+"_tool__g"] if mode!="stock" else [])+(["robot__"+side+"_finger_g"] if mode!="replace" else [])
                    self.assertEqual({m.geom(g).name for g in groups[side]},set(expected))

    def test_missing_custom_does_not_revert_to_stock(self):
        m,meta=self.fixture(("adapter","stock"));meta["plan"]["instances"]=[]
        self.assertEqual(actor_groups(m,meta)["left"],[])
        self.assertTrue(actor_groups(m,meta)["right"])

    def test_duplicate_bindings_rejected(self):
        m,meta=self.fixture(("replace","replace"))
        meta["plan"]["end_effectors"]["right"]["instance"]="left_tool"
        with self.assertRaises(ValidationError):actor_groups(m,meta)

    def test_unnamed_manufacturer_geom_is_bound_by_local_index(self):
        from asset_generation.runtime import InteractionRuntime
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".01"/></worldbody></mujoco>')
        runtime=InteractionRuntime(m,{"interfaces":[]},actor_geom_ids=[0])
        self.assertEqual(runtime.actor_geoms,{0})
        for ids in ([-1],[1],[True]):
            with self.assertRaises(ValueError):InteractionRuntime(m,{"interfaces":[]},actor_geom_ids=ids)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.art=self.root/"artifact";self.art.mkdir();(self.art/"model.xml").write_text('<mujoco/>')
        self.limits=SceneConfig(validation_mode="static");self.contract=numerical_contract({})
        self.q=SimpleNamespace(recoveries=[])
    def cache(self,**kw):
        return Checkpoints(self.root/"cache",self.art,kw.get("limits",self.limits),kw.get("contract",self.contract),kw.get("recovery",False))
    def test_resume_preserves_completed_unknown_and_recovery_evidence(self):
        cache=self.cache()
        def compute():self.q.recoveries.append({"method":"test"});return {"status":"partial","coverage_gaps":[{"error_kind":"unsupported_check"}]}
        result=cache.run("first",compute,self.q)
        with self.assertRaises(RuntimeError):cache.run("second",lambda: (_ for _ in ()).throw(RuntimeError("interrupt")),self.q)
        resumed=self.cache();self.q.recoveries=[]
        self.assertEqual(resumed.run("first",lambda:self.fail("recomputed completed step"),self.q),result)
        self.assertEqual(self.q.recoveries,[{"method":"test"}]);self.assertEqual(resumed.hits,1)
        self.assertEqual(resumed.run("second",lambda:{"status":"fail"},self.q),{"status":"fail"})
    def test_invalidation_includes_geometry_contract_tolerance_recovery_code_and_dependencies(self):
        original=self.cache().key
        for altered in (self.cache(recovery=True),self.cache(limits=replace(self.limits,support_tolerance_m=self.limits.support_tolerance_m/2)),self.cache(contract={**self.contract,"payload_mass_kg":1})):
            self.assertNotEqual(original,altered.key)
        from asset_prep.implementation import package_manifest
        changed={**package_manifest(Path(__file__).resolve().parents[1]/'src'),"changed":"code"}
        with patch('scene_generation.checkpoints.implementation.package_manifest',return_value=changed):self.assertNotEqual(original,self.cache().key)
        with patch('scene_generation.checkpoints.implementation.dependency_versions',return_value={"mujoco":"changed"}):self.assertNotEqual(original,self.cache().key)
        (self.art/"model.xml").write_text('<mujoco model="new"/>');self.assertNotEqual(original,self.cache().key)
    def test_corrupt_checkpoint_is_recomputed_and_cannot_write_into_source(self):
        cache=self.cache();cache.run("step",lambda:1,self.q)
        from asset_prep.storage import digest
        (cache.directory/(digest("step")+".json")).write_text('{broken')
        self.assertEqual(cache.run("step",lambda:2,self.q),2)
        with self.assertRaises(ValidationError):Checkpoints(self.art/"cache",self.art,self.limits,self.contract,False)

    def test_cache_write_failure_is_not_a_new_scene_stop(self):
        with patch('scene_generation.checkpoints.atomic_json',side_effect=OSError("disk full")):
            cache=self.cache()
            self.assertEqual(cache.run("step",lambda:3,self.q),3)
            self.assertTrue(cache.write_warning)

    def test_changed_input_during_check_cannot_publish_cached_evidence(self):
        from asset_prep.errors import StaleEvidence
        cache=self.cache();(self.art/"model.xml").write_text('<mujoco model="mutated"/>')
        with self.assertRaises(StaleEvidence):cache.verify_input()


class TriageLayoutTests(unittest.TestCase):
    def test_mixed_defect_gap_and_program_failure_remain_separate(self):
        report={"checks":[{"id":"support:a","status":"fail","complete":False,"unsupported_components":[{"body":"a__lid","bodies":["a__lid","a__handle"]}],"coverage_gaps":[{"error_kind":"unsupported_check"}]},
                          {"id":"internal","status":"error"},{"id":"collision","status":"partial","unexamined_pairs":7}]}
        value=triage(report)
        self.assertEqual([r["next_step"] for r in value["checks"]],["repair_confirmed_and_measure_unknown","diagnose_checker","obtain_evidence"])
        self.assertEqual(localization(report["checks"][0])["model_entities"],["a__handle","a__lid"])
    def test_fewer_tables_cannot_hide_larger_room_or_table_area(self):
        before={"checks":[],"layout_assessment":{"home_clear":True,"compactness":{"table_count":5,"table_area_m2":25.,"room_area_m2":120.},"instances":[{"instance":"a","targets":[{"target":"open","status":"plausible"}]}]}}
        after=copy.deepcopy(before);m=after["layout_assessment"]["compactness"];m["table_count"]=3
        self.assertTrue(repair_comparison(before,after)["adopt"])
        for field,value in (("table_area_m2",26),("room_area_m2",130)):
            old=m[field];m[field]=value
            self.assertFalse(repair_comparison(before,after)["adopt"]);m[field]=old
        before["layout_assessment"]["compactness"]["workspace_span_m2"]=10
        self.assertFalse(repair_comparison(before,after)["adopt"])

    def test_old_defect_found_on_an_unchanged_asset_does_not_veto_the_round(self):
        before={"checks":[{"id":"scene_collisions","status":"fail","complete":True,"collision_count":1,"collisions":[{"pair":"a__x|b__y"}]},
                          {"id":"asset:J02:visual","status":"pass","complete":True}]}
        after={"checks":[{"id":"scene_collisions","status":"pass","complete":True},
                         {"id":"asset:J02:visual","status":"fail","complete":True}]}
        self.assertEqual(repair_comparison(before,after)["outcome"],"mixed")
        self.assertEqual(repair_comparison(before,after,unchanged_assets={"J01"})["outcome"],"mixed")
        result=repair_comparison(before,after,unchanged_assets={"J02"})
        self.assertTrue(result["adopt"])
        self.assertEqual(result["new_defects"],[])
        self.assertEqual([d["check"] for d in result["discovered_defects"]],["asset:J02:visual"])

    def test_unchanged_assets_compares_artifact_digests_with_the_baseline(self):
        from scene_generation.pipeline import SceneRun
        version=lambda sha:{"assembly_readiness":{"artifact_sha256":sha}}
        run=SimpleNamespace(state={"stages":[dict(directory="s/a",**version("1")),dict(directory="s/b",**version("2")),
                                             {"directory":"s/c"}],
                                   "jobs":{"J01":version("1"),"J02":version("3"),"J03":{}}})
        self.assertEqual(SceneRun.unchanged_assets(run,{"jobs":{"J01":"s/a","J02":"s/b","J03":"s/c"}}),{"J01"})

    def test_a_checker_failure_leaves_the_repair_candidate_unjudged(self):
        from scene_generation.pipeline import SceneRun
        run=SimpleNamespace(state={"repair_history":[{"status":"applied","baseline":{"scene":"s/a"}}]})
        SceneRun.finish_repair(run,{"status":"error","error_kind":"check_execution","checks":[]},None)
        self.assertEqual(run.state["repair_history"][0]["status"],"applied")

    def test_same_body_distinct_islands_are_distinct_repair_obligations(self):
        before={"checks":[{"id":"material_integrity:a","status":"fail","complete":True,"disconnected_count":2,
            "disconnected_components":[{"body":"a__root","geoms":["a__g1"],"detail":"one"},{"body":"a__root","geoms":["a__g2"],"detail":"two"}]}]}
        after=copy.deepcopy(before);after["checks"][0]["disconnected_count"]=1
        after["checks"][0]["disconnected_components"].pop()
        result=repair_comparison(before,after)
        self.assertEqual(result["outcome"],"improved_partial");self.assertTrue(result["adopt"])
        after=copy.deepcopy(before);after["checks"][0]["disconnected_components"][0]["detail"]="wording only"
        self.assertFalse(repair_comparison(before,after)["adopt"])


class IntegratedCheckpointTests(unittest.TestCase):
    def test_real_scene_resume_and_changed_table_invalidate(self):
        from scene_fixtures import environment,write_component,plan
        from scene_generation.assembly import assemble
        from scene_generation.checks import scene_checks
        from asset_generation.material import integrity
        from asset_prep.storage import atomic_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=environment(root/"robot")
            artifacts={"instrument":write_component(root/"instrument"),"tool":write_component(root/"tool",True)}
            art=assemble(plan(),env,artifacts,root/"robot",root/"scene")
            limits=SceneConfig(validation_mode="static");contract=numerical_contract({});cache=root/"cache"
            with patch('asset_generation.material.integrity',wraps=integrity) as check:
                first=scene_checks(art,limits,contract,checkpoint_dir=cache)
                self.assertEqual(check.call_count,2)
            with patch('asset_generation.material.integrity',side_effect=AssertionError("should reuse")), patch('asset_generation.checks.measure_model',side_effect=AssertionError("compile should reuse")):
                second=scene_checks(art,limits,contract,checkpoint_dir=cache)
            self.assertEqual(first,second)
            path=art/"scene.json";value=json.loads(path.read_text());value["plan"]["instances"][0]["yaw"]+=.1;atomic_json(path,value)
            with patch('asset_generation.material.integrity',wraps=integrity) as check:
                scene_checks(art,limits,contract,checkpoint_dir=cache)
                self.assertEqual(check.call_count,2)
