"""Developer-authored review/repair fixtures; no provider or formal asset edits."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asset_generation.review import ask_review
from asset_generation.contracts import GenerationConfig
from asset_prep.errors import ValidationError
from scene_generation.review import scoped_review, resolution_feedback
from scene_generation.evidence import review_with_evidence, validate_inventory
from scene_generation.progress import repair_comparison, repair_tasks
from scene_generation.recovery import decide, repairable_rows
from scene_generation.contracts import SceneConfig
import test_scene_evidence as evidence_fixtures
import test_scene_first as loop_fixtures
from scene_fixtures import plan

ACCEPT={"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":[]}
def answer(key,status="resolved"):
    return {"id":key,"status":status,"evidence":"developer fixture observation"}

class ReviewRepairContractTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
    def reviewer(self,replies,seen):
        class Model:
            def ask(inner,instruction,context,images,phase):
                seen.append(deepcopy(context));value=replies.pop(0)
                if isinstance(value,BaseException):raise value
                return value,{"source":"developer_fixture"}
        return Model()
    def test_small_patch_retains_other_valid_answers(self):
        seen=[];pending=[{"id":k,"stage":"scene"} for k in ("a","b")]
        replies=[{**ACCEPT,"deferred_results":[answer("a")]},{"deferred_results":[answer("b")]}]
        result,_=ask_review(self.reviewer(replies,seen),"fixture",{},[],"test",self.root,
            validator=lambda v:scoped_review(v,"scene",pending))
        self.assertEqual({r["id"] for r in result["deferred_results"]},{"a","b"})
        correction=seen[1]["review_format_correction"]
        self.assertEqual(correction["item_feedback"]["missing_ids"],["b"])
        self.assertEqual(correction["item_feedback"]["retained_ids"],["a"])
    def test_duplicate_conflict_is_not_preserved_and_unknown_is_explicit(self):
        good,errors=resolution_feedback([answer("a"),answer("a","unresolved"),answer("b"),answer("evil")],{"a","b"})
        self.assertEqual(set(good),{"b"})
        self.assertEqual([e["id"] for e in errors],["a","evil"])
        with self.assertRaisesRegex(ValidationError,"unknown=.*extra"):
            scoped_review({**ACCEPT,"extra":1},"asset")
    def test_incremental_receipt_and_attempt_budget_survive_interruption(self):
        seen=[];receipt={};pending=[{"id":k,"stage":"scene"} for k in ("a","b")]
        validate=lambda v:scoped_review(v,"scene",pending)
        with self.assertRaises(KeyboardInterrupt):
            ask_review(self.reviewer([{**ACCEPT,"deferred_results":[answer("a")]},KeyboardInterrupt()],seen),
                "fixture",{},[],"test",self.root,validator=validate,session=receipt)
        self.assertEqual(receipt["attempts"],1)
        result,_=ask_review(self.reviewer([{"deferred_results":[answer("b")]}],seen),"fixture",{},[],"test",self.root,
                            validator=validate,session=receipt)
        self.assertEqual(result["verdict"],"accept");self.assertEqual(receipt["attempts"],2)
        ask_review(self.reviewer([],seen),"fixture",{},[],"test",self.root,validator=validate,session=receipt)
        self.assertEqual(len(seen),3)
    def test_large_scene_review_batches_and_rechecks_union(self):
        seen=[];pending=[{"id":str(i),"stage":"scene"} for i in range(35)]
        replies=[{**ACCEPT,"deferred_results":[answer(d["id"]) for d in pending[i:i+16]]} for i in range(0,35,16)]
        run=SimpleNamespace(model=self.reviewer(replies,seen),sandbox=object(),check_pause=lambda:None)
        result,_=review_with_evidence(run,"fixture",{"deferred_checks":pending,"plan":{"instances":[]}},[],"test",self.root,
            lambda v:scoped_review(v,"scene",pending),self.root,"scene")
        self.assertEqual([len(c["deferred_checks"]) for c in seen],[16,16,3])
        self.assertEqual(len(result["deferred_results"]),35)
    def test_inventory_matches_single_and_multiple_configuration_contract(self):
        request=deepcopy(evidence_fixtures.VIEW)
        single={"bodies":["base","lid"],"geoms":[],"joints":[]}
        validate_inventory([request],single,"asset")
        with self.assertRaisesRegex(ValidationError,"available configuration"):
            validate_inventory([request],{"configurations":{"closed":single}},"asset")
        validate_inventory([{**request,"configuration":"closed"}],{"configurations":{"closed":single}},"asset")
        with self.assertRaisesRegex(ValidationError,"unknown objects"):
            validate_inventory([{**request,"focus_body":"invented"}],single,"asset")
        with self.assertRaisesRegex(ValidationError,"unknown joint"):
            validate_inventory([{**request,"joints":{"invented":1}}],single,"asset")
    def test_successful_duplicate_returns_evidence_without_tool_execution(self):
        fixture=evidence_fixtures.EvidenceLoopTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        u=evidence_fixtures.UNCERTAIN;view=evidence_fixtures.VIEW
        result,_=fixture.run_review([u,{"requests":[view]},u,{"requests":[view]},ACCEPT])
        self.assertEqual(result["verdict"],"accept")
        self.assertEqual(len(fixture.inspections),2)
        self.assertTrue(fixture.requests[-1][0]["supplemental_evidence"][-1]["results"][0]["reused"])
    def test_repeated_cached_only_planning_stops_without_extra_tools(self):
        fixture=evidence_fixtures.EvidenceLoopTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        u=evidence_fixtures.UNCERTAIN;view=evidence_fixtures.VIEW
        result,_=fixture.run_review([u,{"requests":[view]},u,{"requests":[view]},u,{"requests":[view]}],rounds=4)
        self.assertEqual(result["verdict"],"uncertain")
        outcome=json.loads((fixture.root/"evidence_outcome.json").read_text(encoding="utf-8"))
        self.assertEqual(outcome["reason"],"evidence_no_progress")
        self.assertEqual(len(fixture.inspections),2)
    def test_format_failure_does_not_become_geometry_inquiry(self):
        fixture=evidence_fixtures.EvidenceLoopTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        result,_=fixture.run_review([{}, {}, {}])
        self.assertEqual(result["verdict"],"uncertain")
        self.assertEqual(fixture.inspections,[])
        self.assertEqual(len(fixture.requests),3)
    def test_supplemental_review_format_failure_stops_more_inquiry(self):
        fixture=evidence_fixtures.EvidenceLoopTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        result,_=fixture.run_review([evidence_fixtures.UNCERTAIN,{"requests":[evidence_fixtures.VIEW]},{},{},{}])
        self.assertEqual(result["verdict"],"uncertain")
        self.assertEqual(len(fixture.inspections),2)
        outcome=json.loads((fixture.root/"evidence_outcome.json").read_text(encoding="utf-8"))
        self.assertEqual(outcome["reason"],"review_format_exhausted")
    def test_comparison_does_not_resolve_disappeared_unknown_or_skipped_check(self):
        before={"checks":[{"id":"support:a","status":"fail","complete":True}]}
        for row in (None,{"id":"support:a","status":"partial"},{"id":"support:a","status":"skipped"}):
            with self.subTest(row=row):
                comparison=repair_comparison(before,{"checks":[row] if row else []})
                self.assertFalse(comparison["adopt"])
                self.assertFalse(comparison["resolved_checks"])
    def test_repair_tradeoff_preserves_baseline_not_failure_count(self):
        before={"checks":[{"id":"support:a","status":"fail"},{"id":"boundary:a","status":"pass"}]}
        after={"checks":[{"id":"support:a","status":"pass"},{"id":"boundary:a","status":"fail"}]}
        comparison=repair_comparison(before,after)
        self.assertEqual(comparison["outcome"],"mixed");self.assertFalse(comparison["adopt"])
    def test_local_temporary_failure_leaves_only_independent_defect_editable(self):
        report={"status":"error","fault_status_managed":True,"error_kind":"backend_execution",
            "checks":[{"id":"support:a","status":"fail","instances":["a"]},{"id":"support:b","status":"error","instances":["b"]}],
            "recovery_failures":[{"status":"open","kind":"temporary","scope":"checks","remaining_checks":["support:b"]}]}
        self.assertEqual(decide(report),"repair")
        self.assertEqual([r["id"] for r in repairable_rows(report)],["support:a"])
        report["recovery_failures"][0]["kind"]="program"
        self.assertEqual(decide(report),"needs_environment")
    def test_task_preserves_known_defect_and_unknown_separately(self):
        report={"status":"fail","checks":[{"id":"support:station","status":"fail","complete":False,
            "instances":["station"],"unsupported_components":[{"body":"station__cap"}]}]}
        tasks=repair_tasks(report,None,plan(),"current")
        self.assertEqual(tasks[0]["certainty"],"confirmed_defect")
        self.assertTrue(tasks[0]["evidence_incomplete"])
        self.assertEqual(tasks[0]["targets"]["unsupported_components"],[{"body":"station__cap"}])

    def test_protocol_and_deferred_change_do_not_reuse_old_session(self):
        from scene_generation.evidence_state import session
        run=SimpleNamespace(root=self.root,store=SimpleNamespace(state={"identity":{"implementation":"fixture"}}))
        artifact=self.root/"asset";artifact.mkdir()
        (artifact/"model.xml").write_text("developer fixture")
        context={"source":"same source","protocol":{"requirement":"first"},"deferred_checks":[{"id":"a"}]}
        old,_=session(run,artifact,"scene_review","scene",context,self.root)
        for changed in ({**context,"protocol":{"requirement":"second"}},{**context,"deferred_checks":[{"id":"b"}]}):
            fresh,_=session(run,artifact,"scene_review","scene",changed,self.root)
            self.assertNotEqual(old["input_key"],fresh["input_key"])
    def test_confirmed_defects_are_repairable_without_unrelated_evidence_loop(self):
        seen=[]
        run=SimpleNamespace(model=self.reviewer([evidence_fixtures.UNCERTAIN],seen),
            sandbox=SimpleNamespace(inspect_artifact=lambda *args: self.fail("Should repair known defect before further inquiry")),
            generation=GenerationConfig(),state={},check_pause=lambda:None)
        context={"checks":{"status":"fail","checks":[{"id":"support:device","status":"fail","instances":["device"]}]}}
        result,_=review_with_evidence(run,"fixture",context,[],"test",self.root,
            lambda v:scoped_review(v,"scene",instance_ids={"device"}),self.root/"artifact","scene")
        self.assertEqual(result["verdict"],"uncertain")
        self.assertEqual(run.state["evidence_outcomes"]["test"]["reason"],"confirmed_defects_first")
        self.assertEqual(decide(context["checks"],result),"repair")

    def test_material_connection_repair_compares_real_geometry(self):
        import test_engineering_closure as material
        before=material.probe(material.BASE.format(child=material.FLOAT))
        after=material.probe(material.BASE.format(child=material.FLOAT.replace('0 0 .3','0 0 .12')))
        comparison=repair_comparison({"checks":[before]},{"checks":[after]})
        self.assertEqual(before["status"],"fail")
        self.assertEqual(after["status"],"pass",after)
        self.assertEqual(comparison["outcome"],"improved")
    def test_wrong_specification_is_fixed_by_geometry_not_tolerance(self):
        from generation_fixtures import XML,write_artifact
        from asset_generation.checks import validate_artifact
        from asset_generation.contracts import check_profile
        profile=check_profile({"validation_mode":"static","dimensions":[{"id":"width","geoms":["base_geom"],"axis":0,"expected_m":.1,"tolerance_m":.000001}]})
        xml=XML.replace('size="0.05 0.05 0.025"','size="0.06 0.05 0.025"',1)
        self.assertNotEqual(xml,XML)
        write_artifact(self.root,xml)
        before=validate_artifact(self.root,["A01"],profile)
        write_artifact(self.root,XML)
        after=validate_artifact(self.root,["A01"],profile)
        comparison=repair_comparison(before,after)
        self.assertIn("width",comparison["resolved_checks"])
        self.assertEqual(profile["dimensions"][0]["tolerance_m"],.000001)

    def test_same_collision_pair_getting_deeper_is_regression(self):
        row={"id":"scene_collisions","status":"fail","complete":True,"collisions":[{"geom1":"a","geom2":"b","distance_m":-.001}]}
        worse=deepcopy(row);worse["collisions"][0]["distance_m"]=-.01
        result=repair_comparison({"checks":[row]},{"checks":[worse]})
        self.assertEqual(result["outcome"],"regressed")
        self.assertFalse(result["adopt"])
    def test_growing_partial_coverage_gap_is_not_progress(self):
        row={"id":"support:a","status":"partial","complete":False,"measurement_error_count":1}
        result=repair_comparison({"checks":[row]},{"checks":[{**row,"measurement_error_count":5}]})
        self.assertEqual(result["outcome"],"coverage_lost")
    def test_actual_slot_obstruction_at_current_fit_can_be_repaired(self):
        import mujoco
        from asset_generation.distance import DistanceQuery,observe
        from asset_generation.static_state import geometry_state
        def report(blocked):
            model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="rack"><geom name="wall" type="box" pos=".1 0 0" size=".01 .1 .1"/>'+
                ('<geom name="plug" type="box" size=".05 .05 .05"/>' if blocked else '')+
                '</body><body name="tube"><geom name="insert" type="box" size=".02 .02 .02"/></body></worldbody></mujoco>')
            data=mujoco.MjData(model);geometry_state(model,data)
            query=DistanceQuery(model,data);hits=[]
            for name in (["wall","plug"] if blocked else ["wall"]):
                evidence=observe(query,model.geom("insert").id,model.geom(name).id,.2)
                if evidence["distance_m"] is not None and evidence["distance_m"]<-.0005:hits.append(evidence)
            return {"checks":[{"id":"scene_collisions","status":"fail" if hits else "pass","complete":True,"collisions":hits}]}
        before,after=report(True),report(False)
        self.assertEqual(before["checks"][0]["status"],"fail")
        self.assertIn("scene_collisions",repair_comparison(before,after)["resolved_checks"])

class PhysicalRepairLoopTests(unittest.TestCase):
    setUp=loop_fixtures.SceneFirstTests.setUp
    make=loop_fixtures.SceneFirstTests.make
    def backend(self):
        class Physical(loop_fixtures.Backend):
            def check_scene(inner,artifact,destination,limits,contract):
                from scene_generation.checks import scene_checks
                report=scene_checks(artifact,limits,contract)
                report["renders"]=["fixture.png"]
                return report,artifact.parent/"renders"
        return Physical()
    def test_real_support_repair_and_full_recheck(self):
        p=plan();p["instances"][0]["position"][2]=.9
        repair={"jobs":[],"placements":[{"id":"station","position":[.5,0,.8],"yaw":0}],"reason":"developer fixture: lower whole asset onto table"}
        r,t=self.make([p,loop_fixtures.ASSET,loop_fixtures.TOOL,ACCEPT,ACCEPT,repair,ACCEPT,ACCEPT,ACCEPT],self.backend())
        result=r.run()
        self.assertEqual(result["status"],"static_candidate_ready",result.get("stop_reason"))
        history=r.state["repair_history"][0]
        self.assertEqual(history["selection"],"adopted")
        self.assertIn("support:station",history["comparison"]["resolved_checks"])
        self.assertEqual(r.sandbox.build_calls,2)
    def test_real_collision_repair_rejects_new_support_failure_and_keeps_budget(self):
        p=plan();p["instances"].append({"id":"neighbor","job":"instrument","placement":"table","position":[.55,0,.8],"yaw":0})
        bad={"jobs":[],"placements":[{"id":"station","position":[.5,0,1.0],"yaw":0}],"reason":"developer fixture: avoid overlap but introduce floating"}
        # Repeated-instance review explicitly reports one prototype unit.
        unit={**ACCEPT,"prototype_units":1}
        r,t=self.make([p,loop_fixtures.ASSET,loop_fixtures.TOOL,unit,ACCEPT,bad,unit,ACCEPT,ACCEPT],
                      self.backend(),limits=SceneConfig(validation_mode="static",max_iterations=2))
        result=r.run()
        self.assertEqual(result["scene_rounds"],2)
        history=r.state["repair_history"][0]
        self.assertEqual(history["selection"],"baseline_preserved")
        self.assertIn(history["outcome"],{"mixed","regressed"})
        self.assertEqual(r.state["plan"]["instances"][0]["position"],[.5,0,.8])
        self.assertEqual(result["budget"]["model_calls"],len(t.requests))
        self.assertFalse(result["assembly"]["quality_accepted"])
        self.assertEqual(len([s for s in r.state["stages"] if s["kind"]=="scene_candidate"]),2)

    def test_selection_interruption_resumes_before_iteration_budget_exit(self):
        p=plan();p["instances"].append({"id":"neighbor","job":"instrument","placement":"table","position":[.55,0,.8],"yaw":0})
        bad={"jobs":[],"placements":[{"id":"station","position":[.5,0,1.0],"yaw":0}],"reason":"developer regression"}
        unit={**ACCEPT,"prototype_units":1}
        limit=SceneConfig(validation_mode="static",max_iterations=2)
        r,t=self.make([p,loop_fixtures.ASSET,loop_fixtures.TOOL,unit,ACCEPT,bad,unit,ACCEPT,ACCEPT],self.backend(),limits=limit)
        original=r.select_repair_candidate
        def interrupt():
            if r.state.get("repair_history") and r.state["repair_history"][-1]["status"]=="evaluated":raise KeyboardInterrupt()
            return original()
        with patch.object(r,"select_repair_candidate",side_effect=interrupt),self.assertRaises(KeyboardInterrupt):r.run()
        r2,t2=self.make([],self.backend(),resume=True,limits=limit)
        result=r2.run()
        self.assertEqual(t2.requests,[])
        self.assertEqual(result["scene_rounds"],2)
        self.assertEqual(r2.state["repair_history"][0]["selection"],"baseline_preserved")
        self.assertEqual(r2.state["plan"]["instances"][0]["position"],[.5,0,.8])
        self.assertEqual(result["budget"]["model_calls"],len(t.requests))

    def test_explicit_repair_blocker_is_persistent_without_repeat_calls(self):
        p=plan();p["instances"][0]["position"][2]=.9
        blocker={"blocked":{"kind":"unsupported_repair","detail":"developer fixture: no permitted change available"}}
        r,t=self.make([p,loop_fixtures.ASSET,loop_fixtures.TOOL,ACCEPT,ACCEPT]+[blocker]*4,self.backend())
        result=r.run()
        self.assertEqual(result["stop_reason"]["reason"],"repair_tasks_deferred")
        self.assertEqual(len(r.state["repair_planning"]["attempts"]),2)  # Identical scopes share their failed plan attempts.
        r2,t2=self.make([],self.backend(),resume=True)
        result=r2.run()
        self.assertEqual(t2.requests,[])
        self.assertFalse(result["assembly"]["quality_accepted"])

    def test_failed_repair_assembly_preserves_baseline_and_cost(self):
        p=plan();p["instances"][0]["position"][2]=.9
        repair={"jobs":[],"placements":[{"id":"station","position":[.5,0,.8],"yaw":0}],"reason":"developer repair"}
        backend=self.backend();assemble=backend.assemble_candidate
        count=0
        def fail_second(*args):
            nonlocal count
            count+=1
            if count==2:return {"assembly_readiness":{"loadable":False},"detail":"developer assembly fault"},None
            return assemble(*args)
        backend.assemble_candidate=fail_second
        alternate={**repair,"placements":[{**repair["placements"][0],"yaw":.1}]}
        r,t=self.make([p,loop_fixtures.ASSET,loop_fixtures.TOOL,ACCEPT,ACCEPT,repair,alternate,ACCEPT,ACCEPT,ACCEPT],backend)
        result=r.run()
        self.assertEqual(result["status"],"static_candidate_ready")
        self.assertEqual(r.state["repair_history"][0]["selection"],"baseline_preserved")
        self.assertEqual(r.state["plan"]["instances"][0]["position"],[.5,0,.8])
        self.assertEqual(len(r.state["repair_history"]),2)
        self.assertEqual(result["budget"]["model_calls"],len(t.requests))
        self.assertTrue((r.root/result["scene_artifact"]/"model.xml").is_file())
        self.assertTrue(result["assembly"]["quality_accepted"])
