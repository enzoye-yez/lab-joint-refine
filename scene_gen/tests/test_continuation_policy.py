"""Developer-authored offline policies. Synthetic model responses, never network."""
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asset_prep.errors import ContextLimitError, BudgetExceeded, StaleEvidence
from asset_generation.contracts import GenerationConfig
from scene_generation.contracts import SceneConfig
from scene_generation.review import scoped_review
from scene_generation.evidence import review_with_evidence
from asset_generation.check_recovery import recover_checks
import test_request_context as context_fixtures
import test_scene_first as scenes
import test_review_repair_loop as repairs
import test_scene_evidence as evidence_fixtures
from scene_fixtures import plan

ACCEPT=scenes.ACCEPT
UNCERTAIN=evidence_fixtures.UNCERTAIN
VIEW=evidence_fixtures.VIEW


class ContextContinuationTests(unittest.TestCase):
    setUp=context_fixtures.ContextTests.setUp
    large=context_fixtures.ContextTests.large
    model=context_fixtures.ContextTests.model

    def test_read_pages_remain_available_for_cross_page_decision(self):
        seen=[]
        class Transport:
            def post(inner,url,payload,key,timeout):
                c=json.loads(payload['messages'][1]['content'][0]['text']);seen.append(c)
                if len(seen)<3:
                    value={'context_read':{'reference':c['_context']['source_ref'],
                        'path':['feedback','checks',0,'details'],'offset':len(seen)-1,'limit':1}}
                else:
                    self.assertEqual(c['context_page']['data'][0]['name'],'1')
                    self.assertEqual(c['context_pages'][0]['data'][0]['name'],'0')
                    value={'done':True}
                return {'choices':[{'message':{'content':json.dumps(value)}}]}
        m=self.model(Transport())
        self.assertTrue(m.ask('fixture',self.large(),[],'review')[0]['done'])
        self.assertEqual(len(seen),3)

    def test_different_requested_limits_do_not_turn_same_page_into_new_evidence(self):
        seen=[]
        class Transport:
            def post(inner,url,payload,key,timeout):
                c=json.loads(payload['messages'][1]['content'][0]['text']);seen.append(c)
                if len(seen)<3:
                    value={'context_read':{'reference':c['_context']['source_ref'],'limit':16*len(seen)}}
                else:
                    self.assertTrue(c['context_read_progress']['reused'])
                    self.assertEqual(c['context_read_progress']['stalled'],1)
                    value={'done':True}
                return {'choices':[{'message':{'content':json.dumps(value)}}]}
        self.assertTrue(self.model(Transport()).ask('fixture',self.large(),[],'review')[0]['done'])

    def test_reread_earlier_page_then_answer_and_replay_without_new_calls(self):
        seen=[]
        class Transport:
            def post(inner,url,payload,key,timeout):
                c=json.loads(payload["messages"][1]["content"][0]["text"])
                seen.append(c);n=len(seen)
                if n<=3:
                    value={"context_read":{"reference":c["_context"]["source_ref"],
                        "path":["feedback","checks",0,"details"],"offset":0 if n!=2 else 1,"limit":1}}
                else:
                    self.assertEqual(c["context_page"]["data"][0]["name"],"0")
                    self.assertTrue(c["context_read_progress"]["reused"])
                    value={"decision":"developer fixture"}
                return {"choices":[{"message":{"content":json.dumps(value)}}]}
        m=self.model(Transport())
        result,_=m.ask("fixture",self.large(),[],"review")
        self.assertEqual(result,{"decision":"developer fixture"})
        self.assertEqual(len(m.store.state["calls"]),4)
        self.assertEqual(m.ask("fixture",self.large(),[],"review")[0],result)
        self.assertEqual(len(seen),4)

    def test_distinct_pages_cross_read_window_without_losing_budget(self):
        count=0
        class Transport:
            def post(inner,url,payload,key,timeout):
                nonlocal count
                count+=1;c=json.loads(payload["messages"][1]["content"][0]["text"])
                v=({"context_read":{"reference":c["_context"]["source_ref"],
                    "path":["feedback","checks",0,"details"],"offset":count,"limit":1}}
                    if count<=3 else {"done":True})
                return {"choices":[{"message":{"content":json.dumps(v)}}]}
        m=self.model(Transport(),replace(self.config,max_context_reads=1))
        self.assertTrue(m.ask("fixture",self.large(),[],"review")[0]["done"])
        self.assertEqual(len(m.store.state["calls"]),4)
        self.assertGreater(m.store.totals()["accounted_tokens"],0)

    def test_stalled_reader_gets_decision_prompt_before_local_deferral(self):
        seen=[]
        class Transport:
            def post(inner,url,payload,key,timeout):
                c=json.loads(payload["messages"][1]["content"][0]["text"]);seen.append(c)
                return {"choices":[{"message":{"content":json.dumps({"context_read":{
                    "reference":c["_context"]["source_ref"],"limit":1}})}}]}
        m=self.model(Transport(),replace(self.config,max_context_reads=2))
        with self.assertRaises(ContextLimitError):
            m.ask("fixture",self.large(),[],"review")
        self.assertEqual(len(seen),4)
        self.assertIn("Submit the verdict",seen[-1]["context_read_progress"]["instruction"])
        paid=len(m.store.state["calls"])
        with self.assertRaises(ContextLimitError):
            m.ask("fixture",self.large(),[],"review")
        self.assertEqual(len(m.store.state["calls"]),paid)

    def test_invalid_detail_request_can_be_corrected_locally(self):
        seen=[]
        class Transport:
            def post(inner,url,payload,key,timeout):
                c=json.loads(payload["messages"][1]["content"][0]["text"]);seen.append(c)
                if len(seen)==1:v={"context_read":{"reference":"outside"}}
                elif len(seen)==2:
                    self.assertIn("error",c["context_page"])
                    v={"context_read":{"reference":c["_context"]["source_ref"],"limit":1}}
                else:v={"done":True}
                return {"choices":[{"message":{"content":json.dumps(v)}}]}
        m=self.model(Transport())
        self.assertTrue(m.ask("fixture",self.large(),[],"review")[0]["done"])


class TaskContinuationTests(unittest.TestCase):
    setUp=scenes.SceneFirstTests.setUp
    make=scenes.SceneFirstTests.make
    backend=repairs.PhysicalRepairLoopTests.backend

    def test_deferred_asset_review_does_not_block_other_asset_or_scene_review(self):
        r,t=self.make([plan(),scenes.ASSET,scenes.TOOL,ACCEPT])
        ask=r.model.ask;phases=[]
        def deferred(instruction,context,images,phase):
            phases.append(phase)
            if phase=="asset_review:instrument":raise ContextLimitError("developer detail stall")
            return ask(instruction,context,images,phase)
        with patch.object(r.model,"ask",side_effect=deferred):
            result=r.run()
        # The stalled instrument review does not hold back the tool (which passes its static
        # checks and is deferred to the operation stage without a review call) or the scene review.
        self.assertNotIn("asset_review:tool",phases)
        self.assertEqual(result["jobs"]["tool"],"candidate_ready")
        self.assertIn("scene_review",phases)
        self.assertNotEqual(result["status"],"paused")
        self.assertTrue(result["assembly"]["complete"])
        self.assertFalse(result["assembly"]["quality_accepted"])
        self.assertEqual(result["budget"]["model_calls"],4)

    def test_confirmed_support_defect_is_repaired_despite_deferred_asset_review(self):
        p=plan();p["instances"][0]["position"][2]=.9
        fix={"jobs":[],"placements":[{"id":"station","position":[.5,0,.8],"yaw":0}],"reason":"developer support repair"}
        r,t=self.make([p,scenes.ASSET,scenes.TOOL,fix,ACCEPT,ACCEPT],self.backend())
        ask=r.model.ask;once=False;phases=[]
        def deferred(instruction,context,images,phase):
            nonlocal once
            phases.append(phase)
            if phase=="asset_review:instrument" and not once:
                once=True;raise ContextLimitError("developer first review deferred")
            return ask(instruction,context,images,phase)
        with patch.object(r.model,"ask",side_effect=deferred):
            result=r.run()
        # Repair now precedes visual review; a subsequently deferred review stays unknown.
        self.assertEqual(result["status"],"needs_check_support")
        self.assertLess(phases.index("scene_repair"),phases.index("asset_review:instrument"))
        self.assertFalse(result["assembly"]["quality_accepted"])
        support=next(row for row in result["assembly"]["issues"] if row["id"]=="support:station")
        self.assertEqual(support["status"],"pass")
        self.assertTrue(any(row["id"]=="asset:instrument:visual" and row["status"]=="partial"
                            for row in result["assembly"]["issues"]))
        self.assertEqual(len(r.state["repair_history"]),1)
        self.assertEqual(result["budget"]["model_calls"],len(t.requests))

    def test_blocker_is_followed_by_a_working_alternative(self):
        p=plan();p["instances"][0]["position"][2]=.9
        blocked={"blocked":{"kind":"unsupported_repair","detail":"developer initial mistaken claim"}}
        fix={"jobs":[],"placements":[{"id":"station","position":[.5,0,.8],"yaw":0}],"reason":"developer alternative"}
        r,t=self.make([p,scenes.ASSET,scenes.TOOL,blocked,fix,ACCEPT,ACCEPT,ACCEPT],self.backend())
        result=r.run()
        self.assertEqual(result["status"],"static_candidate_ready")
        self.assertTrue(any(a.get("response")==blocked for a in r.state["repair_planning"]["attempts"]))

    def test_bad_joint_plan_yields_to_smaller_job_scope(self):
        p=plan();p["instances"][0]["position"][2]=.9
        fix={"jobs":[],"placements":[{"id":"station","position":[.5,0,.8],"yaw":0}],"reason":"developer focused repair"}
        r,t=self.make([p,scenes.ASSET,scenes.TOOL,{},fix,ACCEPT,ACCEPT,ACCEPT],
                     self.backend(),limits=SceneConfig(validation_mode="static",max_repair_plan_attempts=1))
        result=r.run()
        self.assertEqual(result["status"],"static_candidate_ready")
        self.assertEqual([a["scope"] for a in r.state["repair_planning"]["attempts"]],[0,1])

    def test_initial_assembly_failure_routes_trusted_local_failure_to_repair(self):
        p=plan()
        fix={"jobs":[],"placements":[{"id":"station","position":[.4,0,.8],"yaw":0}],"reason":"developer assembly correction"}
        b=scenes.Backend();assemble=b.assemble_candidate;count=0
        def first_failure(*args):
            nonlocal count
            count+=1
            if count==1:
                return {"status":"fail","checks":[{"id":"scene_build","status":"fail",
                    "instances":["station"],"detail":"developer synthetic layout assembly failure"}],"renders":[]},None
            return assemble(*args)
        b.assemble_candidate=first_failure
        r,t=self.make([p,scenes.ASSET,scenes.TOOL,fix,ACCEPT,ACCEPT,ACCEPT],b)
        result=r.run()
        self.assertEqual(result["status"],"static_candidate_ready")
        self.assertEqual(count,2)
        self.assertEqual(r.state["repair_history"][0]["target_checks"],["scene_build"])

    def test_local_asset_checker_failure_still_checks_other_asset(self):
        b=scenes.Backend();original=b.recheck_asset;checked=[]
        def recheck(artifact,*args):
            checked.append(str(artifact))
            if len(checked)==1:raise OSError("developer local checker unavailable")
            return original(artifact,*args)
        b.recheck_asset=recheck
        r,t=self.make([plan(),scenes.ASSET,scenes.TOOL,ACCEPT,ACCEPT,ACCEPT],b)
        result=r.run()
        self.assertEqual(len(checked),2)
        self.assertTrue(result["assembly"]["available"])
        self.assertFalse(result["assembly"]["quality_accepted"])

    def test_scene_passes_its_retry_limit_to_asset_recovery(self):
        b=scenes.Backend();original=b.recheck_asset;limits=[]
        def partial(*args):
            report,renders=original(*args)
            report["checks"]=[{"id":"material_integrity","status":"partial","complete":False}]
            report["status"]="partial"
            return report,renders
        def recovery(backend,artifact,report,destination,**options):
            limits.append(options["max_retries"])
            return report
        b.recheck_asset=partial
        r,t=self.make([plan(),scenes.ASSET,scenes.TOOL,ACCEPT,ACCEPT,ACCEPT],b,
            limits=SceneConfig(validation_mode="static",max_check_retries=0))
        with patch("asset_generation.check_recovery.recover_checks",side_effect=recovery):
            result=r.run()
        self.assertEqual(limits,[0,0])
        self.assertFalse(result["assembly"]["quality_accepted"])

    def test_user_pause_still_prevents_any_new_call(self):
        r,t=self.make([plan()])
        (r.root/"pause.request").write_text("user")
        from asset_prep.errors import PrepError
        with self.assertRaises(PrepError):r.run()
        self.assertEqual(t.requests,[])


class EvidenceContinuationTests(unittest.TestCase):
    setUp=evidence_fixtures.EvidenceLoopTests.setUp
    run_review=evidence_fixtures.EvidenceLoopTests.run_review

    def test_successful_new_observation_extends_evidence_window(self):
        second={**VIEW,"azimuth":90}
        review,_=self.run_review([UNCERTAIN,{"requests":[VIEW]},UNCERTAIN,{"requests":[second]},ACCEPT],rounds=1)
        self.assertEqual(review["verdict"],"accept")
        self.assertEqual(len(self.inspections),3)

    def test_interrupted_observation_still_counts_as_progress_after_resume(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_review([UNCERTAIN,{"requests":[VIEW]}],rounds=1,image_error=KeyboardInterrupt())
        observations=len(self.inspections)
        second={**VIEW,"azimuth":90}
        review,_=self.run_review([UNCERTAIN,{"requests":[second]},ACCEPT],rounds=1)
        self.assertEqual(review["verdict"],"accept")
        self.assertEqual(len(self.inspections),observations+1)

    def test_unavailable_inventory_defers_only_its_review(self):
        class Model:
            def ask(inner,*args):return deepcopy(UNCERTAIN),{}
        class Backend:
            def inspect_artifact(inner,*args):raise OSError("developer unavailable inventory")
        run=SimpleNamespace(model=Model(),sandbox=Backend(),generation=GenerationConfig(),
                            check_pause=lambda:None,state={})
        review,_=review_with_evidence(run,"fixture",{},[],"review",self.root,
            lambda v:scoped_review(v,"asset"),self.root/"artifact","asset")
        self.assertEqual(review["verdict"],"uncertain")
        self.assertEqual(run.state["evidence_outcomes"]["review"]["reason"],"inspection_unavailable")

    def test_framework_proposal_is_tried_despite_model_blocker(self):
        capability=SimpleNamespace(id="fixture",description="developer test view",propose=lambda report,scope:[VIEW])
        blocked={"requests":[],"blocked":{"kind":"unsupported_check","detail":"developer incorrect capability claim"}}
        with patch.dict("asset_generation.inspection_registry.REGISTRY",{"fixture":capability},clear=True):
            review,_=self.run_review([UNCERTAIN,blocked,ACCEPT])
        self.assertEqual(review["verdict"],"accept")
        self.assertEqual(self.inspections[-1],[VIEW])

    def test_failed_batch_splits_items_and_keeps_unresolved_item(self):
        pending=[{"id":"a","stage":"scene"},{"id":"b","stage":"scene"}]
        replies=[ACCEPT,ACCEPT,ACCEPT,
                 {**ACCEPT,"deferred_results":[{"id":"a","status":"resolved","evidence":"developer observed a"}]},
                 ACCEPT,ACCEPT,ACCEPT]
        seen=[]
        class Model:
            def ask(inner,instruction,context,images,phase):
                seen.append(context)
                return deepcopy(replies.pop(0)),{}
        run=SimpleNamespace(model=Model(),sandbox=object(),check_pause=lambda:None)
        result,_=review_with_evidence(run,"fixture",{"deferred_checks":pending,"plan":{"instances":[]}},[],
            "scene_review",self.root,lambda v:scoped_review(v,"scene",pending),self.root,"scene")
        self.assertEqual(result["verdict"],"uncertain")
        self.assertEqual({r["id"]:r["status"] for r in result["deferred_results"]},{"a":"resolved","b":"unresolved"})
        self.assertEqual(len(seen),7)
        self.assertEqual([len(c["deferred_checks"]) for c in seen],[2,2,2,1,1,1,1])

    def test_asset_check_retry_limit_is_honored_and_not_reset_on_resume(self):
        request={"kind":"material"}
        capability=SimpleNamespace(id="material",propose=lambda report,scope:[request],validate=lambda request,scope:None)
        count=0
        class Backend:
            def inspect_artifact(inner,*args):
                nonlocal count
                count+=1
                raise TimeoutError("developer persistent temporary worker failure")
        report={"status":"partial","checks":[{"id":"material_integrity","status":"partial","complete":False}]}
        asset=self.root/"asset";asset.mkdir()
        with patch.dict("asset_generation.inspection_registry.REGISTRY",{"material":capability},clear=True):
            recover_checks(Backend(),asset,report,self.root/"retry",max_retries=0)
            self.assertEqual(count,1)
            recover_checks(Backend(),asset,report,self.root/"retry",max_retries=0)
            self.assertEqual(count,1)
            recover_checks(Backend(),asset,report,self.root/"retry",max_retries=2)
            self.assertEqual(count,3)
            recover_checks(Backend(),asset,report,self.root/"retry",max_retries=2)
            self.assertEqual(count,3)

    def test_temporary_asset_check_failure_retries_without_geometry_edits(self):
        request={"kind":"material"}
        capability=SimpleNamespace(id="material",description="developer test material",propose=lambda report,scope:[request],validate=lambda request,scope:None)
        count=0
        class Backend:
            def inspect_artifact(inner,artifact,destination,scope,requests):
                nonlocal count
                count+=1
                if count==1:raise TimeoutError("developer temporary worker failure")
                return {"status":"pass","results":[{"request":request,"complete":True,
                    "checks":[{"id":"material_integrity","status":"pass","complete":True}]}]},None
        report={"status":"partial","checks":[{"id":"material_integrity","status":"partial","complete":False}]}
        artifact=self.root/"asset";artifact.mkdir()
        (artifact/"model.xml").write_text("developer immutable geometry")
        with patch.dict("asset_generation.inspection_registry.REGISTRY",{"fixture":capability},clear=True):
            result=recover_checks(Backend(),artifact,report,self.root/"recovery")
        self.assertEqual(count,2)
        self.assertEqual(result["status"],"pass",result)
        self.assertEqual((artifact/"model.xml").read_text(),"developer immutable geometry")


if __name__=="__main__":unittest.main()
