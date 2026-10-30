import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from asset_generation.contracts import GenerationConfig
from asset_prep.storage import atomic_json,digest
from scene_generation.evidence import review_with_evidence
from scene_generation.review import scoped_review


ACCEPT={"verdict":"accept","findings":[],"unverified":[]}
UNCERTAIN={"verdict":"uncertain","findings":[],"unverified":[],"blocking":["缺少内部视图"]}
VIEW={"kind":"view","joints":{},"hide_bodies":["lid"],"focus_body":"base","azimuth":45,"elevation":-90}


class EvidenceLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.requests=[]
        self.inspections=[]

    def run_review(self,replies,scope="asset",status="pass",rounds=3,attempts=2,image_error=None,review_model=None):
        outer=self
        class Model:
            def ask(self,instruction,context,images,phase):
                outer.requests.append((context,images,phase))
                reply=replies.pop(0)
                if isinstance(reply,BaseException):raise reply
                return reply,{"synthetic":True}
        class Backend:
            def inspect_artifact(self,artifact,destination,scope,requests):
                outer.inspections.append(requests)
                destination.mkdir(parents=True)
                if requests[0]["kind"] == "inventory":
                    return {"status":"pass","inventory":{"bodies":["base","lid"]}},None
                current=status.pop(0) if isinstance(status,list) else status
                result={"request":requests[0]}
                if requests[0]['kind']=='view':
                    result['image']='internal.png'
                if current!='pass':
                    result['error']='synthetic measurement failure'
                report={"status":current,"results":[result],"renders":["internal.png"]}
                atomic_json(destination/"report.json",report)
                return report,destination
        def images(path,report):
            if image_error:raise image_error
            return [{"path":"synthetic/internal.png","sha256":"synthetic"}]
        run=SimpleNamespace(model=Model(),sandbox=Backend(),generation=GenerationConfig(),check_pause=lambda:None,
                            limits=SimpleNamespace(max_evidence_rounds=rounds,max_evidence_plan_attempts=attempts),
                            images=images,store=SimpleNamespace(state={"identity":{"model":review_model}},save=lambda:None))
        return review_with_evidence(run,"synthetic",{},[],"test",self.root,
                                    lambda value:scoped_review(value,scope,instance_ids={"device"}),self.root/"artifact",scope)

    def test_review_interruption_reuses_tools_and_preserves_evidence(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_review([UNCERTAIN,{"requests":[VIEW]},KeyboardInterrupt()])
        before=list(self.inspections)
        result,_=self.run_review([ACCEPT])
        self.assertEqual(result["verdict"],"accept")
        self.assertEqual(self.inspections,before)
        self.assertEqual(len(self.requests),4)

    def test_interrupt_after_tool_before_image_loading_reuses_completed_execution(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_review([UNCERTAIN,{"requests":[VIEW]}],image_error=KeyboardInterrupt())
        before=list(self.inspections)
        result,_=self.run_review([ACCEPT])
        self.assertEqual(result["verdict"],"accept")
        self.assertEqual(self.inspections,before)

    def test_new_reviewer_can_resolve_saved_question_with_alternative_view(self):
        question="issue_"+digest(UNCERTAIN["blocking"][0])[:16]
        original={**VIEW,"question":question}
        with self.assertRaises(KeyboardInterrupt):
            self.run_review([UNCERTAIN,{"requests":[original]},KeyboardInterrupt()],
                            status="incomplete",review_model="synthetic_a")
        before=len(self.inspections)
        rewritten={**UNCERTAIN,"blocking":["需要换角度看同一内部部件"]}
        alternative={**original,"azimuth":90}
        result,_=self.run_review([rewritten,rewritten,{"requests":[alternative]},ACCEPT],
                                 review_model="synthetic_b")
        self.assertEqual(result["verdict"],"accept")
        self.assertEqual(self.inspections[before:],[[alternative]])
        plan=self.requests[-2][0]
        self.assertIn(question,plan["questions"])
        outcome=json.loads((self.root/"evidence_outcome.json").read_text(encoding="utf-8"))
        self.assertEqual(outcome["unresolved_checks"],[])

    def test_uncertain_adds_actual_evidence_and_reviews_again(self):
        review,_=self.run_review([UNCERTAIN,{"requests":[VIEW]},ACCEPT])
        self.assertEqual(review["verdict"],"accept")
        self.assertEqual(len(self.inspections),2)
        context,images,_=self.requests[-1]
        self.assertEqual(context["supplemental_evidence"][0]["status"],"pass")
        self.assertEqual(images[0]["path"],"synthetic/internal.png")
        self.assertTrue((self.root/"evidence_1/plan.json").is_file())
        self.assertTrue((self.root/"evidence_1/review.json").is_file())

    def test_scene_measurement_can_route_to_repair_not_accept(self):
        repair={"verdict":"repair","findings":["device:净空不足"],"unverified":[],"affected_instances":["device"]}
        review,_=self.run_review([UNCERTAIN,{"requests":[{"kind":"clearance","instance":"device","distance_m":.05}]},repair],"scene")
        self.assertEqual(review["verdict"],"repair")

    def test_unavailable_evidence_keeps_uncertain_and_can_stop(self):
        review,_=self.run_review([UNCERTAIN,{"requests":[]},{"requests":[]}])
        self.assertEqual(review["verdict"],"uncertain")
        self.assertEqual(len(self.inspections),1)
        outcome=json.loads((self.root/'evidence_outcome.json').read_text(encoding='utf-8'))
        self.assertEqual(outcome['reason'],'evidence_planning_exhausted')

    def test_failed_inspection_cannot_be_accepted(self):
        review,_=self.run_review([UNCERTAIN,{"requests":[VIEW]},ACCEPT,ACCEPT,ACCEPT,{"requests":[]},{"requests":[]}],status="incomplete")
        self.assertEqual(review["verdict"],"uncertain")

    def test_empty_plan_is_corrected_without_discarding_review(self):
        review,_=self.run_review([UNCERTAIN,{'requests':[]},{'requests':[VIEW]},ACCEPT])
        self.assertEqual(review['verdict'],'accept')
        self.assertIn('为空',self.requests[2][0]['validation_feedback'][0]['error'])

    def test_failed_query_can_retry_but_successful_query_cannot_repeat(self):
        review,_=self.run_review([UNCERTAIN,{'requests':[VIEW]},UNCERTAIN,
            {'requests':[VIEW]},ACCEPT],status=['incomplete','pass'])
        self.assertEqual(review['verdict'],'accept')
        self.assertEqual(len(self.inspections),3)

    def test_unrelated_success_cannot_hide_failed_query(self):
        other={**VIEW,'azimuth':90}
        review,_=self.run_review([UNCERTAIN,{'requests':[VIEW]},UNCERTAIN,
            {'requests':[other]},ACCEPT,ACCEPT,ACCEPT],status=['incomplete','pass'],rounds=2)
        self.assertEqual(review['verdict'],'uncertain')
        outcome=json.loads((self.root/'evidence_outcome.json').read_text(encoding='utf-8'))
        self.assertEqual(outcome['reason'],'review_format_exhausted')
        self.assertEqual(outcome['unresolved_checks'][0]['request'],VIEW)
        self.assertEqual(len(self.requests),7)
        self.assertEqual(len(self.inspections),3)

    def test_explicit_missing_capability_has_distinct_stop_reason(self):
        blocked={'requests':[],'blocked':{'kind':'unsupported_check','detail':'synthetic unsupported check'}}
        review,_=self.run_review([UNCERTAIN,blocked,blocked])
        self.assertEqual(review['verdict'],'uncertain')
        self.assertEqual(len(self.requests),3)
        self.assertEqual(len(self.inspections),1)
        outcome=json.loads((self.root/'evidence_outcome.json').read_text(encoding='utf-8'))
        self.assertEqual(outcome['reason'],'capability_blocked')
        self.assertEqual(outcome['blocked'],blocked['blocked'])
        self.assertEqual(len(outcome['planning_feedback']),1)

    def test_malformed_blocked_cannot_turn_planning_failure_into_valid_capability_stop(self):
        blocked={'requests':[],'blocked':{'kind':'invented','detail':'synthetic'}}
        self.run_review([UNCERTAIN,blocked,blocked])
        outcome=json.loads((self.root/'evidence_outcome.json').read_text(encoding='utf-8'))
        self.assertEqual(outcome['reason'],'evidence_planning_exhausted')
        self.assertNotIn('blocked',outcome)

    def test_valid_blocked_keeps_previous_unresolved_measurement(self):
        blocked={'requests':[],'blocked':{'kind':'measurement_error','detail':'No remaining supported measurement'}}
        self.run_review([UNCERTAIN,{'requests':[VIEW]},UNCERTAIN,blocked,blocked],status='incomplete')
        outcome=json.loads((self.root/'evidence_outcome.json').read_text(encoding='utf-8'))
        self.assertEqual(outcome['reason'],'capability_blocked')
        self.assertEqual(outcome['completed_rounds'],1)
        self.assertEqual(outcome['unresolved_checks'][0]['request'],VIEW)


if __name__ == "__main__":
    unittest.main()
