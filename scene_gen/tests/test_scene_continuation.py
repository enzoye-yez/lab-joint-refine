import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from asset_generation.pipeline import snapshot
from asset_prep.errors import ValidationError
from asset_prep.storage import atomic_json, digest, file_hash
from asset_prep.state_archive import read_state

spec = importlib.util.spec_from_file_location("scene_continuation_script", Path(__file__).resolve().parents[1] / "scripts/prepare_scene_continuation.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class SceneContinuationTests(unittest.TestCase):
    def test_resume_copy_preserves_ledger_and_does_not_overwrite_completed_upgrade(self):
        import shutil
        atomic_json(self.root/'stage_review_upgrade.json',{'previous_upgrade':'inherited source receipt'})
        target=self.project/'runs/new';shutil.copytree(self.root,target)
        script.prepare(self.project,'old','new',self.before,resume_copy=True)
        self.assertEqual(read_state(target/'state.json')['calls'],self.state['calls'])
        with self.assertRaises(ValidationError):
            script.prepare(self.project,'old','new',self.before,resume_copy=True)

    def test_resume_copy_rejects_a_changed_ledger(self):
        import shutil
        target=self.project/'runs/new';shutil.copytree(self.root,target)
        atomic_json(target/'state.json',{**self.state,'changed':True})
        with self.assertRaises(ValidationError):
            script.prepare(self.project,'old','new',self.before,resume_copy=True)

    def test_resume_copy_rejects_modified_stage_bytes(self):
        import shutil
        target=self.project/'runs/new';shutil.copytree(self.root,target)
        (target/'stages/00001_asset/model.py').write_text('# modified')
        with self.assertRaises(ValidationError):
            script.prepare(self.project,'old','new',self.before,resume_copy=True)

    def test_context_packing_preserves_bytes_and_overlay_on_second_copy(self):
        from asset_generation.request_context import reference_bytes
        source=self.project/'contexts';folder=source/'binding';folder.mkdir(parents=True)
        payload='{"text":"原始证据","values":[1,2,3]}'.encode('utf-8')
        (folder/'reference.json').write_bytes(payload)
        target=self.project/'packed'
        receipt=script.copy_context_archive(source,target)
        self.assertTrue(receipt[0]['crc_verified'])
        self.assertEqual(receipt[0]['files'],1)
        self.assertFalse((target/'binding/reference.json').exists())
        self.assertEqual(reference_bytes(target/'binding','reference'),payload)
        overlay=target/'binding';overlay.mkdir();(overlay/'new.json').write_bytes(b'{"new":true}')
        second=self.project/'second';script.copy_context_archive(target,second)
        self.assertEqual(reference_bytes(second/'binding','reference'),payload)
        self.assertEqual(reference_bytes(second/'binding','new'),b'{"new":true}')

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name)
        code = self.project / "src/scene_generation/pipeline.py"
        code.parent.mkdir(parents=True)
        code.write_text("# old", encoding="utf-8")
        before = {str(code.relative_to(self.project)):file_hash(code)}
        self.before = self.project / "before.json"
        atomic_json(self.before, before)
        code.write_text("# updated review", encoding="utf-8")
        self.root = self.project / "runs/old"
        stage = self.root / "stages/00001_asset"
        (stage / "artifact").mkdir(parents=True)
        (stage / "artifact/model.xml").write_text("<mujoco/>", encoding="utf-8")
        (stage / "model.py").write_text("# archived", encoding="utf-8")
        (stage / "preview.png").write_bytes(b"synthetic-image")
        self.state = {"identity":{"implementation":digest(before)}, "calls":[],
                      "scene":{"status":"needs_review", "rounds":[],
                               "stages":[{"directory":"stages/00001_asset", "kind":"asset", "files":snapshot(stage)}],
                               "jobs":{"fixture":{"stage":"stages/00001_asset", "status":"needs_review",
                                       "source":"stages/00001_asset/model.py", "artifact":"stages/00001_asset/artifact",
                                       "preview_paths":["stages/00001_asset/preview.png"],
                                       "feedback":{"numerical":{"status":"pass"}}}}}}
        atomic_json(self.root / "state.json", self.state)
        atomic_json(self.root / "result.json", {"status":"needs_review", "budget":{"model_calls":24}})

    def prepare(self):
        return script.prepare(self.project, "old", "new", self.before)

    def test_copy_preserves_original_evidence_and_budget_requires_re_review(self):
        original = snapshot(self.root)
        result = self.prepare()
        self.assertEqual(result["model_calls_added"], 0)
        self.assertEqual({k:v for k,v in snapshot(self.root).items() if k != ".lock"}, original)
        state = json.loads((self.project / "runs/new/state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["scene"]["jobs"]["fixture"]["status"], "review_pending")
        self.assertEqual(state["calls"], self.state["calls"])
        self.assertEqual(state["scene"]["stages"], self.state["scene"]["stages"])
        result = json.loads((self.project / "runs/new/result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["budget"]["model_calls"], 24)
        with self.assertRaises(ValidationError):
            self.prepare()

    def test_evidence_pipeline_upgrade_keeps_unknown_ledger_and_invalidates_quality(self):
        module = self.project / "src/asset_generation/request_context.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("# developer context fixture", encoding="utf-8")
        self.state["identity"]["limits"]={"validation_mode":"static"}
        self.state["calls"]=[{"index":0,"key":"unknown","status":"pending","reserved_tokens":42}]
        self.state["scene"]["jobs"]["fixture"].update(quality_key="old",numeric_audit_key="old",
            stage="stages\\00001_asset",assembly_readiness={"loadable":True},feedback={"numerical":{"status":"partial"}})
        atomic_json(self.root/"state.json",self.state)
        atomic_json(self.root/"result.json",{"status":"needs_review","assembly":{"quality_accepted":True},"budget":{"model_calls":1}})
        before=snapshot(self.root)
        receipt=script.prepare(self.project,"old","new",self.before,"evidence_pipeline")
        current=json.loads((self.project/"runs/new/state.json").read_text(encoding="utf-8"))
        self.assertEqual(current["calls"],self.state["calls"])
        self.assertTrue(receipt["unknown_request_preserved"])
        self.assertNotIn("quality_key",current["scene"]["jobs"]["fixture"])
        self.assertNotIn("assembly_readiness",current["scene"]["jobs"]["fixture"])
        self.assertIn("max_context_tokens",current["identity"]["generation"])
        self.assertEqual(current["identity"]["generation"]["max_context_reads"], 8)
        self.assertFalse(json.loads((self.project/"runs/new/result.json").read_text(encoding="utf-8"))["assembly"]["quality_accepted"])
        self.assertEqual({k:v for k,v in snapshot(self.root).items() if k!=".lock"},before)

    def test_local_repair_upgrade_preserves_models_and_rechecks_scene(self):
        self.state['identity']['limits']={'validation_mode':'static'}
        self.state['scene']['jobs']['fixture'].update(status='assembly_ready',dirty=False)
        self.state['scene'].update(repair_pending=True,repair_error='修复没有产生变化')
        atomic_json(self.root/'state.json',self.state)
        original=snapshot(self.root)
        result=script.prepare(self.project,'old','new',self.before,'local_repair')
        copied=json.loads((self.project/'runs/new/state.json').read_text(encoding='utf-8'))
        self.assertEqual(result['model_calls_added'],0)
        self.assertEqual(copied['scene']['jobs'],self.state['scene']['jobs'])
        self.assertFalse(copied['scene']['repair_pending'])
        self.assertEqual(copied['scene']['stages'],self.state['scene']['stages'])
        self.assertEqual({k:v for k,v in snapshot(self.root).items() if k!='.lock'},original)

    def test_corrupt_evidence_is_rejected_before_copy(self):
        (self.root / "stages/00001_asset/model.py").write_text("modified", encoding="utf-8")
        with self.assertRaises(ValidationError):
            self.prepare()
        self.assertFalse((self.project / "runs/new").exists())

    def test_main_check_upgrade_revalidates_layout_without_changing_old_artifacts(self):
        self.state['identity']['limits']={'validation_mode':'static'}
        self.state['scene']['status']='needs_check_support'
        self.state['scene']['plan']={'jobs':[{'id':'fixture','role':'fixture'}],'instances':[]}
        self.state['scene']['jobs']['fixture'].update(status='assembly_ready',dirty=False)
        self.state['scene']['check_recovery']={'old':{'status':'completed','attempts':1}}
        for name in ('capabilities','dispatch','recovery_geometry'):
            (self.project/f'src/scene_generation/{name}.py').write_text('# developer fixture',encoding='utf-8')
        atomic_json(self.root/'state.json',self.state)
        original=snapshot(self.root)
        script.prepare(self.project,'old','new',self.before,'main_check_recovery')
        copied=json.loads((self.project/'runs/new/state.json').read_text(encoding='utf-8'))
        self.assertEqual(copied['scene']['jobs']['fixture']['status'],'review_pending')
        self.assertIn('layout_revalidation',copied['scene']['jobs']['fixture'])
        restored=read_state(self.project/'runs/new/state.json')
        self.assertEqual(restored['scene']['check_recovery_history'],[self.state['scene']['check_recovery']])
        self.assertEqual(copied['scene']['stages'],self.state['scene']['stages'])
        self.assertEqual({k:v for k,v in snapshot(self.root).items() if k!='.lock'},original)

    def test_check_upgrade_rechecks_without_rewriting_original_models(self):
        self.state['identity']['limits']={'validation_mode':'static'}
        self.state['scene']['jobs']['fixture'].update(status='assembly_ready',dirty=False)
        self.state['scene'].update(repair_pending=True,repair_error='old error')
        atomic_json(self.root/'state.json',self.state)
        code=self.project/'src/asset_generation/distance.py'
        code.parent.mkdir(parents=True)
        code.write_text('# synthetic checked query',encoding='utf-8')
        original=snapshot(self.root)
        result=script.prepare(self.project,'old','new',self.before,'evidence_checks')
        copied=json.loads((self.project/'runs/new/state.json').read_text(encoding='utf-8'))
        self.assertEqual(result['model_calls_added'],0)
        self.assertEqual(copied['scene']['jobs'],self.state['scene']['jobs'])
        self.assertEqual(copied['scene']['stages'],self.state['scene']['stages'])
        self.assertFalse(copied['scene']['repair_pending'])
        self.assertEqual({k:v for k,v in snapshot(self.root).items() if k!='.lock'},original)

    def test_unrelated_source_change_is_not_silently_accepted(self):
        (self.project / "src/scene_generation/checks.py").write_text("changed checks", encoding="utf-8")
        with self.assertRaises(ValidationError):
            self.prepare()

    def test_pending_paid_request_cannot_be_retried_by_migration(self):
        self.state["calls"] = [{"status":"pending"}]
        atomic_json(self.root / "state.json", self.state)
        with self.assertRaises(ValidationError):
            self.prepare()

    def test_evidence_upgrade_preserves_unknown_call_and_original_geometry(self):
        self.state["identity"]["limits"]={"validation_mode":"static"}
        self.state["calls"]=[{"status":"pending","reserved_tokens":1234}]
        self.state["scene"].update(status="paused",pending={"directory":"stages/00002_asset","kind":"asset"})
        pending=self.root/"stages/00002_asset"
        pending.mkdir()
        (pending/"model.py").write_text("# preserve incomplete workspace",encoding="utf-8")
        atomic_json(self.root/"state.json",self.state)
        before=snapshot(self.root)
        result=script.prepare(self.project,"old","new",self.before,"evidence_recovery")
        copied=json.loads((self.project/"runs/new/state.json").read_text(encoding="utf-8"))
        self.assertEqual(copied["calls"],self.state["calls"])
        self.assertTrue(result["unknown_request_preserved"])
        self.assertEqual(copied["scene"]["jobs"]["fixture"]["status"],"review_pending")
        self.assertEqual((self.project/"runs/new/stages/00002_asset/model.py").read_text(),"# preserve incomplete workspace")
        self.assertEqual({k:v for k,v in snapshot(self.root).items() if k != '.lock'},
                         {k:v for k,v in before.items() if k != '.lock'})

    def test_capacity_upgrade_preserves_asset_decisions_and_rebuilds_scene(self):
        (self.project/"src/scene_generation/pipeline.py").write_text("# old",encoding="utf-8")
        (self.project/"src/scene_generation/checks.py").write_text("# scene load scope",encoding="utf-8")
        scene = self.state["scene"]
        scene["jobs"]["fixture"].update(status="assembly_ready",dirty=False)
        scene.update(rounds=["failed"],latest={"report":{"checks":[{"id":"scene_build","detail":"模型规模超限"}]}},
                     repair_pending=True,repair_error="修复没有产生变化",feedback={"status":"fail"})
        atomic_json(self.root/"state.json",self.state)
        result = script.prepare(self.project,"old","new",self.before,"scene_capacity")
        self.assertEqual(result["review_pending"],[])
        copied = json.loads((self.project/"runs/new/state.json").read_text(encoding="utf-8"))
        self.assertEqual(copied["scene"]["jobs"],scene["jobs"])
        self.assertEqual(copied["scene"]["latest"],scene["latest"])
        self.assertFalse(copied["scene"]["repair_pending"])
        self.assertNotIn("feedback",copied["scene"])

    def test_transport_upgrade_preserves_ready_jobs_and_pending_stage(self):
        self.state["identity"]["limits"] = {"validation_mode":"static"}
        self.state["scene"].update(status="paused", repair_pending=True, feedback={"synthetic":True},
            pending={"directory":"stages/00002_asset", "kind":"asset"})
        (self.root / "stages/00002_asset").mkdir()
        self.state["scene"]["jobs"]["fixture"]["status"] = "assembly_ready"
        atomic_json(self.root / "state.json", self.state)
        original = file_hash(self.root / "state.json")
        result = script.prepare(self.project, "old", "new", self.before, "transport_recovery")
        copied = json.loads((self.project / "runs/new/state.json").read_text(encoding="utf-8"))
        self.assertEqual(copied["scene"], self.state["scene"])
        self.assertEqual(copied["calls"], self.state["calls"])
        self.assertEqual(file_hash(self.root / "state.json"), original)
        self.assertEqual(result["review_pending"], [])

    def test_repair_upgrade_carries_archived_directive_and_invalidates_dependencies(self):
        import copy
        (self.project/"src/scene_generation/pipeline.py").write_text("# old",encoding="utf-8")
        (self.project/"src/scene_generation/geometry.py").write_text("# conservative broadphase",encoding="utf-8")
        scene = self.state["scene"]
        scene["jobs"]["fixture"].update(status="assembly_ready",dirty=False)
        scene["jobs"]["tool"] = copy.deepcopy(scene["jobs"]["fixture"])
        scene.update(order=["fixture","tool"],plan={"jobs":[{"id":"fixture","depends_on":[]},{"id":"tool","depends_on":["fixture"]}]},
                     rounds=["failed"],latest={"report":{"checks":[{"id":"scene_build","detail":"近邻碰撞检查对数超限"}]}})
        action = {"jobs":["fixture"],"placements":[],"reason":"one model per instance"}
        directory = self.root/"stages/00002_repair"
        atomic_json(directory/"action.json",action)
        scene["stages"].append({"directory":"stages/00002_repair","kind":"repair","files":snapshot(directory)})
        atomic_json(self.root/"state.json",self.state)
        script.prepare(self.project,"old","new",self.before,"scene_repair")
        copied = json.loads((self.project/"runs/new/state.json").read_text(encoding="utf-8"))["scene"]
        self.assertEqual(copied["repair_directive"],action)
        self.assertEqual(copied["jobs"]["fixture"]["status"],"review_pending")
        self.assertTrue(all(job["dirty"] for job in copied["jobs"].values()))

    def test_preview_upgrade_requires_static_pass_and_preserves_old_uncertainty(self):
        self.state["identity"]["limits"] = {"validation_mode":"static"}
        scene = self.state["scene"]
        scene["jobs"]["fixture"].update(status="assembly_ready",dirty=False)
        scene.update(rounds=["previous"],latest={"report":{"status":"pass","checks":[]},"review":{"verdict":"uncertain"}})
        atomic_json(self.root/"state.json",self.state)
        script.prepare(self.project,"old","new",self.before,"static_preview")
        copied = json.loads((self.project/"runs/new/state.json").read_text(encoding="utf-8"))["scene"]
        self.assertEqual(copied["status"],"paused")
        self.assertEqual(copied["latest"]["review"]["verdict"],"uncertain")
        self.state["identity"]["limits"]["validation_mode"] = "full"
        atomic_json(self.root/"state.json",self.state)
        with self.assertRaises(ValidationError):
            script.prepare(self.project,"old","other",self.before,"static_preview")
