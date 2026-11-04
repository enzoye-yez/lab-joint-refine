"""Offline copy tool for stage-review upgrades: re-verifies old evidence, keeps the ledger, and re-reviews existing geometry."""

import argparse
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from asset_generation.contracts import require
from asset_generation.pipeline import snapshot
from asset_prep import implementation
from asset_prep.storage import atomic_json, digest, file_hash, run_directory, run_lock, safe_path


ALLOWED_CHANGES = {"src/asset_generation/review.py", "src/scene_generation/pipeline.py",
                   "src/scene_generation/review.py"}

BINDING_CHANGES = {'src/runtime_bootstrap.py', 'src/asset_prep/implementation.py',
    'src/asset_generation/local.py', 'src/asset_generation/sandbox.py', 'src/asset_generation/worker.py',
    'src/asset_generation/model.py', 'src/asset_generation/distance.py', 'src/asset_generation/convex.py',
    'src/scene_generation/pipeline.py', 'src/scene_generation/cli.py', 'src/scene_generation/__main__.py', 'src/scene_generation/capabilities.py',
    'src/scene_generation/dispatch.py', 'src/scene_generation/progress.py',
    'src/scene_generation/scene_first.py','src/scene_generation/candidate.py',
    'src/asset_generation/rendering.py','src/asset_generation/checks.py','src/asset_generation/observation.py',
    'src/scene_generation/checks.py','src/scene_generation/preview.py'}


def copy_context_archive(source, target):
    """Preserve every original byte without reproducing hundreds of thousands of files."""
    if not source.exists():return []
    target.mkdir(parents=True,exist_ok=True)
    entries=list(source.iterdir())
    for entry in entries:
        if entry.is_file():shutil.copy2(entry,target/entry.name)
    def pack(entry):
        archive=target/(entry.name+'.zip')
        if (source/(entry.name+'.zip')).exists():
            shutil.copytree(entry,target/entry.name,dirs_exist_ok=True)
            return None  # Preserve overlays over an existing immutable pack.
        temp=archive.with_suffix('.zip.tmp')
        count=size=0
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as packed:
            for path in entry.rglob('*'):
                if path.is_file():
                    packed.write(path,path.relative_to(entry).as_posix())
                    count+=1;size+=path.stat().st_size
        with zipfile.ZipFile(temp) as packed:
            require(packed.testzip() is None,'historical context archive integrity check failed')
        temp.replace(archive)
        return {'directory':entry.name,'files':count,'original_bytes':size,
                'archive_bytes':archive.stat().st_size,'sha256':file_hash(archive),'crc_verified':True}
    receipts=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(pack,entry) for entry in entries if entry.is_dir()]
        for future in as_completed(futures):
            record=future.result()
            if record is not None:
                receipts.append(record)
                print(f"context archive {len(receipts)}: {record['files']} original files saved losslessly",flush=True)
    return receipts


def prepare(project, source_name, target_name, before_file=None, kind="stage_review", *, resume_copy=False):
    source, target = run_directory(project, source_name), run_directory(project, target_name)
    require((source / "state.json").is_file(), "source run does not exist")
    require(resume_copy or not target.exists(), "target run already exists; overwriting is forbidden")
    with run_lock(source), (run_lock(target) if resume_copy else nullcontext()):
        if resume_copy:
            require((target/'state.json').is_file() and file_hash(target/'state.json')==file_hash(source/'state.json'),
                    'only an interrupted copy whose ledger has not changed can be resumed; overwriting a run that was already upgraded or executed is forbidden')
        from asset_prep.state_archive import read_state, write_state
        state = read_state(source / "state.json")
        from asset_prep import implementation as manifest_source
        archive_only = kind == 'archive_revalidation'
        if archive_only:
            require(before_file is None, 'archive revalidation does not pose as an old-source upgrade; do not supply an old source manifest that is not used')
            before = None
        else:
            require(before_file is not None, 'a source upgrade requires --before-sources; verification cannot be skipped automatically when it is missing')
            before = json.loads(Path(before_file).read_text(encoding="utf-8"))
            legacy = digest(before)
            normalized = manifest_source.implementation_digest(before)
            require(state["identity"]["implementation"] in {legacy, normalized},
                    "pre-upgrade source manifest does not match the old run identity")
            before = manifest_source.normalize_manifest(before)
        after = manifest_source.source_manifest(project)
        prefixed = lambda m: {("src/" + key): value for key, value in m.items()}
        changed = None if archive_only else {p for p in prefixed(before).keys() | prefixed(after).keys()
                   if prefixed(before).get(p) != prefixed(after).get(p)}
        options = {"evidence_pipeline":BINDING_CHANGES | ALLOWED_CHANGES | {
                       "src/asset_prep/storage.py", "src/asset_prep/state_archive.py", "src/robot_operation/scene_bridge.py",
                       "src/scene_generation/assembly.py", "src/scene_generation/checkpoints.py",
                       "src/scene_generation/recovery.py","src/scene_generation/prompts.py",
                       "src/scene_generation/gallery.py",
                       "src/asset_generation/contracts.py","src/asset_generation/model.py","src/asset_generation/transport.py",
                       "src/asset_generation/request_context.py","src/asset_generation/check_recovery.py",
                       "src/asset_generation/surface_contact.py","src/asset_generation/material.py","src/asset_prep/errors.py",
                       "src/scene_generation/geometry.py","src/asset_generation/evidence.py",
                       "src/scene_generation/context.py","src/scene_generation/evidence.py","src/scene_generation/evidence_state.py",
                       "src/scene_generation/inspection.py","src/scene_generation/support.py"},
                   "stage_review":ALLOWED_CHANGES,
                   "archive_revalidation":set(),
                   "implementation_binding":BINDING_CHANGES,
                   "main_check_recovery":{"src/asset_generation/distance.py","src/asset_generation/evidence.py",
                       "src/asset_generation/local.py","src/asset_generation/sandbox.py","src/asset_generation/worker.py",
                       "src/scene_generation/geometry.py","src/scene_generation/checks.py","src/scene_generation/inspection.py",
                       "src/scene_generation/contracts.py","src/scene_generation/pipeline.py","src/scene_generation/prompts.py",
                       "src/scene_generation/checkpoint.py","src/scene_generation/recovery.py","src/scene_generation/edge_mounts.py",
                       "src/scene_generation/capabilities.py","src/scene_generation/dispatch.py","src/scene_generation/recovery_geometry.py",
                       "src/scene_generation/faults.py"},
                   "evidence_checks":{"src/asset_generation/distance.py", "src/asset_generation/evidence.py",
                       "src/scene_generation/geometry.py", "src/scene_generation/checks.py", "src/scene_generation/inspection.py",
                       "src/scene_generation/assembly.py", "src/scene_generation/evidence.py", "src/scene_generation/contracts.py",
                       "src/scene_generation/cli.py", "src/scene_generation/pipeline.py"},
                   "local_repair":{"src/asset_prep/storage.py", "src/asset_generation/model.py",
                       "src/scene_generation/contracts.py", "src/scene_generation/table.py", "src/scene_generation/assembly.py",
                       "src/scene_generation/checks.py", "src/scene_generation/prompts.py", "src/scene_generation/pipeline.py",
                       "src/scene_generation/cli.py"},
                   "agent_context":{"src/asset_generation/model.py", "src/asset_generation/codex_transport.py",
                       "src/scene_generation/scene_first.py"},
                   "review_contract":{"src/scene_generation/review.py"},
                   "phase_budget":{"src/asset_prep/phase_budget.py"},
                   "budget_limits":{"src/scene_generation/scene_first.py",
                       # presentation renders only; no check or review reads it
                       "src/asset_generation/studio.py",
                       # line endings only: restored to the repository's CRLF
                       "src/asset_generation/model.py", "src/asset_generation/codex_transport.py",
                       "src/scene_generation/review.py"},
                   "repair_loop":{"src/asset_generation/probes.py", "src/asset_generation/motion.py",
                       "src/asset_generation/checks.py", "src/asset_generation/prompts.py",
                       "src/asset_generation/model.py", "src/asset_generation/review.py",
                       "src/scene_generation/scene_first.py", "src/scene_generation/pipeline.py",
                       "src/scene_generation/review.py",
                       # a repaired asset whose recheck times out is rejected instead of pausing the scene
                       "src/asset_generation/sandbox.py", "src/asset_generation/local.py",
                       "src/scene_generation/local_repair.py",
                       # presentation renders only; no check or review reads it
                       "src/asset_generation/studio.py"},
                   "tool_draft":{"src/scene_generation/pipeline.py", "src/scene_generation/scene_first.py",
                       "src/scene_generation/tool_contract.py", "src/asset_generation/selfcheck.py",
                       "src/asset_generation/codex_transport.py",
                       # scene check reports above 8 MB no longer stop the scene stage
                       "src/asset_generation/sandbox.py"},
                   # a flush face contact was measured as a full-thickness penetration
                   "collision_depth":{"src/asset_generation/distance.py"},
                   "repair_schedule":{"src/scene_generation/scene_first.py", "src/scene_generation/pipeline.py",
                       "src/scene_generation/repair_schedule.py", "src/scene_generation/local_repair.py",
                       "src/scene_generation/evidence_state.py", "src/scene_generation/contracts.py",
                       "src/scene_generation/prompts.py", "src/asset_generation/request_context.py",
                       "src/asset_generation/sandbox.py"},
                   # a regressed asset reverts alone; prefetch waits for re-authored dependencies
                   "partial_revert":{"src/scene_generation/local_repair.py", "src/scene_generation/scene_first.py"},
                   # a mixed scene round keeps its fixes; authors see why a repair was reverted or did nothing
                   "repair_feedback":{"src/scene_generation/local_repair.py", "src/scene_generation/pipeline.py"},
                   # pre-existing defects a re-review finds on unchanged assets are repair work, not a regression
                   "discovered_defects":{"src/scene_generation/progress.py", "src/scene_generation/pipeline.py",
                       "src/scene_generation/prompts.py"},
                   # a measured mount contact passes beside unmeasured pairs; a checker crash leaves the round unjudged
                   "checker_failure":{"src/scene_generation/inspection.py", "src/scene_generation/pipeline.py"},
                   "asset_checks":{"src/asset_generation/probes.py", "src/asset_generation/motion.py",
                       "src/asset_generation/checks.py", "src/asset_generation/local.py"},
                   "transport_recovery":{"src/asset_generation/contracts.py", "src/asset_generation/model.py",
                       "src/asset_generation/provider_diagnostics.py", "src/asset_generation/transport.py",
                       "src/asset_prep/network.py", "src/scene_generation/gallery.py", "src/scene_generation/pipeline.py"},
                   "scene_capacity":{"src/asset_generation/checks.py", "src/scene_generation/checks.py",
                                     "src/scene_generation/preview.py", "src/scene_generation/runtime.py"},
                   "scene_repair":{"src/scene_generation/geometry.py", "src/scene_generation/pipeline.py",
                                   "src/scene_generation/review.py"},
                   "static_preview":{"src/scene_generation/checks.py", "src/scene_generation/pipeline.py",
                                     "src/scene_generation/review.py", "src/scene_generation/cli.py"},
                   "evidence_recovery":{"src/asset_generation/evidence.py", "src/asset_generation/transport.py",
                       "src/asset_generation/model.py", "src/asset_generation/cli.py", "src/asset_generation/local.py",
                       "src/asset_generation/sandbox.py", "src/asset_generation/worker.py",
                       "src/scene_generation/evidence.py", "src/scene_generation/pipeline.py",
                       "src/scene_generation/cli.py", "src/scene_generation/preview.py"}}
        require(kind in options, "invalid upgrade kind")
        allowed = options[kind]
        require(archive_only or changed and {p.replace("\\", "/") for p in changed} <= allowed,
                "source changes outside stage review exist; this upgrade is not supported")
        scene = state["scene"]
        require(not scene.get("pending") or kind in {"evidence_recovery", "transport_recovery", "evidence_pipeline", "agent_context", "review_contract", "phase_budget", "budget_limits", "repair_loop", "tool_draft", "collision_depth", "repair_schedule", "partial_revert", "repair_feedback", "discovered_defects", "checker_failure"}, "an unarchived workspace exists")
        if kind in {'main_check_recovery', 'implementation_binding', 'archive_revalidation', 'evidence_pipeline'}:
            require(state['identity']['limits']['validation_mode']=='static'
                    and scene['status'] in {'needs_check_support','needs_environment','needs_review','paused',
                                            'scene_repair_budget_exhausted'},
                    'a main-check upgrade only supports static runs that have not passed acceptance')
            old_recovery = scene.pop('check_recovery',None)
            if old_recovery:
                scene.setdefault('check_recovery_history',[]).append(old_recovery)
            # Actions, calls and old evidence are kept; the old run has no scope receipt, so it cannot be reused as grounds for a current pass.
            scene.pop('scene_review_resume',None)
        elif kind in {"evidence_recovery", "transport_recovery", "local_repair", "evidence_checks"}:
            require(state["identity"]["limits"]["validation_mode"] == "static"
                    and scene["status"] in {"needs_review","paused"}, "only static runs with insufficient evidence or paused by network errors may be continued")
        elif kind == "budget_limits":
            # The run limits become BUDGET (scripts/run_scene_batch.py); phase ceilings follow them.
            from asset_prep.phase_budget import ceilings
            from run_scene_batch import BUDGET
            allocation = state.get("phase_budget")
            if allocation:
                allocation["ceilings"] = ceilings(allocation["entry"], {"model_calls": BUDGET["max_model_calls"],
                                                                        "accounted_tokens": BUDGET["max_reserved_tokens"]})
            # Key each standing asset review by what it judged, so the re-audit under the new
            # implementation keeps verdicts whose checks and images come out unchanged. Tool jobs
            # depend on the live layout context and are re-reviewed.
            from scene_generation.scene_first import review_key
            for plan_job in scene["plan"]["jobs"]:
                job = scene["jobs"].get(plan_job["id"]) or {}
                numerical = job.get("feedback", {}).get("numerical") or {}
                visual = job.get("feedback", {}).get("visual") or {}
                if plan_job["role"] != "asset" or numerical.get("status") == "fail" or visual.get("verdict") not in {"accept", "repair"}:
                    continue
                images = [file_hash(safe_path(source, path)) for path in job.get("preview_paths", [])]
                dependencies = {k: scene["jobs"].get(k, {}).get("assembly_readiness") for k in plan_job["depends_on"]}
                job["review_key"] = review_key(plan_job, job["assembly_readiness"], None, dependencies, numerical, images)
        elif kind in {"tool_draft", "collision_depth", "partial_revert", "repair_feedback", "discovered_defects", "checker_failure"}:
            # The end-effector is accepted on its static checks and repaired in the operation stage;
            # the implementation digest re-audits every asset, and unchanged reviews stand.
            scene.pop("scene_review_resume", None)
        elif kind == "repair_schedule":
            # The undecided repair was compared with a baseline the old checker assessed; void it and
            # return to that baseline (run_static does the same on start), then key every standing
            # review by what it judged under the new review key, so unchanged verdicts are kept.
            from copy import deepcopy
            from types import SimpleNamespace
            from scene_generation.scene_first import review_key, dependency_geometry
            stages = {s["directory"].replace("\\", "/"): s for s in scene["stages"]}
            for entry in scene.get("repair_history", []):
                undecided = entry["status"] == "applied" or entry["status"] == "evaluated" and "selection" not in entry
                if not undecided or not entry.get("baseline"):
                    continue
                baseline = stages[entry["baseline"]["scene"]]
                require(baseline.get("requested_plan", baseline["plan"]) == scene["plan"], "the voided repair changed the plan")
                jobs = {}
                for key, directory in entry["baseline"]["jobs"].items():
                    jobs[key] = {k: deepcopy(v) for k, v in stages[directory].items() if k not in {"directory", "kind", "files"}}
                    jobs[key].update(stage=directory, dirty=False)
                    jobs[key].setdefault("artifact", directory+"/artifact")
                    jobs[key].setdefault("artifact_stage", directory)
                restored = {k: deepcopy(v) for k, v in baseline.items() if k not in {"kind", "files"}}
                scene.update(jobs=jobs, latest=restored, published_scene=deepcopy(restored))
                for field in ("current_assessment", "repair_directive", "repair_tasks", "scene_review_resume"):
                    scene.pop(field, None)
                entry.update(status="voided", selection="voided", void_reason="The checker changed after the baseline scene was assessed; "
                    "the candidate was not compared, its asset versions were set aside and the baseline is assessed again with the current checker")
            view = SimpleNamespace(state=scene)
            for plan_job in scene["plan"]["jobs"]:
                job = scene["jobs"].get(plan_job["id"]) or {}
                numerical = job.get("feedback", {}).get("numerical") or {}
                visual = job.get("feedback", {}).get("visual") or {}
                if plan_job["role"] != "asset" or numerical.get("status") == "fail" or visual.get("verdict") not in {"accept", "repair"}:
                    continue
                images = [file_hash(safe_path(source, path)) for path in job.get("preview_paths", [])]
                job["review_key"] = review_key(plan_job, job["assembly_readiness"], None, dependency_geometry(view, plan_job), numerical, images)
            # Repairs are no longer capped per round.
            state["identity"]["limits"].pop("max_repairs_per_round", None)
        elif kind == "repair_loop":
            # Checker and repair-loop change: the implementation digest re-audits every asset, and
            # failed assets then go back to their authors within their own repair allowance.
            scene.pop("scene_review_resume", None)
        elif kind == "phase_budget":
            # Only the scene share changed; the ceiling is recomputed from the saved entry totals
            # and the run limits (the final ceiling), never from what has been spent since.
            from asset_prep.phase_budget import ceilings
            require(scene["status"] == "phase_budget_exhausted", "only runs that exhausted the scene budget are continued")
            allocation = state["phase_budget"]
            allocation["ceilings"] = ceilings(allocation["entry"], allocation["ceilings"]["final"])
        elif kind == "asset_checks":
            # Checker faults stopped the run; the new implementation digest re-audits every asset.
            require(state["identity"]["limits"]["validation_mode"] == "static"
                    and scene["status"] in {"needs_environment", "paused"}, "only static runs stopped by a checker fault are continued")
            old_recovery = scene.pop("check_recovery", None)
            if old_recovery:
                scene.setdefault("check_recovery_history", []).append(old_recovery)
            scene.pop("active_check_recovery", None)
            scene.pop("scene_review_resume", None)
        elif kind in {"agent_context", "review_contract"}:
            # Only how author requests reach the agent changed; every asset decision carries over.
            require(scene["status"] in {"generating_candidates", "reviewing_assets", "paused"}, "only runs interrupted during asset generation or asset review are continued")
        elif kind == "stage_review":
            require(not scene.get("rounds") and not scene.get("latest"), "a stage-review upgrade only supports runs that have not been assembled")
        else:
            failures = scene.get("latest", {}).get("report", {}).get("checks", [])
            expected_error = "模型规模超限" if kind == "scene_capacity" else "近邻碰撞检查对数超限"
            if kind == "static_preview":
                require(state["identity"]["limits"]["validation_mode"] == "static"
                        and scene["latest"]["report"]["status"] == "pass"
                        and scene["latest"]["review"]["verdict"] == "uncertain",
                        "a static-preview upgrade only applies to static runs whose engineering checks passed but whose visual evidence is insufficient")
            else:
                require(any(row.get("id") == "scene_build" and expected_error in row.get("detail", "") for row in failures),
                        "a scene upgrade only applies to a confirmed matching engineering limit")
            require(all(j["status"] in {"candidate_ready", "assembly_ready"} and not j.get("dirty")
                        for j in scene["jobs"].values()), "a scene-capacity upgrade does not change any asset review result")
        unknown = any(c["status"] == "pending" for c in state["calls"])
        require(not unknown or kind in {"evidence_recovery", "transport_recovery", "evidence_pipeline", "agent_context", "review_contract", "phase_budget", "budget_limits", "repair_loop", "tool_draft", "collision_depth", "repair_schedule", "partial_revert", "repair_feedback", "discovered_defects", "checker_failure"}, "requests with unknown outcome exist; reconcile the billing ledger first")
        for call in state["calls"]:
            if call["status"] == "done":
                require(file_hash(safe_path(source, call["file"])) == call["sha256"], "call record digest mismatch")
        stages = {}
        for stage in scene["stages"]:
            require(snapshot(safe_path(source, stage["directory"])) == stage["files"], "archived evidence digest mismatch")
            stages[stage["directory"].replace("\\", "/")] = stage
        if archive_only:
            require(scene.get('plan') and set(scene['jobs']) == {j['id'] for j in scene['plan']['jobs']}
                    and all(j['status'] in {'candidate_ready','assembly_ready'} and not j.get('dirty')
                            for j in scene['jobs'].values()),
                    'archive revalidation only accepts static runs whose assets are all archived and that have not passed full scene acceptance')
        recheck = []
        for key, job in scene["jobs"].items():
            if kind in {"agent_context", "review_contract", "asset_checks", "phase_budget", "budget_limits", "repair_loop", "tool_draft", "collision_depth", "repair_schedule", "partial_revert", "repair_feedback", "discovered_defects", "checker_failure"}:
                break  # no job is marked; asset_checks re-audits through the implementation digest
            if kind=="evidence_pipeline":
                owner=job.get("artifact_stage",job["stage"]).replace("\\", "/")
                require(owner in stages and job.get("artifact"),"asset awaiting revalidation has no archived version")
                artifact=safe_path(source,job["artifact"])
                require(artifact.is_relative_to(safe_path(source,owner)) and (artifact/"model.xml").is_file(),"artifact awaiting revalidation is invalid")
                for field in ("quality_key","numeric_audit_key","assembly_readiness"):job.pop(field,None)
                job.update(status="review_pending",dirty=True,needs_implementation_recheck=True)
                recheck.append(key)
                continue
            if job["status"] not in {"candidate_ready", "assembly_ready", "needs_review", "failed"}:
                continue
            if job["status"] == "failed" and job.get("feedback", {}).get("numerical", {}).get("status") != "pass":
                continue
            owner = job.get("artifact_stage", job["stage"]).replace("\\", "/")
            require(owner in stages, "asset version is not archived")
            require(job.get("feedback", {}).get("numerical", {}).get("status") == "pass", "asset awaiting re-review did not pass engineering checks")
            require(job.get("source") and job.get("artifact") and job.get("preview_paths"), "asset awaiting re-review is missing its model or images")
            for relative in [job["source"], *job["preview_paths"]]:
                path = safe_path(source, relative)
                stage_root = safe_path(source, owner)
                belongs = path.is_relative_to(stage_root) and path.relative_to(stage_root).as_posix() in stages[owner]["files"]
                # A read-only recheck may render new images of the original geometry; a verified stage record must explicitly bind them to the original asset.
                if not belongs and relative in job['preview_paths']:
                    for archived in stages.values():
                        evidence_root = safe_path(source, archived['directory'])
                        if (not path.is_relative_to(evidence_root)
                                or path.relative_to(evidence_root).as_posix() not in archived['files']
                                or 'record.json' not in archived['files']):
                            continue
                        evidence = json.loads((evidence_root/'record.json').read_text(encoding='utf-8'))
                        belongs = (evidence.get('kind') in {'asset_review','review_pending'}
                                   and evidence.get('job') == key and evidence.get('artifact_stage') == owner
                                   and evidence.get('artifact') == job['artifact'])
                        if belongs:
                            break
                require(belongs, "model or images awaiting re-review do not belong to a verified archive")
            artifact = safe_path(source, job["artifact"])
            require(artifact.is_relative_to(safe_path(source, owner)) and (artifact / "model.xml").is_file(),
                    "artifact of asset awaiting re-review is invalid")
            layout_recheck=(kind=='main_check_recovery' and
                            next(j['role'] for j in scene['plan']['jobs'] if j['id']==key)!='asset')
            if kind == "stage_review" or kind in {'implementation_binding','archive_revalidation'} or (kind == "evidence_recovery" and job["status"] == "needs_review") or layout_recheck:
                job.update(status="review_pending", dirty=True, artifact_stage=owner)
                if kind in {'implementation_binding','archive_revalidation'}:
                    job['needs_implementation_recheck'] = True
                if kind=='main_check_recovery':
                    job['layout_revalidation']={'reason':'the old version did not pass an authoritative layout; re-verify the original fixture at its actual position; geometry is not edited by hand',
                                                'current_layout':digest(scene['plan']['instances'])}
                recheck.append(key)
        if kind == "scene_repair":
            # Inherit only archived model decisions; do not re-guess from free text which asset should be repaired.
            actions = [json.loads(safe_path(source, s["directory"]+"/action.json").read_text(encoding="utf-8"))
                       for s in scene["stages"] if s["kind"] == "repair"]
            directive = next((a for a in reversed(actions) if a["jobs"]), None)
            require(directive is not None, "no archived asset repair decision to inherit")
            dirty = set(directive["jobs"])
            for key in scene["order"]:
                job = next(j for j in scene["plan"]["jobs"] if j["id"] == key)
                if dirty.intersection(job["depends_on"]):
                    dirty.add(key)
            for key in dirty:
                scene["jobs"][key]["dirty"] = True
            for key in directive["jobs"]:
                scene["jobs"][key]["status"] = "review_pending"
                recheck.append(key)
            scene["repair_directive"] = directive
        require(recheck or kind != "stage_review", "no reusable asset awaiting re-review")
        # Refuse to copy links; the original state file and all archives are left unmodified.
        for path in source.rglob("*"):
            require(not path.is_symlink() and not (hasattr(path, "is_junction") and path.is_junction()), "run directory contains links")
        receipt = {"kind":kind+"_upgrade", "at":time.time(), "source_run":source_name,
                   "source_state_sha256":file_hash(source / "state.json"),
                   "from":state["identity"]["implementation"], "to":digest(after),
                   "changed_files":None if archive_only else sorted(changed),
                   "legacy_source_verified":not archive_only,
                   "before_sources":before, "after_sources":after,
                   "review_pending":recheck, "model_calls_added":0,
                   "unknown_request_preserved":unknown,
                   "pending_workspace_files":snapshot(safe_path(source,scene["pending"]["directory"])) if scene.get("pending") else {},
                   "reason":("the old source manifest was not saved, so the old implementation cannot be fully verified; only the digest-verified original assets, ledger and budget are continued; "
                             "everything is re-measured, re-reviewed and assembly-checked, and old pass evidence is not inherited; this is a continuation after engineering intervention, not a success of the original frozen experiment"
                             if archive_only else
                             "keep assets, call ledger and action history; bind the new implementation, re-measure and re-review existing assets, and rebuild the check scene; old pass conclusions are not inherited"
                             if kind == 'implementation_binding' else
                             "keep the original assets, scene and call records; upgrade local saving, actual-support checks and constrained tabletop repair; after rebuilding checks the model decides; the experiment model is not edited by hand"
                             if kind == "local_repair" else
                             "the original run stays unchanged; the copy enables main-check recovery, actual-layout passing and re-checking; old scene conclusions are not inherited as a pass"
                             if kind == "main_check_recovery" else
                             "keep the original model and ledger; enable distance-evidence consistency verification, targeted checks and bounded evidence-supplement retries; rebuild the re-check scene; old conclusions are not inherited as a pass"
                             if kind == "evidence_checks" else
                             "keep assets, reviews, pending stages and call ledger; only upgrade transport recovery and asset display; acceptance conclusions are unchanged"
                             if kind == "transport_recovery" else
                             "keep assets, checks, pending stages and call ledger; in coding-agent mode, requests deliver the whole context (oversized parts written to files) and prefetch in parallel; acceptance conclusions are unchanged"
                             if kind == "agent_context" else
                             "keep assets, checks and call ledger; the review deferred-item format is spelled out in the prompt and synonymous keys are accepted; reviews that failed format are re-reviewed by the original flow; existing conclusions are unchanged"
                             if kind == "review_contract" else
                             "keep assets, review records and call ledger; fix configuration button probing, button reset judgment, connection-release probe duration and multi-configuration check time limits; "
                             "all assets are re-measured and re-reviewed with the new implementation; old check conclusions are not inherited"
                             if kind == "asset_checks" else
                             "keep assets, checks, reviews and call ledger; the total budget is unchanged; the scene-stage quota follows the new share and shares 90% of the total with the operation stage"
                             if kind == "phase_budget" else
                             "keep assets, checks, reviews and call ledger; the model call count and total token limits are doubled from the original configuration; review conclusions for unchanged assets are carried over"
                             if kind == "budget_limits" else
                             "keep assets, review records and call ledger; fix probe misjudgments, add slide-hold and lock-interlock checks, and repair failing assets directly in the asset stage; "
                             "all assets are re-measured and re-reviewed with the new implementation"
                             if kind == "repair_loop" else
                             "keep assets, reviews and call ledger; the end-effector is accepted on its static and printability checks and handed to the operation stage; "
                             "scene check reports up to 64 MB are read"
                             if kind == "tool_draft" else
                             "keep assets, reviews and call ledger; flush face contacts are no longer measured as full-thickness penetrations; "
                             "every asset and the scene are re-checked, and unchanged reviews stand"
                             if kind == "collision_depth" else
                             "keep assets, reviews and call ledger; the repair compared with an old-checker baseline is voided and the baseline re-assessed; "
                             "every open task is repaired in one round with parallel authoring, checks and reviews; tasks the current evidence no longer reports are retired"
                             if kind == "repair_schedule" else
                             "keep assets, reviews and call ledger; an asset whose local recheck regressed goes back to its baseline version with its edited dependents, "
                             "and the other assets and placement changes go on to the scene check; prefetch no longer authors a job before a dependency this sweep re-authors"
                             if kind == "partial_revert" else
                             "keep assets, reviews and call ledger; a scene round that fixed something and also made some assets worse reverts only those assets and checks the scene again; "
                             "authors see the checks that got a repair reverted and repairs that did not change the model"
                             if kind == "repair_feedback" else
                             "keep assets, reviews and call ledger; a defect a re-review finds on an asset whose geometry the round did not change "
                             "is new repair work and no longer rejects the round"
                             if kind == "discovered_defects" else
                             "keep assets, reviews and call ledger; the right-arm mount check no longer crashes the scene checker, "
                             "and a round whose candidate the checker failed to assess is checked again instead of being rejected"
                             if kind == "checker_failure" else
                             "keep existing geometry and ledger; re-review after enabling read-only evidence supplementation; unknown requests stay pending and automatic resend is forbidden"
                             if kind == "evidence_recovery" else
                             "keep existing geometry and engineering checks; re-run per-stage review; old conclusions are not automatically upgraded to a pass"
                             if kind == "stage_review" else
                             "keep all unverified items; re-run engineering checks and close-up review; deliver only the current static layout preview"
                             if kind == "static_preview" else
                             "keep the old report and re-check; fix scene engineering capacity or broadphase, and fully inherit archived repair decisions")}
        if resume_copy:
            for stage in scene['stages']:
                require(snapshot(safe_path(target,stage['directory']))==stage['files'],'a stage of the interrupted copy is incomplete or modified')
            for call in state['calls']:
                if call.get('file'):
                    require(file_hash(safe_path(target,call['file']))==call['sha256'],'a response of the interrupted copy is incomplete or modified')
        else:
            shutil.copytree(source, target, ignore=lambda directory,names:
                [name for name in names if name=='.lock' or (Path(directory)==source and name=='request_context')])
        receipt['request_context_archives']=copy_context_archive(source/'request_context',target/'request_context')
        if kind=="evidence_pipeline":
            from asset_generation.contracts import GenerationConfig
            state["identity"]["generation"]=GenerationConfig(**state["identity"].get("generation",{})).identity()
            scene.pop("current_assessment",None)
            receipt["reason"]="keep the original assets, scene, ledger and execution history; the copy binds the new evidence data flow and context budget, re-measures and re-reviews, and does not inherit old pass conclusions; unknown requests still block sending"
        state["identity"]["implementation"] = manifest_source.implementation_digest(after)
        state['identity']['binding'] = implementation.from_identity(project, state['identity'])
        implementation.archive_sources(project, target, state['identity']['implementation'])
        state.setdefault("implementation_history", []).append(receipt)
        scene["status"] = "paused"
        if kind not in {"stage_review", "transport_recovery", "agent_context", "review_contract", "phase_budget", "budget_limits", "repair_loop"}:
            scene.update(repair_pending=False, repair_error=None)
            scene.pop("feedback", None)
        write_state(target / "state.json", state)
        atomic_json(target / "stage_review_upgrade.json", receipt)
        result = json.loads((target / "result.json").read_text(encoding="utf-8"))
        result.update(status="paused", jobs={k:v["status"] for k,v in scene["jobs"].items()}, repair_error=None,
                      continuation_note=receipt["reason"])
        if kind == "evidence_pipeline" and "assembly" in result:
            result["assembly"]["quality_accepted"]=False
            result.setdefault("unverified",[]).append("engineering version changed; old check conclusions are kept only as history; re-measure and re-review required")
        if kind == "stage_review":
            result.update(deferred_checks=[], blocking_reasons=[])
        atomic_json(target / "result.json", result)
    return {"run":target_name, "review_pending":recheck, "model_calls_added":0, "unknown_request_preserved":unknown}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--before-sources", type=Path)
    parser.add_argument("--kind", choices=("evidence_pipeline", "archive_revalidation", "implementation_binding", "stage_review", "scene_capacity", "scene_repair", "static_preview", "evidence_recovery", "transport_recovery", "local_repair", "evidence_checks", "main_check_recovery", "agent_context", "review_contract", "asset_checks", "phase_budget", "budget_limits", "repair_loop", "tool_draft", "collision_depth", "repair_schedule", "partial_revert", "repair_feedback", "discovered_defects", "checker_failure"), default="stage_review")
    args = parser.parse_args()
    print(json.dumps(prepare(PROJECT, args.source, args.target, args.before_sources, args.kind), ensure_ascii=False))
