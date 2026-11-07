"""Static-mode targeted scene repair from external robot-trajectory feedback.

A new run is seeded from an immutable source run, bound to this source tree, and given planner
feedback bound to the source scene digest. One repair round then runs plan -> assets -> assemble,
like robot_operation.scene_bridge.SceneBridge.repair, with its progress saved in the run's own
ledger (state["robot_feedback"]) so an interrupted repair resumes where it stopped.
"""

import argparse
import json
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from asset_generation.contracts import GenerationConfig, read_json, require
from asset_generation.pipeline import snapshot
from asset_prep import implementation
from asset_prep.config import Config
from asset_prep.errors import PrepError, ValidationError
from asset_prep.storage import atomic_json, digest, file_hash, run_directory, run_lock, safe_path
from .contracts import SceneConfig, environment
from .feedback import planning_feedback

KIND = "robot_feedback"
CONFIG_FILES = ("provider.json", "generation.json", "scene.json", "environment.json")
# Identity fields a repair may not change; implementation and binding follow this source tree.
FIXED_IDENTITY = ("catalogue", "environment", "contract", "provider", "model", "base_url", "output_token_parameter",
                  "generation", "limits", "robot_files", "backend", "provenance")


def published(scene):
    """The scene a static run currently offers: the admitted publication, never a half-built round."""
    record = scene.get("published_scene") or {}
    if not record.get("artifact"):
        raise ValidationError("The source run has no published scene artifact")
    return record


def source_inputs(project, source, state):
    """Configuration, package and protocol the source run was generated from."""
    receipt = state.get(KIND)
    if receipt:
        return {**receipt["inputs"], "config_dir": str(source/KIND/"config")}
    for path in sorted((source.parent/"batches").glob("*/batch.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        entry = next((e for e in batch.get("entries", []) if e.get("run") == source.name), None)
        if entry:
            return {"config_dir": str(path.parent/"config"), "package": entry["package"], "protocol_pdf": entry.get("pdf")}
    return None


def build_run(project, root, inputs, budget, *, resume, retry_uncertain=False):
    from asset_generation.checks import load_model
    from asset_generation.local import LocalBackend
    from asset_generation.transport import transport_for
    from asset_prep.config import credentials
    from .benchmark_input import bind_environment
    from .inputs import catalog
    from .pipeline import SceneRun, code_identity
    config = root/KIND/"config"
    provider = Config.load(config/"provider.json")
    generation = GenerationConfig.load(config/"generation.json")
    limits = replace(SceneConfig(**read_json(config/"scene.json")), validation_mode="static",
                     max_model_calls=budget["max_model_calls"], max_reserved_tokens=budget["max_reserved_tokens"],
                     max_iterations=budget["max_iterations"])
    env = environment(read_json(config/"environment.json"))
    catalogue = catalog(Path(inputs["package"]), generation,
                        Path(inputs["protocol_pdf"]) if inputs.get("protocol_pdf") else None, provider.max_document_chars)
    robot = safe_path(project, env["robot"]["directory"])
    bind_environment(catalogue, env, robot)
    sandbox = LocalBackend(generation, project)
    sandbox.preflight()
    load_model(robot)
    return SceneRun(catalogue, env, inputs["contract"], provider, generation, limits, credentials(project), root,
                    transport_for(provider, root), sandbox, robot, project, resume, retry_uncertain,
                    expected_implementation=code_identity(project))


def start(project, source, root, feedback, max_calls, max_tokens, config_dir=None, package=None, protocol_pdf=None):
    """Create the run: freeze inputs, validate feedback against the source scene, bind identity."""
    from asset_prep.state_archive import read_state
    require(not root.exists() or not any(p.name != ".lock" for p in root.iterdir()),
            "Run directory is not empty; resume explicitly or use a new name")
    with run_lock(source):
        state = read_state(source/"state.json")
        source_state_sha256 = digest(state)
    scene = state.get("scene", {})
    require(scene.get("plan") and state["identity"]["limits"].get("validation_mode") == "static",
            "The source must be a planned static scene run")
    require(not any(c["status"] == "pending" for c in state["calls"]),
            "The source run has requests with unknown outcome; reconcile them first")
    record = published(scene)
    artifact = safe_path(source, record["artifact"])
    scene_sha256 = digest(snapshot(artifact))
    require(scene_sha256 == record["artifact_sha256"], "The published source scene was modified")
    checked = planning_feedback(feedback, scene_sha256, scene["plan"])
    inputs = source_inputs(project, source, state) or {}
    if config_dir: inputs["config_dir"] = str(config_dir)
    if package: inputs["package"] = str(package)
    if protocol_pdf: inputs["protocol_pdf"] = str(protocol_pdf)
    require(inputs.get("config_dir") and inputs.get("package"),
            "Source configuration not found; pass --config-dir and --package")
    resolved = {"package": str(Path(inputs["package"]).resolve()),
                "protocol_pdf": str(Path(inputs["protocol_pdf"]).resolve()) if inputs.get("protocol_pdf") else None,
                "contract": state["identity"]["contract"]}
    totals = {"model_calls": sum(c["kind"] in {"model", "model_search"} for c in state["calls"]),
              "accounted_tokens": sum(c["reported_total_tokens"] if c.get("settled_by_usage") else
                                      max(c["reserved_tokens"], c.get("reported_total_tokens", 0)) for c in state["calls"])}
    budget = {"max_calls": max_calls, "max_tokens": max_tokens, "inherited": totals,
              "max_model_calls": totals["model_calls"]+max_calls,
              "max_reserved_tokens": totals["accounted_tokens"]+max_tokens,
              "max_iterations": len(scene.get("repair_history", []))+1}
    source_identity = state["identity"]
    config_dir, inputs = Path(inputs["config_dir"]), resolved
    del state, scene
    root.mkdir(parents=True, exist_ok=True)
    config = root/KIND/"config"
    config.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        shutil.copyfile(config_dir/name, config/name)
    atomic_json(root/KIND/"feedback.json", feedback)
    run = build_run(project, root, inputs, budget, resume=False)
    identity = run.store.state["identity"]
    # Worktrees reach the same environment through their own .venv link (see the relocation continuation).
    same = lambda k, v: {f: x for f, x in v.items() if f != "python"} if k == "backend" else v
    changed = [k for k in FIXED_IDENTITY if same(k, identity.get(k)) != same(k, source_identity.get(k))]
    require(not changed, "Repair inputs differ from the source run: "+", ".join(changed))
    run.store.state[KIND] = {"version": 1, "phase": "seeding", "source_run": str(source), "source_state_sha256": source_state_sha256,
        "source_implementation": source_identity["implementation"], "implementation": identity["implementation"],
        "source_scene": {"artifact": record["artifact"], "sha256": scene_sha256, "directory": record.get("directory")},
        "inputs": {k: inputs[k] for k in ("package", "protocol_pdf")}, "budget": budget,
        "feedback": checked, "feedback_sha256": digest(feedback), "sessions": []}
    run.store.save()
    return run


def inject(scene, checked, feedback_sha256):
    """Hand validated planner rows (feedback.planning_feedback) to a static scene as its repair input.

    SceneRun.run refuses planner feedback in static mode; this entry point replaces only that
    gate, and route_repair then plans against these rows alone."""
    for key in ("activity", "stop_reason", "repair_error", "request_failure", "local_failure", "check_failure",
                "repair_planning", "last_decision"):
        scene.pop(key, None)
    scene.update(status="repairing_scene", feedback=deepcopy(checked), repair_pending=True)
    scene["feedback_receipts"] = scene.get("feedback_receipts", [])+[feedback_sha256]
    return scene


def install(run, source):
    """Copy the source stages and ledger into the new run (idempotent), then hand it the feedback."""
    from robot_operation.scene_bridge import seed
    from asset_prep.phase_budget import ceilings
    receipt = run.store.state[KIND]
    copied = seed(source, run.root)
    if copied["source_state_sha256"] != receipt["source_state_sha256"]:
        raise ValidationError("The source run changed while it was being copied")
    state = run.store.state
    state["calls"] = copied["calls"]
    totals = run.store.totals()
    limits = {"model_calls": run.limits.max_model_calls, "accounted_tokens": run.limits.max_reserved_tokens}
    # The whole repair allowance is the scene phase: --max-calls/--max-tokens are exact.
    state["phase_budget"] = {"version": 1, "entry": totals, "ceilings": {p: dict(limits) for p in ceilings(totals, limits)},
                             "phase": "scene"}
    scene = inject(copied["scene"], receipt["feedback"], receipt["feedback_sha256"])
    state["scene"] = scene
    state["source_identity"] = copied["source_identity"]
    state.setdefault("implementation_history", []).append({"kind": KIND, "at": time.time(), "source_run": receipt["source_run"],
        "source_state_sha256": receipt["source_state_sha256"], "from": receipt["source_implementation"],
        "to": receipt["implementation"], "model_calls_added": 0,
        "reason": "keep assets, reviews and call ledger; repair the assets and layout named by measured robot feedback"})
    receipt.update(phase="plan", call_start=len(state["calls"]), totals_start=totals,
                   history_start=len(scene.get("repair_history", [])), baseline_plan=deepcopy(scene["plan"]),
                   baseline_jobs={k: v["stage"] for k, v in scene["jobs"].items()},
                   baseline_stage_count=len(scene["stages"]))
    run.store.save()


def author(run, job):
    """Regenerate one job as collect() does: a failed replacement keeps the previous loadable version."""
    from .candidate import usable
    key = job["id"]
    old = run.state["jobs"].get(key)
    if job["role"] != "asset":
        from .layout_planning import refine
        refine(run)
    ready = run.generate_job(job, None, draft=True)
    candidate = run.state["jobs"][key]
    if ready:
        candidate.pop("regenerate", None)
    elif usable(run, old) and candidate is not old:
        run.state["jobs"][key] = old
        old.pop("regenerate", None)
        old["dirty"] = False
        old["replacement_failure"] = {"stage": candidate["stage"], "feedback": candidate.get("feedback")}
    run.store.save()
    return ready


def assess(run, published_scene, targets):
    """Static assessment; quality repairs of assets the directive did not name are not started here."""
    from .scene_first import assess_scene
    unresolved = run.state.setdefault("asset_repairs_unresolved", {})
    for key in run.state["jobs"]:
        if key not in targets and key not in unresolved:
            unresolved[key] = {"reason": "Outside the robot-feedback repair scope", KIND: True}
    try:
        return assess_scene(run, published_scene)
    finally:
        skipped = sorted(k for k, v in unresolved.items() if v.get(KIND))
        for key in skipped:
            unresolved.pop(key)
        run.store.save()


def repair(run):
    from .candidate import usable
    from .local_repair import precheck
    from .scene_first import publish
    state, receipt = run.state, run.store.state[KIND]
    if receipt["phase"] == "plan":
        if len(state.get("repair_history", [])) == receipt["history_start"]:
            retired = deepcopy(state.get("retired_repair_tasks"))
            routed = run.route_repair(use_review=False)
            # Scheduling against external rows must not retire the source run's own open tasks.
            if retired is None: state.pop("retired_repair_tasks", None)
            else: state["retired_repair_tasks"] = retired
            if not routed:
                receipt.update(phase="done", status="deferred",
                               reason=state.get("repair_error") or (state.get("stop_reason") or {}).get("detail") or state["status"])
                run.store.save()
                return
        receipt["phase"] = "assets"
        run.store.save()
    entry = state["repair_history"][receipt["history_start"]]
    targets = set(entry["action"]["jobs"])
    if receipt["phase"] == "assets":
        jobs = {j["id"]: j for j in state["plan"]["jobs"]}
        done = receipt.setdefault("jobs_done", [])
        for key in state["order"]:
            if key in done: continue
            current = state["jobs"].get(key)
            # Dependents are only revalidated (review_pending): the audit re-measures them, nobody re-authors them.
            if current is None or current.get("regenerate"):
                author(run, jobs[key])
            done.append(key)
            run.store.save()
        receipt["phase"] = "assemble"
        run.store.save()
    if receipt["phase"] == "assemble":
        jobs = {j["id"]: j for j in state["plan"]["jobs"]}
        for _ in range(8):
            if precheck(run):
                receipt.update(phase="done", status="rejected",
                               reason="The local recheck proved the edited assets regressed or did not change; the source scene is kept")
                break
            missing = [k for k in jobs if not usable(run, state["jobs"].get(k))]
            scene = publish(run, missing)
            if scene is None:
                receipt.update(phase="done", status="assembly_failed", reason="The repaired scene could not be assembled")
                break
            report, review = assess(run, scene, targets)
            if report is None:
                # A re-authored target failed its audit and went back to its author (scene_first.schedule_asset_repairs).
                for key in state["order"]:
                    if (state["jobs"].get(key) or {}).get("regenerate"):
                        author(run, jobs[key])
                continue
            from .recovery import decide
            action = decide(report, review)
            state["last_decision"] = {"action": action, "scene": state["published_scene"]["directory"]}
            # One external round: remaining static findings stay recorded for the next feedback round.
            state["status"] = "incomplete_scene" if missing else "repair_tasks_deferred" if action == "repair" else action
            state["stop_reason"] = {"reason": "robot_feedback_round_complete", "decision": action, "missing_jobs": missing,
                                    "detail": "The robot-feedback repair round was assembled and assessed; the robot benchmark judges it next"}
            receipt.update(phase="done", status="candidate" if not missing else "partial_candidate",
                           reason=None if not missing else "Some jobs are missing from the assembled scene")
            break
        else:
            receipt.update(phase="done", status="assessment_exhausted", reason="Asset quality repairs did not converge")
        if "selection" not in entry and entry.get("status") == "evaluated":
            # The robot benchmark judges this candidate; the static comparison is kept as information.
            entry["selection"] = KIND
        run.store.save()


def layout_changes(before, after):
    old = {i["id"]: i for i in before["instances"]}
    moved = [{"instance": i["id"], "before": {k: old[i["id"]].get(k) for k in ("position", "yaw", "placement", "table_id")},
              "after": {k: i.get(k) for k in ("position", "yaw", "placement", "table_id")}}
             for i in after["instances"] if i["id"] in old and i != old[i["id"]]]
    fields = [k for k in ("layout", "table_modifications", "edge_mounts", "end_effectors", "robot_posture")
              if before.get(k) != after.get(k)]
    return {"moved_instances": moved, "added_instances": sorted(set(i["id"] for i in after["instances"])-set(old)),
            "changed_fields": fields}


def write_result(run, started):
    from asset_generation.model import request_summary
    from collections import Counter
    state, store = run.state, run.store
    receipt = store.state[KIND]
    result = run.result() if state.get("plan") else {}
    calls = store.state["calls"][receipt.get("call_start", len(store.state["calls"])):]
    totals, start_totals = store.totals(), receipt.get("totals_start") or store.totals()
    history = state.get("repair_history", [])
    entry = history[receipt["history_start"]] if "history_start" in receipt and len(history) > receipt["history_start"] else None
    stages = state["stages"][receipt.get("baseline_stage_count", len(state["stages"])):]
    record = state.get("published_scene") or {}
    artifact = safe_path(run.root, record["artifact"]) if record.get("artifact") else None
    targets = set((entry or {}).get("target_checks", []))
    directive_jobs = set(((entry or {}).get("action") or {}).get("jobs", []))
    items = {i["id"]: i for i in state.get("plan", {}).get("instances", [])}
    changes = layout_changes(receipt["baseline_plan"], state["plan"]) if receipt.get("baseline_plan") else None
    moved = {m["instance"] for m in (changes or {}).get("moved_instances", [])}
    latest = state.get("latest") or {}
    sessions = receipt.get("sessions", [])
    document = {"version": 1, "kind": KIND, "status": receipt.get("status", "paused" if receipt["phase"] != "done" else "unknown"),
        "phase": receipt["phase"], "reason": receipt.get("reason"),
        "source_run": receipt["source_run"], "source_scene": receipt["source_scene"],
        "scene_artifact": str(artifact) if artifact else None,
        "scene_artifact_relative": record.get("artifact"),
        "scene_sha256": digest(snapshot(artifact)) if artifact and artifact.is_dir() else None,
        "scene_changed": bool(artifact) and record.get("artifact") != receipt["source_scene"]["artifact"],
        "reauthored_jobs": sorted({s["job"] for s in stages if s["kind"] == "asset"}),
        "revalidated_jobs": sorted({k for s in stages if s["kind"] == "repair" for k in s.get("layout_revalidation_jobs", [])}),
        "replacement_failures": {k: v["replacement_failure"] for k, v in state["jobs"].items()
                                 if v.get("replacement_failure") and k in directive_jobs},
        "repair_action": (entry or {}).get("action"), "repair_directive": state.get("repair_directive"),
        "layout_changes": changes,
        "usage": {"model_calls": totals["model_calls"]-start_totals["model_calls"],
                  "accounted_tokens": totals["accounted_tokens"]-start_totals["accounted_tokens"],
                  "reported_total_tokens": totals["reported_total_tokens"]-start_totals["reported_total_tokens"],
                  "by_phase": dict(Counter(c.get("phase", "unknown") for c in calls)),
                  "requests": request_summary(calls), "budget": receipt["budget"]},
        "wall_seconds": sum(s["ended"]-s["started"] for s in sessions if s.get("ended"))+(time.time()-started),
        "static_quality": {"run_status": result.get("status"), "report_status": (latest.get("report") or {}).get("status"),
                           "review_verdict": (latest.get("review") or {}).get("verdict"),
                           "scene_version": latest.get("directory"), "assessed_current": latest.get("artifact") == record.get("artifact"),
                           "comparison_with_source": (entry or {}).get("comparison"),
                           "local_verification": (entry or {}).get("local_verification")},
        "feedback": [{"id": row["id"], "kind": row["kind"], "instances": row["instances"],
                      "jobs": sorted({items[i]["job"] for i in row["instances"] if i in items}),
                      "targeted": row["id"] in targets,
                      "asset_changed": bool({items[i]["job"] for i in row["instances"] if i in items} & directive_jobs),
                      "moved": bool(set(row["instances"]) & moved)}
                     for row in receipt["feedback"]["checks"]],
        "feedback_sha256": receipt["feedback_sha256"], "implementation": receipt["implementation"]}
    atomic_json(run.root/"robot_feedback_result.json", document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", required=True, help="source run: a name under runs/ or a path")
    parser.add_argument("--run", required=True, help="new run name under runs/")
    parser.add_argument("--feedback", required=True, type=Path)
    parser.add_argument("--max-calls", type=int, default=160)
    parser.add_argument("--max-tokens", type=int, default=80000000)
    parser.add_argument("--config-dir", type=Path, help="provider/generation/scene/environment JSON of the source run")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--protocol-pdf", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-uncertain", action="store_true")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    if not getattr(sys, "_harness_runtime", None):
        print("Run through scripts/apply_robot_feedback.py (frozen source)", file=sys.stderr)
        return 2
    source = Path(args.source_run)
    source = (source if source.is_absolute() or source.exists() else project/"runs"/args.source_run).resolve()
    root = run_directory(project, args.run)
    feedback = read_json(args.feedback)
    started = time.time()
    run = None
    try:
        with run_lock(root):
            if (root/"state.json").exists():
                if not args.resume:
                    raise ValidationError("Run already exists; pass --resume to continue it")
                saved = read_json(root/KIND/"feedback.json")
                if digest(saved) != digest(feedback):
                    raise ValidationError("--resume must use the same feedback as the run was started with")
                from asset_prep.state_archive import read_state
                saved = read_state(root/"state.json")
                receipt = saved[KIND]
                if Path(receipt["source_run"]) != source:
                    raise ValidationError("--resume must use the same source run")
                inputs = {**receipt["inputs"], "contract": saved["identity"]["contract"]}
                del saved
                run = build_run(project, root, inputs, receipt["budget"], resume=True, retry_uncertain=args.retry_uncertain)
            else:
                if args.resume:
                    raise ValidationError("No run to resume")
                run = start(project, source, root, feedback, args.max_calls, args.max_tokens,
                            args.config_dir, args.package, args.protocol_pdf)
            receipt = run.store.state[KIND]
            if receipt["phase"] == "seeding":
                install(run, source)
                inputs = {**receipt["inputs"], "contract": run.store.state["identity"]["contract"]}
                run = build_run(project, root, inputs, receipt["budget"], resume=True, retry_uncertain=args.retry_uncertain)
            if run.store.state[KIND]["phase"] != "done":
                repair(run)
            document = write_result(run, started)
            run.store.state[KIND]["sessions"].append({"started": started, "ended": time.time()})
            run.store.save()
        print(json.dumps({k: document[k] for k in ("status", "scene_artifact", "scene_sha256", "reauthored_jobs", "usage")},
                         ensure_ascii=False))
        return 0 if document["status"] in {"candidate", "partial_candidate"} else 3
    except (PrepError, KeyboardInterrupt, OSError) as exc:
        if run is not None and KIND in run.store.state:
            run.state["status"] = "paused"
            run.store.state[KIND]["sessions"].append({"started": started, "ended": time.time()})
            run.store.state[KIND].setdefault("interruptions", []).append({"at": time.time(), "error": type(exc).__name__,
                                                                           "detail": str(exc)[:2000]})
            try:
                run.store.save()
                write_result(run, time.time())
            except Exception as secondary:
                exc.add_note("writing the pause record also raised: "+type(secondary).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
