"""Deterministic (no model call) repair of a static scene run from already-verified edits.

A repair harness verifies a fix in simulation first (a counterfactual edit of the published scene)
and then hands the verified numeric edit here. Nothing is re-planned:

- layout edits (move_instance, clearance) go through contracts.repair_action, then
  arrangement.arrange with per-instance overrides (instance["arrangement"]), then validate_plan;
- joint-parameter and geometry edits (joint_param, geometry_requirement) are not patched into the
  asset: they are written into the owning job's requirements and the job is marked
  dirty/regenerate exactly as SceneRun.route_repair does, so the normal generate_job -> precheck
  -> publish path re-authors and re-checks it.

The run is seeded, built and assembled like external_repair (same phases, ledger key
"directed_repair"). Edits document:

    {"scene_sha256": <published source scene digest>,
     "edits": [{"kind": "move_instance" | "clearance" | "joint_param" | "geometry_requirement",
                "instance": <instance id>, "job": <job id, optional>, "params": {...},
                "requirement": <verified requirement in words>, "evidence": {...}}]}

params: move_instance {dx, dy, dyaw, frame: "world" (default) | "row"} (metres/radians; in the row
frame x runs along the bench and y toward the wall); clearance {gap_m, setback_m, back_m,
dock_standoff_m} (absolute values, arrangement.OVERRIDES ranges); joint_param and
geometry_requirement: any non-empty JSON mapping, passed to the author as stated.
"""

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

from asset_generation.contracts import read_json, require
from asset_prep.errors import PrepError, ValidationError
from asset_prep.storage import atomic_json, digest, run_directory, run_lock

KIND = "directed_repair"
DOCUMENT = "edits.json"
LAYOUT_KINDS = ("move_instance", "clearance")
JOB_KINDS = ("joint_param", "geometry_requirement")
MOVE_KEYS = {"dx": 1., "dy": 1., "dyaw": math.pi}
CLEARANCE_KEYS = ("gap_m", "setback_m", "back_m", "dock_standoff_m")
MAX_EDITS = 64
MAX_TEXT = 2000
MAX_JSON = 4000


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_edits(document, scene_sha256, plan):
    """Validate an edits document against the published source scene; returns normalized edits."""
    from .arrangement import OVERRIDES
    require(isinstance(document, dict) and set(document) - {"note", "source"} == {"scene_sha256", "edits"},
            "Edits must give exactly scene_sha256 and edits")
    require(document["scene_sha256"] == scene_sha256, "The edits were verified against a different scene")
    edits = document["edits"]
    require(isinstance(edits, list) and 0 < len(edits) <= MAX_EDITS, "Edits must be a non-empty list of at most %d" % MAX_EDITS)
    items = {i["id"]: i for i in plan["instances"]}
    jobs = {j["id"]: j for j in plan["jobs"]}
    seen, out = set(), []
    for index, edit in enumerate(edits):
        require(isinstance(edit, dict) and {"kind", "instance", "params", "requirement", "evidence"} <= set(edit)
                <= {"kind", "instance", "job", "params", "requirement", "evidence"}, "Edit %d: invalid fields" % index)
        kind, key = edit["kind"], edit["instance"]
        require(kind in LAYOUT_KINDS+JOB_KINDS, "Edit %d: unknown kind %r" % (index, kind))
        require(isinstance(key, str) and key in items, "Edit %d: unknown instance %r" % (index, key))
        item = items[key]
        job = edit.get("job", item["job"])
        require(isinstance(job, str) and job in jobs, "Edit %d: unknown job %r" % (index, job))
        require(isinstance(edit["requirement"], str) and edit["requirement"].strip() and len(edit["requirement"]) <= MAX_TEXT,
                "Edit %d: requirement must be a non-empty statement" % index)
        require(isinstance(edit["evidence"], dict) and len(json.dumps(edit["evidence"])) <= MAX_JSON,
                "Edit %d: evidence must be a mapping" % index)
        params = edit["params"]
        require(isinstance(params, dict) and params and len(json.dumps(params)) <= MAX_JSON,
                "Edit %d: params must be a non-empty mapping" % index)
        if kind in LAYOUT_KINDS:
            require("layout" in plan, "Edit %d: layout edits need an arranged (mobile) scene" % index)
            require(item["placement"] in {"table", "floor"}, "Edit %d: only table or floor instances can be moved" % index)
            require(job == item["job"], "Edit %d: a layout edit belongs to its instance's job" % index)
            require((kind, key) not in seen, "Edit %d: duplicate %s edit of %s" % (index, kind, key))
            seen.add((kind, key))
        if kind == "move_instance":
            require(set(params) <= set(MOVE_KEYS) | {"frame"} and set(params) & set(MOVE_KEYS),
                    "Edit %d: move_instance takes dx, dy, dyaw and frame" % index)
            require(params.get("frame", "world") in {"world", "row"}, "Edit %d: frame is world or row" % index)
            require("frame" not in params or not item.get("support_instance") or params["frame"] == "world",
                    "Edit %d: a supported instance moves in the world frame" % index)
            for name, bound in MOVE_KEYS.items():
                require(name not in params or number(params[name]) and abs(params[name]) <= bound,
                        "Edit %d: %s out of range" % (index, name))
        elif kind == "clearance":
            require(not item.get("support_instance"), "Edit %d: clearance applies to instances standing in a row" % index)
            require(set(params) <= set(CLEARANCE_KEYS), "Edit %d: clearance takes %s" % (index, ", ".join(CLEARANCE_KEYS)))
            for name, value in params.items():
                lo, hi = OVERRIDES[name]
                require(number(value) and lo <= value <= hi, "Edit %d: %s out of range [%g, %g]" % (index, name, lo, hi))
        out.append({"id": "directed:%d:%s:%s" % (index, kind, key), "kind": kind, "instance": key, "job": job,
                    "params": deepcopy(params), "requirement": edit["requirement"].strip(),
                    "evidence": deepcopy(edit["evidence"])})
    return {"scene_sha256": scene_sha256, "edits": out}


def row_yaw(plan, item):
    """Yaw of the row frame an arranged instance stands in (x along the row, y toward the wall)."""
    tables = {t["id"]: t for t in plan["layout"]["tables"]}
    if item["placement"] == "table" and item.get("table_id") in tables:
        return float(tables[item["table_id"]]["yaw"])
    dock = next((d for d in plan["layout"].get("docks", []) if d["instance"] == item["id"]), None)
    # The dock faces the row: its heading is the row's +Y (arrangement.dock).
    return float(dock["pose"][2])-math.pi/2 if dock else 0.


def descendants(plan, roots):
    found, grown = set(roots), True
    while grown:
        grown = False
        for item in plan["instances"]:
            if item.get("support_instance") in found and item["id"] not in found:
                found.add(item["id"])
                grown = True
    return found


def plan_edits(plan, checked, shapes, env, limits, asset_ids):
    """The updated plan and repair action of a set of checked edits (pure; no run state)."""
    from .arrangement import arrange, overrides
    from .contracts import repair_action, validate_plan
    from .layout import rotate, validate_layout
    edits = checked["edits"]
    items = {i["id"]: i for i in plan["instances"]}
    placements, arranged = [], {}
    for edit in edits:
        if edit["kind"] not in LAYOUT_KINDS:
            continue
        item = items[edit["instance"]]
        record = arranged.setdefault(item["id"], dict(item.get("arrangement") or {}))
        params = edit["params"]
        if edit["kind"] == "clearance":
            record.update(params)
            continue
        delta = [float(params.get("dx", 0.)), float(params.get("dy", 0.))]
        dyaw = float(params.get("dyaw", 0.))
        yaw_row = row_yaw(plan, item)
        world = rotate(delta, yaw_row) if params.get("frame") == "row" else delta
        placements.append({"id": item["id"], "yaw": round(math.remainder(item["yaw"]+dyaw, 2*math.pi), 6),
                           "position": [round(item["position"][0]+world[0], 4), round(item["position"][1]+world[1], 4),
                                        item["position"][2]]})
        if item.get("support_instance"):
            # A supported part keeps its new pose on its support (layout.follow_supports); no row override.
            arranged.pop(item["id"])
            continue
        local = rotate(world, -yaw_row)
        offset = [a+b for a, b in zip(record.get("offset_m", [0., 0.]), local)]
        record["offset_m"] = [round(float(x), 4) for x in offset]
        record["yaw_offset"] = round(math.remainder(record.get("yaw_offset", 0.)+dyaw, 2*math.pi), 6)
    jobs = sorted({e["job"] for e in edits if e["kind"] in JOB_KINDS})
    value = {"jobs": jobs, "placements": placements,
             "reason": "Directed repair of simulation-verified edits: "+", ".join(e["id"] for e in edits)}
    changes = [{"job": e["job"], "operation": e["kind"], "parameters": e["params"]} for e in edits if e["kind"] in JOB_KINDS]
    if changes:
        value["geometry_changes"] = changes
    allowed = descendants(plan, {e["instance"] for e in edits if e["kind"] in LAYOUT_KINDS})
    # Clearance alone changes only arrangement overrides, which repair_action does not carry.
    updated = repair_action(value, plan, jobs, sorted(allowed), limits, env) if jobs or placements else deepcopy(plan)
    by_id = {i["id"]: i for i in updated["instances"]}
    for key, record in arranged.items():
        by_id[key]["arrangement"] = record
        try:
            overrides(by_id[key])
        except ValueError as exc:
            raise ValidationError("%s: %s" % (key, exc)) from exc
    if arranged or placements:
        updated = arrange(updated, shapes, env)
        if not any(e["kind"] == "clearance" for e in edits):
            # A verified move is applied exactly: when re-arranging would also move other instances
            # or benches (the published plan is not the arrangement of its current assets, or some
            # asset has no loadable shape), the edit is refused instead of shifting unverified parts.
            before = {i["id"]: i for i in plan["instances"]}
            stray = sorted(i["id"] for i in updated["instances"] if i["id"] not in allowed and i != before.get(i["id"]))
            require(not stray and updated["layout"]["tables"] == plan["layout"]["tables"],
                    "Re-arranging for the verified move would also move unverified instances or benches: "
                    + ", ".join(stray or ["tables"]))
    blocks = {}
    for edit in edits:
        if edit["kind"] in JOB_KINDS:
            blocks.setdefault(edit["job"], []).append(
                "\n\nVerified repair requirement (%s, instance %s): %s\nVerified parameters: %s\nSimulation evidence: %s"
                % (edit["id"], edit["instance"], edit["requirement"], json.dumps(edit["params"], sort_keys=True),
                   json.dumps(edit["evidence"], sort_keys=True)[:MAX_TEXT]))
    for job in updated["jobs"]:
        if job["id"] in blocks:
            job["requirements"] += "".join(blocks[job["id"]])
    _, order = validate_plan(updated, asset_ids, limits, env)
    validate_layout(updated, env)
    require(digest(updated) != digest(plan), "The directed edits produced no change")
    return updated, value, order


def adopt(scene, receipt):
    """Prepare the copied scene for the directed plan phase (no planner feedback is injected)."""
    for key in ("activity", "stop_reason", "repair_error", "request_failure", "local_failure", "check_failure",
                "repair_planning", "last_decision"):
        scene.pop(key, None)
    scene.update(status="repairing_scene", repair_pending=True)
    scene["feedback_receipts"] = scene.get("feedback_receipts", [])+[receipt["feedback_sha256"]]
    return scene


def tasks_of(checked):
    return [{"id": e["id"], "check_id": e["id"], "kind": e["kind"], "instances": [e["instance"]], "job": e["job"],
             "requirement": e["requirement"], "parameters": e["params"], "evidence": e["evidence"],
             "binding": digest(e), "external_feedback": {"detail": e["requirement"], "evidence": e["evidence"],
                                                         "verified": True}}
            for e in checked["edits"]]


def record_repair(run, updated, value, checked):
    """Commit one directed repair round the way SceneRun.route_repair commits a planned one."""
    from .capabilities import repair_signature, repair_mechanism
    from .pipeline import mark_repair_jobs, repair_scope
    from .robot_contract import validate_model_posture
    state = run.state
    run.validate_prepared_plan(updated)
    validate_model_posture(updated, run.robot_joints)
    run.before_repair(value)
    order = state["order"]
    from .contracts import validate_plan
    _, state["order"] = validate_plan(updated, run.catalogue["asset_ids"], run.limits, run.env)
    try:
        dirty, revalidate = repair_scope(state, updated, value["jobs"])
    except Exception:
        state["order"] = order
        raise
    latest = state.get("latest", {})
    tasks = tasks_of(checked)
    destination = run.begin("repair")
    atomic_json(destination/"action.json", value)
    atomic_json(destination/DOCUMENT, checked)
    state.setdefault("repair_history", []).append({
        "input_key": run.repair_input_key(), "action": value, "cycle": latest.get("directory", str(len(state.get("rounds", [])))),
        "problem_key": digest(checked), "action_key": repair_signature(value), "mechanism": repair_mechanism(value),
        "status": "applied", "kind": KIND,
        "baseline": {"scene": latest["directory"], "jobs": {k: v["stage"] for k, v in state["jobs"].items()}}
        if run.limits.validation_mode == "static" and latest.get("assessed") else None,
        "target_checks": [t["check_id"] for t in tasks], "task_bindings": {t["check_id"]: t["binding"] for t in tasks},
        "call_start": len(run.store.state["calls"])})
    state["plan"] = updated
    state["repair_directive"] = value
    state["repair_tasks"] = {key: [t for t in tasks if t["job"] == key] for key in value["jobs"]}
    dirty, revalidate, pending = mark_repair_jobs(state, updated, value["jobs"], dirty, revalidate)
    run.commit("repair", {"call": None, "directed": True, "dirty_jobs": sorted(dirty),
                          "layout_revalidation_jobs": sorted(revalidate-dirty), "pending_generation_jobs": sorted(pending),
                          "plan": updated})
    state.pop("repair_planning", None)
    state["repair_pending"] = False
    run.store.save()
    return {"dirty_jobs": sorted(dirty), "revalidate_jobs": sorted(revalidate-dirty)}


def plan_phase(run):
    from .arrangement import shapes_for
    state, receipt = run.state, run.store.state[KIND]
    if len(state.get("repair_history", [])) == receipt["history_start"]:
        checked = receipt["feedback"]
        layout = any(e["kind"] in LAYOUT_KINDS for e in checked["edits"])
        shapes = shapes_for(run, state["plan"]) if layout else {}
        updated, value, _ = plan_edits(state["plan"], checked, shapes, run.env, run.limits, run.catalogue["asset_ids"])
        record_repair(run, updated, value, checked)
    receipt["phase"] = "assets"
    run.store.save()


def write_result(run, started):
    from .external_repair import layout_changes
    from asset_generation.pipeline import snapshot
    from asset_prep.storage import safe_path
    state, store = run.state, run.store
    receipt = store.state[KIND]
    history = state.get("repair_history", [])
    entry = history[receipt["history_start"]] if "history_start" in receipt and len(history) > receipt["history_start"] else None
    stages = state["stages"][receipt.get("baseline_stage_count", len(state["stages"])):]
    record = state.get("published_scene") or {}
    artifact = safe_path(run.root, record["artifact"]) if record.get("artifact") else None
    changes = layout_changes(receipt["baseline_plan"], state["plan"]) if receipt.get("baseline_plan") else None
    moved = {m["instance"] for m in (changes or {}).get("moved_instances", [])}
    targets = {e["instance"] for e in receipt["feedback"]["edits"] if e["kind"] in LAYOUT_KINDS}
    totals, start_totals = store.totals(), receipt.get("totals_start") or store.totals()
    sessions = receipt.get("sessions", [])
    document = {"version": 1, "kind": KIND, "status": receipt.get("status", "paused" if receipt["phase"] != "done" else "unknown"),
        "phase": receipt["phase"], "reason": receipt.get("reason"),
        "source_run": receipt["source_run"], "source_scene": receipt["source_scene"],
        "scene_artifact": str(artifact) if artifact else None, "scene_artifact_relative": record.get("artifact"),
        "scene_sha256": digest(snapshot(artifact)) if artifact and artifact.is_dir() else None,
        "scene_changed": bool(artifact) and record.get("artifact") != receipt["source_scene"]["artifact"],
        "reauthored_jobs": sorted({s["job"] for s in stages if s["kind"] == "asset"}),
        "revalidated_jobs": sorted({k for s in stages if s["kind"] == "repair" for k in s.get("layout_revalidation_jobs", [])}),
        "replacement_failures": {k: v["replacement_failure"] for k, v in state["jobs"].items() if v.get("replacement_failure")},
        "repair_action": (entry or {}).get("action"), "layout_changes": changes,
        # Instances the arrangement moved although no edit named them (e.g. neighbours of a wider clearance).
        "collateral_moves": sorted(moved-targets),
        "edits": [{"id": e["id"], "kind": e["kind"], "instance": e["instance"], "job": e["job"],
                   "moved": e["instance"] in moved} for e in receipt["feedback"]["edits"]],
        "usage": {"model_calls": totals["model_calls"]-start_totals["model_calls"],
                  "accounted_tokens": totals["accounted_tokens"]-start_totals["accounted_tokens"], "budget": receipt["budget"]},
        "wall_seconds": sum(s["ended"]-s["started"] for s in sessions if s.get("ended"))+(time.time()-started),
        "edits_sha256": receipt["feedback_sha256"], "implementation": receipt["implementation"]}
    atomic_json(run.root/"directed_repair_result.json", document)
    return document


def main(argv=None):
    from . import external_repair as er
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", required=True, help="source run: a name under runs/ or a path")
    parser.add_argument("--run", required=True, help="new run name under runs/")
    parser.add_argument("--edits", required=True, type=Path)
    parser.add_argument("--max-calls", type=int, default=160, help="model calls for re-authoring edited jobs")
    parser.add_argument("--max-tokens", type=int, default=80000000)
    parser.add_argument("--config-dir", type=Path, help="provider/generation/scene/environment JSON of the source run")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--protocol-pdf", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-uncertain", action="store_true")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    if not getattr(sys, "_harness_runtime", None):
        print("Run through scripts/apply_directed_repair.py (frozen source)", file=sys.stderr)
        return 2
    source = Path(args.source_run)
    source = (source if source.is_absolute() or source.exists() else project/"runs"/args.source_run).resolve()
    root = run_directory(project, args.run)
    edits = read_json(args.edits)
    started = time.time()
    run = None
    try:
        with run_lock(root):
            if (root/"state.json").exists():
                if not args.resume:
                    raise ValidationError("Run already exists; pass --resume to continue it")
                if digest(read_json(root/KIND/DOCUMENT)) != digest(edits):
                    raise ValidationError("--resume must use the same edits as the run was started with")
                from asset_prep.state_archive import read_state
                saved = read_state(root/"state.json")
                receipt = saved[KIND]
                if Path(receipt["source_run"]) != source:
                    raise ValidationError("--resume must use the same source run")
                inputs = {**receipt["inputs"], "contract": saved["identity"]["contract"]}
                del saved
                run = er.build_run(project, root, inputs, receipt["budget"], resume=True,
                                   retry_uncertain=args.retry_uncertain, kind=KIND)
            else:
                if args.resume:
                    raise ValidationError("No run to resume")
                run = er.start(project, source, root, edits, args.max_calls, args.max_tokens, args.config_dir,
                               args.package, args.protocol_pdf, kind=KIND, check=check_edits, document=DOCUMENT)
            receipt = run.store.state[KIND]
            if receipt["phase"] == "seeding":
                er.install(run, source, kind=KIND, adopt=adopt,
                           reason="keep assets, reviews and call ledger; apply simulation-verified directed edits")
                inputs = {**receipt["inputs"], "contract": run.store.state["identity"]["contract"]}
                run = er.build_run(project, root, inputs, receipt["budget"], resume=True,
                                   retry_uncertain=args.retry_uncertain, kind=KIND)
            if run.store.state[KIND]["phase"] == "plan":
                plan_phase(run)
            if run.store.state[KIND]["phase"] != "done":
                er.execute(run, kind=KIND)
            document = write_result(run, started)
            run.store.state[KIND]["sessions"].append({"started": started, "ended": time.time()})
            run.store.save()
        print(json.dumps({k: document[k] for k in ("status", "scene_artifact", "scene_sha256", "reauthored_jobs",
                                                   "collateral_moves", "usage")}, ensure_ascii=False))
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
