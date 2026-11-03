"""Stream a signed trajectory; emit only bounded windows and keep locators into the original record."""
from collections import deque
import gzip
import json
import numpy as np
from .common import need, read


def inspect_trace(context, query):
    need(context.run_root is not None and context.record is not None, "No reusable execution record", "missing_evidence")
    path = context.run_root/"steps.jsonl.gz"
    need(path.is_file(), "Execution produced no per-step trajectory", "missing_evidence")
    graph = read(context.run_root/"graph.json")
    outcome = read(context.run_root/"result.json")
    if query.get("node"):
        need(query["node"] in {n["id"] for n in graph["nodes"]}, "Trajectory query node does not exist")
    model, settings = context.model, context.settings
    bodies = query["bodies"]
    bids = {name: model.body(name).id for name in bodies}
    from .evidence import descendants
    from scene_generation.runtime import actor_groups
    selected = {b for bid in bids.values() for b in descendants(model, bid)}
    retention_pairs=[]
    if outcome.get("code")=="grasp_lost" and context.failure:
        actors=actor_groups(model,context.meta)
        for item in context.failure.get("evidence",{}).get("last",{}).values():
            rule=item.get("condition",{})
            if rule.get("kind")=="held" and not item.get("passed") and rule.get("arm") in actors:
                retention_pairs.append((set(actors[rule["arm"]]),descendants(model,model.body(rule["body"]).id)))
    tail, window = deque(maxlen=query["window"]), []
    first, count, scanned, maximum_force, penetration = None, 0, 0, 0.0, 0.0
    contact_pairs, positions, last_positions, conditions = {}, {}, {}, {}
    failure_step = context.failure.get("sim_steps") if context.failure else None
    def reject(value): raise ValueError("Trajectory contains a non-finite number: "+value)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            context.cancel_guard()
            scanned += 1
            need(scanned <= settings.diagnostic_trace_steps and len(line) <= 32_000_000,
                 "Trajectory analysis budget exhausted; the uncovered part must not be marked as passing", "diagnostic_budget")
            row = json.loads(line, parse_constant=reject)
            if failure_step is not None and row["step"] > failure_step: break
            if query.get("node") and row["node"] != query["node"]: continue
            count += 1
            for key,value in row.get("condition_evidence",{}).items():
                condition=value.get("condition",{})
                if bodies and condition.get("body") not in bodies: continue
                summary=conditions.setdefault(key,{"condition":condition,"first_failed_step":None,"last_step":row["step"],
                                                  "max_relative_drift_m":0.0,"max_rotation_drift_rad":0.0})
                summary["last_step"]=row["step"]
                if not value.get("passed") and summary["first_failed_step"] is None: summary["first_failed_step"]=row["step"]
                detail=value.get("evidence",{})
                summary["max_relative_drift_m"]=max(summary["max_relative_drift_m"],detail.get("relative_drift_m",0))
                summary["max_rotation_drift_rad"]=max(summary["max_rotation_drift_rad"],detail.get("rotation_drift_rad",0))
            hits = []
            for c in row["contacts"]:
                body_ids = [int(model.geom_bodyid[g]) for g in c["geoms"]]
                if selected and not (selected & set(body_ids)): continue
                names = [model.body(b).name for b in body_ids]
                item = {**c, "geoms": [model.geom(g).name or "#"+str(g) for g in c["geoms"]],
                        "bodies": names, "instances": context.instances(names)}
                hits.append(item)
                maximum_force = max(maximum_force, c["force_n"])
                penetration = max(penetration, -c["distance"])
                key = tuple(sorted(item["geoms"]))
                contact_pairs.setdefault(key, {"geoms": list(key), "instances": item["instances"],
                                               "first_step": row["step"], "max_force_n": 0.0, "max_penetration_m": 0.0})
                pair = contact_pairs[key]
                pair["last_step"] = row["step"]
                pair["max_force_n"] = max(pair["max_force_n"], c["force_n"])
                pair["max_penetration_m"] = max(pair["max_penetration_m"], -c["distance"])
                pair["limit_exceeded"] = pair["max_penetration_m"] > settings.penetration_m or pair["max_force_n"] > settings.contact_force_n
                pair["retention_failure_observed"] = (row["node"] == (context.failure or {}).get("node") and
                    any(set(c["geoms"]) & actor and set(body_ids) & target for actor,target in retention_pairs))
            for name, bid in bids.items():
                positions.setdefault(name, row["body_position"][bid])
                last_positions[name] = row["body_position"][bid]
            sample = {"step": row["step"], "time": row["time"], "node": row["node"], "contacts": hits[:32],
                      "contact_count": len(hits), "body_positions": {n: row["body_position"][b] for n, b in bids.items()}}
            exceeded = any(c["force_n"] > settings.contact_force_n or c["distance"] < -settings.penetration_m for c in hits)
            if first is None and exceeded:
                first = {"step": row["step"], "time": row["time"], "node": row["node"]}
                window = list(tail)
            if first is not None and len(window) < 2*query["window"]+1: window.append(sample)
            tail.append(sample)
    if first is None: window = list(tail)
    pairs = sorted(contact_pairs.values(), key=lambda r: r["first_step"])
    return {"status": "valid" if count else "unknown",
            "observations": {"steps": count, "first_limit_candidate": first, "max_contact_force_n": maximum_force,
                             "max_penetration_m": penetration, "contact_pairs": pairs[:128], "pair_count": len(pairs),
                             "condition_summaries":list(conditions.values())[:128],"condition_count":len(conditions),
                             "body_displacement_m": {n: float(np.linalg.norm(np.array(last_positions[n])-p)) for n, p in positions.items()},
                             "window": window},
            "coverage": {"kind": "trace", "node": query.get("node"), "bodies": bodies, "steps": count,
                         "through_failure_step": failure_step, "window_limit": query["window"]},
            "limitations": ["An engineering-threshold candidate is not a root cause; whole-object displacement is not slip relative to the gripper; full data remains in the original per-step record"],
            "source": {"path": str(path), "sha256": context.files["steps.jsonl.gz"]},
            "rows": [{"kind": "trace_contact", "status": "confirmed_observation", "repair_authorized": False,
                      "instances": p["instances"], "observation": p} for p in pairs[:128]]}
