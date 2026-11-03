"""Single-parameter comparison from the same full start state; runs through the original executor and never modifies the real execution site."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .common import need, read, digest, OperationError, atomic_json
from .scene import load


def paired_probe(context, query):
    from .runtime import Session
    from .recording import Recorder
    from .contracts import validate_graph
    need(context.run_root is not None and context.evidence_root is not None,
         "A local comparison requires a complete run record and a separate evidence output directory", "missing_evidence")
    need(context.record.get("physics_started", True), "Dynamics admission not passed; local probing is not allowed")
    task, graph = read(context.run_root/"task.json"), read(context.run_root/"graph.json")
    original = next((n for n in graph["nodes"] if n["id"] == query["node"]), None)
    need(original is not None and original["operation"] not in {"device", "wait", "observe", "check"},
         "A comparison must target an existing mechanical operation")
    parameter = query["parameter"]
    need(parameter in original["parameters"], "A comparison must not add an unused parameter")
    need(original["parameters"][parameter] != query["value"], "Comparison parameter is unchanged")
    if parameter == "targets":
        before, after = original["parameters"][parameter], query["value"]
        need(isinstance(after, list) and len(before) == len(after) and
             all({k: v for k, v in a.items() if k not in {"position", "rotation", "offset"}} ==
                 {k: v for k, v in b.items() if k not in {"position", "rotation", "offset"}} for a, b in zip(before, after)),
             "A target comparison must not change the arm, interface, target object or reference frame")
    if parameter in {"positions", "controls"}:
        need(isinstance(query["value"], dict) and set(query["value"]) == set(original["parameters"][parameter]),
             "A comparison must not change the set of controlled joints or actuators")
    snapshot = (context.failure if query["state"] == "failure" else context.node_start if query["state"] == "node_start"
                else read(context.run_root/"initial.json"))
    need(snapshot is not None, "Missing full comparison start state")
    if query["state"] != "initial":
        need(snapshot.get("node") == original["id"], "Failure snapshot does not belong to the node under comparison")
    root = Path(context.evidence_root)/digest({"query": query, "binding": context.binding(query)})
    need(not root.exists(), "A local probe directory already exists; an interrupted probe cannot be retried implicitly", "interrupted_probe")
    root.mkdir(parents=True)
    outcomes = {}
    for label in ("baseline", "variant"):
        context.guard()
        model, data, meta, manifest = load(context.scene, context.scene_hash)
        session = Session(model, data, meta, manifest, task, context.settings)
        session.restore(deepcopy(snapshot))
        initial_time = float(data.time)
        session.cancel_guard = context.cancel_guard
        node = deepcopy(original)
        if label == "variant": node["parameters"][parameter] = deepcopy(query["value"])
        node.update(timeout=min(float(original["timeout"]), query["seconds"]), on_success="success", on_failure="failed")
        local_graph = {**deepcopy(graph), "entry": node["id"], "nodes": [node],
                       "diagnostic_probe": {"source_graph_sha256": digest(graph), "case": label, "query": query}}
        validate_graph(local_graph, task, context.scene_hash, context.settings)
        (root/label).mkdir()
        recorder = Recorder(root/label, mode="trial", images=False)
        recorder.start(session, context.scene_hash, task, local_graph, context.scene)
        session.recorder = recorder
        result = {"status": "interrupted", "mode": "diagnostic_probe"}
        try:
            need(not session.unbound_events, "Mechanical event has no detector", "capability_missing")
            session.begin(node)
            session.guard()
            need(session.evidence.evaluate(node["precondition"]), "Local probe start state does not satisfy the node precondition", "probe_precondition")
            session.execute(node)
            while not session.evidence.evaluate(node["completion"]): session.tick()
            result.update(status="success", code=None)
        except OperationError as exc:
            atomic_json(root/label/"failure_state.json", getattr(session, "failure_snapshot", session.snapshot()))
            result.update(status="failed", code=exc.code, detail=exc.detail)
        finally:
            result.update(sim_steps=session.sim_steps-snapshot["sim_steps"],
                          simulated_seconds=float(data.time)-initial_time,
                          completion_evidence=deepcopy(session.evidence.last))
            recorder.close(result)
        outcomes[label] = {**result, "directory": str(root/label)}
    valid = all(r.get("code") not in {"probe_precondition", "capability_missing", "simulation_unstable", "execution_error"}
                for r in outcomes.values())
    improved = valid and outcomes["baseline"]["status"] != "success" and outcomes["variant"]["status"] == "success"
    return {"status": "valid" if valid else "unknown", "observations": {"cases": outcomes, "variant_improved": improved},
            "coverage": {"kind": "paired_actuator_probe", "node": query["node"], "parameter": parameter,
                         "state_sha256": digest(snapshot), "seconds_per_case": query["seconds"]},
            "limitations": ["Shows only the effect of this local parameter change; does not prove a unique root cause or full task success"],
            "rows": [{"kind": "trajectory_probe", "status": "confirmed_observation", "repair_authorized": False,
                      "scope": "trajectory", "observation": {"variant_improved": improved}}] if valid else []}
