"""同一完整起态的单参数对照；通过原执行器运行，不修改真实执行现场。"""
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
         "局部对照需要完整运行记录和独立证据输出目录", "missing_evidence")
    need(context.record.get("physics_started", True), "未通过动力学准入，不能局部试探")
    task, graph = read(context.run_root/"task.json"), read(context.run_root/"graph.json")
    original = next((n for n in graph["nodes"] if n["id"] == query["node"]), None)
    need(original is not None and original["operation"] not in {"device", "wait", "observe", "check"},
         "对照必须针对已有机械操作")
    parameter = query["parameter"]
    need(parameter in original["parameters"], "对照不能添加未使用的参数")
    need(original["parameters"][parameter] != query["value"], "对照参数未发生变化")
    if parameter == "targets":
        before, after = original["parameters"][parameter], query["value"]
        need(isinstance(after, list) and len(before) == len(after) and
             all({k: v for k, v in a.items() if k not in {"position", "rotation", "offset"}} ==
                 {k: v for k, v in b.items() if k not in {"position", "rotation", "offset"}} for a, b in zip(before, after)),
             "目标对照不得改变手臂、接口、目标对象或参考坐标系")
    if parameter in {"positions", "controls"}:
        need(isinstance(query["value"], dict) and set(query["value"]) == set(original["parameters"][parameter]),
             "对照不得改变受控关节或执行器集合")
    snapshot = (context.failure if query["state"] == "failure" else context.node_start if query["state"] == "node_start"
                else read(context.run_root/"initial.json"))
    need(snapshot is not None, "缺少完整对照起态")
    if query["state"] != "initial":
        need(snapshot.get("node") == original["id"], "失败现场不属于待对照节点")
    root = Path(context.evidence_root)/digest({"query": query, "binding": context.binding(query)})
    need(not root.exists(), "已有局部试探目录，不能隐式重试中断试探", "interrupted_probe")
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
            need(not session.unbound_events, "机械事件缺少检测器", "capability_missing")
            session.begin(node)
            session.guard()
            need(session.evidence.evaluate(node["precondition"]), "局部试探起态不满足节点前提", "probe_precondition")
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
            "limitations": ["仅说明这一局部参数变化的效果，不证明唯一根因，不代表完整任务成功"],
            "rows": [{"kind": "trajectory_probe", "status": "confirmed_observation", "repair_authorized": False,
                      "scope": "trajectory", "observation": {"variant_improved": improved}}] if valid else []}
