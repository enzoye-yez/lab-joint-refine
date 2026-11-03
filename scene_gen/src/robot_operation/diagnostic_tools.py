"""Composable read-only measurements; the coverage of any measurement never amounts to certifying the whole operation."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import math
import json
import re
import hashlib
import numpy as np
import mujoco

from .common import Settings, need, read, digest, identity, vector, number, OperationError
from .scene import load
from .evidence import contacts, descendants
from .planning import rotation_error

VERSION = 1
FIELDS = {
    "contacts": {"bodies"},
    "configuration": {"positions", "bodies"},
    "evidence_detail": {"evidence_id", "path", "offset", "limit"},
    "relative_pose": {"site_a", "site_b", "axis"},
    "joint_state": {"joints"},
    "geometry": {"geom"},
    "distance": {"geoms", "limit"},
    "condition": {"node", "condition"},
    "mechanism": {"interface", "samples"},
    "fit": {"interface", "body", "distance", "samples"},
    "trace": {"node", "bodies", "window"},
    "probe": {"node", "parameter", "value", "seconds"},
}
CATALOG = [
    {"kind": "contacts", "purpose": "Locate both contact parties, contact position, depth and contact force", "parameters": {"bodies": "optional list of actual body names"}},
    {"kind": "configuration", "purpose": "Check contacts of a given joint pose on a copy; does not claim the pose is dynamically reachable", "parameters": {"positions": "mapping from actual scalar joints to positions", "bodies": "optional list of bodies"}},
    {"kind": "evidence_detail", "purpose": "Read selected fields of a full measurement by evidence id, with paging; no re-simulation", "parameters": {"evidence_id": "id of an evidence summary already returned", "path": "list of field paths", "offset": "page offset, default 0", "limit": "1 to 64 items, default 32"}},
    {"kind": "relative_pose", "purpose": "Relative translation, rotation angle and axial/lateral offset along a given axis", "parameters": {"site_a": "actual site", "site_b": "reference site", "axis": "unit axis in the reference site's local frame, optional"}},
    {"kind": "joint_state", "purpose": "Joint positions, velocities, limits and related actuator control values", "parameters": {"joints": "list of actual joint names"}},
    {"kind": "geometry", "purpose": "Read the actual collision geometry and its associated specification claims; does not infer unobservable internal cavities", "parameters": {"geom": "actual geom name"}},
    {"kind": "distance", "purpose": "Reuse the consistency-checked distance measurement between two geoms; on failure, explicitly returns unknown", "parameters": {"geoms": "two actual geom names", "limit": "measurement range, default 0.05 m"}},
    {"kind": "condition", "purpose": "Re-check a fixed precondition or completion condition of the existing operation graph, keeping hold durations and the grasp reference", "parameters": {"node": "actual operation node", "condition": "precondition or completion"}},
    {"kind": "mechanism", "purpose": "Check a mechanism by sampling along its declared travel; cannot certify when locked or when dependencies are unknown", "parameters": {"interface": "actual interface", "samples": "2 to 56, default 9"}},
    {"kind": "fit", "purpose": "Straight-line fit geometry probe from a given start pose; does not certify actual executability", "parameters": {"interface": "actual interface", "body": "free part", "distance": "signed metres", "samples": "2 to 56, default 9"}},
    {"kind": "trace", "purpose": "Reuse the recorded trajectory to locate abnormal contacts and motion changes", "parameters": {"node": "optional operation node", "bodies": "optional list of bodies", "window": "number of steps kept before and after the anomaly"}},
    {"kind": "probe", "purpose": "Actual-executor comparison of the original action and a single-parameter variant from the same full state", "parameters": {"node": "existing operation node", "parameter": "parameter allowed to change", "value": "variant value", "seconds": "maximum duration of each probe"}},
]


def normalize(query, settings=None):
    settings = settings or Settings()
    need(isinstance(query, dict), "A diagnostic query must be an object")
    kind = query.get("kind")
    need(isinstance(kind,str) and kind in FIELDS, "Diagnostic capability not registered: " + str(kind), "capability_missing")
    need(set(query) <= FIELDS[kind] | {"kind", "state", "question", "hypotheses"}, "Diagnostic query has unknown fields")
    out = {k: deepcopy(v) for k, v in query.items() if k not in {"question", "hypotheses"}}
    out.setdefault("state", "node_start" if kind == "probe" else "initial")
    need(out["state"] in {"initial", "failure", "node_start"}, "Diagnostic state must be the initial state, a node start state or the real failure snapshot")
    if kind in {"trace","geometry","evidence_detail"}:out["state"]="initial"  # Full recordings and model-local geometry are not recomputed because of the state label.
    required = {"relative_pose": ["site_a", "site_b"], "joint_state": ["joints"], "geometry": ["geom"],
                "configuration": ["positions"],
                "evidence_detail": ["evidence_id"],
                "mechanism": ["interface"], "fit": ["interface", "body", "distance"],
                "distance": ["geoms"], "condition": ["node", "condition"],
                "probe": ["node", "parameter", "value"]}
    need(all(k in out for k in required.get(kind, [])), "Diagnostic query is missing parameters")
    if kind=="evidence_detail":
        need(isinstance(out["evidence_id"],str) and re.fullmatch(r"[0-9a-f]{64}",out["evidence_id"]),"Invalid evidence id")
        out.setdefault("path",["observations"]);out.setdefault("offset",0);out.setdefault("limit",32)
        need(isinstance(out["path"],list) and len(out["path"])<=12 and all(type(p) in (str,int) for p in out["path"]),"Invalid evidence field path")
        need(type(out["offset"]) is int and 0<=out["offset"]<=1000000 and type(out["limit"]) is int and 1<=out["limit"]<=64,"Invalid evidence page range")
    for key in ("bodies", "joints"):
        if key in out:
            need(isinstance(out[key], list) and len(out[key]) <= 64 and
                 all(isinstance(x, str) and x for x in out[key]), "Invalid diagnostic object list")
            out[key] = sorted(set(out[key]))
    if kind == "joint_state": need(out["joints"], "Joint list must not be empty")
    if kind in {"contacts","configuration"}: out.setdefault("bodies", [])
    if kind == "configuration":
        need(isinstance(out["positions"],dict) and 0<len(out["positions"])<=32 and
             all(isinstance(k,str) and type(v) in (int,float) for k,v in out["positions"].items()),"A pose query must be a bounded mapping of scalar joints")
    if kind == "distance":
        need(isinstance(out["geoms"],list) and len(out["geoms"])==2 and all(isinstance(x,str) for x in out["geoms"])
             and out["geoms"][0]!=out["geoms"][1], "A distance check must name two different geoms")
        out["geoms"]=sorted(out["geoms"])
        out["limit"]=float(number(out.get("limit",.05),1e-6,1))
    if kind == "condition": need(out["condition"] in {"precondition","completion"}, "Only frozen node conditions can be read")
    if kind in {"fit", "mechanism"}:
        out.setdefault("samples", 9)
        need(type(out["samples"]) is int and 2 <= out["samples"] <= 56, "Sample count must be between 2 and 56")
    if kind == "fit": out["distance"] = float(number(out["distance"], -1, 1))
    if "axis" in out:
        axis = vector(out["axis"], 3)
        need(abs(np.linalg.norm(axis)-1) < 1e-6, "Fit axis must be a unit vector")
        out["axis"] = axis.tolist()
    if kind == "trace":
        out.setdefault("bodies", [])
        out.setdefault("window", settings.diagnostic_trace_window)
        need(type(out["window"]) is int and 1 <= out["window"] <= settings.diagnostic_trace_window,
             "Trajectory evidence window too large")
    if kind == "probe":
        out.setdefault("seconds", settings.diagnostic_probe_seconds)
        out["seconds"] = float(number(out["seconds"], .002, settings.diagnostic_probe_seconds))
        need(out["parameter"] in {"targets", "positions", "controls", "axis", "distance", "angle", "goal"},
             "A probe must not modify the task, object identities, contact permissions, thresholds or actuator specifications")
    # Unify int and float semantics so that 1 and 1.0 cannot bypass deduplication.
    def canonical(value):
        if isinstance(value, dict): return {k: canonical(v) for k, v in sorted(value.items())}
        if isinstance(value, list): return [canonical(v) for v in value]
        if type(value) in (float, int):
            need(math.isfinite(value), "Diagnostic parameters must not contain non-finite values")
            return int(value) if float(value).is_integer() else value
        return value
    return canonical(out)


class DiagnosticContext:
    def __init__(self, scene, scene_hash, settings=None, *, run_root=None, evidence_root=None, guard=lambda: None, cancel_guard=None):
        self.scene, self.scene_hash = str(scene), scene_hash
        self.settings, self.guard = settings or Settings(), guard
        self.cancel_guard=cancel_guard or guard
        self.model, self.initial, self.meta, self.manifest = load(scene, scene_hash)
        self.run_root = Path(run_root) if run_root else None
        self.evidence_root = Path(evidence_root) if evidence_root else None
        self.record = self.files = self.failure = self.node_start = self.initial_snapshot = None
        if self.run_root:
            from asset_prep.storage import file_hash
            self.files = read(self.run_root/"files.json")
            for name in ("recording.json", "initial.json", "failure_state.json", "node_start_state.json", "task.json", "graph.json", "result.json", "events.jsonl", "steps.jsonl.gz"):
                path = self.run_root/name
                if name in self.files or path.exists():
                    need(path.is_file() and self.files.get(name) == file_hash(path), "Diagnostic source missing or modified: "+name, "stale_evidence")
            self.record = read(self.run_root/"recording.json")
            need(self.record["scene_sha256"] == scene_hash, "Diagnostic run belongs to an older scene", "stale_evidence")
            if (self.run_root/"initial.json").exists(): self.initial_snapshot = read(self.run_root/"initial.json")
            if (self.run_root/"failure_state.json").exists(): self.failure = read(self.run_root/"failure_state.json")
            if (self.run_root/"node_start_state.json").exists(): self.node_start = read(self.run_root/"node_start_state.json")

    def verify_sources(self):
        """Cached deliveries check sources too; a cache hit cannot skip version verification."""
        from asset_prep.storage import file_hash
        self.guard()
        need(identity(self.scene)==self.scene_hash,"Diagnostic scene version changed","stale_scene")
        if self.run_root:
            need(digest(read(self.run_root/"files.json"))==digest(self.files),"Original run receipt changed","stale_evidence")
            for name in ("recording.json","initial.json","failure_state.json","node_start_state.json","task.json","graph.json","result.json","events.jsonl","steps.jsonl.gz"):
                path=self.run_root/name
                if name in self.files or path.exists():
                    need(path.is_file() and self.files.get(name)==file_hash(path),"Original run evidence missing or changed: "+name,"stale_evidence")

    def binding(self, query):
        state = self.failure if query["state"] == "failure" else self.node_start if query["state"] == "node_start" else self.initial_snapshot
        binding = {"scene_sha256": self.scene_hash, "tool_version": VERSION,
                   "mujoco_version":mujoco.__version__,
                   "settings_sha256": digest(asdict(self.settings)),
                   "state_sha256": digest(state) if state else query["state"]}
        if query["kind"]=="geometry":
            binding["state_sha256"]="model_local_geometry"
            try:
                value=self.geometry_values(query["geom"])
                g=self.model.geom(query["geom"]);mesh=int(g.dataid[0])
                if g.type[0]==mujoco.mjtGeom.mjGEOM_MESH:
                    m=self.model
                    for field,adr,num in (("vertices",m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]),("faces",m.mesh_faceadr[mesh],m.mesh_facenum[mesh])):
                        array=m.mesh_vert if field=="vertices" else m.mesh_face
                        value[field]=hashlib.sha256(array[adr:adr+num].tobytes()).hexdigest()
                binding["local_geometry_sha256"]=digest(value)
            except (KeyError,TypeError):pass  # For invalid objects the measurement tool returns cacheable invalid evidence.
        if query["kind"]=="trace":binding["state_sha256"]="recorded_sequence"
        if query["kind"] in {"trace", "probe", "condition"}: binding["run_sha256"] = digest(self.files)
        return binding

    def geometry_values(self,name):
        m=self.model;g=m.geom(name)
        return {"geom":name,"body":m.body(g.bodyid[0]).name,"type":int(g.type[0]),
                "size_parameters":g.size.tolist(),"local_position":g.pos.tolist(),"local_quaternion":g.quat.tolist(),
                "friction":g.friction.tolist(),"interface_declarations":[i for i in self.manifest["interfaces"] if i["body"]==m.body(g.bodyid[0]).name]}

    def data(self, query):
        data = mujoco.MjData(self.model)
        from .common import copy_data
        copy_data(data, self.model, self.initial)
        if query["state"] != "initial" or self.initial_snapshot is not None:
            snapshot = self.failure if query["state"] == "failure" else self.node_start if query["state"] == "node_start" else self.initial_snapshot
            need(snapshot is not None, "No matching real state record", "missing_evidence")
            physics = np.asarray(snapshot["physics"], float)
            need(physics.shape == (mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION),)
                 and np.isfinite(physics).all(), "Invalid integration state in the failure snapshot")
            mujoco.mj_setState(self.model, data, physics, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(self.model, data)
        return data

    def instances(self, bodies):
        ids = [i["id"] for i in self.meta["plan"]["instances"]]
        return sorted({i for name in bodies for i in ids if name.startswith(i+"__")})

    def roles(self):
        jobs = {j["id"]: j.get("role", "asset") for j in self.meta["plan"].get("jobs", [])}
        tools = {v.get("instance") for v in self.meta["plan"].get("end_effectors", {}).values()}
        return {i["id"]: ("fixture" if jobs.get(i["job"]) in {"fixture", "tool"} or i["id"] in tools else "asset")
                for i in self.meta["plan"]["instances"]}

    def measure(self, query):
        self.guard()
        need(identity(self.scene) == self.scene_hash, "Scene version changed before measurement", "stale_scene")
        result = self._measure(query)
        need(identity(self.scene) == self.scene_hash, "Scene version changed during measurement", "stale_scene")
        if self.run_root:
            from asset_prep.storage import file_hash
            need(digest(read(self.run_root/"files.json"))==digest(self.files),"Original run receipt changed during measurement","stale_evidence")
            names=["recording.json","initial.json","failure_state.json","node_start_state.json","task.json","graph.json","result.json"]
            if query["kind"]=="trace":names.append("steps.jsonl.gz")
            for name in names:
                if name in self.files:need(file_hash(self.run_root/name)==self.files[name],"Original run evidence changed during measurement: "+name,"stale_evidence")
        self.guard()
        return result

    def _measure(self, q):
        kind = q["kind"]
        if kind=="evidence_detail":
            from asset_prep.storage import file_hash
            record=getattr(self,"evidence_records",{}).get(q["evidence_id"])
            need(record is not None and record.get("result") is not None,"Evidence does not exist or is not yet complete","missing_evidence")
            reference=record.get("result_reference")
            if reference:
                need(self.evidence_root is not None,"No evidence archive root is bound","missing_evidence")
                path=Path(reference["path"])
                need(path.resolve().parent==(self.evidence_root/"evidence").resolve() and re.fullmatch(r"[0-9a-f]{64}",path.stem)
                     and path.is_file() and file_hash(path)==reference["sha256"],"Evidence archive missing or modified","stale_evidence")
                value=read(path)
            else:value=deepcopy(record["result"])
            for field in q["path"]:
                need(isinstance(value,dict) and field in value or isinstance(value,list) and type(field) is int and 0<=field<len(value),"Evidence field does not exist")
                value=value[field]
            offset,limit=q["offset"],q["limit"]
            if isinstance(value,list):total=len(value);page=value[offset:offset+limit]
            elif isinstance(value,dict):total=len(value);page=dict(list(value.items())[offset:offset+limit])
            elif isinstance(value,str):total=len(value);page=value[offset:offset+limit]
            else:total=1;page=value if offset==0 else None
            need(len(json.dumps(page,ensure_ascii=False))<=65536,"Evidence page too large; drill into a field or reduce the page size","query_too_broad")
            return {"status":"valid","observations":{"data":page,"total":total,"next_offset":offset+limit if offset+limit<total else None},
                    "coverage":{"evidence_id":q["evidence_id"],"path":q["path"],"offset":offset,"limit":limit},
                    "limitations":["Reads details of an existing measurement; produces no new physical measurement or repair authorization"],"rows":[]}
        if kind == "trace":
            from .diagnostic_trace import inspect_trace
            return inspect_trace(self, q)
        if kind == "probe":
            from .diagnostic_probe import paired_probe
            return paired_probe(self, q)
        m, d = self.model, self.data(q)
        result = {"status": "valid", "observations": {}, "coverage": {"kind": kind, "state": q["state"]},
                  "limitations": [], "rows": []}
        if kind in {"contacts","configuration"}:
            if kind == "configuration":
                for name,value in q["positions"].items():
                    j=m.joint(name)
                    need(j.type[0] in (2,3) and (not j.limited[0] or j.range[0]<=value<=j.range[1]),"A diagnostic pose must not change the free root or exceed joint limits")
                    need(not any(d.eq_active[e] and m.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and
                                 j.id in {int(m.eq_obj1id[e]),int(m.eq_obj2id[e])} for e in range(m.neq)),
                         "Diagnosis cannot move a coupling-constrained joint on its own","missing_evidence")
                    d.qpos[j.qposadr[0]]=value
                mujoco.mj_forward(m,d)
                result["coverage"]["positions"]=q["positions"]
                result["limitations"]=["Geometric contact observation for a given joint pose; does not prove robot reachability, valid mechanism dependencies or real execution success"]
            selected = set()
            for name in q["bodies"]: selected |= descendants(m, m.body(name).id)
            rows = []
            for c in contacts(m, d):
                bids = [int(m.geom_bodyid[g]) for g in c["geoms"]]
                if selected and not (selected & set(bids)): continue
                bodies = [m.body(b).name for b in bids]
                rows.append({**c, "geoms": [m.geom(g).name or "#"+str(g) for g in c["geoms"]],
                             "bodies": bodies, "instances": self.instances(bodies)})
            result["observations"] = {"contacts": rows}
            result["coverage"]["bodies"] = q["bodies"]
            result["rows"] = [{"kind": "collision" if r["distance"] < -self.settings.penetration_m else "contact_observation",
                                "status": "confirmed_observation", "geoms": r["geoms"],
                                "instances": r["instances"], "repair_authorized":
                                kind == "contacts" and q["state"] == "initial" and r["distance"] < -self.settings.penetration_m and
                                not any(b.startswith("robot__") for b in r["bodies"]),
                                "observation": {**r, "force_is_execution_evidence":kind=="contacts",
                                                "limit_exceeded": r["distance"] < -self.settings.penetration_m or
                                                (kind=="contacts" and r["force_n"] > self.settings.contact_force_n)}} for r in rows]
        elif kind == "relative_pose":
            a, b = m.site(q["site_a"]).id, m.site(q["site_b"]).id
            rotation = d.site_xmat[b].reshape(3, 3)
            delta = rotation.T @ (d.site_xpos[a]-d.site_xpos[b])
            values = {"translation_m": delta.tolist(), "angle_rad": float(np.linalg.norm(rotation_error(rotation, d.site_xmat[a].reshape(3, 3))))}
            if "axis" in q:
                axial = float(delta @ np.asarray(q["axis"]))
                values.update(axial_m=axial, lateral_m=float(np.linalg.norm(delta-axial*np.asarray(q["axis"]))))
            result["observations"] = values
            result["coverage"].update(site_a=q["site_a"], site_b=q["site_b"], axis=q.get("axis"))
        elif kind == "joint_state":
            rows = []
            for name in q["joints"]:
                j = m.joint(name)
                need(j.type[0] in (2, 3), "A free joint cannot be checked as a scalar mechanism")
                aids = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j.id]
                rows.append({"joint": name, "position": float(d.qpos[j.qposadr[0]]), "velocity": float(d.qvel[j.dofadr[0]]),
                             "limited": bool(j.limited[0]), "range": j.range.tolist(),
                             "actuators": [{"name": m.actuator(a).name, "control": float(d.ctrl[a]),
                                             "force": float(d.actuator_force[a]), "control_limited": bool(m.actuator_ctrllimited[a]),
                                             "control_range": m.actuator_ctrlrange[a].tolist()} for a in aids]})
            result["observations"] = {"joints": rows}
        elif kind == "distance":
            from asset_generation.distance import observe
            from scene_generation.recovery_geometry import RecoveryQuery
            value=observe(RecoveryQuery(m,d),*[m.geom(n).id for n in q["geoms"]],q["limit"])
            result["observations"]=value
            result["status"]="unknown" if value["outcome"]=="unknown" else "valid"
            result["coverage"].update(geoms=q["geoms"],limit_m=q["limit"])
            result["limitations"]=["A separation lower bound is not an exact distance; a single geom pair does not represent the whole opening or continuous motion"]
        elif kind == "condition":
            need(self.run_root is not None,"Missing frozen operation graph","missing_evidence")
            from .runtime import Session
            graph,task=read(self.run_root/"graph.json"),read(self.run_root/"task.json")
            node=next((n for n in graph["nodes"] if n["id"]==q["node"]),None)
            need(node is not None,"Condition-check node does not exist")
            snapshot=self.failure if q["state"]=="failure" else self.node_start if q["state"]=="node_start" else read(self.run_root/"initial.json")
            need(snapshot is not None,"Missing full state for the condition check","missing_evidence")
            session=Session(m,d,self.meta,self.manifest,task,self.settings);session.restore(deepcopy(snapshot))
            passed=session.evidence.evaluate(node[q["condition"]])
            result["observations"]={"passed":passed,"evidence":session.evidence.last[digest(node[q["condition"]])]}
            result["coverage"].update(node=q["node"],condition=q["condition"])
            result["limitations"]=["Re-checks existing conditions only; a failure by itself cannot authorize modifying the real instrument"]
        elif kind == "geometry":
            result["observations"] = self.geometry_values(q["geom"])
            result["limitations"] = ["Raw geometry parameters are not internal clearance; an interface's source claim is not an independent specification certification"]
        else:
            from .diagnosis import inspect_scene
            query = {k: v for k, v in q.items() if k != "state"}
            snapshot = self.failure if q["state"] == "failure" else self.node_start if q["state"] == "node_start" else self.initial_snapshot
            report = inspect_scene(self.scene, self.scene_hash, self.settings, snapshot, [query])
            row = deepcopy(report["rows"][-1])
            # The new diagnostic never treats self-contact from an arbitrary mechanism sweep directly as a defect: coupling must remain unknown.
            if kind == "mechanism":
                interface = next(i for i in self.manifest["interfaces"] if i["id"] == q["interface"])
                op = interface.get("operation", {})
                source_range = op.get("basis", {}).get("range", {}).get("kind") == "source_specification"
                instance=next((i for i in self.roles() if interface["body"].startswith(i+"__")),None)
                coupled_unknown=instance is not None and sum(m.joint(j).name.startswith(instance+"__") and m.jnt_type[j] in (2,3)
                                                              for j in range(m.njnt))>1
                if not source_range or coupled_unknown:
                    row["repair_authorized"] = False
                    if row.get("status") == "confirmed_defect": row["status"] = "confirmed_observation"
                    row.setdefault("coverage_gaps", []).append("Multi-joint dependency not yet certified" if coupled_unknown else "Required travel lacks a specification source; scan results cannot authorize changing the real mechanism")
            result["rows"] = [row]
            result["status"] = "unknown" if row["status"] == "unknown" else "valid"
            result["coverage"].update(interface=q["interface"], samples=q["samples"])
            result["limitations"] = row.get("coverage_gaps", [])
        return result
