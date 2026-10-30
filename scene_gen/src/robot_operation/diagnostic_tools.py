"""可组合的只读量测；任何量测的覆盖范围都不等同于整项操作认证。"""
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
    {"kind": "contacts", "purpose": "定位接触双方、位置、深度与接触力", "parameters": {"bodies": "可选实际身体名列表"}},
    {"kind": "configuration", "purpose": "副本中检查给定关节姿态的接触，不声称该姿态动力学可达", "parameters": {"positions": "实际标量关节到位置的映射", "bodies": "可选身体列表"}},
    {"kind": "evidence_detail", "purpose": "按证据编号读取完整量测的指定字段和分页，不重新仿真", "parameters": {"evidence_id": "已返回的证据摘要编号", "path": "字段路径列表", "offset": "分页起点，默认0", "limit": "1至64项，默认32"}},
    {"kind": "relative_pose", "purpose": "相对平移、转角及沿给定轴的轴向/横向偏差", "parameters": {"site_a": "实际坐标点", "site_b": "参考坐标点", "axis": "参考坐标点局部单位轴，可选"}},
    {"kind": "joint_state", "purpose": "关节位置、速度、限位和相关执行器控制值", "parameters": {"joints": "实际关节名列表"}},
    {"kind": "geometry", "purpose": "读取实际碰撞几何及关联规格声明，不推断不可观测内腔", "parameters": {"geom": "实际几何名"}},
    {"kind": "distance", "purpose": "复用带一致性校验的两几何距离量测，失败明确返回未知", "parameters": {"geoms": "两个实际几何名", "limit": "量测范围，默认0.05米"}},
    {"kind": "condition", "purpose": "复查既有操作图的固定前置或完成条件，保留保持时长与夹持参考", "parameters": {"node": "实际操作节点", "condition": "precondition或completion"}},
    {"kind": "mechanism", "purpose": "沿声明行程采样检查机构；锁止或依赖未知时不能认证", "parameters": {"interface": "实际接口", "samples": "2至56，默认9"}},
    {"kind": "fit", "purpose": "指定起姿的直线配合几何试探，不认证实际可执行性", "parameters": {"interface": "实际接口", "body": "自由部件", "distance": "有符号米", "samples": "2至56，默认9"}},
    {"kind": "trace", "purpose": "复用录制轨迹定位异常接触及运动变化", "parameters": {"node": "可选操作节点", "bodies": "可选身体列表", "window": "异常前后保留步数"}},
    {"kind": "probe", "purpose": "同一完整状态下原动作与单参数变体的实际执行器对照", "parameters": {"node": "已有操作节点", "parameter": "允许调整的参数", "value": "变体值", "seconds": "每次试探时长上限"}},
]


def normalize(query, settings=None):
    settings = settings or Settings()
    need(isinstance(query, dict), "诊断查询必须为对象")
    kind = query.get("kind")
    need(isinstance(kind,str) and kind in FIELDS, "诊断能力未注册：" + str(kind), "capability_missing")
    need(set(query) <= FIELDS[kind] | {"kind", "state", "question", "hypotheses"}, "诊断查询存在未知字段")
    out = {k: deepcopy(v) for k, v in query.items() if k not in {"question", "hypotheses"}}
    out.setdefault("state", "node_start" if kind == "probe" else "initial")
    need(out["state"] in {"initial", "failure", "node_start"}, "诊断状态只能为初态、节点起态或真实失败现场")
    if kind in {"trace","geometry","evidence_detail"}:out["state"]="initial"  # 整段录制和模型局部几何不因状态标签重算。
    required = {"relative_pose": ["site_a", "site_b"], "joint_state": ["joints"], "geometry": ["geom"],
                "configuration": ["positions"],
                "evidence_detail": ["evidence_id"],
                "mechanism": ["interface"], "fit": ["interface", "body", "distance"],
                "distance": ["geoms"], "condition": ["node", "condition"],
                "probe": ["node", "parameter", "value"]}
    need(all(k in out for k in required.get(kind, [])), "诊断查询缺少参数")
    if kind=="evidence_detail":
        need(isinstance(out["evidence_id"],str) and re.fullmatch(r"[0-9a-f]{64}",out["evidence_id"]),"证据编号无效")
        out.setdefault("path",["observations"]);out.setdefault("offset",0);out.setdefault("limit",32)
        need(isinstance(out["path"],list) and len(out["path"])<=12 and all(type(p) in (str,int) for p in out["path"]),"证据字段路径无效")
        need(type(out["offset"]) is int and 0<=out["offset"]<=1000000 and type(out["limit"]) is int and 1<=out["limit"]<=64,"证据分页范围无效")
    for key in ("bodies", "joints"):
        if key in out:
            need(isinstance(out[key], list) and len(out[key]) <= 64 and
                 all(isinstance(x, str) and x for x in out[key]), "诊断对象列表无效")
            out[key] = sorted(set(out[key]))
    if kind == "joint_state": need(out["joints"], "关节列表不能为空")
    if kind in {"contacts","configuration"}: out.setdefault("bodies", [])
    if kind == "configuration":
        need(isinstance(out["positions"],dict) and 0<len(out["positions"])<=32 and
             all(isinstance(k,str) and type(v) in (int,float) for k,v in out["positions"].items()),"姿态查询须为有界标量关节映射")
    if kind == "distance":
        need(isinstance(out["geoms"],list) and len(out["geoms"])==2 and all(isinstance(x,str) for x in out["geoms"])
             and out["geoms"][0]!=out["geoms"][1], "距离检查须指定两个不同几何")
        out["geoms"]=sorted(out["geoms"])
        out["limit"]=float(number(out.get("limit",.05),1e-6,1))
    if kind == "condition": need(out["condition"] in {"precondition","completion"}, "只能读取冻结的节点条件")
    if kind in {"fit", "mechanism"}:
        out.setdefault("samples", 9)
        need(type(out["samples"]) is int and 2 <= out["samples"] <= 56, "采样数须在2至56内")
    if kind == "fit": out["distance"] = float(number(out["distance"], -1, 1))
    if "axis" in out:
        axis = vector(out["axis"], 3)
        need(abs(np.linalg.norm(axis)-1) < 1e-6, "配合轴须为单位向量")
        out["axis"] = axis.tolist()
    if kind == "trace":
        out.setdefault("bodies", [])
        out.setdefault("window", settings.diagnostic_trace_window)
        need(type(out["window"]) is int and 1 <= out["window"] <= settings.diagnostic_trace_window,
             "轨迹证据窗口超限")
    if kind == "probe":
        out.setdefault("seconds", settings.diagnostic_probe_seconds)
        out["seconds"] = float(number(out["seconds"], .002, settings.diagnostic_probe_seconds))
        need(out["parameter"] in {"targets", "positions", "controls", "axis", "distance", "angle", "goal"},
             "试探不能修改任务、对象身份、接触许可、阈值或执行器规格")
    # 统一整数与浮点的语义，避免 1 和 1.0 绕过去重。
    def canonical(value):
        if isinstance(value, dict): return {k: canonical(v) for k, v in sorted(value.items())}
        if isinstance(value, list): return [canonical(v) for v in value]
        if type(value) in (float, int):
            need(math.isfinite(value), "诊断参数禁止非有限数值")
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
                    need(path.is_file() and self.files.get(name) == file_hash(path), "诊断来源缺失或被修改："+name, "stale_evidence")
            self.record = read(self.run_root/"recording.json")
            need(self.record["scene_sha256"] == scene_hash, "诊断运行属于旧场景", "stale_evidence")
            if (self.run_root/"initial.json").exists(): self.initial_snapshot = read(self.run_root/"initial.json")
            if (self.run_root/"failure_state.json").exists(): self.failure = read(self.run_root/"failure_state.json")
            if (self.run_root/"node_start_state.json").exists(): self.node_start = read(self.run_root/"node_start_state.json")

    def verify_sources(self):
        """缓存交付同样检查来源，不能借命中缓存跳过版本验证。"""
        from asset_prep.storage import file_hash
        self.guard()
        need(identity(self.scene)==self.scene_hash,"诊断场景版本变化","stale_scene")
        if self.run_root:
            need(digest(read(self.run_root/"files.json"))==digest(self.files),"原运行回执变化","stale_evidence")
            for name in ("recording.json","initial.json","failure_state.json","node_start_state.json","task.json","graph.json","result.json","events.jsonl","steps.jsonl.gz"):
                path=self.run_root/name
                if name in self.files or path.exists():
                    need(path.is_file() and self.files.get(name)==file_hash(path),"原运行证据缺失或变化："+name,"stale_evidence")

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
            except (KeyError,TypeError):pass  # 无效对象由量测工具返回可缓存的无效证据。
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
            need(snapshot is not None, "没有相符的真实状态记录", "missing_evidence")
            physics = np.asarray(snapshot["physics"], float)
            need(physics.shape == (mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION),)
                 and np.isfinite(physics).all(), "失败现场积分状态无效")
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
        need(identity(self.scene) == self.scene_hash, "量测前场景版本变化", "stale_scene")
        result = self._measure(query)
        need(identity(self.scene) == self.scene_hash, "量测期间场景版本变化", "stale_scene")
        if self.run_root:
            from asset_prep.storage import file_hash
            need(digest(read(self.run_root/"files.json"))==digest(self.files),"量测期间原运行回执变化","stale_evidence")
            names=["recording.json","initial.json","failure_state.json","node_start_state.json","task.json","graph.json","result.json"]
            if query["kind"]=="trace":names.append("steps.jsonl.gz")
            for name in names:
                if name in self.files:need(file_hash(self.run_root/name)==self.files[name],"量测期间原运行证据变化："+name,"stale_evidence")
        self.guard()
        return result

    def _measure(self, q):
        kind = q["kind"]
        if kind=="evidence_detail":
            from asset_prep.storage import file_hash
            record=getattr(self,"evidence_records",{}).get(q["evidence_id"])
            need(record is not None and record.get("result") is not None,"证据不存在或尚未完成","missing_evidence")
            reference=record.get("result_reference")
            if reference:
                need(self.evidence_root is not None,"未绑定证据归档根目录","missing_evidence")
                path=Path(reference["path"])
                need(path.resolve().parent==(self.evidence_root/"evidence").resolve() and re.fullmatch(r"[0-9a-f]{64}",path.stem)
                     and path.is_file() and file_hash(path)==reference["sha256"],"证据归档缺失或被修改","stale_evidence")
                value=read(path)
            else:value=deepcopy(record["result"])
            for field in q["path"]:
                need(isinstance(value,dict) and field in value or isinstance(value,list) and type(field) is int and 0<=field<len(value),"证据字段不存在")
                value=value[field]
            offset,limit=q["offset"],q["limit"]
            if isinstance(value,list):total=len(value);page=value[offset:offset+limit]
            elif isinstance(value,dict):total=len(value);page=dict(list(value.items())[offset:offset+limit])
            elif isinstance(value,str):total=len(value);page=value[offset:offset+limit]
            else:total=1;page=value if offset==0 else None
            need(len(json.dumps(page,ensure_ascii=False))<=65536,"证据页过大，请下钻字段或减小分页","query_too_broad")
            return {"status":"valid","observations":{"data":page,"total":total,"next_offset":offset+limit if offset+limit<total else None},
                    "coverage":{"evidence_id":q["evidence_id"],"path":q["path"],"offset":offset,"limit":limit},
                    "limitations":["读取已有量测详情，不产生新的物理量测或返修授权"],"rows":[]}
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
                    need(j.type[0] in (2,3) and (not j.limited[0] or j.range[0]<=value<=j.range[1]),"诊断姿态不能改变自由根或越过关节限位")
                    need(not any(d.eq_active[e] and m.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and
                                 j.id in {int(m.eq_obj1id[e]),int(m.eq_obj2id[e])} for e in range(m.neq)),
                         "诊断不能单独移动受联动约束的关节","missing_evidence")
                    d.qpos[j.qposadr[0]]=value
                mujoco.mj_forward(m,d)
                result["coverage"]["positions"]=q["positions"]
                result["limitations"]=["给定关节姿态的几何接触观测，不证明机器人可达、机构依赖合法或真实执行成功"]
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
                need(j.type[0] in (2, 3), "自由关节不能当成标量机构检查")
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
            result["limitations"]=["分离下界不是精确距离；单个几何对不代表整个入口或连续运动"]
        elif kind == "condition":
            need(self.run_root is not None,"缺少已冻结操作图","missing_evidence")
            from .runtime import Session
            graph,task=read(self.run_root/"graph.json"),read(self.run_root/"task.json")
            node=next((n for n in graph["nodes"] if n["id"]==q["node"]),None)
            need(node is not None,"条件检查节点不存在")
            snapshot=self.failure if q["state"]=="failure" else self.node_start if q["state"]=="node_start" else read(self.run_root/"initial.json")
            need(snapshot is not None,"缺少条件检查完整状态","missing_evidence")
            session=Session(m,d,self.meta,self.manifest,task,self.settings);session.restore(deepcopy(snapshot))
            passed=session.evidence.evaluate(node[q["condition"]])
            result["observations"]={"passed":passed,"evidence":session.evidence.last[digest(node[q["condition"]])]}
            result["coverage"].update(node=q["node"],condition=q["condition"])
            result["limitations"]=["只复查既有条件；失败本身不能授权修改真实仪器"]
        elif kind == "geometry":
            result["observations"] = self.geometry_values(q["geom"])
            result["limitations"] = ["原始几何参数不是内腔间隙；接口来源声明不是独立规格认证"]
        else:
            from .diagnosis import inspect_scene
            query = {k: v for k, v in q.items() if k != "state"}
            snapshot = self.failure if q["state"] == "failure" else self.node_start if q["state"] == "node_start" else self.initial_snapshot
            report = inspect_scene(self.scene, self.scene_hash, self.settings, snapshot, [query])
            row = deepcopy(report["rows"][-1])
            # 新诊断器不把任意机构扫描的自身接触直接认作缺陷：必须保留耦合未知。
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
                    row.setdefault("coverage_gaps", []).append("多关节依赖尚未认证" if coupled_unknown else "必要行程缺少规格依据，不能由扫描结果授权改变真实机构")
            result["rows"] = [row]
            result["status"] = "unknown" if row["status"] == "unknown" else "valid"
            result["coverage"].update(interface=q["interface"], samples=q["samples"])
            result["limitations"] = row.get("coverage_gaps", [])
        return result
