"""独立加载实际MJCF并检查；不执行生成脚本或模型自写测试。"""

import math
from pathlib import Path
from xml.etree.ElementTree import ParseError
from defusedxml.common import DefusedXmlException

from defusedxml import ElementTree as ET

from asset_prep.errors import ValidationError
from asset_prep.storage import safe_path
from .contracts import finite, read_json, require


def _load_model(root, *, scope="asset", model_file="model.xml", enforce_geom_limit=True, enforce_solver_limit=True):
    import mujoco
    require(scope in {"asset", "scene"}, "模型加载范围无效")
    max_geoms = 50000 if scope == "scene" else 10000
    max_xml_bytes = 32000000 if scope == "scene" else 8000000
    root = Path(root)
    xml = safe_path(root, model_file)
    require(xml.is_file() and xml.stat().st_size <= max_xml_bytes, "缺少或过大的 model.xml")
    tree = ET.parse(xml)
    require(tree.getroot().tag == "mujoco", "根节点必须为 mujoco")
    for node in tree.iter():
        require(node.tag not in {"include", "plugin", "extension", "attach"}, "第一版不支持外部包含或插件")
        if node.tag == "compiler":
            require(not any(k in node.attrib for k in ("assetdir", "meshdir", "texturedir")), "资源直接使用产物内相对路径")
            require(node.get("angle", "degree") == "radian", "必须明确使用弧度")
        if node.tag == "flag":
            require(all(node.get(key) != "disable" for key in ("contact", "constraint", "equality", "limit")),
                    "不能关闭接触、约束或关节限位以绕过检查")
        for key,value in node.attrib.items():
            if key.startswith("file"):
                target = safe_path(root, value)
                require(target.is_file(), "引用的资源不存在")
    require(tree.find("compiler") is not None, "必须明确提供 compiler angle=radian")
    model = mujoco.MjModel.from_xml_path(str(xml))
    require(0 < model.ngeom and (not enforce_geom_limit or model.ngeom <= max_geoms),
            f"模型几何规模超限：scope={scope}, ngeom={model.ngeom}/{max_geoms}, "
            f"nbody={model.nbody}/4096, nv={model.nv}/2048")
    if enforce_solver_limit:
        require(model.nbody <= 4096 and model.nv <= 2048,
                f"资产准入或动力学检查资源超限：nbody={model.nbody}/4096, nv={model.nv}/2048；静态场景入口不使用全局动力学求解器")
    require(0.000001 <= model.opt.timestep <= 0.01, "仿真步长超出工程检查范围")
    return model


def load_model(root, *, scope="asset", model_file="model.xml"):
    """Strict compatibility loader; asset admission retains its geometry limit."""
    return _load_model(root, scope=scope, model_file=model_file)


def load_scene_model(root, *, model_file="model.xml", dynamics=False):
    """Scene publication, inspection and preview share one validated loading policy."""
    return _load_model(root, scope="scene", model_file=model_file, enforce_geom_limit=False,
                       enforce_solver_limit=dynamics)


def measure_model(root, *, scope="scene", model_file="model.xml", static=False):
    """Load past the geom cap and report the actual scale instead of refusing to look.

    The fixed cap turned a large but valid scene into an outright failure: nothing was
    measured, so the run could neither accept nor locate a defect. Checks that cannot
    complete at this size degrade to an explicit partial coverage row instead, which
    keeps "unknown" distinguishable from "passed" while still producing evidence.
    """
    model = _load_model(root, scope=scope, model_file=model_file, enforce_geom_limit=False,
                        enforce_solver_limit=not (scope == "scene" and static))
    return model, {"ngeom": int(model.ngeom), "nbody": int(model.nbody), "nv": int(model.nv),
                   "limit": 50000 if scope == "scene" else 10000,
                   "dynamics_within_limits":model.nbody <= 4096 and model.nv <= 2048}


def scale_limits(model, scope="scene"):
    """Per-model work budgets derived from actual size, never from a single threshold."""
    base = {"scene": 50000, "asset": 10000}[scope]
    ratio = max(1.0, model.ngeom / base)
    return {"pair_budget": int(200000 / min(ratio, 4.0)),
            "joint_samples": max(56, int(1024 / min(ratio, 4.0))),
            "degraded": model.ngeom > base or getattr(model,"nbody",0)>4096 or getattr(model,"nv",0)>2048,
            "ngeom": int(model.ngeom), "base_limit": base}


def validate_manifest(value, model, asset_ids):
    import mujoco
    import numpy as np
    require(isinstance(value, dict) and value.get("schema_version") == "1.0", "交互清单版本无效")
    require(value.get("asset_ids") == asset_ids, f"清单 asset_ids 必须与任务输入完全相同：期望 {asset_ids!r}，实际 {value.get('asset_ids')!r}；tool/fixture 中这些编号表示服务范围，不要求重复交付仪器")
    for key in ("parts", "interfaces", "assumptions", "approximations", "buttons", "connections"):
        require(isinstance(value.get(key), list) and len(value[key]) <= 2048, f"{key} 必须为有界列表")
    require(value["parts"], "至少声明一个语义部件")
    records=value.get("construction_records",[])
    require(isinstance(records,list) and len(records)<=2048,"构建记录须为有界列表")
    for record in records:
        require(isinstance(record,dict) and isinstance(record.get("tool"),str) and type(record.get("version")) is int
                and isinstance(record.get("parameters"),dict) and record.get("provenance")=="engineering_constructor"
                and record.get("certified") is False,"构建工具记录不得伪装为模型自主设计或质量认证")
    for key in ("assumptions", "approximations"):
        require(all(isinstance(x, str) for x in value[key]), "假设和近似须为文本")
    for key in ("parts", "interfaces", "buttons", "connections"):
        ids = [entry.get("id") for entry in value[key] if isinstance(entry, dict)]
        require(len(ids) == len(value[key]) and all(isinstance(i, str) and i for i in ids)
                and len(set(ids)) == len(ids), f"{key} 编号缺失或重复")
    for collection,fields in (('parts',('body',)),('interfaces',('asset_id','kind','body')),
                              ('buttons',('joint','geom','event')),('connections',('equality','release_event'))):
        for entry in value[collection]:
            require(all(isinstance(entry.get(field),str) and entry[field] for field in fields),
                    f'{collection} 的引用必须是非空名称，不能是数字索引')
    for interface in value['interfaces']:
        require(interface['kind']=='software' or isinstance(interface.get('site'),str) and interface['site'],
                '接口坐标系必须按名称引用')
    try:
        for part in value["parts"]:
            require(model.body(part["body"]).id > 0, "语义部件不能引用世界根节点")
        for interface in value["interfaces"]:
            require(interface["asset_id"] in asset_ids and interface["kind"] in {
                "grasp", "press", "rotate", "insert", "mount", "cut", "software"}, "接口类别或资产编号无效")
            body = model.body(interface["body"]).id
            if interface["kind"] != "software":
                site = model.site(interface["site"]).id
                require(model.site_bodyid[site] == body,
                        f"接口坐标系不属于声明的部件：接口 {interface['id']}，site={interface['site']}，"
                        f"实际body={model.body(int(model.site_bodyid[site])).name}，声明body={interface['body']}")
                if "approach_axis" in interface:
                    axis=interface["approach_axis"]
                    require(isinstance(axis,list) and len(axis)==3 and all(finite(v) for v in axis)
                            and abs(sum(v*v for v in axis)-1)<1e-5, "approach_axis须为接口坐标系中指向操作侧的单位向量")
        from .operation_interface import validate_operation
        for interface in value["interfaces"]:
            validate_operation(interface, model, {i["id"] for i in value["interfaces"]})
        events = {i["id"] for i in value["interfaces"] if i["kind"] == "software"}
        for button in value["buttons"]:
            joint = model.joint(button["joint"])
            geom = model.geom(button["geom"])
            require(joint.type[0] == mujoco.mjtJoint.mjJNT_SLIDE and joint.limited[0], "按钮需有限位移动关节")
            require(geom.bodyid[0] == joint.bodyid[0], "按钮接触面和关节不属于同一部件")
            require(finite(button["pressed_at"]) and finite(button["released_at"])
                    and joint.range[0] <= button["released_at"] < button["pressed_at"] <= joint.range[1], "按钮触发范围无效")
            require(isinstance(button["event"], str) and button["event"] and button["event"] not in events, "按钮事件重复或无效")
            events.add(button["event"])
        mechanical = value.get("mechanical_events", [])
        require(isinstance(mechanical,list) and len(mechanical)<=2048, "机械事件须为有界列表")
        by_id = {i["id"]:i for i in value["interfaces"]}
        for event in mechanical:
            require(isinstance(event,dict) and set(event)=={"id","interface","action"}, "机械事件字段须为id/interface/action")
            require(isinstance(event["id"],str) and event["id"] and event["id"] not in events, "机械事件编号重复或无效")
            interface = by_id.get(event["interface"]) if isinstance(event["interface"],str) else None
            require(interface is not None and interface["kind"] in {"grasp","press","rotate","insert","cut"},
                    "机械事件须引用真实操作接口，不能用装配点或软件接口占位")
            require(event["action"] in {"release","attach"}, "机械事件动作须为release或attach")
            events.add(event["id"])
        mechanical_actions = {e["id"]:e["action"] for e in mechanical}
        def under(body, root):
            while body and body != root:
                body = int(model.body_parentid[body])
            return body == root

        def attach_contract(connection, sides, kind):
            """Shared re-attachment contract; sides are the two units the sites must belong to."""
            require(not connection["irreversible"] and connection["attach_event"] in events
                    and connection["attach_event"] != connection["release_event"], "不可逆连接不能重新接合；接合事件须独立声明")
            sites = connection.get("capture_sites")
            require(isinstance(sites,list) and len(sites) == 2 and len(set(sites)) == 2,"重新接合须有两个不同定位坐标")
            require(finite(connection.get("position_tolerance_m")) and 0 < connection["position_tolerance_m"] <= .005
                    and finite(connection.get("angle_tolerance_rad")) and 0 < connection["angle_tolerance_rad"] <= .2,
                    "重新接合捕获公差无效")
            site_ids = [model.site(n).id for n in sites]
            if kind == "latch":
                # 一个坐标在盖体上，另一个在盖体之外：锁止对不要求两坐标分属不同运动树。
                lid, base = sides
                require(under(int(model.site_bodyid[site_ids[0]]),lid) != under(int(model.site_bodyid[site_ids[1]]),lid),
                        "铰链锁止的两个接合定位坐标须一个属于被锁部件、另一个位于其支撑侧，不能同在盖体一侧")
            else:
                for sid,expected_body in zip(site_ids,sides):
                    require(under(int(model.site_bodyid[sid]),expected_body),"接合定位坐标须分别属于约束两端")
            data = mujoco.MjData(model)
            from .static_state import geometry_state
            geometry_state(model,data)
            require(np.linalg.norm(data.site_xpos[site_ids[0]]-data.site_xpos[site_ids[1]]) <= 1e-6
                    and np.allclose(data.site_xmat[site_ids[0]],data.site_xmat[site_ids[1]],atol=1e-6),
                    "默认锁合姿态的定位坐标必须重合，避免激活错误参考位姿")

        for connection in value["connections"]:
            require(mechanical_actions.get(connection["release_event"],"release")=="release"
                    and mechanical_actions.get(connection.get("attach_event"),"attach")=="attach", "机械事件动作与连接用途不一致")
            eq = model.equality(connection["equality"])
            kind = connection.get("kind", "detachable")
            require(kind in {"detachable", "latch"},
                    f"连接 {connection['id']} 的 kind 只能为 detachable（可整体拆离）或 latch（铰链锁止）；"
                    "盖子这类始终铰接、解锁后只能翻开的部件用 latch，能整体取下的部件用 detachable")
            require(connection["release_event"] in events and type(connection["irreversible"]) is bool,
                    f"连接 {connection['id']} 的 release_event（释放事件）或 irreversible 无效")
            require(eq.active0[0], f"连接 {connection['id']} 的约束须初始启用，锁止状态才算生效")
            if kind == "latch":
                require(eq.type[0] == mujoco.mjtEq.mjEQ_JOINT,
                        f"连接 {connection['id']} 声明为 latch（解除锁止后仍由铰链连接），"
                        f"equality {connection['equality']!r} 必须是 joint 型约束以锁住铰链角度，当前为 {int(eq.type[0])}。"
                        "若该部件确实要整体拆离，请改为 kind=detachable 并使用 weld 约束")
                joint = int(eq.obj1id[0])
                require(0 <= joint < model.njnt, f"连接 {connection['id']} 的 joint 约束未引用实际关节")
                require(model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE),
                        f"连接 {connection['id']} 锁住的关节 {model.joint(joint).name!r} 须为 hinge 或 slide；"
                        "ball/free 关节不能表达锁止后的铰链运动")
                lid = int(model.jnt_bodyid[joint])
                require(lid > 0, f"连接 {connection['id']} 锁住的关节不能位于世界根节点")
                require(not connection["irreversible"],
                        f"连接 {connection['id']} 是铰链锁止，必须能再次锁止；irreversible 只适用于可拆连接")
                if "attach_event" in connection:
                    attach_contract(connection, [lid, int(model.body_parentid[lid])], "latch")
            else:
                require(eq.type[0] == mujoco.mjtEq.mjEQ_WELD,
                        f"连接 {connection['id']} 声明为 detachable（整体拆离），"
                        f"equality {connection['equality']!r} 须为 weld 约束，当前为 {int(eq.type[0])}。"
                        "若部件只是被锁住、始终与主体铰接，请改为 kind=latch 并使用 joint 约束")
                bodies = [int(eq.obj1id[0]), int(eq.obj2id[0])]
                roots = []
                for body in bodies:
                    while model.body_parentid[body] != 0:
                        body = int(model.body_parentid[body])
                    roots.append(body)
                require(roots[0] != roots[1],
                        f"连接 {connection['id']} 声明为可拆件，但约束两端仍属同一永久运动树（根部件 {roots[0]}）；"
                        "可拆件一开始就须是独立运动单元。盖子这类始终铰接的部件应改为 kind=latch")
                require(any(model.body_jntnum[b] and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
                            for b in roots if b),
                        f"连接 {connection['id']} 声明为可拆件，但两端都没有自由运动能力，拆下后无法独立运动")
                if "attach_event" in connection:
                    attach_contract(connection, bodies, "detachable")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError("交互清单缺少字段或引用不存在") from exc
    return value


def extent(model, data, names, axis):
    """在默认姿态测量指定几何的世界轴包络；不采用模型自报尺寸。"""
    import mujoco
    import numpy as np
    lows, highs = [], []
    for name in names:
        g = model.geom(name).id
        rotation = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g]
        kind, size = model.geom_type[g], model.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            adr, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[adr:adr+count] @ rotation.T + center
            lows.append(float(vertices[:, axis].min())); highs.append(float(vertices[:, axis].max()))
            continue
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            half = float(np.abs(rotation[axis]) @ size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = float(size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            half = float(np.linalg.norm(rotation[axis] * size))
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            vertical = abs(float(rotation[axis, 2]))
            half = float(size[1]) * vertical + float(size[0]) * (
                1 if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else math.sqrt(max(0, 1 - vertical*vertical)))
        else:
            raise ValidationError("此尺寸检查暂不支持指定几何类型")
        lows.append(float(center[axis] - half)); highs.append(float(center[axis] + half))
    return max(highs) - min(lows)


def _validate_one(root, asset_ids, profile, render_dir=None, *, model_file="model.xml", manifest_value=None):
    from .check_report import report_status, validate_report
    import mujoco
    import numpy as np
    static = profile.get("validation_mode", "full") == "static"
    results = []
    def add(name, status, detail, **evidence):
        results.append(dict(id=name, status=status, detail=detail, **evidence))
    try:
        model = load_model(root, model_file=model_file)
        require(not any(t in (mujoco.mjtGeom.mjGEOM_PLANE,mujoco.mjtGeom.mjGEOM_HFIELD) for t in model.geom_type),
                "单资产不能生成全局地面或高度场，请使用有限尺寸几何")
        require(all(model.body(b).name for b in range(1,model.nbody)), "资产所有物理部件必须命名，才能跟踪装配与质量")
        require(any(model.geom_contype[g] or model.geom_conaffinity[g] for g in range(model.ngeom)),
                "资产必须包含有效碰撞几何，不能只提交视觉外壳")
        require(all(model.geom(g).name and model.geom_bodyid[g] > 0 for g in range(model.ngeom)
                    if model.geom_contype[g] or model.geom_conaffinity[g]),
                "资产碰撞几何须命名且属于物理部件，不能放在世界根节点")
        manifest = validate_manifest(manifest_value if manifest_value is not None else read_json(Path(root)/"interfaces.json"), model, asset_ids)
        add("compile_and_interfaces", "pass", "已独立编译并解析实际引用")
    except (ValidationError, ValueError, KeyError, ParseError, DefusedXmlException) as exc:
        add("compile_and_interfaces", "fail", str(exc)[:2000])
        return {"status": "fail", "checks": results, "renders": [], "manifest": None,
                "coverage": ["其他检查因编译或接口错误未执行"]}
    data = mujoco.MjData(model)
    from .static_state import geometry_state
    if static:
        geometry_state(model, data, contacts=True)
    else:
        mujoco.mj_forward(model, data)
    from .material import integrity
    results.append(integrity(model,data,tolerance=profile["penetration_tolerance_m"]))
    geometry_summary = {"unit":"m", "interfaces":[], "bodies":[]}
    for interface in manifest["interfaces"]:
        if interface.get("site"):
            site = model.site(interface["site"]).id
            geometry_summary["interfaces"].append({"id":interface["id"],"site":interface["site"],
                "position":data.site_xpos[site].tolist(),"rotation":data.site_xmat[site].reshape(3,3).tolist()})
    for part in manifest["parts"]:
        body = model.body(part["body"]).id
        geometry_summary["bodies"].append({"id":part["id"],"body":part["body"],
            "position":data.xpos[body].tolist(),"mass_kg":float(model.body_mass[body])})
    for item in profile["dimensions"]:
        try:
            actual = extent(model, data, item["geoms"], item["axis"])
            add(item["id"], "pass" if abs(actual-item["expected_m"]) <= item["tolerance_m"] else "fail",
                "默认姿态下实际几何包络", actual_m=actual, expected_m=item["expected_m"], tolerance_m=item["tolerance_m"])
        except (KeyError, ValidationError) as exc:
            add(item["id"], "fail", str(exc))
    interfaces = {i["id"] for i in manifest["interfaces"]}
    for name in profile["required_interfaces"]:
        add("interface:"+name, "pass" if name in interfaces else "fail", "只读要求中的接口存在性")
    kinds = {"hinge": mujoco.mjtJoint.mjJNT_HINGE, "slide": mujoco.mjtJoint.mjJNT_SLIDE,
             "ball": mujoco.mjtJoint.mjJNT_BALL, "free": mujoco.mjtJoint.mjJNT_FREE}
    for item in profile["joints"]:
        try:
            j = model.joint(item["name"])
            valid = j.type[0] == kinds[item["type"]] and np.allclose(j.range, item["range"], rtol=0, atol=item["tolerance"])
            add("joint:"+item["name"], "pass" if valid else "fail", "实际关节类型及限位", actual_range=j.range.tolist())
        except KeyError:
            add("joint:"+item["name"], "fail", "缺少指定关节")
    contacts = [{"geom1": model.geom(int(c.geom1)).name, "geom2": model.geom(int(c.geom2)).name,
                 "distance_m": float(c.dist)} for c in data.contact if c.dist < -profile["penetration_tolerance_m"]]
    add("initial_contacts", "fail" if contacts else "pass", "默认碰撞过滤下初始接触穿透", contacts=contacts[:40])
    poses = [("initial", data.qpos.copy())]
    sample_count = 0
    sweep_collisions = []
    for j in (range(model.njnt) if not static else ()):
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) or not model.jnt_limited[j]:
            continue
        for q in np.linspace(*model.jnt_range[j], 7):
            mujoco.mj_resetData(model, data)
            data.qpos[model.jnt_qposadr[j]] = q
            mujoco.mj_forward(model, data)
            sample_count += 1
            if any(c.dist < -profile["penetration_tolerance_m"] for c in data.contact):
                sweep_collisions.append({"joint": model.joint(j).name, "position": float(q)})
        if len(poses) < 4:
            poses.append((f"joint_{j}_end", data.qpos.copy()))
    add("sampled_joint_sweep", "skipped" if static else ("fail" if sweep_collisions else "pass"), "静态模式跳过" if static else "逐关节7点采样，非连续碰撞证明",
        samples=sample_count, collisions=sweep_collisions[:40])
    mujoco.mj_resetData(model, data)
    for _ in range(0 if static else profile["settle_steps"]):
        mujoco.mj_step(model, data)
    stable = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and not any(w.number for w in data.warning)
    add("numerical_stability", "skipped" if static else ("pass" if stable else "fail"), "静态模式跳过" if static else "仅检查短时数值有限性与引擎警告，不证明有支撑或静止",
        steps=0 if static else profile["settle_steps"], simulation_seconds=float(data.time))
    renders = []
    if render_dir is not None:
        try:
            renders = render_views(model, poses, Path(render_dir))
            add("render", "pass", "静态初始姿态多视角" if static else "初始姿态多视角及有限数量的关节末端姿态")
        except Exception as exc:
            add("render", "error", type(exc).__name__ + ": 无法生成审查图像")
    coverage = ["未自动证明静态几何间无交叠、支撑稳定、夹具适配或机器人任务可执行",
                "关节检查为独立姿态采样；按钮力学、连接释放与复合机构需专门操作探测",
                "资料中未绑定只读数值检查的尺寸仍未获独立量测认证"]
    if static:
        add("interaction_probes", "skipped", "静态模式不按压按钮或释放连接")
        coverage[1] = "静态模式：未执行关节开合、重力步进、按钮或连接操作验证"
    elif manifest["buttons"] or manifest["connections"]:
        from .probes import interaction_probes
        try:
            results.extend(interaction_probes(root,manifest))
        except Exception as exc:
            add("interaction_probes","error",str(exc)[:2000],error_kind='check_execution')
        coverage[1] = "按钮及释放使用框架压头/声明软件事件短探测；复合机构、重新接合与完整机器人路径仍需任务验证"
    status = report_status(results)
    return validate_report({"status": status, "checks": results, "renders": renders, "manifest": manifest,
            "geometry_summary":geometry_summary, "coverage": coverage,
            "validation_mode":"static" if static else "full"},domain='asset',mode='static' if static else 'full')


def render_views(model, poses, output, *, width=640, height=480):
    import mujoco
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    paths = []
    require(type(width) is int and type(height) is int and 64 <= width <= 2560 and 64 <= height <= 1920,
            "渲染尺寸超出支持范围")
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,width)
    model.vis.global_.offheight = max(model.vis.global_.offheight,height)
    # 框架提供基础照明，不能让生成代码遗漏灯光而产生不可审查的全黑图。
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    from .rendering import renderer as make_renderer
    with make_renderer(model, height=height, width=width) as renderer:
        for index, (label, qpos) in enumerate(poses):
            data.qpos[:] = qpos
            from .static_state import geometry_state
            geometry_state(model, data)
            for azimuth in ([45, 135, 225] if index == 0 else [45]):
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, camera)
                camera.azimuth = azimuth
                camera.elevation = -25
                renderer.update_scene(data, camera)
                name = f"{label}_{azimuth}.png"
                Image.fromarray(renderer.render()).save(output/name)
                paths.append(name)
    return paths


def validate_artifact(root, asset_ids, profile, render_dir=None):
    from .configurations import catalogue, selection
    from .check_report import report_status
    from asset_prep.storage import digest, file_hash
    if not (Path(root)/"interfaces.json").is_file():
        return _validate_one(root,asset_ids,profile,render_dir)
    try:
        variants = catalogue(root)
    except (ValidationError, OSError, ValueError) as exc:
        return {"status":"fail", "checks":[{"id":"configurations","status":"fail","detail":str(exc)}],
                "renders":[], "manifest":None, "coverage":["configuration delivery invalid"]}
    if set(variants) == {"default"}:
        return _validate_one(root, asset_ids, profile, render_dir)
    reports, rows, renders = {}, [], []
    for name in sorted(variants):
        _, model_file, manifest = selection(root, name)
        output = Path(render_dir)/name if render_dir is not None else None
        report = _validate_one(root, asset_ids, profile, output, model_file=model_file, manifest_value=manifest)
        report["configuration"] = name
        report["model_sha256"] = file_hash(Path(root)/model_file)
        report["manifest_sha256"] = digest(manifest)
        reports[name] = report
        rows.extend({**row, "id":name+":"+row["id"], "configuration":name} for row in report["checks"])
        renders.extend(name+"/"+path for path in report.get("renders",[]))
    return {"status":report_status(rows),"checks":rows,"renders":renders,
            "manifest":read_json(Path(root)/"interfaces.json"),
            "configurations":reports,"geometry_summary":{"configurations":{k:v.get("geometry_summary",{}) for k,v in reports.items()}},
            "coverage":["Every declared configuration was independently loaded; evidence is configuration-specific."],
            "validation_mode":profile.get("validation_mode","full")}
