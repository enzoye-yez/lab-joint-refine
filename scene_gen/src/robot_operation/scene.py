"""从不可变静态产物派生自由底盘场景，按名字迁移状态。"""
from dataclasses import asdict
from pathlib import Path
import shutil
from defusedxml import ElementTree as ET
from xml.etree.ElementTree import SubElement, tostring
import mujoco
from .common import copy_data
import numpy as np

from asset_generation.checks import load_scene_model
from scene_generation.runtime import actor_groups
from .common import Settings, atomic_json, identity, need, new_directory, read


def joint_width(kind):
    return (7, 6) if kind == mujoco.mjtJoint.mjJNT_FREE else ((4, 3) if kind == mujoco.mjtJoint.mjJNT_BALL else (1, 1))


def initial(model, metadata):
    data = mujoco.MjData(model)
    need(len(metadata["home_qpos"]) == model.nq and len(metadata["home_ctrl"]) == model.nu, "初态维度不匹配")
    data.qpos[:] = metadata["home_qpos"]
    data.ctrl[:] = metadata["home_ctrl"]
    if "home_qvel" in metadata:
        data.qvel[:] = metadata["home_qvel"]
    if "home_eq_active" in metadata:
        data.eq_active[:] = metadata["home_eq_active"]
    mujoco.mj_forward(model, data)
    return data


def prepare(source, destination, expected_hash, settings=None):
    settings = settings or Settings()
    source = Path(source).resolve()
    source_hash = identity(source)
    need(source_hash == expected_hash, "来源场景摘要不一致", "stale_scene")
    meta, manifest = read(source/"scene.json"), read(source/"interfaces.json")
    old = load_scene_model(source, dynamics=True)
    before = initial(old, meta)
    # 此绑定还会拒绝未实际安装的工具；没有原厂夹爪回退。
    actor_groups(old, meta)
    tree = ET.parse(source/"model.xml")
    root = tree.getroot()
    # Surface grasps need a contact patch, not a single arbitrary convex-mesh corner.
    # This solver change belongs to the derived scene; vendor resources stay immutable.
    option=root.find("option")
    if option is None:option=SubElement(root,"option")
    flags=option.find("flag")
    if flags is None:flags=SubElement(option,"flag")
    flags.set("multiccd","enable")
    body = root.find(".//body[@name='robot__layout_root']")
    need(body is not None, "首版动力学转换要求整机的平面布局根")
    joints = body.findall("joint")
    removed = {"robot__base_x", "robot__base_y", "robot__base_yaw"}
    need({j.get("name") for j in joints} == removed, "底盘根结构与转换契约不符")
    root_id = old.body("robot__layout_root").id
    for node in joints:
        body.remove(node)
    SubElement(body, "freejoint", {"name": "robot__free_base"})
    for key in root.findall("keyframe"):
        root.remove(key)
    out = new_directory(destination, [source])
    for path in source.rglob("*"):
        if path.is_file():
            target = out/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    (out/"model.xml").write_text(tostring(root, encoding="unicode"), encoding="utf-8")
    model = load_scene_model(out, dynamics=True)
    data = mujoco.MjData(model)
    for j in range(old.njnt):
        name = old.joint(j).name
        if name in removed:
            continue
        other = model.joint(name)
        nq, nv = joint_width(old.jnt_type[j])
        need(other.type[0] == old.jnt_type[j], "转换改变了非底盘关节类型")
        qa, va = int(old.jnt_qposadr[j]), int(old.jnt_dofadr[j])
        qb, vb = int(other.qposadr[0]), int(other.dofadr[0])
        data.qpos[qb:qb+nq] = before.qpos[qa:qa+nq]
        data.qvel[vb:vb+nv] = before.qvel[va:va+nv]
    base = model.joint("robot__free_base")
    q = int(base.qposadr[0])
    data.qpos[q:q+3] = before.xpos[root_id]
    data.qpos[q+3:q+7] = before.xquat[root_id]
    for a in range(old.nu):
        name = old.actuator(a).name
        need(bool(name), "执行器必须有稳定名称")
        data.ctrl[model.actuator(name).id] = before.ctrl[a]
    # 未命名的原厂夹爪联动也必须保持；转换没有增删或重排约束。
    need(old.neq == model.neq and np.array_equal(old.eq_type,model.eq_type), "转换改变了连接约束")
    data.eq_active[:] = before.eq_active
    mujoco.mj_forward(model, data)
    # 检查迁移保持所有身体世界位姿，防止法兰/自由工具的隐式跳变。
    need(model.nbody == old.nbody and np.array_equal(model.body_parentid, old.body_parentid), "转换改变了身体拓扑")
    need(all(old.body(b).name == model.body(b).name for b in range(old.nbody)), "转换改变了身体顺序")
    errors = {str(b)+":"+old.body(b).name: float(np.linalg.norm(before.xpos[b]-data.xpos[b]))
              for b in range(1, old.nbody)}
    need(max(errors.values(), default=0) < 1e-6, "自由根迁移改变了装配世界位姿")
    need(np.max(np.abs(before.xmat-data.xmat), initial=0) < 1e-6, "自由根迁移改变了装配方向")
    meta.update(home_qpos=data.qpos.tolist(), home_qvel=data.qvel.tolist(), home_ctrl=data.ctrl.tolist(),
                home_eq_active=data.eq_active.tolist())
    meta["operation_conversion"] = {"version": 1, "source_sha256": source_hash,
        "root": "robot__free_base", "removed_joints": sorted(removed),
        "joint_mapping": "name_and_type", "max_body_position_error_m": max(errors.values(), default=0),
        "approximations": ["沿用来源资产质量、摩擦、执行器及工具惯性建模值；不认证真实硬件动力学"],
        "solver":{"multiple_convex_contacts":True,"source_enableflags":int(old.opt.enableflags),
                  "derived_enableflags":int(model.opt.enableflags)},
        "source": str(source)}
    atomic_json(out/"scene.json", meta)
    # 准入检查在副本中进行，不暗中改变任务初态。
    probe = mujoco.MjData(model)
    copy_data(probe, model, data)
    warnings = []
    for _ in range(max(1, int(settings.settle_seconds/model.opt.timestep))):
        mujoco.mj_step(model, probe)
        if not np.isfinite(probe.qpos).all() or any(w.number for w in probe.warning):
            warnings.append("非有限状态或求解器警告")
            break
    penetration = max((-c.dist for c in probe.contact), default=0.)
    tilt = float(np.arccos(np.clip(probe.xmat[model.body("robot__layout_root").id].reshape(3,3)[2,2], -1, 1)))
    settled = float(np.max(np.abs(probe.qvel), initial=0))
    if penetration > settings.penetration_m: warnings.append("初始稳定过程存在过深接触")
    if tilt > settings.base_tilt_rad: warnings.append("底盘倾斜超过工程边界")
    if settled > settings.stopped_speed: warnings.append("初始状态未稳定")
    report = {"status": "blocked" if warnings else "pass", "warnings": warnings,
        "max_penetration_m": float(penetration), "base_tilt_rad": tilt, "max_speed": settled,
        "settings": asdict(settings), "initial_state_changed": False,
        "coverage": "动力学初始化、绑定和静置；不认证完整任务或来源场景待检查项",
        "mechanical_events_pending": [e["id"] for e in manifest.get("mechanical_events", [])]}
    atomic_json(out/"admission.json", report)
    return {"scene": str(out), "scene_sha256": identity(out), "admission": report}


def load(path, expected_hash):
    root = Path(path)
    need(identity(root) == expected_hash, "操作场景已变更，旧轨迹失效", "stale_scene")
    meta, manifest = read(root/"scene.json"), read(root/"interfaces.json")
    need(meta.get("operation_conversion", {}).get("version") == 1, "不是操作动力学场景")
    model = load_scene_model(root, dynamics=True)
    need(model.joint("robot__free_base").type[0] == mujoco.mjtJoint.mjJNT_FREE, "操作底盘必须自由运动")
    return model, initial(model, meta), meta, manifest
