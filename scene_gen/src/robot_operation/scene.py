"""Derive a free-base scene from immutable static artifacts and migrate state by name."""
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
    need(len(metadata["home_qpos"]) == model.nq and len(metadata["home_ctrl"]) == model.nu, "Initial state dimension mismatch")
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
    need(source_hash == expected_hash, "Source scene digest mismatch", "stale_scene")
    meta, manifest = read(source/"scene.json"), read(source/"interfaces.json")
    old = load_scene_model(source, dynamics=True)
    before = initial(old, meta)
    # This binding also rejects tools that are not actually mounted; there is no fallback to the stock gripper.
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
    need(body is not None, "The first-version dynamics conversion requires the integrated robot's planar layout root")
    joints = body.findall("joint")
    removed = {"robot__base_x", "robot__base_y", "robot__base_yaw"}
    need({j.get("name") for j in joints} == removed, "Base root structure does not match the conversion contract")
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
        need(other.type[0] == old.jnt_type[j], "Conversion changed a non-base joint type")
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
        need(bool(name), "Actuators must have stable names")
        data.ctrl[model.actuator(name).id] = before.ctrl[a]
    # Unnamed stock-gripper couplings must also be kept; the conversion adds, removes or reorders no constraints.
    need(old.neq == model.neq and np.array_equal(old.eq_type,model.eq_type), "Conversion changed connection constraints")
    data.eq_active[:] = before.eq_active
    mujoco.mj_forward(model, data)
    # Check that migration preserves every body's world pose, preventing implicit jumps of flanges or free tools.
    need(model.nbody == old.nbody and np.array_equal(model.body_parentid, old.body_parentid), "Conversion changed the body topology")
    need(all(old.body(b).name == model.body(b).name for b in range(old.nbody)), "Conversion changed the body order")
    errors = {str(b)+":"+old.body(b).name: float(np.linalg.norm(before.xpos[b]-data.xpos[b]))
              for b in range(1, old.nbody)}
    need(max(errors.values(), default=0) < 1e-6, "Free-root migration changed the assembly world pose")
    need(np.max(np.abs(before.xmat-data.xmat), initial=0) < 1e-6, "Free-root migration changed the assembly orientation")
    meta.update(home_qpos=data.qpos.tolist(), home_qvel=data.qvel.tolist(), home_ctrl=data.ctrl.tolist(),
                home_eq_active=data.eq_active.tolist())
    meta["operation_conversion"] = {"version": 1, "source_sha256": source_hash,
        "root": "robot__free_base", "removed_joints": sorted(removed),
        "joint_mapping": "name_and_type", "max_body_position_error_m": max(errors.values(), default=0),
        "approximations": ["Reuses the source assets' modelled mass, friction, actuator and tool inertia values; does not certify real-hardware dynamics"],
        "solver":{"multiple_convex_contacts":True,"source_enableflags":int(old.opt.enableflags),
                  "derived_enableflags":int(model.opt.enableflags)},
        "source": str(source)}
    atomic_json(out/"scene.json", meta)
    # Admission checks run on a copy and never silently change the task's initial state.
    probe = mujoco.MjData(model)
    copy_data(probe, model, data)
    warnings = []
    for _ in range(max(1, int(settings.settle_seconds/model.opt.timestep))):
        mujoco.mj_step(model, probe)
        if not np.isfinite(probe.qpos).all() or any(w.number for w in probe.warning):
            warnings.append("Non-finite state or solver warning")
            break
    penetration = max((-c.dist for c in probe.contact), default=0.)
    tilt = float(np.arccos(np.clip(probe.xmat[model.body("robot__layout_root").id].reshape(3,3)[2,2], -1, 1)))
    settled = float(np.max(np.abs(probe.qvel), initial=0))
    if penetration > settings.penetration_m: warnings.append("Excessively deep contact during initial settling")
    if tilt > settings.base_tilt_rad: warnings.append("Base tilt exceeds the engineering bound")
    if settled > settings.stopped_speed: warnings.append("Initial state did not settle")
    report = {"status": "blocked" if warnings else "pass", "warnings": warnings,
        "max_penetration_m": float(penetration), "base_tilt_rad": tilt, "max_speed": settled,
        "settings": asdict(settings), "initial_state_changed": False,
        "coverage": "Dynamics initialization, binding and settling; does not certify the full task or pending source-scene checks",
        "mechanical_events_pending": [e["id"] for e in manifest.get("mechanical_events", [])]}
    atomic_json(out/"admission.json", report)
    return {"scene": str(out), "scene_sha256": identity(out), "admission": report}


def load(path, expected_hash):
    root = Path(path)
    need(identity(root) == expected_hash, "Operation scene changed; old trajectories are invalid", "stale_scene")
    meta, manifest = read(root/"scene.json"), read(root/"interfaces.json")
    need(meta.get("operation_conversion", {}).get("version") == 1, "Not an operation dynamics scene")
    model = load_scene_model(root, dynamics=True)
    need(model.joint("robot__free_base").type[0] == mujoco.mjtJoint.mjJNT_FREE, "The operation base must move freely")
    return model, initial(model, meta), meta, manifest
