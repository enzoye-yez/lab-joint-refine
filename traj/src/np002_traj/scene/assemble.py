"""Replace the published scene's robot with ours, keeping the room, instruments and layout.

Everything is driven by the scene metadata: the old robot and the tools mounted on it are the
`robot__` subtree and the instances listed in scene.json `mounts`. No instrument is named here.
The result is cached as XML (editable) and MJB (fast to load) under runs/scene_cache/<digest>/.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from ..robot import PROFILE, ROOT

OLD_ROBOT = "robot__"


def digest(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def _refers(element, prefixes):
    """Whether any name-valued attribute of a spec element starts with a removed prefix."""
    for field in ("name", "name1", "name2", "target", "objname", "refname", "bodyname1", "bodyname2",
                  "geomname1", "geomname2", "slidersite", "cranksite"):
        value = getattr(element, field, None)
        if isinstance(value, str) and value.startswith(prefixes):
            return True
    return False


def _strip(spec, prefixes):
    removed = {}
    for kind in ("actuators", "sensors", "equalities", "excludes", "pairs", "tendons", "keys"):
        items = [e for e in getattr(spec, kind) if kind == "keys" or _refers(e, prefixes)]
        for item in items:
            spec.delete(item)
        removed[kind] = len(items)
    roots = [b for b in spec.bodies if b.name.startswith(prefixes) and not b.parent.name.startswith(prefixes)]
    for body in roots:
        spec.delete(body)
    removed["bodies"] = len(roots)
    for kind in ("meshes", "materials", "textures"):
        items = [e for e in getattr(spec, kind) if e.name.startswith(prefixes)]
        for item in items:
            spec.delete(item)
        removed[kind] = len(items)
    return removed


def _named_values(model, data_qpos, ctrl):
    joints = {model.joint(j).name: data_qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + _width(model, j)].copy()
              for j in range(model.njnt)}
    actuators = {model.actuator(a).name: float(ctrl[a]) for a in range(model.nu)}
    return joints, actuators


def _width(model, joint):
    return {0: 7, 1: 4, 2: 1, 3: 1}[int(model.jnt_type[joint])]


def assemble(source, out_root=None, prefix="robot__"):
    source = Path(source)
    meta = json.loads((source / "scene.json").read_text(encoding="utf-8"))
    robot_xml = ROOT / "data/robot/model.xml"
    key = digest([source / "model.xml", source / "scene.json", robot_xml, PROFILE])
    out = Path(out_root or ROOT / "runs/scene_cache") / key
    if (out / "model.mjb").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)

    spec = mujoco.MjSpec.from_file(str(source / "model.xml"))
    original = spec.compile()
    room_joints, room_actuators = _named_values(original, original.key_qpos[0], original.key_ctrl[0])
    prefixes = tuple([OLD_ROBOT] + [m["instance"] + "__" for m in meta.get("mounts", [])])
    removed = _strip(spec, prefixes)
    spec.nkey = 0   # a reserved <size nkey> would put blank keyframes before ours

    robot = mujoco.MjSpec.from_file(str(robot_xml))
    home_model = robot.compile()
    robot_joints, robot_actuators = _named_values(home_model, home_model.key_qpos[0], home_model.key_ctrl[0])
    for item in list(robot.keys):
        robot.delete(item)
    x, y, yaw = meta["plan"]["layout"]["robot_home"]
    frame = spec.worldbody.add_frame(pos=[x, y, 0.0], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    spec.attach(robot, prefix=prefix, frame=frame)
    model = spec.compile()

    qpos, ctrl = model.qpos0.copy(), np.zeros(model.nu)
    for j in range(model.njnt):
        name = model.joint(j).name
        value = room_joints.get(name) if not name.startswith(prefix) else robot_joints.get(name[len(prefix):])
        if value is not None:
            qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + len(value)] = value
    for a in range(model.nu):
        name = model.actuator(a).name
        ctrl[a] = room_actuators.get(name, 0.0) if not name.startswith(prefix) else robot_actuators.get(name[len(prefix):], 0.0)
    spec.add_key(name="assembled", qpos=qpos.tolist(), ctrl=ctrl.tolist())
    model = spec.compile()

    xml = spec.to_xml()
    xml = xml.replace('file="resources/', f'file="{source / "resources"}/')
    xml = xml.replace('file="assets/', f'file="{ROOT / "data/vendor/fr3/assets"}/')
    (out / "model.xml").write_text(xml, encoding="utf-8")
    mujoco.mj_saveModel(model, str(out / "model.mjb"))
    (out / "assembly.json").write_text(json.dumps({
        "source": str(source), "digest": key, "robot_prefix": prefix, "removed_prefixes": list(prefixes),
        "removed": removed, "robot_home": [x, y, yaw], "nq": model.nq, "nv": model.nv, "ngeom": model.ngeom,
    }, indent=1), encoding="utf-8")
    return out
