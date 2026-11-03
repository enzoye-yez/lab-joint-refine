"""One tool contract for authoring, admission and assembly; no quality exemptions."""
import numpy as np
import mujoco
from defusedxml import ElementTree as ET
from xml.etree.ElementTree import SubElement, tostring
from asset_generation.contracts import require
from asset_generation.static_state import geometry_state

TOOL_POLICY = """The tool's local origin is the robot mounting flange and +Z is the flange axis; a tcp operation site is required. The main-body root is fixed-mounted and has no free joint.
Exchangeable parts must be separate bodies directly under worldbody with a freejoint, connected to the main body by initially active welds declared in the manifest connections (chained connections are allowed).
Do not put detachable parts in the main body's permanent subtree, and do not deliver free parts that are not connected to the main body. The framework mounts only the main body on the flange; independent parts keep their world-root free joints and are transformed to the actual flange pose.
The manifest's release/engage events, locating sites and tolerances follow the common connection contract; static checks do not certify operational capability. The mass of the complete device, including detachable parts, counts toward the payload. Active mechanisms contain real joints and actuators."""


def tool_roots(model, manifest):
    """Call after validate_manifest, which validates each releasable weld."""
    roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]
    free = {b for b in roots if model.body_jntnum[b] and
            model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE}
    fixed = set(roots) - free
    require(fixed, "end-effector must have a fixed-mounted main body; not every root may have a free joint")
    require(all(model.body(b).name for b in roots), "end-effector roots must be named")
    require(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp") >= 0, "end-effector must provide a tcp operation frame")
    actors = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    require(actors and all(model.geom(g).name for g in actors), "end-effector collision geoms must exist and be named")
    edges = []
    for connection in manifest["connections"]:
        eq = model.equality(connection["equality"])
        edges.append(tuple(int(model.body_rootid[int(b)]) for b in (eq.obj1id[0], eq.obj2id[0])))
    reached = set(fixed)
    while True:
        expanded = reached | {b for edge in edges if reached.intersection(edge) for b in edge if b}
        if expanded == reached: break
        reached = expanded
    require(free <= reached, "detachable end-effector parts must connect to the mounted main body through declared, initially active welds; not connected: " +
            ", ".join(model.body(b).name for b in sorted(free - reached)))
    return roots, free


def pose(position, quaternion, local_pos, local_quat):
    pos, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(pos, quat, np.asarray(position), np.asarray(quaternion),
                      np.asarray(local_pos), np.asarray(local_quat))
    return pos, quat


def attach_tool(spec, child, model, manifest, prefix, site_name):
    roots, free = tool_roots(model, manifest)
    if not free:
        spec.attach(child, prefix=prefix, site=spec.site(site_name))
        return spec
    # Attach once at world level to preserve cross-root welds, actuators and resources.
    # Move only mounted roots beneath the flange; free joints must remain world children.
    spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame())
    current = spec.compile()
    data = mujoco.MjData(current)
    geometry_state(current, data)
    sid = current.site(site_name).id
    flange_quat = np.zeros(4)
    mujoco.mju_mat2Quat(flange_quat, data.site_xmat[sid])
    tree = ET.fromstring(spec.to_xml())
    parents = {node: parent for parent in tree.iter() for node in parent}
    bodies = {b.get("name"): b for b in tree.iter("body")}
    parent_name = current.body(int(current.site_bodyid[sid])).name
    mount = SubElement(tree.find("worldbody") if current.site_bodyid[sid] == 0 else bodies[parent_name], "frame", pos=" ".join(map(str, current.site_pos[sid])),
                       quat=" ".join(map(str, current.site_quat[sid])))
    for b in roots:
        node = bodies[prefix + model.body(b).name]
        parents[node].remove(node)
        pos, quat = model.body_pos[b], model.body_quat[b]
        if b in free:
            pos, quat = pose(data.site_xpos[sid], flange_quat, pos, quat)
        node.set("pos", " ".join(map(str, pos)))
        node.set("quat", " ".join(map(str, quat)))
        (tree.find("worldbody") if b in free else mount).append(node)
    return mujoco.MjSpec.from_string(tostring(tree, encoding="unicode"), assets=spec.assets)


def align_free_roots(model, data, mounts):
    """Set initial detached-component poses at the robot home pose, without stepping."""
    geometry_state(model, data)
    for mount in mounts:
        sid = model.site(mount["attachment_site"]).id
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
        for root in mount["roots"]:
            if not root.get("free"): continue
            b = model.body(root["body"]).id
            q = model.jnt_qposadr[model.body_jntadr[b]]
            pos, rotation = pose(data.site_xpos[sid], quat, root["local_pos"], root["local_quat"])
            data.qpos[q:q+7] = np.r_[pos, rotation]


def has_free_parts(model, prefix):
    return any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and
               model.body(int(model.jnt_bodyid[j])).name.startswith(prefix) for j in range(model.njnt))
