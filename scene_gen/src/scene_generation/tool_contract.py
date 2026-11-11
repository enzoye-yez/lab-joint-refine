"""One tool contract for authoring, admission and assembly; no quality exemptions."""
import numpy as np
import mujoco
from defusedxml import ElementTree as ET
from xml.etree.ElementTree import SubElement, tostring
from asset_generation.checks import load_model
from asset_generation.configurations import catalogue, selection
from asset_generation.contracts import require
from asset_generation.static_state import geometry_state

TOOL_POLICY = """The tool's local origin is the robot mounting flange and +Z is the flange axis; a tcp operation site is required. The main-body root is fixed-mounted and has no free joint.
Exchangeable parts must be separate bodies directly under worldbody with a freejoint, connected to the main body by initially active welds declared in the manifest connections (chained connections are allowed).
Do not put detachable parts in the main body's permanent subtree, and do not deliver free parts that are not connected to the main body. The framework mounts only the main body on the flange; independent parts keep their world-root free joints and are transformed to the actual flange pose.
The manifest's release/engage events, locating sites and tolerances follow the common connection contract; static checks do not certify operational capability. The mass of the complete device, including detachable parts, counts toward the payload. Active mechanisms contain real joints and actuators.
Keep this a simple first version that makes the robot's work easy: a flange adapter plus few parts (at most 6 printed parts), passive features (V-grooves, pegs, hooks, compliant pads, alignment chamfers) before extra actuators, and an actuator only where the protocol cannot be done without one. Its use (grasps, actuation, release) is verified and repaired later in the robot-operation stage, not now: deliver a clean, buildable, mountable draft.
It must be printable on a common FDM printer: every printed part fits a 220 x 220 x 220 mm build volume, walls and features are at least 1.5 mm thick (2 mm where they carry load), holes are at least 3 mm across, there are no enclosed voids and no interpenetrating parts, and each part has a flat face to print on. Moving joints use standard hardware (M3 screws, 3 mm pins, 608 bearings) or printed hinges with 0.3 mm clearance. Give printed parts PLA/PETG density (about 1250 kg/m^3). Blades, springs, screws, bearings, magnets and electronics are bought parts: declare them in the manifest parts with "fabrication": "purchased"; every other part counts as printed and is checked."""

OPERATION_REVIEW = {"verdict": "accept", "source": "operation_stage", "findings": [], "unverified": [], "motion_coverage": []}
OPERATION_CHECK = {"stage": "operation", "status": "pending",
                   "detail": "End-effector use is unverified: grasp, actuate and release every served interface "
                             "with a trajectory, and repair the tool where it fails"}
BUILD_VOLUME_M = .22
MIN_FEATURE_M = .0015


def printability(model, manifest):
    """3D-printing feasibility of each printed body: its collision geometry fits the build volume
    and no collision feature is thinner than MIN_FEATURE_M. Bodies of parts declared
    fabrication="purchased" are skipped. Returns problem strings; empty means printable."""
    purchased = {model.body(p["body"]).id for p in manifest["parts"] if p.get("fabrication") == "purchased"}
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    points, problems = {}, []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0 or b in purchased or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        name = model.geom(g).name or str(g)
        center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        kind = model.geom_type[g]
        size = model.geom_size[g]
        thickness = 2*size[0] if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                          mujoco.mjtGeom.mjGEOM_CYLINDER) else 2*float(np.min(half))
        if thickness < MIN_FEATURE_M:
            problems.append(f"geom {name} on body {model.body(b).name} is {thickness*1000:.1f} mm thick; printed "
                            f"features need at least {MIN_FEATURE_M*1000:.1f} mm (thicken it, make cosmetic detail visual-only, or mark a bought part fabrication=purchased)")
        rotation = model.geom_quat[g]
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, rotation)
        points.setdefault(b, []).append(model.geom_pos[g] + (center + corners*half) @ matrix.reshape(3, 3).T)
    for b, chunks in points.items():
        extent = np.ptp(np.vstack(chunks), axis=0)
        if extent.max() > BUILD_VOLUME_M:
            problems.append(f"printed body {model.body(b).name} spans {extent.max()*1000:.0f} mm; it must fit a "
                            f"{BUILD_VOLUME_M*1000:.0f} mm build volume, so split it into parts joined with screws")
    return problems


def printability_check(root):
    """The printability check row over every delivered configuration of a tool artifact."""
    problems = []
    for name in sorted(catalogue(root)):
        _, filename, manifest = selection(root, name)
        problems += [f"{name}: {p}" for p in printability(load_model(root, model_file=filename), manifest)]
    return {"id": "printability", "status": "fail" if problems else "pass", "problems": problems[:24],
            "detail": f"Printed parts fit a {BUILD_VOLUME_M*1000:.0f} mm build volume and have no collision feature "
                      f"thinner than {MIN_FEATURE_M*1000:.1f} mm; {len(problems)} problems"}


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
