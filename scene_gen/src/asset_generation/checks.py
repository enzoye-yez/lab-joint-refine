"""Independently load the actual MJCF and check it; never execute the generation script or tests the model wrote itself."""

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
    require(scope in {"asset", "scene"}, "Invalid model loading scope")
    max_geoms = 50000 if scope == "scene" else 10000
    max_xml_bytes = 32000000 if scope == "scene" else 8000000
    root = Path(root)
    xml = safe_path(root, model_file)
    require(xml.is_file() and xml.stat().st_size <= max_xml_bytes, "Missing or oversized model.xml")
    tree = ET.parse(xml)
    require(tree.getroot().tag == "mujoco", "Root element must be mujoco")
    for node in tree.iter():
        require(node.tag not in {"include", "plugin", "extension", "attach"}, "External includes and plugins are not supported in this version")
        if node.tag == "compiler":
            require(not any(k in node.attrib for k in ("assetdir", "meshdir", "texturedir")), "Resources must use paths relative to the artifact directly")
            require(node.get("angle", "degree") == "radian", "Radians must be set explicitly")
        if node.tag == "flag":
            require(all(node.get(key) != "disable" for key in ("contact", "constraint", "equality", "limit")),
                    "Contacts, constraints or joint limits must not be disabled to bypass checks")
        for key,value in node.attrib.items():
            if key.startswith("file"):
                target = safe_path(root, value)
                require(target.is_file(), "Referenced resource does not exist")
    require(tree.find("compiler") is not None, "Must explicitly provide compiler angle=radian")
    model = mujoco.MjModel.from_xml_path(str(xml))
    require(0 < model.ngeom and (not enforce_geom_limit or model.ngeom <= max_geoms),
            f"Model geometry size exceeds limit: scope={scope}, ngeom={model.ngeom}/{max_geoms}, "
            f"nbody={model.nbody}/4096, nv={model.nv}/2048")
    if enforce_solver_limit:
        require(model.nbody <= 4096 and model.nv <= 2048,
                f"Resources for asset admission or dynamics checks exceed limit: nbody={model.nbody}/4096, nv={model.nv}/2048; the static scene entry does not use the global dynamics solver")
    require(0.000001 <= model.opt.timestep <= 0.01, "Simulation timestep outside the engineering check range")
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
    require(isinstance(value, dict) and value.get("schema_version") == "1.0", "Invalid interaction manifest version")
    require(value.get("asset_ids") == asset_ids, f"manifest asset_ids must be exactly identical to the task input: expected {asset_ids!r}, got {value.get('asset_ids')!r}; for tool/fixture these ids denote the service scope and do not require re-delivering the instruments")
    for key in ("parts", "interfaces", "assumptions", "approximations", "buttons", "connections"):
        require(isinstance(value.get(key), list) and len(value[key]) <= 2048, f"{key} must be a bounded list")
    require(value["parts"], "Declare at least one semantic part")
    records=value.get("construction_records",[])
    require(isinstance(records,list) and len(records)<=2048,"Construction records must be a bounded list")
    for record in records:
        require(isinstance(record,dict) and isinstance(record.get("tool"),str) and type(record.get("version")) is int
                and isinstance(record.get("parameters"),dict) and record.get("provenance")=="engineering_constructor"
                and record.get("certified") is False,"Construction tool records must not masquerade as the model's own design or as quality certification")
    for key in ("assumptions", "approximations"):
        require(all(isinstance(x, str) for x in value[key]), "Assumptions and approximations must be text")
    for key in ("parts", "interfaces", "buttons", "connections"):
        ids = [entry.get("id") for entry in value[key] if isinstance(entry, dict)]
        require(len(ids) == len(value[key]) and all(isinstance(i, str) and i for i in ids)
                and len(set(ids)) == len(ids), f"{key} ids missing or duplicated")
    for collection,fields in (('parts',('body',)),('interfaces',('asset_id','kind','body')),
                              ('buttons',('joint','geom','event')),('connections',('equality','release_event'))):
        for entry in value[collection]:
            require(all(isinstance(entry.get(field),str) and entry[field] for field in fields),
                    f'{collection} references must be non-empty names, not numeric indices')
    for interface in value['interfaces']:
        require(interface['kind']=='software' or isinstance(interface.get('site'),str) and interface['site'],
                'Interface frames must be referenced by name')
    try:
        for part in value["parts"]:
            require(model.body(part["body"]).id > 0, "A semantic part must not reference the world root")
        for interface in value["interfaces"]:
            require(interface["asset_id"] in asset_ids and interface["kind"] in {
                "grasp", "press", "rotate", "insert", "mount", "cut", "software"}, "Invalid interface kind or asset id")
            body = model.body(interface["body"]).id
            if interface["kind"] != "software":
                site = model.site(interface["site"]).id
                require(model.site_bodyid[site] == body,
                        f"Interface frame does not belong to the declared part: interface {interface['id']}, site={interface['site']}, "
                        f"actual body={model.body(int(model.site_bodyid[site])).name}, declared body={interface['body']}")
                if "approach_axis" in interface:
                    axis=interface["approach_axis"]
                    require(isinstance(axis,list) and len(axis)==3 and all(finite(v) for v in axis)
                            and abs(sum(v*v for v in axis)-1)<1e-5, "approach_axis must be a unit vector in the interface frame pointing toward the operating side")
        from .operation_interface import validate_operation
        for interface in value["interfaces"]:
            validate_operation(interface, model, {i["id"] for i in value["interfaces"]})
        events = {i["id"] for i in value["interfaces"] if i["kind"] == "software"}
        for button in value["buttons"]:
            joint = model.joint(button["joint"])
            geom = model.geom(button["geom"])
            require(joint.type[0] == mujoco.mjtJoint.mjJNT_SLIDE and joint.limited[0], "A button needs a limited slide joint")
            require(geom.bodyid[0] == joint.bodyid[0], "Button contact face and joint do not belong to the same part")
            require(finite(button["pressed_at"]) and finite(button["released_at"])
                    and joint.range[0] <= button["released_at"] < button["pressed_at"] <= joint.range[1], "Invalid button trigger range")
            require(isinstance(button["event"], str) and button["event"] and button["event"] not in events, "Button event duplicated or invalid")
            events.add(button["event"])
        mechanical = value.get("mechanical_events", [])
        require(isinstance(mechanical,list) and len(mechanical)<=2048, "Mechanical events must be a bounded list")
        by_id = {i["id"]:i for i in value["interfaces"]}
        for event in mechanical:
            require(isinstance(event,dict) and set(event) in ({"id","interface","action"},{"id","interface","action","joint","at"}),
                    "Mechanical event fields must be id/interface/action, optionally with both a trigger joint (joint) and a threshold (at)")
            if "joint" in event:
                trigger = model.joint(event["joint"])
                require(trigger.type[0] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and trigger.limited[0]
                        and finite(event["at"]) and trigger.range[0] <= event["at"] <= trigger.range[1],
                        "Mechanical event trigger joint must be a limited hinge/slide, and threshold at must lie within its travel")
            require(isinstance(event["id"],str) and event["id"] and event["id"] not in events, "Mechanical event id duplicated or invalid")
            interface = by_id.get(event["interface"]) if isinstance(event["interface"],str) else None
            require(interface is not None and interface["kind"] in {"grasp","press","rotate","insert","cut"},
                    "Mechanical events must reference a real operating interface, not an assembly point or software interface as a placeholder")
            require(event["action"] in {"release","attach"}, "Mechanical event action must be release or attach")
            events.add(event["id"])
        mechanical_actions = {e["id"]:e["action"] for e in mechanical}
        def under(body, root):
            while body and body != root:
                body = int(model.body_parentid[body])
            return body == root

        def attach_contract(connection, sides, kind):
            """Shared re-attachment contract; sides are the two units the sites must belong to."""
            require(not connection["irreversible"] and connection["attach_event"] in events
                    and connection["attach_event"] != connection["release_event"], "An irreversible connection cannot re-engage; the attach event must be declared separately")
            sites = connection.get("capture_sites")
            require(isinstance(sites,list) and len(sites) == 2 and len(set(sites)) == 2,"Re-engagement requires two distinct locating sites")
            require(finite(connection.get("position_tolerance_m")) and 0 < connection["position_tolerance_m"] <= .005
                    and finite(connection.get("angle_tolerance_rad")) and 0 < connection["angle_tolerance_rad"] <= .2,
                    "Invalid re-engagement capture tolerance")
            site_ids = [model.site(n).id for n in sites]
            if kind == "latch":
                # One site on the lid, the other outside the lid: a lock pair does not require the two sites to be in different kinematic trees.
                lid, base = sides
                require(under(int(model.site_bodyid[site_ids[0]]),lid) != under(int(model.site_bodyid[site_ids[1]]),lid),
                        "For a hinge lock, one of the two engagement locating sites must belong to the locked part and the other to its supporting side; they cannot both be on the lid side")
            else:
                for sid,expected_body in zip(site_ids,sides):
                    require(under(int(model.site_bodyid[sid]),expected_body),"Engagement locating sites must belong to the two ends of the constraint respectively")
            data = mujoco.MjData(model)
            from .static_state import geometry_state
            geometry_state(model,data)
            require(np.linalg.norm(data.site_xpos[site_ids[0]]-data.site_xpos[site_ids[1]]) <= 1e-6
                    and np.allclose(data.site_xmat[site_ids[0]],data.site_xmat[site_ids[1]],atol=1e-6),
                    "Locating sites must coincide in the default locked pose, to avoid activating a wrong reference pose")

        for connection in value["connections"]:
            require(mechanical_actions.get(connection["release_event"],"release")=="release"
                    and mechanical_actions.get(connection.get("attach_event"),"attach")=="attach", "Mechanical event action does not match the connection's use")
            eq = model.equality(connection["equality"])
            kind = connection.get("kind", "detachable")
            require(kind in {"detachable", "latch"},
                    f"Connection {connection['id']} kind must be detachable (removable as a whole) or latch (hinge-locked); "
                    "use latch for parts such as lids that are always hinged and can only swing open once unlocked; use detachable for parts that can be removed as a whole")
            require(connection["release_event"] in events and type(connection["irreversible"]) is bool,
                    f"Connection {connection['id']}: release_event or irreversible is invalid")
            require(eq.active0[0], f"Connection {connection['id']}: constraint must be initially active for the locked state to take effect")
            if kind == "latch":
                require(eq.type[0] == mujoco.mjtEq.mjEQ_JOINT,
                        f"Connection {connection['id']} is declared latch (still hinge-connected after unlocking), "
                        f"equality {connection['equality']!r} must be a joint-type constraint that locks the hinge angle; currently {int(eq.type[0])}. "
                        "If the part really should be removed as a whole, change to kind=detachable and use a weld constraint")
                joint = int(eq.obj1id[0])
                require(0 <= joint < model.njnt, f"Connection {connection['id']}: joint constraint does not reference an actual joint")
                require(model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE),
                        f"Connection {connection['id']}: locked joint {model.joint(joint).name!r} must be hinge or slide; "
                        "ball/free joints cannot express hinge motion after locking")
                lid = int(model.jnt_bodyid[joint])
                require(lid > 0, f"Connection {connection['id']}: locked joint must not be on the world root")
                require(not connection["irreversible"],
                        f"Connection {connection['id']} is a hinge lock and must be able to lock again; irreversible applies only to detachable connections")
                if "attach_event" in connection:
                    attach_contract(connection, [lid, int(model.body_parentid[lid])], "latch")
            else:
                require(eq.type[0] == mujoco.mjtEq.mjEQ_WELD,
                        f"Connection {connection['id']} is declared detachable (removable as a whole), "
                        f"equality {connection['equality']!r} must be a weld constraint; currently {int(eq.type[0])}. "
                        "If the part is only locked and always hinged to the main body, change to kind=latch and use a joint constraint")
                bodies = [int(eq.obj1id[0]), int(eq.obj2id[0])]
                roots = []
                for body in bodies:
                    while model.body_parentid[body] != 0:
                        body = int(model.body_parentid[body])
                    roots.append(body)
                require(roots[0] != roots[1],
                        f"Connection {connection['id']} is declared detachable, but both constraint ends still belong to the same permanent kinematic tree (root part {roots[0]}); "
                        "a detachable part must be an independent motion unit from the start. Parts such as lids that are always hinged should use kind=latch")
                require(any(model.body_jntnum[b] and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
                            for b in roots if b),
                        f"Connection {connection['id']} is declared detachable, but neither end can move freely, so it cannot move independently once detached")
                if "attach_event" in connection:
                    attach_contract(connection, bodies, "detachable")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError("Interaction manifest is missing fields or references something that does not exist") from exc
    return value


def extent(model, data, names, axis):
    """Measure the world-axis envelope of the given geoms in the default pose; never use dimensions self-reported by the model."""
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
            raise ValidationError("This dimension check does not yet support the given geom type")
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
                "A single asset must not generate a global ground or heightfield; use finite-size geometry")
        require(all(model.body(b).name for b in range(1,model.nbody)), "All physical parts of the asset must be named so assembly and mass can be tracked")
        require(any(model.geom_contype[g] or model.geom_conaffinity[g] for g in range(model.ngeom)),
                "The asset must contain valid collision geometry, not just a visual shell")
        require(all(model.geom(g).name and model.geom_bodyid[g] > 0 for g in range(model.ngeom)
                    if model.geom_contype[g] or model.geom_conaffinity[g]),
                "Asset collision geoms must be named and belong to a physical part, not the world root")
        manifest = validate_manifest(manifest_value if manifest_value is not None else read_json(Path(root)/"interfaces.json"), model, asset_ids)
        add("compile_and_interfaces", "pass", "Independently compiled and resolved the actual references")
    except (ValidationError, ValueError, KeyError, ParseError, DefusedXmlException) as exc:
        add("compile_and_interfaces", "fail", str(exc)[:2000])
        return {"status": "fail", "checks": results, "renders": [], "manifest": None,
                "coverage": ["Other checks not run because of compile or interface errors"]}
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
                "Actual geometry envelope in the default pose", actual_m=actual, expected_m=item["expected_m"], tolerance_m=item["tolerance_m"])
        except (KeyError, ValidationError) as exc:
            add(item["id"], "fail", str(exc))
    interfaces = {i["id"] for i in manifest["interfaces"]}
    for name in profile["required_interfaces"]:
        add("interface:"+name, "pass" if name in interfaces else "fail", "Presence of interfaces from the read-only requirements")
    kinds = {"hinge": mujoco.mjtJoint.mjJNT_HINGE, "slide": mujoco.mjtJoint.mjJNT_SLIDE,
             "ball": mujoco.mjtJoint.mjJNT_BALL, "free": mujoco.mjtJoint.mjJNT_FREE}
    for item in profile["joints"]:
        try:
            j = model.joint(item["name"])
            valid = j.type[0] == kinds[item["type"]] and np.allclose(j.range, item["range"], rtol=0, atol=item["tolerance"])
            add("joint:"+item["name"], "pass" if valid else "fail", "Actual joint type and limits", actual_range=j.range.tolist())
        except KeyError:
            add("joint:"+item["name"], "fail", "Specified joint missing")
    contacts = [{"geom1": model.geom(int(c.geom1)).name, "geom2": model.geom(int(c.geom2)).name,
                 "distance_m": float(c.dist)} for c in data.contact if c.dist < -profile["penetration_tolerance_m"]]
    add("initial_contacts", "fail" if contacts else "pass", "Initial contact penetration under default collision filtering", contacts=contacts[:40])
    poses = [("initial", data.qpos.copy())]
    # Motion is part of finishing an asset, in every scene validation mode.
    from . import motion
    qpos0 = data.qpos.copy()
    results.append(motion.mechanism_contract(model, manifest))
    sweep_rows, mechanism_poses = motion.mechanism_sweep(model, manifest, qpos0, profile["penetration_tolerance_m"])
    uncovered = [model.joint(j).name for j in motion.movable(model)
                 if not any(model.joint(j).name in m.get("joints", []) for m in motion.declared(manifest))]
    results.extend(sweep_rows)
    if uncovered or not sweep_rows:
        add("sampled_joint_sweep", "fail" if uncovered else "pass",
            "Joints not assigned to a mechanism cannot be swept per mechanism" if uncovered else "Asset has no movable joints", unswept_joints=uncovered[:40])
    poses.extend(mechanism_poses[:8])
    results.extend(motion.mechanism_retention(model, manifest, qpos0))
    # Settle first: every later check simulates, and a diverging model only burns the time
    # budget there (MuJoCo resets and keeps stepping) before the whole recheck times out.
    stability = motion.settle(safe_path(Path(root), model_file), qpos0)
    results.append(stability)
    unstable = stability.get("unstable", False)
    stops = []
    if not unstable:
        results.extend(motion.mechanism_drive(model, manifest, qpos0, stops))
    # Where a drive stopped short, the author and reviewer see that pose, blockers highlighted.
    poses.extend(stops[:4])
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    renders = []
    if render_dir is not None:
        try:
            renders = render_views(model, poses, Path(render_dir))
            add("render", "pass", "Initial pose from multiple views and both end poses of each mechanism")
        except Exception as exc:
            add("render", "error", type(exc).__name__ + ": cannot generate review images")
    coverage = ["Not automatically proven: no overlap between static geometry, stable support, fixture fit, or executability of the robot task",
                "Joint checks are independent pose samples; button mechanics, connection release and compound mechanisms need dedicated operation probes",
                "Dimensions in the materials not bound to a read-only numeric check are still not independently measured or certified"]
    from .probes import interaction_probes
    if unstable:
        add("dynamic_checks", "fail", "Mechanism drive, button, release and lock probes were not run: the model is "
            "numerically unstable (see numerical_stability)", repair_hint=stability["repair_hint"])
    try:
        probe_rows = [] if unstable else interaction_probes(root, manifest, model_file) if manifest["buttons"] or manifest["connections"] else []
        probe_rows = [r for r in probe_rows if not (r["id"].startswith("release_probe:") and r.get("complete") is False)]
        if not unstable:
            probe_rows.extend(motion.release_probes(model, manifest, qpos0))
        add("interaction_probes", "fail" if any(r["status"] == "fail" for r in probe_rows) else "pass",
            "Summary of button presser, software release and mechanical release probes; cross-asset insertion/removal and grasping are verified in the trajectory stage",
            probes=[r["id"] for r in probe_rows])
        results.extend(probe_rows)
    except Exception as exc:
        add("interaction_probes", "error", str(exc)[:2000], error_kind='check_execution')
    coverage[1] = "Within the single asset, mechanism sweep, drive, locking and release have been probed; cross-asset insertion/removal, grasping and robot paths are verified in the trajectory stage"
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
            "Render size outside the supported range")
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,width)
    model.vis.global_.offheight = max(model.vis.global_.offheight,height)
    # The framework provides basic lighting so generated code that forgets lights cannot produce unreviewable all-black images.
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    from .rendering import renderer as make_renderer
    with make_renderer(model, height=height, width=width) as renderer:
        for index, (label, qpos, *rest) in enumerate(poses):
            data.qpos[:] = qpos
            from .static_state import geometry_state
            geometry_state(model, data)
            # Optional third element: geoms to paint magenta (contacts that blocked a drive).
            highlight = sorted(rest[0]) if rest else []
            saved = model.geom_rgba[highlight].copy(), model.geom_matid[highlight].copy()
            model.geom_rgba[highlight], model.geom_matid[highlight] = [1, 0, 1, 1], -1
            for azimuth in ([45, 135, 225] if index == 0 else [45]):
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, camera)
                camera.azimuth = azimuth
                camera.elevation = -25
                renderer.update_scene(data, camera)
                name = f"{label}_{azimuth}.png"
                Image.fromarray(renderer.render()).save(output/name)
                paths.append(name)
            model.geom_rgba[highlight], model.geom_matid[highlight] = saved
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
