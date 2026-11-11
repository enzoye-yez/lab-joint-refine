"""Independent whole-scene checks and operating-pose probing; discrete reachability is never called a complete trajectory."""

from pathlib import Path
import numpy as np
import mujoco

from asset_generation.checks import render_views
from asset_generation.contracts import require, read_json
from .geometry import aggregate_bounds, collisions, collision_scan, owner, site_surface_distance, bounds
from .table import POLICY as TABLE_EDIT_POLICY, validate_cuts
from asset_generation.distance import DistanceQuery, MeasurementError, observe
from .recovery import report_status
from .edge_mounts import boundary_evidence, validate_mounts
from .layout import top_prefix, table_for, table_geometry


SUPPORT_OVERHANG_M = .03


def support_fit(item, bounds, support_bounds):
    """An object resting on another instance has to fit on it in its home pose. A part much
    wider than its support (a flat 203x254 mm filter on a 54x68 mm folded foil) is in the
    wrong initial state: assembly never folds, loads or moves anything."""
    span = np.asarray(bounds[1][:2])-np.asarray(bounds[0][:2])
    base = np.asarray(support_bounds[1][:2])-np.asarray(support_bounds[0][:2])
    excess = span-base
    allowed = np.maximum(SUPPORT_OVERHANG_M, .5*base)
    ok = bool(np.all(excess <= allowed))
    row = {"id":"support_fit:"+item["id"],"status":"pass" if ok else "fail",
           "detail":"Home-pose footprint of an instance resting on another instance against the footprint of that support",
           "instances":[item["id"],item["support_instance"]],
           "footprint_mm":[round(float(v)*1000,1) for v in span],"support_footprint_mm":[round(float(v)*1000,1) for v in base]}
    if not ok:
        row["repair_hint"] = (f"{item['id']} spans {row['footprint_mm'][0]:.0f} x {row['footprint_mm'][1]:.0f} mm at home but its "
            f"support {item['support_instance']} is {row['support_footprint_mm'][0]:.0f} x {row['support_footprint_mm'][1]:.0f} mm. "
            "The model's default pose (every joint at 0) is the scene's initial state; build the part in the state the "
            "protocol starts in (folded, loaded, closed) instead of expecting scene assembly to change it.")
    return row


def representative(hits, limit=100):
    """The deepest hit of every instance pair first, then the rest by depth: a cut at limit
    keeps every colliding pair visible (a plain cut hid whole pairs behind one large one)."""
    depth=lambda h:h.get('distance_m') if type(h.get('distance_m')) in (int,float) else 0.
    ordered=sorted(hits,key=depth)
    first,rest,seen=[],[],set()
    for hit in ordered:
        pair=tuple(sorted(hit.get('instances',[])))
        (rest if pair in seen else first).append(hit);seen.add(pair)
    return (first+rest)[:limit]


def upward_resultant(contacts):
    """Whether the normal cones of several frictionless contacts can form a vertical resultant force; does not certify torque or stability."""
    normals = np.array([c['normal'] for c in contacts])
    if not len(normals):
        return False
    horizontal = np.linalg.norm(normals[:,:2],axis=1)
    if np.any(horizontal <= 1e-7):
        return True
    angles = np.sort(np.arctan2(normals[:,1],normals[:,0]))
    largest_gap = np.diff(np.r_[angles,angles[0]+2*np.pi]).max()
    return bool(largest_gap <= np.pi+1e-7)


def support_evidence(model, data, item, tolerance, collision_tolerance, geometry, query=None,
                     object_geoms=None, support_geoms=None, max_pairs=200000):
    from asset_generation.material import surface_batch
    query = query or DistanceQuery(model,data)
    with surface_batch(model,data,query):
        return _support_evidence(model,data,item,tolerance,collision_tolerance,geometry,query,
                                 object_geoms,support_geoms,max_pairs)


def _support_evidence(model, data, item, tolerance, collision_tolerance, geometry, query=None,
                     object_geoms=None, support_geoms=None, max_pairs=200000):
    """Find actual upward-facing support contacts, allowing flange/lip support; side-wall proximity is not support."""
    own = [g for g in geometry if owner(model,g) == item["id"]]
    query = query or DistanceQuery(model,data)
    parent = item.get("support_instance")
    if parent:
        parents = [g for g in geometry if owner(model,g) == parent]
    elif item["placement"] == "table":
        parents = [g for g in geometry if (model.geom(g).name or "").startswith(top_prefix(item))]
    else:
        parents = [model.geom("environment__floor").id]
    for selection, ids in ((object_geoms,own),(support_geoms,parents)):
        if selection is not None:
            require(set(selection) <= {model.geom(g).name for g in ids}, "Support-check geoms fall outside the specified instance or the real supporting object")
    if object_geoms is not None:
        own = [g for g in own if model.geom(g).name in object_geoms]
    if support_geoms is not None:
        parents = [g for g in parents if model.geom(g).name in support_geoms]
    contacts, nearby, errors, examined, nearest = [], [], [], 0, None
    error_count = skipped = 0
    uncertain_objects = set()
    from .geometry import support_candidates
    for g1,g2 in support_candidates(geometry,parents,own,tolerance):
        if examined >= max_pairs:
            skipped += 1
            uncertain_objects.add(g2)
            continue
        examined += 1
        evidence = observe(query,g1,g2,tolerance+.001)
        if evidence['outcome'] == 'unknown':
            error_count += 1
            uncertain_objects.add(g2)
            errors.append({**evidence,"geom1":model.geom(g1).name,"geom2":model.geom(g2).name,
                           "instances":[item['id'],parent or 'environment']})
            continue
        distance = evidence["distance_m"]
        if distance is None:
            if evidence.get('lower_bound_m') is None or evidence['lower_bound_m'] <= tolerance:
                error_count += 1
                uncertain_objects.add(g2)
                errors.append({**evidence,'outcome':'unknown',
                    'geom1':model.geom(g1).name,'geom2':model.geom(g2).name,
                    'instances':[item['id'],parent or 'environment'],
                    'error':'Distance bounds cannot prove a support contact or rule out a support candidate'})
            continue
        nearest = distance if nearest is None else min(nearest,distance)
        if not -collision_tolerance <= distance <= tolerance:
            continue
        upward = evidence["normal"] is not None and evidence["normal"][2] > 1e-7
        if upward and distance <= min(tolerance,1e-5):
            from asset_generation.material import visible_witness
            points=evidence.get("points")
            witnesses=([visible_witness(model,data,query,g1,np.array(points[:3])),
                        visible_witness(model,data,query,g2,np.array(points[3:]))] if points is not None else [None,None])
            if any(w is None for w in witnesses):
                from asset_generation.surface_contact import contact
                replacement=contact(query,g1,g2,evidence.get("normal"))
                if replacement is not None and replacement["normal"][2]>1e-7:
                    points=replacement["points"]
                    candidates=[visible_witness(model,data,query,g1,np.array(points[:3])),
                                visible_witness(model,data,query,g2,np.array(points[3:]))]
                    if all(candidates):
                        evidence=replacement;distance=replacement["distance_m"];witnesses=candidates
            if any(w is None for w in witnesses):
                error_count+=1;uncertain_objects.add(g2)
                errors.append({"geom1":model.geom(g1).name,"geom2":model.geom(g2).name,
                    "instances":[item["id"]],"error_kind":"unsupported_check",
                    "detail":"collision support witness is not certified on displayed material"})
                continue
            evidence={**evidence,"visible_witnesses":witnesses}
        if upward:
            row = {**evidence,"support_geom":model.geom(g1).name,"object_geom":model.geom(g2).name}
            (contacts if distance <= min(tolerance,1e-5) else nearby).append(row)
    contacts.sort(key=lambda c:(abs(c['distance_m']),-c['normal'][2],c['object_geom']))
    nearby.sort(key=lambda c:abs(c['distance_m']))
    if skipped:
        errors.append({"error_kind":"resource_limit","unexamined_pairs":skipped,
                       "instances":[item['id'],parent or 'environment']})
    supported = upward_resultant(contacts)
    components = {'component_scope':'selected_geoms'}
    if object_geoms is None and support_geoms is None:
        from .support import component_support
        components = component_support(model,data,item,geometry,contacts,query,tolerance,collision_tolerance,
                                       max(0,max_pairs-examined),uncertain_objects,upward_resultant)
        supported = components['all_components_supported']
        error_count += components['component_error_count']
        errors.extend(components['component_errors'])
        skipped += components['component_unexamined_pairs']
        if components['component_unexamined_pairs']:
            errors.append({'error_kind':'resource_limit','unexamined_pairs':components['component_unexamined_pairs'],
                           'instances':[item['id']],'detail':'Support contacts between independent components have not been fully checked'})
    return {"support_tolerance_m":tolerance,"collision_tolerance_m":collision_tolerance,
            "supported":supported,"complete":not error_count and not skipped,
            "gap_m":min((c["distance_m"] for c in contacts),default=nearest),
            "candidate_pairs":examined,"support_contacts":contacts[:12],"contact_count":len(contacts),
            "nearby_noncontacts":nearby[:6],"measurement_errors":errors,"measurement_error_count":error_count,
            "unexamined_pairs":skipped,"contact_epsilon_m":min(tolerance,1e-5),
            "vertical_resultant_available":upward_resultant(contacts),
            **components,
            "reason":"Every independent component has support or an actual connection chain; connection manufacturability/fit and torque balance not checked" if supported and components.get('component_scope')=='whole_instance' else
                     "Some independent components have no verified support chain; see component_support" if components.get('component_scope')=='whole_instance' and contacts else
                     "Contact normals can form an upward vertical resultant force; torque balance not checked" if supported else
                     "Inclined contacts exist but their normals cannot be shown to form a vertical resultant force" if contacts else
                     "No nearby support geometry found" if not examined else "Nearby geometry has no support contact satisfying the gap and upward-direction criteria"}


def operating_azimuth(model, data, prefix):
    """Free-camera azimuth (degrees) that looks at an instance's operating face, its root body's local -y."""
    roots = [b for b in range(1,model.nbody) if owner_of_body(model,b) == prefix and owner_of_body(model,int(model.body_parentid[b])) != prefix]
    if not roots:
        return 45.0
    face = -data.xmat[roots[0]].reshape(3,3)[:,1]
    if np.hypot(face[0],face[1]) < 1e-6:
        return 45.0
    # The camera sits on the face side, a quarter turn off square so depth stays readable.
    return float(np.degrees(np.arctan2(-face[1],-face[0]))) + 25.0


def owner_of_body(model, body):
    while body:
        name = model.body(int(body)).name
        if name and "__" in name:
            return name.split("__",1)[0]
        body = int(model.body_parentid[body])
    return "unassigned"


def segment_hits_box(start, end, lo, hi):
    """Slab test: does the segment start->end pass through the axis-aligned box [lo, hi]?"""
    direction = end - start
    near, far = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if not lo[axis] <= start[axis] <= hi[axis]:
                return False
            continue
        a, b = sorted(((lo[axis]-start[axis])/direction[axis], (hi[axis]-start[axis])/direction[axis]))
        near, far = max(near,a), min(far,b)
        if near > far:
            return False
    return True


def fade_occluders(model, data, scene, prefix, eye, lo, hi, alpha=.12):
    """Make other instances' and the room's geometry between the camera and the target nearly transparent."""
    from .geometry import owner, bounds
    center = (lo+hi)/2
    targets = [center] + [center + .5*(np.array(c)-center) for c in np.stack(np.meshgrid(*zip(lo,hi)),-1).reshape(-1,3)]
    for g in range(scene.ngeom):
        geom = scene.geoms[g]
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or owner(model,geom.objid) == prefix \
                or model.geom_type[geom.objid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        box = bounds(model,data,geom.objid)
        if any(segment_hits_box(eye,t,box[0],box[1]) for t in targets):
            geom.rgba[3] = min(geom.rgba[3],alpha)


def render_details(model, home, plan, output):
    """Close-up of an instance in the current whole scene, seen from its operating face with occluding geometry faded;
    does not change pose or advance physics time."""
    from PIL import Image
    output = Path(output)
    data = mujoco.MjData(model)
    data.qpos[:] = home
    from asset_generation.static_state import geometry_state
    geometry_state(model,data)
    paths, seen = [], set()
    from asset_generation.rendering import renderer as make_renderer
    with make_renderer(model,height=480,width=640) as renderer:
        for item in plan["instances"]:
            variant=(item["job"],item.get("configuration"))
            if variant in seen:
                continue
            seen.add(variant)
            lo,hi = aggregate_bounds(model,data,item["id"])
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model,camera)
            camera.lookat[:] = (lo+hi)/2
            camera.distance = max(.15,float(np.linalg.norm(hi-lo))*1.8)
            camera.azimuth, camera.elevation = operating_azimuth(model,data,item["id"]),-30
            renderer.update_scene(data,camera)
            az, el = np.radians(camera.azimuth), np.radians(camera.elevation)
            forward = np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
            fade_occluders(model,data,renderer.scene,item["id"],camera.lookat-camera.distance*forward,lo,hi)
            name = "detail_"+item["id"]+".png"
            Image.fromarray(renderer.render()).save(output/name)
            paths.append(name)
    return paths


def pose_error(data, sid, position, rotation):
    current = data.site_xmat[sid].reshape(3,3)
    quat, angle = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(quat, (rotation @ current.T).reshape(-1))
    if quat[0] < 0:
        quat *= -1
    mujoco.mju_quat2Vel(angle, quat, 1)
    return np.r_[position-data.site_xpos[sid], angle]


def reach_target(model, home, env, target, limits, recovery=False):
    from .tool_contract import has_free_parts
    if has_free_parts(model, target["tool_site"].split("__")[0]+"__"):
        return {"status":"partial", "target":target,
                "coverage_gaps":[{"error_kind":"unsupported_check", "detail":"A detachable end-effector requires a constraint-coupled kinematic solve"}],
                "detail":"Current inverse kinematics solves only the permanent kinematic tree and cannot certify operating poses involving free detachable parts"}
    data = mujoco.MjData(model)
    data.qpos[:] = home
    mujoco.mj_forward(model,data)
    site = model.site(target["site"]).id
    tcp = model.site(target["tool_site"]).id
    # Only ancestors of the selected TCP can contribute. Never move the other arm,
    # wheels or head to solve one hand's pose.
    ancestors=set()
    body=int(model.site_bodyid[tcp])
    while body:
        ancestors.add(body); body=int(model.body_parentid[body])
    robot_joints=[j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in ancestors
                  and (model.joint(j).name or "").removeprefix("robot__") in env["robot"]["joints"]
                  and model.jnt_limited[j]]
    if not robot_joints:
        return {"status":"partial","target":target,"coverage_gaps":[{"error_kind":"unsupported_check",
                "detail":"The target tool center has no solvable robot joint chain"}],"detail":"The target tool center has no solvable robot joint chain"}
    columns, addresses = model.jnt_dofadr[robot_joints], model.jnt_qposadr[robot_joints]
    destination = data.site_xpos[site] + data.site_xmat[site].reshape(3,3) @ np.array(target["offset"])
    local_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(local_rotation,np.array(target["quaternion"]))
    rotation = data.site_xmat[site].reshape(3,3) @ local_rotation.reshape(3,3)
    rng = np.random.default_rng(0)
    best = None
    unknown = []
    for restart in range(limits.ik_restarts):
        data.qpos[:] = home
        if restart:
            data.qpos[addresses] = rng.uniform(model.jnt_range[robot_joints,0], model.jnt_range[robot_joints,1])
        for _ in range(limits.ik_iterations):
            mujoco.mj_forward(model,data)
            error = pose_error(data,tcp,destination,rotation)
            pe, ae = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
            if best is None or pe + .1*ae < best[0]:
                best = (pe+.1*ae,pe,ae)
            if pe <= target["position_tolerance_m"] and ae <= target["angle_tolerance_rad"]:
                scan = collision_scan(model,data,limits.collision_tolerance_m,target['allowed_contacts'],
                                      query=DistanceQuery(model,data,recovery=recovery))
                if not scan['collisions'] and scan['complete']:
                    return {"status":"pass", "position_error_m":pe, "angle_error_rad":ae,
                            "qpos":data.qpos.tolist(), "target":target, "detail":"Found a collision-free discrete operating pose; approach/retreat path not verified"}
                unknown.extend(scan['coverage_gaps'][:8])
                break
            jacp, jacr = np.zeros((3,model.nv)), np.zeros((3,model.nv))
            mujoco.mj_jacSite(model,data,jacp,jacr,tcp)
            jac = np.vstack([jacp[:,columns], .25*jacr[:,columns]])
            weighted = np.r_[error[:3], .25*error[3:]]
            step = jac.T @ np.linalg.solve(jac @ jac.T + .002*np.eye(6),weighted)
            data.qpos[addresses] = np.clip(data.qpos[addresses]+np.clip(step,-.15,.15),
                                            model.jnt_range[robot_joints,0], model.jnt_range[robot_joints,1])
    return {"status":"partial" if unknown else "fail", "position_error_m":best[1], "angle_error_rad":best[2], "target":target,
            "coverage_gaps":unknown[:32],
            "detail":"Bounded inverse-kinematics search found no usable pose; this does not prove mathematical unreachability"}


def _body_chain(model, body):
    """Yield a body and all its ancestors, so a joint's moving subtree can be found."""
    while body:
        yield body
        body = int(model.body_parentid[body])


def scene_checks(root, limits, contract, render_dir=None, recovery=False, checkpoint_dir=None):
    from asset_generation.check_report import validate_report
    from asset_generation.checks import measure_model, scale_limits
    static = limits.validation_mode == "static"
    from asset_generation.timing import stage
    from .checkpoints import Checkpoints
    checkpoints = Checkpoints(checkpoint_dir,root,limits,contract,recovery)
    with stage("scene_load"):
        model, scale = checkpoints.load_model(lambda: measure_model(root, scope="scene", static=static))
    budgets = scale_limits(model, "scene")
    meta = read_json(Path(root)/"scene.json", 8000000)
    manifest = read_json(Path(root)/"interfaces.json", 8000000)
    plan, env = meta["plan"], meta["environment"]
    data = mujoco.MjData(model)
    data.qpos[:] = meta["home_qpos"]
    data.ctrl[:] = meta["home_ctrl"]
    from asset_generation.static_state import geometry_state
    if static:
        geometry_state(model,data,contacts=True)
    else:
        mujoco.mj_forward(model,data)
    home = data.qpos.copy()
    rows = []
    def add(kind, ok, detail, instances=(), **evidence):
        rows.append({"id":kind,"status":"pass" if ok else "fail", "detail":detail,
                     "instances":list(instances), **evidence})
    query = DistanceQuery(model,data,recovery=recovery)
    # Budgets scale with the actual model instead of one fixed number, so a large scene
    # degrades in coverage (reported) rather than timing out without evidence.
    pair_budget = budgets["pair_budget"] * (4 if recovery else 1)
    scan = checkpoints.run("scene_collisions",lambda: collision_scan(model,data,
        limits.collision_tolerance_m,query=query,max_pairs=pair_budget),query)
    hits, gaps = scan['collisions'], scan['coverage_gaps']
    add("scene_collisions",not hits,"Actual cross-instance geometric distances and robot self-contact",
        sorted({i for h in hits for i in h['instances']}),collisions=representative(hits),
        collision_count=len(hits),coverage_gaps=gaps,examined_pairs=scan['examined_pairs'],
        complete=scan['complete'],unexamined_pairs=scan['unexamined_pairs'],
        scale=scale,budget=pair_budget)
    if gaps and not hits:
        rows[-1]['status']='partial'
    geometry = {g:bounds(model,data,g) for g in range(model.ngeom)
                if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE and (model.geom_contype[g] or model.geom_conaffinity[g])}
    from asset_generation.material import integrity
    for item in plan["instances"]:
        rows.append(checkpoints.run("material:"+item["id"],lambda: integrity(model,data,
            instance=item["id"],tolerance=limits.collision_tolerance_m,max_pairs=pair_budget,query=query),query))
    cuts = validate_cuts(plan.get("table_modifications", []), plan, env)
    mounts = validate_mounts(plan.get('edge_mounts',[]),plan)
    local_geometry = {t["id"]:table_geometry(model,data,geometry,t) for t in
                      plan.get("layout",{}).get("tables",[])}
    for cut in cuts:
        lo,hi = aggregate_bounds(model,data,cut["instance"])
        if "layout" in plan:
            item = next(i for i in plan["instances"] if i["id"]==cut["instance"])
            boxes = [b for g,b in local_geometry[item["table_id"]].items() if owner(model,g)==item["id"]]
            lo,hi = np.min([b[0] for b in boxes],axis=0),np.max([b[1] for b in boxes],axis=0)
        center,half = np.array(cut["center"]),np.array(cut["size"])/2
        margin = TABLE_EDIT_POLICY["max_instance_margin_m"]
        add("table_modification:"+cut["id"],bool(np.all(center-half >= lo[:2]-margin) and np.all(center+half <= hi[:2]+margin)),
            "Machined region lies near the associated instance envelope; material load-bearing strength not certified",[cut["instance"]],cut=cut)
    for item in plan["instances"]:
        if item["placement"] == "robot":
            continue
        lo, hi = aggregate_bounds(model,data,item["id"])
        support = item.get("support_instance")
        evidence = checkpoints.run("support:"+item["id"],lambda: support_evidence(model,data,item,
            limits.support_tolerance_m,limits.collision_tolerance_m,geometry,query,max_pairs=pair_budget),query)
        add("support:"+item["id"], evidence.pop("supported"),
            "Per-independent-component support contacts and actual kinematic-tree/enabled-weld connection chains; does not prove connection manufacturability/fit, center-of-mass stability or load-bearing strength",
            [item["id"]]+([support] if support else []), **evidence)
        if not evidence['complete']:
            rows[-1].update(status="fail" if evidence.get('unsupported_components') else "partial",
                            coverage_gaps=evidence['measurement_errors'])
        if support:
            rows.append(support_fit(item, (lo, hi), aggregate_bounds(model,data,support)))
        if item["placement"] == "table":
            boundary = boundary_evidence(model,local_geometry.get(item.get("table_id"),geometry),
                                         item,{**env,"table":table_for(item,plan,env)},mounts)
            add("table_boundary:"+item["id"],boundary.pop('valid'),
                "Ordinary parts lie within the tabletop; declared table-edge clamp parts are limited to the mounting zone and are not exempt from collision and support",[item['id']],
                bounds=[lo.tolist(),hi.tolist()],**boundary)
    from .robot_contract import arms, validate_bindings
    selected=validate_bindings(plan,env,partial=True)
    tools = [i for i in plan["instances"] if i["placement"] == "robot"]
    if "arms" not in env["robot"]:
        require(len(tools)==1 or static and not tools and meta.get("partial_without_tool") is True,
                "A legacy scene must have exactly one mounted end-effector; missing parts require an explicit partial-assembly record")
    tool = tools[0] if tools else None
    entries=arms(env)
    if "arms" in env["robot"]:
        from .inspection import mount_contact
        for tool_item in tools:
            rows.append(mount_contact(model,data,meta,tool_item["id"],
                                      limits.support_tolerance_m,limits.collision_tolerance_m,query))
    require(set(contract.get("arm_payloads",{})) <= {a["id"] for a in entries}, "Workload references an unknown robot arm")
    for arm in entries:
        binding=selected[arm["id"]]
        target=next((i for i in tools if i["id"]==binding["instance"]),None)
        kind="payload:"+arm["id"] if "arms" in env["robot"] else "payload"
        workload=contract.get("arm_payloads",{}).get(arm["id"],contract["payload_mass_kg"])
        if len(entries)>1 and contract["payload_mass_kg"] and arm["id"] not in contract.get("arm_payloads",{}):
            rows.append({"id":kind,"status":"partial","instances":[],"detail":"Non-zero multi-arm workloads must be declared per arm; the total load is neither split nor treated as a single-arm load"})
            continue
        if binding["mode"]!="stock" and target is None:
            rows.append({"id":kind,"status":"partial","instances":[],"detail":"This arm's planned custom end-effector is missing; the full payload cannot be verified"})
            continue
        if binding["mode"]=="stock" and arm["stock"] is None:
            rows.append({"id":kind,"status":"partial","instances":[],"detail":"End-effector not yet assembled; payload cannot be verified"})
            continue
        mass=sum(float(model.body_mass[b]) for b in range(model.nbody)
                 if target and (model.body(b).name or "").startswith(target["id"]+"__"))
        stock_mass=arm["stock"]["mass_kg"] if arm["stock"] and binding["mode"]!="replace" else 0
        add(kind,mass+stock_mass+workload<=arm["payload_kg"],
            "Per arm: actual custom assembly mass + nominal mass of the retained stock gripper + the arm's declared workload",
            [target["id"]] if target else ["robot"],arm=arm["id"],tool_mass_kg=mass,
            stock_mass_kg=stock_mass,declared_payload_kg=workload,limit_kg=arm["payload_kg"])
    for fit in contract["fits"]:
        if static and "id" in fit and any(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,n) < 0
                          for end in ("outer","inner") for n in fit[end]):
            rows.append({"id":"fit:"+fit["id"],"status":"partial","instances":[],
                         "detail":"Partial scene lacks mating markers; original acceptance requirements retained"})
            continue
        widths = []
        surface_errors = {n:site_surface_distance(model,data,n) for end in ("outer","inner") for n in fit[end]}
        for end in ("outer","inner"):
            sites = [model.site(n).id for n in fit[end]]
            local_axis = data.site_xmat[sites[0]].reshape(3,3)[:,fit["axis"]]
            widths.append(abs(float((data.site_xpos[sites[1]]-data.site_xpos[sites[0]]) @ local_axis)))
        clearance = widths[0]-widths[1]
        add("fit:"+fit["id"], fit["clearance_m"][0] <= clearance <= fit["clearance_m"][1]
            and all(x <= .0005 for x in surface_errors.values()),
            "Marker spacing differences on actual collision surfaces; hole-wall semantics still require source and visual verification",
            sorted({n.split("__")[0] for key in ("outer","inner") for n in fit[key]}), clearance_m=clearance,
            surface_errors_m=surface_errors,evidence=fit["evidence"])
    # Sample each joint in the whole scene to catch instrument lids that hit neighbours when opened.
    sample_failures, sample_count, sample_gaps = [], 0, []
    from .tool_contract import has_free_parts
    coupled_tools = [i["id"]+"__" for i in tools if has_free_parts(model,i["id"]+"__")]
    for j in (range(model.njnt) if not static else ()):
        name = model.joint(j).name
        if name.startswith("robot__") or model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) or not model.jnt_limited[j]:
            continue
        if any(name.startswith(prefix) for prefix in coupled_tools):
            sample_gaps.append({"error_kind":"unsupported_check", "joint":name,
                                "detail":"Joint sweeps with a detachable end-effector require constraint coupling; moving only the permanent kinematic tree is not enough"})
            continue
        if sample_count+7 > budgets["joint_samples"]:
            sample_gaps.append({'error_kind':'resource_limit','joint':name,
                                'detail':'Joint-sweep sample budget exhausted (scaled down with model size; %d poses swept)' % sample_count})
            continue
        # Only the instances containing this joint can change contact during the sweep;
        # everything else is unchanged geometry, so rescanning it adds cost but no evidence.
        moving = {owner(model,g) for g in range(model.ngeom) if model.geom_bodyid[g] == int(model.jnt_bodyid[j])}
        moving |= {owner(model,g) for g in range(model.ngeom)
                   if any(ancestor == int(model.jnt_bodyid[j])
                          for ancestor in _body_chain(model, int(model.geom_bodyid[g])))}
        for q in np.linspace(*model.jnt_range[j],7):
            data.qpos[:] = home
            data.qpos[model.jnt_qposadr[j]] = q
            mujoco.mj_forward(model,data)
            sample_count += 1
            scan = collision_scan(model,data,limits.collision_tolerance_m,query=query,max_pairs=pair_budget,
                                  only_owners=moving)
            found = scan['collisions']
            if scan['coverage_gaps']:
                sample_gaps.append({'joint':name,'q':float(q),'coverage_gaps':scan['coverage_gaps'][:8]})
            if found:
                sample_failures.append({"joint":name,"q":float(q),"collisions":found[:8]})
    add("scene_joint_sweeps",not sample_failures,"Per-joint discrete open/close clearance check", sorted({i for f in sample_failures for h in f["collisions"] for i in h["instances"]}),samples=sample_count,failures=sample_failures[:40])
    if sample_gaps:
        rows[-1].update(coverage_gaps=sample_gaps[:40],status='fail' if sample_failures else 'partial')
    if static:
        rows[-1].update(status="skipped", detail="Static mode skips the joint open/close clearance check")
    data.qpos[:] = home
    if static:
        geometry_state(model,data)
    else:
        mujoco.mj_forward(model,data)
    initial = data.xpos.copy()
    for _ in range(0 if static else limits.settle_steps):
        mujoco.mj_step(model,data)
    motion = []
    for b in range(1,model.nbody):
        if model.body_jntnum[b] and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE:
            shift = float(np.linalg.norm(data.xpos[b]-initial[b]))
            if shift > .02:
                motion.append({"body":model.body(b).name,"displacement_m":shift})
    add("settling",not motion and np.isfinite(data.qpos).all() and not any(w.number for w in data.warning),
        "Finite-time check under gravity for free parts falling or moving noticeably", sorted({m["body"].split("__")[0] for m in motion}),moving_bodies=motion,seconds=float(data.time))
    if static:
        rows[-1].update(status="skipped", detail="Static mode skips dynamics stepping")
    targets = list(contract["targets"]) if not static else []
    automatic_targets = not targets and not static
    if automatic_targets:
        # Explicit multi-arm targets retain the chosen tool_site. Automatic probing
        # proposes each available hand independently; it is not a two-hand obligation.
        available=[]
        for arm in entries:
            binding=selected[arm["id"]]
            tcp=("robot__"+arm["tcp_site"] if binding["mode"]=="stock" else
                 str(binding["instance"])+"__tcp")
            if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,tcp)>=0:
                available.append((arm["id"],tcp))
        targets = [{"id":i["id"],"site":i["site"],"tool_site":tcp,"offset":[0,0,.03],
                    "quaternion":[1,0,0,0],"position_tolerance_m":.005,"angle_tolerance_rad":.1,"allowed_contacts":[]}
                   for i in manifest["interfaces"] if i.get("site") and i["kind"] not in {"software","mount"}
                   and not any(i["site"].startswith(t["id"]+"__") for t in tools)
                   for arm,tcp in available]
    target_groups={}
    for target in targets:
        target_groups.setdefault(target["id"],[]).append(target)
    if not static:
        require(len(target_groups) <= 256,"Too many operating poses")
    poses = [("scene",home)]
    for alternatives in (target_groups.values() if not static else ()):
        attempts=[]
        for target in alternatives:
            result=reach_target(model,home,env,target,limits,recovery=recovery)
            attempts.append((target,result))
            if result["status"]=="pass":break
        target,result=min(attempts,key=lambda pair:{"pass":0,"partial":1,"fail":2}[pair[1]["status"]])
        add("reach:"+target["id"],result["status"] == "pass",result["detail"],
            [target["site"].split("__")[0],target["tool_site"].split("__")[0]], probe=result)
        if result['status']=='partial':
            rows[-1].update(status='partial',coverage_gaps=result['coverage_gaps'])
        if result["status"] == "pass" and len(poses) < 4:
            poses.append(("reach_"+str(len(poses)),np.array(result["qpos"])))
    renders = []
    if render_dir is not None:
        with stage("scene_render"):
            renders = render_views(model,poses,Path(render_dir))
            renders.extend(render_details(model,home,plan,render_dir))
    coverage = ["Discrete geometry and operating-pose probing do not prove a continuous trajectory or completion of the full experiment",
                "Mesh collision distances use the engine convex hull; concave cavities must be split into separate collision geoms",
                "Per-component support is combined with visible-material connectivity and internal-overlap checks; support points are checked against displayed surfaces. Full surface equivalence, manufacturability/fit of moving connections and center-of-mass stability remain unproven",
                "The rated-payload check does not yet certify the manufacturer's full torque and inertia envelope"]
    if static:
        rows.append({"id":"robot_reachability", "status":"skipped", "instances":[], "detail":"Static mode does not solve operating poses"})
        coverage.append("Static mode: motion clearance, dynamic stability, operating interaction and robot reachability not verified")
    if not contract["payload_mass_kg"] and not any(contract.get("arm_payloads",{}).values()):
        coverage.append("No carried-object mass declared; only the unloaded end-effector mass is checked for now")
    if automatic_targets and not static:
        coverage.append("Operating poses are inferred from interface frames with a fixed 30 mm approach offset; direction still needs confirmation by task planning")
    if not targets and not static:
        coverage.append("No probeable operation target; robot reachability not verified")
    layout_advice = {}
    if "layout" in plan:
        from .mobile import assess, boundaries, articulation
        rows.extend(boundaries(plan,env,geometry,model))
        layout_advice = assess(model,data,plan,env,geometry,manifest)
        layout_advice["articulations"] = articulation(model,data)
    checkpoints.verify_input()
    return validate_report({"status":report_status(rows),
            **({"layout_assessment":layout_advice} if layout_advice else {}),
            "error_kind":"measurement_error" if any(r["status"] == "error" for r in rows) else None,
            "distance_recoveries":query.recoveries,
            "scale":scale,"degraded":budgets["degraded"],
            "checks":rows,"renders":renders,"coverage":coverage,"manifest":manifest,
            "validation_mode":limits.validation_mode},domain='scene',mode=limits.validation_mode)
