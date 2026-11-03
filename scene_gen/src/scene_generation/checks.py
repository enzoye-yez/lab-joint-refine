"""完整场景独立检查与操作姿态探测；不将离散可达性称为完整轨迹。"""

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
    """多个无摩擦接触的法向锥能否形成竖直合力；不认证力矩或稳定性。"""
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
    """寻找朝上的实际承托接触，允许凸缘承托；侧壁接近不等同于支撑。"""
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
            require(set(selection) <= {model.geom(g).name for g in ids}, "承托检查几何越出指定实例或真实支撑对象")
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
                    'error':'距离界限不能证明承托接触或排除承托候选'})
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
                           'instances':[item['id']],'detail':'独立组件之间的承托接触尚未检查完'})
    return {"support_tolerance_m":tolerance,"collision_tolerance_m":collision_tolerance,
            "supported":supported,"complete":not error_count and not skipped,
            "gap_m":min((c["distance_m"] for c in contacts),default=nearest),
            "candidate_pairs":examined,"support_contacts":contacts[:12],"contact_count":len(contacts),
            "nearby_noncontacts":nearby[:6],"measurement_errors":errors,"measurement_error_count":error_count,
            "unexamined_pairs":skipped,"contact_epsilon_m":min(tolerance,1e-5),
            "vertical_resultant_available":upward_resultant(contacts),
            **components,
            "reason":"各独立组件均有承托或实际连接链；未检查连接制造适配及力矩平衡" if supported and components.get('component_scope')=='whole_instance' else
                     "部分独立组件没有已验证的承托链，详见component_support" if components.get('component_scope')=='whole_instance' and contacts else
                     "接触法向可以形成向上的竖直合力；未检查力矩平衡" if supported else
                     "存在倾斜接触但不能证明其法向形成竖直合力" if contacts else
                     "未找到邻近承托几何" if not examined else "邻近几何没有满足间隙与朝上方向的承托接触"}


def render_details(model, home, plan, output):
    """当前完整场景中的实例近景；不切换姿态、不推进物理时间。"""
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
            camera.azimuth, camera.elevation = 45,-35
            renderer.update_scene(data,camera)
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
                "coverage_gaps":[{"error_kind":"unsupported_check", "detail":"可拆末端需要约束耦合的运动学求解"}],
                "detail":"当前逆运动学只求解永久运动树，不能认证含自由可拆部件的操作姿态"}
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
                "detail":"目标工具中心没有可求解的机器人关节链"}],"detail":"目标工具中心没有可求解的机器人关节链"}
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
                            "qpos":data.qpos.tolist(), "target":target, "detail":"找到无碰撞离散操作姿态；未验证进出路径"}
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
            "detail":"有限次逆运动学搜索未找到可用姿态，不证明数学上不可达"}


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
    add("scene_collisions",not hits,"跨实例实际几何距离与机器人自接触",
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
            "加工区域位于关联实例包络附近；不认证材料承重强度",[cut["instance"]],cut=cut)
    for item in plan["instances"]:
        if item["placement"] == "robot":
            continue
        lo, hi = aggregate_bounds(model,data,item["id"])
        support = item.get("support_instance")
        evidence = checkpoints.run("support:"+item["id"],lambda: support_evidence(model,data,item,
            limits.support_tolerance_m,limits.collision_tolerance_m,geometry,query,max_pairs=pair_budget),query)
        add("support:"+item["id"], evidence.pop("supported"),
            "逐独立组件的承托接触及实际运动树/启用焊接连接链；不证明连接制造适配、重心稳定或承重强度",
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
                "普通部件在桌面内；声明的桌边夹持部件限于安装区，不豁免碰撞和支撑",[item['id']],
                bounds=[lo.tolist(),hi.tolist()],**boundary)
    from .robot_contract import arms, validate_bindings
    selected=validate_bindings(plan,env,partial=True)
    tools = [i for i in plan["instances"] if i["placement"] == "robot"]
    if "arms" not in env["robot"]:
        require(len(tools)==1 or static and not tools and meta.get("partial_without_tool") is True,
                "旧场景须恰有一个安装末端；缺件必须有显式部分装配记录")
    tool = tools[0] if tools else None
    entries=arms(env)
    if "arms" in env["robot"]:
        from .inspection import mount_contact
        for tool_item in tools:
            rows.append(mount_contact(model,data,meta,tool_item["id"],
                                      limits.support_tolerance_m,limits.collision_tolerance_m,query))
    require(set(contract.get("arm_payloads",{})) <= {a["id"] for a in entries}, "工作负载引用未知机械臂")
    for arm in entries:
        binding=selected[arm["id"]]
        target=next((i for i in tools if i["id"]==binding["instance"]),None)
        kind="payload:"+arm["id"] if "arms" in env["robot"] else "payload"
        workload=contract.get("arm_payloads",{}).get(arm["id"],contract["payload_mass_kg"])
        if len(entries)>1 and contract["payload_mass_kg"] and arm["id"] not in contract.get("arm_payloads",{}):
            rows.append({"id":kind,"status":"partial","instances":[],"detail":"多臂非零工作负载须逐臂声明，不把总负载分摊或当成单臂负载"})
            continue
        if binding["mode"]!="stock" and target is None:
            rows.append({"id":kind,"status":"partial","instances":[],"detail":"该臂计划定制末端缺件，不能验证完整负载"})
            continue
        if binding["mode"]=="stock" and arm["stock"] is None:
            rows.append({"id":kind,"status":"partial","instances":[],"detail":"末端尚未装配，不能验证负载"})
            continue
        mass=sum(float(model.body_mass[b]) for b in range(model.nbody)
                 if target and (model.body(b).name or "").startswith(target["id"]+"__"))
        stock_mass=arm["stock"]["mass_kg"] if arm["stock"] and binding["mode"]!="replace" else 0
        add(kind,mass+stock_mass+workload<=arm["payload_kg"],
            "逐臂：定制总成实际质量＋保留原厂夹爪名义质量＋该臂声明工作负载",
            [target["id"]] if target else ["robot"],arm=arm["id"],tool_mass_kg=mass,
            stock_mass_kg=stock_mass,declared_payload_kg=workload,limit_kg=arm["payload_kg"])
    for fit in contract["fits"]:
        if static and "id" in fit and any(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,n) < 0
                          for end in ("outer","inner") for n in fit[end]):
            rows.append({"id":"fit:"+fit["id"],"status":"partial","instances":[],
                         "detail":"部分场景缺少配合标记，保留原验收要求"})
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
            "实际碰撞表面上的标记间距差；孔壁语义仍须来源和视觉核对",
            sorted({n.split("__")[0] for key in ("outer","inner") for n in fit[key]}), clearance_m=clearance,
            surface_errors_m=surface_errors,evidence=fit["evidence"])
    # 在整场景中逐关节采样，补充仪器开盖撞到邻居的问题。
    sample_failures, sample_count, sample_gaps = [], 0, []
    from .tool_contract import has_free_parts
    coupled_tools = [i["id"]+"__" for i in tools if has_free_parts(model,i["id"]+"__")]
    for j in (range(model.njnt) if not static else ()):
        name = model.joint(j).name
        if name.startswith("robot__") or model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) or not model.jnt_limited[j]:
            continue
        if any(name.startswith(prefix) for prefix in coupled_tools):
            sample_gaps.append({"error_kind":"unsupported_check", "joint":name,
                                "detail":"可拆末端关节扫描需要约束耦合；不能只移动永久运动树"})
            continue
        if sample_count+7 > budgets["joint_samples"]:
            sample_gaps.append({'error_kind':'resource_limit','joint':name,
                                'detail':'关节扫描样本预算耗尽（随模型规模缩减，已扫描 %d 个姿态）' % sample_count})
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
    add("scene_joint_sweeps",not sample_failures,"逐关节离散开合空间检查", sorted({i for f in sample_failures for h in f["collisions"] for i in h["instances"]}),samples=sample_count,failures=sample_failures[:40])
    if sample_gaps:
        rows[-1].update(coverage_gaps=sample_gaps[:40],status='fail' if sample_failures else 'partial')
    if static:
        rows[-1].update(status="skipped", detail="静态模式跳过关节开合空间检查")
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
        "重力下有限时间检查自由部件掉落或明显移动", sorted({m["body"].split("__")[0] for m in motion}),moving_bodies=motion,seconds=float(data.time))
    if static:
        rows[-1].update(status="skipped", detail="静态模式跳过动力学步进")
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
        require(len(target_groups) <= 256,"操作姿态数超限")
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
    coverage = ["离散几何与操作姿态探测不证明连续轨迹或完整实验完成",
                "网格碰撞距离采用引擎凸包；凹腔必须拆分碰撞几何",
                "逐组件承托另结合可见材料连通与内部交叠检查；承托点核对显示表面。完整表面等价、活动连接制造适配及质心稳定仍未证明",
                "额定负载检查尚未认证制造商完整力矩及惯量包络"]
    if static:
        rows.append({"id":"robot_reachability", "status":"skipped", "instances":[], "detail":"静态模式不求解操作姿态"})
        coverage.append("静态模式：未验证运动开合空间、动力学稳定性、操作交互或机器人可达性")
    if not contract["payload_mass_kg"] and not any(contract.get("arm_payloads",{}).values()):
        coverage.append("没有声明搬运物质量，目前只核对空载末端装置质量")
    if automatic_targets and not static:
        coverage.append("操作姿态由接口坐标系和固定30毫米接近偏移推断，尚需任务规划确认方向")
    if not targets and not static:
        coverage.append("没有可探测操作目标，未验证机器人可达性")
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
