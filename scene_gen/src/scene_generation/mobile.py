"""Bounded geometric advice for layout; never an operation/trajectory certificate."""
import math
from collections import deque

import numpy as np

from .geometry import owner, bounds
from .layout import world_xy


def segment_box(start, end, low, high):
    """Conservative open-segment/AABB intersection; bounds are not precise collisions."""
    enter,leave=1e-5,1-1e-5
    for a,b,lo,hi in zip(start,end,low,high):
        delta=b-a
        if abs(delta)<1e-12:
            if a<lo or a>hi:return False
        else:
            t1,t2=sorted(((lo-a)/delta,(hi-a)/delta))
            enter,leave=max(enter,t1),min(leave,t2)
            if enter>leave:return False
    return True


def assess(model, data, plan, env, geometry, manifest=None):
    """Circular base envelope + conservative grid connectivity + arm distance.

    Grid edges sample whole-cell clearance, so diagonal corner-cutting is absent.
    No inverse kinematics, wheel dynamics or manipulation path is claimed.
    """
    layout = plan["layout"]
    from .robot_contract import arms
    import mujoco
    entries=arms(env)
    home = np.array(layout["robot_home"][:2])
    yaw=layout["robot_home"][2]
    rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    # Shoulder positions from the actual selected model and posture. A legacy
    # source without shoulder sites explicitly retains its approximate reference.
    probe=mujoco.MjData(model); probe.qpos[:]=data.qpos
    lift_ids=[]
    for name in env["robot"].get("shared_joints",[]):
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"robot__"+name)
        if j>=0 and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE and model.jnt_limited[j]:
            lift_ids.append(j)
    postures=[{}]+[{model.joint(j).name.removeprefix("robot__"):float(q)}
                   for j in lift_ids for q in model.jnt_range[j]]
    shoulders=[]
    custom={i["id"] for i in plan["instances"] if i["placement"]=="robot"}
    from .tool_contract import has_free_parts
    coupled=any(has_free_parts(model,key+"__") for key in custom)
    if coupled:
        # Free roots need the assembly's explicit weld transforms when lifting.
        # This read-only geometric sampler cannot invent those transforms.
        postures=[{}]
    robot_geoms=[g for g in range(model.ngeom) if owner(model,g) in custom|{"robot"}
                 and model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE]
    radius=math.hypot(*env["robot"]["mobile"]["base_size"][:2])/2
    current_radius=radius
    for index,posture in enumerate(postures):
        probe.qpos[:]=data.qpos
        for name,q in posture.items():
            probe.qpos[model.joint("robot__"+name).qposadr[0]]=q
        mujoco.mj_kinematics(model,probe)
        for g in robot_geoms:
            lo,hi=bounds(model,probe,g)
            radius=max(radius,max(np.linalg.norm(np.array([x,y])-home)
                                  for x in (lo[0],hi[0]) for y in (lo[1],hi[1])))
        if index==0: current_radius=radius
        for arm in entries:
            sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"robot__"+arm["shoulder_site"]) if arm["shoulder_site"] else -1
            pos=probe.site_xpos[sid].copy() if sid>=0 else np.r_[home,env["robot"]["mobile"]["mount_height"]]
            shoulders.append({"arm":arm["id"],"xy":(pos[:2]-home)@rotation,
                              "height":float(pos[2]),"reach":arm["reach_m"],"posture":posture,
                              "source":"model_shoulder_site" if sid>=0 else "legacy_mount_approximation"})
    def distances(positions, target, shoulder, angles):
        c,s=np.cos(angles),np.sin(angles)
        offset=shoulder["xy"]
        xy=positions+np.column_stack((c*offset[0]-s*offset[1],s*offset[0]+c*offset[1]))
        return np.sqrt(np.sum((xy-target[:2])**2,axis=1)+(target[2]-shoulder["height"])**2)
    room = np.array(layout["room"]["size"])/2
    by_instance = {}
    for g, box in geometry.items():
        by_instance.setdefault(owner(model,g),[]).append(box)
    obstacles = []
    for t in layout["tables"]:
        corners = np.array([world_xy([x,y],t) for x in (-t["size"][0]/2,t["size"][0]/2)
                            for y in (-t["size"][1]/2,t["size"][1]/2)])
        obstacles.append((corners.min(axis=0),corners.max(axis=0)))
    for i in plan["instances"]:
        if i["placement"] == "robot":
            continue
        boxes = by_instance.get(i["id"],[])
        if boxes:
            obstacles.append((np.min([b[0][:2] for b in boxes],axis=0),
                              np.max([b[1][:2] for b in boxes],axis=0)))
    step = max(.15, float(max(room)*2/160))
    clearance = radius+step/math.sqrt(2)
    xs,ys = [np.arange(-r+clearance,r-clearance+1e-9,step) for r in room]
    def clear(p, margin=clearance):
        return (bool(np.all(np.abs(p)+margin <= room-.025)) and
                all(np.linalg.norm(np.maximum(np.maximum(lo-p,p-hi),0)) > margin
                    for lo,hi in obstacles))
    free = {(x,y) for x,a in enumerate(xs) for y,b in enumerate(ys) if clear(np.array([a,b]))}
    home = np.array(layout["robot_home"][:2])
    # A nearby grid center is only connected if the whole segment is clear.
    seeds = [cell for cell in free if np.linalg.norm(np.array([xs[cell[0]],ys[cell[1]]])-home) <= step*1.5
             and all(clear(home+(np.array([xs[cell[0]],ys[cell[1]]])-home)*u, radius+step/10)
                     for u in np.linspace(0,1,16))]
    reached = set(seeds) if clear(home,radius) else set()
    queue = deque(reached)
    while queue:
        x,y = queue.popleft()
        for nxt in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if nxt in free and nxt not in reached:
                reached.add(nxt);queue.append(nxt)
    positions = np.array([[xs[x],ys[y]] for x,y in sorted(reached)]).reshape(-1,2)
    def connected(point):
        if not clear(point,radius):
            return False
        nearby = positions[np.linalg.norm(positions-point,axis=1)<=step*1.5]
        return any(all(clear(point+(p-point)*u,radius+step/10) for u in np.linspace(0,1,16)) for p in nearby)
    grouped_bounds={key:(np.min([b[0] for b in boxes],axis=0),np.max([b[1] for b in boxes],axis=0))
                    for key,boxes in by_instance.items() if boxes and key!="environment"}
    grouped_bounds.update({"environment:"+str(g):box for g,box in geometry.items() if owner(model,g)=="environment"})
    rows = []
    for item in plan["instances"]:
        if item["placement"] == "robot":
            continue
        # Only declared operation interfaces; assembly/measurement sites are not tasks.
        declared_interfaces = [i for i in (manifest or {}).get("interfaces",[])
                               if i.get("site", "").startswith(item["id"]+"__")
                               and i.get("kind") in {"grasp","press","rotate","insert","cut"}]
        targets = [(i["id"],data.site_xpos[model.site(i["site"]).id]) for i in declared_interfaces]
        target_source = "operation_interfaces" if targets else "geometry_center_approximation"
        boxes = by_instance.get(item["id"],[])
        if not targets and boxes:
            targets = [("geometry_center", (np.min([b[0] for b in boxes],axis=0)+
                                          np.max([b[1] for b in boxes],axis=0))/2)]
        outcomes = []
        target_interfaces={i["id"]:i for i in declared_interfaces}
        for label,target in targets:
            interface=target_interfaces.get(label,{})
            axis=interface.get("approach_axis")
            normal=(data.site_xmat[model.site(interface["site"]).id].reshape(3,3)@np.array(axis)) if axis else None
            pose = None
            best=None
            angles=np.arctan2(target[1]-positions[:,1],target[0]-positions[:,0])
            for shoulder in shoulders:
                distance=distances(positions,target,shoulder,angles)
                candidates=np.flatnonzero(distance<=shoulder["reach"])
                # Bounded conservative approach screening; absence of a suggestion is not a collision certificate.
                for index in candidates[np.argsort(distance[candidates])][:8]:
                    angle=angles[index];c,s=math.cos(angle),math.sin(angle)
                    offset=shoulder["xy"]
                    start=np.r_[positions[index]+[c*offset[0]-s*offset[1],s*offset[0]+c*offset[1]],shoulder["height"]]
                    if normal is not None and np.dot(start-target,normal)<=0: continue
                    if any(segment_box(start,target,lo,hi) for key,(lo,hi) in grouped_bounds.items()
                           if key not in {item["id"],"robot",*custom}): continue
                    if best is None or distance[index]<best[0]:
                        best=(float(distance[index]),index,shoulder)
                    break
            if best:
                _,index,shoulder=best
                pose=[*positions[index].tolist(),float(angles[index])]
            outcomes.append({"target":label,"position":target.tolist(),"suggested_dock":pose,
                             "arm":best[2]["arm"] if best else None,
                             "robot_posture":best[2]["posture"] if best else None,
                             "shoulder_source":best[2]["source"] if best else None,
                             "approach_direction":"declared_axis_screened" if normal is not None else "not_declared",
                             "obstruction_screen":"other_object_bounds_only",
                             "screen_limit_per_arm_posture":8,"operation_verified":False,
                             "distance_m":best[0] if best else None,
                             "status":"plausible" if pose else "needs_layout_review"})
        declared = []
        for dock in layout["docks"]:
            if dock["instance"] != item["id"]:
                continue
            point = np.array(dock["pose"][:2])
            per_arm=[{"arm":s["arm"],"robot_posture":s["posture"],
                      "target_distances_m":[float(distances(point[None,:],t,s,np.array([dock["pose"][2]]))[0])
                                            for _,t in targets]} for s in shoulders]
            declared.append({"id":dock["id"],"pose":dock["pose"],"base_clear":clear(point,current_radius),
                             "whole_robot_envelope_clear":clear(point,current_radius),
                             "grid_connected":connected(point),"arm_distances":per_arm})
        rows.append({"instance":item["id"],"targets":outcomes,"declared_docks":declared,
                     "target_source":target_source,"operation_verified":False,
                     "status":"plausible" if outcomes and all(t["suggested_dock"] for t in outcomes)
                              else "needs_layout_review"})
    from .layout_metrics import measure
    return {"compactness":measure(plan,by_instance), "method":"whole_robot_envelope_and_per_arm_shoulder_distance",
            "qualification":"使用实际姿态整机水平外廓（含双臂和已装工具）的保守圆包络及逐臂肩部距离；升降建议使用所有采样姿态外廓的并集。包络相交仅需布局复核，不证明实际碰撞；操作接口方向仅在声明时筛查，肩部到目标线段仅筛查其他对象包围盒，每臂每升降姿态最多8个候选，自身机构和完整开合未验证；未求解操作姿态、臂部绕障、轮式导航或操作轨迹，不能认证任务可执行",
            "base_radius_m":radius,"declared_posture_radius_m":current_radius,"robot_posture":plan.get("robot_posture",{}),
            "lift_sampling":"current_only_coupled_tool" if coupled else "current_and_limits",
            "grid_step_m":step,"reachable_cells":len(reached),
            "home_clear":clear(home,radius),"instances":rows}


def articulation(model, data):
    """Verify kinematic motion exists; do not infer collision-free travel or a motor."""
    import mujoco
    probe=mujoco.MjData(model)
    probe.qpos[:]=data.qpos
    mujoco.mj_kinematics(model,probe)
    positions, rotations=probe.xpos.copy(),probe.xmat.copy()
    rows=[]
    for j in range(model.njnt):
        name=model.joint(j).name or ""
        if name.startswith("robot__") or model.jnt_type[j] not in (
                mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        adr=model.jnt_qposadr[j]
        lo,hi=model.jnt_range[j]
        q=hi if abs(data.qpos[adr]-hi)>abs(data.qpos[adr]-lo) else lo
        if not model.jnt_limited[j]:
            q=data.qpos[adr]+.1
        probe.qpos[:]=data.qpos;probe.qpos[adr]=q;mujoco.mj_kinematics(model,probe)
        b=model.jnt_bodyid[j]
        moved=bool(np.linalg.norm(probe.xpos[b]-positions[b])>1e-8 or
                   np.linalg.norm(probe.xmat[b]-rotations[b])>1e-8)
        rows.append({"joint":name,"instance":name.split("__")[0],"range": [float(lo),float(hi)],
                     "limited":bool(model.jnt_limited[j]),"kinematic_motion":moved,
                     "qualification":"只检查单关节运动学变化；未认证闭链约束、驱动、空间无碰撞或操作可用性"})
    return rows


def boundaries(plan, env, geometry, model):
    """Finite physical boundary checks, separate from rough approach advice."""
    room = np.array(plan["layout"]["room"]["size"])/2
    result = []
    for i in plan["instances"]:
        if i["placement"]=="robot":
            continue
        boxes = [b for g,b in geometry.items() if owner(model,g)==i["id"]]
        ok = bool(boxes) and all(np.all(lo[:2]>=-room) and np.all(hi[:2]<=room) for lo,hi in boxes)
        result.append({"id":"room_boundary:"+i["id"],"status":"pass" if ok else "fail",
                       "instances":[i["id"]],"detail":"实际几何处于已确定房间平面边界内"})
    violations = []
    for t in plan["layout"]["tables"]:
        points = [world_xy([x,y],t) for x in (-t["size"][0]/2,t["size"][0]/2)
                  for y in (-t["size"][1]/2,t["size"][1]/2)]
        if not all(np.all(np.abs(p)<=room) for p in points):
            violations.append({"tables":[t["id"]],"kind":"outside_room"})
    tables = plan["layout"]["tables"]
    for n,a in enumerate(tables):
        for b in tables[n+1:]:
            def corners(t):
                return np.array([world_xy([x,y],t) for x in (-t["size"][0]/2,t["size"][0]/2)
                                 for y in (-t["size"][1]/2,t["size"][1]/2)])
            ca,cb=corners(a),corners(b)
            axes=[np.array([math.cos(t["yaw"]),math.sin(t["yaw"])]) for t in (a,b)]
            axes += [np.array([-v[1],v[0]]) for v in axes]
            overlap=all(min(max(ca@v),max(cb@v))-max(min(ca@v),min(cb@v)) > 1e-9 for v in axes)
            if overlap:
                violations.append({"tables":[a["id"],b["id"]],"kind":"overlapping_footprints"})
    ids = {t for v in violations for t in v["tables"]}
    result.append({"id":"workspace_layout","status":"fail" if violations else "pass",
                   "instances":["environment"]+[i["id"] for i in plan["instances"] if i.get("table_id") in ids],
                   "detail":"工作台位于房间内且独立地面占用区域不重叠；非实际几何穿透量测",
                   "violations":violations})
    return result
