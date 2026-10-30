"""只读检查工具：编译已归档几何、摆静态检查姿态、拍图和量距；从不执行建模代码。"""

from pathlib import Path
import math

import mujoco
import numpy as np
from PIL import Image

from .checks import load_model, load_scene_model
from .static_state import geometry_state
from .contracts import read_json, require
from asset_prep.errors import ValidationError
from .distance import MeasurementError


def requests_valid(requests, scope):
    require(isinstance(scope,str) and scope in {'asset','scene'}, '检查范围无效')
    require(isinstance(requests, list) and 1 <= len(requests) <= 4, "一次补证据须有1至4个请求")
    for request in requests:
        require(isinstance(request, dict), "检查请求须为对象")
        if "question" in request:
            require(isinstance(request["question"],str) and 1<=len(request["question"])<=80,"invalid question identity")
        configuration = request.get("configuration")
        if configuration is not None:
            require(scope=="asset" and isinstance(configuration,str), "configuration only supported for assets")
        request = {k:v for k,v in request.items() if k not in {"configuration","question"}}
        kind = request.get("kind")
        require(isinstance(kind,str), '检查种类须为文本')
        from .inspection_registry import REGISTRY
        if kind in REGISTRY:
            REGISTRY[kind].validate(request,scope)
        elif kind == "inventory":
            require(set(request) == {"kind"}, "清单请求不接受其他字段")
        elif kind == "view":
            require({"kind","joints","hide_bodies","focus_body","azimuth","elevation"} <= set(request) <=
                    {"kind","joints","hide_bodies","focus_body","azimuth","elevation","focus_geoms","hide_sites","zoom","diagnostic_light"}, "视图字段无效")
            for flag in ("hide_sites","diagnostic_light"):
                require(flag not in request or type(request[flag]) is bool, "observation flag must be boolean")
            if "focus_geoms" in request:
                require(isinstance(request["focus_geoms"],list) and 1<=len(request["focus_geoms"])<=128 and
                        all(isinstance(g,str) and g for g in request["focus_geoms"]), "invalid target geometries")
            require(type(request.get("zoom",1)) in (int,float) and math.isfinite(request.get("zoom",1)) and
                    .25 <= request.get("zoom",1) <= 4, "invalid observation zoom")
            require(isinstance(request["joints"], dict) and len(request["joints"]) <= 16, "检查姿态关节数超限")
            require(all(isinstance(n,str) and type(v) in (int,float) and math.isfinite(v)
                        for n,v in request["joints"].items()), "关节姿态无效")
            require(isinstance(request["hide_bodies"], list) and len(request["hide_bodies"]) <= 16
                    and all(isinstance(n,str) for n in request["hide_bodies"]), "隐藏部件列表无效")
            require(request["focus_body"] is None or isinstance(request["focus_body"],str), "视图焦点无效")
            require(all(type(request[k]) in (int,float) and math.isfinite(request[k]) for k in ("azimuth","elevation"))
                    and -360 <= request["azimuth"] <= 360 and -90 <= request["elevation"] <= 90, "相机角度超限")
        elif kind in {"clearance","perimeter","mount"}:
            require(scope == "scene" and set(request) == {"kind","instance","distance_m"}, "仅场景支持实例净空量测")
            require(isinstance(request["instance"],str) and type(request["distance_m"]) in (int,float)
                    and math.isfinite(request["distance_m"]) and 0 < request["distance_m"] <= 1, "量测范围无效")
        elif kind == "pairs":
            require(scope == 'scene' and set(request)=={'kind','pairs','distance_m'},'几何对检查字段无效')
            require(isinstance(request['pairs'],list) and 1<=len(request['pairs'])<=32 and
                    all(isinstance(p,list) and len(p)==2 and all(isinstance(n,str) and n for n in p) for p in request['pairs']),'几何对须为最多32个实际名称对')
            require(type(request['distance_m']) in (int,float) and math.isfinite(request['distance_m']) and 0<request['distance_m']<=1,'量测范围无效')
        elif kind == 'support':
            require(scope=='scene' and {'kind','instance'}<=set(request)<={'kind','instance','object_geoms','support_geoms','support_tolerance_m','collision_tolerance_m'}
                    and isinstance(request['instance'],str),'承托检查字段无效')
            for field in ('support_tolerance_m','collision_tolerance_m'):
                require(field not in request or type(request[field]) in (int,float) and math.isfinite(request[field]) and request[field]>=0,'invalid fixed tolerance')
            for key in ('object_geoms','support_geoms'):
                if key in request:
                    require(isinstance(request[key],list) and 1<=len(request[key])<=128 and all(isinstance(n,str) and n for n in request[key]),'承托几何选择无效')
        else:
            require(False, "不支持的补证据工具")
    from asset_prep.storage import digest
    require(len({digest(r) for r in requests})==len(requests), '检查请求不能重复')
    return requests


def descendants(model, name):
    try:
        root = model.body(name).id
    except KeyError:
        raise ValueError("未知部件：" + name) from None
    require(root > 0, "不能隐藏或聚焦世界根节点")
    result = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def inventory(model, data, scope):
    from scene_generation.geometry import owner
    from scene_generation.geometry import bounds
    return {"unit":"m", "angle_unit":"rad", "scope":scope,
            "geometry":[{"name":model.geom(g).name,"body":model.body(int(model.geom_bodyid[g])).name,
                         "bounds":[x.tolist() for x in bounds(model,data,g)]}
                        for g in range(model.ngeom) if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE],
            "joints":[{"name":model.joint(j).name, "type":int(model.jnt_type[j]),
                       "limited":bool(model.jnt_limited[j]), "range":model.jnt_range[j].tolist()}
                      for j in range(model.njnt)],
            "joint_types":{"hinge":3,"slide":2,"ball":1,"free":0},
            "bodies":[model.body(b).name for b in range(1,model.nbody)],
            "geoms":[model.geom(g).name for g in range(model.ngeom)],
            "sites":[model.site(s).name for s in range(model.nsite)],
            "instances":sorted({owner(model,g) for g in range(model.ngeom)}) if scope == "scene" else []}


def clearance(model, data, request, max_pairs=200000):
    from .distance import DistanceQuery, observe
    query=DistanceQuery(model,data)
    from scene_generation.geometry import bounds, owner
    limit = float(request["distance_m"])
    owners = [owner(model,g) for g in range(model.ngeom)]
    omitted = {"unassigned", "environment", "table", "floor"}
    target = request["instance"]
    require(target in owners and target not in omitted, "净空目标不是已知场景实例")
    # 有限视觉几何也参与；不受碰撞位掩码过滤。承托台面、地面不属于周边设备净空。
    ids = [g for g in range(model.ngeom) if owners[g] not in omitted]
    boxes = {g:bounds(model,data,g) for g in ids}
    left = [g for g in ids if owners[g] == target]
    right = [g for g in ids if owners[g] != target]
    require(left and right, "净空量测缺少目标或其他实例几何")
    lower = np.array([boxes[g][0] for g in right])
    upper = np.array([boxes[g][1] for g in right])
    nearest, pair, count = limit, None, 0
    errors=[]; skipped=0; error_count=0
    for g in left:
        lo, hi = boxes[g]
        gaps = np.maximum(np.maximum(lower-hi, lo-upper),0)
        candidates = np.flatnonzero(np.linalg.norm(gaps,axis=1) <= limit)
        for index in candidates:
            other = right[index]
            if count>=max_pairs:
                skipped+=1
                continue
            count+=1
            measurement=observe(query,g,other,limit)
            if measurement['outcome']=='unknown':
                error_count+=1
                errors.append({**measurement,'geom1':model.geom(g).name,'geom2':model.geom(other).name,
                                   'instances':[target,owners[other]]})
                continue
            distance=measurement['distance_m']
            if distance is None:
                continue
            require(math.isfinite(distance), "距离结果非有限值")
            if distance < nearest:
                nearest, pair = distance, [model.geom(g).name,model.geom(other).name]
    if skipped:
        errors.append({'error_kind':'resource_limit','unexamined_pairs':skipped})
    return {"kind":"clearance", "instance":target, "query_distance_m":limit,
            "minimum_distance_m":nearest if pair and not errors else None,
            "measured_minimum_m":nearest if pair else None,"lower_bound_m":None if pair or errors else limit,
            "measurement_errors":errors,"measurement_error_count":error_count,"complete":not errors,
            "nearest_geoms":pair, "examined_pairs":count,
            "distance_recoveries":query.recoveries,
            "other_instances":sorted({owners[g] for g in right}),
            "excluded_owners":sorted(omitted),
            "meaning":"当前初始姿态的三维碰撞形状距离；网格使用凸包。阈值来自模型查询，不自动作为用户验收标准；不证明其他姿态或散热性能。"}


def inspect_artifact(root, scope, requests, output):
    from scene_generation.geometry import bounds
    require(scope in {"asset","scene"}, "补证据阶段无效")
    requests_valid(requests,scope)
    root, output = Path(root),Path(output)
    from .configurations import selection, catalogue
    selected = {r.get("configuration") for r in requests}
    require(len(selected)==1, "one inspection batch must use one configuration")
    if scope == "asset":
        available = catalogue(root)
        selected_name = next(iter(selected))
        if selected_name is None and set(available) != {"default"}:
            if all(r["kind"]=="inventory" for r in requests):
                inventories = {}
                for name in available:
                    _, filename, _ = selection(root,name)
                    m=load_model(root,model_file=filename); d=mujoco.MjData(m); geometry_state(m,d)
                    inventories[name]=inventory(m,d,scope)
                return {"status":"pass","results":[],"renders":[],"simulation_steps":0,"source_mutated":False,
                        "inventory":{"configurations":inventories}}
            require(False,"choose configuration explicitly")
        _, filename, _ = selection(root,selected_name)
        model=load_model(root,model_file=filename)
    else:
        model=load_scene_model(root)
    data = mujoco.MjData(model)
    if scope == "scene":
        meta = read_json(root/"scene.json",8000000)
        data.qpos[:],data.ctrl[:] = meta["home_qpos"],meta["home_ctrl"]
    home = data.qpos.copy()
    geometry_state(model,data)
    report = {"status":"pass", "results":[], "renders":[], "simulation_steps":0,
              "source_mutated":False, "inventory":inventory(model,data,scope)}
    for number, request in enumerate(requests):
        data.qpos[:], data.qvel[:] = home,0
        geometry_state(model,data)
        try:
            from .inspection_registry import REGISTRY
            if request["kind"] in REGISTRY:
                result=REGISTRY[request["kind"]].inspect(model,data,request,root,scope,output,number)
                report["results"].append(result)
                if result.get("complete") is False: report["status"]="incomplete"
                continue
            if request["kind"] == "inventory":
                continue
            if request["kind"] == "clearance":
                result=clearance(model,data,request)
                report["results"].append({**result, 'request':request})
                if result.get('measurement_errors'):
                    report['status']='incomplete'
                    report['results'][-1]['error_kind']='measurement_error'
                continue
            if request['kind'] in {'pairs','support','perimeter','mount'}:
                from scene_generation import inspection
                kind=request['kind']
                if kind=='pairs':
                    result=inspection.pairs(model,data,request)
                elif kind=='perimeter':
                    result=inspection.perimeter(root,model,data,meta,request)
                else:
                    result=getattr(inspection,kind)(model,data,meta,request)
                report['results'].append({**result, 'request':request})
                if result.get('measurement_errors'):
                    report['status']='incomplete'
                    report['results'][-1]['error_kind']='measurement_error'
                continue
            for name,value in request["joints"].items():
                joint = model.joint(name).id
                require(model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)
                        and model.jnt_limited[joint], "检查姿态只支持有明确限位的铰链或滑动关节")
                require(model.jnt_range[joint,0] <= value <= model.jnt_range[joint,1], "检查姿态超过原模型限位")
                data.qpos[model.jnt_qposadr[joint]] = value
            geometry_state(model,data)
            from .observation import render_observation
            result=render_observation(model,data,request,output,number,descendants)
            report["results"].append(result)
            report["renders"].append(result["image"])
            if result.get("error_kind"): report["status"]="incomplete"
        except (ValueError,KeyError,RuntimeError,ValidationError) as exc:
            report["results"].append({"request":request,"error":str(exc)[:1500],
                                      "error_kind":"measurement_error" if isinstance(exc,MeasurementError) else "inspection_error"})
            report["status"] = "incomplete"
    return report
