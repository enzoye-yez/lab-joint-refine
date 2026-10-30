"""Independent bounded diagnostics on copied state. No simulator step or world edits."""
from pathlib import Path
import numpy as np
import mujoco
from asset_prep.storage import file_hash, digest
from .common import Settings, copy_data, need, read, atomic_json, identity
from .scene import load


def owner(model, geom, instances):
    name=model.body(int(model.geom_bodyid[geom])).name
    return next((i for i in sorted(instances,key=len,reverse=True) if name.startswith(i+"__")),
                "robot" if name.startswith("robot__") else "environment")


def contact_evidence(model, data, instances, tolerance):
    return [{"geoms":[model.geom(int(c.geom1)).name or "#"+str(c.geom1),model.geom(int(c.geom2)).name or "#"+str(c.geom2)],
        "geom_indices":[int(c.geom1),int(c.geom2)],"bodies":[model.body(int(model.geom_bodyid[c.geom1])).name,model.body(int(model.geom_bodyid[c.geom2])).name],
        "instances":sorted({owner(model,c.geom1,instances),owner(model,c.geom2,instances)}),
        "distance_m":float(c.dist),"limit_m":tolerance}
        for c in data.contact if c.dist < -tolerance]


def inspect_scene(scene, scene_hash, settings=None, snapshot=None, queries=()):
    settings=settings or Settings()
    model,data,meta,manifest=load(scene,scene_hash)
    if snapshot is not None:
        state=np.asarray(snapshot["physics"],float)
        need(state.shape==(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION),) and np.isfinite(state).all(),"失败状态维度或数值无效")
        mujoco.mj_setState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION); mujoco.mj_forward(model,data)
    original=data.qpos.copy()
    instances={i["id"] for i in meta["plan"]["instances"]}
    contacts=contact_evidence(model,data,instances,settings.penetration_m)
    rows=[]
    for c in contacts:
        kind="asset" if len(c["instances"])==1 and c["instances"][0] not in {"robot","environment"} else "layout"
        # A collision observed during execution is a fact, not proof the original asset is wrong.
        rows.append({"kind":"collision","scope":kind,"status":"confirmed_observation",
                     "repair_authorized":snapshot is None and "robot" not in c["instances"],**c})
    need(len(queries)<=32,"局部诊断请求过多")
    by_id={i["id"]:i for i in manifest["interfaces"]}
    for query in queries:
        need(set(query)<={"kind","interface","body","distance","samples"} and query.get("kind") in {"mechanism","fit"},"未知诊断请求")
        interface=by_id.get(query.get("interface")); need(interface is not None,"诊断接口不存在")
        op=interface.get("operation",{}); count=query.get("samples",9)
        need(type(count) is int and 2<=count<=56,"诊断姿态数须在2至56内")
        samples=[]; coverage=[]; clone=mujoco.MjData(model); copy_data(clone,model,data)
        if query["kind"]=="mechanism":
            if not all(k in op for k in ("joint","range")):
                rows.append({"kind":"mechanism","interface":interface["id"],"status":"unknown","repair_authorized":False,"coverage_gaps":["缺少声明的实际关节和必要行程"]}); continue
            jid=model.joint(op["joint"]).id; body=int(model.jnt_bodyid[jid])
            locked=any(model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and data.eq_active[e] and
                       (model.eq_obj1id[e]==jid or model.eq_obj2id[e]==jid) for e in range(model.neq))
            from .evidence import descendants
            moving=descendants(model,body)
            locked |= any(data.eq_active[e] and model.eq_type[e] in (mujoco.mjtEq.mjEQ_WELD,mujoco.mjtEq.mjEQ_CONNECT)
                          and ((int(model.eq_obj1id[e]) in moving)!=(int(model.eq_obj2id[e]) in moving)) for e in range(model.neq))
            if locked:
                rows.append({"kind":"mechanism","interface":interface["id"],"status":"unknown","repair_authorized":False,"coverage_gaps":["关节当前锁止，须由真实解锁事件推进，诊断不擅自解除"]}); continue
            own=next((i for i in instances if interface["body"].startswith(i+"__")),None)
            affected=set()
            for b in range(1,model.nbody):
                a=b
                while a and a!=body: a=int(model.body_parentid[a])
                if a==body: affected.add(b)
            for value in np.linspace(*op["range"],count):
                clone.qpos[:]=data.qpos; clone.qpos[model.jnt_qposadr[jid]]=value; mujoco.mj_forward(model,clone)
                hits=[c for c in contact_evidence(model,clone,instances,settings.penetration_m)
                      if any(int(model.geom_bodyid[g]) in affected for g in c["geom_indices"])]
                samples.append({"joint_value":float(value),"collisions":hits})
            intrinsic=[c for s in samples for c in s["collisions"] if c["instances"]==[own]]
            external=[c for s in samples for c in s["collisions"] if c["instances"]!=[own]]
            rows.append({"kind":"mechanism","scope":"asset" if intrinsic else "layout","interface":interface["id"],
                "instances":sorted({i for c in intrinsic+external for i in c["instances"] if i in instances}),
                "status":"confirmed_defect" if intrinsic else "confirmed_observation" if external else "sampled_clear",
                "repair_authorized":bool(intrinsic),"samples":samples,
                "coverage_gaps":["离散几何姿态检查，不认证连续路径、驱动力或接触稳定性"]})
        else:
            # Relative placement probe: move only an explicitly named free object in a copy.
            body_name=query.get("body"); need(isinstance(body_name,str) and not body_name.startswith("robot__"),"配合诊断须指定资产自由部件")
            bid=model.body(body_name).id; jid=int(model.body_jntadr[bid])
            need(model.body_jntnum[bid]>0 and model.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE,"配合诊断部件必须具有自由根")
            axis=op.get("axis",interface.get("approach_axis"))
            if axis is None:
                rows.append({"kind":"fit","status":"unknown","repair_authorized":False,"coverage_gaps":["缺少有依据的配合轴"]});continue
            distance=query.get("distance")
            need(type(distance) in (int,float) and np.isfinite(distance) and abs(distance)<=1,"局部配合诊断距离无效")
            sid=model.site(interface["site"]).id; axis=data.site_xmat[sid].reshape(3,3)@np.asarray(axis)
            qa=int(model.jnt_qposadr[jid])
            for value in np.linspace(0,distance,count):
                clone.qpos[:]=data.qpos; clone.qpos[qa:qa+3]+=axis*value; mujoco.mj_forward(model,clone)
                from .evidence import descendants
                affected=descendants(model,bid)
                hits=[c for c in contact_evidence(model,clone,instances,settings.penetration_m)
                      if any(int(model.geom_bodyid[g]) in affected for g in c["geom_indices"])]
                samples.append({"offset_m":float(value),"collisions":hits})
            rows.append({"kind":"fit","scope":"asset_or_operation","status":"confirmed_observation" if any(s["collisions"] for s in samples) else "sampled_clear",
                "repair_authorized":False,"interface":interface["id"],"samples":samples,
                "coverage_gaps":["仅检查指定起姿与直线配合；错误接近姿态也可能导致干涉，不自动授权改几何"]})
    gaps=["未扫描的姿态和操作仍未知"]
    gaps.extend(str(row.get("interface",row["kind"]))+": "+gap for row in rows for gap in row.get("coverage_gaps",[]))
    need(np.array_equal(original,data.qpos),"诊断改变了输入状态")
    need(identity(scene)==scene_hash,"诊断期间输入变化","stale_scene")
    return {"version":1,"scene_sha256":scene_hash,"checker":"operation_diagnostics_v1","rows":rows,
            "queries":list(queries),"state_sha256":digest(snapshot) if snapshot is not None else "initial",
            "coverage_gaps":sorted(set(gaps)),"network_calls":0}


def diagnose_run(run_root, output=None, queries=()):
    root=Path(run_root); record=read(root/"recording.json"); result=read(root/"result.json")
    files=read(root/"files.json")
    # Recorder stores a flat file -> digest manifest.
    for name in ("recording.json","initial.json","failure_state.json","task.json","graph.json","result.json","events.jsonl","steps.jsonl.gz"):
        if (root/name).exists():
            expected=files.get(name)
            need(expected==file_hash(root/name),"运行证据被修改："+name,"stale_evidence")
    state=read(root/"failure_state.json") if (root/"failure_state.json").exists() else None
    report=inspect_scene(record["scene_path"],record["scene_sha256"],Settings(**record["settings"]),state,queries)
    code=result.get("code","")
    scope="engineering" if code in {"simulation_unstable","execution_error"} else "operation"
    report["failure"]={"code":code,"scope":scope,"completed_stages":result.get("completed_stages",[])}
    if state:
        report["predicate_evidence"]=state.get("evidence",{}).get("last",{})
        report["holding"]=state.get("holding",{})
        for observed in report["predicate_evidence"].values():
            kind=observed.get("condition",{}).get("kind")
            if kind in {"held","supported","relative_pose","joint_range"} and not observed.get("passed"):
                report["rows"].append({"kind":"grasp" if kind=="held" else "support" if kind=="supported" else "execution_condition",
                    "scope":"operation","status":"confirmed_observation","repair_authorized":False,
                    "observation":observed,"coverage_gaps":["失败条件不独立证明资产结构缺陷"]})
    report["run_sha256"]=digest(files)
    if output: atomic_json(output,report)
    return report


def packet(report,limit=16):
    """Small working state, with the complete report separately retained and hashed."""
    rows=report.get("rows",[])
    return {"scene_sha256":report["scene_sha256"],"report_sha256":digest(report),
            "failure":report.get("failure"),"rows":rows[:limit],"row_count":len(rows),
            "coverage_gaps":report.get("coverage_gaps",[]),"full_report_available":True}
