"""Shared arm and end-effector semantics, including legacy single-arm inputs."""
from copy import deepcopy
import re
from asset_generation.contracts import require, finite


def arms(env):
    robot = env["robot"]
    if "arms" in robot:
        return robot["arms"]
    return [{"id":"main","joints":robot["joints"],"wrist_site":robot["attachment_site"],
             "tcp_site":robot["attachment_site"],"shoulder_site":None,
             "reach_m":robot["reach_m"],"payload_kg":robot["payload_kg"],"stock":None}]


def validate_robot(robot):
    require(isinstance(robot,dict), "机器人配置无效")
    common={"name","directory","joints","home"}
    extra={"mobile"}
    required=(common|{"arms","shared_joints","model_variants"} if "arms" in robot else
              common|{"attachment_site","payload_kg","reach_m"})
    require(required <= set(robot) <= required|extra, "机器人配置字段无效")
    require(all(isinstance(robot[k],str) and robot[k] for k in ("name","directory")), "机器人来源缺失")
    joints=robot["joints"]
    require(isinstance(joints,list) and joints and all(isinstance(j,str) and j for j in joints)
            and len(joints)==len(set(joints)), "机器人关节无效")
    require(isinstance(robot["home"],list) and len(robot["home"])==len(joints)
            and all(finite(q) for q in robot["home"]), "机器人初始姿态无效")
    entries=arms({"robot":robot})
    require(isinstance(entries,list) and entries, "至少声明一只机械臂")
    ids=[]
    grouped=list(robot.get("shared_joints",[]))
    for arm in entries:
        require(isinstance(arm,dict) and set(arm)=={"id","joints","wrist_site","tcp_site","shoulder_site","reach_m","payload_kg","stock"},
                "机械臂字段无效")
        require(isinstance(arm["id"],str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}",arm["id"]), "机械臂编号无效")
        ids.append(arm["id"])
        require(isinstance(arm["joints"],list) and arm["joints"] and all(isinstance(j,str) for j in arm["joints"]), "机械臂关节组无效")
        require(all(isinstance(arm[k],str) and arm[k] for k in ("wrist_site","tcp_site")), "机械臂接口缺失")
        require(arm["shoulder_site"] is None or isinstance(arm["shoulder_site"],str), "肩部接口无效")
        require(all(finite(arm[k]) and arm[k]>0 for k in ("reach_m","payload_kg")), "机械臂规格无效")
        grouped.extend(arm["joints"])
        stock=arm["stock"]
        if stock is not None:
            require(isinstance(stock,dict) and set(stock)=={"name","joints","adapter_site","mass_kg","mass_source"}, "原厂末端字段无效")
            require(all(isinstance(stock[k],str) and stock[k] for k in ("name","adapter_site","mass_source")), "原厂末端来源或接口缺失")
            require(isinstance(stock["joints"],list) and all(isinstance(j,str) for j in stock["joints"])
                    and finite(stock["mass_kg"]) and stock["mass_kg"]>0, "原厂末端关节或质量无效")
            grouped.extend(stock["joints"])
    require(len(ids)==len(set(ids)), "机械臂编号重复")
    if "arms" in robot:
        require(len(grouped)==len(set(grouped)) and set(grouped)==set(joints), "公共、手臂、夹爪关节必须无重叠且覆盖初始关节")
        variants=robot["model_variants"]
        require(isinstance(variants,dict) and variants.get("")=="model.xml", "整机默认变体缺失")
        for key,value in variants.items():
            require(isinstance(key,str) and isinstance(value,str) and value.endswith(".xml")
                    and re.fullmatch(r"[A-Za-z0-9_.-]+",value), "整机变体路径无效")
            require(not key or key==",".join(sorted(set(key.split(",")))) and set(key.split(","))<=set(ids), "整机变体编号无效")


def bindings(plan, env):
    if "end_effectors" in plan:
        return plan["end_effectors"]
    entries=arms(env)
    tools=[i for i in plan["instances"] if i["placement"]=="robot"]
    return {a["id"]:{"mode":"replace" if tools and len(entries)==1 else "stock",
                    "instance":tools[0]["id"] if tools and len(entries)==1 else None,
                    "reason":"旧单臂输入"} for a in entries}


def validate_bindings(plan, env, *, partial=False):
    entries={a["id"]:a for a in arms(env)}
    value=bindings(plan,env)
    require(isinstance(value,dict) and set(value)==set(entries), "末端选择必须覆盖每只机械臂")
    items={i["id"]:i for i in plan["instances"]}
    used=[]
    removed=[]
    for key,binding in value.items():
        require(isinstance(binding,dict) and set(binding)=={"mode","instance","reason"}, "末端选择字段无效")
        mode=binding["mode"]
        require(mode in ("stock","adapter","replace") and isinstance(binding["reason"],str)
                and 0<len(binding["reason"])<=4000, "末端选择须有明确理由")
        if mode=="stock":
            require(binding["instance"] is None, "原厂末端不能引用生成实例")
            require(entries[key]["stock"] is not None or partial, "裸腕不能冒充原厂夹爪")
        else:
            target=binding["instance"]
            require(isinstance(target,str) and target not in used, "定制末端实例缺失或重复占用")
            require(partial and target not in items or target in items and items[target]["placement"]=="robot", "末端须引用机器人上的实例")
            used.append(target)
            if mode=="adapter":
                require(entries[key]["stock"] is not None, "没有原厂夹爪不能装夹爪适配件")
            elif entries[key]["stock"] is not None:
                removed.append(key)
    require(set(used)&set(items)=={i["id"] for i in items.values() if i["placement"]=="robot"}, "生成末端必须恰好绑定一次")
    key=",".join(sorted(removed))
    require(not key or key in env["robot"].get("model_variants",{}), "整机缺少所选腕部替换变体")
    return value


def model_file(plan, env):
    selected=validate_bindings(plan,env,partial=True)
    removed=[a["id"] for a in arms(env) if a["stock"] and selected[a["id"]]["mode"]=="replace"]
    return env["robot"].get("model_variants",{"":"model.xml"})[",".join(sorted(removed))]


def mount_for(plan, env, instance):
    for arm in arms(env):
        binding=bindings(plan,env)[arm["id"]]
        if binding["instance"]==instance:
            site=arm["stock"]["adapter_site"] if binding["mode"]=="adapter" else arm["wrist_site"]
            return {"arm":arm["id"],"mode":binding["mode"],"site":site,"reason":binding["reason"]}
    raise ValueError("工具没有声明机械臂安装点："+instance)


def validate_posture(value, env):
    require(isinstance(value,dict), "机器人姿态须为关节到数值的映射")
    require(not value or env is not None, "机器人姿态缺少环境约定")
    if value:
        require(all(isinstance(k,str) and k in env["robot"]["joints"] and finite(v)
                    for k,v in value.items()), "机器人姿态引用未知关节或无效数值")


def joint_catalogue(root, env):
    """Compact actual-model contract, never the full robot source in a prompt."""
    import mujoco
    from asset_generation.checks import load_model
    model=load_model(root)
    coupled={int(model.eq_obj1id[e]):int(model.eq_obj2id[e]) for e in range(model.neq)
             if model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e]>=0}
    result={}
    for name in env["robot"]["joints"]:
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        require(j>=0,"配置关节不存在："+name)
        result[name]={"type":"slide" if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                      "range":model.jnt_range[j].tolist() if model.jnt_limited[j] else None,
                      "follows":model.joint(coupled[j]).name if j in coupled else None}
    return result


def validate_model_posture(plan, catalogue):
    for name,q in plan.get("robot_posture",{}).items():
        require(name in catalogue,"姿态引用未知模型关节："+name)
        spec=catalogue[name]
        require(not spec["follows"],"不能独立指定联动从关节："+name)
        require(spec["range"] is None or spec["range"][0]<=q<=spec["range"][1],"机器人姿态越界："+name)


def initialize(model, data, env, prefix="robot__", posture=None):
    """Only position servos receive joint positions. Velocity/effort start at zero."""
    import mujoco
    validate_posture(posture or {},env)
    pose=dict(zip(env["robot"]["joints"],env["robot"]["home"]))
    pose.update(posture or {})
    for name,q in pose.items():
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,prefix+name)
        if jid<0:
            # Only removable stock joints may be absent in a verified variant.
            require(any(a["stock"] and name in a["stock"]["joints"] for a in arms(env)), "机器人关节缺失："+name)
            continue
        require(model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE), "机器人初始化只支持标量关节")
        require(not model.jnt_limited[jid] or model.jnt_range[jid,0]<=q<=model.jnt_range[jid,1], "机器人初始姿态越界："+name)
        data.qpos[model.jnt_qposadr[jid]]=q
    for aid in range(model.nu):
        jid=int(model.actuator_trnid[aid,0])
        if model.actuator_trntype[aid]!=mujoco.mjtTrn.mjTRN_JOINT or jid<0:
            continue
        if not (model.joint(jid).name or "").startswith(prefix):
            continue
        data.ctrl[aid]=0
        if model.actuator_biastype[aid]==mujoco.mjtBias.mjBIAS_AFFINE and model.actuator_biasprm[aid,1]!=0:
            data.ctrl[aid]=data.qpos[model.jnt_qposadr[jid]]*model.actuator_gear[aid,0]
    synchronize_grippers(model,data,prefix)


def synchronize_grippers(model,data,prefix="robot__"):
    """Apply scalar joint equality kinematics for a non-stepping viewer/initial pose."""
    import mujoco
    for eid in range(model.neq):
        if model.eq_type[eid]!=mujoco.mjtEq.mjEQ_JOINT or not data.eq_active[eid]:
            continue
        one,two=int(model.eq_obj1id[eid]),int(model.eq_obj2id[eid])
        if two<0 or any(model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) for j in (one,two)):
            continue
        if not all((model.joint(j).name or "").startswith(prefix) for j in (one,two)):
            continue
        x=data.qpos[model.jnt_qposadr[two]]-model.qpos0[model.jnt_qposadr[two]]
        data.qpos[model.jnt_qposadr[one]]=model.qpos0[model.jnt_qposadr[one]]+sum(c*x**i for i,c in enumerate(model.eq_data[eid,:5]))


POLICY = """
机器人以environment.robot中的公共机构和arms机械臂清单为准。不同手臂共用底盘和升降，不是两个独立机器人。
计划增加end_effectors:{机械臂id:{mode:"stock|adapter|replace",instance:生成末端实例id或null,reason:依据}}，覆盖每只机械臂。
stock保留原厂夹爪，instance=null，不生成假工具任务。adapter保留夹爪，在声明的固定外壳adapter_site装适配总成；
不能假装固定外壳接口会随夹指运动。replace移除对应原厂夹爪，在wrist_site装完整专用末端。一次任务每臂最多一个定制总成。
按需生成0到机械臂数量个tool任务；每个tool实例placement=robot，局部原点对应该选择的安装点，位置为零。
所有定制实例必须恰好绑定一次。优先使用原厂夹爪，不能只因有两只手就生成两套复杂工具。原厂整机不是模型生成资产。
计划及修复可选robot_posture:{实际关节名:位置}指定当前移动与静态展示姿态，默认environment.robot.home。
未给出的关节沿用默认值，必须遵守整机实际限位；夹爪联动从主关节推导，不能独立指定被动指节。
不允许修改机器人源码、官方关节限位、夹爪联动或验收标准。粗略接近性不是双臂协同轨迹证明。
修复可提供完整end_effectors更新安装选择；仅能换装本轮允许修改的工具或tool_additions新增工具。
若原厂夹爪经确认不足，tool_additions:[{job:完整tool任务,instance:完整robot实例}]可按需增加工具，
同时提供end_effectors绑定和明确原因。新增工具须列在jobs修复清单中，共用当前预算，不重新提取资产或增加假仪器。
"""
