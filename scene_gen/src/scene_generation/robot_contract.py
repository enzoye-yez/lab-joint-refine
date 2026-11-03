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
    require(isinstance(robot,dict), "Invalid robot configuration")
    common={"name","directory","joints","home"}
    extra={"mobile"}
    required=(common|{"arms","shared_joints","model_variants"} if "arms" in robot else
              common|{"attachment_site","payload_kg","reach_m"})
    require(required <= set(robot) <= required|extra, "Invalid robot configuration fields")
    require(all(isinstance(robot[k],str) and robot[k] for k in ("name","directory")), "Robot source missing")
    joints=robot["joints"]
    require(isinstance(joints,list) and joints and all(isinstance(j,str) and j for j in joints)
            and len(joints)==len(set(joints)), "Invalid robot joints")
    require(isinstance(robot["home"],list) and len(robot["home"])==len(joints)
            and all(finite(q) for q in robot["home"]), "Invalid robot initial posture")
    entries=arms({"robot":robot})
    require(isinstance(entries,list) and entries, "Declare at least one robot arm")
    ids=[]
    grouped=list(robot.get("shared_joints",[]))
    for arm in entries:
        require(isinstance(arm,dict) and set(arm)=={"id","joints","wrist_site","tcp_site","shoulder_site","reach_m","payload_kg","stock"},
                "Invalid robot arm fields")
        require(isinstance(arm["id"],str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}",arm["id"]), "Invalid robot arm id")
        ids.append(arm["id"])
        require(isinstance(arm["joints"],list) and arm["joints"] and all(isinstance(j,str) for j in arm["joints"]), "Invalid robot arm joint group")
        require(all(isinstance(arm[k],str) and arm[k] for k in ("wrist_site","tcp_site")), "Robot arm interface missing")
        require(arm["shoulder_site"] is None or isinstance(arm["shoulder_site"],str), "Invalid shoulder interface")
        require(all(finite(arm[k]) and arm[k]>0 for k in ("reach_m","payload_kg")), "Invalid robot arm specification")
        grouped.extend(arm["joints"])
        stock=arm["stock"]
        if stock is not None:
            require(isinstance(stock,dict) and set(stock)=={"name","joints","adapter_site","mass_kg","mass_source"}, "Invalid stock end-effector fields")
            require(all(isinstance(stock[k],str) and stock[k] for k in ("name","adapter_site","mass_source")), "Stock end-effector source or interface missing")
            require(isinstance(stock["joints"],list) and all(isinstance(j,str) for j in stock["joints"])
                    and finite(stock["mass_kg"]) and stock["mass_kg"]>0, "Invalid stock end-effector joints or mass")
            grouped.extend(stock["joints"])
    require(len(ids)==len(set(ids)), "Duplicate robot arm id")
    if "arms" in robot:
        require(len(grouped)==len(set(grouped)) and set(grouped)==set(joints), "Shared, arm and gripper joints must not overlap and must cover the initial joints")
        variants=robot["model_variants"]
        require(isinstance(variants,dict) and variants.get("")=="model.xml", "Default whole-robot variant missing")
        for key,value in variants.items():
            require(isinstance(key,str) and isinstance(value,str) and value.endswith(".xml")
                    and re.fullmatch(r"[A-Za-z0-9_.-]+",value), "Invalid whole-robot variant path")
            require(not key or key==",".join(sorted(set(key.split(",")))) and set(key.split(","))<=set(ids), "Invalid whole-robot variant id")


def bindings(plan, env):
    if "end_effectors" in plan:
        return plan["end_effectors"]
    entries=arms(env)
    tools=[i for i in plan["instances"] if i["placement"]=="robot"]
    return {a["id"]:{"mode":"replace" if tools and len(entries)==1 else "stock",
                    "instance":tools[0]["id"] if tools and len(entries)==1 else None,
                    "reason":"legacy single-arm input"} for a in entries}


def validate_bindings(plan, env, *, partial=False):
    entries={a["id"]:a for a in arms(env)}
    value=bindings(plan,env)
    require(isinstance(value,dict) and set(value)==set(entries), "End-effector selection must cover every robot arm")
    items={i["id"]:i for i in plan["instances"]}
    used=[]
    removed=[]
    for key,binding in value.items():
        require(isinstance(binding,dict) and set(binding)=={"mode","instance","reason"}, "Invalid end-effector selection fields")
        mode=binding["mode"]
        require(mode in ("stock","adapter","replace") and isinstance(binding["reason"],str)
                and 0<len(binding["reason"])<=4000, "End-effector selection requires an explicit reason")
        if mode=="stock":
            require(binding["instance"] is None, "A stock end-effector cannot reference a generated instance")
            require(entries[key]["stock"] is not None or partial, "A bare wrist cannot pose as a stock gripper")
        else:
            target=binding["instance"]
            require(isinstance(target,str) and target not in used, "Custom end-effector instance missing or already bound")
            require(partial and target not in items or target in items and items[target]["placement"]=="robot", "The end-effector must reference an instance placed on the robot")
            used.append(target)
            if mode=="adapter":
                require(entries[key]["stock"] is not None, "A gripper adapter cannot be mounted without a stock gripper")
            elif entries[key]["stock"] is not None:
                removed.append(key)
    require(set(used)&set(items)=={i["id"] for i in items.values() if i["placement"]=="robot"}, "Each generated end-effector must be bound exactly once")
    key=",".join(sorted(removed))
    require(not key or key in env["robot"].get("model_variants",{}), "The whole robot lacks the selected wrist replacement variant")
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
    raise ValueError("Tool declares no robot arm mount point: "+instance)


def validate_posture(value, env):
    require(isinstance(value,dict), "Robot posture must map joints to numbers")
    require(not value or env is not None, "Robot posture lacks an environment contract")
    if value:
        require(all(isinstance(k,str) and k in env["robot"]["joints"] and finite(v)
                    for k,v in value.items()), "Robot posture references an unknown joint or an invalid value")


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
        require(j>=0,"Configured joint does not exist: "+name)
        result[name]={"type":"slide" if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                      "range":model.jnt_range[j].tolist() if model.jnt_limited[j] else None,
                      "follows":model.joint(coupled[j]).name if j in coupled else None}
    return result


def validate_model_posture(plan, catalogue):
    for name,q in plan.get("robot_posture",{}).items():
        require(name in catalogue,"Posture references an unknown model joint: "+name)
        spec=catalogue[name]
        require(not spec["follows"],"A coupled follower joint cannot be set independently: "+name)
        require(spec["range"] is None or spec["range"][0]<=q<=spec["range"][1],"Robot posture out of range: "+name)


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
            require(any(a["stock"] and name in a["stock"]["joints"] for a in arms(env)), "Robot joint missing: "+name)
            continue
        require(model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE), "Robot initialization supports only scalar joints")
        require(not model.jnt_limited[jid] or model.jnt_range[jid,0]<=q<=model.jnt_range[jid,1], "Robot initial posture out of range: "+name)
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
The robot is defined by the shared mechanisms and the arms list in environment.robot. Different arms share the base and lift; they are not independent robots.
The plan adds end_effectors:{arm id:{mode:"stock|adapter|replace",instance:generated end-effector instance id or null,reason:justification}}, covering every arm.
stock keeps the stock gripper, instance=null, and generates no fake tool job. adapter keeps the gripper and mounts an adapter assembly on the declared fixed-housing adapter_site;
do not pretend the fixed-housing interface moves with the fingers. replace removes that arm's stock gripper and mounts a complete dedicated end-effector on wrist_site. At most one custom assembly per arm per task.
Generate 0 up to the number of arms tool jobs as needed; each tool instance has placement=robot, its local origin at the selected mount point, position zero.
Every custom instance must be bound exactly once. Prefer stock grippers; do not generate two complex tools merely because there are two hands. The stock whole robot is not a model-generated asset.
Plans and repairs may give robot_posture:{actual joint name:position} for the current mobile and static display posture; default environment.robot.home.
Unlisted joints keep their defaults and must respect the whole robot's actual limits; gripper coupling is derived from the driving joint, and passive finger joints cannot be set independently.
Do not modify the robot source, official joint limits, gripper coupling or acceptance criteria. Rough proximity is not proof of a coordinated dual-arm trajectory.
A repair may provide a complete end_effectors to update mount choices; it may only swap tools allowed to change this round or tools added via tool_additions.
If stock grippers are confirmed insufficient, tool_additions:[{job:complete tool job,instance:complete robot instance}] may add tools as needed,
together with the end_effectors binding and an explicit reason. Added tools must be listed in the jobs repair list and share the current budget; do not re-extract assets or add fake instruments.
"""
