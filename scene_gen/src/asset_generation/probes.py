"""框架编写的短接触探测，不运行模型提供的验证程序。"""

from pathlib import Path
import numpy as np
import mujoco

from .runtime import InteractionRuntime
from .contracts import require


# 压头闭环推进：按钮到达行程末端（REACHED）即停，最多压过剩余行程 PRESS_OVERTRAVEL 以抵消
# 接触压缩；继续压只会把按钮推出软限位、制造假读数。
PRESS_OVERTRAVEL = .003
REACHED = .98
SETTLE_STEPS_SECONDS = .3


def _front_depth(model, data, g, axis):
    """受压几何沿 -axis 方向最外表面到几何中心的距离（按 geom_aabb，适用于所有几何类型）。"""
    rotation = data.geom_xmat[g].reshape(3, 3)
    center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    local = rotation.T@axis
    return float(-local@center+np.abs(local)@half)


def _probe_model(root, model_file, joint):
    """资产加一个独立运动学压头；按钮所在部件的自由根被固定在原位，压头压的是按钮而不是坠落中的仪器。"""
    spec = mujoco.MjSpec.from_file(str(Path(root)/model_file))
    require(spec.geom("harness_probe") is None and spec.body("harness_probe_body") is None,"资产使用保留探测名称")
    plain = spec.compile()
    root_body = plain.body(int(plain.body_rootid[plain.jnt_bodyid[plain.joint(joint).id]])).name
    for item in list(spec.joints):
        if item.type == mujoco.mjtJoint.mjJNT_FREE and item.parent.name == root_body:
            spec.delete(item)
    actor = spec.worldbody.add_body(name="harness_probe_body",mocap=True,pos=[0,0,10])
    actor.add_geom(name="harness_probe",type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.003,0,0],mass=.01,rgba=[1,0,0,1])
    return spec.compile(), root_body


def interaction_probes(root, manifest, model_file="model.xml"):
    """使用独立运动学压头推动实际按钮；不直接写按钮关节或冒充机械臂执行。

    model_file 是被检查的构型文件；多构型资产的按钮只存在于各自的构型模型中。
    按钮先在重力下静置得到静止位置；压头推进到行程 98% 且事件触发即停（最多压过行程末端 3 mm）；撤回后再静置，
    以静止位置判断复位。静止位置已越过 released_at 说明弹簧撑不住按钮自重，是真缺陷。"""
    rows = []
    require(len(manifest["buttons"]) <= 32,"按钮探测数超限")
    released_connections = set()
    for button in manifest["buttons"]:
        model, root_body = _probe_model(root, model_file, button["joint"])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        j = model.joint(button["joint"]).id
        g = model.geom(button["geom"]).id
        q = model.jnt_qposadr[j]
        settle = int(SETTLE_STEPS_SECONDS/model.opt.timestep)
        for _ in range(settle):
            mujoco.mj_step(model,data)
        rest = float(data.qpos[q])
        radius = float(model.geom_size[model.geom("harness_probe").id,0])
        axis = data.xaxis[j].copy()
        surface = data.geom_xpos[g]-axis*_front_depth(model,data,g,axis)
        stroke = float(model.jnt_range[j,1])-rest
        start = surface-axis*(radius+.005)
        finish = surface+axis*(stroke+PRESS_OVERTRAVEL-radius)
        mocap = model.body("harness_probe_body").mocapid[0]
        runtime = InteractionRuntime(model,manifest,["harness_probe"])
        records, peak = [], rest
        top = float(model.jnt_range[j,0])+REACHED*float(model.jnt_range[j,1]-model.jnt_range[j,0])
        for position in np.linspace(start,finish,360):
            data.mocap_pos[mocap] = position
            mujoco.mj_step(model,data)
            records.extend(runtime.update(data))
            peak = max(peak,float(data.qpos[q]))
            # Stop once the button is at the end of travel and has fired; otherwise press on to the cap.
            if peak >= top and any(r.get("name") == button["event"] for r in records):
                break
        finish = position
        event = any(r.get("name") == button["event"] for r in records)
        released_connections.update(r["id"] for r in records if r["type"] == "connection_released")
        for position in np.linspace(finish,start,120):
            data.mocap_pos[mocap] = position
            mujoco.mj_step(model,data)
            runtime.update(data)
        data.mocap_pos[mocap] = [0,0,10]
        for _ in range(settle):
            mujoco.mj_step(model,data)
            runtime.update(data)
        final = float(data.qpos[q])
        stable = np.isfinite(data.qpos).all() and not any(w.number for w in data.warning)
        span = float(model.jnt_range[j,1]-model.jnt_range[j,0])
        returned = abs(final-rest) <= max(.1*span, 2e-4)
        sag_ok = rest < button["released_at"]
        released = not runtime.pressed[button["id"]]
        failed = [name for name, ok in (
            (f"重力下静止位置 {rest*1000:.2f} mm 已越过 released_at {button['released_at']*1000:.2f} mm（弹簧撑不住按钮自重或 released_at 过低）", sag_ok),
            ("未触发事件", event), ("压头撤回后未回到静止位置", returned),
            ("压头撤回后按钮状态未复位", released), ("数值不稳定", stable)) if not ok]
        rows.append({"id":"button_probe:"+button["id"],"status":"fail" if failed else "pass",
                     "detail":("；".join(failed)+"。" if failed else "")+
                              f"按钮所在部件根 {root_body} 固定；独立压头推进到按钮行程末端即停（最多压过 {PRESS_OVERTRAVEL*1000:g} mm）并撤回；未直接写按钮关节，不认证机器人按压轨迹",
                     "rest_position":rest,"peak_position":peak,"final_position":final,"stroke":span,
                     "triggered":event,"released_after_retreat":released,"events":records[:50]})
    # 明确声明的软件控制连接可验证释放逻辑；不能把物理接口伪装成软件事件。
    software = {i["id"] for i in manifest["interfaces"] if i["kind"] == "software"}
    if software:
        model = mujoco.MjModel.from_xml_path(str(Path(root)/model_file))
        data = mujoco.MjData(model)
        runtime = InteractionRuntime(model,manifest)
        events = runtime.update(data,sorted(software))
        released_connections.update(r["id"] for r in events if r["type"] == "connection_released")
    mechanical={e["id"] for e in manifest.get("mechanical_events",[])}
    for connection in manifest["connections"]:
        if connection["release_event"] in mechanical:
            rows.append({"id":"release_probe:"+connection["id"],"status":"partial","complete":False,
                         "detail":"机械动作仅声明真实接口；尚无对应操作探测，未认证释放，不能据此判定几何错误"})
            continue
        rows.append({"id":"release_probe:"+connection["id"],
                     "status":"pass" if connection["id"] in released_connections else "fail",
                     "detail":"已声明触发事件确实使连接等式约束失活；不证明分离路径或重新锁合"})
    return rows
