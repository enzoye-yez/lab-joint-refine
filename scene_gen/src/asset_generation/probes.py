"""框架编写的短接触探测，不运行模型提供的验证程序。"""

from pathlib import Path
import numpy as np
import mujoco

from .runtime import InteractionRuntime
from .contracts import require


def interaction_probes(root, manifest):
    """使用独立运动学压头推动实际按钮；不直接写按钮关节或冒充机械臂执行。"""
    rows = []
    require(len(manifest["buttons"]) <= 32,"按钮探测数超限")
    released_connections = set()
    for button in manifest["buttons"]:
        spec = mujoco.MjSpec.from_file(str(Path(root)/"model.xml"))
        require(spec.geom("harness_probe") is None and spec.body("harness_probe_body") is None,"资产使用保留探测名称")
        actor = spec.worldbody.add_body(name="harness_probe_body",mocap=True,pos=[0,0,10])
        radius = .003
        actor.add_geom(name="harness_probe",type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[radius,0,0],
                       mass=.01,rgba=[1,0,0,1])
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        j = model.joint(button["joint"]).id
        g = model.geom(button["geom"]).id
        axis = data.xaxis[j].copy()
        center = data.geom_xpos[g].copy()
        start = center-axis*(model.geom_rbound[g]+radius+.005)
        finish = center+axis*(model.jnt_range[j,1]-data.qpos[model.jnt_qposadr[j]]+radius)
        mocap = model.body("harness_probe_body").mocapid[0]
        runtime = InteractionRuntime(model,manifest,["harness_probe"])
        records, peak = [], float(data.qpos[model.jnt_qposadr[j]])
        for position in np.linspace(start,finish,240):
            data.mocap_pos[mocap] = position
            mujoco.mj_step(model,data)
            records.extend(runtime.update(data))
            peak = max(peak,float(data.qpos[model.jnt_qposadr[j]]))
        event = any(r.get("name") == button["event"] for r in records)
        released_connections.update(r["id"] for r in records if r["type"] == "connection_released")
        for position in np.linspace(finish,start,120):
            data.mocap_pos[mocap] = position
            mujoco.mj_step(model,data)
            runtime.update(data)
        stable = np.isfinite(data.qpos).all() and not any(w.number for w in data.warning)
        rows.append({"id":"button_probe:"+button["id"],"status":"pass" if event and stable else "fail",
                     "detail":"独立压头产生真实接触并推动按钮；未直接写按钮关节，不认证机器人按压轨迹",
                     "peak_position":peak,"triggered":event,"released_after_retreat":not runtime.pressed[button["id"]],
                     "events":records[:50]})
    # 明确声明的软件控制连接可验证释放逻辑；不能把物理接口伪装成软件事件。
    software = {i["id"] for i in manifest["interfaces"] if i["kind"] == "software"}
    if software:
        model = mujoco.MjModel.from_xml_path(str(Path(root)/"model.xml"))
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
