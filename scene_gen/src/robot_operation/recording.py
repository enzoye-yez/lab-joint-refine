"""流式证据和物理命令重放；图像缺失不能伪装为完整数据集。"""
from dataclasses import asdict
from pathlib import Path
import json
import gzip
import platform
import sys
import mujoco
import numpy as np
from PIL import Image
from asset_generation.runtime import InteractionRuntime
from .common import atomic_json, digest, identity, need, read
from .evidence import Evidence, contacts
from .scene import load
from scene_generation.runtime import actor_groups


class Recorder:
    def __init__(self, root, mode="final", images=True):
        self.root=Path(root); self.mode=mode; self.images=images
        self.trace=gzip.open(self.root/"steps.jsonl.gz","wt",encoding="utf-8",compresslevel=3)
        self.events=(self.root/"events.jsonl").open("w",encoding="utf-8")
        self.renderer=None; self.next_image=0.; self.frame=0; self.image_error=None
        self.closed=False

    def start(self,session,scene_hash,task,graph,scene_path):
        from asset_prep.implementation import package_manifest
        base=Path(__file__).resolve().parents[1]
        pinned=getattr(sys,"_harness_runtime",None)
        files=pinned["manifest"] if pinned else package_manifest(base)
        implementation={"manifest":files,"implementation":digest(files)}
        for name in files:
            target=self.root/"implementation_sources"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(pinned["sources"][name] if pinned else (base/name).read_bytes())
        atomic_json(self.root/"initial.json",session.snapshot())
        atomic_json(self.root/"task.json",task); atomic_json(self.root/"graph.json",graph)
        m=session.model
        self.robot_joint_ids=[j for j in range(m.njnt) if m.joint(j).name.startswith("robot__") and m.jnt_type[j] in (2,3)]
        self.metadata={"version":1,"mode":self.mode,"provenance":graph["provenance"],"scene_sha256":scene_hash,
            "scene_path":str(Path(scene_path).resolve()),"task_sha256":digest(task),"graph_sha256":digest(graph),
            "mujoco_version":mujoco.__version__,"python_version":platform.python_version(),
            "implementation_sha256":implementation["implementation"],"implementation_files":implementation["manifest"],
            "timestep":float(m.opt.timestep),"integrator":int(m.opt.integrator),"solver":int(m.opt.solver),
            "iterations":int(m.opt.iterations),"tolerance":float(m.opt.tolerance),"settings":asdict(session.settings),
            "observation_fields":["images","robot_joint_positions","robot_joint_velocities"],
            "privileged_fields":["physics_state","body_poses","contacts","device_state"],
            "geom_names":[m.geom(i).name for i in range(m.ngeom)],"body_names":[m.body(i).name for i in range(m.nbody)],
            "joint_names":[m.joint(i).name for i in range(m.njnt)],"actuator_names":[m.actuator(i).name for i in range(m.nu)],
            "observation_joint_names":[m.joint(i).name for i in self.robot_joint_ids],
            "images_requested":self.images,"image_error":None}
        atomic_json(self.root/"recording.json",self.metadata)

    def event(self,row):
        self.events.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n"); self.events.flush()

    def capture_images(self,session):
        if not self.images or self.image_error: return
        try: self.render(session)
        except Exception as exc:
            self.image_error=f"{type(exc).__name__}: {exc}"
            self.event({"type":"image_error","detail":self.image_error,"time":float(session.data.time)})
        self.next_image=session.data.time+session.settings.image_period

    def step(self,session,before,controls,software,logical_extra):
        d,m=session.data,session.model
        state=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        row={"step":session.sim_steps,"node":session.node,"time_before":before,"time":float(d.time),
             "logical_time":session.logical_time,"control":controls.tolist(),"software_requests":software,
             "logical_extra":logical_extra,"state":state.tolist(),"qpos":d.qpos.tolist(),"qvel":d.qvel.tolist(),
             "body_position":d.xpos.tolist(),"body_quaternion":d.xquat.tolist(),
             "contacts":session.evidence.contact_rows,"events":session.last_events,"eq_active":d.eq_active.tolist(),
             "devices":session.evidence.devices}
        row["condition_evidence"]={key:value for key,value in session.evidence.last.items()
            if value.get("time")==float(d.time) and (not value.get("passed") or value.get("condition",{}).get("kind")=="held")}
        row["observation"]={"joint_position":d.qpos[m.jnt_qposadr[self.robot_joint_ids]].tolist(),
                            "joint_velocity":d.qvel[m.jnt_dofadr[self.robot_joint_ids]].tolist()}
        self.trace.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
        if session.sim_steps%100==0: self.trace.flush()
        if self.images and not self.image_error and d.time+1e-10>=self.next_image:
            self.capture_images(session)

    def render(self,session):
        if self.renderer is None:
            self.renderer=mujoco.Renderer(session.model,height=240,width=320)
            (self.root/"images").mkdir(exist_ok=True)
        # 同一仿真时刻分别渲染全局及左右工具近景，记录自由相机实际外参。
        cameras=[("overview",None)]+[(arm,site) for arm,site in session.tcps.items()]
        for label,site in cameras:
            camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
            camera.lookat[:]=session.model.stat.center if site is None else session.data.site_xpos[session.model.site(site).id]
            camera.distance=max(1.,float(session.model.stat.extent)*1.3) if site is None else .35
            camera.azimuth=120; camera.elevation=-35
            self.renderer.update_scene(session.data,camera=camera)
            pixels=self.renderer.render()
            name=f"{self.frame:07d}_{label}.png"
            Image.fromarray(pixels).save(self.root/"images"/name)
            optical=[]
            for cam in self.renderer.scene.camera:
                optical.append({"position":cam.pos.tolist(),"forward":cam.forward.tolist(),"up":cam.up.tolist(),
                    "frustum_near":float(cam.frustum_near),"frustum_far":float(cam.frustum_far),
                    "frustum_top":float(cam.frustum_top),"frustum_bottom":float(cam.frustum_bottom),
                    "frustum_center":float(cam.frustum_center),"frustum_width":float(cam.frustum_width)})
            self.event({"type":"image","file":"images/"+name,"time":float(session.data.time),
                "logical_time":session.logical_time,"node":session.node,"camera":label,"width":320,"height":240,"optical":optical})
        self.frame+=1

    def close(self,result):
        if self.closed: return
        self.closed=True
        self.trace.close(); self.events.close()
        if self.renderer: self.renderer.close()
        self.metadata.update(image_error=self.image_error,frames=self.frame,
            dataset_complete=bool(self.mode=="final" and self.images and not self.image_error and self.frame>0 and result["status"]=="success"))
        atomic_json(self.root/"recording.json",self.metadata)
        atomic_json(self.root/"result.json",result)
        from asset_generation.pipeline import snapshot
        atomic_json(self.root/"files.json",snapshot(self.root))


def replay(run,scene=None,tolerance=1e-8):
    root=Path(run); metadata=read(root/"recording.json")
    from asset_prep.implementation import package_manifest
    need(digest(package_manifest(Path(__file__).resolve().parents[1]))==metadata["implementation_sha256"],"重放代码版本不同")
    need(mujoco.__version__==metadata["mujoco_version"],"重放仿真器版本不同")
    from asset_prep.storage import file_hash, safe_path
    for name,expected in read(root/"files.json").items():
        need(file_hash(safe_path(root,name))==expected,"运行证据被修改")
    m,d,meta,manifest=load(scene or metadata["scene_path"],metadata["scene_sha256"])
    state=read(root/"initial.json")
    mujoco.mj_setState(m,d,np.asarray(state["physics"]),mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m,d)
    task=read(root/"task.json")
    from .common import Settings
    actors=actor_groups(m,meta)
    evidence=Evidence(m,d,actors,task,Settings(**metadata["settings"]))
    evidence.restore(state["evidence"])
    interaction=InteractionRuntime(m,manifest,actor_geom_ids=sorted({g for gs in actors.values() for g in gs}))
    interaction.pressed=state["interaction"]["pressed"]; interaction.released=set(state["interaction"]["released"])
    interfaces={device["interface"]:device["id"] for device in task.get("devices",[])}
    maximum=0.; count=0
    with gzip.open(root/"steps.jsonl.gz","rt",encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line); count+=1
            need(abs(d.time-row["time_before"])<=tolerance,"记录不是连续时间序列")
            for request in row["software_requests"]:
                need(request in interfaces,"重放包含无设备门控的软件请求")
                evidence.start(interfaces[request])
            d.ctrl[:]=row["control"]
            mujoco.mj_step(m,d); mujoco.mj_forward(m,d)
            mechanical=evidence.update(m.opt.timestep,row["logical_extra"])
            events=interaction.update(d,row["software_requests"],verified_mechanical_events=mechanical)
            state_vector=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(m,d,state_vector,mujoco.mjtState.mjSTATE_INTEGRATION)
            delta=max(float(np.max(np.abs(d.qpos-np.array(row["qpos"])),initial=0)),
                      float(np.max(np.abs(d.qvel-np.array(row["qvel"])),initial=0)))
            maximum=max(maximum,delta)
            need(delta<=tolerance,"命令重放状态不一致","replay_mismatch")
            need(float(np.max(np.abs(state_vector-np.array(row["state"])),initial=0))<=tolerance,"完整积分状态重放不一致","replay_mismatch")
            need(d.eq_active.tolist()==row["eq_active"] and events==row["events"],"连接或事件重放不一致","replay_mismatch")
            need(evidence.devices==row["devices"],"设备状态重放不一致","replay_mismatch")
    return {"status":"pass","steps":count,"max_state_error":maximum,
            "coverage":"固定代码、仿真器和场景下的命令重放；不替代任务评分或真实硬件验证"}
