"""Kinematic inspection of the pinned robot or an assembled scene. No API calls."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from scene_generation.robot_contract import initialize, synchronize_grippers


def main():
    parser=argparse.ArgumentParser(description="FFW-SG2 整机分组关节查看；只更新运动学，不执行实验和动力学")
    parser.add_argument("--scene",type=Path,help="已有场景的 model.xml（模型文件）")
    parser.add_argument("--render",type=Path,help="只输出效果图，随后退出")
    args=parser.parse_args()
    path=args.scene or PROJECT/"data/robots/ffw_sg2/model.xml"
    if args.scene:
        meta=json.loads((path.parent/"scene.json").read_text(encoding="utf-8"))
        env=meta["environment"];prefix="robot__"
        model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
        data.qpos[:]=meta["home_qpos"];data.ctrl[:]=meta["home_ctrl"]
    else:
        env=json.loads((PROJECT/"config/scene_environment_ffw.json").read_text(encoding="utf-8"));prefix=""
        spec=mujoco.MjSpec.from_file(str(path))
        spec.worldbody.add_light(pos=[2,-2,4],dir=[-1,1,-2])
        spec.worldbody.add_geom(name="viewer_floor",type=mujoco.mjtGeom.mjGEOM_PLANE,
                               pos=[0,0,-env["robot"]["mobile"].get("ground_offset_m",0)],
                               size=[3,3,.1],rgba=[.25,.28,.32,1])
        spec.visual.global_.offwidth=960;spec.visual.global_.offheight=960
        model=spec.compile();data=mujoco.MjData(model)
        initialize(model,data,env,prefix="")
    home=data.qpos.copy()
    mounts=meta.get("mounts",[]) if args.scene else []
    mujoco.mj_forward(model,data)
    if args.render:
        from PIL import Image
        camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
        camera.lookat[:]=data.xpos[model.body(prefix+"base_link").id]+[0,0,.65]
        camera.distance=2.8;camera.azimuth=135;camera.elevation=-12
        option=mujoco.MjvOption();option.geomgroup[3:]=0;option.sitegroup[:]=0
        args.render.parent.mkdir(parents=True,exist_ok=True)
        with mujoco.Renderer(model,height=960,width=960) as renderer:
            renderer.update_scene(data,camera,scene_option=option)
            Image.fromarray(renderer.render()).save(args.render)
        print(args.render)
        return
    import tkinter as tk
    from tkinter import ttk
    from mujoco import viewer as mjviewer
    window=tk.Tk();window.title("FFW-SG2 关节查看（运动学）")
    ttk.Label(window,text="滑块改变关节位置；夹爪从关节自动联动。此入口不进行动力学步进。").pack(padx=10,pady=8)
    notebook=ttk.Notebook(window);notebook.pack(fill="both",expand=True)
    joints={model.joint(j).name:j for j in range(model.njnt)}
    followers={int(model.eq_obj1id[e]) for e in range(model.neq) if model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT}
    groups={"底盘":["base_x","base_y","base_yaw"]+[j for j in env["robot"]["shared_joints"] if "wheel" in j],
            "升降及头部":[j for j in env["robot"]["shared_joints"] if "wheel" not in j]}
    for arm in env["robot"]["arms"]:
        label="左臂" if arm["id"]=="left" else "右臂"
        groups[label]=arm["joints"]
        groups[label+"夹爪"]=arm["stock"]["joints"] if arm["stock"] else []
    variables={}
    for group,names in groups.items():
        panel=ttk.Frame(notebook);notebook.add(panel,text=group)
        for row,name in enumerate(names):
            j=joints.get(prefix+name)
            if j is None or j in followers:continue
            if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_SLIDE,mujoco.mjtJoint.mjJNT_HINGE):continue
            lo,hi=model.jnt_range[j] if model.jnt_limited[j] else (-3.14,3.14)
            value=tk.DoubleVar(value=float(data.qpos[model.jnt_qposadr[j]]))
            variables[j]=value
            ttk.Label(panel,text=name).grid(row=row,column=0,sticky="w")
            tk.Scale(panel,variable=value,from_=float(lo),to=float(hi),resolution=.001,
                     orient="horizontal",length=340).grid(row=row,column=1)
    def reset():
        for j,var in variables.items():var.set(float(home[model.jnt_qposadr[j]]))
    ttk.Button(window,text="恢复场景初始姿态",command=reset).pack(pady=8)
    with mjviewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:]=[0,0,.8];viewer.cam.distance=2.8
        viewer.cam.azimuth=135;viewer.cam.elevation=-12
        viewer.opt.geomgroup[3:]=0;viewer.opt.sitegroup[:]=0
        def tick():
            if not viewer.is_running():
                window.destroy();return
            with viewer.lock():
                for j,var in variables.items():
                    data.qpos[model.jnt_qposadr[j]]=var.get()
                synchronize_grippers(model,data,prefix)
                if mounts:
                    from scene_generation.tool_contract import align_free_roots
                    align_free_roots(model,data,mounts)
                mujoco.mj_forward(model,data)
            viewer.sync()
            window.after(30,tick)
        window.after(30,tick)
        window.mainloop()


if __name__=="__main__":
    main()
