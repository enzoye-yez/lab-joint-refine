"""Interactive kinematic inspection, without running a protocol or dynamics."""
import argparse
from pathlib import Path
import time
import mujoco
import mujoco.viewer

parser=argparse.ArgumentParser(description="TIAGo OMNI 整机关节与外观查看；不执行实验或动力学")
parser.add_argument("--flange",action="store_true",help="查看生产装配用的无夹爪腕部配置")
args=parser.parse_args()
project=Path(__file__).resolve().parents[1]
path=project/"data/robots/tiago_omni"/("model.xml" if args.flange else "stock_gripper.xml")
model=mujoco.MjModel.from_xml_path(str(path))
data=mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,0)
mujoco.mj_forward(model,data)
with mujoco.viewer.launch_passive(model,data) as viewer:
    viewer.cam.lookat[:]=[0,0,.7]
    viewer.cam.distance=2.6
    viewer.cam.azimuth=135
    viewer.cam.elevation=-15
    viewer.opt.geomgroup[3:]=0
    viewer.opt.sitegroup[:]=0
    while viewer.is_running():
        with viewer.lock():
            mujoco.mj_forward(model,data)
        viewer.sync()
        time.sleep(.02)
