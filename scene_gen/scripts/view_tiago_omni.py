"""Interactive kinematic inspection, without running a protocol or dynamics."""
import argparse
from pathlib import Path
import time
import mujoco
import mujoco.viewer

parser=argparse.ArgumentParser(description="TIAGo OMNI whole-robot joint and appearance viewer; runs no experiment or dynamics")
parser.add_argument("--flange",action="store_true",help="view the gripperless wrist configuration used for production assembly")
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
