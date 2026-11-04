"""Robot description: every name and limit comes from data/robot/profile.json."""
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..geometry import body_geoms, subtree_bodies

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "data/robot/profile.json"


@dataclass
class Arm:
    side: str
    joints: np.ndarray       # joint ids
    qpos: np.ndarray         # qpos addresses
    dofs: np.ndarray         # dof addresses
    actuators: np.ndarray    # torque actuator ids
    kp: np.ndarray
    kd: np.ndarray
    tcp: int                 # site id
    gripper: int             # actuator id
    gripper_open: float
    pads: tuple              # geom ids
    fingers: tuple           # body ids
    home: np.ndarray


class Robot:
    """Binds the profile to a compiled model (the robot alone or inside a scene)."""

    def __init__(self, model, profile=None, prefix=""):
        self.profile = profile or json.loads(PROFILE.read_text(encoding="utf-8"))
        self.model, self.prefix = model, prefix
        p, name = self.profile, lambda n: prefix + n
        self.base_joints = [model.joint(name(j)).id for j in p["base"]["joints"]]
        self.base_actuators = np.array([model.actuator(name(a)).id for a in p["base"]["actuators"]])
        self.lift_joint = model.joint(name(p["lift"]["joint"])).id
        self.lift_actuator = model.actuator(name(p["lift"]["actuator"])).id
        self.root = model.body(name(p["root_body"])).id
        self.arms = {}
        for side, a in p["arms"].items():
            joints = np.array([model.joint(name(j)).id for j in a["joints"]])
            g = a["gripper"]
            self.arms[side] = Arm(side, joints, model.jnt_qposadr[joints], model.jnt_dofadr[joints],
                                  np.array([model.actuator(name(x)).id for x in a["actuators"]]),
                                  np.array(a["gains"]["kp"], float), np.array(a["gains"]["kd"], float),
                                  model.site(name(a["tcp_site"])).id, model.actuator(name(g["actuator"])).id,
                                  g["open"], tuple(model.geom(name(x)).id for x in g["pads"].values()),
                                  tuple(model.body(name(x)).id for x in g["finger_bodies"]), np.array(a["home"], float))
        self.bodies = subtree_bodies(model, self.root)
        self.geoms = body_geoms(model, self.bodies)


def load_robot():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "data/robot/model.xml"))
    return model, Robot(model)
