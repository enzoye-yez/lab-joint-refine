"""Robot description: every name and limit comes from data/robot/profile.json."""
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..geometry import body_geoms, subtree_bodies

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "data/robot/profile.json"
CAPABILITY = ROOT / "data/robot/capability.json"


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
    pin_actuator: int = -1   # retractable press pin (if the tool has one)
    pin_qpos: int = -1
    pin_extended: float = 0.0
    pin_tip: np.ndarray = None   # pin tip relative to the TCP when extended, tool frame
    blade_actuator: int = -1     # retractable pry blade
    blade_qpos: int = -1
    blade_extended: float = 0.0
    blade_tip: np.ndarray = None
    suction: dict = None         # cup body/site/equality ids, face offset and normal in the tool frame


class Robot:
    """Binds the profile to a compiled model (the robot alone or inside a scene)."""

    def __init__(self, model, profile=None, prefix="", capability=None):
        self.profile = profile or json.loads(PROFILE.read_text(encoding="utf-8"))
        self.capability = capability or load_capability()
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
            pin = g.get("press_pin")
            if pin:
                arm = self.arms[side]
                arm.pin_actuator = model.actuator(name(pin["actuator"])).id
                arm.pin_qpos = int(model.jnt_qposadr[model.joint(name(pin["joint"])).id])
                arm.pin_extended, arm.pin_tip = pin["extended"], np.array(pin["tip_from_tcp_m"], float)
            blade = g.get("pry_blade")
            if blade:
                arm = self.arms[side]
                arm.blade_actuator = model.actuator(name(blade["actuator"])).id
                arm.blade_qpos = int(model.jnt_qposadr[model.joint(name(blade["joint"])).id])
                arm.blade_extended, arm.blade_tip = blade["extended"], np.array(blade["tip_from_tcp_m"], float)
            cup = g.get("suction")
            if cup:
                self.arms[side].suction = {
                    "body": model.body(name(cup["body"])).id, "geom": model.geom(name(cup["geom"])).id,
                    "equality": model.equality(name(cup["equality"])).id, "max_mass_kg": cup["max_mass_kg"],
                    "face": np.array(cup["face_from_tcp_m"], float), "normal": np.array(cup["normal"], float),
                    "radius": cup["radius_m"]}
        self.bodies = subtree_bodies(model, self.root)
        self.geoms = body_geoms(model, self.bodies)
        self.base_qpos = model.jnt_qposadr[self.base_joints]
        self.lift_qpos = int(model.jnt_qposadr[self.lift_joint])
        # The planar joints live in the frame the robot was attached with (its home pose in the room).
        self.frame_position = model.body_pos[self.root][:2].copy()
        q = model.body_quat[self.root]
        self.frame_yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def _rotate(self, xy, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])

    def base_pose(self, qpos):
        """World (x, y, yaw) of the base for a qpos vector."""
        x, y, yaw = qpos[self.base_qpos]
        return np.array([*(self.frame_position + self._rotate((x, y), self.frame_yaw)), self.frame_yaw + yaw])

    def base_joint_values(self, pose):
        """Planar joint values that put the base at world (x, y, yaw)."""
        local = self._rotate(np.asarray(pose[:2]) - self.frame_position, -self.frame_yaw)
        return np.array([local[0], local[1], pose[2] - self.frame_yaw])

    def base_command(self, data, velocity_xy, yaw_rate):
        """World-frame base velocity command."""
        local = self._rotate(velocity_xy, -self.frame_yaw)
        data.ctrl[self.base_actuators] = [local[0], local[1], yaw_rate]


def load_capability():
    """The frozen capability profile (forces, payload, reach) the task code and the repair requirements use."""
    return json.loads(CAPABILITY.read_text(encoding="utf-8"))


def load_robot():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "data/robot/model.xml"))
    return model, Robot(model)
