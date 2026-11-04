"""Assemble the NP-002 robot: holonomic base, lift column, two Franka FR3 arms and parallel grippers.

Writes data/robot/model.xml and data/robot/profile.json. The profile is the only source of
names and limits for the rest of the package; nothing downstream hard-codes robot names.
Run: .venv/bin/python scripts/build_robot.py
"""
import json
import os
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FR3 = ROOT / "data/vendor/fr3/model.xml"
OUT = ROOT / "data/robot"

BASE_HALF = (0.30, 0.25, 0.15)      # m, box half extents
BASE_CLEARANCE = 0.01               # m, gap to the floor; the planar joints carry the base
LIFT_RANGE = (0.0, 0.60)            # m
PLATFORM_Z = 0.45                   # m, platform top above the floor at lift 0
SHOULDER_Y = 0.22                   # m, lateral offset of each arm mount
SHOULDER_X = 0.08                   # m, forward offset of the arm mounts
FINGER_STROKE = 0.04                # m per finger, 0-80 mm total opening
GRIP_FORCE = 70.0                   # N per finger, actuator force limit
PAD_FRICTION = (0.9, 0.02, 0.001)   # TPU pads
ARM_GAINS = {"kp": [600, 600, 600, 600, 300, 300, 200], "kd": [50, 50, 50, 40, 20, 20, 10]}


def fr3_arm(prefix):
    spec = mujoco.MjSpec.from_file(str(FR3))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def gripper(spec, flange, side):
    """Temporary parallel gripper on the FR3 flange; replaced by the generated tool later."""
    hand = flange.add_body(name=f"{side}_hand", pos=[0, 0, 0])
    hand.add_geom(name=f"{side}_hand_housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.09, 0.03],
                  pos=[0, 0, 0.03], mass=0.55, rgba=[0.25, 0.25, 0.28, 1])
    hand.add_geom(name=f"{side}_wrist_camera", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.02, 0.012],
                  pos=[0.045, 0, 0.03], mass=0.05, rgba=[0.1, 0.1, 0.1, 1])
    hand.add_camera(name=f"{side}_wrist_camera", pos=[0.06, 0, 0.03], xyaxes=[0, -1, 0, 0.8, 0, 0.6])
    joints = []
    for name, sign in (("a", 1), ("b", -1)):
        finger = hand.add_body(name=f"{side}_finger_{name}", pos=[0, 0, 0.06])
        joint = finger.add_joint(name=f"{side}_finger_{name}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                                 axis=[0, sign, 0], range=[0, FINGER_STROKE], armature=0.01, damping=5.0)
        joints.append(joint)
        finger.add_geom(name=f"{side}_finger_{name}_link", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.006, 0.025],
                        pos=[0, sign * 0.010, 0.025], mass=0.04, rgba=[0.6, 0.6, 0.62, 1])
        finger.add_geom(name=f"{side}_pad_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.011, 0.002, 0.012],
                        pos=[0, sign * 0.002, 0.038], mass=0.005, friction=list(PAD_FRICTION), condim=4,
                        rgba=[0.15, 0.4, 0.8, 1])
    # Closed (q=0) the pads meet at y=0; each finger opens outwards by q.
    spec.add_exclude(bodyname1=f"{side}_finger_a", bodyname2=f"{side}_finger_b")
    hand.add_site(name=f"{side}_tcp", pos=[0, 0, 0.098], size=[0.004, 0, 0])
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f"{side}_finger_a", name2=f"{side}_finger_b",
                      data=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], name=f"{side}_finger_coupling")
    actuator = spec.add_actuator(name=f"{side}_gripper", target=f"{side}_finger_a",
                                 trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[0, FINGER_STROKE],
                                 forcerange=[-GRIP_FORCE, GRIP_FORCE], forcelimited=True, ctrllimited=True)
    actuator.set_to_position(kp=3000, kv=60)


def home_key(model, arms, lift=0.3):
    """Arms tucked forward-down, lift mid-travel, grippers open."""
    qpos, ctrl = model.qpos0.copy(), np.zeros(model.nu)
    def set_joint(name, value, actuator=None):
        qpos[model.jnt_qposadr[model.joint(name).id]] = value
        if actuator:
            ctrl[model.actuator(actuator).id] = value
    set_joint("lift", lift, "lift_drive")
    for arm in arms.values():
        for joint, value in zip(arm["joints"], arm["home"]):
            set_joint(joint, value)
        for joint in arm["gripper"]["joints"]:
            set_joint(joint, arm["gripper"]["open"])
        ctrl[model.actuator(arm["gripper"]["actuator"]).id] = arm["gripper"]["open"]
    return {"qpos": qpos.tolist(), "ctrl": ctrl.tolist()}


def build():
    spec = mujoco.MjSpec()
    spec.modelname = "np002_robot"
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    base = spec.worldbody.add_body(name="robot_base")
    for name, kind, axis in (("base_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
                             ("base_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0]),
                             ("base_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1])):
        base.add_joint(name=name, type=kind, axis=axis, armature=1.0, damping=1.0)
    base.add_geom(name="base_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BASE_HALF),
                  pos=[0, 0, BASE_CLEARANCE + BASE_HALF[2]], mass=60.0, rgba=[0.2, 0.22, 0.25, 1])
    base.add_geom(name="lift_column", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.12, 0.35],
                  pos=[-0.18, 0, BASE_CLEARANCE + 2 * BASE_HALF[2] + 0.35], mass=8.0, rgba=[0.3, 0.32, 0.35, 1])
    for name, joint, kv, force in (("base_x_drive", "base_x", 3000, 800), ("base_y_drive", "base_y", 3000, 800),
                                   ("base_yaw_drive", "base_yaw", 600, 300)):
        actuator = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                     forcerange=[-force, force], forcelimited=True)
        actuator.set_to_velocity(kv=kv)

    lift = base.add_body(name="lift_platform", pos=[0, 0, PLATFORM_Z])
    lift.add_joint(name="lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=list(LIFT_RANGE),
                   armature=5.0, damping=100.0)
    lift.add_geom(name="platform", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.16, 0.34, 0.03],
                  pos=[0, 0, -0.03], mass=10.0, rgba=[0.35, 0.37, 0.4, 1])
    lift.add_geom(name="head_mast", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.25],
                  pos=[-0.12, 0, 0.25], mass=1.0, rgba=[0.3, 0.3, 0.3, 1])
    lift.add_geom(name="head_camera_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.06, 0.025],
                  pos=[-0.1, 0, 0.52], mass=0.3, rgba=[0.1, 0.1, 0.1, 1])
    lift.add_camera(name="head_camera", pos=[-0.07, 0, 0.53], xyaxes=[0, -1, 0, 0.6, 0, 0.8])
    actuator = spec.add_actuator(name="lift_drive", target="lift", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                 ctrlrange=list(LIFT_RANGE), ctrllimited=True, forcerange=[-2500, 2500], forcelimited=True)
    actuator.set_to_position(kp=1e6, kv=3e4)
    spec.add_exclude(bodyname1="robot_base", bodyname2="lift_platform")

    arms = {}
    for side, sign in (("left", 1), ("right", -1)):
        frame = lift.add_frame(pos=[SHOULDER_X, sign * SHOULDER_Y, 0])
        prefix = f"{side}_"
        spec.attach(fr3_arm(prefix), prefix=prefix, frame=frame)
        spec.add_exclude(bodyname1="lift_platform", bodyname2=prefix + "fr3_link0")
        spec.add_exclude(bodyname1="lift_platform", bodyname2=prefix + "fr3_link1")
        joints = [f"{prefix}fr3_joint{i}" for i in range(1, 8)]
        for joint in joints:
            limit = spec.joint(joint).actfrcrange[1]
            actuator = spec.add_actuator(name=joint + "_motor", target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                         ctrlrange=[-limit, limit], ctrllimited=True)
            actuator.set_to_motor()
        flange = spec.site(prefix + "attachment_site").parent
        site = spec.site(prefix + "attachment_site")
        gripper_body = flange.add_body(name=f"{side}_tool_mount", pos=list(site.pos), quat=list(site.quat))
        gripper(spec, gripper_body, side)
        spec.add_exclude(bodyname1=prefix + "fr3_link7", bodyname2=f"{side}_hand")
        arms[side] = {"joints": joints, "actuators": [j + "_motor" for j in joints], "gains": ARM_GAINS,
                      "flange_site": prefix + "attachment_site", "tcp_site": f"{side}_tcp",
                      "gripper": {"actuator": f"{side}_gripper", "open": FINGER_STROKE, "closed": 0.0,
                                  "joints": [f"{side}_finger_a", f"{side}_finger_b"], "max_force_n": GRIP_FORCE,
                                  "pads": {"a": f"{side}_pad_a", "b": f"{side}_pad_b"},
                                  "finger_bodies": [f"{side}_finger_a", f"{side}_finger_b"]},
                      "wrist_camera": f"{side}_wrist_camera",
                      "home": [0, -0.6, 0, -2.2, 0, 1.6, -0.785],
                      "travel": [0, -1.5, 0, -2.8, 0, 1.4, -0.785],
                      "lateral_m": sign * SHOULDER_Y,
                      "payload_kg": 3.0, "reach_m": 0.855}

    model = spec.compile()
    home = home_key(model, arms)
    spec.add_key(name="home", qpos=home["qpos"], ctrl=home["ctrl"])
    model = spec.compile()
    OUT.mkdir(parents=True, exist_ok=True)
    # Mesh paths are relative to the FR3 directory; the written file lives in data/robot/.
    xml = spec.to_xml().replace('file="assets/', f'file="{os.path.relpath(FR3.parent, OUT)}/assets/')
    (OUT / "model.xml").write_text(xml, encoding="utf-8")
    profile = {
        "model": "data/robot/model.xml",
        "root_body": "robot_base",
        "base": {"type": "planar_velocity", "joints": ["base_x", "base_y", "base_yaw"],
                 "actuators": ["base_x_drive", "base_y_drive", "base_yaw_drive"],
                 "max_speed_mps": 0.5, "max_yaw_rate": 1.0, "footprint_half_m": list(BASE_HALF[:2])},
        "lift": {"joint": "lift", "actuator": "lift_drive", "range_m": list(LIFT_RANGE), "platform_z_m": PLATFORM_Z},
        "arms": arms,
        "head_camera": "head_camera",
        "force_limits_n": {"robot_default": 80.0, "glass": 30.0, "instrument_door": 40.0},
        "mass_kg": float(sum(model.body_mass[1:])),
    }
    (OUT / "profile.json").write_text(json.dumps(profile, indent=1), encoding="utf-8")
    print(f"wrote {OUT/'model.xml'}: nq={model.nq} nu={model.nu} ngeom={model.ngeom} mass={profile['mass_kg']:.1f} kg")


if __name__ == "__main__":
    build()
