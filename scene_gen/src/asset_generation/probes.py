"""Short contact probes written by the framework; never runs verification programs supplied by the model."""

from pathlib import Path
import numpy as np
import mujoco

from .runtime import InteractionRuntime
from .contracts import require


# Closed-loop presser advance: stop as soon as the button reaches the end of travel (REACHED), overtravel at most PRESS_OVERTRAVEL to offset
# contact compression; pressing further only pushes the button past its soft limit and produces false readings.
PRESS_OVERTRAVEL = .003
REACHED = .98
SETTLE_STEPS_SECONDS = .3


def _front_depth(model, data, g, axis):
    """Distance from the outermost surface of the pressed geom along -axis to the geom centre (via geom_aabb; works for every geom type)."""
    rotation = data.geom_xmat[g].reshape(3, 3)
    center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    local = rotation.T@axis
    return float(-local@center+np.abs(local)@half)


def _probe_model(root, model_file, joint):
    """The asset plus an independent kinematic presser; the free root of the button's part is fixed in place so the presser presses the button, not a falling instrument."""
    spec = mujoco.MjSpec.from_file(str(Path(root)/model_file))
    require(spec.geom("harness_probe") is None and spec.body("harness_probe_body") is None,"Asset uses a reserved probe name")
    plain = spec.compile()
    root_body = plain.body(int(plain.body_rootid[plain.jnt_bodyid[plain.joint(joint).id]])).name
    for item in list(spec.joints):
        if item.type == mujoco.mjtJoint.mjJNT_FREE and item.parent.name == root_body:
            spec.delete(item)
    actor = spec.worldbody.add_body(name="harness_probe_body",mocap=True,pos=[0,0,10])
    actor.add_geom(name="harness_probe",type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.003,0,0],mass=.01,rgba=[1,0,0,1])
    return spec.compile(), root_body


def interaction_probes(root, manifest, model_file="model.xml"):
    """Push the actual button with an independent kinematic presser; never write the button joint directly or pretend to be a robot arm.

    model_file is the configuration file under check; in multi-configuration assets the buttons exist only in their own configuration models.
    The button first rests under gravity to get its rest position; the presser advances to 98% of travel and stops once the event fires (overtravel at most 3 mm past the end of travel); after retracting it rests again,
    and return is judged by the rest position. A rest position already past released_at means the spring cannot hold the button's own weight, a real defect."""
    rows = []
    require(len(manifest["buttons"]) <= 32,"Too many button probes")
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
            (f"Gravity rest position {rest*1000:.2f} mm is already past released_at {button['released_at']*1000:.2f} mm (spring cannot hold the button's own weight, or released_at is too low)", sag_ok),
            ("event not triggered", event), ("did not return to rest position after presser retracted", returned),
            ("button state not reset after presser retracted", released), ("numerically unstable", stable)) if not ok]
        rows.append({"id":"button_probe:"+button["id"],"status":"fail" if failed else "pass",
                     "detail":("; ".join(failed)+". " if failed else "")+
                              f"Root {root_body} of the button's part is fixed; an independent presser advances to the end of button travel and stops (overtravel at most {PRESS_OVERTRAVEL*1000:g} mm), then retracts; the button joint is never written directly, and the robot pressing trajectory is not certified",
                     "rest_position":rest,"peak_position":peak,"final_position":final,"stroke":span,
                     "triggered":event,"released_after_retreat":released,"events":records[:50]})
    # An explicitly declared software-controlled connection can verify release logic; a physical interface must not be disguised as a software event.
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
                         "detail":"Mechanical action only declares a real interface; no matching operation probe yet, release not certified, and no geometry error can be concluded from it"})
            continue
        rows.append({"id":"release_probe:"+connection["id"],
                     "status":"pass" if connection["id"] in released_connections else "fail",
                     "detail":"The declared trigger event really deactivates the connection's equality constraint; does not prove a separation path or re-locking"})
    return rows
