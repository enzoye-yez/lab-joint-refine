"""Compute conditions, hold durations and device events from the actual simulation state."""
from copy import deepcopy
import mujoco
import numpy as np
from .common import OperationError, digest, need, number, vector
from .planning import rotation_error


def descendants(model, body):
    root = model.body(body).id if isinstance(body,str) else body
    if root == 0: return {0}  # The world's own ground geometry, excluding all descendant robots and objects.
    ids = set()
    for b in range(model.nbody):
        p = b
        while p and p != root: p = int(model.body_parentid[p])
        if p == root: ids.add(b)
    return ids


def body_geoms(model, body):
    ids = descendants(model, body)
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in ids}


def contacts(model, data):
    rows = []
    for i,c in enumerate(data.contact):
        if c.dist > 0: continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model,data,i,force)
        rows.append({"geoms":[int(c.geom1),int(c.geom2)], "distance":float(c.dist),
                     "force_n":float(np.linalg.norm(force[:3])), "position":c.pos.tolist(),
                     "normal":c.frame[:3].tolist()})
    return rows


class Evidence:
    def __init__(self, model, data, actors, task, settings):
        self.model, self.data, self.actors, self.task, self.settings = model,data,actors,task,settings
        self.timers, self.holds = {}, {}
        self.devices = {d["id"]:{"state":"idle","elapsed":0.} for d in task.get("devices",[])}
        self.rules = {d["id"]:d for d in task.get("devices",[])}
        self.mechanical_latched = set()
        self.last = {}
        self.contact_rows = []
        self.geometry_cache = {}
        self.monitored = {}
        self.register([task["initial"],task["final"]]+[s["goal"] for s in task["stages"]]+task.get("invariants",[]))
        self.register([r[key] for r in task.get("devices",[]) for key in ("start_condition","running_condition")])

    def register(self, rules):
        for rule in rules:
            self.monitored[digest(rule)] = deepcopy(rule)

    def geoms(self, body):
        if body not in self.geometry_cache: self.geometry_cache[body]=body_geoms(self.model,body)
        return self.geometry_cache[body]

    def snapshot(self):
        return deepcopy({"timers":self.timers,"holds":self.holds,"devices":self.devices,
            "mechanical_latched":sorted(self.mechanical_latched),"last":self.last,"monitored":self.monitored})

    def restore(self, value):
        self.timers, self.holds, self.devices = deepcopy(value["timers"]), deepcopy(value["holds"]), deepcopy(value["devices"])
        self.mechanical_latched = set(value["mechanical_latched"])
        self.last = deepcopy(value["last"])
        self.monitored = deepcopy(value.get("monitored",self.monitored))
        self.contact_rows = contacts(self.model,self.data)

    def touching(self, a, b):
        return [r for r in self.contact_rows if (r["geoms"][0] in a and r["geoms"][1] in b) or
                (r["geoms"][1] in a and r["geoms"][0] in b)]

    def speed(self, body):
        value = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.model.body(body).id,value,0)
        return float(np.linalg.norm(value[:3])),float(np.linalg.norm(value[3:]))

    def evaluate(self, rule):
        key, kind = digest(rule), rule["kind"]
        self.monitored[key] = deepcopy(rule)
        m,d = self.model,self.data
        detail = {}
        if kind in {"all","any"}:
            values = [self.evaluate(r) for r in rule["items"]]
            ok = all(values) if kind == "all" else any(values)
        elif kind == "not": ok = not self.evaluate(rule["item"])
        elif kind == "joint_range":
            j = m.joint(rule["joint"])
            need(j.type[0] in (2,3), "Joint-range detection supports only scalar joints")
            value = float(d.qpos[j.qposadr[0]])
            lo,hi = vector(rule["range"],2)
            ok,detail = lo <= value <= hi, {"value":value}
        elif kind == "contact":
            rows = self.touching(self.geoms(rule["body_a"]),self.geoms(rule["body_b"]))
            threshold = number(rule.get("min_force_n",.01),0,self.settings.contact_force_n)
            ok = any(r["force_n"] >= threshold for r in rows)
            detail = {"contacts":rows}
        elif kind == "relative_pose":
            a,b = m.site(rule["site_a"]).id,m.site(rule["site_b"]).id
            rot = d.site_xmat[b].reshape(3,3)
            delta = rot.T@(d.site_xpos[a]-d.site_xpos[b])-vector(rule.get("offset",[0,0,0]),3)
            angle = float(np.linalg.norm(rotation_error(rot,d.site_xmat[a].reshape(3,3))))
            ok = np.linalg.norm(delta) <= number(rule.get("position_tolerance",self.settings.position_tolerance_m),1e-8,1)
            if "angle_tolerance" in rule: ok &= angle <= number(rule["angle_tolerance"],1e-8,3.15)
            detail = {"distance_m":float(np.linalg.norm(delta)),"angle_rad":angle}
        elif kind in {"body_stable","base_stopped"}:
            body = rule.get("body","robot__layout_root")
            angular,linear = self.speed(body)
            ok = max(angular,linear) <= number(rule.get("speed",self.settings.stopped_speed),1e-8,1)
            detail = {"angular_speed":angular,"linear_speed":linear}
        elif kind == "held":
            arm,body = rule["arm"],rule["body"]
            need(arm in self.actors, "The selected arm does not exist")
            opposed,touched=self.grip_contacts(arm,body)
            sid,bid = m.site(rule["tcp"]).id,m.body(body).id
            rot = d.site_xmat[sid].reshape(3,3)
            relative = rot.T@(d.xpos[bid]-d.site_xpos[sid])
            relative_rot = rot.T@d.xmat[bid].reshape(3,3)
            previous = self.holds.get(key)
            if not opposed:
                self.holds.pop(key,None); ok=False
            else:
                if previous is None:
                    previous={"position":relative.tolist(),"rotation":relative_rot.tolist()}; self.holds[key]=previous
                drift = float(np.linalg.norm(relative-np.array(previous["position"])))
                angle = float(np.linalg.norm(rotation_error(previous["rotation"],relative_rot)))
                ok = drift <= self.settings.position_tolerance_m and angle <= self.settings.rotation_tolerance_rad
                detail={"relative_drift_m":drift,"rotation_drift_rad":angle,"contact_geoms":sorted(touched)}
                if not ok and not previous.get("locked"):
                    # Reset the observation window after a failed sample, not the physical pose.
                    # An established held-object guard still fails on this very sample.
                    self.holds[key]={"position":relative.tolist(),"rotation":relative_rot.tolist()}
        elif kind == "supported":
            rows=self.touching(self.geoms(rule["body"]),self.geoms(rule["support"]))
            touching_hand=any(self.touching(set(gs),self.geoms(rule["body"])) for gs in self.actors.values())
            angular,linear=self.speed(rule["body"])
            ok=bool(rows) and not touching_hand and max(angular,linear)<=self.settings.stopped_speed
            detail={"contacts":rows,"touching_hand":touching_hand}
        elif kind == "device_state":
            need(rule["device"] in self.devices,"Unknown device state")
            ok=self.devices[rule["device"]]["state"]==rule["state"]
            detail=deepcopy(self.devices[rule["device"]])
        else: raise OperationError("capability_missing","Unknown detector")
        if ok:
            self.timers.setdefault(key,float(d.time))
        else: self.timers.pop(key,None)
        passed=bool(ok and d.time-self.timers.get(key,d.time)+1e-10 >= rule.get("duration",0))
        self.last[key]={"time":float(d.time),"passed":passed,"condition":rule,"evidence":detail}
        return passed

    def lock_hold(self,rule):
        # Commit the acquisition anchor once; repeated reads must never erase later slip.
        need(self.evaluate(rule),"Grasp is not yet stable; cannot fix the hold reference","grasp_lost")
        immediate={**rule,"duration":0};anchor=deepcopy(self.holds[digest(rule)]);anchor["locked"]=True
        self.holds[digest(immediate)]=anchor;self.register([immediate])

    def grip_contacts(self,arm,body):
        need(arm in self.actors,"The selected arm does not exist")
        rows=self.touching(set(self.actors[arm]),self.geoms(body))
        touched={g for r in rows for g in r["geoms"] if g in self.actors[arm]}
        bodies={int(self.model.geom_bodyid[g]) for g in touched}
        normals=[np.array(r["normal"])*(1 if r["geoms"][0] in self.actors[arm] else -1) for r in rows]
        opposed=len(bodies)>=2 and any(float(a@b)<-.5 for i,a in enumerate(normals) for b in normals[i+1:])
        return opposed,touched

    def update(self, seconds, logical_extra=0):
        self.contact_rows=contacts(self.model,self.data)
        for rule in list(self.monitored.values()): self.evaluate(rule)
        for device,state in self.devices.items():
            if state["state"]=="running":
                if not self.evaluate(self.rules[device]["running_condition"]):
                    state["state"]="fault"
                    raise OperationError("device_interlock","Loading or closure condition failed while running",{"device":device})
                state["elapsed"]+=seconds+logical_extra
                if state["elapsed"]>=self.rules[device]["duration_seconds"]: state["state"]="complete"
        events=[]
        for detector in self.task.get("mechanical_detectors",[]):
            name=detector["event"]
            if self.evaluate(detector["condition"]):
                if name not in self.mechanical_latched: events.append(name); self.mechanical_latched.add(name)
            else: self.mechanical_latched.discard(name)
        return events

    def start(self, device):
        need(device in self.rules,"Unknown device")
        state,rule=self.devices[device],self.rules[device]
        need(state["state"]=="idle" and self.evaluate(rule["start_condition"]),"Device start condition not met","device_interlock")
        state["state"]="running"
        return rule["interface"]
