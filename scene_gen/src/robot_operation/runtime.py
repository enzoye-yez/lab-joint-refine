"""The real world advances only through actuators; candidate search and restore are used only for explicit trial and error."""
from copy import deepcopy
from dataclasses import asdict
import math
import mujoco
from .common import copy_data
import numpy as np

from asset_generation.runtime import InteractionRuntime
from scene_generation.runtime import actor_groups
from scene_generation.robot_contract import arms, validate_bindings
from .common import OperationError, Settings, need, vector, number
from .evidence import Evidence, body_geoms, contacts, descendants
from .planning import Kinematics, astar, edge_valid, interpolate, rrt_connect, rotation_error, wheel_commands


class Session:
    def __init__(self, model, data, metadata, manifest, task, settings=None, recorder=None):
        self.model,self.data,self.meta,self.manifest,self.task=model,data,metadata,manifest,task
        self.settings=settings or Settings()
        for body in task["bindings"].values():
            need(isinstance(body,str) and mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)>=0,"Task object identity is not bound to an actual body")
        self.recorder=recorder
        self.rng=np.random.default_rng(self.settings.seed)
        self.actors=actor_groups(model,metadata)
        self.arms={a["id"]:a for a in arms(metadata["environment"])}
        selected=validate_bindings(metadata["plan"],metadata["environment"],partial=True)
        self.tcps={a:(selected[a]["instance"]+"__tcp" if selected[a]["mode"]!="stock" else "robot__"+v["tcp_site"])
                   for a,v in self.arms.items()}
        for site in self.tcps.values(): model.site(site)
        self.interaction=InteractionRuntime(model,manifest,actor_geom_ids=sorted({g for gs in self.actors.values() for g in gs}))
        self.evidence=Evidence(model,data,self.actors,task,self.settings)
        self.geoms=self.evidence.geoms
        self.evidence.contact_rows=contacts(model,data)
        self.node=None; self.deadline=math.inf; self.logical_time=0.; self.holding={}
        self.allowed=[]; self.held_checks={}; self.last_events=[]
        self.base=model.joint("robot__free_base")
        self.base_body=int(model.jnt_bodyid[self.base.id])
        self.robot_geoms=body_geoms(model,self.base_body)
        self.wheels=self._wheels()
        self.sim_steps=0
        self.devices_by_interface={d["interface"]:d["id"] for d in task.get("devices",[])}
        software={i["id"] for i in manifest["interfaces"] if i["kind"]=="software"}
        need(set(self.devices_by_interface)<=software,"A device request must be bound to a declared software interface")
        events={e["id"] for e in manifest.get("mechanical_events",[])}
        need({e["event"] for e in task.get("mechanical_detectors",[])}<=events,"Mechanical detector event is not declared")
        bound={e["event"] for e in task.get("mechanical_detectors",[])}
        self.unbound_events=sorted(events-bound)
        self.mechanical_interfaces={e["id"]:next(i for i in manifest["interfaces"] if i["id"]==e["interface"])
                                    for e in manifest.get("mechanical_events",[])}
        self.position_actuators={}
        for a in range(model.nu):
            if model.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and model.actuator_biasprm[a,1]<0:
                self.position_actuators[int(model.actuator_trnid[a,0])]=a

    def _wheels(self):
        m,d=self.model,self.data
        result=[]
        rotation=d.xmat[self.base_body].reshape(3,3)
        for stem in ("left","right","rear"):
            steer=m.joint("robot__"+stem+"_wheel_steer_joint")
            drive=m.joint("robot__"+stem+"_wheel_drive_joint")
            ga=[g for g in range(m.ngeom) if m.geom_bodyid[g]==drive.bodyid[0] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_CYLINDER
                and (m.geom_contype[g] or m.geom_conaffinity[g])]
            need(len(ga)==1,"Each wheel unit must have a unique contact cylinder; other robots need an explicit wheel adapter")
            actuator=lambda j: next((a for a in range(m.nu) if m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a,0]==j.id),None)
            sa,da=actuator(steer),actuator(drive)
            need(sa is not None and da is not None,"Wheel unit has no drive actuator")
            need(m.actuator_biasprm[sa,1]<0 and m.actuator_biasprm[da,2]<0 and m.actuator_biasprm[da,1]==0,"Wheel unit actuator type mismatch")
            result.append({"xy":(rotation.T@(d.xanchor[steer.id]-d.xpos[self.base_body]))[:2].tolist(),
                "radius":float(m.geom_size[ga[0],0]),"steer":sa,"drive":da,"steer_joint":steer.id,"geom":ga[0],"sign":1})
        return result

    def snapshot(self):
        state=np.zeros(mujoco.mj_stateSize(self.model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model,self.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        return {"physics":state.tolist(),"rng":deepcopy(self.rng.bit_generator.state),"evidence":self.evidence.snapshot(),
            "interaction":{"pressed":deepcopy(self.interaction.pressed),"released":sorted(self.interaction.released)},
            "logical_time":self.logical_time,"holding":deepcopy(self.holding),"held_checks":deepcopy(self.held_checks),
            "allowed":deepcopy(self.allowed),"node":self.node,"deadline_remaining":None if math.isinf(self.deadline) else self.deadline-self.data.time,
            "sim_steps":self.sim_steps,"selected_arm":getattr(self,"selected_arm",None)}

    def restore(self, state):
        need(self.recorder is None or self.recorder.mode=="trial","Restoring checkpoints is not allowed in formal continuous execution")
        mujoco.mj_setState(self.model,self.data,np.asarray(state["physics"]),mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model,self.data)
        self.rng.bit_generator.state=deepcopy(state["rng"])
        self.evidence.restore(state["evidence"])
        self.interaction.pressed=deepcopy(state["interaction"]["pressed"])
        self.interaction.released=set(state["interaction"]["released"])
        self.logical_time=state["logical_time"]; self.holding=deepcopy(state["holding"])
        self.held_checks=deepcopy(state["held_checks"]); self.allowed=deepcopy(state["allowed"])
        self.node=state["node"]; self.sim_steps=state["sim_steps"]
        if state.get("selected_arm"):self.selected_arm=state["selected_arm"]
        elif hasattr(self,"selected_arm"):del self.selected_arm
        self.deadline=math.inf if state["deadline_remaining"] is None else self.data.time+state["deadline_remaining"]
        if self.recorder: self.recorder.event({"type":"checkpoint_restore","time":float(self.data.time)})

    def begin(self,node):
        from .quality import check_node
        check_node(self,node)
        self.node=node["id"]; self.deadline=float(self.data.time)+node["timeout"]
        self.allowed=[]
        for pair in node["parameters"].get("allowed_contacts",[]):
            need(isinstance(pair,list) and len(pair)==2,"An allowed contact must be a pair of body names")
            self.model.body(pair[0]); self.model.body(pair[1]); self.allowed.append(pair)

    def acceptable_contact(self,g1,g2, data=None):
        m=self.model
        floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"environment__floor")
        if floor in (g1,g2) and any(w["geom"] in (g1,g2) for w in self.wheels): return True
        for a,b in self.allowed:
            ga,gb=self.geoms(a),self.geoms(b)
            if g1 in ga and g2 in gb or g2 in ga and g1 in gb: return True
        for arm,body in self.holding.items():
            ga,gb=set(self.actors[arm]),self.geoms(body)
            if g1 in ga and g2 in gb or g2 in ga and g1 in gb: return True
        return False

    def collision_valid(self, data):
        carried=set().union(*(self.geoms(b) for b in self.holding.values())) if self.holding else set()
        for c in data.contact:
            a,b=int(c.geom1),int(c.geom2)
            if c.dist < -self.settings.penetration_m:
                self.last_collision={"geom_indices":[a,b],"geoms":[self.model.geom(a).name,self.model.geom(b).name],"distance_m":float(c.dist)}
                return False
            if c.dist<=0 and (a in self.robot_geoms|carried or b in self.robot_geoms|carried) and not self.acceptable_contact(a,b): return False
        return True

    def guard(self):
        m,d,s=self.model,self.data,self.settings
        need(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not any(w.number for w in d.warning),"Simulation unstable","simulation_unstable")
        tilt=math.acos(float(np.clip(d.xmat[self.base_body].reshape(3,3)[2,2],-1,1)))
        need(tilt<=s.base_tilt_rad,"Base tilt out of bounds","base_unstable")
        for row in self.evidence.contact_rows:
            a,b=row["geoms"]
            need(row["distance"]>=-s.penetration_m,"Contact penetration exceeds the limit","collision")
            if a in self.robot_geoms or b in self.robot_geoms:
                need(self.acceptable_contact(a,b),"Unexpected robot contact","collision")
                if not any(w["geom"] in (a,b) for w in self.wheels):
                    need(row["force_n"]<=s.contact_force_n,"Operation contact force exceeds the engineering bound","contact_limit")
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in (2,3):
                value=d.qpos[m.jnt_qposadr[j]]
                need(m.jnt_range[j,0]-.03<=value<=m.jnt_range[j,1]+.03,"Observed joint out of bounds","joint_limit")
        for rule in self.task.get("invariants",[]):
            need(self.evidence.evaluate(rule),"Task invariant violated","task_invariant")
        for rule in self.held_checks.values():
            need(self.evidence.evaluate({**rule,"duration":0}),"Held object slipped or grasp failed","grasp_lost")

    def tick(self, software=(), logical_extra=0):
        getattr(self,"cancel_guard",lambda:None)()
        need(self.data.time < self.deadline+1e-9,"Node execution timed out","timeout")
        need(self.data.time < self.settings.max_sim_seconds,"Total simulation budget exhausted","budget_exhausted")
        before=float(self.data.time)
        controls=self.data.ctrl.copy()
        mujoco.mj_step(self.model,self.data)
        # Update kinematics and constraint forces used for evidence without changing the integration state.
        mujoco.mj_forward(self.model,self.data)
        self.logical_time+=self.model.opt.timestep+logical_extra
        try:
            self.last_events=[]
            mechanical=self.evidence.update(self.model.opt.timestep,logical_extra)
            actors=set().union(*(set(gs) for gs in self.actors.values()))
            actors |= set().union(*(self.geoms(b) for b in self.holding.values())) if self.holding else set()
            for event in self.unbound_events:
                target=self.geoms(self.mechanical_interfaces[event]['body'])
                need(not self.evidence.touching(actors,target),
                    'Mechanical event of the contacted object has no detector','capability_missing')
            # A mechanical threshold also requires real contact between the acting body and that interface part; it cannot be triggered by an unrelated joint.
            for event in mechanical:
                target=self.geoms(self.mechanical_interfaces[event]["body"])
                actors=set().union(*(set(gs) for gs in self.actors.values()))
                actors |= set().union(*(self.geoms(b) for b in self.holding.values())) if self.holding else set()
                need(bool(self.evidence.touching(actors,target)),"Mechanical event lacks contact evidence from the acting body","mechanical_evidence_missing")
            self.last_events=self.interaction.update(self.data,software,verified_mechanical_events=mechanical)
            self.guard()
        finally:
            self.sim_steps+=1
            if self.recorder: self.recorder.step(self,before,controls,list(software),logical_extra)

    def advance(self,seconds):
        end=self.data.time+seconds
        while self.data.time<end-1e-10: self.tick()

    def stop(self):
        for wheel in self.wheels: self.data.ctrl[wheel["drive"]]=0
        for j,a in self.position_actuators.items():
            self.data.ctrl[a]=self.data.qpos[self.model.jnt_qposadr[j]]

    def settled(self):
        rule={"kind":"base_stopped","duration":self.settings.stable_seconds}
        while not self.evidence.evaluate(rule): self.advance(self.settings.control_period)

    def controls_for(self,joints,q):
        for name,value in zip(joints,q):
            j=self.model.joint(name)
            need(j.id in self.position_actuators,"Joint has no position servo","capability_missing")
            a=self.position_actuators[j.id]
            need(j.range[0]<=value<=j.range[1],"Position command outside the joint range")
            if self.model.actuator_ctrllimited[a]:
                need(self.model.actuator_ctrlrange[a,0]<=value<=self.model.actuator_ctrlrange[a,1],"Control command outside the actuator range")
            self.data.ctrl[a]=value

    def joints_for(self,arms_selected,include_lift=False):
        need(arms_selected and len(set(arms_selected))==len(arms_selected) and set(arms_selected)<=set(self.arms),"Invalid arm selection")
        joints=["robot__"+j for arm in arms_selected for j in self.arms[arm]["joints"]]
        if include_lift:
            shared=self.meta["environment"]["robot"].get("shared_joints",[])
            # Shared vertical prismatic mechanisms, not an arbitrary hard-coded joint name.
            lifts=["robot__"+j for j in shared if self.model.joint("robot__"+j).type[0]==2
                   and abs(float(self.model.jnt_axis[self.model.joint("robot__"+j).id,2]))>.99]
            need(len(lifts)==1,"Lift mechanism lacks a unique actual shared joint")
            joints.extend(lifts)
        return joints

    def query(self,joints,base_data=None):
        m,d=self.model,self.data
        clone=mujoco.MjData(m); copy_data(clone,m,base_data if base_data is not None else d)
        qids=[int(m.joint(j).qposadr[0]) for j in joints]
        # Carried objects follow the end effector only in the candidate copy; the execution world still relies solely on gripper contact.
        carry=[]
        for arm,body in self.holding.items():
            bid=m.body(body).id; sid=m.site(self.tcps[arm]).id
            jid=int(m.body_jntadr[bid])
            need(m.body_jntnum[bid]>0 and m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE,"A carried object requires its own free root joint","capability_missing")
            rot=d.site_xmat[sid].reshape(3,3)
            carry.append((int(m.jnt_qposadr[jid]),sid,rot.T@(d.xpos[bid]-d.site_xpos[sid]),rot.T@d.xmat[bid].reshape(3,3)))
        def valid(q):
            clone.qpos[qids]=q
            mujoco.mj_forward(m,clone)
            for address,sid,pos,rot in carry:
                frame=clone.site_xmat[sid].reshape(3,3)
                clone.qpos[address:address+3]=clone.site_xpos[sid]+frame@pos
                mujoco.mju_mat2Quat(clone.qpos[address+3:address+7],(frame@rot).reshape(9))
            mujoco.mj_forward(m,clone)
            return self.collision_valid(clone)
        return clone,qids,valid

    def follow(self,joints,path):
        qids=[int(self.model.joint(j).qposadr[0]) for j in joints]
        for target in path[1:]:
            limits=np.array([self.model.joint(j).range for j in joints])
            start=np.clip(self.data.qpos[qids],limits[:,0],limits[:,1])
            duration=max(self.settings.control_period,float(np.max(np.abs(target-start)))/self.settings.joint_speed)
            steps=max(1,int(math.ceil(duration/self.settings.control_period)))
            aids=[self.position_actuators[self.model.joint(j).id] for j in joints]
            # Preserve existing load compensation instead of resetting the servo equilibrium.
            offset=self.data.ctrl[aids]-self.data.qpos[qids]
            for k in range(1,steps+1):
                command=start+(target-start)*k/steps+offset
                self.controls_for(joints,np.clip(command,limits[:,0],limits[:,1]))
                self.advance(self.settings.control_period)
            # Gravity produces a nonzero static error in the vendor position servos.
            # Correct through bounded actuator commands, never by overwriting measured state.
            for _ in range(101):
                error=target-self.data.qpos[qids]
                if np.max(np.abs(error))<=self.settings.joint_tolerance:break
                delta=np.clip(error*.5,-self.settings.joint_speed*self.settings.control_period,
                              self.settings.joint_speed*self.settings.control_period)
                self.controls_for(joints,np.clip(self.data.ctrl[aids]+delta,limits[:,0],limits[:,1]))
                self.advance(self.settings.control_period)
            else:raise OperationError("tracking_failed","Bounded joint tracking did not reach the target; the real error is kept")

    def joint_move(self,p):
        selected=p["arms"]; allowed=self.joints_for(selected,p.get("include_lift",False))
        joints=list(p["positions"])
        need(set(joints)<=set(allowed) and joints,"Motion may control only the selected arms and an explicitly selected lift joint")
        _,qids,valid=self.query(joints)
        limits=np.array([self.model.joint(j).range for j in joints])
        goal=vector([p["positions"][j] for j in joints],len(joints))
        path=rrt_connect(self.data.qpos[qids].copy(),goal,limits,valid,self.rng,self.settings.rrt_iterations,self.settings.path_resolution,start_tolerance=self.settings.joint_tolerance)
        self.follow(joints,path)

    def targets(self,rows,data=None):
        data=self.data if data is None else data
        result=[]
        for row in rows:
            arm=row["arm"]; need(arm in self.tcps,"Unknown arm")
            if "interface" in row:
                interface=next((i for i in self.manifest["interfaces"] if i["id"]==row["interface"]),None)
                need(interface is not None and interface.get("site"),"Operation target interface has no actual site")
                need("frame" not in row,"An interface target must not also specify another frame")
                row={**row,"frame":interface["site"]}
            position=vector(row["position"],3)
            quat=vector(row["quaternion"],4)
            need(abs(np.linalg.norm(quat)-1)<1e-5,"Target quaternion must be normalized")
            rotation=np.zeros(9); mujoco.mju_quat2Mat(rotation,quat); rotation=rotation.reshape(3,3)
            if "frame" in row:
                sid=self.model.site(row["frame"]).id
                r=data.site_xmat[sid].reshape(3,3)
                position=data.site_xpos[sid]+r@position; rotation=r@rotation
            result.append({"site":self.tcps[arm],"position":position,"rotation":rotation})
        return result

    def cartesian(self,p,contact=False):
        stop=p.get("stop_when")
        if stop is not None:
            from .contracts import predicate
            predicate(stop);need(contact,"A stop condition is allowed only on a bounded contact path")
            if self.evidence.evaluate(stop):return
        rows=p["targets"]; joints=self.joints_for([r["arm"] for r in rows],p.get("include_lift",False))
        targets=self.targets(rows)
        _,qids,valid=self.query(joints)
        kin=Kinematics(self.model,self.data,joints,valid,self.settings,self.rng)
        if not contact:
            goal=kin.solve(targets)
            path=rrt_connect(self.data.qpos[qids].copy(),goal,kin.limits,valid,self.rng,self.settings.rrt_iterations,self.settings.path_resolution,start_tolerance=self.settings.joint_tolerance)
            self.follow(joints,path)
            self.track_targets(joints,targets); return
        starts=[(self.data.site_xpos[self.model.site(t["site"]).id].copy(),self.data.site_xmat[self.model.site(t["site"]).id].reshape(3,3).copy()) for t in targets]
        distance=max(np.linalg.norm(t["position"]-start[0]) for t,start in zip(targets,starts))
        angle=max(np.linalg.norm(rotation_error(t["rotation"],start[1])) for t,start in zip(targets,starts))
        steps=max(1,int(math.ceil(max(distance/self.settings.cartesian_speed_mps,angle/.2)/self.settings.control_period)))
        for k in range(1,steps+1):
            if stop is not None and self.evidence.evaluate(stop):return
            interpolated=[]
            for target,(pos,rot) in zip(targets,starts):
                delta=rotation_error(target["rotation"],rot)*k/steps
                magnitude=np.linalg.norm(delta); quat=np.array([1.,0,0,0]); matrix=np.zeros(9)
                if magnitude>1e-12: mujoco.mju_axisAngle2Quat(quat,delta/magnitude,magnitude)
                mujoco.mju_quat2Mat(matrix,quat)
                interpolated.append({"site":target["site"],"position":pos+(target["position"]-pos)*k/steps,"rotation":matrix.reshape(3,3)@rot})
            _,qids,valid=self.query(joints)
            kin=Kinematics(self.model,self.data,joints,valid,self.settings,self.rng)
            q=kin.solve(interpolated)
            need(edge_valid(self.data.qpos[qids],q,valid,self.settings.path_resolution),"Intermediate state of the contact path is blocked","path_not_found")
            # Slow down as contact load rises; crossing the hard threshold is stopped by guard and enters the explicit recovery branch.
            force=max((r["force_n"] for r in self.evidence.contact_rows
                       if any(g in self.robot_geoms for g in r["geoms"])
                       and not any(w["geom"] in r["geoms"] for w in self.wheels)),default=0.)
            scale=.5 if force>self.settings.contact_force_n*.5 else 1.
            start=np.clip(self.data.qpos[qids],kin.limits[:,0],kin.limits[:,1])
            segments=max(1,int(np.ceil(np.max(np.abs(q-start))/(self.settings.joint_speed*self.settings.control_period*scale))))
            for segment in range(1,segments+1):
                # IK returns a desired physical configuration, not an actuator equilibrium.
                # Retain servo effort already balancing gravity/contact instead of resetting it
                # to the measured pose at every tiny Cartesian increment.
                command=np.array([self.data.ctrl[self.position_actuators[self.model.joint(j).id]] for j in joints])
                delta=(q-self.data.qpos[qids])*.5/segments
                self.controls_for(joints,np.clip(command+delta,kin.limits[:,0],kin.limits[:,1]))
                self.advance(self.settings.control_period)
                if stop is not None and self.evidence.evaluate(stop):return
        self.track_targets(joints,targets)
        if stop is not None:
            need(self.evidence.evaluate(stop),"Stop condition not met within the bounded contact travel","contact_not_found")

    def track_targets(self,joints,targets):
        # Bounded closed-loop correction; simulator state is never assigned here.
        qids=[int(self.model.joint(j).qposadr[0]) for j in joints]
        stable_since=None
        for attempt in range(101):
            residuals=[]
            for target in targets:
                sid=self.model.site(target["site"]).id
                residuals.append({"site":target["site"],"position_error_m":float(np.linalg.norm(self.data.site_xpos[sid]-target["position"])),
                    "rotation_error_rad":float(np.linalg.norm(rotation_error(target["rotation"],self.data.site_xmat[sid].reshape(3,3))))})
            self.last_tracking={"iterations":attempt,"residuals":residuals}
            if all(r["position_error_m"]<=self.settings.position_tolerance_m*2 and r["rotation_error_rad"]<=self.settings.rotation_tolerance_rad*2 for r in residuals):
                if stable_since is None:stable_since=float(self.data.time)
                if self.data.time-stable_since>=self.settings.stable_seconds:return
                if attempt==100:raise OperationError("tracking_error","End effector did not settle within the tracking budget",self.last_tracking)
                self.advance(self.settings.control_period);continue
            stable_since=None
            if attempt==100:raise OperationError("tracking_error","End effector closed-loop tracking did not reach the target within budget",self.last_tracking)
            _,_,valid=self.query(joints)
            kin=Kinematics(self.model,self.data,joints,valid,self.settings,self.rng)
            q=kin.solve(targets)
            need(edge_valid(self.data.qpos[qids],q,valid,self.settings.path_resolution),"Tracking correction edge is blocked","path_not_found")
            command=np.array([self.data.ctrl[self.position_actuators[self.model.joint(j).id]] for j in joints])
            correction=np.clip((q-self.data.qpos[qids])*.5,-self.settings.joint_speed*self.settings.control_period,self.settings.joint_speed*self.settings.control_period)
            self.controls_for(joints,np.clip(command+correction,kin.limits[:,0],kin.limits[:,1]))
            self.advance(self.settings.control_period)

    def gripper(self,arm,values,*,apply=True):
        need(arm in self.actors and isinstance(values,dict) and values,"Gripper command missing")
        bodies={int(self.model.geom_bodyid[g]) for g in self.actors[arm]}
        commands={}
        for name,value in values.items():
            a=self.model.actuator(name).id; j=int(self.model.actuator_trnid[a,0])
            need(self.model.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and int(self.model.jnt_bodyid[j]) in bodies,"A gripper command cannot control other actuators")
            number(value,-1e5,1e5)
            if self.model.actuator_ctrllimited[a]: need(self.model.actuator_ctrlrange[a,0]<=value<=self.model.actuator_ctrlrange[a,1],"Gripper command out of range")
            commands[a]=value
        if apply:
            for actuator,value in commands.items():self.data.ctrl[actuator]=value

    def grasp(self,p):
        arm,body=p["arm"],p["body"]
        need(arm not in self.holding,"This arm is already carrying an object")
        need(bool(self.geoms(body)),"Grasp target has no geometry")
        if "approach" in p: self.cartesian(p["approach"])
        check={"kind":"held","arm":arm,"body":body,"tcp":self.tcps[arm],"duration":self.settings.stable_seconds}
        # Ramp toward the requested closure and hold once opposed contact is established.
        # A closing target is a bound, never permission to keep squeezing through the object.
        targets=p["controls"];self.gripper(arm,targets,apply=False)
        speed=number(p.get("closure_speed",self.settings.joint_speed),1e-5,self.settings.joint_speed)
        minimum=number(p.get("contact_force_n",0),0,self.settings.contact_force_n)
        def force_ready():
            if not self.evidence.grip_contacts(arm,body)[0]:return False
            forces={}
            for row in self.evidence.touching(set(self.actors[arm]),self.geoms(body)):
                for g in row["geoms"]:
                    if g in self.actors[arm]:
                        b=int(self.model.geom_bodyid[g]);forces[b]=forces.get(b,0)+row["force_n"]
            return len(forces)>=2 and min(forces.values())>=minimum
        while not (force_ready() and self.evidence.evaluate(check)):
            if not force_ready():
                command={}
                for name,target in targets.items():
                    current=float(self.data.ctrl[self.model.actuator(name).id])
                    command[name]=current+float(np.clip(target-current,-speed*self.settings.control_period,speed*self.settings.control_period))
                self.gripper(arm,command)
            self.advance(self.settings.control_period)
        if "hold_controls" in p:
            need(set(p["hold_controls"])==set(targets),"Hold commands must target the same set of gripper actuators")
            self.gripper(arm,p["hold_controls"])
            self.advance(self.settings.stable_seconds)
            while not self.evidence.evaluate(check):self.advance(self.settings.control_period)
        self.evidence.lock_hold(check)
        self.holding[arm]=body; self.held_checks[arm]=check

    def place(self,p):
        arm=p["arm"]; need(self.holding.get(arm)==p["body"],"The selected arm is not holding the selected object","wrong_object")
        if "approach" in p: self.cartesian(p["approach"],True)
        self.gripper(arm,p["controls"])
        self.holding.pop(arm); self.held_checks.pop(arm)
        if "retreat" in p: self.cartesian(p["retreat"],True)
        check={"kind":"supported","body":p["body"],"support":p["support"],"duration":self.settings.stable_seconds}
        while not self.evidence.evaluate(check): self.advance(self.settings.control_period)

    def axis_motion(self,p,rotate=False):
        arm=p["arm"]; sid=self.model.site(self.tcps[arm]).id
        axis=vector(p["axis"],3); need(abs(np.linalg.norm(axis)-1)<1e-5,"Operation axis must be normalized")
        if "frame" in p: axis=self.data.site_xmat[self.model.site(p["frame"]).id].reshape(3,3)@axis
        position=self.data.site_xpos[sid].copy(); rotation=self.data.site_xmat[sid].reshape(3,3).copy()
        if rotate:
            # Rotation must follow an arc; straight-line interpolation between start and end must not replace motion about the axis.
            total=number(p["angle"],-2*math.pi,2*math.pi)
            count=max(1,int(math.ceil(abs(total)/.02)))
            start_position, start_rotation=position.copy(),rotation.copy()
            for step in range(1,count+1):
                quat=np.zeros(4); matrix=np.zeros(9)
                mujoco.mju_axisAngle2Quat(quat,axis,total*step/count)
                mujoco.mju_quat2Mat(matrix,quat); rot=matrix.reshape(3,3)
                rotation=rot@start_rotation
                position=start_position.copy()
                if "pivot" in p:
                    pivot=vector(p["pivot"],3); position=pivot+rot@(start_position-pivot)
                mujoco.mju_mat2Quat(quat,rotation.reshape(9))
                rows=[{"arm":arm,"position":position.tolist(),"quaternion":quat.tolist()}]
                for other in p.get("hold_arms",[]):
                    other_id=self.model.site(self.tcps[other]).id; other_quat=np.zeros(4)
                    mujoco.mju_mat2Quat(other_quat,self.data.site_xmat[other_id])
                    rows.append({"arm":other,"position":self.data.site_xpos[other_id].tolist(),"quaternion":other_quat.tolist()})
                self.cartesian({"targets":rows},True)
            return
        position+=axis*number(p["distance"],-1,1)
        quat=np.zeros(4); mujoco.mju_mat2Quat(quat,rotation.reshape(9))
        rows=[{"arm":arm,"position":position.tolist(),"quaternion":quat.tolist()}]
        for other in p.get("hold_arms",[]):
            sid=self.model.site(self.tcps[other]).id; q=np.zeros(4)
            mujoco.mju_mat2Quat(q,self.data.site_xmat[sid])
            rows.append({"arm":other,"position":self.data.site_xpos[sid].tolist(),"quaternion":q.tolist()})
        self.cartesian({"targets":rows,**({"stop_when":p["stop_when"]} if "stop_when" in p else {})},True)

    def navigate(self,p,prepare_transport=True):
        if prepare_transport: self.joint_move(p["transport"])
        target=vector(p["goal"],3)
        m,d=self.model,self.data; q=int(self.base.qposadr[0]); rotation=d.xmat[self.base_body].reshape(3,3)
        yaw=math.atan2(rotation[1,0],rotation[0,0])
        clone=mujoco.MjData(m); copy_data(clone,m,d)
        def valid(point,theta=yaw):
            clone.qpos[:]=d.qpos
            clone.qpos[q:q+2]=point
            clone.qpos[q+3:q+7]=[math.cos(theta/2),0,0,math.sin(theta/2)]
            mujoco.mj_forward(m,clone)
            # Carried objects follow the base rigidly only for collision queries.
            for arm,body in self.holding.items():
                bid=m.body(body).id; jid=int(m.body_jntadr[bid]); address=int(m.jnt_qposadr[jid])
                need(m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE,"A transported object requires its own free root joint")
                sid=m.site(self.tcps[arm]).id; oldrot=d.site_xmat[sid].reshape(3,3); newrot=clone.site_xmat[sid].reshape(3,3)
                clone.qpos[address:address+3]=clone.site_xpos[sid]+newrot@oldrot.T@(d.xpos[bid]-d.site_xpos[sid])
                mujoco.mju_mat2Quat(clone.qpos[address+3:address+7],(newrot@oldrot.T@d.xmat[bid].reshape(3,3)).reshape(9))
            mujoco.mj_forward(m,clone)
            return self.collision_valid(clone)
        route=astar(d.qpos[q:q+2],target[:2],p["bounds"],valid,self.settings.grid_m)
        # Keep the sweep-checked turning points to avoid stopping repeatedly at every grid cell center.
        simplified=[route[0]]; index=0
        while index<len(route)-1:
            for candidate in range(min(len(route)-1,index+20),index,-1):
                if edge_valid(route[index],route[candidate],valid,self.settings.grid_m/3):
                    simplified.append(route[candidate]); index=candidate; break
        route=simplified
        delta=(target[2]-yaw+math.pi)%(2*math.pi)-math.pi
        need(all(valid(target[:2],yaw+t*delta) for t in np.linspace(0,1,max(2,int(abs(delta)/.03)+1))),"Final turning sweep is blocked","path_not_found")
        for index,point in enumerate(route[1:]+[target[:2]]):
            desired_yaw=target[2] if index==len(route)-1 else yaw
            while True:
                rotation=d.xmat[self.base_body].reshape(3,3); actual_yaw=math.atan2(rotation[1,0],rotation[0,0])
                error=np.asarray(point)-d.qpos[q:q+2]
                angular=(desired_yaw-actual_yaw+math.pi)%(2*math.pi)-math.pi
                if np.linalg.norm(error)<self.settings.position_tolerance_m and abs(angular)<self.settings.rotation_tolerance_rad: break
                velocity=error*1.5; norm=np.linalg.norm(velocity)
                if norm>self.settings.base_speed_mps: velocity*=self.settings.base_speed_mps/norm
                local=rotation[:2,:2].T@velocity
                commands=wheel_commands(*local,float(np.clip(angular,-self.settings.base_yaw_rate,self.settings.base_yaw_rate)),self.wheels)
                for wheel,(steer,speed) in zip(self.wheels,commands):
                    d.ctrl[wheel["steer"]]=steer
                    actual=d.qpos[m.jnt_qposadr[wheel["steer_joint"]]]
                    d.ctrl[wheel["drive"]]=speed*max(0.,math.cos(steer-actual)) if abs(steer-actual)<.15 else 0.
                self.advance(self.settings.control_period)
        for wheel in self.wheels: d.ctrl[wheel["drive"]]=0
        self.settled()

    def dock(self,p):
        self.joint_move(p["transport"])
        interface=next((i for i in self.manifest["interfaces"] if i["id"]==p["interface"]),None)
        need(interface is not None and "site" in interface,"Docking target has no operation site")
        target=self.data.site_xpos[self.model.site(interface["site"]).id].copy()
        radii=p.get("radii",[p.get("radius",.65)])
        need(isinstance(radii,list) and 1<=len(radii)<=4,"Docking radius candidates must have 1 to 4 entries")
        choices=list(self.arms) if p["arm"]=="auto" else [p["arm"]]
        need(set(choices)<=set(self.arms),"Unknown docking arm")
        candidates=[]
        for radius in radii:
            number(radius,.2,2)
            for angle in np.linspace(-math.pi,math.pi,16,endpoint=False):
                pos=target[:2]+radius*np.array([math.cos(angle),math.sin(angle)])
                candidates.append([*pos,math.atan2(target[1]-pos[1],target[0]-pos[0])])
        candidates.sort(key=lambda x:np.linalg.norm(np.array(x[:2])-self.data.xpos[self.base_body,:2]))
        before_steps=self.sim_steps; failures=[]
        for candidate in candidates:
            clone=mujoco.MjData(self.model); copy_data(clone,self.model,self.data)
            q=int(self.base.qposadr[0]); clone.qpos[q:q+2]=candidate[:2]
            clone.qpos[q+3:q+7]=[math.cos(candidate[2]/2),0,0,math.sin(candidate[2]/2)]
            mujoco.mj_forward(self.model,clone)
            for arm in choices:
                rows=[{**r,"arm":arm if r["arm"]=="auto" else r["arm"]} for r in p["reach_targets"]]
                joints=self.joints_for(list(dict.fromkeys(r["arm"] for r in rows)),p.get("include_lift",False))
                _,qids,valid=self.query(joints,clone)
                if not valid(clone.qpos[qids]):continue
                try:
                    goal=Kinematics(self.model,clone,joints,valid,self.settings,self.rng).solve(self.targets(rows,clone))
                except OperationError as exc:
                    failures.append({"arm":arm,"pose":candidate,"code":exc.code,"evidence":exc.evidence});continue
                try:
                    result=self.navigate({**p,"goal":candidate},prepare_transport=False)
                    self.selected_arm=arm
                    if self.recorder:self.recorder.event({"type":"dock_selection","arm":arm,"pose":candidate,
                        "joint_candidate":dict(zip(joints,goal.tolist())),"qualification":"search_witness_not_completed_manipulation"})
                    return result
                except OperationError as exc:
                    if self.sim_steps!=before_steps or exc.code!="path_not_found":raise
                    failures.append({"arm":arm,"pose":candidate,"code":exc.code,"evidence":exc.evidence})
        raise OperationError("path_not_found","This bounded docking search found no solution; this is not a proof of unreachability",{"attempts":len(failures),"last_failures":failures[-8:]})

    def execute(self,node):
        self.begin(node)
        op,p=node["operation"],deepcopy(node["parameters"])
        if op!="dock":
            def resolve(value):
                if isinstance(value,list):return [resolve(v) for v in value]
                if not isinstance(value,dict):return value
                out={k:resolve(v) for k,v in value.items()}
                if out.get("arm")=="selected":
                    need(hasattr(self,"selected_arm"),"No arm has been selected by an executed docking")
                    out["arm"]=self.selected_arm
                return out
            p=resolve(p)
        if op not in {"navigate","dock","observe","check","wait","device"}: self.settled()
        if op=="observe":
            result={"time":float(self.data.time),"logical_time":self.logical_time,
                "qpos":self.data.qpos.tolist(),"qvel":self.data.qvel.tolist(),
                "body_position":self.data.xpos.tolist(),"body_quaternion":self.data.xquat.tolist(),
                "contacts":contacts(self.model,self.data),"devices":deepcopy(self.evidence.devices),
                "visibility":"planner_privileged_state"}
            if self.recorder:
                self.recorder.event({"type":"observation","node":self.node,**result})
                self.recorder.capture_images(self)
            return result
        if op=="check": return
        if op=="joint_move": return self.joint_move(p)
        if op=="navigate": return self.navigate(p)
        if op=="dock": return self.dock(p)
        if op=="cartesian": return self.cartesian(p,p.get("contact",False))
        if op=="dual_arm":
            need(len(p["targets"])==2,"A dual-arm node must have two targets")
            return self.cartesian(p,True)
        if op in {"press","insert"}: return self.axis_motion(p)
        if op=="rotate": return self.axis_motion(p,True)
        if op=="grasp": return self.grasp(p)
        if op=="place": return self.place(p)
        if op=="device":
            event=self.evidence.start(p["device"])
            return self.tick([event])
        if op=="wait":
            extra=number(p.get("logical_seconds",0),0,1e9)
            if extra:
                need(self.task.get("allow_logical_time") is True,"The task does not allow logical time acceleration")
                need(any(d["state"]=="running" for d in self.evidence.devices.values()),"No device is running; the wait cannot be skipped")
                self.tick(logical_extra=extra)
            return self.advance(number(p.get("seconds",self.settings.control_period),0,self.settings.max_sim_seconds))
        raise OperationError("capability_missing","Operation not supported yet")
