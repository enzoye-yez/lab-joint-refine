"""实际世界只通过执行器推进；候选搜索和恢复仅用于显式试错。"""
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
            need(isinstance(body,str) and mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)>=0,"任务对象身份未绑定实际身体")
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
        need(set(self.devices_by_interface)<=software,"设备请求必须绑定已声明的软件接口")
        events={e["id"] for e in manifest.get("mechanical_events",[])}
        need({e["event"] for e in task.get("mechanical_detectors",[])}<=events,"机械检测事件未声明")
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
            need(len(ga)==1,"轮组须有唯一接触圆柱；其他机器人需显式轮组适配")
            actuator=lambda j: next((a for a in range(m.nu) if m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a,0]==j.id),None)
            sa,da=actuator(steer),actuator(drive)
            need(sa is not None and da is not None,"轮组缺少驱动")
            need(m.actuator_biasprm[sa,1]<0 and m.actuator_biasprm[da,2]<0 and m.actuator_biasprm[da,1]==0,"轮组执行器类型不匹配")
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
        need(self.recorder is None or self.recorder.mode=="trial","正式连续执行禁止恢复检查点")
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
            need(isinstance(pair,list) and len(pair)==2,"允许接触须为两个身体名称")
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
        need(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not any(w.number for w in d.warning),"仿真不稳定","simulation_unstable")
        tilt=math.acos(float(np.clip(d.xmat[self.base_body].reshape(3,3)[2,2],-1,1)))
        need(tilt<=s.base_tilt_rad,"底盘倾斜越界","base_unstable")
        for row in self.evidence.contact_rows:
            a,b=row["geoms"]
            need(row["distance"]>=-s.penetration_m,"接触穿透超限","collision")
            if a in self.robot_geoms or b in self.robot_geoms:
                need(self.acceptable_contact(a,b),"非预期机器人接触","collision")
                if not any(w["geom"] in (a,b) for w in self.wheels):
                    need(row["force_n"]<=s.contact_force_n,"操作接触力超过工程边界","contact_limit")
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in (2,3):
                value=d.qpos[m.jnt_qposadr[j]]
                need(m.jnt_range[j,0]-.03<=value<=m.jnt_range[j,1]+.03,"观测关节越界","joint_limit")
        for rule in self.task.get("invariants",[]):
            need(self.evidence.evaluate(rule),"任务不变量失效","task_invariant")
        for rule in self.held_checks.values():
            need(self.evidence.evaluate({**rule,"duration":0}),"持有物滑落或夹持失效","grasp_lost")

    def tick(self, software=(), logical_extra=0):
        getattr(self,"cancel_guard",lambda:None)()
        need(self.data.time < self.deadline+1e-9,"节点执行超时","timeout")
        need(self.data.time < self.settings.max_sim_seconds,"仿真总预算耗尽","budget_exhausted")
        before=float(self.data.time)
        controls=self.data.ctrl.copy()
        mujoco.mj_step(self.model,self.data)
        # 更新用于证据的运动学和约束力，不改变积分状态。
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
                    '接触对象的机械事件缺少检测器','capability_missing')
            # 机械阈值还必须有执行主体与该接口部件的真实接触，不能靠无关关节触发。
            for event in mechanical:
                target=self.geoms(self.mechanical_interfaces[event]["body"])
                actors=set().union(*(set(gs) for gs in self.actors.values()))
                actors |= set().union(*(self.geoms(b) for b in self.holding.values())) if self.holding else set()
                need(bool(self.evidence.touching(actors,target)),"机械事件缺少操作主体接触证据","mechanical_evidence_missing")
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
            need(j.id in self.position_actuators,"关节没有位置伺服","capability_missing")
            a=self.position_actuators[j.id]
            need(j.range[0]<=value<=j.range[1],"位置命令超出关节范围")
            if self.model.actuator_ctrllimited[a]:
                need(self.model.actuator_ctrlrange[a,0]<=value<=self.model.actuator_ctrlrange[a,1],"控制命令超出执行器范围")
            self.data.ctrl[a]=value

    def joints_for(self,arms_selected,include_lift=False):
        need(arms_selected and len(set(arms_selected))==len(arms_selected) and set(arms_selected)<=set(self.arms),"机械臂选择无效")
        joints=["robot__"+j for arm in arms_selected for j in self.arms[arm]["joints"]]
        if include_lift:
            shared=self.meta["environment"]["robot"].get("shared_joints",[])
            # Shared vertical prismatic mechanisms, not an arbitrary hard-coded joint name.
            lifts=["robot__"+j for j in shared if self.model.joint("robot__"+j).type[0]==2
                   and abs(float(self.model.jnt_axis[self.model.joint("robot__"+j).id,2]))>.99]
            need(len(lifts)==1,"升降机构缺少唯一的实际共享关节")
            joints.extend(lifts)
        return joints

    def query(self,joints,base_data=None):
        m,d=self.model,self.data
        clone=mujoco.MjData(m); copy_data(clone,m,base_data if base_data is not None else d)
        qids=[int(m.joint(j).qposadr[0]) for j in joints]
        # 携带物仅在候选副本中随末端变换；执行世界仍只使用夹爪接触。
        carry=[]
        for arm,body in self.holding.items():
            bid=m.body(body).id; sid=m.site(self.tcps[arm]).id
            jid=int(m.body_jntadr[bid])
            need(m.body_jntnum[bid]>0 and m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE,"携带物需要独立自由根","capability_missing")
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
            else:raise OperationError("tracking_failed","有界关节跟踪未达到目标；保留真实误差")

    def joint_move(self,p):
        selected=p["arms"]; allowed=self.joints_for(selected,p.get("include_lift",False))
        joints=list(p["positions"])
        need(set(joints)<=set(allowed) and joints,"运动只能控制所选机械臂及明确选择的升降关节")
        _,qids,valid=self.query(joints)
        limits=np.array([self.model.joint(j).range for j in joints])
        goal=vector([p["positions"][j] for j in joints],len(joints))
        path=rrt_connect(self.data.qpos[qids].copy(),goal,limits,valid,self.rng,self.settings.rrt_iterations,self.settings.path_resolution,start_tolerance=self.settings.joint_tolerance)
        self.follow(joints,path)

    def targets(self,rows,data=None):
        data=self.data if data is None else data
        result=[]
        for row in rows:
            arm=row["arm"]; need(arm in self.tcps,"未知机械臂")
            if "interface" in row:
                interface=next((i for i in self.manifest["interfaces"] if i["id"]==row["interface"]),None)
                need(interface is not None and interface.get("site"),"操作目标接口缺少实际坐标点")
                need("frame" not in row,"接口目标不得同时指定其他坐标系")
                row={**row,"frame":interface["site"]}
            position=vector(row["position"],3)
            quat=vector(row["quaternion"],4)
            need(abs(np.linalg.norm(quat)-1)<1e-5,"目标四元数须归一化")
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
            predicate(stop);need(contact,"停止条件仅用于有界接触路径")
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
            need(edge_valid(self.data.qpos[qids],q,valid,self.settings.path_resolution),"接触路径中间状态受阻","path_not_found")
            # 接触负荷升高时减速；越过硬阈值由 guard 停止并进入显式恢复分支。
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
            need(self.evidence.evaluate(stop),"限定接触行程内未满足停止条件","contact_not_found")

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
                if attempt==100:raise OperationError("tracking_error","末端未在跟踪预算内稳定",self.last_tracking)
                self.advance(self.settings.control_period);continue
            stable_since=None
            if attempt==100:raise OperationError("tracking_error","末端闭环跟踪预算内未达到目标",self.last_tracking)
            _,_,valid=self.query(joints)
            kin=Kinematics(self.model,self.data,joints,valid,self.settings,self.rng)
            q=kin.solve(targets)
            need(edge_valid(self.data.qpos[qids],q,valid,self.settings.path_resolution),"跟踪修正边受阻","path_not_found")
            command=np.array([self.data.ctrl[self.position_actuators[self.model.joint(j).id]] for j in joints])
            correction=np.clip((q-self.data.qpos[qids])*.5,-self.settings.joint_speed*self.settings.control_period,self.settings.joint_speed*self.settings.control_period)
            self.controls_for(joints,np.clip(command+correction,kin.limits[:,0],kin.limits[:,1]))
            self.advance(self.settings.control_period)

    def gripper(self,arm,values,*,apply=True):
        need(arm in self.actors and isinstance(values,dict) and values,"夹爪指令缺失")
        bodies={int(self.model.geom_bodyid[g]) for g in self.actors[arm]}
        commands={}
        for name,value in values.items():
            a=self.model.actuator(name).id; j=int(self.model.actuator_trnid[a,0])
            need(self.model.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and int(self.model.jnt_bodyid[j]) in bodies,"不能借夹爪指令控制其他执行器")
            number(value,-1e5,1e5)
            if self.model.actuator_ctrllimited[a]: need(self.model.actuator_ctrlrange[a,0]<=value<=self.model.actuator_ctrlrange[a,1],"夹爪命令越界")
            commands[a]=value
        if apply:
            for actuator,value in commands.items():self.data.ctrl[actuator]=value

    def grasp(self,p):
        arm,body=p["arm"],p["body"]
        need(arm not in self.holding,"此手臂已有携带物")
        need(bool(self.geoms(body)),"抓取对象没有几何")
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
            need(set(p["hold_controls"])==set(targets),"保持命令必须对应同一组夹爪执行器")
            self.gripper(arm,p["hold_controls"])
            self.advance(self.settings.stable_seconds)
            while not self.evidence.evaluate(check):self.advance(self.settings.control_period)
        self.evidence.lock_hold(check)
        self.holding[arm]=body; self.held_checks[arm]=check

    def place(self,p):
        arm=p["arm"]; need(self.holding.get(arm)==p["body"],"所选手臂未持有所选对象","wrong_object")
        if "approach" in p: self.cartesian(p["approach"],True)
        self.gripper(arm,p["controls"])
        self.holding.pop(arm); self.held_checks.pop(arm)
        if "retreat" in p: self.cartesian(p["retreat"],True)
        check={"kind":"supported","body":p["body"],"support":p["support"],"duration":self.settings.stable_seconds}
        while not self.evidence.evaluate(check): self.advance(self.settings.control_period)

    def axis_motion(self,p,rotate=False):
        arm=p["arm"]; sid=self.model.site(self.tcps[arm]).id
        axis=vector(p["axis"],3); need(abs(np.linalg.norm(axis)-1)<1e-5,"操作轴必须归一化")
        if "frame" in p: axis=self.data.site_xmat[self.model.site(p["frame"]).id].reshape(3,3)@axis
        position=self.data.site_xpos[sid].copy(); rotation=self.data.site_xmat[sid].reshape(3,3).copy()
        if rotate:
            # 旋转必须沿圆弧，不以起终点的直线插值替代绕轴运动。
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
            # 携带物随底盘刚性变换仅用于碰撞查询。
            for arm,body in self.holding.items():
                bid=m.body(body).id; jid=int(m.body_jntadr[bid]); address=int(m.jnt_qposadr[jid])
                need(m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE,"运输物需独立自由根")
                sid=m.site(self.tcps[arm]).id; oldrot=d.site_xmat[sid].reshape(3,3); newrot=clone.site_xmat[sid].reshape(3,3)
                clone.qpos[address:address+3]=clone.site_xpos[sid]+newrot@oldrot.T@(d.xpos[bid]-d.site_xpos[sid])
                mujoco.mju_mat2Quat(clone.qpos[address+3:address+7],(newrot@oldrot.T@d.xmat[bid].reshape(3,3)).reshape(9))
            mujoco.mj_forward(m,clone)
            return self.collision_valid(clone)
        route=astar(d.qpos[q:q+2],target[:2],p["bounds"],valid,self.settings.grid_m)
        # 保留经过扫掠检查的拐点，避免在每个栅格中心反复停靠。
        simplified=[route[0]]; index=0
        while index<len(route)-1:
            for candidate in range(min(len(route)-1,index+20),index,-1):
                if edge_valid(route[index],route[candidate],valid,self.settings.grid_m/3):
                    simplified.append(route[candidate]); index=candidate; break
        route=simplified
        delta=(target[2]-yaw+math.pi)%(2*math.pi)-math.pi
        need(all(valid(target[:2],yaw+t*delta) for t in np.linspace(0,1,max(2,int(abs(delta)/.03)+1))),"终点转向扫掠受阻","path_not_found")
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
        need(interface is not None and "site" in interface,"停靠目标没有操作坐标")
        target=self.data.site_xpos[self.model.site(interface["site"]).id].copy()
        radii=p.get("radii",[p.get("radius",.65)])
        need(isinstance(radii,list) and 1<=len(radii)<=4,"停靠半径候选须为1至4项")
        choices=list(self.arms) if p["arm"]=="auto" else [p["arm"]]
        need(set(choices)<=set(self.arms),"未知停靠机械臂")
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
        raise OperationError("path_not_found","本次有界停靠搜索未找到方案；不构成不可达证明",{"attempts":len(failures),"last_failures":failures[-8:]})

    def execute(self,node):
        self.begin(node)
        op,p=node["operation"],deepcopy(node["parameters"])
        if op!="dock":
            def resolve(value):
                if isinstance(value,list):return [resolve(v) for v in value]
                if not isinstance(value,dict):return value
                out={k:resolve(v) for k,v in value.items()}
                if out.get("arm")=="selected":
                    need(hasattr(self,"selected_arm"),"尚无已执行停靠选定的机械臂")
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
            need(len(p["targets"])==2,"双臂节点必须有两个目标")
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
                need(self.task.get("allow_logical_time") is True,"任务未允许逻辑时间加速")
                need(any(d["state"]=="running" for d in self.evidence.devices.values()),"没有运行设备，不能跳过等待")
                self.tick(logical_extra=extra)
            return self.advance(number(p.get("seconds",self.settings.control_period),0,self.settings.max_sim_seconds))
        raise OperationError("capability_missing","尚未支持的操作")
