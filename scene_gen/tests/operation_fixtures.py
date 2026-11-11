"""开发者合成移动双臂，不能作为正式模型结果或真实硬件验证。"""
from pathlib import Path
import mujoco
from asset_prep.storage import atomic_json


def fixture(root, grasp_ready=False):
    root=Path(root); root.mkdir(parents=True)
    wheels=[]; actuators=[]
    for name,x,y in (("left",.13,.12),("right",.13,-.12),("rear",-.15,0)):
        wheels.append(f'''<body name="robot__{name}_wheel_steer" pos="{x} {y} -.05">
          <inertial pos="0 0 0" mass=".1" diaginertia=".001 .001 .001"/>
          <joint name="robot__{name}_wheel_steer_joint" axis="0 0 1" range="-1.58 1.58" damping="2"/>
          <body name="robot__{name}_wheel_drive"><joint name="robot__{name}_wheel_drive_joint" axis="0 1 0" limited="false" damping=".05"/>
          <geom name="robot__{name}_wheel" type="cylinder" size=".1 .025" quat=".70710678 .70710678 0 0" mass=".5" friction="1 .01 .001"/>
          </body></body>''')
        actuators.extend([f'<position name="robot__{name}_steer" joint="robot__{name}_wheel_steer_joint" kp="100" kv="10"/>',
                          f'<velocity name="robot__{name}_drive" joint="robot__{name}_wheel_drive_joint" kv="5" ctrlrange="-10 10"/>'])
    armxml=[]; arms=[]
    alljoints=[f"{name}_wheel_{kind}_joint" for name in ("left","right","rear") for kind in ("steer","drive")]
    for name,y in (("left",.2),("right",-.2)):
        joints=[name+"_"+axis for axis in "xyz"]
        fingers=[name+"_finger"+str(i) for i in (1,2)]
        armxml.append(f'<body name="robot__{name}_arm" pos="0 {y} .35"><joint name="robot__{name}_x" type="slide" axis="1 0 0" range="-.3 .5" damping="10"/><geom type="sphere" size=".025" mass=".1" contype="0" conaffinity="0"/>')
        armxml.append(f'<body name="robot__{name}_slide_y"><joint name="robot__{name}_y" type="slide" axis="0 1 0" range="-.2 .2" damping="10"/><geom type="sphere" size=".02" mass=".1" contype="0" conaffinity="0"/>')
        armxml.append(f'<body name="robot__{name}_hand"><joint name="robot__{name}_z" type="slide" axis="0 0 1" range="-.3 .3" damping="10"/><geom type="sphere" size=".015" mass=".1" contype="0" conaffinity="0"/><site name="robot__{name}_tcp"/><site name="robot__{name}_wrist"/>')
        for index,sign in ((1,-1),(2,1)):
            armxml.append(f'<body name="robot__{name}_finger_body{index}" pos="0 {sign*.04} 0"><joint name="robot__{name}_finger{index}" type="slide" axis="0 {-sign} 0" range="0 .03" damping="1"/><geom name="robot__{name}_finger_geom{index}" type="box" size=".02 .006 .015" mass=".02"/></body>')
            actuators.append(f'<position name="robot__{name}_grip{index}" joint="robot__{name}_finger{index}" kp="100" kv="2" ctrlrange="0 .03"/>')
        armxml.append('</body></body></body>')
        actuators.extend(f'<position name="robot__{j}_motor" joint="robot__{j}" kp="1000" kv="40" ctrlrange="-.5 .5"/>' for j in joints)
        arms.append({"id":name,"joints":joints,"tcp_site":name+"_tcp","wrist_site":name+"_wrist","shoulder_site":None,
                     "reach_m":1.,"payload_kg":1.,"stock":{"joints":fingers,"adapter_site":name+"_wrist","mass_kg":.04,"name":"fixture"}})
        alljoints+=joints+fingers
    object_position="0 .2 .5" if grasp_ready else ".6 0 .03"
    object_size=".02 .02 .02" if grasp_ready else ".03 .03 .03"
    support='<body name="support" pos="0 .2 .4"><geom name="support_geom" type="box" size=".08 .08 .03"/></body>' if grasp_ready else ''
    xml=f'''<mujoco><compiler angle="radian"/><option timestep=".002" integrator="implicitfast"/>
      <worldbody><light pos="0 0 3"/><geom name="environment__floor" type="plane" size="3 3 .1"/>
      <body name="robot__layout_root" pos="0 0 .15">
      <joint name="robot__base_x" type="slide" axis="1 0 0"/><joint name="robot__base_y" type="slide" axis="0 1 0"/><joint name="robot__base_yaw" axis="0 0 1"/>
      <geom name="robot__chassis" type="box" pos="0 0 .11" size=".17 .12 .04" mass="10"/>
      {''.join(wheels)}{''.join(armxml)}</body>
      {support}<body name="object__body" pos="{object_position}"><freejoint name="object__free"/><geom name="object__geom" type="box" size="{object_size}" mass=".01" friction="1 .01 .001"/><site name="object__site"/></body>
      </worldbody><actuator>{''.join(actuators)}</actuator></mujoco>'''
    (root/"model.xml").write_text(xml,encoding="utf-8")
    model=mujoco.MjModel.from_xml_path(str(root/"model.xml")); data=mujoco.MjData(model)
    if grasp_ready:
        for index in (1,2):
            data.qpos[model.joint(f"robot__left_finger{index}").qposadr[0]]=.016
            data.ctrl[model.actuator(f"robot__left_grip{index}").id]=.016
    env={"robot":{"arms":arms,"joints":alljoints,"shared_joints":alljoints[:6],"model_variants":{"":"model.xml"}}}
    plan={"instances":[{"id":"object","job":"asset","placement":"table"}],"end_effectors":{
        arm:{"mode":"stock","instance":None,"reason":"开发者合成原厂夹爪"} for arm in ("left","right")}}
    atomic_json(root/"scene.json",{"plan":plan,"environment":env,"home_qpos":data.qpos.tolist(),"home_ctrl":data.ctrl.tolist()})
    atomic_json(root/"interfaces.json",{"interfaces":[{"id":"object__grasp","kind":"grasp","body":"object__body","site":"object__site","asset_id":"A1"}],"parts":[],"buttons":[],"connections":[],"mechanical_events":[]})
    return root


def task_graph(scene_hash):
    from asset_prep.storage import digest
    condition={"kind":"joint_range","joint":"robot__left_x","range":[-.001,.001]}
    task={"version":1,"input_sha256":"developer_fixture","public_profile_sha256":"developer_fixture",
          "provenance":"developer_fixture","bindings":{"object":"object__body"},"initial":condition,
          "stages":[{"id":"check","source":"developer fixture","depends_on":[],"goal":condition}],"final":condition}
    graph={"version":1,"scene_sha256":scene_hash,"task_sha256":digest(task),"provenance":"developer_fixture","entry":"one",
           "nodes":[{"id":"one","operation":"wait","stage":"check","parameters":{"seconds":.1},
                     "precondition":condition,"completion":condition,"timeout":2,"on_success":"success","on_failure":"failed"}]}
    return task,graph
