import os, json

def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    xml = r'''<mujoco model="A10_vortex_mixer">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom condim="3" friction="0.8 0.25 0.2" density="900"/>
    <joint damping="0.15" armature="0.001"/>
  </default>
  <asset>
    <material name="dark_gray" rgba="0.16 0.17 0.18 1"/>
    <material name="black" rgba="0.035 0.04 0.045 1"/>
    <material name="rubber" rgba="0.025 0.025 0.025 1"/>
    <material name="silver" rgba="0.55 0.57 0.58 1"/>
    <material name="white" rgba="0.88 0.88 0.86 1"/>
    <material name="red" rgba="0.75 0.06 0.04 1"/>
    <material name="blue" rgba="0.04 0.18 0.55 1"/>
  </asset>
  <worldbody>
    <body name="vortex_base" pos="0 0 0.09">
      <geom name="base_main" type="box" size="0.061 0.0825 0.075" material="dark_gray"/>
      <geom name="base_lower" type="box" pos="0 0 -0.062" size="0.064 0.085 0.013" material="black"/>
      <geom name="front_panel" type="box" pos="0 -0.080 0.004" size="0.048 0.003 0.045" material="black"/>
      <geom name="front_label" type="box" pos="0 -0.084 0.026" size="0.041 0.001 0.014" material="white" contype="0" conaffinity="0"/>
      <geom name="label_red_band" type="box" pos="0 -0.085 0.041" size="0.040 0.001 0.002" material="red" contype="0" conaffinity="0"/>
      <geom name="foot_left" type="cylinder" pos="-0.043 0.060 -0.078" size="0.009 0.006" material="rubber"/>
      <geom name="foot_right" type="cylinder" pos="0.043 0.060 -0.078" size="0.009 0.006" material="rubber"/>
      <geom name="foot_front_left" type="cylinder" pos="-0.043 -0.060 -0.078" size="0.009 0.006" material="rubber"/>
      <geom name="foot_front_right" type="cylinder" pos="0.043 -0.060 -0.078" size="0.009 0.006" material="rubber"/>
      <body name="cup_head" pos="0 0 0.083">
        <freejoint name="cup_head_free"/>
        <geom name="cup_stem" type="cylinder" size="0.018 0.008" material="black"/>
        <body name="cup_platform" pos="0 0 0.012">
          <joint name="cup_rotation" type="hinge" axis="0 0 1" limited="true" range="-0.25 0.25"/>
          <geom name="cup_platform_disc" type="cylinder" size="0.031 0.006" material="black"/>
          <geom name="cup_platform_rim" type="cylinder" pos="0 0 0.006" size="0.029 0.002" material="silver"/>
          <site name="cup_grasp_site" pos="0 0 0.010" size="0.004" rgba="0.1 0.8 0.1 1"/>
        </body>
      </body>
      <body name="speed_knob" pos="0.025 -0.088 0.006">
        <joint name="speed_knob_rotate" type="hinge" axis="0 1 0" limited="true" range="-1.57 1.57"/>
        <geom name="speed_knob" type="cylinder" size="0.012 0.006" quat="0.7071 0.7071 0 0" material="black"/>
        <site name="speed_knob_site" pos="0 -0.008 0" size="0.003"/>
      </body>
      <body name="touch_switch" pos="-0.027 -0.088 0.019">
        <joint name="touch_switch_slide" type="slide" axis="0 0 1" limited="true" range="0 0.002"/>
        <geom name="touch_switch" type="box" size="0.006 0.004 0.004" material="silver"/>
        <site name="touch_switch_site" pos="0 -0.005 0" size="0.0025"/>
      </body>
      <body name="holder_cradle" pos="0.080 0 0.010">
        <geom name="cradle_shelf" type="box" pos="0.025 0 0.006" size="0.026 0.060 0.006" material="black"/>
        <geom name="cradle_back" type="box" pos="0.052 0 0.027" size="0.004 0.060 0.021" material="black"/>
      </body>
    </body>
    <body name="tube_holder" pos="0.105 0 0.030">
      <freejoint name="tube_holder_free"/>
      <geom name="holder_disc" type="cylinder" size="0.057 0.004" material="white"/>
      <geom name="holder_rim" type="cylinder" pos="0 0 0.0045" size="0.058 0.002" material="silver"/>
      <site name="holder_grasp_site" pos="0 0 0.010" size="0.004" rgba="0.1 0.8 0.1 1"/>
      <site name="holder_mate_site" pos="0 0 -0.006" size="0.003"/>
      <body name="clip_01" pos="0.050 0 0.006" euler="0 0 0">
        <geom name="clip_01_a" type="box" pos="0 0.006 0" size="0.009 0.0025 0.006" material="white"/>
        <geom name="clip_01_b" type="box" pos="0 -0.006 0" size="0.009 0.0025 0.006" material="white"/>
      </body>
'''
    for i in range(1,24):
        import math
        a = 2.0*math.pi*i/24.0
        x,y = 0.050*math.cos(a),0.050*math.sin(a)
        # radial jaw pair; each pair leaves a real tube slot between jaws
        xml += f'''      <body name="clip_{i+1:02d}" pos="{x:.6f} {y:.6f} 0.006" euler="0 0 {a:.6f}">
        <geom name="clip_{i+1:02d}_a" type="box" pos="0 0.006 0" size="0.009 0.0025 0.006" material="white"/>
        <geom name="clip_{i+1:02d}_b" type="box" pos="0 -0.006 0" size="0.009 0.0025 0.006" material="white"/>
      </body>
'''
    xml += '''    </body>
  </worldbody>
  <equality>
    <weld name="cup_head_mount" body1="vortex_base" body2="cup_head" solref="0.01 1" solimp="0.9 0.95 0.01"/>
  </equality>
  <contact>
    <exclude body1="vortex_base" body2="cup_head"/>
  </contact>
</mujoco>
'''
    with open(os.path.join(output_dir, "model.xml"), "w", encoding="utf-8") as f:
        f.write(xml)
    interfaces = {
        "schema_version":"1.0",
        "asset_ids":["A10"],
        "parts":[
            {"id":"A10_vortex_mixer","body":"vortex_base"},
            {"id":"A10_cup_head","body":"cup_head"},
            {"id":"A10_horizontal_24_position_holder","body":"tube_holder"}
        ],
        "interfaces":[
            {"id":"A10_cup_grasp","asset_id":"A10","kind":"grasp","body":"cup_head","site":"cup_grasp_site","approach_axis":[0,0,1]},
            {"id":"A10_holder_grasp","asset_id":"A10","kind":"grasp","body":"tube_holder","site":"holder_grasp_site","approach_axis":[0,0,1]},
            {"id":"A10_short_mix_contact","asset_id":"A10","kind":"press","body":"cup_head","site":"cup_grasp_site","approach_axis":[0,0,1],"operation":{"status":"declared","axis":[0,0,-1],"joint":"cup_rotation","range":[-0.25,0.25],"clearance_m":0.002,"grasp_geoms":["cup_platform_disc"],"basis":{"range":{"kind":"assumption","reference":"短时接触混合的工程近似"}}}},
            {"id":"A10_speed_control","asset_id":"A10","kind":"rotate","body":"speed_knob","site":"speed_knob_site","approach_axis":[0,-1,0],"operation":{"status":"declared","axis":[0,1,0],"joint":"speed_knob_rotate","range":[-1.57,1.57],"basis":{"range":{"kind":"assumption","reference":"面板旋钮可调范围建模值"}}}},
            {"id":"A10_touch_start","asset_id":"A10","kind":"press","body":"touch_switch","site":"touch_switch_site","approach_axis":[0,-1,0],"operation":{"status":"declared","axis":[0,0,1],"joint":"touch_switch_slide","range":[0,0.002],"basis":{"range":{"kind":"assumption","reference":"图示接触启动开关的建模行程"}}}}
        ],
        "assumptions":[
            "未提供制造商三视图；主体采用规格外廓165 mm深、122 mm宽、165 mm高的工程化近似。",
            "杯头、管架夹片、底脚和侧置管架收纳托盘的尺寸为建模假设。",
            "水平管架按直径约114 mm的圆盘、24个径向双爪夹位建模，夹口保留真实贯通间隙。",
            "管架初始状态放置在主体侧置收纳托盘内；杯头通过可释放焊接约束装配在主体顶部。"
        ],
        "approximations":[
            "持续振荡用杯头绕Z轴的小角度铰链近似；未模拟偏心旋转、电机和振幅控制。",
            "杯头与主体的可拆装关系用不同运动树上的freejoint加weld等式表达。",
            "管架夹片用成对刚体块近似柔性塑料卡扣，保持插槽的实际空隙。"
        ],
        "buttons":[
            {"id":"A10_touch_button","joint":"touch_switch_slide","geom":"touch_switch","pressed_at":0.002,"released_at":0.0005,"event":"A10_touch_start_event"}
        ],
        "connections":[
            {"id":"A10_detachable_cup","kind":"detachable","equality":"cup_head_mount","release_event":"A10_touch_start_event","irreversible":false,"attach_event":"A10_touch_start_event","capture_sites":["cup_grasp_site","cup_grasp_site"],"position_tolerance_m":0.005,"angle_tolerance_rad":0.2}
        ],
        "mechanical_events":[
            {"id":"A10_touch_start_event","interface":"A10_touch_start","action":"release"}
        ]
    }
    with open(os.path.join(output_dir, "interfaces.json"), "w", encoding="utf-8") as f:
        json.dump(interfaces, f, ensure_ascii=False, indent=2)
