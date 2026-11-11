import os
import json
import math


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    xml = r'''<mujoco model="centrifuge_5424R_A11">
  <compiler angle="radian" coordinate="local" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom contype="1" conaffinity="1" density="600" friction="0.7 0.25 0.15"/>
    <joint damping="0.08" armature="0.001"/>
    <site size="0.006" rgba="0.1 0.8 0.9 1"/>
  </default>
  <asset>
    <material name="housing_white" rgba="0.82 0.84 0.84 1"/>
    <material name="housing_dark" rgba="0.12 0.14 0.15 1"/>
    <material name="panel_dark" rgba="0.08 0.09 0.10 1"/>
    <material name="button_green" rgba="0.18 0.55 0.22 1"/>
    <material name="button_red" rgba="0.72 0.12 0.10 1"/>
    <material name="button_gray" rgba="0.43 0.46 0.47 1"/>
    <material name="rotor_metal" rgba="0.38 0.42 0.44 1"/>
    <material name="rotor_dark" rgba="0.08 0.10 0.11 1"/>
    <material name="lid_window" rgba="0.18 0.24 0.26 0.55"/>
    <material name="label_white" rgba="0.95 0.95 0.92 1"/>
  </asset>
  <worldbody>
    <body name="centrifuge_base" pos="0 0 0.115">
      <geom name="base_lower_shell" type="box" pos="0 0 0" size="0.145 0.235 0.115" material="housing_white"/>
      <geom name="base_front_lower" type="box" pos="0 -0.238 0.025" size="0.135 0.008 0.075" material="housing_white"/>
      <geom name="base_top_deck" type="box" pos="0 0 0.112" size="0.138 0.224 0.012" material="housing_white"/>
      <geom name="left_side_vent" type="box" pos="-0.147 0.035 0.075" size="0.004 0.105 0.055" material="panel_dark"/>
      <geom name="right_side_vent" type="box" pos="0.147 0.035 0.075" size="0.004 0.105 0.055" material="panel_dark"/>
      <geom name="front_control_panel" type="box" pos="0 -0.247 0.128" size="0.132 0.009 0.043" material="panel_dark"/>
      <geom name="display_window" type="box" pos="-0.035 -0.258 0.151" size="0.060 0.004 0.014" material="housing_dark"/>
      <geom name="brand_badge" type="box" pos="0.078 -0.259 0.062" size="0.040 0.003 0.008" material="label_white"/>

      <body name="button_start_body" pos="0.091 -0.261 0.145">
        <joint name="start_button_joint" type="slide" axis="0 0 1" range="0 0.004"/>
        <geom name="start_button_geom" type="cylinder" axisangle="1 0 0 1.5708" size="0.012 0.004" material="button_green"/>
        <site name="start_button_site" pos="0 0 0.006"/>
      </body>
      <body name="button_stop_body" pos="0.091 -0.261 0.112">
        <joint name="stop_button_joint" type="slide" axis="0 0 1" range="0 0.004"/>
        <geom name="stop_button_geom" type="cylinder" axisangle="1 0 0 1.5708" size="0.012 0.004" material="button_red"/>
        <site name="stop_button_site" pos="0 0 0.006"/>
      </body>
      <body name="button_unlock_body" pos="0.058 -0.261 0.111">
        <joint name="unlock_button_joint" type="slide" axis="0 0 1" range="0 0.004"/>
        <geom name="unlock_button_geom" type="cylinder" axisangle="1 0 0 1.5708" size="0.009 0.004" material="button_gray"/>
        <site name="unlock_button_site" pos="0 0 0.006"/>
      </body>

      <body name="rotor_body" pos="0 0 0.132">
        <joint name="rotor_spin_joint" type="hinge" axis="0 0 1" range="-6.283 6.283" damping="0.03" armature="0.002"/>
        <geom name="rotor_hub" type="cylinder" pos="0 0 0.018" size="0.030 0.014" material="rotor_metal"/>
        <geom name="rotor_spoke_n" type="box" pos="0 0.060 0.018" size="0.015 0.060 0.006" material="rotor_metal"/>
        <geom name="rotor_spoke_s" type="box" pos="0 -0.060 0.018" size="0.015 0.060 0.006" material="rotor_metal"/>
        <geom name="rotor_spoke_e" type="box" pos="0.060 0 0.018" size="0.060 0.015 0.006" material="rotor_metal"/>
        <geom name="rotor_spoke_w" type="box" pos="-0.060 0 0.018" size="0.060 0.015 0.006" material="rotor_metal"/>
        <geom name="rotor_ring_outer" type="torus" pos="0 0 0.020" size="0.073 0.004" material="rotor_metal"/>
        <geom name="rotor_ring_inner" type="torus" pos="0 0 0.021" size="0.040 0.004" material="rotor_dark"/>
        <site name="rotor_load_site" pos="0 0 0.034"/>
        <site name="rotor_cover_mate_site" pos="0 0 0.039"/>
'''
    # Add 24 annular tube positions. The toroidal geometry leaves the tube openings visibly open.
    for i in range(24):
        a = 2.0 * math.pi * i / 24.0
        r = 0.084
        x, y = r * math.cos(a), r * math.sin(a)
        xml += f'        <geom name="tube_well_ring_{i+1:02d}" type="torus" pos="{x:.6f} {y:.6f} 0.027" size="0.0085 0.0022" material="rotor_dark"/>' + "\n"
    xml += r'''      </body>

      <body name="lid_hinge_body" pos="0 0.205 0.225">
        <joint name="lid_hinge_joint" type="hinge" axis="1 0 0" range="0 1.85" damping="0.16"/>
        <geom name="lid_shell" type="box" pos="0 -0.100 0.026" size="0.140 0.108 0.026" material="housing_white"/>
        <geom name="lid_front_lip" type="box" pos="0 -0.207 0.006" size="0.138 0.009 0.020" material="housing_white"/>
        <geom name="lid_window" type="cylinder" pos="0 -0.092 0.056" axisangle="1 0 0 0" size="0.073 0.004" material="lid_window"/>
        <geom name="lid_center_handle" type="cylinder" pos="0 -0.092 0.066" axisangle="1 0 0 0" size="0.012 0.005" material="housing_dark"/>
        <site name="lid_grasp_site" pos="0 -0.205 0.029"/>
        <site name="lid_lock_site" pos="0 -0.218 0.010"/>
      </body>
    </body>

    <body name="rotor_cover_body" pos="0 0 0.194">
      <freejoint name="rotor_cover_freejoint"/>
      <geom name="rotor_cover_disk" type="cylinder" pos="0 0 0.010" size="0.101 0.008" material="rotor_dark"/>
      <geom name="rotor_cover_center_cap" type="cylinder" pos="0 0 0.021" size="0.017 0.004" material="rotor_metal"/>
      <site name="rotor_cover_grasp_site" pos="0 0 0.031"/>
      <site name="rotor_cover_capture_site" pos="0 0 0.000"/>
    </body>
  </worldbody>
  <equality>
    <weld name="rotor_cover_lock" body1="rotor_cover_body" body2="rotor_body" solref="0.02 1" solimp="0.9 0.95 0.01"/>
  </equality>
</mujoco>
'''

    with open(os.path.join(output_dir, "model.xml"), "w", encoding="utf-8") as f:
        f.write(xml)

    interfaces = {
        "schema_version": "1.0",
        "asset_ids": ["A11"],
        "parts": [
            {"id": "centrifuge_shell", "body": "centrifuge_base"},
            {"id": "rotor_24_hole", "body": "rotor_body"},
            {"id": "rotor_cover", "body": "rotor_cover_body"},
            {"id": "hinged_lid", "body": "lid_hinge_body"}
        ],
        "interfaces": [
            {"id": "A11_mount", "asset_id": "A11", "kind": "mount", "body": "centrifuge_base", "site": "rotor_load_site", "approach_axis": [0, 0, 1]},
            {"id": "A11_load_tubes", "asset_id": "A11", "kind": "insert", "body": "rotor_body", "site": "rotor_load_site", "approach_axis": [0, 0, 1]},
            {"id": "A11_lid_grasp", "asset_id": "A11", "kind": "grasp", "body": "lid_hinge_body", "site": "lid_grasp_site", "approach_axis": [0, -1, 0], "operation": {"status": "declared", "axis": [1, 0, 0], "joint": "lid_hinge_joint", "range": [0, 1.85], "clearance_m": 0.02, "grasp_geoms": ["lid_front_lip"], "basis": {"range": {"kind": "assumption", "reference": "hinged upper cover opening envelope"}}}},
            {"id": "A11_rotor_cover_grasp", "asset_id": "A11", "kind": "grasp", "body": "rotor_cover_body", "site": "rotor_cover_grasp_site", "approach_axis": [0, 0, 1]},
            {"id": "A11_start", "asset_id": "A11", "kind": "press", "body": "button_start_body", "site": "start_button_site", "approach_axis": [0, 0, 1]},
            {"id": "A11_stop", "asset_id": "A11", "kind": "press", "body": "button_stop_body", "site": "stop_button_site", "approach_axis": [0, 0, 1]},
            {"id": "A11_unlock", "asset_id": "A11", "kind": "press", "body": "button_unlock_body", "site": "unlock_button_site", "approach_axis": [0, 0, 1]},
            {"id": "A11_remove_tubes", "asset_id": "A11", "kind": "grasp", "body": "rotor_body", "site": "rotor_load_site", "approach_axis": [0, 0, 1]}
        ],
        "assumptions": [
            "局部坐标原点位于设备底面中心，设备支撑面为z=0；外形尺寸采用290 mm宽、480 mm深、260 mm高。",
            "前方定义为局部-y方向，控制面、启动/停止/解锁按钮和开盖抓取位置布置在前侧。",
            "转子中心、24孔等角度分布、孔半径、转子盖厚度和盖体捕获位置为建模假设，未宣称为厂家测量值。",
            "转子盖以独立自由体和weld等式表达，便于后续取下/重新装配；当前静态阶段保持锁合。"
        ],
        "approximations": [
            "机壳使用有厚度的轴对齐实体盒近似圆角外壳；侧面通风区以凹色条带近似。",
            "24个管孔使用环形实体几何和十字支撑结构近似，保留中心开口，不用整体凸包封闭孔位。",
            "启动、停止和解锁按钮使用短行程滑动关节；真实电子控制和转速控制未模拟。",
            "转子使用单一绕Z轴铰链表示旋转，未模拟离心加速、制冷和内部气流。"
        ],
        "buttons": [
            {"id": "A11_start_button", "joint": "start_button_joint", "geom": "start_button_geom", "pressed_at": 0.004, "released_at": 0.0005, "event": "centrifuge_start"},
            {"id": "A11_stop_button", "joint": "stop_button_joint", "geom": "stop_button_geom", "pressed_at": 0.004, "released_at": 0.0005, "event": "centrifuge_stop"},
            {"id": "A11_unlock_button", "joint": "unlock_button_joint", "geom": "unlock_button_geom", "pressed_at": 0.004, "released_at": 0.0005, "event": "centrifuge_unlock"}
        ],
        "connections": [
            {"id": "A11_rotor_cover_connection", "kind": "detachable", "equality": "rotor_cover_lock", "release_event": "centrifuge_unlock", "irreversible": false, "attach_event": "centrifuge_cover_attach", "capture_sites": ["rotor_cover_capture_site", "rotor_cover_mate_site"], "position_tolerance_m": 0.005, "angle_tolerance_rad": 0.2}
        ],
        "mechanical_events": [
            {"id": "centrifuge_cover_attach", "interface": "A11_rotor_cover_grasp", "action": "attach"},
            {"id": "centrifuge_unlock", "interface": "A11_unlock", "action": "release"}
        ]
    }
    with open(os.path.join(output_dir, "interfaces.json"), "w", encoding="utf-8") as f:
        json.dump(interfaces, f, ensure_ascii=False, indent=2)
