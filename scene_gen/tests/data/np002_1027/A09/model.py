import os, json


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    xml = '''<mujoco model="A09_precision_gp05_water_bath">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <material name="bath_cream" rgba="0.88 0.90 0.78 1"/>
    <material name="bath_trim" rgba="0.12 0.14 0.13 1"/>
    <material name="panel_blue" rgba="0.08 0.55 0.78 1"/>
    <material name="display_dark" rgba="0.025 0.045 0.035 1"/>
    <material name="button_white" rgba="0.82 0.90 0.88 1"/>
    <material name="water_blue" rgba="0.20 0.52 0.66 0.42"/>
    <material name="clear_cover" rgba="0.72 0.88 0.88 0.30"/>
    <material name="rack_metal" rgba="0.28 0.31 0.30 1"/>
    <material name="rubber" rgba="0.025 0.03 0.028 1"/>
  </asset>
  <worldbody>
    <body name="waterbath_body" pos="0 0 0">
      <geom name="waterbath_bottom_collision" type="box" pos="0 0 0.025" size="0.123 0.1775 0.025" material="bath_cream" mass="2.0"/>
      <geom name="waterbath_left_wall_collision" type="box" pos="-0.100 0 0.126" size="0.023 0.1775 0.106" material="bath_cream" mass="1.0"/>
      <geom name="waterbath_right_wall_collision" type="box" pos="0.100 0 0.126" size="0.023 0.1775 0.106" material="bath_cream" mass="1.0"/>
      <geom name="waterbath_front_wall_collision" type="box" pos="0 -0.16375 0.126" size="0.077 0.01375 0.106" material="bath_cream" mass="0.7"/>
      <geom name="waterbath_rear_wall_collision" type="box" pos="0 0.16375 0.126" size="0.077 0.01375 0.106" material="bath_cream" mass="0.7"/>
      <geom name="waterbath_inner_water_visual" type="box" pos="0 0 0.190" size="0.077 0.150 0.002" material="water_blue" contype="0" conaffinity="0"/>

      <geom name="waterbath_top_left_trim" type="box" pos="-0.100 0 0.232" size="0.0245 0.180 0.006" material="bath_trim" mass="0.15"/>
      <geom name="waterbath_top_right_trim" type="box" pos="0.100 0 0.232" size="0.0245 0.180 0.006" material="bath_trim" mass="0.15"/>
      <geom name="waterbath_top_front_trim" type="box" pos="0 -0.164 0.232" size="0.076 0.015 0.006" material="bath_trim" mass="0.10"/>
      <geom name="waterbath_top_rear_trim" type="box" pos="0 0.164 0.232" size="0.076 0.015 0.006" material="bath_trim" mass="0.10"/>

      <geom name="waterbath_control_panel" type="box" pos="0 -0.184 0.105" size="0.070 0.008 0.060" material="panel_blue" mass="0.10"/>
      <geom name="waterbath_display" type="box" pos="-0.015 -0.193 0.130" size="0.038 0.002 0.017" material="display_dark" contype="0" conaffinity="0"/>
      <geom name="waterbath_control_button_left" type="box" pos="-0.047 -0.194 0.087" size="0.010 0.002 0.007" material="button_white" mass="0.005"/>
      <geom name="waterbath_control_button_mid" type="box" pos="-0.020 -0.194 0.087" size="0.010 0.002 0.007" material="button_white" mass="0.005"/>
      <geom name="waterbath_control_button_right" type="box" pos="0.007 -0.194 0.087" size="0.010 0.002 0.007" material="button_white" mass="0.005"/>
      <geom name="waterbath_control_knob" type="cylinder" pos="0.042 -0.194 0.116" size="0.014 0.004" zaxis="0 1 0" material="bath_trim" mass="0.02"/>

      <geom name="waterbath_foot_front_left" type="cylinder" pos="-0.092 -0.135 0.004" size="0.012 0.004" material="rubber" mass="0.03"/>
      <geom name="waterbath_foot_front_right" type="cylinder" pos="0.092 -0.135 0.004" size="0.012 0.004" material="rubber" mass="0.03"/>
      <geom name="waterbath_foot_rear_left" type="cylinder" pos="-0.092 0.135 0.004" size="0.012 0.004" material="rubber" mass="0.03"/>
      <geom name="waterbath_foot_rear_right" type="cylinder" pos="0.092 0.135 0.004" size="0.012 0.004" material="rubber" mass="0.03"/>

      <site name="waterbath_rack_insert_site" pos="0 0 0.232" size="0.012" rgba="0.1 0.8 0.8 1"/>
      <site name="waterbath_body_grasp_site" pos="0.115 -0.105 0.115" size="0.012" rgba="0.9 0.5 0.1 1"/>
    </body>

    <body name="immersed_tube_rack" pos="0 0 0">
      <freejoint name="immersed_tube_rack_freejoint"/>
      <inertial pos="0 0 0.12" mass="0.32" diaginertia="0.0012 0.0012 0.0008"/>
      <geom name="rack_front_rail" type="box" pos="0 -0.132 0.188" size="0.073 0.006 0.006" material="rack_metal" mass="0.03"/>
      <geom name="rack_rear_rail" type="box" pos="0 0.132 0.188" size="0.073 0.006 0.006" material="rack_metal" mass="0.03"/>
      <geom name="rack_left_rail" type="box" pos="-0.068 0 0.188" size="0.006 0.126 0.006" material="rack_metal" mass="0.03"/>
      <geom name="rack_right_rail" type="box" pos="0.068 0 0.188" size="0.006 0.126 0.006" material="rack_metal" mass="0.03"/>
      <geom name="rack_crossbar_front" type="box" pos="0 -0.066 0.188" size="0.073 0.004 0.004" material="rack_metal" mass="0.018"/>
      <geom name="rack_crossbar_rear" type="box" pos="0 0.066 0.188" size="0.073 0.004 0.004" material="rack_metal" mass="0.018"/>
      <geom name="rack_left_leg" type="box" pos="-0.068 -0.118 0.095" size="0.006 0.006 0.095" material="rack_metal" mass="0.025"/>
      <geom name="rack_right_leg" type="box" pos="0.068 -0.118 0.095" size="0.006 0.006 0.095" material="rack_metal" mass="0.025"/>
      <geom name="rack_left_leg_rear" type="box" pos="-0.068 0.118 0.095" size="0.006 0.006 0.095" material="rack_metal" mass="0.025"/>
      <geom name="rack_right_leg_rear" type="box" pos="0.068 0.118 0.095" size="0.006 0.006 0.095" material="rack_metal" mass="0.025"/>
      <geom name="rack_handle" type="box" pos="0 0 0.220" size="0.040 0.006 0.006" material="rack_metal" mass="0.025"/>
      <site name="rack_grasp_site" pos="0 0 0.225" size="0.010" rgba="0.9 0.5 0.1 1"/>
      <site name="rack_insert_site" pos="0 0 0.188" size="0.010" rgba="0.1 0.8 0.8 1"/>
      <site name="rack_tube_slot_1" pos="-0.040 -0.090 0.194" size="0.006"/>
      <site name="rack_tube_slot_2" pos="0.000 -0.090 0.194" size="0.006"/>
      <site name="rack_tube_slot_3" pos="0.040 -0.090 0.194" size="0.006"/>
      <site name="rack_tube_slot_4" pos="-0.040 -0.030 0.194" size="0.006"/>
      <site name="rack_tube_slot_5" pos="0.000 -0.030 0.194" size="0.006"/>
      <site name="rack_tube_slot_6" pos="0.040 -0.030 0.194" size="0.006"/>
      <site name="rack_tube_slot_7" pos="-0.040 0.030 0.194" size="0.006"/>
      <site name="rack_tube_slot_8" pos="0.000 0.030 0.194" size="0.006"/>
      <site name="rack_tube_slot_9" pos="0.040 0.030 0.194" size="0.006"/>
      <site name="rack_tube_slot_10" pos="-0.040 0.090 0.194" size="0.006"/>
      <site name="rack_tube_slot_11" pos="0.000 0.090 0.194" size="0.006"/>
      <site name="rack_tube_slot_12" pos="0.040 0.090 0.194" size="0.006"/>
    </body>

    <body name="transparent_cover" pos="0 0 0.232">
      <freejoint name="transparent_cover_freejoint"/>
      <inertial pos="0 0 0.030" mass="0.42" diaginertia="0.004 0.004 0.002"/>
      <geom name="cover_top_collision" type="box" pos="0 0 0.050" size="0.112 0.168 0.004" material="clear_cover" mass="0.12"/>
      <geom name="cover_front_panel_collision" type="box" pos="0 -0.155 0.025" size="0.112 0.004 0.030" material="clear_cover" mass="0.06"/>
      <geom name="cover_rear_panel_collision" type="box" pos="0 0.155 0.025" size="0.112 0.004 0.030" material="clear_cover" mass="0.06"/>
      <geom name="cover_left_panel_collision" type="box" pos="-0.108 0 0.025" size="0.004 0.155 0.030" material="clear_cover" mass="0.06"/>
      <geom name="cover_right_panel_collision" type="box" pos="0.108 0 0.025" size="0.004 0.155 0.030" material="clear_cover" mass="0.06"/>
      <geom name="cover_handle_left" type="box" pos="-0.032 0 0.082" size="0.006 0.010 0.032" material="bath_trim" mass="0.025"/>
      <geom name="cover_handle_right" type="box" pos="0.032 0 0.082" size="0.006 0.010 0.032" material="bath_trim" mass="0.025"/>
      <geom name="cover_handle_grip" type="box" pos="0 0 0.114" size="0.038 0.010 0.006" material="bath_trim" mass="0.025"/>
      <site name="cover_grasp_site" pos="0 0 0.122" size="0.012" rgba="0.9 0.5 0.1 1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="rack_installed_weld" body1="waterbath_body" body2="immersed_tube_rack" solref="0.02 1" solimp="0.9 0.95 0.01"/>
    <weld name="cover_installed_weld" body1="waterbath_body" body2="transparent_cover" solref="0.02 1" solimp="0.9 0.95 0.01"/>
  </equality>
</mujoco>'''

    interfaces = {
        "schema_version": "1.0",
        "asset_ids": ["A09"],
        "parts": [
            {"id": "A09_body", "body": "waterbath_body"},
            {"id": "A09_immersed_rack", "body": "immersed_tube_rack"},
            {"id": "A09_cover", "body": "transparent_cover"}
        ],
        "interfaces": [
            {"id": "A09_cover_grasp", "asset_id": "A09", "kind": "grasp", "body": "transparent_cover", "site": "cover_grasp_site", "approach_axis": [0, 0, 1], "operation": {"status": "declared", "axis": [0, 0, 1], "grasp_geoms": ["cover_handle_grip"], "basis": {"kind": "source_specification", "reference": "资料图片及A09要求中的透明可移除盖和手柄"}}},
            {"id": "A09_rack_grasp", "asset_id": "A09", "kind": "grasp", "body": "immersed_tube_rack", "site": "rack_grasp_site", "approach_axis": [0, 0, 1], "operation": {"status": "declared", "axis": [0, 0, 1], "grasp_geoms": ["rack_handle"], "basis": {"kind": "assumption", "reference": "管架取放需要的顶部抓取手柄"}}},
            {"id": "A09_rack_insert", "asset_id": "A09", "kind": "insert", "body": "immersed_tube_rack", "site": "rack_insert_site", "approach_axis": [0, 0, 1], "operation": {"status": "declared", "axis": [0, 0, 1], "range": [0.0, 0.12], "clearance_m": 0.002, "mate_interface": "A09_bath_rack_mate", "basis": {"kind": "assumption", "reference": "按研磨管外形建立的浸没式管架建模值"}}},
            {"id": "A09_bath_rack_mate", "asset_id": "A09", "kind": "mount", "body": "waterbath_body", "site": "waterbath_rack_insert_site", "approach_axis": [0, 0, 1], "operation": {"status": "declared", "axis": [0, 0, 1], "clearance_m": 0.002, "mate_interface": "A09_rack_insert", "basis": {"kind": "assumption", "reference": "按腔体尺寸和管架承托关系确定的定位接口"}}}
        ],
        "assumptions": [
            "水浴外形、腔体和容量按资料给出的Precision GP 05参考规格建模：外形246 x 355 x 232 mm，腔体154 x 300 x 150 mm。",
            "资料未给出管架孔位、管架壁厚和研磨管精确外形；采用12位、3列4排、孔位间距约40 x 60 mm的建模值。",
            "水面以非碰撞视觉薄层表示，不模拟液体或温度过程；腔体保持真实开口。",
            "控制面板和旋钮为图片中可见外观的低细节几何，未声明为可验证按钮事件。",
            "盖子按可整体取下的透明罩建模，初始由边缘承托并通过可释放焊接约束装配。"
        ],
        "approximations": [
            "透明罩和管架使用带自由根的刚体近似，并以weld等式表达初始装配；运行时释放操作未验证。",
            "温控、加热、液体和防浮功能仅保留静态结构，不模拟热过程或流体。"
        ],
        "buttons": [],
        "connections": [
            {"id": "A09_rack_connection", "kind": "detachable", "equality": "rack_installed_weld", "release_event": "A09_rack_grasp", "irreversible": false, "attach_event": "A09_rack_insert", "capture_sites": ["waterbath_rack_insert_site", "rack_insert_site"], "position_tolerance_m": 0.005, "angle_tolerance_rad": 0.2},
            {"id": "A09_cover_connection", "kind": "detachable", "equality": "cover_installed_weld", "release_event": "A09_cover_grasp", "irreversible": false, "capture_sites": ["waterbath_rack_insert_site", "cover_grasp_site"], "position_tolerance_m": 0.005, "angle_tolerance_rad": 0.2}
        ],
        "mechanical_events": [],
        "construction_records": []
    }

    with open(os.path.join(output_dir, "model.xml"), "w", encoding="utf-8") as f:
        f.write(xml)
    with open(os.path.join(output_dir, "interfaces.json"), "w", encoding="utf-8") as f:
        json.dump(interfaces, f, ensure_ascii=False, indent=2)
