import os
import json


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    xml = '''<mujoco model="PL303_precision_balance">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81"/>
  <default>
    <geom contype="1" conaffinity="1" friction="0.7 0.1 0.1" density="650"/>
    <joint damping="0.25" armature="0.0001"/>
  </default>
  <asset>
    <material name="body_dark_gray" rgba="0.22 0.24 0.26 1"/>
    <material name="body_light_gray" rgba="0.62 0.64 0.66 1"/>
    <material name="black_panel" rgba="0.035 0.04 0.045 1"/>
    <material name="display_green" rgba="0.38 0.62 0.48 1"/>
    <material name="steel" rgba="0.72 0.74 0.76 1"/>
    <material name="pan_metal" rgba="0.82 0.84 0.86 1"/>
    <material name="glass" rgba="0.72 0.88 0.94 0.22"/>
    <material name="rubber" rgba="0.08 0.085 0.09 1"/>
    <material name="label_white" rgba="0.88 0.89 0.87 1"/>
  </asset>
  <worldbody>
    <body name="pl303_balance_body" pos="0 0 0">
      <inertial pos="0 0 0.09" mass="5.1" diaginertia="0.045 0.060 0.035"/>
      <geom name="balance_lower_housing" type="box" pos="0 0 0.074" size="0.119 0.1676 0.074" material="body_dark_gray"/>
      <geom name="balance_upper_deck" type="box" pos="0 0.005 0.119" size="0.108 0.145 0.018" material="body_light_gray"/>
      <geom name="front_control_panel" type="box" pos="-0.101 -0.005 0.096" size="0.004 0.115 0.040" material="black_panel"/>
      <geom name="display_window" type="box" pos="-0.106 -0.005 0.118" size="0.002 0.040 0.014" material="display_green"/>
      <geom name="control_knob" type="cylinder" pos="-0.107 -0.060 0.088" size="0.010 0.004" material="black_panel" zaxis="0 1 0"/>
      <geom name="control_key_1" type="cylinder" pos="-0.107 -0.030 0.084" size="0.007 0.003" material="steel" zaxis="0 1 0"/>
      <geom name="control_key_2" type="cylinder" pos="-0.107 -0.010 0.084" size="0.007 0.003" material="steel" zaxis="0 1 0"/>
      <geom name="control_key_3" type="cylinder" pos="-0.107 0.010 0.084" size="0.007 0.003" material="steel" zaxis="0 1 0"/>
      <geom name="control_key_4" type="cylinder" pos="-0.107 0.030 0.084" size="0.007 0.003" material="steel" zaxis="0 1 0"/>

      <body name="weighing_pan_support" pos="0 0 0.119">
        <inertial pos="0 0 0.008" mass="0.08" diaginertia="0.00008 0.00008 0.00012"/>
        <geom name="pan_support_post" type="cylinder" pos="0 0 0.008" size="0.018 0.008" material="steel"/>
        <geom name="pan_support_ring" type="cylinder" pos="0 0 0.016" size="0.058 0.003" material="steel"/>
        <body name="weighing_pan" pos="0 0 0.012">
          <inertial pos="0 0 0" mass="0.045" diaginertia="0.00006 0.00006 0.00010"/>
          <geom name="round_weighing_pan" type="cylinder" pos="0 0 0" size="0.050 0.003" material="pan_metal"/>
          <geom name="pan_inner_surface" type="cylinder" pos="0 0 0.0032" size="0.043 0.0008" material="steel"/>
          <site name="weighing_pan_center" pos="0 0 0.004" size="0.006" rgba="0.1 0.8 0.2 0.7"/>
        </body>
      </body>

      <body name="glass_enclosure_fixed" pos="0 0 0">
        <inertial pos="0 0 0.205" mass="0.34" diaginertia="0.003 0.003 0.003"/>
        <geom name="glass_left_panel" type="box" pos="-0.109 0.010 0.209" size="0.002 0.145 0.075" material="glass"/>
        <geom name="glass_rear_panel" type="box" pos="0 0.151 0.209" size="0.109 0.002 0.075" material="glass"/>
        <geom name="glass_top_panel" type="box" pos="0 0.010 0.284" size="0.109 0.145 0.003" material="glass"/>
        <geom name="rear_top_rail" type="box" pos="0 0.145 0.279" size="0.112 0.010 0.006" material="body_light_gray"/>
        <geom name="left_side_rail" type="box" pos="-0.111 0.010 0.132" size="0.006 0.153 0.008" material="body_light_gray"/>
        <geom name="right_side_rail" type="box" pos="0.111 0.010 0.132" size="0.006 0.153 0.008" material="body_light_gray"/>
        <site name="enclosure_service_side" pos="0.115 -0.015 0.210" size="0.008" rgba="0.1 0.5 1 0.7"/>
      </body>

      <body name="side_sliding_door" pos="0.111 -0.005 0.209">
        <joint name="side_door_slide_joint" type="slide" axis="0 1 0" range="-0.105 0.105" limited="true"/>
        <inertial pos="0 0 0" mass="0.16" diaginertia="0.0015 0.0015 0.0015"/>
        <geom name="sliding_glass_panel" type="box" pos="0 0 0" size="0.0025 0.145 0.075" material="glass"/>
        <geom name="door_upper_frame" type="box" pos="0 0 0.077" size="0.006 0.150 0.005" material="body_light_gray"/>
        <geom name="door_lower_frame" type="box" pos="0 0 -0.077" size="0.006 0.150 0.005" material="body_light_gray"/>
        <geom name="door_handle_mount" type="box" pos="0 -0.005 0.010" size="0.006 0.012 0.026" material="steel"/>
        <geom name="door_handle" type="capsule" fromto="0 -0.020 0.010 0 0.020 0.010" size="0.004" material="steel"/>
        <site name="side_door_handle_grasp" pos="0 -0.005 0.010" size="0.008" rgba="0.9 0.4 0.1 0.8"/>
        <site name="side_door_panel_center" pos="0 0 0" size="0.006" rgba="0.2 0.7 1 0.6"/>
      </body>

      <body name="foot_front_left" pos="-0.092 -0.132 0.018">
        <inertial pos="0 0 0" mass="0.035" diaginertia="0.00001 0.00001 0.00001"/>
        <geom name="front_left_foot" type="cylinder" size="0.012 0.018" material="rubber"/>
      </body>
      <body name="foot_front_right" pos="0.092 -0.132 0.018">
        <inertial pos="0 0 0" mass="0.035" diaginertia="0.00001 0.00001 0.00001"/>
        <geom name="front_right_foot" type="cylinder" size="0.012 0.018" material="rubber"/>
      </body>
      <body name="foot_rear_left" pos="-0.092 0.132 0.018">
        <inertial pos="0 0 0" mass="0.035" diaginertia="0.00001 0.00001 0.00001"/>
        <geom name="rear_left_foot" type="cylinder" size="0.012 0.018" material="rubber"/>
      </body>
      <body name="foot_rear_right" pos="0.092 0.132 0.018">
        <inertial pos="0 0 0" mass="0.035" diaginertia="0.00001 0.00001 0.00001"/>
        <geom name="rear_right_foot" type="cylinder" size="0.012 0.018" material="rubber"/>
      </body>

      <site name="balance_front_grasp" pos="-0.115 -0.105 0.095" size="0.008" rgba="0.9 0.4 0.1 0.7"/>
      <site name="balance_mount_reference" pos="0 0 0.018" size="0.008" rgba="0.2 0.8 0.2 0.7"/>
    </body>
  </worldbody>
</mujoco>
'''

    interfaces = {
        "schema_version": "1.0",
        "asset_ids": ["A02"],
        "parts": [
            {"id": "A02_balance", "body": "pl303_balance_body"},
            {"id": "A02_weighing_pan", "body": "weighing_pan"},
            {"id": "A02_glass_enclosure", "body": "glass_enclosure_fixed"},
            {"id": "A02_side_sliding_door", "body": "side_sliding_door"},
            {"id": "A02_support_feet", "body": "foot_front_left"}
        ],
        "interfaces": [
            {
                "id": "A02_pan_grasp",
                "asset_id": "A02",
                "kind": "grasp",
                "body": "weighing_pan",
                "site": "weighing_pan_center",
                "approach_axis": [0, 0, 1],
                "operation": {
                    "status": "declared",
                    "axis": [0, 0, 1],
                    "joint": "",
                    "range": [0, 0],
                    "clearance_m": 0.005,
                    "grasp_geoms": ["round_weighing_pan"],
                    "basis": {
                        "clearance_m": {"kind": "assumption", "reference": "建模操作间隙"}
                    }
                }
            },
            {
                "id": "A02_side_door_operation",
                "asset_id": "A02",
                "kind": "insert",
                "body": "side_sliding_door",
                "site": "side_door_handle_grasp",
                "approach_axis": [1, 0, 0],
                "operation": {
                    "status": "declared",
                    "axis": [0, 1, 0],
                    "joint": "side_door_slide_joint",
                    "range": [-0.105, 0.105],
                    "clearance_m": 0.003,
                    "grasp_geoms": ["door_handle"],
                    "basis": {
                        "axis": {"kind": "source_specification", "reference": "资料明确说明侧玻璃门沿深度方向滑动"},
                        "range": {"kind": "assumption", "reference": "满足100 mm称量盘进出所需的建模行程"}
                    }
                }
            },
            {
                "id": "A02_balance_mount",
                "asset_id": "A02",
                "kind": "mount",
                "body": "pl303_balance_body",
                "site": "balance_mount_reference",
                "approach_axis": [0, 0, -1]
            }
        ],
        "assumptions": [
            "PL303-IC后缀未确认，按PL303系列外形和机构要求建模。",
            "局部坐标取X为宽度238 mm、Y为深度335.2 mm、Z为高度287 mm，支撑面为z=0。",
            "外壳曲面按分段实体包络近似，保持指定总体外形尺寸；未使用原始网格资料。",
            "玻璃板、门框、门把手、支脚和内部承托结构的厚度、材料及质量为建模假设。",
            "内部净高按称量盘上表面至玻璃罩内顶面150 mm布置。"
        ],
        "approximations": [
            "精密天平外壳采用带有真实厚度的分段盒体近似。",
            "玻璃罩采用薄实体玻璃板组合，侧门采用直线滑移关节近似实际滑轨。",
            "显示区和控制键作为静态外观几何，不声明为可操作电子按钮。"
        ],
        "buttons": [],
        "connections": [],
        "mechanical_events": []
    }

    with open(os.path.join(output_dir, "model.xml"), "w", encoding="utf-8") as f:
        f.write(xml)
    with open(os.path.join(output_dir, "interfaces.json"), "w", encoding="utf-8") as f:
        json.dump(interfaces, f, ensure_ascii=False, indent=2)
