import os, json


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def box(name, pos, size, rgba="0.72 0.76 0.78 1", contype="1", conaffinity="1", mass=None):
        attrs = [f'name="{name}"', f'pos="{pos[0]} {pos[1]} {pos[2]}"', f'size="{size[0]/2} {size[1]/2} {size[2]/2}"', f'material="mat_{rgba.replace(" ", "_")}"', f'contype="{contype}"', f'conaffinity="{conaffinity}"']
        if mass is not None:
            attrs.append(f'mass="{mass}"')
        return "<geom type=\"box\" " + " ".join(attrs) + "/>"

    def cyl(name, pos, radius, length, axis="z", rgba="0.18 0.20 0.21 1", mass=None):
        attrs = [f'name="{name}"', f'pos="{pos[0]} {pos[1]} {pos[2]}"', f'size="{radius} {length/2}"', f'axis="{axis}"', f'material="mat_{rgba.replace(" ", "_")}"', 'contype="1"', 'conaffinity="1"']
        if mass is not None:
            attrs.append(f'mass="{mass}"')
        return "<geom type=\"cylinder\" " + " ".join(attrs) + "/>"

    mats = """
    <material name="mat_0.72_0.76_0.78_1" rgba="0.72 0.76 0.78 1"/>
    <material name="mat_0.16_0.18_0.19_1" rgba="0.16 0.18 0.19 1"/>
    <material name="mat_0.05_0.06_0.07_1" rgba="0.05 0.06 0.07 1"/>
    <material name="mat_0.82_0.84_0.84_1" rgba="0.82 0.84 0.84 1"/>
    <material name="mat_0.30_0.65_0.78_0.28" rgba="0.30 0.65 0.78 0.28" specular="0.8"/>
    <material name="mat_0.88_0.89_0.88_1" rgba="0.88 0.89 0.88 1"/>
    <material name="mat_0.40_0.43_0.44_1" rgba="0.40 0.43 0.44 1"/>
    """

    geoms = []
    # Fixed support frame: four legs, upper perimeter rails, lower braces.
    leg_x = (-0.57, 0.57)
    leg_y = (-0.31, 0.31)
    for i, x in enumerate(leg_x):
        for j, y in enumerate(leg_y):
            geoms.append(box(f"support_leg_{i}_{j}", (x, y, 0.375), (0.055, 0.055, 0.75), "0.40 0.43 0.44 1", mass=8))
    geoms += [
        box("front_support_rail", (0, -0.31, 0.69), (1.22, 0.055, 0.08), "0.40 0.43 0.44 1", mass=4),
        box("rear_support_rail", (0, 0.31, 0.69), (1.22, 0.055, 0.08), "0.40 0.43 0.44 1", mass=4),
        box("left_support_rail", (-0.57, 0, 0.69), (0.055, 0.62, 0.08), "0.40 0.43 0.44 1", mass=3),
        box("right_support_rail", (0.57, 0, 0.69), (0.055, 0.62, 0.08), "0.40 0.43 0.44 1", mass=3),
        box("lower_front_brace", (0, -0.31, 0.25), (1.1, 0.035, 0.035), "0.16 0.18 0.19 1", mass=1),
        box("lower_rear_brace", (0, 0.31, 0.25), (1.1, 0.035, 0.035), "0.16 0.18 0.19 1", mass=1),
    ]

    # Cabinet enclosure, with open front and real internal cavity.
    geoms += [
        box("left_side_panel", (-0.625, 0.0, 1.511), (0.05, 0.802, 1.522), mass=18),
        box("right_side_panel", (0.625, 0.0, 1.511), (0.05, 0.802, 1.522), mass=18),
        box("top_panel", (0, 0.0, 2.247), (1.30, 0.802, 0.07), mass=10),
        box("rear_upper_panel", (0, 0.365, 1.94), (1.20, 0.055, 0.58), mass=8),
        box("rear_lower_panel", (0, 0.365, 1.00), (1.20, 0.055, 0.26), mass=5),
        box("work_surface", (0, 0.015, 0.785), (1.20, 0.63, 0.07), "0.88 0.89 0.88 1", mass=12),
        box("front_edge", (0, -0.355, 0.845), (1.30, 0.075, 0.16), "0.16 0.18 0.19 1", mass=5),
        box("front_upper_header", (0, -0.365, 2.04), (1.20, 0.06, 0.34), mass=5),
        box("left_front_post", (-0.59, -0.365, 1.43), (0.08, 0.07, 1.55), mass=5),
        box("right_front_post", (0.59, -0.365, 1.43), (0.08, 0.07, 1.55), mass=5),
    ]

    # Rear exhaust grille: spaced slats preserve visible openings.
    for i, x in enumerate((-0.48, -0.32, -0.16, 0.0, 0.16, 0.32, 0.48)):
        geoms.append(box(f"rear_grille_vertical_{i}", (x, 0.333, 1.55), (0.055, 0.025, 0.28), "0.05 0.06 0.07 1", mass=0.25))
    for i, z in enumerate((1.42, 1.50, 1.58, 1.66, 1.74)):
        geoms.append(box(f"rear_grille_horizontal_{i}", (0, 0.325, z), (1.08, 0.025, 0.035), "0.05 0.06 0.07 1", mass=0.3))

    # Internal filter support and suction-funnel clearance fixtures.
    geoms += [
        box("filter_clip_left", (-0.45, -0.06, 0.835), (0.16, 0.035, 0.025), "0.16 0.18 0.19 1", mass=0.12),
        box("filter_clip_right", (0.45, -0.06, 0.835), (0.16, 0.035, 0.025), "0.16 0.18 0.19 1", mass=0.12),
        box("rear_funnel_clearance_rail", (0, 0.27, 0.84), (0.80, 0.035, 0.035), "0.16 0.18 0.19 1", mass=0.25),
        cyl("internal_drain_port", (0.0, 0.28, 0.86), 0.045, 0.04, "z", "0.16 0.18 0.19 1", 0.2),
    ]

    fixed_xml = "\n".join(geoms)

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="MSC_Advantage_1p2_biosafety_cabinet">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint damping="3" armature="0.001"/>
    <geom friction="0.7 0.1 0.1" solref="0.01 1" solimp="0.85 0.95 0.01"/>
  </default>
  <asset>{mats}</asset>
  <worldbody>
    <body name="A03_cabinet_root" pos="0 0 0">
      {fixed_xml}
      <body name="front_window_carriage" pos="0 -0.405 0.75">
        <joint name="front_window_lift_joint" type="slide" axis="0 0 1" range="0 0.333" limited="true" damping="8"/>
        <geom name="front_window_glass" type="box" pos="0 0 0.25" size="0.575 0.018 0.225" material="mat_0.30_0.65_0.78_0.28" contype="1" conaffinity="1" mass="7"/>
        <geom name="front_window_lower_rail" type="box" pos="0 0.025 0.025" size="0.60 0.035 0.035" material="mat_0.16_0.18_0.19_1" contype="1" conaffinity="1" mass="1"/>
        <geom name="front_window_left_rail" type="box" pos="-0.59 0 0.25" size="0.025 0.035 0.25" material="mat_0.16_0.18_0.19_1" contype="1" conaffinity="1" mass="0.7"/>
        <geom name="front_window_right_rail" type="box" pos="0.59 0 0.25" size="0.025 0.035 0.25" material="mat_0.16_0.18_0.19_1" contype="1" conaffinity="1" mass="0.7"/>
        <geom name="front_window_handle" type="box" pos="0 -0.045 0.08" size="0.14 0.025 0.025" material="mat_0.05_0.06_0.07_1" contype="1" conaffinity="1" mass="0.25"/>
        <site name="front_window_grasp_site" pos="0 -0.075 0.08" size="0.025" rgba="0.9 0.4 0.1 1"/>
      </body>
      <site name="cabinet_work_surface_site" pos="0 -0.05 0.84" size="0.025" rgba="0.1 0.8 0.3 1"/>
      <site name="cabinet_front_access_site" pos="0 -0.43 1.05" size="0.03" rgba="0.1 0.5 0.9 1"/>
      <site name="cabinet_mount_site" pos="0 0 0.04" size="0.025" rgba="0.8 0.2 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
'''

    interfaces = {
        "schema_version": "1.0",
        "asset_ids": ["A03"],
        "parts": [
            {"id": "cabinet_enclosure", "body": "A03_cabinet_root"},
            {"id": "fixed_support_bracket", "body": "A03_cabinet_root"},
            {"id": "front_window", "body": "front_window_carriage"},
            {"id": "rear_exhaust_grille", "body": "A03_cabinet_root"},
            {"id": "internal_work_surface", "body": "A03_cabinet_root"}
        ],
        "interfaces": [
            {"id": "A03_mount_to_table", "asset_id": "A03", "kind": "mount", "body": "A03_cabinet_root", "site": "cabinet_mount_site", "approach_axis": [0, 0, -1], "operation": {"status": "declared", "axis": [0, 0, 1], "clearance_m": 0.01, "basis": {"clearance_m": {"kind": "assumption", "reference": "建模间隙值"}}}},
            {"id": "A03_window_grasp", "asset_id": "A03", "kind": "grasp", "body": "front_window_carriage", "site": "front_window_grasp_site", "approach_axis": [0, -1, 0], "operation": {"status": "declared", "axis": [0, 0, 1], "joint": "front_window_lift_joint", "range": [0.2, 0.533], "clearance_m": 0.01, "grasp_geoms": ["front_window_handle"], "basis": {"range": {"kind": "source_specification", "reference": "MSC-Advantage资料：正常开窗0.2 m，最大开窗0.533 m"}}}},
            {"id": "A03_front_access", "asset_id": "A03", "kind": "insert", "body": "A03_cabinet_root", "site": "cabinet_front_access_site", "approach_axis": [0, -1, 0]},
            {"id": "A03_work_surface", "asset_id": "A03", "kind": "mount", "body": "A03_cabinet_root", "site": "cabinet_work_surface_site", "approach_axis": [0, 0, 1]}
        ],
        "assumptions": [
            "外壳宽1.300 m、深0.802 m、高1.522 m按资料规格建模；固定支架另建于外壳下方，工作面局部高度0.750 m。",
            "资料未给出支架截面、板厚、窗框截面、后格栅具体孔距和质量，采用工程建模假设。",
            "局部坐标约定为X左右、Y前后（前方为负Y）、Z向上；资产支撑面为z=0。"
        ],
        "approximations": [
            "前窗采用带导轨的刚体垂直滑动近似，保留0.2 m正常开窗位置和0.533 m最大开窗位置；未模拟配重、气弹簧和真实开闭力。",
            "后部排风格栅采用带真实间隙的水平与竖向条栅近似，未模拟风机和气流。",
            "内部滤膜夹与抽滤设备空间采用固定夹点、导轨和排液口几何近似，不模拟生化过程或空气洁净性能。"
        ],
        "buttons": [],
        "connections": [],
        "construction_records": []
    }

    with open(os.path.join(output_dir, "model.xml"), "w", encoding="utf-8") as f:
        f.write(xml)
    with open(os.path.join(output_dir, "interfaces.json"), "w", encoding="utf-8") as f:
        json.dump(interfaces, f, ensure_ascii=False, indent=2)
