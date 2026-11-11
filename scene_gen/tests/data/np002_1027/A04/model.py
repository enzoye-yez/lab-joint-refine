import os, json, math


def _box(name, pos, size, rgba, contype=1, conaffinity=1):
    sx, sy, sz = size
    return '<geom name="%s" type="box" pos="%.6f %.6f %.6f" size="%.6f %.6f %.6f" rgba="%s" contype="%d" conaffinity="%d"/>' % (name, pos[0], pos[1], pos[2], sx/2, sy/2, sz/2, rgba, contype, conaffinity)


def _cyl(name, pos, radius, halfheight, rgba, contype=1, conaffinity=1):
    return '<geom name="%s" type="cylinder" pos="%.6f %.6f %.6f" size="%.6f %.6f" rgba="%s" contype="%d" conaffinity="%d"/>' % (name, pos[0], pos[1], pos[2], radius, halfheight, rgba, contype, conaffinity)


def _dish_wall(prefix, radius, thickness, z0, height, rgba):
    out = []
    n = 32
    radial = thickness / 2.0
    tangential = radius * math.sin(math.pi / n) * 1.03
    for i in range(n):
        a = 2.0 * math.pi * i / n
        x = (radius - radial) * math.cos(a)
        y = (radius - radial) * math.sin(a)
        out.append(_box('%s_%02d' % (prefix, i), (x, y, z0 + height/2),
                         (thickness, 2*tangential, height), rgba))
    return ''.join(out)


def _inertial(mass, pos=(0,0,0), diaginertia=(1e-5,1e-5,1e-5)):
    return '<inertial pos="%.6f %.6f %.6f" mass="%.6f" diaginertia="%.8f %.8f %.8f"/>' % (pos + (mass,) + diaginertia)


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    clear = '0.88 0.90 0.93 1'
    white = '0.92 0.93 0.95 1'
    foil_rgba = '0.68 0.70 0.74 1'
    membrane_rgba = '0.12 0.13 0.15 1'
    dark = '0.18 0.20 0.23 1'
    blue = '0.18 0.38 0.72 1'

    board = []
    board.append(_box('cutting_board_collision', (0,0,0.006), (0.230,0.280,0.012), dark))
    board.append(_box('cutting_board_visual', (0,0,0.006), (0.230,0.280,0.012), dark, 0, 0))
    # Raised cutting perimeter and two retaining bars; they preserve a usable open cutting area.
    board.append(_box('cutting_edge_front', (-0.055,-0.130,0.014), (0.120,0.006,0.010), blue))
    board.append(_box('cutting_edge_back', (-0.055,0.130,0.014), (0.120,0.006,0.010), blue))
    board.append(_box('membrane_retainer_left', (-0.108,0.025,0.016), (0.006,0.115,0.006), blue))
    board.append(_box('membrane_retainer_right', (0.000,0.025,0.016), (0.006,0.115,0.006), blue))
    board.append('<site name="cutting_surface_site" pos="-0.055 0 0.016" size="0.008" rgba="0.1 0.5 1 1"/>')
    board.append('<site name="foil_grasp_site" pos="-0.075 -0.105 0.016" size="0.006" rgba="0.8 0.8 0.2 1"/>')

    foil = []
    foil.append(_inertial(0.012, diaginertia=(0.00002,0.00002,0.00003)))
    foil.append('<joint name="foil_free_joint" type="free"/>')
    foil.append(_box('aluminum_foil_collision', (0,0,0.00015), (0.2032,0.2540,0.00030), foil_rgba))
    foil.append(_box('aluminum_foil_visual', (0,0,0.00016), (0.2032,0.2540,0.00020), '0.78 0.80 0.84 1', 0, 0))
    # A folded-back edge is represented as a supported, low rigid flap rather than a deformable sheet.
    foil.append(_box('foil_folded_edge', (0.092,-0.108,0.0010), (0.018,0.095,0.0010), foil_rgba))
    foil.append('<site name="foil_surface_site" pos="-0.075 -0.105 0.0005" size="0.005" rgba="0.9 0.8 0.2 1"/>')

    dish = []
    dish.append(_inertial(0.075, diaginertia=(0.00005,0.00005,0.00008)))
    dish.append(_cyl('dish_bottom_collision', (0,0,0.0010), 0.0420, 0.0010, clear))
    dish.append(_cyl('dish_bottom_visual', (0,0,0.0011), 0.0420, 0.0008, '0.78 0.86 0.92 1', 0, 0))
    dish.append(_dish_wall('dish_wall', 0.044715, 0.0025, 0.0018, 0.01738, clear))
    dish.append('<site name="dish_cavity_site" pos="0 0 0.005" size="0.006" rgba="0.1 0.7 0.9 1"/>')
    dish.append('<site name="dish_grasp_site" pos="0.044 0 0.010" size="0.004" rgba="0.9 0.7 0.1 1"/>')

    membrane = []
    membrane.append(_inertial(0.001, diaginertia=(0.0000002,0.0000002,0.0000003)))
    membrane.append('<joint name="membrane_free_joint" type="free"/>')
    membrane.append(_cyl('filter_membrane_collision', (0,0,0.00018), 0.0235, 0.00018, membrane_rgba))
    membrane.append(_cyl('filter_membrane_visual', (0,0,0.00020), 0.0235, 0.00012, '0.10 0.11 0.12 1', 0, 0))
    membrane.append('<site name="membrane_grasp_site" pos="0.018 0 0.0005" size="0.003" rgba="0.9 0.7 0.1 1"/>')

    lid = []
    lid.append(_inertial(0.018, diaginertia=(0.000015,0.000015,0.000025)))
    lid.append('<joint name="lid_free_joint" type="free"/>')
    lid.append(_dish_wall('lid_rim', 0.0455, 0.0022, 0.0000, 0.0030, '0.72 0.82 0.90 0.72'))
    lid.append(_cyl('lid_top_collision', (0,0,0.0022), 0.0448, 0.00035, '0.72 0.82 0.90 0.72'))
    lid.append(_cyl('lid_top_visual', (0,0,0.0022), 0.0448, 0.00025, '0.72 0.82 0.90 0.38', 0, 0))
    lid.append('<site name="lid_grasp_site" pos="0.044 0 0.002" size="0.004" rgba="0.9 0.7 0.1 1"/>')

    xml = '''<mujoco model="A04_falcon_353003_membrane_support">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom friction="0.7 0.1 0.01" condim="3"/>
    <joint damping="0.02" armature="0.0001"/>
  </default>
  <worldbody>
    <body name="a04_cutting_support" pos="0 0 0">
      %s
    </body>
    <body name="a04_aluminum_foil" pos="0 0 0.0123">
      %s
    </body>
    <body name="a04_petri_dish" pos="0.055 0 0.0123">
      %s
    </body>
    <body name="a04_filter_membrane" pos="0.055 0 0.01565">
      %s
    </body>
    <body name="a04_petri_lid" pos="0.055 0 0.03048">
      %s
    </body>
  </worldbody>
  <equality>
    <weld name="a04_lid_detachable_weld" body1="a04_petri_lid" body2="a04_petri_dish" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </equality>
</mujoco>''' % (''.join(board), ''.join(foil), ''.join(dish), ''.join(membrane), ''.join(lid))

    with open(os.path.join(output_dir, 'model.xml'), 'w', encoding='utf-8') as f:
        f.write(xml)

    interfaces = {
        "schema_version": "1.0",
        "asset_ids": ["A04"],
        "parts": [
            {"id": "cutting_support", "body": "a04_cutting_support"},
            {"id": "aluminum_foil", "body": "a04_aluminum_foil"},
            {"id": "petri_dish_353003", "body": "a04_petri_dish"},
            {"id": "filter_membrane_47mm", "body": "a04_filter_membrane"},
            {"id": "petri_lid", "body": "a04_petri_lid"}
        ],
        "interfaces": [
            {"id": "a04_cut_membrane", "asset_id": "A04", "kind": "cut", "body": "a04_cutting_support", "site": "cutting_surface_site", "approach_axis": [0,0,1], "operation": {"status": "declared", "axis": [0,0,1], "clearance_m": 0.004, "grasp_geoms": ["membrane_retainer_left", "membrane_retainer_right"], "basis": {"axis": {"kind": "source_specification", "reference": "A04 membrane cutting support requirement"}, "clearance_m": {"kind": "assumption", "reference": "rigid cutting clearance"}}}},
            {"id": "a04_grasp_foil", "asset_id": "A04", "kind": "grasp", "body": "a04_aluminum_foil", "site": "foil_grasp_site", "approach_axis": [0,0,1]},
            {"id": "a04_insert_membrane", "asset_id": "A04", "kind": "insert", "body": "a04_petri_dish", "site": "dish_cavity_site", "approach_axis": [0,0,1]},
            {"id": "a04_grasp_dish", "asset_id": "A04", "kind": "grasp", "body": "a04_petri_dish", "site": "dish_grasp_site", "approach_axis": [1,0,0]},
            {"id": "a04_grasp_lid", "asset_id": "A04", "kind": "grasp", "body": "a04_petri_lid", "site": "lid_grasp_site", "approach_axis": [1,0,0]}
        ],
        "assumptions": [
            "培养皿主体外径0.08943 m、高0.01918 m按任务固定规格建模；壁厚、底厚和盖体细节为建模假设。",
            "203.2 mm × 254 mm铝箔作为刚体薄片近似，边缘折返用独立刚体折边表示。",
            "47 mm滤膜以圆形薄刚体表示；切割后的约20 mm²小片不在初始模型中重复生成。",
            "切割承托面尺寸、夹持条位置和材料颜色为建模布局假设。"
        ],
        "approximations": [
            "铝箔和滤膜采用带厚度的刚体近似，不模拟柔性变形。",
            "培养皿与盖子的透明塑料采用半透明实体几何近似。",
            "盖子采用自由体加weld等式表达可整体拆离关系；当前静态状态为闭合。"
        ],
        "buttons": [],
        "connections": [
            {"id": "a04_lid_connection", "kind": "detachable", "equality": "a04_lid_detachable_weld", "release_event": "a04_grasp_lid", "irreversible": false, "attach_event": "a04_grasp_lid", "capture_sites": ["dish_cavity_site", "lid_grasp_site"], "position_tolerance_m": 0.005, "angle_tolerance_rad": 0.2}
        ]
    }
    with open(os.path.join(output_dir, 'interfaces.json'), 'w', encoding='utf-8') as f:
        json.dump(interfaces, f, ensure_ascii=False, indent=2)
