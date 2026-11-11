import os, json, math


def write_obj(path, kind):
    verts = []
    faces = []
    n = 64
    def ring(r, z):
        ids = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            verts.append((r * math.cos(a), r * math.sin(a), z))
            ids.append(len(verts))
        return ids
    def quad(a, b):
        for i in range(n):
            j = (i + 1) % n
            faces.append((a[i], a[j], b[j], b[i]))
    if kind == 'tube':
        ro0 = ring(0.0035, 0.0)
        ro1 = ring(0.01486, 0.0120)
        ro2 = ring(0.01486, 0.11455)
        ri0 = ring(0.0028, 0.0030)
        ri1 = ring(0.0128, 0.0120)
        ri2 = ring(0.013905, 0.11455)
        quad(ro0, ro1)
        quad(ro1, ro2)
        quad(ri0, ri1)
        quad(ri1, ri2)
        quad(ro2, ri2)
        quad(ro0, ri0)
    else:
        # Cap: annular threaded skirt and closed top disk.
        outer0 = ring(0.017375, 0.10259)
        outer1 = ring(0.017375, 0.11470)
        outer2 = ring(0.0169, 0.11580)
        inner0 = ring(0.015825, 0.10259)
        inner1 = ring(0.015825, 0.1138)
        quad(outer0, outer1)
        quad(outer1, outer2)
        quad(inner0, inner1)
        quad(outer0, inner0)
        quad(outer2, inner1)
    with open(path, 'w', encoding='utf-8') as f:
        for v in verts:
            f.write('v %.8f %.8f %.8f\n' % v)
        for face in faces:
            f.write('f ' + ' '.join(str(x) for x in face) + '\n')


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)
    write_obj(os.path.join(output_dir, 'meshes', 'falcon352070_tube.obj'), 'tube')
    write_obj(os.path.join(output_dir, 'meshes', 'falcon358206_cap.obj'), 'cap')

    xml = r'''<mujoco model="A05_falcon_50ml_tube_and_stand">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <asset>
    <mesh name="falcon_tube_mesh" file="meshes/falcon352070_tube.obj"/>
    <mesh name="falcon_cap_mesh" file="meshes/falcon358206_cap.obj"/>
  </asset>
  <default>
    <geom contype="1" conaffinity="1" friction="0.7 0.05 0.02"/>
  </default>
  <worldbody>
    <body name="support_stand" pos="0 0 0">
      <geom name="stand_base" type="box" pos="0 0 0.006" size="0.075 0.048 0.006" rgba="0.18 0.20 0.23 1"/>
      <geom name="stand_left_rail" type="box" pos="-0.052 0 0.020" size="0.006 0.044 0.014" rgba="0.20 0.22 0.25 1"/>
      <geom name="stand_right_rail" type="box" pos="0.052 0 0.020" size="0.006 0.044 0.014" rgba="0.20 0.22 0.25 1"/>
      <geom name="stand_endstop_front" type="box" pos="0 -0.040 0.019" size="0.070 0.004 0.013" rgba="0.20 0.22 0.25 1"/>
      <geom name="stand_endstop_back" type="box" pos="0 0.040 0.019" size="0.070 0.004 0.013" rgba="0.20 0.22 0.25 1"/>
      <geom name="position_a_left_guide" type="box" pos="-0.038 -0.022 0.031" size="0.004 0.018 0.012" rgba="0.28 0.30 0.34 1"/>
      <geom name="position_a_right_guide" type="box" pos="-0.038 0.022 0.031" size="0.004 0.018 0.012" rgba="0.28 0.30 0.34 1"/>
      <geom name="position_b_left_guide" type="box" pos="0.038 -0.022 0.031" size="0.004 0.018 0.012" rgba="0.28 0.30 0.34 1"/>
      <geom name="position_b_right_guide" type="box" pos="0.038 0.022 0.031" size="0.004 0.018 0.012" rgba="0.28 0.30 0.34 1"/>
      <site name="stand_position_a" pos="-0.038 0 0.032" size="0.003"/>
      <site name="stand_position_b" pos="0.038 0 0.032" size="0.003"/>
      <site name="cap_rest_area" pos="0 0.034 0.030" size="0.012 0.012 0.003" type="box"/>
    </body>

    <body name="sample_tube_352070" pos="-0.038 0 0.032">
      <joint name="sample_tube_free" type="free"/>
      <geom name="sample_tube_shell" type="mesh" mesh="falcon_tube_mesh" rgba="0.78 0.82 0.88 0.42" mass="0.012"/>
      <site name="sample_tube_grasp" pos="0 0 0.072" size="0.004"/>
      <site name="sample_tube_mouth" pos="0 0 0.11455" size="0.003"/>
    </body>
    <body name="sample_cap_358206" pos="-0.038 0 0.032">
      <joint name="sample_cap_free" type="free"/>
      <geom name="sample_cap_shell" type="mesh" mesh="falcon_cap_mesh" rgba="0.03 0.16 0.72 1" mass="0.006"/>
      <geom name="sample_cap_thread_rib_1" type="torus" pos="0 0 0.1050" size="0.0167 0.00065" rgba="0.03 0.16 0.72 1"/>
      <geom name="sample_cap_thread_rib_2" type="torus" pos="0 0 0.1080" size="0.0167 0.00065" rgba="0.03 0.16 0.72 1"/>
      <site name="sample_cap_grasp" pos="0 0 0.109" size="0.004"/>
    </body>

    <body name="balance_tube_352070" pos="0.038 0 0.032">
      <joint name="balance_tube_free" type="free"/>
      <geom name="balance_tube_shell" type="mesh" mesh="falcon_tube_mesh" rgba="0.78 0.82 0.88 0.42" mass="0.012"/>
      <site name="balance_tube_grasp" pos="0 0 0.072" size="0.004"/>
      <site name="balance_tube_mouth" pos="0 0 0.11455" size="0.003"/>
    </body>
    <body name="balance_cap_358206" pos="0.038 0 0.032">
      <joint name="balance_cap_free" type="free"/>
      <geom name="balance_cap_shell" type="mesh" mesh="falcon_cap_mesh" rgba="0.03 0.16 0.72 1" mass="0.006"/>
      <geom name="balance_cap_thread_rib_1" type="torus" pos="0 0 0.1050" size="0.0167 0.00065" rgba="0.03 0.16 0.72 1"/>
      <geom name="balance_cap_thread_rib_2" type="torus" pos="0 0 0.1080" size="0.0167 0.00065" rgba="0.03 0.16 0.72 1"/>
      <site name="balance_cap_grasp" pos="0 0 0.109" size="0.004"/>
    </body>
  </worldbody>
</mujoco>'''
    with open(os.path.join(output_dir, 'model.xml'), 'w', encoding='utf-8') as f:
        f.write(xml)

    manifest = {
        "schema_version": "1.0",
        "asset_ids": ["A05"],
        "parts": [
            {"id": "A05_sample_tube", "body": "sample_tube_352070"},
            {"id": "A05_sample_cap", "body": "sample_cap_358206"},
            {"id": "A05_balance_tube", "body": "balance_tube_352070"},
            {"id": "A05_balance_cap", "body": "balance_cap_358206"},
            {"id": "A05_support_stand", "body": "support_stand"}
        ],
        "interfaces": [
            {"id": "A05_sample_grasp", "asset_id": "A05", "kind": "grasp", "body": "sample_tube_352070", "site": "sample_tube_grasp", "approach_axis": [0, 0, 1]},
            {"id": "A05_sample_cap_rotate", "asset_id": "A05", "kind": "rotate", "body": "sample_cap_358206", "site": "sample_cap_grasp", "approach_axis": [0, 0, 1], "operation": {"status": "declared", "axis": [0, 0, 1], "joint": "sample_cap_free", "range": [-6.283, 6.283], "clearance_m": 0.003, "grasp_geoms": ["sample_cap_shell"], "basis": {"axis": {"kind": "assumption", "reference": "cap rotational symmetry and screw-cap function"}}}},
            {"id": "A05_balance_grasp", "asset_id": "A05", "kind": "grasp", "body": "balance_tube_352070", "site": "balance_tube_grasp", "approach_axis": [0, 0, 1]},
            {"id": "A05_stand_mount", "asset_id": "A05", "kind": "mount", "body": "support_stand", "site": "stand_position_a", "approach_axis": [0, 0, 1]},
            {"id": "A05_pipette_mouth", "asset_id": "A05", "kind": "insert", "body": "sample_tube_352070", "site": "sample_tube_mouth", "approach_axis": [0, 0, 1]}
        ],
        "assumptions": [
            "支撑孔采用两个开放式导向位置，名义导向间隙约0.8毫米；孔径、深度和导轨坐标为建模值而非厂家测量。",
            "管体底部留在支撑架导向区域上方约32毫米局部承托高度，支撑架两侧导轨提供防旋转反力。",
            "盖子的螺纹以同轴自由刚体和环形螺纹筋近似，未认证真实螺纹啮合、旋盖扭矩或密封。",
            "平衡管为同型独立实体，用于离心配平；未生成液体或生化内容物。"
        ],
        "approximations": [
            "透明管采用具有真实内腔的程序化薄壁网格，内底锥面和顶部开口保留。",
            "358206盖采用薄壁裙边、顶盖和两道环形筋的刚体近似。",
            "支撑架采用开放导轨和端挡块近似，可支持旋盖、取放和倾倒姿态；实际夹持力未验证。"
        ],
        "buttons": [],
        "connections": [],
        "construction_records": []
    }
    with open(os.path.join(output_dir, 'interfaces.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
