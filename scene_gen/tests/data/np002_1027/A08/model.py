import os, json, math
from xml.etree import ElementTree as ET


def annular_mesh(path, ro, ri, z0, z1, n=48):
    verts = []
    faces = []
    for z in (z0, z1):
        for r in (ro, ri):
            for i in range(n):
                a = 2.0 * math.pi * i / n
                verts.append((r * math.cos(a), r * math.sin(a), z))
    # index: layer 0 bottom outer, 1 bottom inner, 2 top outer, 3 top inner
    bo, bi, to, ti = 0, n, 2*n, 3*n
    for i in range(n):
        j = (i + 1) % n
        faces += [
            (bo+i, bo+j, to+j), (bo+i, to+j, to+i),
            (bi+j, bi+i, ti+i), (bi+j, ti+i, ti+j),
            (to+i, to+j, ti+j), (to+i, ti+j, ti+i),
            (bo+j, bo+i, bi+i), (bo+j, bi+i, bi+j)
        ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# annular tube shell\n')
        for v in verts:
            f.write('v %.8f %.8f %.8f\n' % v)
        for a,b,c in faces:
            f.write('f %d %d %d\n' % (a+1,b+1,c+1))


def ring_mesh(path, ro, ri, h, n=48):
    annular_mesh(path, ro, ri, 0.0, h, n)


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    meshdir = os.path.join(output_dir, 'meshes')
    os.makedirs(meshdir, exist_ok=True)

    tube_mesh = 'meshes/grinding_tube_shell.obj'
    support_mesh = 'meshes/tube_support_ring.obj'
    annular_mesh(os.path.join(output_dir, tube_mesh), 0.0060, 0.00475, 0.005, 0.055, 48)
    ring_mesh(os.path.join(output_dir, support_mesh), 0.0078, 0.0057, 0.004, 48)

    root = ET.Element('mujoco', {'model': 'A08_powerbead_grinding_tube_set'})
    ET.SubElement(root, 'compiler', {'angle': 'radian', 'coordinate': 'local'})
    ET.SubElement(root, 'option', {'gravity': '0 0 -9.81', 'integrator': 'implicitfast'})

    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', {'name': 'grinding_tube_shell_mesh', 'file': tube_mesh})
    ET.SubElement(asset, 'mesh', {'name': 'tube_support_ring_mesh', 'file': support_mesh})
    ET.SubElement(asset, 'material', {'name': 'clear_polypropylene', 'rgba': '0.82 0.92 0.96 0.28', 'reflectance': '0.15'})
    ET.SubElement(asset, 'material', {'name': 'cap_material', 'rgba': '0.72 0.82 0.86 0.72', 'reflectance': '0.18'})
    ET.SubElement(asset, 'material', {'name': 'bead_material', 'rgba': '0.72 0.76 0.78 1', 'reflectance': '0.1'})
    ET.SubElement(asset, 'material', {'name': 'support_material', 'rgba': '0.18 0.22 0.24 1'})

    world = ET.SubElement(root, 'worldbody')
    base = ET.SubElement(world, 'body', {'name': 'A08_support_plate', 'pos': '0 0 0'})
    ET.SubElement(base, 'geom', {'name': 'A08_support_plate_geom', 'type': 'box', 'pos': '0 0 0.002', 'size': '0.032 0.018 0.002', 'material': 'support_material', 'mass': '0.025'})

    positions = [(-0.012, 0.0, 'sample'), (0.012, 0.0, 'balance')]
    tube_names = {}
    cap_names = {}
    for x, y, tag in positions:
        tube = ET.SubElement(world, 'body', {'name': 'A08_%s_tube' % tag, 'pos': '%.5f %.5f 0' % (x,y)})
        tube_names[tag] = tube.get('name')
        ET.SubElement(tube, 'freejoint', {'name': 'A08_%s_tube_free' % tag})
        ET.SubElement(tube, 'inertial', {'pos': '0 0 0.032', 'mass': '0.0065', 'diaginertia': '1.1e-6 1.1e-6 2.0e-8'})
        ET.SubElement(tube, 'geom', {'name': 'A08_%s_tube_shell' % tag, 'type': 'mesh', 'mesh': 'grinding_tube_shell_mesh', 'material': 'clear_polypropylene', 'contype': '1', 'conaffinity': '1'})
        ET.SubElement(tube, 'site', {'name': 'A08_%s_tube_grasp' % tag, 'pos': '0 0 0.031', 'size': '0.0032', 'type': 'sphere', 'rgba': '0.1 0.8 0.2 0.5'})
        ET.SubElement(tube, 'site', {'name': 'A08_%s_tube_mouth' % tag, 'pos': '0 0 0.055', 'size': '0.002', 'type': 'sphere', 'rgba': '0.2 0.6 0.9 0.5'})
        if tag == 'sample':
            for i, (bx, by, bz) in enumerate([
                (-0.0022,-0.0015,0.018), (0.0018,-0.0018,0.019), (0.0002,0.0019,0.021),
                (-0.0020,0.0016,0.025), (0.0021,0.0011,0.026), (-0.0007,-0.0020,0.029),
                (0.0020,-0.0002,0.032), (-0.0018,-0.0006,0.034), (0.0004,0.0018,0.036),
                (-0.0012,0.0012,0.039), (0.0015,0.0008,0.041), (0.0,-0.0012,0.043)]):
                ET.SubElement(tube, 'geom', {'name': 'A08_grinding_bead_%02d' % (i+1), 'type': 'sphere', 'pos': '%.5f %.5f %.5f' % (bx,by,bz), 'size': '0.00125', 'material': 'bead_material', 'mass': '0.00012'})
        cap = ET.SubElement(world, 'body', {'name': 'A08_%s_cap' % tag, 'pos': '%.5f %.5f 0' % (x,y)})
        cap_names[tag] = cap.get('name')
        ET.SubElement(cap, 'freejoint', {'name': 'A08_%s_cap_free' % tag})
        ET.SubElement(cap, 'inertial', {'pos': '0 0 0.061', 'mass': '0.0022', 'diaginertia': '2.0e-7 2.0e-7 2.5e-8'})
        ET.SubElement(cap, 'geom', {'name': 'A08_%s_cap_body' % tag, 'type': 'cylinder', 'pos': '0 0 0.061', 'size': '0.0065 0.006', 'material': 'cap_material', 'mass': '0.002'})
        for j, z in enumerate((0.0555, 0.0580, 0.0605)):
            ET.SubElement(cap, 'geom', {'name': 'A08_%s_cap_thread_%d' % (tag,j+1), 'type': 'cylinder', 'pos': '0 0 %.5f' % z, 'size': '0.0068 0.00045', 'material': 'cap_material', 'mass': '0.00008'})
        ET.SubElement(cap, 'site', {'name': 'A08_%s_cap_grasp' % tag, 'pos': '0 0 0.0665', 'size': '0.003', 'type': 'sphere', 'rgba': '0.9 0.6 0.1 0.6'})
        ET.SubElement(cap, 'site', {'name': 'A08_%s_cap_axis' % tag, 'pos': '0 0 0.067', 'size': '0.0015', 'type': 'sphere', 'rgba': '0.9 0.2 0.1 0.6'})

    # Visible physical support rings for both independent tubes.
    for x, y, tag in positions:
        ET.SubElement(base, 'geom', {'name': 'A08_%s_support_ring' % tag, 'type': 'mesh', 'mesh': 'tube_support_ring_mesh', 'pos': '%.5f %.5f 0.004' % (x,y), 'material': 'support_material', 'mass': '0.004'})

    eq = ET.SubElement(root, 'equality')
    ET.SubElement(eq, 'weld', {'name': 'A08_sample_cap_weld', 'body1': tube_names['sample'], 'body2': cap_names['sample'], 'solref': '0.02 1', 'solimp': '0.9 0.95 0.001'})
    ET.SubElement(eq, 'weld', {'name': 'A08_balance_cap_weld', 'body1': tube_names['balance'], 'body2': cap_names['balance'], 'solref': '0.02 1', 'solimp': '0.9 0.95 0.001'})

    xml_path = os.path.join(output_dir, 'model.xml')
    ET.ElementTree(root).write(xml_path, encoding='utf-8', xml_declaration=True)

    manifest = {
        'schema_version': '1.0',
        'asset_ids': ['A08'],
        'parts': [
            {'id': 'A08_sample_grinding_tube', 'body': 'A08_sample_tube'},
            {'id': 'A08_sample_screw_cap', 'body': 'A08_sample_cap'},
            {'id': 'A08_balance_tube', 'body': 'A08_balance_tube'},
            {'id': 'A08_balance_screw_cap', 'body': 'A08_balance_cap'},
            {'id': 'A08_support_plate', 'body': 'A08_support_plate'}
        ],
        'interfaces': [
            {'id': 'A08_sample_grasp', 'asset_id': 'A08', 'kind': 'grasp', 'body': 'A08_sample_tube', 'site': 'A08_sample_tube_grasp', 'approach_axis': [0,0,1]},
            {'id': 'A08_sample_cap_rotate', 'asset_id': 'A08', 'kind': 'rotate', 'body': 'A08_sample_cap', 'site': 'A08_sample_cap_grasp', 'approach_axis': [0,0,1], 'operation': {'status': 'declared', 'axis': [0,0,1], 'joint': 'A08_sample_cap_free', 'range': [-6.2832,6.2832], 'clearance_m': 0.001, 'grasp_geoms': ['A08_sample_cap_body'], 'basis': {'axis': {'kind':'assumption','reference':'螺旋盖旋拧轴线按管体中心轴建模'}}}},
            {'id': 'A08_balance_grasp', 'asset_id': 'A08', 'kind': 'grasp', 'body': 'A08_balance_tube', 'site': 'A08_balance_tube_grasp', 'approach_axis': [0,0,1]},
            {'id': 'A08_balance_cap_rotate', 'asset_id': 'A08', 'kind': 'rotate', 'body': 'A08_balance_cap', 'site': 'A08_balance_cap_grasp', 'approach_axis': [0,0,1], 'operation': {'status': 'declared', 'axis': [0,0,1], 'joint': 'A08_balance_cap_free', 'range': [-6.2832,6.2832], 'clearance_m': 0.001, 'grasp_geoms': ['A08_balance_cap_body'], 'basis': {'axis': {'kind':'assumption','reference':'螺旋盖旋拧轴线按管体中心轴建模'}}}},
            {'id': 'A08_sample_insert', 'asset_id': 'A08', 'kind': 'insert', 'body': 'A08_sample_tube', 'site': 'A08_sample_tube_mouth', 'approach_axis': [0,0,1]},
            {'id': 'A08_balance_insert', 'asset_id': 'A08', 'kind': 'insert', 'body': 'A08_balance_tube', 'site': 'A08_balance_tube_mouth', 'approach_axis': [0,0,1]}
        ],
        'assumptions': [
            '未公开的管体尺寸按外径约12 mm、有效长度50 mm、内径约9.5 mm建模。',
            '配平管采用与研磨管相同的外形和螺旋盖尺寸；其内部不生成研磨珠。',
            '螺纹用多道环形凸台近似，未模拟真实螺旋牙型和密封压缩。',
            '底部支撑环为本地建模的水平固定托架，用于表达离心前后承托关系。'
        ],
        'approximations': [
            '研磨珠以12个独立球体表示，不模拟颗粒间碰撞或生化过程。',
            '管盖以自由转动/平移刚体加weld装配约束近似旋紧状态；运行时操作释放尚未验证。',
            '透明塑料仅用于视觉表达，管壁碰撞采用同一封闭环形网格。'
        ],
        'buttons': [],
        'connections': [
            {'id':'A08_sample_cap_connection','kind':'detachable','equality':'A08_sample_cap_weld','release_event':'A08_sample_cap_rotate','irreversible':False,'attach_event':'A08_sample_cap_rotate','capture_sites':['A08_sample_tube_mouth','A08_sample_cap_axis'],'position_tolerance_m':0.005,'angle_tolerance_rad':0.2},
            {'id':'A08_balance_cap_connection','kind':'detachable','equality':'A08_balance_cap_weld','release_event':'A08_balance_cap_rotate','irreversible':False,'attach_event':'A08_balance_cap_rotate','capture_sites':['A08_balance_tube_mouth','A08_balance_cap_axis'],'position_tolerance_m':0.005,'angle_tolerance_rad':0.2}
        ],
        'mechanical_events': [
            {'id':'A08_sample_cap_rotate','interface':'A08_sample_cap_rotate','action':'release'},
            {'id':'A08_balance_cap_rotate','interface':'A08_balance_cap_rotate','action':'release'}
        ]
    }
    with open(os.path.join(output_dir, 'interfaces.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    build(os.path.dirname(os.path.abspath(__file__)))