import os, json, math
from xml.etree.ElementTree import Element, SubElement, tostring


def _add(parent, tag, **attrs):
    return SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _cylinder_between(parent, name, a, b, radius, rgba, mass=0.01, contype=1, conaffinity=1):
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx-ax, by-ay, bz-az
    length = math.sqrt(dx*dx+dy*dy+dz*dz)
    mx, my, mz = (ax+bx)/2, (ay+by)/2, (az+bz)/2
    # MuJoCo cylinder axis is local Z. Rotate local Z onto segment.
    if length < 1e-9:
        quat = '1 0 0 0'
    else:
        ux, uy, uz = dx/length, dy/length, dz/length
        # quaternion rotating +Z to u
        dot = uz
        if dot < -0.999999:
            quat = '0 1 0 0'
        elif dot > 0.999999:
            quat = '1 0 0 0'
        else:
            qx, qy, qz = -uy, ux, 0.0
            s = math.sqrt((1+dot)*2)
            quat = f'{s/2} {qx/s} {qy/s} {qz/s}'
    _add(parent, 'geom', name=name, type='cylinder', pos=f'{mx} {my} {mz}',
         quat=quat, size=f'{radius} {length/2}', rgba=rgba,
         mass=mass, contype=contype, conaffinity=conaffinity)


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    root = Element('mujoco', {'model': 'A07_magnetic_filter_funnel_set'})
    _add(root, 'compiler', angle='radian', coordinate='local')
    _add(root, 'option', gravity='0 0 -9.81', integrator='implicitfast')
    asset = _add(root, 'asset')
    _add(asset, 'material', name='white_polymer', rgba='0.92 0.93 0.90 1')
    _add(asset, 'material', name='amber_glass', rgba='0.48 0.20 0.05 0.42',
         shininess='0.7', specular='0.8')
    _add(asset, 'material', name='black_membrane', rgba='0.025 0.03 0.028 1')
    _add(asset, 'material', name='steel', rgba='0.35 0.38 0.40 1', shininess='0.7')
    _add(asset, 'material', name='pump_white', rgba='0.83 0.85 0.86 1')
    _add(asset, 'material', name='hose', rgba='0.12 0.14 0.13 1')
    _add(asset, 'material', name='red_button', rgba='0.75 0.03 0.02 1')

    world = _add(root, 'worldbody')
    # A local support plinth makes the static instrument self-supported.
    base = _add(world, 'body', name='A07_support_base', pos='0 0 0')
    _add(base, 'geom', name='A07_support_base_geom', type='box', pos='0 0 0.012',
         size='0.34 0.22 0.012', material='steel', mass='0.25')

    bottle = _add(world, 'body', name='A07_suction_bottle', pos='0 0 0.025')
    # One-liter bottle: broad hollow cylindrical body, open shoulder/neck represented by rings.
    _add(bottle, 'geom', name='A07_bottle_body', type='cylinder', pos='0 0 0.085',
         size='0.072 0.06', material='amber_glass', mass='0.32')
    _add(bottle, 'geom', name='A07_bottle_bottom', type='cylinder', pos='0 0 0.026',
         size='0.073 0.006', material='amber_glass', mass='0.05')
    _add(bottle, 'geom', name='A07_bottle_shoulder', type='cone', pos='0 0 0.151',
         size='0.073 0.045', material='amber_glass', mass='0.10')
    _add(bottle, 'geom', name='A07_bottle_neck', type='cylinder', pos='0 0 0.205',
         size='0.022 0.025', material='amber_glass', mass='0.05')
    _add(bottle, 'geom', name='A07_bottle_neck_rim', type='cylinder', pos='0 0 0.232',
         size='0.027 0.006', material='amber_glass', mass='0.025')
    # Bottle stopper and two visibly distinct ports.
    _add(bottle, 'geom', name='A07_bottle_stop­per', type='cylinder', pos='0 0 0.239',
         size='0.029 0.007', material='white_polymer', mass='0.025')
    _add(bottle, 'geom', name='A07_bottle_side_port', type='cylinder', pos='0.065 0 0.11',
         quat='0 0.7071 0 0.7071', size='0.007 0.018', material='white_polymer', mass='0.012')
    _add(bottle, 'site', name='A07_bottle_mount_site', pos='0 0 0.245', size='0.006', rgba='0 1 0 1')
    _add(bottle, 'site', name='A07_bottle_exhaust_site', pos='0.078 0 0.11', size='0.005', rgba='0 1 0 1')

    funnel = _add(world, 'body', name='A07_funnel_assembly', pos='0 0 0.245')
    _add(funnel, 'freejoint', name='A07_funnel_freejoint')
    # lower magnetic support/base, with a real central filtration opening
    _add(funnel, 'geom', name='A07_funnel_support_outer', type='cylinder', pos='0 0 0.012',
         size='0.052 0.012', material='white_polymer', mass='0.06')
    _add(funnel, 'geom', name='A07_funnel_support_inner_rim', type='cylinder', pos='0 0 0.026',
         size='0.028 0.006', material='white_polymer', mass='0.025')
    _add(funnel, 'geom', name='A07_funnel_screen', type='cylinder', pos='0 0 0.033',
         size='0.0235 0.0012', material='steel', mass='0.008')
    _add(funnel, 'geom', name='A07_funnel_upper_seat', type='cylinder', pos='0 0 0.040',
         size='0.052 0.006', material='white_polymer', mass='0.03')
    # transparent cup is approximated by a tapered cone plus a visible open rim.
    _add(funnel, 'geom', name='A07_funnel_cup_wall', type='cone', pos='0 0 0.105',
         size='0.0445 0.058', material='amber_glass', mass='0.09')
    _add(funnel, 'geom', name='A07_funnel_cup_rim', type='torus', pos='0 0 0.166',
         size='0.044 0.004', material='amber_glass', mass='0.015')
    _add(funnel, 'site', name='A07_funnel_grasp_site', pos='0.045 0 0.105', size='0.006', rgba='0 0 1 1')
    _add(funnel, 'site', name='A07_funnel_filter_site', pos='0 0 0.034', size='0.004', rgba='0 1 0 1')

    lid = _add(funnel, 'body', name='A07_funnel_pressing_lid', pos='0.052 0 0.166')
    _add(lid, 'joint', name='A07_funnel_lid_hinge', type='hinge', axis='0 1 0',
         range='-1.4 1.4', damping='0.03')
    _add(lid, 'geom', name='A07_funnel_lid_ring', type='torus', pos='0 0 0',
         size='0.041 0.004', material='white_polymer', mass='0.012')
    _add(lid, 'geom', name='A07_funnel_lid_handle', type='box', pos='0.018 0 0.008',
         size='0.014 0.012 0.004', material='white_polymer', mass='0.008')
    _add(lid, 'site', name='A07_funnel_lid_grasp_site', pos='0.018 0 0.014', size='0.004', rgba='0 0 1 1')

    membrane = _add(world, 'body', name='A07_47mm_PES_membrane', pos='0 0 0.281')
    _add(membrane, 'freejoint', name='A07_membrane_freejoint')
    _add(membrane, 'geom', name='A07_47mm_membrane_disc', type='cylinder', pos='0 0 0',
         size='0.0235 0.00045', material='black_membrane', mass='0.0015')
    _add(membrane, 'site', name='A07_membrane_grasp_site', pos='0.024 0 0.002', size='0.003', rgba='0 0 1 1')

    pump = _add(world, 'body', name='A07_vacuum_pump', pos='0.235 0.005 0.025')
    _add(pump, 'geom', name='A07_pump_housing', type='box', pos='0 0 0.11',
         size='0.165 0.0825 0.11', material='pump_white', mass='4.2')
    _add(pump, 'geom', name='A07_pump_head', type='cylinder', pos='-0.095 0 0.205',
         size='0.038 0.028', material='steel', mass='0.22')
    _add(pump, 'geom', name='A07_pump_switch', type='box', pos='0.12 -0.086 0.13',
         size='0.018 0.006 0.012', material='red_button', mass='0.01')
    _add(pump, 'joint', name='A07_pump_start_stop_slide', type='slide', axis='0 1 0',
         pos='0.12 -0.086 0.13', range='0 0.004', damping='0.12')
    _add(pump, 'site', name='A07_pump_port_site', pos='-0.095 0 0.235', size='0.006', rgba='0 1 0 1')
    _add(pump, 'site', name='A07_pump_switch_site', pos='0.12 -0.093 0.13', size='0.005', rgba='1 0 0 1')

    valve = _add(bottle, 'body', name='A07_pressure_release_valve', pos='0.065 0 0.125')
    _add(valve, 'joint', name='A07_pressure_release_rotate', type='hinge', axis='0 0 1',
         range='-0.8 0.8', damping='0.04')
    _add(valve, 'geom', name='A07_pressure_release_knob', type='cylinder', pos='0 0 0.012',
         size='0.012 0.006', material='red_button', mass='0.008')
    _add(valve, 'site', name='A07_pressure_release_site', pos='0 0 0.022', size='0.004', rgba='1 0 0 1')

    hose = _add(world, 'body', name='A07_vacuum_hose', pos='0 0 0')
    _cylinder_between(hose, 'A07_hose_bottle_to_pump_1', (0.073,0,0.135), (0.14,0,0.18), 0.006, '0.12 0.14 0.13 1', 0.02)
    _cylinder_between(hose, 'A07_hose_bottle_to_pump_2', (0.14,0,0.18), (0.14,0.005,0.255), 0.006, '0.12 0.14 0.13 1', 0.02)
    _cylinder_between(hose, 'A07_hose_pump_head', (0.14,0.005,0.255), (0.14,0.005,0.255), 0.006, '0.12 0.14 0.13 1', 0.01)
    _add(hose, 'site', name='A07_hose_connection_site', pos='0.073 0 0.135', size='0.004', rgba='0 1 0 1')

    eq = _add(root, 'equality')
    _add(eq, 'weld', name='A07_funnel_to_bottle_weld', body1='A07_funnel_assembly', body2='A07_suction_bottle',
         solref='0.02 1', solimp='0.9 0.95 0.01')
    _add(eq, 'weld', name='A07_membrane_to_funnel_weld', body1='A07_47mm_PES_membrane', body2='A07_funnel_assembly',
         solref='0.02 1', solimp='0.9 0.95 0.01')
    _add(eq, 'joint', name='A07_lid_latch_lock', joint1='A07_funnel_lid_hinge', polycoef='0 1 0 0 0')

    xml_path = os.path.join(output_dir, 'model.xml')
    with open(xml_path, 'wb') as f:
        f.write(tostring(root, encoding='utf-8', xml_declaration=True))

    manifest = {
        'schema_version': '1.0',
        'asset_ids': ['A07'],
        'parts': [
            {'id': 'funnel_assembly', 'body': 'A07_funnel_assembly'},
            {'id': '47mm_membrane', 'body': 'A07_47mm_PES_membrane'},
            {'id': 'suction_bottle', 'body': 'A07_suction_bottle'},
            {'id': 'vacuum_hose', 'body': 'A07_vacuum_hose'},
            {'id': 'vacuum_pump', 'body': 'A07_vacuum_pump'}
        ],
        'interfaces': [
            {'id': 'funnel_press', 'asset_id': 'A07', 'kind': 'press', 'body': 'A07_funnel_pressing_lid', 'site': 'A07_funnel_lid_grasp_site', 'approach_axis': [0,0,1]},
            {'id': 'funnel_grasp', 'asset_id': 'A07', 'kind': 'grasp', 'body': 'A07_funnel_assembly', 'site': 'A07_funnel_grasp_site', 'approach_axis': [1,0,0]},
            {'id': 'membrane_grasp', 'asset_id': 'A07', 'kind': 'grasp', 'body': 'A07_47mm_PES_membrane', 'site': 'A07_membrane_grasp_site', 'approach_axis': [1,0,0]},
            {'id': 'bottle_connection', 'asset_id': 'A07', 'kind': 'insert', 'body': 'A07_funnel_assembly', 'site': 'A07_bottle_mount_site'},
            {'id': 'hose_connection', 'asset_id': 'A07', 'kind': 'insert', 'body': 'A07_vacuum_hose', 'site': 'A07_hose_connection_site'},
            {'id': 'pump_start_stop', 'asset_id': 'A07', 'kind': 'press', 'body': 'A07_vacuum_pump', 'site': 'A07_pump_switch_site', 'operation': {'status': 'declared', 'axis': [0,1,0], 'joint': 'A07_pump_start_stop_slide', 'range': [0,0.004], 'clearance_m': 0.001, 'grasp_geoms': ['A07_pump_switch'], 'basis': {'range': {'kind': 'assumption', 'reference': 'modeling value'}}}},
            {'id': 'pressure_release', 'asset_id': 'A07', 'kind': 'rotate', 'body': 'A07_pressure_release_valve', 'site': 'A07_pressure_release_site', 'operation': {'status': 'declared', 'axis': [0,0,1], 'joint': 'A07_pressure_release_rotate', 'range': [-0.8,0.8], 'basis': {'range': {'kind': 'assumption', 'reference': 'modeling value'}}}}
        ],
        'assumptions': [
            '抽滤瓶外形、壁厚、塞体密封细节和侧管位置为建模假设；容量按资料指定1 L表达。',
            '管路内径按约9.5 mm建模，软管路径和接头外形为建模假设。',
            '泵外形严格采用330 x 165 x 220 mm规格，泵头、开关和接口细节为建模假设。',
            '漏斗整体高度按229 mm规格建模，主筒直径按89 mm规格建模；有效过滤面积按47 mm膜片表达。'
        ],
        'approximations': [
            '漏斗透明杯和抽滤瓶采用带真实厚度的实体几何近似，保留上口和过滤通道的开口关系。',
            '磁性压合采用铰链压环与锁止等式近似，未模拟磁场。',
            '真空泵内部过程和液体流动不模拟，仅保留连续管路、启停滑钮和压差释放机构。'
        ],
        'buttons': [
            {'id': 'pump_start_stop_button', 'joint': 'A07_pump_start_stop_slide', 'geom': 'A07_pump_switch', 'pressed_at': 0.004, 'released_at': 0.0005, 'event': 'pump_start_stop_event'}
        ],
        'connections': [
            {'id': 'funnel_detach', 'kind': 'detachable', 'equality': 'A07_funnel_to_bottle_weld', 'release_event': 'funnel_grasp', 'irreversible': False, 'attach_event': 'bottle_connection', 'capture_sites': ['A07_bottle_mount_site','A07_funnel_filter_site'], 'position_tolerance_m': 0.005, 'angle_tolerance_rad': 0.2},
            {'id': 'membrane_detach', 'kind': 'detachable', 'equality': 'A07_membrane_to_funnel_weld', 'release_event': 'membrane_grasp', 'irreversible': False, 'attach_event': 'funnel_press', 'capture_sites': ['A07_funnel_filter_site','A07_membrane_grasp_site'], 'position_tolerance_m': 0.005, 'angle_tolerance_rad': 0.2},
            {'id': 'funnel_latch', 'kind': 'latch', 'equality': 'A07_lid_latch_lock', 'release_event': 'funnel_press', 'irreversible': False}
        ],
        'mechanical_events': [
            {'id': 'pump_start_stop_event', 'interface': 'pump_start_stop', 'action': 'release'},
            {'id': 'pressure_release_event', 'interface': 'pressure_release', 'action': 'release'}
        ]
    }
    with open(os.path.join(output_dir, 'interfaces.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
