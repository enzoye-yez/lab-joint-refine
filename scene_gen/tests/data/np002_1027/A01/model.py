import os
import json
import xml.etree.ElementTree as ET


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def box(parent, name, pos, size, rgba, mass=None, contype=1, conaffinity=1):
    a = dict(name=name, type='box', pos='%g %g %g' % tuple(pos), size='%g %g %g' % tuple(size), rgba='%g %g %g %g' % tuple(rgba), contype=contype, conaffinity=conaffinity)
    if mass is not None:
        a['mass'] = str(mass)
    return add(parent, 'geom', **a)


def cyl(parent, name, pos, radius, half, rgba, axis='0 0 1', mass=None, contype=1, conaffinity=1):
    a = dict(name=name, type='cylinder', pos='%g %g %g' % tuple(pos), size='%g %g' % (radius, half), axis=axis, rgba='%g %g %g %g' % tuple(rgba), contype=contype, conaffinity=conaffinity)
    if mass is not None:
        a['mass'] = str(mass)
    return add(parent, 'geom', **a)


def site(parent, name, pos, size='0.012', rgba='0.1 0.8 0.2 1'):
    return add(parent, 'site', name=name, pos='%g %g %g' % tuple(pos), size=size, rgba=rgba)


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    root = ET.Element('mujoco', {'model': 'VFC_PM10_air_sampler_A01'})
    compiler = add(root, 'compiler', angle='radian', autolimits='true')
    add(root, 'option', timestep='0.002', integrator='implicitfast', gravity='0 0 -9.81')
    asset = add(root, 'asset')
    add(asset, 'material', name='sampler_white', rgba='0.82 0.84 0.86 1', roughness='0.65')
    add(asset, 'material', name='dark_metal', rgba='0.07 0.08 0.09 1', roughness='0.42', metallic='0.35')
    add(asset, 'material', name='rubber', rgba='0.025 0.03 0.035 1', roughness='0.9')
    add(asset, 'material', name='quartz_filter', rgba='0.12 0.13 0.14 1', roughness='0.95')
    add(asset, 'material', name='orange_lock', rgba='0.82 0.35 0.06 1', roughness='0.7')
    add(asset, 'material', name='steel', rgba='0.42 0.45 0.48 1', roughness='0.35', metallic='0.65')

    world = add(root, 'worldbody')

    chassis = add(world, 'body', name='sampler_chassis', pos='0 0 0')
    # Four independently visible feet provide the actual ground support.
    for i, (x, y) in enumerate(((-0.27, -0.21), (0.27, -0.21), (-0.27, 0.21), (0.27, 0.21))):
        foot = add(chassis, 'body', name='sampler_foot_%d' % i, pos='%g %g 0.045' % (x, y))
        box(foot, 'sampler_foot_geom_%d' % i, (0, 0, 0), (0.045, 0.045, 0.045), (0.25, 0.27, 0.29, 1), 0.35)
    box(chassis, 'sampler_lower_case', (0, 0, 0.34), (0.31, 0.255, 0.29), (0.82, 0.84, 0.86, 1), 13.0)
    box(chassis, 'sampler_front_panel', (0, 0.255, 0.65), (0.31, 0.018, 0.22), (0.76, 0.78, 0.80, 1), 2.0)
    box(chassis, 'sampler_left_panel', (-0.292, 0, 0.67), (0.018, 0.24, 0.25), (0.80, 0.82, 0.84, 1), 1.7)
    box(chassis, 'sampler_right_panel', (0.292, 0, 0.67), (0.018, 0.24, 0.25), (0.80, 0.82, 0.84, 1), 1.7)
    box(chassis, 'sampler_back_panel', (0, -0.238, 0.67), (0.29, 0.018, 0.25), (0.78, 0.80, 0.82, 1), 1.5)
    # The top is a frame, not a solid cover: the inlet/filter opening remains accessible.
    box(chassis, 'top_frame_front', (0, 0.205, 0.865), (0.31, 0.045, 0.025), (0.72, 0.74, 0.76, 1), 0.7)
    box(chassis, 'top_frame_back', (0, -0.205, 0.865), (0.31, 0.045, 0.025), (0.72, 0.74, 0.76, 1), 0.7)
    box(chassis, 'top_frame_left', (-0.265, 0, 0.865), (0.045, 0.165, 0.025), (0.72, 0.74, 0.76, 1), 0.7)
    box(chassis, 'top_frame_right', (0.265, 0, 0.865), (0.045, 0.165, 0.025), (0.72, 0.74, 0.76, 1), 0.7)
    # PM10 inlet represented by a short hollow-looking stacked ring/frame around the opening.
    inlet = add(chassis, 'body', name='pm10_inlet', pos='0 0 0.92')
    box(inlet, 'inlet_front_wall', (0, 0.18, 0), (0.22, 0.025, 0.035), (0.14, 0.15, 0.16, 1), 0.35)
    box(inlet, 'inlet_back_wall', (0, -0.18, 0), (0.22, 0.025, 0.035), (0.14, 0.15, 0.16, 1), 0.35)
    box(inlet, 'inlet_left_wall', (-0.195, 0, 0), (0.025, 0.155, 0.035), (0.14, 0.15, 0.16, 1), 0.35)
    box(inlet, 'inlet_right_wall', (0.195, 0, 0), (0.025, 0.155, 0.035), (0.14, 0.15, 0.16, 1), 0.35)
    site(inlet, 'pm10_air_inlet_site', (0, 0, 0.045), '0.018')
    site(chassis, 'sampler_clamp_mount_site', (0, 0, 0.89), '0.015')

    # Hinged sampler lid, initially closed and mechanically connected to the chassis.
    lid = add(chassis, 'body', name='sampler_lid', pos='0 -0.245 0.91')
    add(lid, 'joint', name='sampler_lid_hinge', type='hinge', axis='1 0 0', range='-1.5708 0.12', damping='0.08', armature='0.002')
    box(lid, 'lid_panel', (0, 0.245, 0.025), (0.31, 0.245, 0.018), (0.84, 0.86, 0.88, 1), 2.2)
    box(lid, 'lid_front_rail', (0, 0.455, 0.055), (0.31, 0.025, 0.04), (0.18, 0.19, 0.20, 1), 0.35)
    box(lid, 'lid_left_rail', (-0.285, 0.245, 0.055), (0.025, 0.215, 0.04), (0.18, 0.19, 0.20, 1), 0.35)
    box(lid, 'lid_right_rail', (0.285, 0.245, 0.055), (0.025, 0.215, 0.04), (0.18, 0.19, 0.20, 1), 0.35)
    site(lid, 'lid_grasp_site', (0, 0.46, 0.07), '0.014')

    # Independent removable filter clamp; weld equality represents the installed locked state.
    clamp = add(world, 'body', name='filter_clamp', pos='0 0 0.905')
    add(clamp, 'freejoint', name='filter_clamp_freejoint')
    box(clamp, 'clamp_base_front', (0, 0.105, 0), (0.145, 0.018, 0.012), (0.08, 0.09, 0.10, 1), 0.28)
    box(clamp, 'clamp_base_back', (0, -0.105, 0), (0.145, 0.018, 0.012), (0.08, 0.09, 0.10, 1), 0.28)
    box(clamp, 'clamp_base_left', (-0.145, 0, 0), (0.018, 0.09, 0.012), (0.08, 0.09, 0.10, 1), 0.28)
    box(clamp, 'clamp_base_right', (0.145, 0, 0), (0.018, 0.09, 0.012), (0.08, 0.09, 0.10, 1), 0.28)
    box(clamp, 'clamp_upper_front', (0, 0.105, 0.018), (0.145, 0.018, 0.012), (0.16, 0.17, 0.18, 1), 0.22)
    box(clamp, 'clamp_upper_back', (0, -0.105, 0.018), (0.145, 0.018, 0.012), (0.16, 0.17, 0.18, 1), 0.22)
    box(clamp, 'clamp_upper_left', (-0.145, 0, 0.018), (0.018, 0.09, 0.012), (0.16, 0.17, 0.18, 1), 0.22)
    box(clamp, 'clamp_upper_right', (0.145, 0, 0.018), (0.018, 0.09, 0.012), (0.16, 0.17, 0.18, 1), 0.22)
    lock = add(clamp, 'body', name='filter_clamp_lock', pos='0 0.132 0.018')
    add(lock, 'joint', name='filter_clamp_lock_slide', type='slide', axis='0 1 0', range='0 0.010', damping='0.15', armature='0.001')
    box(lock, 'filter_clamp_lock_button', (0, 0, 0), (0.026, 0.012, 0.018), (0.82, 0.35, 0.06, 1), 0.08)
    site(clamp, 'filter_clamp_grasp_site', (0, 0, 0.035), '0.014')
    site(clamp, 'filter_clamp_capture_site', (0, 0, 0.025), '0.009')

    # Full specified quartz filter. The very small thickness is retained, with collision volume.
    filt = add(world, 'body', name='quartz_filter_203x254', pos='0 0 0.934')
    add(filt, 'freejoint', name='quartz_filter_freejoint')
    box(filt, 'quartz_filter_membrane', (0, 0, 0), (0.1016, 0.127, 0.00035), (0.12, 0.13, 0.14, 1), 0.018)
    # Thin perimeter strips make the membrane edge and cuttable boundary visually legible without sealing the sampler opening.
    box(filt, 'quartz_filter_edge_front', (0, 0.1262, 0.0005), (0.100, 0.0008, 0.00035), (0.20, 0.21, 0.22, 1), 0.002)
    box(filt, 'quartz_filter_edge_back', (0, -0.1262, 0.0005), (0.100, 0.0008, 0.00035), (0.20, 0.21, 0.22, 1), 0.002)
    box(filt, 'quartz_filter_edge_left', (-0.1008, 0, 0.0005), (0.0008, 0.125, 0.00035), (0.20, 0.21, 0.22, 1), 0.002)
    box(filt, 'quartz_filter_edge_right', (0.1008, 0, 0.0005), (0.0008, 0.125, 0.00035), (0.20, 0.21, 0.22, 1), 0.002)
    site(filt, 'quartz_filter_grasp_site', (0, 0, 0.002), '0.012')
    site(filt, 'quartz_filter_cut_site', (0.0508, 0, 0.002), '0.008')

    eq = add(root, 'equality')
    add(eq, 'weld', name='filter_clamp_installed_weld', body1='sampler_chassis', body2='filter_clamp', solref='0.02 1', solimp='0.9 0.95 0.01')

    actuator = add(root, 'actuator')
    add(actuator, 'position', name='lock_button_press_actuator', joint='filter_clamp_lock_slide', kp='20', ctrllimited='true', ctrlrange='0 0.010')

    contact = add(root, 'contact')
    add(contact, 'exclude', name='lid_hinge_self_clearance', body1='sampler_lid', body2='sampler_chassis')

    xml_path = os.path.join(output_dir, 'model.xml')
    ET.ElementTree(root).write(xml_path, encoding='utf-8', xml_declaration=True)

    manifest = {
        'schema_version': '1.0',
        'asset_ids': ['A01'],
        'parts': [
            {'id': 'A01_sampler', 'body': 'sampler_chassis'},
            {'id': 'A01_filter_clamp', 'body': 'filter_clamp'},
            {'id': 'A01_quartz_filter', 'body': 'quartz_filter_203x254'},
            {'id': 'A01_pm10_inlet', 'body': 'pm10_inlet'},
            {'id': 'A01_lid', 'body': 'sampler_lid'}
        ],
        'interfaces': [
            {'id': 'A01_grasp_filter', 'asset_id': 'A01', 'kind': 'grasp', 'body': 'quartz_filter_203x254', 'site': 'quartz_filter_grasp_site', 'approach_axis': [0, 0, 1]},
            {'id': 'A01_grasp_filter_clamp', 'asset_id': 'A01', 'kind': 'grasp', 'body': 'filter_clamp', 'site': 'filter_clamp_grasp_site', 'approach_axis': [0, 0, 1]},
            {'id': 'A01_insert_filter', 'asset_id': 'A01', 'kind': 'insert', 'body': 'quartz_filter_203x254', 'site': 'quartz_filter_grasp_site', 'approach_axis': [0, 0, -1]},
            {'id': 'A01_rotate_lid', 'asset_id': 'A01', 'kind': 'rotate', 'body': 'sampler_lid', 'site': 'lid_grasp_site', 'approach_axis': [0, 1, 0]},
            {'id': 'A01_press_clamp_lock', 'asset_id': 'A01', 'kind': 'press', 'body': 'filter_clamp', 'site': 'filter_clamp_capture_site', 'approach_axis': [0, 1, 0]},
            {'id': 'A01_cut_filter', 'asset_id': 'A01', 'kind': 'cut', 'body': 'quartz_filter_203x254', 'site': 'quartz_filter_cut_site', 'approach_axis': [0, 0, 1]}
        ],
        'assumptions': [
            'VFC PM10 sampler外形、工作高度及夹具锁止坐标未在资料中给出；采用约0.62 m宽、0.51 m深、0.96 m高的工程建模值。',
            'PM10进样口采用矩形开放式框架表达，保留滤膜进入方向和空气通道；内部气动过程仅作为状态参数。',
            '采样流量状态参数为1.13 m^3/min，未模拟气流或颗粒捕集。',
            '滤膜初始姿态为平展状态，置于独立滤膜夹内；折叠和切割由刚体膜片近似，未模拟柔性变形。'
        ],
        'approximations': [
            '滤膜使用有真实厚度的刚体薄片近似；切割使用预定义cut接口，不模拟材料断裂。',
            '滤膜夹锁止使用短行程滑块近似真实压扣，夹具通过可释放weld表达安装状态。',
            '进样器内部风道、流量控制和颗粒分离不进行生化或流体仿真。'
        ],
        'buttons': [
            {'id': 'A01_clamp_lock_button', 'joint': 'filter_clamp_lock_slide', 'geom': 'filter_clamp_lock_button', 'pressed_at': 0.010, 'released_at': 0.0005, 'event': 'A01_unlock_filter_clamp'}
        ],
        'connections': [
            {'id': 'A01_filter_clamp_connection', 'kind': 'detachable', 'equality': 'filter_clamp_installed_weld', 'release_event': 'A01_unlock_filter_clamp', 'irreversible': False, 'attach_event': 'A01_attach_filter_clamp', 'capture_sites': ['sampler_clamp_mount_site', 'filter_clamp_capture_site'], 'position_tolerance_m': 0.005, 'angle_tolerance_rad': 0.2}
        ],
        'mechanical_events': [
            {'id': 'A01_attach_filter_clamp', 'interface': 'A01_grasp_filter_clamp', 'action': 'attach'},
            {'id': 'A01_unlock_filter_clamp', 'interface': 'A01_press_clamp_lock', 'action': 'release'}
        ],
        'construction_records': [],
        'state_parameters': {'sampling_flow_m3_per_min': 1.13, 'filter_size_m': [0.2032, 0.254]}
    }
    with open(os.path.join(output_dir, 'interfaces.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
