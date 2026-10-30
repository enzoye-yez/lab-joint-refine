import os
import math
import json
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont


def build(output_dir):
    """Build one empty, assembled A15 rack in local metres, with its feet at z=0."""
    os.makedirs(output_dir, exist_ok=True)
    mesh_dir = os.path.join(output_dir, 'meshes')
    texture_dir = os.path.join(output_dir, 'textures')
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(texture_dir, exist_ok=True)

    def nums(values):
        return ' '.join(format(float(v), '.10g') for v in values)

    def node(parent, tag, **attrs):
        return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})

    # One reusable, positive-volume convex sector. The assembly of sectors,
    # not a convex hull of a complete ring, defines each open tube socket.
    segments = 32
    inner_r = 0.0058
    outer_r = 0.0077
    thickness = 0.003
    half_angle = math.pi / segments
    vertices = []
    for z in (-thickness / 2, thickness / 2):
        for r, a in ((inner_r, -half_angle), (outer_r, -half_angle),
                     (outer_r, half_angle), (inner_r, half_angle)):
            vertices.append((r * math.cos(a), r * math.sin(a), z))
    faces = [(1, 4, 3), (1, 3, 2), (5, 6, 7), (5, 7, 8),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
             (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]
    with open(os.path.join(mesh_dir, 'socket_sector.obj'), 'w', encoding='utf-8') as f:
        for v in vertices:
            f.write('v ' + nums(v) + '\n')
        for face in faces:
            f.write('f %d %d %d\n' % face)

    # A small safety marking, not a control or software interface.
    image = Image.new('RGB', (256, 256), (176, 179, 177))
    draw = ImageDraw.Draw(image)
    draw.polygon([(128, 15), (242, 229), (14, 229)], fill=(22, 24, 22))
    draw.polygon([(128, 39), (222, 216), (34, 216)], fill=(250, 211, 37))
    draw.line([(89, 160), (89, 105), (109, 87), (145, 87), (166, 105), (166, 160)], fill=(20, 23, 21), width=16)
    draw.rectangle((79, 150, 99, 177), fill=(20, 23, 21))
    draw.rectangle((156, 150, 176, 177), fill=(20, 23, 21))
    image.save(os.path.join(texture_dir, 'magnet_warning.png'))

    root = ET.Element('mujoco', model='A15_DynaMag_2')
    node(root, 'compiler', angle='radian', autolimits='true')
    node(root, 'option', timestep='0.002', gravity='0 0 -9.81', integrator='implicitfast')
    defaults = node(root, 'default')
    node(defaults, 'geom', friction='0.45 0.006 0.0001', condim='4',
         solref='0.008 1', solimp='0.95 0.99 0.001', margin='0', density='1100')
    node(defaults, 'site', size='0.001', rgba='0.15 0.6 0.8 0', group='4')
    assets = node(root, 'asset')
    node(assets, 'mesh', name='A15_socket_sector_mesh', file='meshes/socket_sector.obj')
    node(assets, 'material', name='A15_ivory', rgba='0.84 0.85 0.82 1', specular='0.28', roughness='0.55')
    node(assets, 'material', name='A15_magnet_gray', rgba='0.48 0.50 0.49 1', specular='0.3', roughness='0.48')
    node(assets, 'material', name='A15_clear_polymer', rgba='0.79 0.85 0.83 0.48', specular='0.65', shininess='0.65')
    node(assets, 'material', name='A15_ink', rgba='0.23 0.27 0.25 1')
    node(assets, 'texture', name='A15_warning_texture', type='2d', file='textures/magnet_warning.png')
    node(assets, 'material', name='A15_warning_material', texture='A15_warning_texture', texuniform='false', specular='0')
    world = node(root, 'worldbody')
    base = node(world, 'body', name='A15_magnet_base', pos='0 0 0')

    def box(body, name, pos, half, material, **kwargs):
        return node(body, 'geom', name=name, type='box', pos=nums(pos),
                    size=nums(half), material=material, **kwargs)

    # Each end housing is visibly continuous from its contact shoe to its
    # horizontal rack-support shoulder. The central slab is recessed below
    # the sockets and does not close either row of tube passages.
    for side, x in (('left', -0.080), ('right', 0.080)):
        box(base, 'A15_' + side + '_support_shoe', (x, 0, 0.004),
            (0.009, 0.022, 0.004), 'A15_ivory')
        box(base, 'A15_' + side + '_end_housing', (x, 0, 0.0305),
            (0.006, 0.022, 0.0225), 'A15_ivory')
        node(base, 'site', name='A15_' + side + '_support_surface',
             pos=nums((x, 0, 0.053)))
    box(base, 'A15_magnet_encapsulated_slab', (0, 0, 0.0265),
        (0.074, 0.0043, 0.0185), 'A15_magnet_gray')
    box(base, 'A15_warning_decal', (-0.065, -0.004345, 0.017),
        (0.004, 0.000045, 0.004), 'A15_warning_material',
        contype='0', conaffinity='0', mass='0')
    node(base, 'site', name='A15_base_mount_site', pos='0 0 0.053')
    node(base, 'site', name='A15_base_grasp_site', pos='-0.080 -0.022 0.032')

    # Separate permanent motion tree from the first frame; weld assembly
    # supplements, rather than substitutes for, the end-shoulder supports.
    rack = node(world, 'body', name='A15_upper_rack', pos='0 0 0.053')
    node(rack, 'freejoint', name='A15_upper_rack_free')
    mass = 0.055
    sx, sy, sz = 0.172, 0.0486, 0.006
    inertia = (mass * (sy * sy + sz * sz) / 12,
               mass * (sx * sx + sz * sz) / 12,
               mass * (sx * sx + sy * sy) / 12)
    node(rack, 'inertial', pos='0 0 0', mass=str(mass), diaginertia=nums(inertia))
    for side, x in (('left', -0.080), ('right', 0.080)):
        box(rack, 'A15_' + side + '_rack_saddle', (x, 0, 0.0015),
            (0.006, 0.0202, 0.0015), 'A15_clear_polymer')
    for row, y in (('front', -0.019), ('rear', 0.019)):
        box(rack, 'A15_' + row + '_longitudinal_rail', (0, y, -0.0005),
            (0.074, 0.0012, 0.0015), 'A15_clear_polymer')
    box(rack, 'A15_central_rack_spine', (0, 0, -0.0005),
        (0.074, 0.0038, 0.0015), 'A15_clear_polymer')
    node(rack, 'site', name='A15_rack_mount_site', pos='0 0 0')
    node(rack, 'site', name='A15_rack_left_grasp_site', pos='-0.080 0 0.003')
    node(rack, 'site', name='A15_rack_right_grasp_site', pos='0.080 0 0.003')

    interfaces = [
        {'id': 'A15.base.grasp', 'asset_id': 'A15', 'kind': 'grasp',
         'body': 'A15_magnet_base', 'site': 'A15_base_grasp_site',
         'approach_axis': [0, -1, 0]},
        {'id': 'A15.rack.grasp.left', 'asset_id': 'A15', 'kind': 'grasp',
         'body': 'A15_upper_rack', 'site': 'A15_rack_left_grasp_site',
         'approach_axis': [0, 0, 1]},
        {'id': 'A15.rack.grasp.right', 'asset_id': 'A15', 'kind': 'grasp',
         'body': 'A15_upper_rack', 'site': 'A15_rack_right_grasp_site',
         'approach_axis': [0, 0, 1]},
        {'id': 'A15.base.mount', 'asset_id': 'A15', 'kind': 'mount',
         'body': 'A15_magnet_base', 'site': 'A15_base_mount_site',
         'approach_axis': [0, 0, 1]},
        {'id': 'A15.rack.mount', 'asset_id': 'A15', 'kind': 'mount',
         'body': 'A15_upper_rack', 'site': 'A15_rack_mount_site'}
    ]
    sockets = []
    # Geometric numerals, so all sixteen socket identifiers remain visible
    # without any font files or nonportable font dependency.
    strokes = {
        'a': ((0, 1), (1, 1)), 'b': ((1, 1), (1, 0.5)),
        'c': ((1, 0.5), (1, 0)), 'd': ((0, 0), (1, 0)),
        'e': ((0, 0.5), (0, 0)), 'f': ((0, 1), (0, 0.5)),
        'g': ((0, 0.5), (1, 0.5))
    }
    digits = {'0': 'abcdef', '1': 'bc', '2': 'abged', '3': 'abgcd',
              '4': 'fgbc', '5': 'afgcd', '6': 'afgecd', '7': 'abc',
              '8': 'abcdefg', '9': 'abfgcd'}
    for row, y, offset in (('front', -0.0108, 0), ('rear', 0.0108, 8)):
        sign = -1 if y < 0 else 1
        for i in range(8):
            number = offset + i + 1
            key = '%02d' % number
            x = (i - 3.5) * 0.018
            for sector in range(segments):
                angle = 2 * math.pi * sector / segments
                node(rack, 'geom', name='A15_socket_' + key + '_sector_%02d' % sector,
                     type='mesh', mesh='A15_socket_sector_mesh',
                     pos=nums((x, y, -0.0005)),
                     quat=nums((math.cos(angle / 2), 0, 0, math.sin(angle / 2))),
                     material='A15_clear_polymer')
            box(rack, 'A15_socket_' + key + '_number_tab',
                (x, sign * 0.022, -0.0005), (0.0033, 0.0023, 0.0015),
                'A15_clear_polymer')
            text = str(number)
            digit_w, digit_h, digit_gap = 0.00105, 0.0020, 0.00045
            text_w = len(text) * digit_w + (len(text) - 1) * digit_gap
            for d_index, char in enumerate(text):
                left = x - text_w / 2 + d_index * (digit_w + digit_gap)
                bottom = sign * 0.022 - digit_h / 2
                for stroke in digits[char]:
                    p, q = strokes[stroke]
                    a = (left + p[0] * digit_w, bottom + p[1] * digit_h, 0.001075)
                    b = (left + q[0] * digit_w, bottom + q[1] * digit_h, 0.001075)
                    node(rack, 'geom', name='A15_label_' + key + '_%d_' % d_index + stroke,
                         type='capsule', fromto=nums(a + b), size='0.000075',
                         material='A15_ink', contype='0', conaffinity='0', mass='0')
            site_name = 'A15_socket_' + key + '_site'
            node(rack, 'site', name=site_name, pos=nums((x, y, 0.001)), size='0.002')
            insert_id = 'A15.socket.' + key + '.insert'
            rotate_id = 'A15.socket.' + key + '.rotate'
            interfaces.append({'id': insert_id, 'asset_id': 'A15', 'kind': 'insert',
                               'body': 'A15_upper_rack', 'site': site_name,
                               'approach_axis': [0, 0, 1]})
            interfaces.append({'id': rotate_id, 'asset_id': 'A15', 'kind': 'rotate',
                               'body': 'A15_upper_rack', 'site': site_name,
                               'approach_axis': [0, 0, 1],
                               'operation': {'status': 'declared', 'axis': [0, 0, 1],
                                             'range': [-math.pi, math.pi],
                                             'basis': {
                                                 'axis': {'kind': 'assumption', 'reference': 'Circular vertical tube bore; external tube rotates about its own axis.'},
                                                 'range': {'kind': 'source_specification', 'reference': 'Protocol pdf:p7 step 21 requires repeated 90-degree tube turns; no axial antirotation feature is modeled.'}
                                             }}})
            sockets.append({'number': number, 'row': row, 'site': site_name,
                            'center_local_m': [x, y, 0.054],
                            'nominal_bore_diameter_m': 2 * inner_r,
                            'minimum_polygon_bore_diameter_m': 2 * inner_r * math.cos(math.pi / segments),
                            'support_surface_z_m': 0.054,
                            'tube_support': 'External microtube upper collar on annular top; tube bottom remains unobstructed.',
                            'compatible_tube_envelope_assumption': {
                                'body_max_diameter_m': 0.011,
                                'collar_diameter_range_m': [0.0122, 0.0145],
                                'maximum_depth_below_collar_m': 0.043,
                                'maximum_tip_radius_m_at_bottom': 0.002},
                            'interaction_status': 'unverified'})

    equality = node(root, 'equality')
    node(equality, 'weld', name='A15_rack_base_weld', body1='A15_magnet_base',
         body2='A15_upper_rack', active='true', solref='0.004 1', solimp='0.98 0.995 0.001')

    manifest = {
        'schema_version': '1.0',
        'asset_ids': ['A15'],
        'parts': [
            {'id': 'A15.magnet_base', 'body': 'A15_magnet_base'},
            {'id': 'A15.removable_upper_rack', 'body': 'A15_upper_rack'}
        ],
        'interfaces': interfaces,
        'buttons': [],
        'connections': [{
            'id': 'A15.rack_base_connection', 'kind': 'detachable',
            'equality': 'A15_rack_base_weld',
            'release_event': 'A15.rack.remove', 'attach_event': 'A15.rack.replace',
            'irreversible': False,
            'capture_sites': ['A15_base_mount_site', 'A15_rack_mount_site'],
            'position_tolerance_m': 0.0015, 'angle_tolerance_rad': 0.05
        }],
        'mechanical_events': [
            {'id': 'A15.rack.remove', 'interface': 'A15.rack.mount', 'action': 'release'},
            {'id': 'A15.rack.replace', 'interface': 'A15.rack.mount', 'action': 'attach'}
        ],
        'assumptions': [
            'Source identity: DynaMag-2, Cat. No. 12321D; source selected_sections specifies sixteen numbered positions, eight per side, 1.5–2 ml microtubes and 10–2000 microlitre working volume.',
            'Reference appearance follows supplied dynamag2.png: a long gray magnetic center, ivory end housings, and a translucent detachable upper rack. No undisclosed source files are read.',
            'No manufacturer overall dimensions, bore pitch, removal force, masses or exact mating dimensions were supplied. All geometric dimensions below are engineering modeling assumptions, not manufacturer measurements.',
            'Modeled overall envelope is 0.178 x 0.0486 x 0.056 m; longitudinal tube pitch 0.018 m; row centers y=+/-0.0108 m; nominal bore 0.0116 m; support collar surface z=0.054 m.',
            'External 1.5–2 ml microtubes are assumed to have body diameter at most 0.011 m, collar diameter 0.0122–0.0145 m and depth below collar at most 0.043 m. Actual A13 mating and cap clearance remain unverified because no dependency geometry was supplied. 0.2 ml reaction tubes are not supported.',
            'The modeled magnet enclosure has half-thickness 0.0043 m. A centered 0.011 m diameter tube has 0.001 m minimum lateral clearance from its face. The two tube corridors have no bottom plate.',
            'Upper rack mass is assumed 0.055 kg; its diagonal inertia is the positive box-envelope approximation. Contact friction and solver parameters are modeling values.',
            'Initial state contains the empty upper rack installed on the magnet base. No tubes, liquid, robot, table, global floor, camera or other inventory assets are generated.',
            'Local support plane is z=0. The static base must be placed on the scene worktable by the assembler. The free rack is independently supported by two visible horizontal end shoulders at z=0.053 m.',
            'Mechanical events declare manual rack removal/replacement only. Runtime must implement event handling and enforce capture tolerances; declaration alone does not trigger an action.',
            'All operation interfaces, grasp access, tube insertion/rotation and reattachment remain unverified in this static candidate.'
        ],
        'approximations': [
            'Magnetic separation is represented solely by rack-on-base versus rack-removed operating status. No magnetic forces, tube adsorption, beads, liquids or biochemical reactions are simulated.',
            'The detachable assembly weld is a releasable assembly approximation, not a measured snap fit or removal-force model; no invented latch, button or lid is added.',
            'Each real open bore is built from 32 shared convex annular sectors. The minimum polygonal opening is recorded per socket; collision and visible socket geometry use the same sectors.',
            'Translucent molded parts, end housings and magnet encapsulation use finite-thickness rigid geometry. Cosmetic numbers and warning decals alone are noncolliding.',
            'Tube washing rotation is an external tube motion in a circular, unkeyed bore, not a fictitious rack motor or built-in rotary joint. A13 tubes remain separate objects and are not welded or magnetically attached to this asset.',
            'Source tube transfer, aspiration, washing and resuspension remain external workflow actions, enabled by open upper access and removable rack; this asset does not assert their execution.'
        ],
        'socket_geometry': sockets,
        'operating_states': [
            {'id': 'A15.magnetic_separation_position',
             'condition': 'Upper rack seated on magnet base with assembly weld enabled.',
             'representation': 'status_only', 'interaction_status': 'unverified'},
            {'id': 'A15.off_magnet_resuspension_position',
             'condition': 'Assembly weld disabled and rack physically lifted clear by an external grasp.',
             'representation': 'status_only', 'interaction_status': 'unverified'}
        ],
        'initial_state': {'rack_connection': 'assembled', 'tube_count': 0,
                          'support_plane_z_m': 0,
                          'rack_support': ['A15_left_end_housing', 'A15_right_end_housing']},
        'source_references': ['input.source.selected_sections:A15', 'input.protocol:pdf:p5',
                              'input.protocol:pdf:p7:steps19-24', 'input.images:images/dynamag2.png'],
        'backend': {'name': 'mujoco', 'target_version': '3.3.7'},
        'validation_scope': 'Static candidate; interactions not verified.'
    }
    model_path = os.path.join(output_dir, 'model.xml')
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(model_path, encoding='utf-8', xml_declaration=True)
    with open(os.path.join(output_dir, 'interfaces.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write('\n')

    # Load only the files just authored. Compilation is not a claim of
    # successful grasping, assembly, magnetic separation or workflow execution.
    import mujoco
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return {'model': 'model.xml', 'interfaces': 'interfaces.json'}
