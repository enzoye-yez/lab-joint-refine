import os, json, math


def box(name, pos, size, rgba, mass=None, contype=1, conaffinity=1):
    m = '' if mass is None else f' mass="{mass}"'
    inert = '' if mass is None else f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{mass*size[1]*size[1]/12:.8g} {mass*size[0]*size[0]/12:.8g} {mass*(size[0]*size[0]+size[1]*size[1])/12:.8g}"/>'
    return f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">{inert}<geom name="{name}_geom" type="box" size="{size[0]/2} {size[1]/2} {size[2]/2}" rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}"{m}/></body>'


def cyl(name, pos, radius, height, rgba, mass=None, axis='z', contype=1, conaffinity=1):
    m = '' if mass is None else f' mass="{mass}"'
    inert = '' if mass is None else f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{mass*(3*radius*radius+height*height)/12:.8g} {mass*(3*radius*radius+height*height)/12:.8g} {mass*radius*radius/2:.8g}"/>'
    quat = ''
    if axis == 'x': quat = ' quat="0.7071068 0 0.7071068"'
    if axis == 'y': quat = ' quat="0.7071068 0.7071068 0 0"'
    return f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}"{quat}>{inert}<geom name="{name}_geom" type="cylinder" size="{radius} {height/2}" rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}"{m}/></body>'


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    xml = []
    xml.append('''<mujoco model="A14_single_channel_pipette_set">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint damping="0.04" armature="0.0001"/>
    <geom friction="0.7 0.1 0.1" density="650"/>
  </default>
  <worldbody>''')
    xml.append(box('support_base', (0,0,0.012), (0.48,0.30,0.024), '0.18 0.20 0.23 1', 2.0))
    xml.append(box('pipette_cradle', (0,0,0.040), (0.44,0.24,0.032), '0.35 0.38 0.42 1', 0.8))

    ranges = [('p025','0.1_2.5',0.00070), ('p010','0.5_10',0.00078), ('p020','2_20',0.00086), ('p200','20_200',0.00100), ('p1000','100_1000',0.00115)]
    xs = [-0.17,-0.085,0,0.085,0.17]
    for i, (tag, label, nozzle_r) in enumerate(ranges):
        x = xs[i]
        root = f'''<body name="pipette_{tag}" pos="{x} 0 0.072">
          <freejoint name="{tag}_freejoint"/>
          <inertial pos="0 0 0.10" mass="0.105" diaginertia="0.00025 0.00025 0.00003"/>
          <geom name="{tag}_handle" type="capsule" fromto="0 0 0.025 0 0 0.185" size="0.017" rgba="0.93 0.94 0.91 1"/>
          <geom name="{tag}_head" type="box" pos="0 0 0.185" size="0.032 0.026 0.032" rgba="0.80 0.82 0.80 1"/>
          <geom name="{tag}_finger_rest" type="box" pos="-0.033 0 0.174" size="0.022 0.026 0.008" rgba="0.86 0.87 0.84 1"/>
          <site name="{tag}_grasp_site" pos="0 0 0.145" size="0.004" rgba="0.1 0.8 0.2 1"/>
          <site name="{tag}_tip_interface" pos="0 0 0.018" size="0.003" rgba="0.9 0.2 0.1 1"/>
          <body name="{tag}_knob" pos="0 0 0.227">
            <joint name="{tag}_knob_rotate" type="hinge" axis="0 0 1" range="-3.14 3.14" limited="true"/>
            <inertial pos="0 0 0" mass="0.012" diaginertia="0.000002 0.000002 0.000002"/>
            <geom name="{tag}_knob_geom" type="cylinder" size="0.020 0.014" rgba="0.74 0.76 0.73 1"/>
            <geom name="{tag}_knob_grip" type="cylinder" pos="0 0 0.014" size="0.016 0.004" rgba="0.62 0.64 0.62 1"/>
          </body>
          <body name="{tag}_stage1" pos="0 0 0.212">
            <joint name="{tag}_stage1_slide" type="slide" axis="0 0 1" range="0 0.006" limited="true"/>
            <inertial pos="0 0 0" mass="0.006" diaginertia="0.000001 0.000001 0.000001"/>
            <geom name="{tag}_stage1_button" type="cylinder" size="0.010 0.008" rgba="0.18 0.20 0.20 1"/>
            <body name="{tag}_stage2" pos="0 0 0.008">
              <joint name="{tag}_stage2_slide" type="slide" axis="0 0 1" range="0 0.006" limited="true"/>
              <inertial pos="0 0 0" mass="0.004" diaginertia="0.0000006 0.0000006 0.0000006"/>
              <geom name="{tag}_stage2_button" type="cylinder" size="0.008 0.006" rgba="0.12 0.14 0.14 1"/>
            </body>
          </body>
          <body name="{tag}_collar" pos="0 0 0.026">
            <geom name="{tag}_collar_geom" type="cylinder" size="0.011 0.008" rgba="0.70 0.72 0.69 1"/>
          </body>
          <body name="{tag}_tip" pos="0 0 -0.045">
            <freejoint name="{tag}_tip_freejoint"/>
            <inertial pos="0 0 -0.045" mass="0.0008" diaginertia="0.00000003 0.00000003 0.000000001"/>
            <geom name="{tag}_tip_upper" type="cylinder" size="{nozzle_r*1.7:.7g} 0.010" rgba="0.86 0.88 0.86 1"/>
            <geom name="{tag}_tip_taper" type="cone" pos="0 0 -0.030" size="{nozzle_r*1.7:.7g} 0.00018 0.060" rgba="0.91 0.92 0.89 1"/>
            <site name="{tag}_tip_site" pos="0 0 -0.061" size="0.0008"/>
          </body>
        </body>'''
        xml.append(root)

    # 96-position refillable tip box, with open cavity and separate suspended tips.
    xml.append('''<body name="tip_box" pos="0.0 0.165 0.045">
      <freejoint name="tip_box_freejoint"/>
      <inertial pos="0 0 0.025" mass="0.34" diaginertia="0.001 0.001 0.001"/>
      <geom name="tip_box_floor" type="box" pos="0 0 0.010" size="0.075 0.060 0.006" rgba="0.88 0.89 0.87 1"/>
      <geom name="tip_box_front_wall" type="box" pos="0 -0.064 0.035" size="0.082 0.006 0.035" rgba="0.88 0.89 0.87 1"/>
      <geom name="tip_box_back_wall" type="box" pos="0 0.064 0.035" size="0.082 0.006 0.035" rgba="0.88 0.89 0.87 1"/>
      <geom name="tip_box_left_wall" type="box" pos="-0.082 0 0.035" size="0.006 0.070 0.035" rgba="0.88 0.89 0.87 1"/>
      <geom name="tip_box_right_wall" type="box" pos="0.082 0 0.035" size="0.006 0.070 0.035" rgba="0.88 0.89 0.87 1"/>
      <geom name="tip_grid_plate" type="box" pos="0 0 0.039" size="0.074 0.058 0.003" rgba="0.78 0.80 0.78 1" contype="0" conaffinity="0"/>
      <site name="tip_box_grasp_site" pos="0 0 0.082" size="0.006"/>
      <body name="tip_box_lid" pos="0 0.070 0.075">
        <joint name="tip_box_lid_hinge" type="hinge" axis="1 0 0" range="0 2.5" limited="true"/>
        <inertial pos="0 -0.035 0.003" mass="0.06" diaginertia="0.00008 0.00008 0.00012"/>
        <geom name="tip_box_lid_panel" type="box" pos="0 -0.035 0.003" size="0.085 0.070 0.003" rgba="0.25 0.45 0.78 0.32" contype="1" conaffinity="1"/>
        <site name="tip_box_lid_grasp_site" pos="0 -0.035 0.009" size="0.005"/>
      </body>
    </body>''')

    # Tip inserts: 96 separate, reusable-consumable objects. Their narrow bodies remain visibly supported by the grid.
    for r in range(8):
        for c in range(12):
            x = -0.066 + c*0.012
            y = 0.165 - 0.055 + r*0.014
            n = f'box_tip_{r+1:02d}_{c+1:02d}'
            xml.append(f'''<body name="{n}" pos="{x:.5f} {y:.5f} 0.018">
              <freejoint name="{n}_freejoint"/>
              <inertial pos="0 0 0.025" mass="0.00045" diaginertia="0.00000002 0.00000002 0.000000001"/>
              <geom name="{n}_shaft" type="cylinder" size="0.0032 0.018" rgba="0.92 0.94 0.91 1"/>
              <geom name="{n}_point" type="cone" pos="0 0 -0.025" size="0.0032 0.0003 0.032" rgba="0.96 0.96 0.93 1"/>
              <site name="{n}_grasp_site" pos="0 0 0.015" size="0.001"/>
            </body>''')

    xml.append('</worldbody><equality>')
    for tag, _, _ in ranges:
        xml.append(f'<weld name="{tag}_tip_detachable_weld" body1="pipette_{tag}" body2="{tag}_tip" solref="0.01 1" solimp="0.9 0.95 0.01"/>')
    xml.append('</equality></mujoco>')
    with open(os.path.join(output_dir, 'model.xml'), 'w', encoding='utf-8') as f:
        f.write(''.join(xml))

    parts = [{'id':'pipette_0.1_2.5','body':'pipette_p025'}, {'id':'pipette_0.5_10','body':'pipette_p010'}, {'id':'pipette_2_20','body':'pipette_p020'}, {'id':'pipette_20_200','body':'pipette_p200'}, {'id':'pipette_100_1000','body':'pipette_p1000'}, {'id':'tip_box_96_position','body':'tip_box'}]
    interfaces = []
    buttons = []
    for tag, label, _ in ranges:
        interfaces += [
            {'id':f'{tag}_grasp','asset_id':'A14','kind':'grasp','body':f'pipette_{tag}','site':f'{tag}_grasp_site','approach_axis':[0,0,1]},
            {'id':f'{tag}_tip_attach','asset_id':'A14','kind':'insert','body':f'pipette_{tag}','site':f'{tag}_tip_interface','approach_axis':[0,0,-1]},
            {'id':f'{tag}_knob_rotate','asset_id':'A14','kind':'rotate','body':f'pipette_{tag}','site':f'{tag}_grasp_site','operation':{'status':'declared','axis':[0,0,1],'joint':f'{tag}_knob_rotate','range':[-3.14159,3.14159],'basis':{'joint':{'kind':'assumption','reference':'旋钮转动机构的建模自由度'}}}},
            {'id':f'{tag}_release','asset_id':'A14','kind':'press','body':f'pipette_{tag}','site':f'{tag}_grasp_site','approach_axis':[0,0,1]},
            {'id':f'{tag}_eject','asset_id':'A14','kind':'press','body':f'pipette_{tag}','site':f'{tag}_grasp_site','approach_axis':[0,0,1]}]
        buttons += [
            {'id':f'{tag}_first_stop','joint':f'{tag}_stage1_slide','geom':f'{tag}_stage1_button','pressed_at':0.006,'released_at':0.0005,'event':f'{tag}_first_stop_event'},
            {'id':f'{tag}_second_stop','joint':f'{tag}_stage2_slide','geom':f'{tag}_stage2_button','pressed_at':0.006,'released_at':0.0005,'event':f'{tag}_second_stop_event'}]
    interfaces += [{'id':'tip_box_grasp','asset_id':'A14','kind':'grasp','body':'tip_box','site':'tip_box_grasp_site','approach_axis':[0,0,1]}, {'id':'tip_box_lid_open','asset_id':'A14','kind':'rotate','body':'tip_box_lid','site':'tip_box_lid_grasp_site','approach_axis':[0,-1,0]}]
    data = {'schema_version':'1.0','asset_ids':['A14'],'parts':parts,'interfaces':interfaces,'assumptions':['五支移液器按资料给定量程建模；外形、孔距、壁厚和内部弹簧尺寸为建模假设。','吸头盒按96孔、8行12列建模，孔距和开口形状为建模假设。','每支移液器初始带一支独立吸头；盒内另有96支未使用吸头，作为独立可拆对象。'],'approximations':['旋钮、一级/二级按压和弹射用转动关节及两级滑动关节刚体近似。','吸头锥体采用圆柱加圆锥分段刚体近似，未模拟柔性薄壁。'],'buttons':buttons,'connections':[{'id':f'{tag}_tip_connection','kind':'detachable','equality':f'{tag}_tip_detachable_weld','release_event':f'{tag}_eject','irreversible':False} for tag,_,_ in ranges]}
    with open(os.path.join(output_dir, 'interfaces.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
