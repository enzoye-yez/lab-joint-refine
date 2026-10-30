import os, json
from xml.sax.saxutils import escape


def box(name, size, pos=(0,0,0), rgba="0.55 0.58 0.62 1", mass=None, group=0, contype=1, conaffinity=1):
    at = f' name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" size="{size[0]/2} {size[1]/2} {size[2]/2}" rgba="{rgba}" group="{group}" contype="{contype}" conaffinity="{conaffinity}"'
    if mass is not None:
        at += f' mass="{mass}"'
    return f'<geom type="box"{at}/>'


def cyl(name, radius, height, pos=(0,0,0), rgba="0.55 0.58 0.62 1", mass=None, contype=1, conaffinity=1):
    at = f' name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" size="{radius} {height/2}" rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}"'
    if mass is not None:
        at += f' mass="{mass}"'
    return f'<geom type="cylinder"{at}/>'


def site(name, pos, size="0.012", rgba="1 0.2 0.1 1"):
    return f'<site name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" size="{size}" rgba="{rgba}"/>'


def bucket_xml(i, x, y, ang):
    # The bucket is a real open-top rectangular receptacle, suspended from the rotor.
    walls = []
    walls.append(box(f"bucket_{i}_bottom", (0.145,0.175,0.012), (0,0,-0.071), "0.28 0.31 0.35 1", 0.08))
    walls.append(box(f"bucket_{i}_leftwall", (0.012,0.175,0.145), (-0.0665,0,0), "0.28 0.31 0.35 1", 0.04))
    walls.append(box(f"bucket_{i}_rightwall", (0.012,0.175,0.145), (0.0665,0,0), "0.28 0.31 0.35 1", 0.04))
    walls.append(box(f"bucket_{i}_frontwall", (0.145,0.012,0.145), (0,-0.0815,0), "0.28 0.31 0.35 1", 0.04))
    walls.append(box(f"bucket_{i}_backwall", (0.145,0.012,0.145), (0,0.0815,0), "0.28 0.31 0.35 1", 0.04))
    # Three 50 ml adapter wells, separated so the tube openings remain open.
    wells=[]
    for j,px in enumerate((-0.043,0,0.043)):
        wells.append(cyl(f"adapter_{i}_{j}_sleeve",0.016,0.034,(px,0,-0.047),"0.72 0.74 0.77 1",0.012))
        wells.append(cyl(f"adapter_{i}_{j}_tube",0.0145,0.116,(px,0,0.010),"0.82 0.84 0.88 1",0.018))
        wells.append(box(f"adapter_{i}_{j}_retainer",(0.029,0.024,0.006),(px,0,-0.066),"0.72 0.74 0.77 1",0.006))
    cover = f'''<body name="bucket_{i}_cover" pos="0 0 0.067">
        <freejoint name="bucket_{i}_cover_free"/>
        {box(f"bucket_{i}_cover_plate",(0.145,0.175,0.010),(0,0,0),"0.32 0.35 0.39 1",0.035)}
        {box(f"bucket_{i}_cover_handle",(0.030,0.018,0.012),(0,0,0.012),"0.18 0.20 0.22 1",0.008)}
        {site(f"bucket_{i}_cover_grasp",(0,0,0.020),"0.010")}
    </body>'''
    body=f'''<body name="bucket_{i}" pos="{x} {y} 0" quat="0.7071 0 0 {0.7071 if ang==0 else (-0.7071 if ang==3.141592653589793 else 0)}">
        <joint name="bucket_{i}_swing" type="hinge" axis="0 1 0" range="-1.5708 1.5708" damping="0.08"/>
        {''.join(walls)}
        {''.join(wells)}
        {site(f"bucket_{i}_grasp",(0,0,0.085),"0.012")}
        {site(f"bucket_{i}_tube_center",(0,0,0.070),"0.009","0.1 0.8 1 1")}
        {cover}
    </body>'''
    return body


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    buckets = ''.join([
        bucket_xml(1, 0.145, 0.0, 0.0),
        bucket_xml(2, 0.0, 0.145, 1.5707963268),
        bucket_xml(3, -0.145, 0.0, 3.1415926536),
        bucket_xml(4, 0.0, -0.145, -1.5707963268)
    ])
    xml = f'''<mujoco model="A06_5810R_refrigerated_centrifuge">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint limited="true" armature="0.001"/>
    <geom friction="0.65 0.12 0.08" condim="4"/>
  </default>
  <worldbody>
    <body name="centrifuge_base" pos="0 0 0.1725">
      {box("centrifuge_lower_shell",(0.700,0.608,0.270),(0,0,-0.020),"0.56 0.59 0.63 1",90)}
      {box("centrifuge_front_panel",(0.675,0.018,0.210),(0,-0.305,0.012),"0.62 0.64 0.68 1",4)}
      {box("centrifuge_left_side",(0.018,0.560,0.230),(-0.351,0,0.010),"0.48 0.52 0.56 1",3)}
      {box("centrifuge_right_side",(0.018,0.560,0.230),(0.351,0,0.010),"0.48 0.52 0.56 1",3)}
      {box("centrifuge_top_deck",(0.690,0.590,0.040),(0,0,0.145),"0.66 0.68 0.72 1",5)}
      {box("centrifuge_front_control_bezel",(0.255,0.035,0.105),(0,-0.325,0.105),"0.18 0.22 0.28 1",0.8)}
      {box("display_panel",(0.170,0.008,0.052),(0.025,-0.346,0.126),"0.04 0.08 0.11 1",0.06,0,0,0)}
      {box("start_button",(0.026,0.012,0.012),(-0.078,-0.350,0.095),"0.05 0.35 0.85 1",0.02,0)}
      {box("stop_button",(0.026,0.012,0.012),(-0.035,-0.350,0.095),"0.85 0.10 0.08 1",0.02,0)}
      {box("unlock_button",(0.026,0.012,0.012),(0.008,-0.350,0.095),"0.15 0.45 0.85 1",0.02,0)}
      <body name="start_button_body" pos="-0.078 -0.350 0.095">
        <joint name="start_button_slide" type="slide" axis="0 1 0" range="0 0.004" damping="0.15"/>
        {box("start_button_press_geom",(0.025,0.010,0.010),(0,0,0),"0.05 0.35 0.85 1",0.02)}
        {site("start_button_site",(0,-0.010,0),"0.006")}
      </body>
      <body name="stop_button_body" pos="-0.035 -0.350 0.095">
        <joint name="stop_button_slide" type="slide" axis="0 1 0" range="0 0.004" damping="0.15"/>
        {box("stop_button_press_geom",(0.025,0.010,0.010),(0,0,0),"0.85 0.10 0.08 1",0.02)}
        {site("stop_button_site",(0,-0.010,0),"0.006")}
      </body>
      <body name="unlock_button_body" pos="0.008 -0.350 0.095">
        <joint name="unlock_button_slide" type="slide" axis="0 1 0" range="0 0.004" damping="0.15"/>
        {box("unlock_button_press_geom",(0.025,0.010,0.010),(0,0,0),"0.15 0.45 0.85 1",0.02)}
        {site("unlock_button_site",(0,-0.010,0),"0.006")}
      </body>
      <body name="main_lid" pos="0 0.285 0.165">
        <joint name="main_lid_hinge" type="hinge" axis="1 0 0" range="0 1.80" damping="0.25"/>
        {box("main_lid_outer",(0.690,0.550,0.040),(0,0.0,0.035),"0.60 0.63 0.68 1",5)}
        {box("main_lid_inner",(0.610,0.470,0.018),(0,-0.005,0.010),"0.32 0.35 0.39 1",2)}
        {cyl("lid_hinge_bar",0.012,0.56,(0,0,0),"0.18 0.20 0.22 1",0.15)}
        {site("main_lid_grasp",(0,-0.230,0.055),"0.014")}
      </body>
      <body name="rotor_carrier" pos="0 0 0.175">
        {cyl("rotor_hub",0.035,0.030,(0,0,0.015),"0.25 0.28 0.32 1",0.20)}
        {cyl("rotor_disk",0.185,0.018,(0,0,0.0),"0.42 0.45 0.50 1",0.60)}
        {cyl("rotor_center_bore",0.020,0.025,(0,0,0.015),"0.12 0.14 0.17 1",0.03)}
        {site("rotor_grasp_site",(0,0,0.040),"0.014")}
        {buckets}
      </body>
      {box("left_foot",(0.090,0.090,0.035),(-0.285,-0.235,-0.170),"0.18 0.20 0.22 1",2)}
      {box("right_foot",(0.090,0.090,0.035),(0.285,-0.235,-0.170),"0.18 0.20 0.22 1",2)}
      {box("rear_left_foot",(0.090,0.090,0.035),(-0.285,0.235,-0.170),"0.18 0.20 0.22 1",2)}
      {box("rear_right_foot",(0.090,0.090,0.035),(0.285,0.235,-0.170),"0.18 0.20 0.22 1",2)}
      {site("centrifuge_front_interface",(0,-0.335,0.08),"0.015")}
      {site("centrifuge_rotor_interface",(0,0,0.215),"0.015","0.1 0.8 1 1")}
    </body>
  </worldbody>
  <equality>
    <weld name="bucket_1_cover_latch" body1="bucket_1" body2="bucket_1_cover" solref="0.02 1" solimp="0.90 0.95 0.01"/>
    <weld name="bucket_2_cover_latch" body1="bucket_2" body2="bucket_2_cover" solref="0.02 1" solimp="0.90 0.95 0.01"/>
    <weld name="bucket_3_cover_latch" body1="bucket_3" body2="bucket_3_cover" solref="0.02 1" solimp="0.90 0.95 0.01"/>
    <weld name="bucket_4_cover_latch" body1="bucket_4" body2="bucket_4_cover" solref="0.02 1" solimp="0.90 0.95 0.01"/>
  </equality>
</mujoco>'''
    with open(os.path.join(output_dir, "model.xml"), "w", encoding="utf-8") as f:
        f.write(xml)
    data = {
      "schema_version":"1.0",
      "asset_ids":["A06"],
      "parts":[
        {"id":"centrifuge_body","body":"centrifuge_base"},
        {"id":"main_lid","body":"main_lid"},
        {"id":"rotor","body":"rotor_carrier"},
        {"id":"bucket_1","body":"bucket_1"},{"id":"bucket_2","body":"bucket_2"},
        {"id":"bucket_3","body":"bucket_3"},{"id":"bucket_4","body":"bucket_4"},
        {"id":"bucket_covers","body":"bucket_1_cover"}
      ],
      "interfaces":[
        {"id":"A06_front_grasp","asset_id":"A06","kind":"grasp","body":"centrifuge_base","site":"centrifuge_front_interface","approach_axis":[0,-1,0]},
        {"id":"A06_lid_grasp","asset_id":"A06","kind":"grasp","body":"main_lid","site":"main_lid_grasp","approach_axis":[0,-1,0],"operation":{"status":"declared","axis":[1,0,0],"joint":"main_lid_hinge","range":[0,1.80],"clearance_m":0.02,"grasp_geoms":["main_lid_outer"],"basis":{"joint":{"kind":"assumption","reference":"hinged upper cover mechanism inferred from reference image"}}}},
        {"id":"A06_rotor_grasp","asset_id":"A06","kind":"rotate","body":"rotor_carrier","site":"rotor_grasp_site","approach_axis":[0,0,1]},
        {"id":"A06_bucket_load","asset_id":"A06","kind":"insert","body":"bucket_1","site":"bucket_1_tube_center","approach_axis":[0,0,1]},
        {"id":"A06_start","asset_id":"A06","kind":"press","body":"start_button_body","site":"start_button_site","approach_axis":[0,-1,0]},
        {"id":"A06_stop","asset_id":"A06","kind":"press","body":"stop_button_body","site":"stop_button_site","approach_axis":[0,-1,0]},
        {"id":"A06_unlock","asset_id":"A06","kind":"press","body":"unlock_button_body","site":"unlock_button_site","approach_axis":[0,-1,0]}
      ],
      "assumptions":[
        "离心机外形采用资料固定尺寸700 x 608 x 345 mm；底脚将机体支承面置于局部z=0。",
        "转子中心、四个吊篮的径向位置、吊篮壁厚、铰链轴线和质量为建模假设，依据A-4-62示意图布置。",
        "50 ml管按直径不超过30 mm建模；管体有效高度采用0.116 m，桶盖限位与管高的0.20 mm名义差异未宣称已验证。",
        "前面板按钮的尺寸、行程和显示屏为外观及交互建模值，不代表厂商测量。",
        "桶盖按可拆卸防气溶胶盖建模，初始通过weld等式装配；解锁动作的运行时验证未完成。"
      ],
      "approximations":[
        "旋转转子采用静态转子盘与四个独立摆动吊篮刚体近似，未模拟离心力或高速动力学。",
        "适配件采用三只独立圆筒套和底部定位件表示三个50 ml管位，保留管位间空隙。",
        "主盖采用单铰链刚体近似，未模拟锁销、阻尼器和联锁电气逻辑。"
      ],
      "buttons":[
        {"id":"A06_start_button","joint":"start_button_slide","geom":"start_button_press_geom","pressed_at":0.002,"released_at":0.0005,"event":"A06_start_event"},
        {"id":"A06_stop_button","joint":"stop_button_slide","geom":"stop_button_press_geom","pressed_at":0.002,"released_at":0.0005,"event":"A06_stop_event"},
        {"id":"A06_unlock_button","joint":"unlock_button_slide","geom":"unlock_button_press_geom","pressed_at":0.002,"released_at":0.0005,"event":"A06_unlock_event"}
      ],
      "connections":[
        {"id":"A06_bucket1_cover","kind":"detachable","equality":"bucket_1_cover_latch","release_event":"A06_unlock_event","irreversible":false,"attach_event":"A06_unlock_event","capture_sites":["bucket_1_cover_grasp","bucket_1_grasp"],"position_tolerance_m":0.005,"angle_tolerance_rad":0.2},
        {"id":"A06_bucket2_cover","kind":"detachable","equality":"bucket_2_cover_latch","release_event":"A06_unlock_event","irreversible":false,"attach_event":"A06_unlock_event","capture_sites":["bucket_2_cover_grasp","bucket_2_grasp"],"position_tolerance_m":0.005,"angle_tolerance_rad":0.2},
        {"id":"A06_bucket3_cover","kind":"detachable","equality":"bucket_3_cover_latch","release_event":"A06_unlock_event","irreversible":false,"attach_event":"A06_unlock_event","capture_sites":["bucket_3_cover_grasp","bucket_3_grasp"],"position_tolerance_m":0.005,"angle_tolerance_rad":0.2},
        {"id":"A06_bucket4_cover","kind":"detachable","equality":"bucket_4_cover_latch","release_event":"A06_unlock_event","irreversible":false,"attach_event":"A06_unlock_event","capture_sites":["bucket_4_cover_grasp","bucket_4_grasp"],"position_tolerance_m":0.005,"angle_tolerance_rad":0.2}
      ],
      "mechanical_events":[
        {"id":"A06_start_event","interface":"A06_start","action":"attach"},
        {"id":"A06_stop_event","interface":"A06_stop","action":"release"},
        {"id":"A06_unlock_event","interface":"A06_unlock","action":"release"}
      ]
    }
    with open(os.path.join(output_dir, "interfaces.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
