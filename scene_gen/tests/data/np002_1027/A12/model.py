import os, json


def box(name, pos, size, mat, contype=1, conaffinity=1, mass=None):
    attrs = 'name="%s" pos="%g %g %g" size="%g %g %g" material="%s" contype="%d" conaffinity="%d"' % (
        name, pos[0], pos[1], pos[2], size[0]/2, size[1]/2, size[2]/2, mat, contype, conaffinity)
    if mass is not None:
        attrs += ' mass="%g"' % mass
    return '<geom type="box" %s/>' % attrs


def cyl(name, pos, radius, halfheight, mat, axis='z', contype=1, conaffinity=1, mass=None):
    attrs = 'name="%s" pos="%g %g %g" size="%g %g" material="%s" contype="%d" conaffinity="%d"' % (
        name, pos[0], pos[1], pos[2], radius, halfheight, mat, contype, conaffinity)
    if axis != 'z':
        if axis == 'x':
            attrs += ' euler="0 1.57079632679 0"'
        elif axis == 'y':
            attrs += ' euler="1.57079632679 0 0"'
    if mass is not None:
        attrs += ' mass="%g"' % mass
    return '<geom type="cylinder" %s/>' % attrs


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    xml = r'''<mujoco model="A12_TSX505GA_refrigerator">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <size njmax="200" nconmax="200"/>
  <asset>
    <material name="cabinet_white" rgba="0.82 0.84 0.82 1"/>
    <material name="cabinet_dark" rgba="0.10 0.13 0.14 1"/>
    <material name="metal" rgba="0.65 0.68 0.67 1"/>
    <material name="shelf_wire" rgba="0.72 0.74 0.72 1"/>
    <material name="glass" rgba="0.20 0.30 0.31 0.34"/>
    <material name="blue_handle" rgba="0.02 0.30 0.55 1"/>
    <material name="display_black" rgba="0.015 0.018 0.02 1"/>
    <material name="display_green" rgba="0.08 0.85 0.60 1"/>
    <material name="tube_clear" rgba="0.75 0.88 0.88 0.55"/>
    <material name="tube_cap" rgba="0.12 0.16 0.17 1"/>
    <material name="rack" rgba="0.28 0.31 0.30 1"/>
  </asset>
  <worldbody>
    <body name="refrigerator_cabinet" pos="0 0 0.4065">
      <!-- closed external shell; the front remains an actual opening -->
      <geom name="cabinet_bottom" type="box" pos="0 0 0.025" size="0.30 0.3305 0.025" material="cabinet_dark"/>
      <geom name="cabinet_left_wall" type="box" pos="-0.285 0 0.4065" size="0.015 0.3305 0.3815" material="cabinet_white"/>
      <geom name="cabinet_right_wall" type="box" pos="0.285 0 0.4065" size="0.015 0.3305 0.3815" material="cabinet_white"/>
      <geom name="cabinet_back" type="box" pos="0 0.315 0.4065" size="0.285 0.015 0.3815" material="cabinet_dark"/>
      <geom name="cabinet_top" type="box" pos="0 0 0.788" size="0.30 0.3305 0.025" material="cabinet_white"/>
      <geom name="cabinet_top_front" type="box" pos="0 -0.316 0.748" size="0.285 0.014 0.065" material="cabinet_white"/>
      <geom name="cabinet_left_front_frame" type="box" pos="-0.270 -0.320 0.405" size="0.030 0.020 0.350" material="cabinet_dark"/>
      <geom name="cabinet_right_front_frame" type="box" pos="0.270 -0.320 0.405" size="0.030 0.020 0.350" material="cabinet_dark"/>
      <geom name="cabinet_lower_front_frame" type="box" pos="0 -0.320 0.065" size="0.270 0.020 0.055" material="cabinet_dark"/>
      <geom name="cabinet_upper_front_frame" type="box" pos="0 -0.320 0.735" size="0.270 0.020 0.040" material="cabinet_dark"/>

      <!-- three removable-looking but mechanically supported shelves -->
      <body name="shelf_lower" pos="0 0 0.225">
        <geom name="shelf_lower_panel" type="box" pos="0 0 0" size="0.255 0.270 0.008" material="shelf_wire"/>
        <geom name="shelf_lower_front_rail" type="box" pos="0 -0.255 0.018" size="0.255 0.008 0.018" material="metal"/>
      </body>
      <body name="shelf_middle" pos="0 0 0.430">
        <geom name="shelf_middle_panel" type="box" pos="0 0 0" size="0.255 0.270 0.008" material="shelf_wire"/>
        <geom name="shelf_middle_front_rail" type="box" pos="0 -0.255 0.018" size="0.255 0.008 0.018" material="metal"/>
      </body>
      <body name="shelf_upper" pos="0 0 0.625">
        <geom name="shelf_upper_panel" type="box" pos="0 0 0" size="0.255 0.270 0.008" material="shelf_wire"/>
        <geom name="shelf_upper_front_rail" type="box" pos="0 -0.255 0.018" size="0.255 0.008 0.018" material="metal"/>
      </body>

      <!-- stable internal 2 ml tube rack, visibly supported by the lower shelf -->
      <body name="tube_rack_2ml" pos="0.105 0.045 0.255">
        <geom name="tube_rack_base" type="box" pos="0 0 0.018" size="0.105 0.105 0.018" material="rack"/>
        <geom name="tube_rack_left_rail" type="box" pos="-0.095 0 0.060" size="0.010 0.105 0.045" material="rack"/>
        <geom name="tube_rack_right_rail" type="box" pos="0.095 0 0.060" size="0.010 0.105 0.045" material="rack"/>
        <geom name="tube_rack_front_rail" type="box" pos="0 -0.095 0.060" size="0.105 0.010 0.045" material="rack"/>
        <geom name="tube_rack_back_rail" type="box" pos="0 0.095 0.060" size="0.105 0.010 0.045" material="rack"/>
        <geom name="tube_rack_crossbar" type="box" pos="0 0 0.083" size="0.105 0.105 0.008" material="rack" contype="0" conaffinity="0"/>
        <site name="tube_rack_grasp_site" pos="0 -0.125 0.075" size="0.012" rgba="0.1 0.8 0.2 1"/>
        <body name="collection_tube_2ml" pos="0 0 0.102">
          <geom name="collection_tube_body" type="cylinder" pos="0 0 0.050" size="0.009 0.050" material="tube_clear"/>
          <geom name="collection_tube_cap" type="cylinder" pos="0 0 0.110" size="0.0105 0.010" material="tube_cap"/>
          <site name="collection_tube_grasp_site" pos="0 -0.012 0.080" size="0.006" rgba="0.2 0.9 0.2 1"/>
        </body>
      </body>

      <!-- fixed reference joint used to lock the hinged door in its initial closed state -->
      <body name="door_lock_reference" pos="-0.270 -0.322 0.405">
        <joint name="door_lock_ref" type="hinge" axis="0 0 1" range="0 0" limited="true" damping="1"/>
        <geom name="door_lock_reference_geom" type="box" pos="0 0 0" size="0.002 0.002 0.002" material="cabinet_dark" contype="0" conaffinity="0"/>
      </body>

      <!-- independently articulated glass door; initial angle is zero (closed) -->
      <body name="refrigerator_door" pos="-0.270 -0.322 0.405">
        <joint name="door_hinge" type="hinge" axis="0 0 1" range="0 1.75" limited="true" damping="2" armature="0.002"/>
        <geom name="door_outer_panel" type="box" pos="0.270 -0.022 0" size="0.270 0.022 0.365" material="cabinet_dark"/>
        <geom name="door_glass_panel" type="box" pos="0.270 -0.048 0" size="0.240 0.006 0.300" material="glass" contype="1" conaffinity="1"/>
        <geom name="door_left_frame" type="box" pos="0.030 -0.050 0" size="0.014 0.012 0.350" material="cabinet_white"/>
        <geom name="door_right_frame" type="box" pos="0.510 -0.050 0" size="0.014 0.012 0.350" material="cabinet_white"/>
        <geom name="door_top_frame" type="box" pos="0.270 -0.050 0.335" size="0.240 0.012 0.015" material="cabinet_white"/>
        <geom name="door_bottom_frame" type="box" pos="0.270 -0.050 -0.335" size="0.240 0.012 0.015" material="cabinet_white"/>
        <geom name="door_handle" type="box" pos="0.535 -0.080 0.205" size="0.018 0.018 0.115" material="blue_handle"/>
        <geom name="door_handle_mount_upper" type="box" pos="0.510 -0.060 0.300" size="0.025 0.012 0.012" material="blue_handle"/>
        <geom name="door_handle_mount_lower" type="box" pos="0.510 -0.060 0.110" size="0.025 0.012 0.012" material="blue_handle"/>
        <site name="door_handle_grasp_site" pos="0.555 -0.082 0.205" size="0.018" rgba="0.1 0.7 0.9 1"/>
      </body>

      <!-- control fascia and display details -->
      <geom name="control_fascia" type="box" pos="-0.040 -0.334 0.760" size="0.085 0.008 0.030" material="display_black"/>
      <geom name="temperature_display" type="box" pos="-0.040 -0.343 0.760" size="0.030 0.003 0.016" material="display_green" contype="0" conaffinity="0"/>
      <site name="door_operation_site" pos="0.285 -0.405 0.205" size="0.015" rgba="0.9 0.3 0.1 1"/>
    </body>
  </worldbody>
  <equality>
    <joint name="door_closed_latch" joint1="door_hinge" joint2="door_lock_ref" polycoef="0 1 0 0 0" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </equality>
</mujoco>
'''
    with open(os.path.join(output_dir, 'model.xml'), 'w', encoding='utf-8') as f:
        f.write(xml)

    interfaces = {
        "schema_version": "1.0",
        "asset_ids": ["A12"],
        "parts": [
            {"id": "refrigerator_body", "body": "refrigerator_cabinet"},
            {"id": "hinged_glass_door", "body": "refrigerator_door"},
            {"id": "internal_2ml_tube_rack", "body": "tube_rack_2ml"},
            {"id": "2ml_collection_tube", "body": "collection_tube_2ml"}
        ],
        "interfaces": [
            {"id": "A12_door_grasp", "asset_id": "A12", "kind": "rotate", "body": "refrigerator_door", "site": "door_handle_grasp_site", "approach_axis": [0, -1, 0], "operation": {"status": "declared", "axis": [0, 0, 1], "joint": "door_hinge", "range": [0, 1.75], "clearance_m": 0.02, "grasp_geoms": ["door_handle"], "basis": {"joint": {"kind": "assumption", "reference": "hinge axis chosen from front-door construction"}, "range": {"kind": "assumption", "reference": "opening clearance for undercounter door"}}}},
            {"id": "A12_tube_grasp", "asset_id": "A12", "kind": "grasp", "body": "collection_tube_2ml", "site": "collection_tube_grasp_site", "approach_axis": [0, -1, 0]},
            {"id": "A12_tube_insert", "asset_id": "A12", "kind": "insert", "body": "collection_tube_2ml", "site": "collection_tube_grasp_site", "approach_axis": [0, 0, -1]},
            {"id": "A12_internal_mount", "asset_id": "A12", "kind": "mount", "body": "tube_rack_2ml", "site": "tube_rack_grasp_site"}
        ],
        "assumptions": [
            "TSX505GA outer dimensions are taken directly from the supplied specification: 0.600 x 0.661 x 0.813 m; handle depth is represented by the front door geometry.",
            "Interior shelf elevations and the 2 ml rack hole layout are modeling assumptions because exact shelf and rack drawings were not supplied.",
            "The refrigerator is modeled at its local origin with its lower cabinet surface at z=0; the scene table provides the external support.",
            "Door hinge axis and opening limit are modeling values selected to preserve front access and avoid cabinet collision.",
            "The tube is visually distinguished from the rack and remains capped during the initial state."
        ],
        "approximations": [
            "Wire shelves are represented by thin collision-bearing panels and rails rather than individual wire rods.",
            "The refrigerator compressor, seals, sensor, refrigeration circuit and temperature dynamics are not simulated.",
            "The 2 ml tube rack is a rigid support approximation with a single representative capped tube.",
            "Door latch operation is represented by an equality lock on the hinge; runtime operation is not verified in this static delivery."
        ],
        "buttons": [],
        "connections": [
            {"id": "A12_door_latch", "kind": "latch", "equality": "door_closed_latch", "release_event": "A12_door_grasp", "irreversible": False}
        ],
        "mechanical_events": [],
        "construction_records": []
    }
    with open(os.path.join(output_dir, 'interfaces.json'), 'w', encoding='utf-8') as f:
        json.dump(interfaces, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    build('.')