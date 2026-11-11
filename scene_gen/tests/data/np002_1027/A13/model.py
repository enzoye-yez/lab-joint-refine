import os
import json
import math
import xml.etree.ElementTree as ET


def _mesh_obj(path, height, r_outer, r_inner, bottom_r, n=32):
    # Hollow tapered tube with a closed bottom and open top.
    rings = [
        (0.000, bottom_r),
        (0.004, r_outer * 0.82),
        (0.012, r_outer),
        (height - 0.004, r_outer),
        (height, r_outer * 0.92),
    ]
    verts = []
    for z, r in rings:
        for i in range(n):
            a = 2.0 * math.pi * i / n
            verts.append((r * math.cos(a), r * math.sin(a), z))
    inner_z0 = 0.005
    inner_z1 = height - 0.003
    inner0 = len(verts)
    for z, r in ((inner_z0, r_inner * 0.92), (inner_z1, r_inner)):
        for i in range(n):
            a = 2.0 * math.pi * i / n
            verts.append((r * math.cos(a), r * math.sin(a), z))
    faces = []
    def quad(a, b, c, d):
        faces.append((a, b, c, d))
    # outer wall
    for j in range(len(rings) - 1):
        a0, b0 = j * n, (j + 1) * n
        for i in range(n):
            k = (i + 1) % n
            quad(a0 + i, a0 + k, b0 + k, b0 + i)
    # bottom underside
    for i in range(n):
        k = (i + 1) % n
        faces.append((i, k, 0))
    # inner wall, reversed
    for i in range(n):
        k = (i + 1) % n
        quad(inner0 + i, inner0 + n + i, inner0 + n + k, inner0 + k)
    # annular rim
    outer_top = (len(rings) - 1) * n
    inner_top = inner0 + n
    for i in range(n):
        k = (i + 1) % n
        quad(outer_top + i, outer_top + k, inner_top + k, inner_top + i)
    # inside bottom closure
    inner_bottom = inner0
    for i in range(n):
        k = (i + 1) % n
        faces.append((inner_bottom + i, inner_bottom + k, 0))
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write("v %.7f %.7f %.7f\n" % v)
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def _add_geom(parent, name, gtype, **attrs):
    attrs = {k: str(v) for k, v in attrs.items()}
    attrs["name"] = name
    return ET.SubElement(parent, "geom", {"type": gtype, **attrs})


def build(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    _mesh_obj(os.path.join(mesh_dir, "tube_15mm.obj"), 0.038, 0.0055, 0.00425, 0.0030)
    _mesh_obj(os.path.join(mesh_dir, "tube_20mm.obj"), 0.045, 0.0060, 0.00475, 0.0032)

    root = ET.Element("mujoco", {"model": "A13_microtube_set"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81", "integrator": "implicitfast"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "mesh_tube15", "file": "meshes/tube_15mm.obj"})
    ET.SubElement(asset, "mesh", {"name": "mesh_tube20", "file": "meshes/tube_20mm.obj"})
    ET.SubElement(asset, "material", {"name": "clear_polypropylene", "rgba": "0.82 0.90 0.92 0.34", "reflectance": "0.15"})
    ET.SubElement(asset, "material", {"name": "clear_cap", "rgba": "0.88 0.93 0.94 0.48", "reflectance": "0.12"})
    ET.SubElement(asset, "material", {"name": "sample_liquid", "rgba": "0.12 0.38 0.72 0.48"})
    ET.SubElement(asset, "material", {"name": "wash_liquid", "rgba": "0.35 0.72 0.88 0.42"})
    ET.SubElement(asset, "material", {"name": "eluate_liquid", "rgba": "0.62 0.20 0.72 0.48"})
    ET.SubElement(asset, "material", {"name": "waste_liquid", "rgba": "0.34 0.34 0.34 0.40"})

    world = ET.SubElement(root, "worldbody")
    group = ET.SubElement(world, "body", {"name": "A13_microtube_set", "pos": "0 0 0"})

    # Three 2 ml collection tubes and nine 1.5 ml process tubes.
    specs = []
    for i in range(3):
        specs.append(("tube2_%02d" % (i + 1), "2ml_collection_%02d" % (i + 1), 2, "sample"))
    categories = ["extract_a", "extract_b", "combined_product", "library_round1", "library_round2", "library_round3", "screen_keep_supernatant", "screen_keep_beads", "final_eluate"]
    for i, cat in enumerate(categories):
        specs.append(("tube15_%02d" % (i + 1), "process_%s" % cat, 1.5, "eluate" if cat in ("combined_product", "final_eluate") else ("waste" if cat.startswith("screen") else "sample")))
    # Additional matching tubes used as explicit balance/centrifugation-stage tubes.
    for i in range(6):
        specs.append(("balance15_%02d" % (i + 1), "balance_stage_%02d" % (i + 1), 1.5, "waste"))

    cols = 6
    for idx, (short, semantic, capacity, liquid_kind) in enumerate(specs):
        row, col = divmod(idx, cols)
        x = -0.050 + col * 0.021
        y = -0.045 + row * 0.025
        h = 0.045 if capacity == 2 else 0.038
        mesh = "mesh_tube20" if capacity == 2 else "mesh_tube15"
        body = ET.SubElement(group, "body", {"name": "body_" + short, "pos": "%.5f %.5f 0" % (x, y)})
        _add_geom(body, "geom_" + short + "_shell", "mesh", mesh=mesh, material="clear_polypropylene", density="920", contype="1", conaffinity="1")
        # Visible liquid state remains inside the open tube and is non-colliding.
        liquid_mat = {"sample": "sample_liquid", "eluate": "eluate_liquid", "waste": "waste_liquid"}[liquid_kind]
        fill = 0.014 if capacity == 2 else 0.011
        _add_geom(body, "geom_" + short + "_liquid", "cylinder", pos="0 0 %.5f" % (0.006 + fill / 2.0), size="%.5f %.5f" % ((0.00435 if capacity == 2 else 0.00385), fill / 2.0), material=liquid_mat, contype="0", conaffinity="0", group="1")
        # Integral flip cap: the cap is a separate moving body connected by a real hinge.
        cap = ET.SubElement(body, "body", {"name": "body_" + short + "_flipcap", "pos": "0.0048 0 %.5f" % h})
        ET.SubElement(cap, "joint", {"name": "joint_" + short + "_lid", "type": "hinge", "axis": "0 1 0", "range": "0 1.95", "limited": "true", "damping": "0.0008", "armature": "0.00001"})
        _add_geom(cap, "geom_" + short + "_cap", "box", pos="0.006 0 0.0012", size="0.0085 0.0068 0.0012", material="clear_cap", density="780", contype="1", conaffinity="1")
        # Interfaces are local to each tube and remain usable after the tube is placed in a scene.
        ET.SubElement(body, "site", {"name": "site_" + short + "_grasp", "pos": "0 0 0.020", "size": "0.0025", "rgba": "0.1 0.8 0.2 0.8"})
        ET.SubElement(body, "site", {"name": "site_" + short + "_pipette", "pos": "0 0 %.5f" % (h - 0.003), "size": "0.002", "rgba": "0.2 0.5 1 0.8"})
        ET.SubElement(cap, "site", {"name": "site_" + short + "_cap_grasp", "pos": "0.006 0 0.004", "size": "0.002", "rgba": "1 0.7 0.1 0.8"})

    equality = ET.SubElement(root, "equality")
    # No weld/equality is used: each tube remains an independently movable component.
    xml_path = os.path.join(output_dir, "model.xml")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)

    parts = []
    interfaces = []
    for short, semantic, capacity, liquid_kind in specs:
        body_name = "body_" + short
        parts.append({"id": semantic, "body": body_name})
        interfaces.append({"id": semantic + "_grasp", "asset_id": "A13", "kind": "grasp", "body": body_name, "site": "site_" + short + "_grasp", "approach_axis": [0, 0, 1]})
        interfaces.append({"id": semantic + "_pipette", "asset_id": "A13", "kind": "insert", "body": body_name, "site": "site_" + short + "_pipette", "approach_axis": [0, 0, 1]})
        interfaces.append({"id": semantic + "_lid", "asset_id": "A13", "kind": "rotate", "body": body_name + "_flipcap", "site": "site_" + short + "_cap_grasp", "approach_axis": [0, 0, 1], "operation": {"status": "declared", "axis": [0, 1, 0], "joint": "joint_" + short + "_lid", "range": [0.0, 1.95], "clearance_m": 0.002, "grasp_geoms": ["geom_" + short + "_cap"], "basis": {"axis": {"kind": "assumption", "reference": "flip-cap hinge axis inferred from supplied Axygen reference image"}, "range": {"kind": "assumption", "reference": "modeling clearance for cap opening"}}}})

    data = {
        "schema_version": "1.0",
        "asset_ids": ["A13"],
        "parts": parts,
        "interfaces": interfaces,
        "assumptions": [
            "Tube lengths, diameters, cone transitions, wall thicknesses and hinge coordinates are modeling values because the supplied catalog images do not provide numeric dimensions.",
            "The asset is a single reusable set containing three 2 ml collection tubes, fifteen 1.5 ml tubes including six balance-stage tubes, and no external rack.",
            "Tube bottoms are modeled as supported directly by the receiving table at local z=0; the tubes are not welded to one another."
        ],
        "approximations": [
            "Clear polypropylene is represented by translucent mesh and simple translucent liquid volumes.",
            "The one-piece flip cap is represented as a rigid cap on a hinge joint; cap latch friction and snap engagement are not simulated.",
            "Liquid colors encode retained sample, eluate and waste identity states; no biochemical reaction is simulated."
        ],
        "buttons": [],
        "connections": [],
        "mechanical_events": [],
        "construction_records": []
    }
    with open(os.path.join(output_dir, "interfaces.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
