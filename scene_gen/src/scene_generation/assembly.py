"""The framework owns assembly and namespaces; the model cannot submit the whole scene or replace the robot."""

import copy
import math
from pathlib import Path
from defusedxml import ElementTree as ET
from xml.etree.ElementTree import tostring

from asset_generation.checks import load_model, validate_manifest
from asset_generation.timing import stage
from asset_generation.contracts import read_json, require
from asset_prep.storage import atomic_json, digest, safe_path, file_hash
from .table import validate_cuts, table_cells
from .layout import tables, validate_layout, mobile


def component(root, prefix, model_file="model.xml", *, diagnostic_names=None):
    import mujoco
    root = Path(root)
    with stage("component_load:"+prefix):
        load_model(root, model_file=model_file)  # Independent safe parse; generated scripts are never executed here.
    tree = ET.parse(root/model_file)
    # Vendor models legitimately contain unnamed geoms. Give the derived scene
    # resolvable diagnostic references, without modifying geometry or vendor files.
    geoms=tree.findall(".//worldbody//geom")
    names={node.get("name") for node in geoms if node.get("name")}
    for index,node in enumerate(geoms):
        if node.get("name"):continue
        name=f"diagnostic_geom_{index}"
        while name in names:name+="_"
        node.set("name",name);names.add(name)
        if diagnostic_names is not None:
            diagnostic_names[prefix+name]={"source_geom_ordinal":index,"original_name":None}
    for keyframe in tree.findall("keyframe"):
        tree.getroot().remove(keyframe)
    assets = {}
    for node in tree.iter():
        for attribute in list(node.attrib):
            if not attribute.startswith("file"):
                continue
            source = safe_path(root, node.get(attribute))
            # When the MJCF declares no name, infer it from the file name; keep the original name before rewriting paths.
            if attribute == "file" and node.tag in {"mesh", "texture", "hfield", "skin"} and not node.get("name"):
                node.set("name", source.stem)
            relative = "resources/" + prefix + file_hash(source)[:20] + source.suffix.lower()
            node.set(attribute, relative)
            assets[relative] = source.read_bytes()
    return mujoco.MjSpec.from_string(tostring(tree.getroot(), encoding="unicode"), assets=assets)


def namespace_manifest(value, prefix):
    result = copy.deepcopy(value)
    for group in ("parts", "interfaces", "buttons", "connections", "mechanical_events"):
        for item in result.get(group,[]):
            for key in ("id", "body", "site", "joint", "geom", "event", "equality", "release_event", "attach_event", "interface"):
                if key in item:
                    item[key] = prefix + item[key]
            if "capture_sites" in item:
                item["capture_sites"] = [prefix+n for n in item["capture_sites"]]
    from asset_generation.operation_interface import namespace_operation
    for interface in result.get("interfaces", []):
        namespace_operation(interface, prefix)
    return result


def assemble(plan, env, artifacts, robot_root, output):
    import mujoco
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_layout(plan, env)
    cuts = validate_cuts(plan.get("table_modifications", []), plan, env)
    spec = mujoco.MjSpec.from_string(f'''<mujoco model="generated_lab">
      <compiler angle="radian"/><option timestep="0.002" integrator="implicitfast"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody><light pos="0 0 3"/>
        <geom name="environment__floor" type="plane" size="5 5 .1" rgba=".25 .27 .30 1"/>
      </worldbody></mujoco>''')
    instances = {i["id"]: i for i in plan["instances"]}
    for table in tables(plan, env):
        sx, sy, thickness = table["size"]
        height = table["height"]
        selected = [c for c in cuts if instances[c["instance"]].get("table_id", "main") == table["id"]]
        suffix = "_"+table["id"] if "layout" in plan else ""
        quat = [math.cos(table["yaw"]/2),0,0,math.sin(table["yaw"]/2)]
        body = spec.worldbody.add_body(name="environment__table"+suffix,
            pos=[*table["position"],height-thickness/2], quat=quat)
        for index, (pos, size) in enumerate(table_cells(table, selected)):
            ending = suffix+f"__{index}" if suffix else (f"_{index}" if selected else "")
            body.add_geom(name="environment__table_top"+ending,
                type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=[.65,.68,.7,1])
        for index, (x, y) in enumerate(( (x,y) for x in (-sx/2+.08,sx/2-.08) for y in (-sy/2+.08,sy/2-.08))):
            body.add_geom(name=f"environment__leg{suffix}_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x,y,-height/2], size=[.035,.035,(height-thickness)/2], rgba=[.3,.3,.3,1])
    if "layout" in plan:
        # Visible room boundary is physical; doors and navigation are outside this phase.
        rx, ry = plan["layout"]["room"]["size"]
        spec.geom("environment__floor").size = [rx/2,ry/2,.1]
        for index, (pos, size) in enumerate((
                ([0,-ry/2,.15],[rx/2,.025,.15]), ([0,ry/2,.15],[rx/2,.025,.15]),
                ([-rx/2,0,.15],[.025,ry/2,.15]), ([rx/2,0,.15],[.025,ry/2,.15]))):
            spec.worldbody.add_geom(name=f"environment__room_boundary_{index}",
                type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=[.4,.42,.46,1])
    from .robot_contract import model_file, mount_for, arms, initialize
    robot_variant = model_file(plan,env)
    robot_diagnostic_names={}
    robot = component(robot_root, "robot__", robot_variant,diagnostic_names=robot_diagnostic_names)
    # MjSpec.attach does not inherit child global solver settings. Preserve the manufacturer's
    # contact formulation instead of silently softening high-friction wheel contacts.
    spec.option.cone=robot.option.cone
    spec.option.impratio=robot.option.impratio
    robot_dynamics={"cone":int(robot.option.cone),"impratio":float(robot.option.impratio),
                    "source_model_sha256":file_hash(Path(robot_root)/robot_variant)}
    for arm in arms(env):
        require(robot.site(arm["wrist_site"]) is not None, "robot wrist interface does not exist")
    if mobile(env) and env["robot"]["mobile"].get("integrated"):
        # The whole robot already contains its manufacturer base and torso.
        # Validate its adapter coordinates before applying world-frame layout poses.
        from .robot import validate_integrated
        validate_integrated(robot.compile())
        robot_frame = spec.worldbody.add_frame(name="robot_mount",
            pos=[0,0,env["robot"]["mobile"].get("ground_offset_m",0)])
    elif mobile(env):
        platform = env["robot"]["mobile"]
        bx, by, bz = platform["base_size"]
        base = spec.worldbody.add_body(name="robot__mobile_base")
        for axis, kind, direction in (("x",mujoco.mjtJoint.mjJNT_SLIDE,[1,0,0]),
                                      ("y",mujoco.mjtJoint.mjJNT_SLIDE,[0,1,0]),
                                      ("yaw",mujoco.mjtJoint.mjJNT_HINGE,[0,0,1])):
            base.add_joint(name="robot__base_"+axis, type=kind, axis=direction, damping=100)
        base.add_geom(name="robot__base_hull", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0,0,bz/2],size=[bx/2,by/2,bz/2],mass=platform["mass_kg"],rgba=[.8,.8,.83,1])
        # Cosmetic wheel disks sit within the declared base envelope. Planar joints
        # supply motion; these disks do not pretend to implement wheel-ground dynamics.
        for index,(x,y) in enumerate(( (x,y) for x in (-bx*.32,bx*.32) for y in (-by*.46,by*.46))):
            base.add_geom(name=f"robot__wheel_visual_{index}",type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[x,y,.09],quat=[math.sqrt(.5),math.sqrt(.5),0,0],size=[.09,by*.035,0],
                contype=0,conaffinity=0,mass=0,rgba=[.08,.08,.09,1])
        mount = platform["mount_height"]
        base.add_geom(name="robot__mast",type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0,0,(mount+bz)/2],size=[.09,.09,(mount-bz)/2],mass=1,rgba=[.3,.3,.35,1])
        robot_frame = base.add_frame(name="robot_mount",pos=[0,0,mount])
    else:
        robot_frame = spec.worldbody.add_frame(name="robot_mount", pos=[0,0,env["table"]["height"]])
    spec.attach(robot, prefix="robot__", frame=robot_frame)
    manifest = {"schema_version":"1.0", "asset_ids":[], "parts":[], "interfaces":[],
                "assumptions":list(plan["assumptions"]), "approximations":[], "buttons":[], "connections":[], "mechanical_events":[],"construction_records":[]}
    jobs = {j["id"]: j for j in plan["jobs"]}
    mounts = []
    component_versions = []
    for item in plan["instances"]:
        prefix = item["id"] + "__"
        root = Path(artifacts[item["job"]])
        from asset_generation.configurations import selection
        configuration, model_file, selected_manifest = selection(root, item.get("configuration"))
        from .candidate import assembly_manifest
        local, interface_issues = assembly_manifest(selected_manifest, load_model(root, model_file=model_file), jobs[item["job"]]["asset_ids"], diagnostic=jobs[item["job"]]["role"]!="tool")
        component_versions.append({"instance":item["id"],"job":item["job"],"configuration":configuration,
            "model_file":model_file,"model_sha256":file_hash(root/model_file),
            "manifest_sha256":digest(selected_manifest),"interface_issues":interface_issues})
        diagnostic_names={}
        child = component(root, prefix, model_file,diagnostic_names=diagnostic_names)
        component_versions[-1]["diagnostic_geometry_names"]=diagnostic_names
        if item["placement"] == "robot":
            tool_model = load_model(root, model_file=model_file)
            from .tool_contract import tool_roots, attach_tool
            roots, free_roots = tool_roots(tool_model, local)
            mount = mount_for(plan,env,item['id'])
            mounts.append({'instance':item['id'],'arm':mount['arm'],'mode':mount['mode'],'attachment_site':'robot__'+mount['site'],
                'source_model_sha256':file_hash(root/model_file),
                'roots':[{'body':prefix+tool_model.body(b).name,'local_pos':tool_model.body_pos[b].tolist(),
                          'local_quat':tool_model.body_quat[b].tolist(), 'free':b in free_roots} for b in roots]})
            spec = attach_tool(spec, child, tool_model, local, prefix, "robot__"+mount["site"])
        else:
            frame = spec.worldbody.add_frame(name=prefix+"placement", pos=item["position"],
                                            quat=[math.cos(item["yaw"]/2),0,0,math.sin(item["yaw"]/2)])
            spec.attach(child, prefix=prefix, frame=frame)
        manifest["construction_records"].extend({"instance":item["id"],"record":r} for r in local.get("construction_records",[]))
        namespaced = namespace_manifest(local, prefix)
        for group in ("parts", "interfaces", "assumptions", "approximations", "buttons", "connections", "mechanical_events"):
            manifest[group].extend(namespaced.get(group,[]))
        manifest["asset_ids"].extend(a for a in local["asset_ids"] if a not in manifest["asset_ids"])
    with stage("scene_compile"):
        model = spec.compile()
    data = mujoco.MjData(model)
    if mobile(env):
        for axis, value in zip(("x","y","yaw"), plan["layout"]["robot_home"]):
            data.qpos[model.joint("robot__base_"+axis).qposadr[0]] = value
        manifest["approximations"].extend(env["robot"]["mobile"]["approximations"])
        manifest["assumptions"].append("mobile platform source: "+env["robot"]["mobile"]["source"])
    initialize(model,data,env,posture=plan.get("robot_posture"))
    from .tool_contract import align_free_roots
    align_free_roots(model, data, mounts)
    if mobile(env):
        # Fit the initial view to occupied workspace, not distant room boundaries.
        import numpy as np
        from .geometry import bounds
        from asset_generation.static_state import geometry_state
        geometry_state(model,data)
        boxes=[bounds(model,data,g) for g in range(model.ngeom)
               if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE
               and not (model.geom(g).name or "").startswith("environment__room_boundary_")]
        if boxes:
            lo,hi=np.min([b[0] for b in boxes],axis=0),np.max([b[1] for b in boxes],axis=0)
            spec.stat.center=(lo+hi)/2
            spec.stat.extent=max(1,float(np.linalg.norm(hi-lo))*.8)
    spec.add_key(name="scene_home", qpos=data.qpos.tolist(), ctrl=data.ctrl.tolist())
    # MjSpec.attach also prefixes asset file names; export from the final virtual file system, do not reuse child-model paths.
    for relative, content in spec.assets.items():
        target = safe_path(output,relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    # The XML keeps inertias to 6 significant digits: a flat part whose principal moments sit exactly on
    # A + B = C (a membrane, a foil) can fail that check when the written scene is loaded again.
    # A near-massless part (a membrane piece of ~1 mg) can also lose its mass and moments to that rounding.
    spec.compiler.balanceinertia = True
    spec.compiler.boundmass, spec.compiler.boundinertia = 1e-7, 1e-13
    (output/"model.xml").write_text(spec.to_xml(), encoding="utf-8")
    atomic_json(output/"interfaces.json", manifest)
    atomic_json(output/"scene.json", {"plan":plan, "environment":env,
                                    "partial_without_tool":not any(i["placement"]=="robot" for i in plan["instances"]),
                                    "robot_dynamics":robot_dynamics,"robot_variant":robot_variant,"robot_model_sha256":file_hash(Path(robot_root)/robot_variant),
                                    "robot_diagnostic_geometry_names":robot_diagnostic_names,
                                    "robot_ground_offset_m":env["robot"].get("mobile",{}).get("ground_offset_m",0),
                                    "home_qpos":data.qpos.tolist(),
                                    "home_ctrl":data.ctrl.tolist(),"mounts":mounts,"component_versions":component_versions})
    for original, exported in (("LICENSE","robot_LICENSE.txt"),("provenance.json","robot_provenance.json")):
        source = Path(robot_root)/original
        if source.is_file():
            (output/exported).write_bytes(source.read_bytes())
    return output
