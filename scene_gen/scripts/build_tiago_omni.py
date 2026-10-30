"""Offline conversion of pinned PAL data. No controller or generated code runs."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/".cache/tiago_build_deps"))
import mujoco
import numpy as np
import trimesh
import xacro
import xacro.substitution_args

SOURCES = PROJECT/".cache/tiago_omni_sources"
OUTPUT = PROJECT/"data/robots/tiago_omni"
PACKAGES = {p.parent.name: p.parent for p in SOURCES.rglob("package.xml")}
HOME = {"torso_lift_joint": .20, **dict(zip(
    [f"arm_{i}_joint" for i in range(1,8)], [.2,-1.0,-.2,1.94,-1.57,1.37,0])),
    "head_1_joint": 0, "head_2_joint": 0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(gripper):
    xacro.substitution_args._eval_find = lambda name: PACKAGES[name].as_posix()
    doc = xacro.process_file(str(PACKAGES["tiago_description"]/"robots/tiago.urdf.xacro"),
        mappings={"base_type":"omni_base", "end_effector":gripper,
                  "ft_sensor":"no-ft-sensor", "is_public_sim":"true"})
    return ET.fromstring(doc.toxml())


def build(gripper, filename):
    root = expand(gripper)
    source_name = "source_"+filename.replace(".xml",".urdf")
    ET.indent(root)
    ET.ElementTree(root).write(OUTPUT/source_name, encoding="utf-8", xml_declaration=True)
    source_limits = {j.get("name"): dict(j.find("limit").attrib)
                     for j in root.findall("joint") if j.find("limit") is not None}
    for element in list(root):
        if element.tag not in {"link","joint","material"}:
            root.remove(element)  # ROS/Gazebo controller extensions are not MuJoCo.
    m = ET.SubElement(root,"mujoco")
    ET.SubElement(m,"compiler",discardvisual="false",fusestatic="false",
                  strippath="false",balanceinertia="false")
    for link in root.findall("link"):
        for kind in ("visual","collision"):
            for index, geom in enumerate(link.findall(kind)):
                geom.set("name", f'{link.get("name")}__{kind}_{index}')
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename")
        assert uri.startswith("package://"), uri
        package, relative = uri[len("package://"):].split("/",1)
        source = (PACKAGES[package]/relative).resolve()
        assert source.is_relative_to(PACKAGES[package].resolve())
        dest = OUTPUT/"assets"/(sha(source)[:20]+".stl")
        if not dest.exists():
            if source.suffix.lower()==".stl":
                shutil.copyfile(source,dest)
            else:
                scene = trimesh.load(source, force="scene")
                converted = scene.to_geometry()
                converted.export(dest, file_type="stl")
        mesh.set("filename",dest.as_posix())
    temp = OUTPUT/("import_"+filename.replace(".xml",".urdf"))
    ET.ElementTree(root).write(temp,encoding="utf-8",xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(temp))
    converted = OUTPUT/filename
    mujoco.mj_saveLastXML(str(converted),model)
    xml = ET.parse(converted)
    # mj_saveLastXML rounds numbers. Restore full precision for kinematic and
    # inertial quantities from the imported model, instead of loosening tests.
    def numbers(values):
        return " ".join(format(float(v),".17g") for v in np.atleast_1d(values))
    for body in xml.iter("body"):
        bid=model.body(body.get("name")).id
        body.set("pos",numbers(model.body_pos[bid]))
        body.set("quat",numbers(model.body_quat[bid]))
        inertial=body.find("inertial")
        if inertial is not None:
            inertial.attrib.clear()
            inertial.attrib.update(pos=numbers(model.body_ipos[bid]),quat=numbers(model.body_iquat[bid]),
                                   mass=numbers(model.body_mass[bid]),diaginertia=numbers(model.body_inertia[bid]))
    for joint in xml.iter("joint"):
        if not joint.get("name"):
            continue
        jid=model.joint(joint.get("name")).id
        joint.set("pos",numbers(model.jnt_pos[jid]))
        joint.set("axis",numbers(model.jnt_axis[jid]))
        if model.jnt_limited[jid]:
            joint.set("range",numbers(model.jnt_range[jid]))
    compiler = xml.getroot().find("compiler")
    compiler.attrib.clear()
    compiler.set("angle","radian")
    for mesh in xml.iter("mesh"):
        mesh.set("file","assets/"+Path(mesh.get("file")).name)
    world = xml.getroot().find("worldbody")
    base = world.find("body")
    assert base is not None and base.get("name") in {"base_footprint","base_link"}
    for name,kind,axis in (("x","slide","1 0 0"),("y","slide","0 1 0"),("yaw","hinge","0 0 1")):
        ET.SubElement(base,"joint",name="base_"+name,type=kind,axis=axis,damping="100",limited="false")
    flange = next(b for b in xml.iter("body") if b.get("name")=="arm_tool_link")
    ET.SubElement(flange,"site",name="attachment_site",size=".008",rgba="0 0.8 0.3 1",group="4")
    for g in xml.iter("geom"):
        visual = "__visual_" in g.get("name","")
        g.set("group","2" if visual else "3")
        if visual:
            g.set("contype","0");g.set("conaffinity","0")
    # Engineering position servos; force bounds come from official URDF.
    # They are not claimed to reproduce the manufacturer's closed-loop controller.
    actuators = ET.SubElement(xml.getroot(),"actuator")
    for j in xml.iter("joint"):
        name = j.get("name","")
        limit = source_limits.get(name,{})
        if not name or name.startswith(("base_","wheel_")) or "lower" not in limit:
            continue
        kp = 2000 if name=="torso_lift_joint" else (800 if name.startswith("arm_") else 100)
        ET.SubElement(actuators,"position",name=name+"_position",joint=name,
                      kp=str(kp),kv=str(kp*.1),ctrlrange=limit["lower"]+" "+limit["upper"],
                      forcerange="-"+limit["effort"]+" "+limit["effort"])
    ET.SubElement(xml.getroot(),"option",timestep=".002",integrator="implicitfast")
    ET.SubElement(xml.getroot(),"visual").append(ET.Element("global",offwidth="1280",offheight="960"))
    ET.indent(xml)
    xml.write(converted,encoding="utf-8",xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(converted))
    data = mujoco.MjData(model)
    home = dict(HOME)
    if gripper=="pal-gripper":
        home.update(gripper_left_finger_joint=.035,gripper_right_finger_joint=.035)
    for name,value in home.items():
        data.qpos[model.joint(name).qposadr[0]]=value
    for a in range(model.nu):
        data.ctrl[a]=data.qpos[model.jnt_qposadr[model.actuator_trnid[a,0]]]
    mujoco.mj_forward(model,data)
    key = ET.SubElement(xml.getroot(),"keyframe")
    ET.SubElement(key,"key",name="home",qpos=" ".join(map(str,data.qpos)),ctrl=" ".join(map(str,data.ctrl)))
    ET.indent(xml);xml.write(converted,encoding="utf-8",xml_declaration=True)
    return {"file":filename,"end_effector":gripper,"joints":model.njnt,"geoms":model.ngeom,
            "home":home,"shoulder_height":float(data.xpos[model.body("arm_1_link").id,2]),
            "mass_kg":float(model.body_mass.sum()),"source_limits":source_limits}


def main():
    if (OUTPUT/"provenance.json").exists():
        record=json.loads((OUTPUT/"provenance.json").read_text(encoding="utf-8"))
        for relative, expected in record["files"].items():
            path=(OUTPUT/relative).resolve()
            if not path.is_relative_to(OUTPUT.resolve()) or sha(path)!=expected:
                raise ValueError("Existing robot modified; preserved without overwrite: "+relative)
        print("Existing TIAGo OMNI verified; no files overwritten.")
        return
    if OUTPUT.exists():
        raise ValueError("Incomplete robot directory preserved; inspect it before rebuilding.")
    from fetch_tiago_sources import REPOS
    for repo, revision in REPOS.items():
        folder=SOURCES/repo
        record=json.loads((folder/"source.json").read_text(encoding="utf-8"))
        if record["revision"]!=revision:
            raise ValueError("Unexpected source revision: "+repo)
        for relative, expected in record["files"].items():
            path=(folder/relative).resolve()
            if not path.is_relative_to(folder.resolve()) or sha(path)!=expected:
                raise ValueError("Source changed: "+relative)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"assets").mkdir(exist_ok=True)
    variants = [build("no-end-effector","model.xml"),build("pal-gripper","stock_gripper.xml")]
    sources = [json.loads(p.read_text(encoding="utf-8")) for p in SOURCES.glob("*/source.json")]
    licenses=[]
    for p in SOURCES.glob("*/LICENSE*"):
        licenses.append(f"\nSOURCE: {p.parent.name}\n"+p.read_text(encoding="utf-8"))
    (OUTPUT/"LICENSE").write_text("\n".join(licenses),encoding="utf-8")
    provenance={"source":"PAL Robotics official TIAGo with base_type=omni_base",
        "sources":sources,"variants":variants,"generated_by":"offline engineering conversion, no model calls",
        "approximations":["Planar x/y/yaw base coordinates; no Mecanum roller-ground dynamics.",
                          "Manufacturer collision primitives/meshes retained; MuJoCo uses convex mesh contacts.",
                          "Engineering position servos, not manufacturer controllers.",
                          "Default assembly variant has a bare wrist for the pipeline-generated end effector; stock_gripper.xml retains PAL gripper."],
        "files":{p.relative_to(OUTPUT).as_posix():sha(p) for p in OUTPUT.rglob("*") if p.is_file() and p.name!="provenance.json"}}
    (OUTPUT/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(json.dumps(variants,indent=2))

if __name__=="__main__":
    main()
