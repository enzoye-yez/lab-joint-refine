"""Fetch pinned manufacturer data and derive static-layout variants. No upstream code execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

REVISION = "d8344c0dbe7a00208d0301111523dde65efc174a"
BASE = f"https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_mujoco_menagerie/{REVISION}/robotis_ffw/"


def prepare(project, rebuild=False):
    import mujoco
    root = project / "data/robots/ffw_sg2"
    root.mkdir(parents=True, exist_ok=True)
    record = root / "provenance.json"
    if record.exists():
        saved = json.loads(record.read_text("utf-8"))
        for name, sha in saved["files"].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha, name
        if not rebuild:
            print("Existing pinned FFW-SG2 resources verified.")
            return

    def fetch(relative, local=None):
        assert ".." not in relative and ":" not in relative
        path = root / (local or relative)
        if path.exists():
            return path.read_bytes()
        request = urllib.request.Request(BASE+relative, headers={"User-Agent": "harness-ffw-setup"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(64000001)
        assert len(data) <= 64000000
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    source = fetch("ffw_sg2.xml", "source.xml")
    fetch("LICENSE")
    doc = ET.fromstring(source)
    for mesh in doc.findall("./asset/mesh"):
        fetch("assets/"+mesh.attrib["file"])
        mesh.set("file", "assets/"+mesh.attrib["file"])
    doc.find("compiler").attrib.pop("meshdir")
    # Add only a massless coordinate frame. Manufacturer body coordinates remain intact.
    world = doc.find("worldbody")
    base = world.find("body")
    base.remove(base.find("freejoint"))
    world.remove(base)
    frame = ET.SubElement(world, "body", name="layout_root")
    for name, kind, axis in (("base_x","slide","1 0 0"),("base_y","slide","0 1 0"),("base_yaw","hinge","0 0 1")):
        ET.SubElement(frame,"joint",name=name,type=kind,axis=axis,limited="false")
    frame.append(base)
    for side, code in (("left","l"),("right","r")):
        wrist = doc.find(f".//body[@name='arm_{code}_link7']")
        shoulder = doc.find(f".//body[@name='arm_{code}_link1']")
        ET.SubElement(shoulder,"site",name=side+"_shoulder",size=".005",rgba="0 0 0 0")
        ET.SubElement(wrist,"site",name=side+"_wrist",pos="0 0 -0.078",quat="0 0 1 0",size=".005")
        ET.SubElement(wrist,"site",name=side+"_tcp",pos="0 0 -0.22",quat="0 0 1 0",size=".005")
        # Fixed housing accessory mount, not a fictitious fingertip attachment.
        ET.SubElement(wrist,"site",name=side+"_adapter",pos="0 0 -0.078",quat="0 0 1 0",size=".005")
        for geom in wrist.findall("geom"):
            if geom.get("mesh") == "gripper_base":
                geom.set("name",side+"_stock_housing")

    variants = {}
    for removed, filename in (((),"model.xml"),(("left",),"without_left.xml"),
                              (("right",),"without_right.xml"),(("left","right"),"without_both.xml")):
        variant = copy.deepcopy(doc)
        for side in removed:
            code = {"left":"l","right":"r"}[side]
            removed_names = set()
            for parent in list(variant.iter()):
                for child in list(parent):
                    if child.tag == "body" and child.get("name","").startswith(f"gripper_{code}_"):
                        removed_names.update(e.get("name") for e in child.iter() if e.get("name"))
                        parent.remove(child)
                    elif child.tag in {"geom","site"} and child.get("name") in {side+"_stock_housing",side+"_tcp",side+"_adapter"}:
                        parent.remove(child)
            for section in ("actuator","equality","contact"):
                parent=variant.find(section)
                for child in list(parent):
                    if any(v in removed_names for k,v in child.attrib.items() if k in {"joint","joint1","joint2","body1","body2"}):
                        parent.remove(child)
        path=root/filename
        ET.indent(variant)
        path.write_text(ET.tostring(variant,encoding="unicode"),"utf-8")
        model=mujoco.MjModel.from_xml_path(str(path))
        variants[",".join(removed)]=filename
        print(filename, model.njnt, model.nu, model.neq)
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob("*")) if p.is_file() and p != record}
    record.write_text(json.dumps({
        "manufacturer":"ROBOTIS","model":"FFW-SG2","revision":REVISION,"source":BASE,
        "files":files,"model_variants":variants,
        "adaptations":["新增无实体平面布局坐标，保留原整机根部高度及所有原连杆姿态。",
                      "腕部、肩部及工具中心标记为工程接口；工具中心为夹爪中间的工程参考点。",
                      "替换变体移除对应夹爪可分离几何、手指、驱动、联动及接触排除；保留上游腕部合并惯性，无法据此认证换装后的精确动力学。",
                      "上游仿真关节范围与硬件文档存在差异；本次保留固定模型范围，未用网页参数覆盖。",
                      "标准夹爪负载使用厂商标称0.5kg；不把上游合并惯性当成独立夹爪称重结果。"]
    },ensure_ascii=False,indent=2),"utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rebuild-adapter",action="store_true")
    args=parser.parse_args()
    prepare(args.project,args.rebuild_adapter)
