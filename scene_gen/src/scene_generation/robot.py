"""固定版本的公开FR3模型准备；不读取密钥、不运行下载的程序。"""

import hashlib
import base64
import json
import urllib.request
from pathlib import Path
from defusedxml import ElementTree as ET
from xml.etree.ElementTree import tostring

from asset_generation.checks import load_model
from asset_generation.contracts import require
from asset_prep.storage import atomic_json, safe_path, file_hash


def validate_integrated(model):
    """The whole-robot adapter uses one floor-root planar frame; no added pedestal."""
    import mujoco
    import numpy as np
    bodies = []
    for name, kind, axis in (("base_x",mujoco.mjtJoint.mjJNT_SLIDE,[1,0,0]),
                             ("base_y",mujoco.mjtJoint.mjJNT_SLIDE,[0,1,0]),
                             ("base_yaw",mujoco.mjtJoint.mjJNT_HINGE,[0,0,1])):
        try:
            joint = model.joint(name)
        except KeyError as exc:
            from asset_prep.errors import ValidationError
            raise ValidationError("整机模型缺少平面底盘坐标："+name) from exc
        bodies.append(int(joint.bodyid[0]))
        require(joint.type[0] == kind and np.allclose(joint.axis,axis) and np.allclose(joint.pos,0)
                and model.qpos0[joint.qposadr[0]] == 0
                and not joint.limited[0], "整机底盘坐标类型、轴或范围不匹配")
    body = bodies[0]
    require(len(set(bodies)) == 1 and model.body_parentid[body] == 0
            and np.allclose(model.body_pos[body],0) and np.allclose(model.body_quat[body],[1,0,0,0]),
            "整机底盘坐标须位于世界原点的同一根节点")
    require(model.body_jntnum[body] == 3 and
            [model.joint(int(model.body_jntadr[body])+i).name for i in range(3)] ==
            ["base_x","base_y","base_yaw"], "整机底盘坐标顺序须为世界X/Y和平面转角")


def prepare_environment(project, env):
    """Existing integrated resources are verified offline, never replaced by FR3."""
    if not env["robot"].get("mobile",{}).get("integrated"):
        return prepare(project)
    root = safe_path(project,env["robot"]["directory"])
    require((root/"provenance.json").is_file(),
            "整机资源未准备；请按对应整机接入文档的离线构建步骤准备，不回退为其他机器人")
    from asset_generation.contracts import read_json
    record = read_json(root/"provenance.json")
    require(isinstance(record.get("files"),dict) and "model.xml" in record["files"],
            "整机来源记录缺少模型摘要")
    for relative, expected in record["files"].items():
        path = safe_path(root,relative)
        require(path.is_file() and file_hash(path)==expected,"整机资源与来源记录不一致："+relative)
    model = load_model(root)
    validate_integrated(model)
    from .robot_contract import arms, initialize
    import mujoco
    for arm in arms(env):
        require(model.site(arm["wrist_site"]).id >= 0,"整机腕部接口不存在")
    initialize(model,mujoco.MjData(model),env,prefix="")
    return root

REVISION = "0059d4335f8156206f63a35662313385f7ad6d74"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/franka_fr3/"


def download(relative, max_bytes=16000000):
    require(relative in {"fr3.xml","LICENSE"} or relative.startswith("assets/"), "模型下载范围无效")
    require(".." not in relative and ":" not in relative, "下载路径无效")
    url = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/franka_fr3/{relative}?ref={REVISION}"
    request = urllib.request.Request(url,headers={"User-Agent":"harness-fr3-setup"})
    with urllib.request.urlopen(request,timeout=60) as response:
        require(response.url == url, "模型下载不接受重定向")
        raw = response.read(max_bytes*2+1)
    require(len(raw) <= max_bytes*2,"模型响应超限")
    metadata = json.loads(raw)
    require(metadata.get("size",max_bytes+1) <= max_bytes,"模型文件超限")
    if metadata.get("encoding") != "base64":
        url = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/blobs/"+metadata["sha"]
        request = urllib.request.Request(url,headers={"User-Agent":"harness-fr3-setup"})
        with urllib.request.urlopen(request,timeout=60) as response:
            require(response.url == url,"模型下载不接受重定向")
            raw = response.read(max_bytes*2+1)
        require(len(raw) <= max_bytes*2,"模型响应超限")
        metadata = json.loads(raw)
    require(metadata.get("encoding") == "base64","公开模型内容编码不支持")
    content = base64.b64decode("".join(metadata["content"].split()),validate=True)
    require(len(content) <= max_bytes,"模型文件超限")
    require(hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest() == metadata["sha"],"公开模型内容摘要不一致")
    return content


def prepare(project, fetch=download):
    root = Path(project)/"data/robots/fr3"
    require(not root.exists(), "机器人目录已存在；保留现有模型，不自动覆盖")
    xml = fetch("fr3.xml")
    tree = ET.fromstring(xml)
    compiler = tree.find("compiler")
    require(compiler is not None and compiler.get("meshdir") == "assets", "公开模型结构变化，需要核对适配器")
    compiler.attrib.pop("meshdir")
    files = {"LICENSE":fetch("LICENSE"),"source/fr3.xml":xml}
    for node in tree.iter():
        require(node.tag not in {"include","plugin","extension","attach"}, "公开机器人模型含未支持内容")
        if node.tag == "mesh" and node.get("file"):
            name = node.get("file")
            require(Path(name).name == name and ":" not in name and "\\" not in name, "网格路径无效")
            # 修改file前固定隐式mesh名称。
            if not node.get("name"):
                node.set("name",Path(name).stem)
            relative = "assets/"+name
            files[relative] = fetch(relative)
            node.set("file",relative)
    require(sum(len(x) for x in files.values()) <= 64000000, "机器人资源总量超限")
    files["model.xml"] = tostring(tree,encoding="utf-8")
    root.mkdir(parents=True)
    for name,data in files.items():
        path = safe_path(root,name)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    load_model(root)
    provenance = {"source":"https://github.com/google-deepmind/mujoco_menagerie/tree/"+REVISION+"/franka_fr3",
                  "revision":REVISION,"model":"Franka Research 3","license":"LICENSE",
                  "normalization":"网格路径展开并保留隐式名称；无几何、运动学或动力学改动",
                  "files":{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
    atomic_json(root/"provenance.json",provenance)
    return root
