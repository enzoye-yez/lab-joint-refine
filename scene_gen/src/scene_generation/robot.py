"""Prepare the pinned public FR3 model; reads no secrets and runs no downloaded programs."""

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
            raise ValidationError("Whole-robot model lacks planar base coordinate: "+name) from exc
        bodies.append(int(joint.bodyid[0]))
        require(joint.type[0] == kind and np.allclose(joint.axis,axis) and np.allclose(joint.pos,0)
                and model.qpos0[joint.qposadr[0]] == 0
                and not joint.limited[0], "Whole-robot base coordinate type, axis or range mismatch")
    body = bodies[0]
    require(len(set(bodies)) == 1 and model.body_parentid[body] == 0
            and np.allclose(model.body_pos[body],0) and np.allclose(model.body_quat[body],[1,0,0,0]),
            "Whole-robot base coordinates must sit on the same root body at the world origin")
    require(model.body_jntnum[body] == 3 and
            [model.joint(int(model.body_jntadr[body])+i).name for i in range(3)] ==
            ["base_x","base_y","base_yaw"], "Whole-robot base coordinate order must be world X/Y then planar yaw")


def prepare_environment(project, env):
    """Existing integrated resources are verified offline, never replaced by FR3."""
    if not env["robot"].get("mobile",{}).get("integrated"):
        return prepare(project)
    root = safe_path(project,env["robot"]["directory"])
    require((root/"provenance.json").is_file(),
            "Whole-robot resources not prepared; prepare them with the offline build steps in the matching robot integration doc; no fallback to another robot")
    from asset_generation.contracts import read_json
    record = read_json(root/"provenance.json")
    require(isinstance(record.get("files"),dict) and "model.xml" in record["files"],
            "Whole-robot source record lacks the model digest")
    for relative, expected in record["files"].items():
        path = safe_path(root,relative)
        require(path.is_file() and file_hash(path)==expected,"Whole-robot resource does not match the source record: "+relative)
    model = load_model(root)
    validate_integrated(model)
    from .robot_contract import arms, initialize
    import mujoco
    for arm in arms(env):
        require(model.site(arm["wrist_site"]).id >= 0,"Whole-robot wrist interface does not exist")
    initialize(model,mujoco.MjData(model),env,prefix="")
    return root

REVISION = "0059d4335f8156206f63a35662313385f7ad6d74"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/franka_fr3/"


def download(relative, max_bytes=16000000):
    require(relative in {"fr3.xml","LICENSE"} or relative.startswith("assets/"), "Invalid model download scope")
    require(".." not in relative and ":" not in relative, "Invalid download path")
    url = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/franka_fr3/{relative}?ref={REVISION}"
    request = urllib.request.Request(url,headers={"User-Agent":"harness-fr3-setup"})
    with urllib.request.urlopen(request,timeout=60) as response:
        require(response.url == url, "Model download does not accept redirects")
        raw = response.read(max_bytes*2+1)
    require(len(raw) <= max_bytes*2,"Model response exceeds the limit")
    metadata = json.loads(raw)
    require(metadata.get("size",max_bytes+1) <= max_bytes,"Model file exceeds the limit")
    if metadata.get("encoding") != "base64":
        url = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/blobs/"+metadata["sha"]
        request = urllib.request.Request(url,headers={"User-Agent":"harness-fr3-setup"})
        with urllib.request.urlopen(request,timeout=60) as response:
            require(response.url == url,"Model download does not accept redirects")
            raw = response.read(max_bytes*2+1)
        require(len(raw) <= max_bytes*2,"Model response exceeds the limit")
        metadata = json.loads(raw)
    require(metadata.get("encoding") == "base64","Unsupported public model content encoding")
    content = base64.b64decode("".join(metadata["content"].split()),validate=True)
    require(len(content) <= max_bytes,"Model file exceeds the limit")
    require(hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest() == metadata["sha"],"Public model content digest mismatch")
    return content


def prepare(project, fetch=download):
    root = Path(project)/"data/robots/fr3"
    require(not root.exists(), "Robot directory already exists; keeping the existing model, no automatic overwrite")
    xml = fetch("fr3.xml")
    tree = ET.fromstring(xml)
    compiler = tree.find("compiler")
    require(compiler is not None and compiler.get("meshdir") == "assets", "Public model structure changed; the adapter must be checked")
    compiler.attrib.pop("meshdir")
    files = {"LICENSE":fetch("LICENSE"),"source/fr3.xml":xml}
    for node in tree.iter():
        require(node.tag not in {"include","plugin","extension","attach"}, "Public robot model contains unsupported content")
        if node.tag == "mesh" and node.get("file"):
            name = node.get("file")
            require(Path(name).name == name and ":" not in name and "\\" not in name, "Invalid mesh path")
            # Pin implicit mesh names before rewriting file.
            if not node.get("name"):
                node.set("name",Path(name).stem)
            relative = "assets/"+name
            files[relative] = fetch(relative)
            node.set("file",relative)
    require(sum(len(x) for x in files.values()) <= 64000000, "Total robot resources exceed the limit")
    files["model.xml"] = tostring(tree,encoding="utf-8")
    root.mkdir(parents=True)
    for name,data in files.items():
        path = safe_path(root,name)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    load_model(root)
    provenance = {"source":"https://github.com/google-deepmind/mujoco_menagerie/tree/"+REVISION+"/franka_fr3",
                  "revision":REVISION,"model":"Franka Research 3","license":"LICENSE",
                  "normalization":"mesh paths expanded with implicit names kept; no geometry, kinematics or dynamics changes",
                  "files":{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
    atomic_json(root/"provenance.json",provenance)
    return root
