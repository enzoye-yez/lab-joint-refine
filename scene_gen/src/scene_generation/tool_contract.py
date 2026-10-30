"""One tool contract for authoring, admission and assembly; no quality exemptions."""
import numpy as np
import mujoco
from defusedxml import ElementTree as ET
from xml.etree.ElementTree import SubElement, tostring
from asset_generation.contracts import require
from asset_generation.static_state import geometry_state

TOOL_POLICY = """工具局部原点是机器人安装法兰，+Z为法兰轴；必须提供tcp操作site。主体根部固定安装，不含自由关节。
可更换部件须另建worldbody直属body并带freejoint，通过清单connections声明的初始启用weld连接到主体（允许逐级连接）。
不能把可拆件放在主体的永久子树内，也不能交付未连接到主体的自由部件。框架只把主体安装到法兰；独立部件保留世界根部自由关节，并转换到法兰实际姿态。
清单的释放/接合事件、定位site和公差遵守通用连接契约；静态检查不认证操作能力。完整装置包括可拆件的质量计入负载。主动机构包含真实关节和actuator。"""


def tool_roots(model, manifest):
    """Call after validate_manifest, which validates each releasable weld."""
    roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]
    free = {b for b in roots if model.body_jntnum[b] and
            model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE}
    fixed = set(roots) - free
    require(fixed, "末端须有固定安装主体；不能所有根部都带自由关节")
    require(all(model.body(b).name for b in roots), "末端根部必须命名")
    require(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp") >= 0, "末端必须提供 tcp 操作坐标")
    actors = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    require(actors and all(model.geom(g).name for g in actors), "末端碰撞几何必须存在且命名")
    edges = []
    for connection in manifest["connections"]:
        eq = model.equality(connection["equality"])
        edges.append(tuple(int(model.body_rootid[int(b)]) for b in (eq.obj1id[0], eq.obj2id[0])))
    reached = set(fixed)
    while True:
        expanded = reached | {b for edge in edges if reached.intersection(edge) for b in edge if b}
        if expanded == reached: break
        reached = expanded
    require(free <= reached, "可拆末端部件须经初始启用且声明的 weld 连接到安装主体；未连接：" +
            ", ".join(model.body(b).name for b in sorted(free - reached)))
    return roots, free


def pose(position, quaternion, local_pos, local_quat):
    pos, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(pos, quat, np.asarray(position), np.asarray(quaternion),
                      np.asarray(local_pos), np.asarray(local_quat))
    return pos, quat


def attach_tool(spec, child, model, manifest, prefix, site_name):
    roots, free = tool_roots(model, manifest)
    if not free:
        spec.attach(child, prefix=prefix, site=spec.site(site_name))
        return spec
    # Attach once at world level to preserve cross-root welds, actuators and resources.
    # Move only mounted roots beneath the flange; free joints must remain world children.
    spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame())
    current = spec.compile()
    data = mujoco.MjData(current)
    geometry_state(current, data)
    sid = current.site(site_name).id
    flange_quat = np.zeros(4)
    mujoco.mju_mat2Quat(flange_quat, data.site_xmat[sid])
    tree = ET.fromstring(spec.to_xml())
    parents = {node: parent for parent in tree.iter() for node in parent}
    bodies = {b.get("name"): b for b in tree.iter("body")}
    parent_name = current.body(int(current.site_bodyid[sid])).name
    mount = SubElement(tree.find("worldbody") if current.site_bodyid[sid] == 0 else bodies[parent_name], "frame", pos=" ".join(map(str, current.site_pos[sid])),
                       quat=" ".join(map(str, current.site_quat[sid])))
    for b in roots:
        node = bodies[prefix + model.body(b).name]
        parents[node].remove(node)
        pos, quat = model.body_pos[b], model.body_quat[b]
        if b in free:
            pos, quat = pose(data.site_xpos[sid], flange_quat, pos, quat)
        node.set("pos", " ".join(map(str, pos)))
        node.set("quat", " ".join(map(str, quat)))
        (tree.find("worldbody") if b in free else mount).append(node)
    return mujoco.MjSpec.from_string(tostring(tree, encoding="unicode"), assets=spec.assets)


def align_free_roots(model, data, mounts):
    """Set initial detached-component poses at the robot home pose, without stepping."""
    geometry_state(model, data)
    for mount in mounts:
        sid = model.site(mount["attachment_site"]).id
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
        for root in mount["roots"]:
            if not root.get("free"): continue
            b = model.body(root["body"]).id
            q = model.jnt_qposadr[model.body_jntadr[b]]
            pos, rotation = pose(data.site_xpos[sid], quat, root["local_pos"], root["local_quat"])
            data.qpos[q:q+7] = np.r_[pos, rotation]


def has_free_parts(model, prefix):
    return any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and
               model.body(int(model.jnt_bodyid[j])).name.startswith(prefix) for j in range(model.njnt))
