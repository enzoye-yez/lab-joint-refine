"""Shared layout contract and transforms; legacy fixed-table scenes stay readable.

All positions in plans are world coordinates. Tables are addressed by id; moving
a table carries its support tree, without scaling any generated asset.
"""
import copy
import math
import re

from asset_generation.contracts import finite, require
from asset_prep.storage import digest


def vector(value, n):
    require(isinstance(value, list) and len(value) == n and all(finite(v) for v in value),
            "布局坐标须为有限数值数组")
    return value


def name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", value),
            "布局编号无效")


def reason(value):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 4000, "布局变化须有明确理由")


def mobile(env):
    return bool(env.get("robot", {}).get("mobile"))


def table_policy(env):
    from .table import POLICY
    if not mobile(env):
        return POLICY
    return {**POLICY, "robot_protected_half_size_m":0,
            "rule":"各工作台独立计算加工面积；center为关联工作台局部坐标。移动机器人不占用桌面中央保护区；"
                   "仍保留桌边、桌腿和凹槽底厚要求。整体工作台尺寸及数量由layout设计，不缩放仪器。"}


def author_layout_policy(env):
    coordinates = ("关联table_id的工作台局部坐标" if mobile(env) else "世界坐标")
    setting = ("按scene_layout中实际多工作台及移动机器人布局设计，桌面中央没有固定机械臂保护区。"
               if mobile(env) else "桌面整体和机械臂安装区固定。")
    return (setting + "可按table_edit_policy局部开槽开孔。优先兼容已有安装条件；确需新增加工时在assumptions中明确"
            + coordinates + "、大小、深度和支撑方式，供场景协调器声明table_modifications。"
            "不得假设已有未声明凹槽，不能关闭碰撞。资产作者不能自行改变场景布局或缩放依赖仪器。")


def review_targets(plan):
    """Typed layout identities, separate from generated asset instance ids."""
    layout = plan.get("layout")
    if not layout:
        return []
    return ["robot_home", *("table:" + t["id"] for t in layout["tables"]),
            *("dock:" + d["id"] for d in layout.get("docks", []))]


def validate_mobile(value):
    required = {"base_size", "mount_height", "mass_kg", "source", "approximations"}
    require(isinstance(value, dict) and required <= set(value) <= required | {"integrated","ground_offset_m"},
            "移动平台须明确尺寸、安装高度、质量、来源与近似")
    require(type(value.get("integrated", False)) is bool, "整机标记须为布尔值")
    require(finite(value.get("ground_offset_m",0)) and abs(value.get("ground_offset_m",0))<=1,
            "整机落地偏移须来自资源测量且在一米以内")
    require(all(v > 0 for v in vector(value["base_size"], 3)), "底盘尺寸须为正数")
    require(finite(value["mount_height"]) and value["mount_height"] > value["base_size"][2],
            "机械臂安装高度须高于底盘")
    require(finite(value["mass_kg"]) and value["mass_kg"] > 0, "平台质量无效")
    reason(value["source"])
    require(isinstance(value["approximations"], list) and
            all(isinstance(v, str) for v in value["approximations"]), "平台近似说明无效")


def tables(plan, env):
    if "layout" in plan:
        return plan["layout"]["tables"]
    return [{"id": "main", **env["table"], "position": [0, 0], "yaw": 0,
             "reason": "旧版固定工作台"}]


def table_for(item, plan, env):
    table_id = item.get("table_id", "main")
    return next(t for t in tables(plan, env) if t["id"] == table_id)


def top_prefix(item):
    return ("environment__table_top_" + item["table_id"] + "__"
            if "table_id" in item else "environment__table_top")


def rotate(point, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [c*point[0]-s*point[1], s*point[0]+c*point[1]]


def local_xy(point, table):
    return rotate([point[k]-table["position"][k] for k in range(2)], -table["yaw"])


def world_xy(point, table):
    q = rotate(point, table["yaw"])
    return [q[k]+table["position"][k] for k in range(2)]


def revision(plan):
    return digest({k: plan.get(k) for k in
                   ("instances", "layout", "table_modifications", "edge_mounts", "end_effectors", "robot_posture")})


def validate_layout(plan, env=None):
    """Validate structure, not quality. Out-of-room/colliding proposals go to repair."""
    value = plan.get("layout")
    if value is None:
        require(not env or not mobile(env), "移动场景须提出 layout（房间、多桌、机器人初始位置及停靠候选）")
        require(all("table_id" not in i for i in plan["instances"]), "旧版场景不能引用多桌编号")
        return
    require(isinstance(value, dict) and set(value) == {"room", "tables", "robot_home", "docks"},
            "layout字段须为 room,tables,robot_home,docks")
    room = value["room"]
    require(isinstance(room, dict) and set(room) == {"size", "reason"}, "房间字段无效")
    require(all(0 < v <= 100 for v in vector(room["size"], 2)), "房间尺寸须在0到100米内")
    reason(room["reason"])
    require(isinstance(value["tables"], list) and 1 <= len(value["tables"]) <= 32, "工作台数量须为1到32")
    ids = set()
    for t in value["tables"]:
        require(isinstance(t, dict) and set(t) == {"id", "size", "height", "position", "yaw", "reason"},
                "工作台字段无效")
        name(t["id"])
        require(t["id"] not in ids, "工作台编号重复")
        ids.add(t["id"])
        require(all(0 < v <= 100 for v in vector(t["size"], 3)) and
                t["size"][0] > .24 and t["size"][1] > .24, "工作台尺寸不足以容纳桌腿")
        require(finite(t["height"]) and t["size"][2] < t["height"] <= 10, "桌面高度无效")
        vector(t["position"], 2)
        require(finite(t["yaw"]), "桌子朝向无效")
        reason(t["reason"])
    vector(value["robot_home"], 3)
    instances = {i["id"]: i for i in plan["instances"]}
    for i in instances.values():
        if i["placement"] == "table":
            require(isinstance(i.get("table_id"), str) and i["table_id"] in ids, "桌面实例须引用实际工作台")
        else:
            require("table_id" not in i, "非桌面实例不能指定工作台")
        parent = instances.get(i.get("support_instance"))
        if parent:
            require(i["placement"] == parent["placement"] and i.get("table_id") == parent.get("table_id"),
                    "支撑链必须使用同一地面或工作台")
    require(isinstance(value["docks"], list) and len(value["docks"]) <= 256, "停靠候选数量无效")
    seen = set()
    for d in value["docks"]:
        require(isinstance(d, dict) and set(d) == {"id", "instance", "pose", "reason"}, "停靠候选字段无效")
        name(d["id"])
        require(d["id"] not in seen and isinstance(d["instance"], str) and d["instance"] in instances
                and instances[d["instance"]]["placement"] != "robot", "停靠候选重复或关联对象无效")
        seen.add(d["id"])
        vector(d["pose"], 3)
        reason(d["reason"])
    if env:
        require(mobile(env), "多桌布局须选择移动机器人环境配置")


def edit_layout(value, plan, updated, allowed, env):
    """Atomic table edits, world-coordinate instance overrides, fixed room boundary."""
    if "layout" not in value:
        return
    require("layout" in plan and mobile(env or {}), "当前运行未授权移动多桌布局")
    new = copy.deepcopy(value["layout"])
    # Structural validation after following attachments and explicit placement overrides.
    require(isinstance(new, dict), "布局须为对象")
    if new.get("room") != plan["layout"]["room"]:
        reason(value.get("room_revision_reason"))
        require(all(i["id"] in allowed for i in plan["instances"] if i["placement"]!="robot"),
                "房间边界修改须纳入全部环境实例复核范围")
    require(isinstance(new.get("tables"), list), "工作台列表无效")
    old_tables = {t["id"]: t for t in plan["layout"]["tables"]}
    new_tables = {}
    for t in new["tables"]:
        require(isinstance(t, dict) and isinstance(t.get("id"), str), "工作台编号缺失")
        require(t["id"] not in new_tables, "工作台编号重复")
        new_tables[t["id"]] = t
    changed = {key for key in old_tables.keys() | new_tables.keys()
               if old_tables.get(key) != new_tables.get(key)}
    explicit = {p["id"] for p in value["placements"]}
    for old in plan["instances"]:
        tid = old.get("table_id")
        if tid not in changed:
            continue
        require(old["id"] in allowed, "桌子修改影响未授权实例；请联合申请相关布局范围")
        item = next(i for i in updated["instances"] if i["id"] == old["id"])
        if tid not in new_tables:
            require(old["id"] in explicit and (item.get("table_id") in new_tables
                    or item.get("placement") == "floor" and "table_id" not in item),
                    "删除桌子前须重新安排其所有实例")
            continue
        if old["id"] in explicit:
            continue
        before, after = old_tables[tid], new_tables[tid]
        # Required fields get validated before arithmetic.
        vector(after.get("position"), 2)
        require(finite(after.get("yaw")) and finite(after.get("height")), "桌子变换无效")
        xy = world_xy(local_xy(old["position"], before), after)
        item["position"] = xy + [old["position"][2] + after["height"]-before["height"]]
        item["yaw"] = old["yaw"] + after["yaw"]-before["yaw"]
    updated["layout"] = new


def follow_supports(plan, updated, explicit):
    """Carry descendants once, topologically, preserving relative world transforms."""
    before = {i["id"]: i for i in plan["instances"]}
    after = {i["id"]: i for i in updated["instances"]}
    visited = set()
    def visit(key):
        if key in visited:
            return
        old, item = before[key], after[key]
        parent = old.get("support_instance")
        if parent:
            visit(parent)
            a, b = before[parent], after[parent]
            if key not in explicit and any(a.get(k) != b.get(k) for k in
                                          ("position", "yaw", "placement", "table_id")):
                delta = rotate([old["position"][n]-a["position"][n] for n in range(2)], b["yaw"]-a["yaw"])
                item["position"] = [b["position"][n]+delta[n] for n in range(2)] + [
                    old["position"][2]+b["position"][2]-a["position"][2]]
                item["yaw"] = old["yaw"]+b["yaw"]-a["yaw"]
                item["placement"] = b["placement"]
                if "table_id" in b:
                    item["table_id"] = b["table_id"]
                else:
                    item.pop("table_id", None)
        visited.add(key)
    for key in before:
        visit(key)


def table_geometry(model, data, geometry, table):
    """Recompute actual geometry in the table frame, avoiding twice-expanded boxes."""
    import numpy as np
    from types import SimpleNamespace
    from .geometry import bounds
    c, s = math.cos(table["yaw"]), math.sin(table["yaw"])
    rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    positions = (data.geom_xpos - np.array([*table["position"], 0])) @ rotation.T
    matrices = np.einsum("ij,njk->nik", rotation, data.geom_xmat.reshape(-1, 3, 3))
    local = SimpleNamespace(geom_xpos=positions, geom_xmat=matrices)
    return {g: bounds(model, local, g) for g in geometry}
