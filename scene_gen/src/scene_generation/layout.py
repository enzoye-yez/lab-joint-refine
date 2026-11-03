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
            "Layout coordinates must be arrays of finite numbers")
    return value


def name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", value),
            "Invalid layout id")


def reason(value):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 4000, "A layout change needs an explicit reason")


def mobile(env):
    return bool(env.get("robot", {}).get("mobile"))


def table_policy(env):
    from .table import POLICY
    if not mobile(env):
        return POLICY
    return {**POLICY, "robot_protected_half_size_m":0,
            "rule":"Machining area is computed per worktable; center is in the associated worktable's local frame. The mobile robot does not occupy a protected central tabletop zone; "
                   "table-edge, table-leg and recess floor-thickness requirements still apply. Overall worktable size and count are designed by layout; instruments are not scaled."}


def author_layout_policy(env):
    coordinates = ("local coordinates of the worktable named by table_id" if mobile(env) else "world coordinates")
    setting = ("Design against the actual multi-worktable and mobile-robot layout in scene_layout; there is no protected fixed-arm zone at the tabletop center. "
               if mobile(env) else "The tabletop as a whole and the robot-arm mounting zone are fixed. ")
    return (setting + "Local slots and holes may be cut per table_edit_policy. Prefer compatibility with existing mounting conditions; if new machining is truly needed, state in assumptions the "
            + coordinates + ", size, depth and support method, so the scene coordinator can declare table_modifications. "
            "Do not assume undeclared recesses exist; do not disable collisions. Asset authors may not change the scene layout or scale dependency instruments on their own.")


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
            "Mobile platform must state size, mounting height, mass, source and approximations")
    require(type(value.get("integrated", False)) is bool, "Integrated-robot flag must be a boolean")
    require(finite(value.get("ground_offset_m",0)) and abs(value.get("ground_offset_m",0))<=1,
            "Integrated-robot ground offset must come from resource measurement and be within one meter")
    require(all(v > 0 for v in vector(value["base_size"], 3)), "Base size must be positive")
    require(finite(value["mount_height"]) and value["mount_height"] > value["base_size"][2],
            "Robot-arm mounting height must be above the base")
    require(finite(value["mass_kg"]) and value["mass_kg"] > 0, "Invalid platform mass")
    reason(value["source"])
    require(isinstance(value["approximations"], list) and
            all(isinstance(v, str) for v in value["approximations"]), "Invalid platform approximation notes")


def tables(plan, env):
    if "layout" in plan:
        return plan["layout"]["tables"]
    return [{"id": "main", **env["table"], "position": [0, 0], "yaw": 0,
             "reason": "Legacy fixed worktable"}]


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
        require(not env or not mobile(env), "A mobile scene must propose layout (room, multiple tables, robot home and dock candidates)")
        require(all("table_id" not in i for i in plan["instances"]), "A legacy scene cannot reference multi-table ids")
        return
    require(isinstance(value, dict) and set(value) == {"room", "tables", "robot_home", "docks"},
            "layout fields must be room,tables,robot_home,docks")
    room = value["room"]
    require(isinstance(room, dict) and set(room) == {"size", "reason"}, "Invalid room fields")
    require(all(0 < v <= 100 for v in vector(room["size"], 2)), "Room size must be within 0 to 100 m")
    reason(room["reason"])
    require(isinstance(value["tables"], list) and 1 <= len(value["tables"]) <= 32, "Worktable count must be 1 to 32")
    ids = set()
    for t in value["tables"]:
        require(isinstance(t, dict) and set(t) == {"id", "size", "height", "position", "yaw", "reason"},
                "Invalid worktable fields")
        name(t["id"])
        require(t["id"] not in ids, "Duplicate worktable id")
        ids.add(t["id"])
        require(all(0 < v <= 100 for v in vector(t["size"], 3)) and
                t["size"][0] > .24 and t["size"][1] > .24, "Worktable too small to fit table legs")
        require(finite(t["height"]) and t["size"][2] < t["height"] <= 10, "Invalid tabletop height")
        vector(t["position"], 2)
        require(finite(t["yaw"]), "Invalid table yaw")
        reason(t["reason"])
    vector(value["robot_home"], 3)
    instances = {i["id"]: i for i in plan["instances"]}
    for i in instances.values():
        if i["placement"] == "table":
            require(isinstance(i.get("table_id"), str) and i["table_id"] in ids, "A tabletop instance must reference an existing worktable")
        else:
            require("table_id" not in i, "A non-tabletop instance cannot specify a worktable")
        parent = instances.get(i.get("support_instance"))
        if parent:
            require(i["placement"] == parent["placement"] and i.get("table_id") == parent.get("table_id"),
                    "A support chain must use the same floor or worktable")
    require(isinstance(value["docks"], list) and len(value["docks"]) <= 256, "Invalid number of dock candidates")
    seen = set()
    for d in value["docks"]:
        require(isinstance(d, dict) and set(d) == {"id", "instance", "pose", "reason"}, "Invalid dock candidate fields")
        name(d["id"])
        require(d["id"] not in seen and isinstance(d["instance"], str) and d["instance"] in instances
                and instances[d["instance"]]["placement"] != "robot", "Duplicate dock candidate or invalid associated object")
        seen.add(d["id"])
        vector(d["pose"], 3)
        reason(d["reason"])
    if env:
        require(mobile(env), "A multi-table layout requires a mobile-robot environment configuration")


def edit_layout(value, plan, updated, allowed, env):
    """Atomic table edits, world-coordinate instance overrides, fixed room boundary."""
    if "layout" not in value:
        return
    require("layout" in plan and mobile(env or {}), "The current run does not authorize a mobile multi-table layout")
    new = copy.deepcopy(value["layout"])
    # Structural validation after following attachments and explicit placement overrides.
    require(isinstance(new, dict), "Layout must be an object")
    if new.get("room") != plan["layout"]["room"]:
        reason(value.get("room_revision_reason"))
        require(all(i["id"] in allowed for i in plan["instances"] if i["placement"]!="robot"),
                "A room-boundary change must include all environment instances in the recheck scope")
    require(isinstance(new.get("tables"), list), "Invalid worktable list")
    old_tables = {t["id"]: t for t in plan["layout"]["tables"]}
    new_tables = {}
    for t in new["tables"]:
        require(isinstance(t, dict) and isinstance(t.get("id"), str), "Missing worktable id")
        require(t["id"] not in new_tables, "Duplicate worktable id")
        new_tables[t["id"]] = t
    changed = {key for key in old_tables.keys() | new_tables.keys()
               if old_tables.get(key) != new_tables.get(key)}
    explicit = {p["id"] for p in value["placements"]}
    for old in plan["instances"]:
        tid = old.get("table_id")
        if tid not in changed:
            continue
        require(old["id"] in allowed, "Table change affects unauthorized instances; request the related layout scope jointly")
        item = next(i for i in updated["instances"] if i["id"] == old["id"])
        if tid not in new_tables:
            require(old["id"] in explicit and (item.get("table_id") in new_tables
                    or item.get("placement") == "floor" and "table_id" not in item),
                    "All instances on a table must be rearranged before the table is removed")
            continue
        if old["id"] in explicit:
            continue
        before, after = old_tables[tid], new_tables[tid]
        # Required fields get validated before arithmetic.
        vector(after.get("position"), 2)
        require(finite(after.get("yaw")) and finite(after.get("height")), "Invalid table transform")
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
