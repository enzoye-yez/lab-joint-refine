"""Deterministic room arrangement from a semantic layout and the actual asset geometry.

The planning model decides what goes together: which bench an instrument stands on, the order
along it, what stands on the floor and what rests on what. Coordinates come from here, once the
assets exist, so that a mobile robot can reach every instrument and the room reads like a lab:

- benches stand against the long walls in two rows facing one central aisle;
- every instrument stands in one row along its bench, operating face to the aisle, front face a
  few centimetres behind the bench edge, so its controls sit inside the arm's working band;
- floor instruments (refrigerator, cabinet) join the rows; every member stands against the wall
  at its own depth, so a deep instrument deepens only its own bench;
- every instrument gets a dock: the chassis square to the row, just short of the bench edge;
- whatever rests on another instance keeps its pose relative to that support.

Nothing here scales an asset or changes its geometry; arranged plans go through the same checks.
"""
import copy
import math

import numpy as np

from .layout import follow_supports, local_xy, rotate, world_xy

OPERATION_KINDS = {"grasp", "press", "rotate", "insert", "cut"}
SETBACK_M = .04          # instrument front face behind the bench edge
GAP_M = .10              # free space between neighbours (lids, side doors, a gripper's fingers)
BENCH_END_M = .10        # bench end to the first instrument
BACK_M = .08             # behind the deepest instrument (cables, rear lids)
MIN_DEPTH_M = .60
WALL_M = .03
AISLE_M = 1.8            # chassis 0.63 m docked on either side and room to turn (half diagonal 0.44 m)
ROW_END_M = .40
DOCK_STANDOFF_M = .03    # chassis front to the bench edge


def shape(artifact, configuration):
    """Local home-pose bounds of one configuration and its operation targets (position, approach normal)."""
    import mujoco
    from asset_generation.configurations import selection
    from .geometry import bounds
    _, model_file, manifest = selection(artifact, configuration)
    model = mujoco.MjModel.from_xml_path(str(artifact/model_file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boxes = [bounds(model, data, g) for g in range(model.ngeom) if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
    targets = []
    for interface in manifest.get("interfaces", []):
        if interface.get("kind") not in OPERATION_KINDS or not interface.get("site"):
            continue
        site = model.site(interface["site"]).id
        axis = interface.get("approach_axis")
        normal = data.site_xmat[site].reshape(3, 3)@np.asarray(axis, float) if axis else None
        targets.append((data.site_xpos[site].copy(), normal))
    return {"lo": lo, "hi": hi, "targets": targets}


def front(item_shape):
    """Local horizontal unit axis the instrument is operated from.

    Declared approach directions decide; otherwise the side of the bounds nearest the operation
    targets; otherwise local -Y (the authoring convention)."""
    lo, hi, targets = item_shape["lo"], item_shape["hi"], item_shape["targets"]
    normals = [n[:2] for _, n in targets if n is not None and np.linalg.norm(n[:2]) > .3]
    axes = [np.array(a, float) for a in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    if normals:
        mean = np.mean([n/np.linalg.norm(n) for n in normals], axis=0)
        if np.linalg.norm(mean) > .3:
            return max(axes, key=lambda a: float(a@mean))
    if targets:
        centroid = np.mean([p[:2] for p, _ in targets], axis=0)
        gaps = {0: hi[0]-centroid[0], 1: centroid[0]-lo[0], 2: hi[1]-centroid[1], 3: centroid[1]-lo[1]}
        side = min(gaps, key=gaps.get)
        if gaps[side] < .25*min(hi[0]-lo[0], hi[1]-lo[1]):
            return axes[side]
    return axes[3]


def facing(item_shape, direction):
    """Yaw that turns the instrument's front onto the given world direction, with the rotated
    local bounds (x along the row, y toward the back)."""
    f = front(item_shape)
    yaw = math.atan2(direction[1], direction[0])-math.atan2(f[1], f[0])
    yaw = math.remainder(yaw, 2*math.pi)
    corners = [rotate([x, y], yaw) for x in (item_shape["lo"][0], item_shape["hi"][0])
               for y in (item_shape["lo"][1], item_shape["hi"][1])]
    return yaw, np.min(corners, axis=0), np.max(corners, axis=0)


def arrange(plan, shapes, env):
    """A new plan with arranged tables, instances, docks and robot home; semantics unchanged.

    shapes maps instance id -> shape(). Instances without a shape (assets not generated yet)
    and tables holding edge-mounted fixtures keep their planned poses relative to their table."""
    if "layout" not in plan:
        return plan
    updated = copy.deepcopy(plan)
    layout = updated["layout"]
    items = {i["id"]: i for i in updated["instances"]}
    old_tables = {t["id"]: t for t in plan["layout"]["tables"]}
    mounted = {items[m["instance"]].get("table_id") for m in plan.get("edge_mounts", []) if m["instance"] in items}
    roots = [i for i in updated["instances"] if i["placement"] in {"table", "floor"} and not i.get("support_instance")]
    # Members of a row: benches with their instruments, and free-standing floor instruments.
    members = []
    for table in layout["tables"]:
        on = sorted((i for i in roots if i.get("table_id") == table["id"]),
                    key=lambda i: local_xy(i["position"], old_tables[table["id"]])[0])
        if on:
            members.append({"kind": "table", "table": table, "roots": on})
    for item in roots:
        if item["placement"] == "floor":
            members.append({"kind": "floor", "item": item})
    if not members:
        return plan
    # Bench-local packing: front to -Y, one row along X.
    for member in members:
        if member["kind"] == "floor":
            item = member["item"]
            s = shapes.get(item["id"])
            if s is None:
                member.update(length=.6, depth=.6, rigid=True)
                continue
            yaw, lo, hi = facing(s, (0, -1))
            member.update(yaw=yaw, lo=lo, hi=hi, length=float(hi[0]-lo[0]), depth=float(hi[1]-lo[1]))
            continue
        table = member["table"]
        if table["id"] in mounted or any(i["id"] not in shapes for i in member["roots"]):
            member.update(length=table["size"][0], depth=table["size"][1], rigid=True)
            continue
        slots = []
        for item in member["roots"]:
            yaw, lo, hi = facing(shapes[item["id"]], (0, -1))
            slots.append((item, yaw, lo, hi))
        member["slots"] = slots
        member["length"] = 2*BENCH_END_M+sum(float(hi[0]-lo[0]) for _, _, lo, hi in slots)+GAP_M*(len(slots)-1)
        member["depth"] = max(MIN_DEPTH_M, max(float(hi[1]-lo[1]) for _, _, lo, hi in slots)+SETBACK_M+BACK_M)
        member["depth"] = math.ceil(member["depth"]/.05-1e-9)*.05
    rows = split_rows(members)
    depths = [max((m["depth"] for m in row), default=MIN_DEPTH_M) for row in rows]
    lengths = [sum(m["length"] for m in row)+GAP_M*max(0, len(row)-1) for row in rows]
    room = [max(lengths)+2*ROW_END_M, 2*WALL_M+depths[0]+AISLE_M+depths[1]]
    old_room = plan["layout"]["room"]["size"]
    layout["room"] = {"size": [round(v, 3) for v in room],
                      "reason": "Two bench rows against the long walls facing one %.1f m robot aisle; "
                                "length from the instruments in a single row along each bench" % AISLE_M}
    walls = [room[1]/2-WALL_M, -room[1]/2+WALL_M]
    front_lines = [walls[0]-depths[0], walls[1]+depths[1]]
    docks = []
    for side, row in enumerate(rows):
        # Row frame: origin on the front line, local -Y toward the aisle, local +Y into the wall.
        # North row faces -Y (yaw 0); south row faces +Y (yaw pi).
        yaw_row = 0. if side == 0 else math.pi
        inward = np.array(rotate([0, 1], yaw_row))
        cursor = -lengths[side]/2
        for member in row:
            centre_x = cursor+member["length"]/2
            cursor += member["length"]+GAP_M
            # Every member stands against the wall at its own depth.
            front_point = np.array([centre_x, walls[side]])-inward*member["depth"]
            if member["kind"] == "table":
                table, depth = member["table"], member["depth"]
                old = old_tables[table["id"]]
                centre = front_point+inward*depth/2
                table.update(size=[round(member["length"], 3), round(depth, 3), table["size"][2]],
                             position=[round(float(centre[0]), 4), round(float(centre[1]), 4)], yaw=yaw_row)
                if member.get("rigid"):
                    for item in member["roots"]:
                        item["position"] = world_xy(local_xy(item["position"], old), table)+[item["position"][2]]
                        item["yaw"] = item["yaw"]-old["yaw"]+yaw_row
                    continue
                x = -member["length"]/2+BENCH_END_M
                for item, yaw, lo, hi in member["slots"]:
                    local = [x-float(lo[0]), -depth/2+SETBACK_M-float(lo[1])]
                    x += float(hi[0]-lo[0])+GAP_M
                    item["position"] = world_xy(local, table)+[item["position"][2]]
                    item["yaw"] = round(math.remainder(yaw+yaw_row, 2*math.pi), 6)
                    docks.append(dock(item, shapes[item["id"]], front_point, inward, env))
                continue
            item = member["item"]
            frame = {"position": front_point.tolist(), "yaw": yaw_row}
            if member.get("rigid"):
                item["position"] = world_xy([0, .3], frame)+[item["position"][2]]
                continue
            local = [-float(member["lo"][0]+member["hi"][0])/2, -float(member["lo"][1])]
            item["position"] = world_xy(local, frame)+[item["position"][2]]
            item["yaw"] = round(math.remainder(member["yaw"]+yaw_row, 2*math.pi), 6)
            docks.append(dock(item, shapes[item["id"]], front_point, inward, env))
    layout["tables"] = [m["table"] for row in rows for m in row if m["kind"] == "table"]
    layout["robot_home"] = [0., round((front_lines[0]+front_lines[1])/2, 4), 0.]  # aisle centre
    layout["docks"] = docks
    move_cuts(plan, updated)
    follow_supports(plan, updated, {i["id"] for i in roots})
    updated["arrangement"] = {"method": "deterministic_rows", "aisle_m": AISLE_M, "setback_m": SETBACK_M,
                              "gap_m": GAP_M, "previous_room_m": old_room}
    return updated


def split_rows(members):
    """Two rows of similar length, keeping the planner's order (north first)."""
    total = sum(m["length"] for m in members)
    north, south, run = [], [], 0.
    for member in members:
        (north if run+member["length"]/2 <= total/2 or not north else south).append(member)
        if member in north:
            run += member["length"]
    return [north, south]


def dock(item, item_shape, front_point, inward, env):
    """Chassis square to the row, its front DOCK_STANDOFF_M short of the front line, centred
    on the operation targets (or the instrument) along the row."""
    yaw = item["yaw"]
    origin = np.array(item["position"][:2])
    targets = [origin+rotate(p[:2], yaw) for p, _ in item_shape["targets"]]
    centre = np.mean(targets, axis=0) if targets else origin
    on_line = centre-inward*float((centre-front_point)@inward)
    base = on_line-inward*(DOCK_STANDOFF_M+env["robot"]["mobile"]["base_size"][0]/2)
    return {"id": ("dock_"+item["id"])[:48], "instance": item["id"],
            "pose": [round(float(base[0]), 4), round(float(base[1]), 4), round(math.atan2(inward[1], inward[0]), 6)],
            "reason": "Chassis square to the row, %.0f mm short of the edge, centred on the operation targets"
                      % (DOCK_STANDOFF_M*1000)}


def move_cuts(plan, updated):
    """Table cuts follow their instance in the table frame."""
    before = {i["id"]: i for i in plan["instances"]}
    after = {i["id"]: i for i in updated["instances"]}
    old_tables = {t["id"]: t for t in plan["layout"]["tables"]}
    new_tables = {t["id"]: t for t in updated["layout"]["tables"]}
    for cut in updated.get("table_modifications", []):
        a, b = before[cut["instance"]], after[cut["instance"]]
        t0, t1 = old_tables[a["table_id"]], new_tables[b["table_id"]]
        o0, o1 = local_xy(a["position"], t0), local_xy(b["position"], t1)
        turn = (b["yaw"]-t1["yaw"])-(a["yaw"]-t0["yaw"])
        relative = rotate([cut["center"][k]-o0[k] for k in range(2)], turn)
        cut["center"] = [round(o1[k]+relative[k], 4) for k in range(2)]


_SHAPES = {}


def shapes_for(run, plan):
    """Shapes of every placed instance whose asset is usable, cached by artifact and configuration."""
    from asset_prep.storage import safe_path
    from .candidate import usable
    out = {}
    for item in plan["instances"]:
        record = run.state["jobs"].get(item["job"])
        if item["placement"] == "robot" or not usable(run, record):
            continue
        artifact = safe_path(run.root, record["artifact"])
        key = (str(artifact), (record.get("assembly_readiness") or {}).get("artifact_sha256"), item.get("configuration"))
        if key not in _SHAPES:
            _SHAPES[key] = shape(artifact, item.get("configuration"))
        out[item["id"]] = _SHAPES[key]
    return out
