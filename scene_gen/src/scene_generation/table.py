"""Local tabletop machining contract and solid partitioning; the model has no write access to the workbench source."""

from asset_generation.contracts import require, finite


POLICY = {"max_cuts":8, "max_area_fraction":.10, "edge_margin_m":.08,
          "leg_corner_keep_m":.12,
          "robot_protected_half_size_m":.25, "min_pocket_floor_m":.01,
          "max_instance_margin_m":.03,
          "rule":"Only axis-aligned rectangular pockets or through-holes under linked tabletop instances are allowed; overall dimensions, table legs and the robot mounting area are fixed. Do not add bases or inserts, and do not disable collisions."}


def validate_cuts(cuts, plan, env=None):
    require(isinstance(cuts, list) and len(cuts) <= POLICY["max_cuts"], "Too many local table modifications")
    instances = {i["id"]:i for i in plan["instances"]}
    seen = set()
    for cut in cuts:
        require(isinstance(cut, dict) and set(cut) == {"id","instance","center","size","depth","reason"}, "Invalid table modification fields")
        require(isinstance(cut["id"], str) and cut["id"] and len(cut["id"]) <= 48 and cut["id"] not in seen, "Duplicate or invalid table modification id")
        seen.add(cut["id"])
        require(isinstance(cut["instance"], str) and cut["instance"] in instances
                and instances[cut["instance"]]["placement"] == "table", "A table modification must be linked to a tabletop instance")
        for key in ("center", "size"):
            require(isinstance(cut[key], list) and len(cut[key]) == 2 and all(finite(x) for x in cut[key]), "Invalid table machining coordinates")
        require(all(x > .0001 for x in cut["size"]) and finite(cut["depth"]) and cut["depth"] > 0, "Invalid table machining dimensions")
        require(isinstance(cut["reason"], str) and 0 < len(cut["reason"]) <= 4000, "A table modification must state its mounting rationale")
        if env is None:
            continue
        from .layout import table_for
        sx, sy, thickness = table_for(instances[cut["instance"]], plan, env)["size"]
        x, y = cut["center"]; w, h = cut["size"]
        require(abs(x)+w/2 <= sx/2-POLICY["edge_margin_m"] and
                abs(y)+h/2 <= sy/2-POLICY["edge_margin_m"], "Machined region intrudes into the table edge or leg keep-out zone")
        require(not (abs(x)+w/2 > sx/2-POLICY["leg_corner_keep_m"] and
                     abs(y)+h/2 > sy/2-POLICY["leg_corner_keep_m"]), "Machined region intrudes into the table leg mounting keep-out zone")
        keep = POLICY["robot_protected_half_size_m"]
        require("layout" in plan or abs(x)-w/2 >= keep or abs(y)-h/2 >= keep, "Machined region intrudes into the robot mounting keep-out zone")
        require(cut["depth"] <= thickness and (abs(cut["depth"]-thickness) < 1e-9 or
                cut["depth"] <= thickness-POLICY["min_pocket_floor_m"]), "Pocket floor too thin; through-hole depth must equal the tabletop thickness")
    if env is not None:
        from .layout import tables
        for table in tables(plan, env):
            related = [c for c in cuts if instances[c["instance"]].get("table_id", "main") == table["id"]]
            require(sum(c["size"][0]*c["size"][1] for c in related) <=
                    table["size"][0]*table["size"][1]*POLICY["max_area_fraction"], "Total machined area on one table exceeds the limit")
    return cuts


def table_cells(table, cuts):
    """Exact rectangular difference: keeps solid slabs and pocket floors; through-holes provide no spurious support."""
    sx, sy, thickness = table["size"]
    xs = sorted({-sx/2, sx/2, *(c["center"][0]+sign*c["size"][0]/2 for c in cuts for sign in (-1,1))})
    ys = sorted({-sy/2, sy/2, *(c["center"][1]+sign*c["size"][1]/2 for c in cuts for sign in (-1,1))})
    cells = []
    for a,b in zip(xs,xs[1:]):
        for c,d in zip(ys,ys[1:]):
            x,y = (a+b)/2,(c+d)/2
            depth = max([q["depth"] for q in cuts if abs(x-q["center"][0]) < q["size"][0]/2
                         and abs(y-q["center"][1]) < q["size"][1]/2] or [0])
            remaining = thickness-depth
            if remaining > 1e-9:
                cells.append(([x,y,-depth/2],[(b-a)/2,(d-c)/2,remaining/2]))
    return cells
