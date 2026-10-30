"""Descriptive compactness, not a quality tolerance or automatic packing policy."""
from .layout import local_xy


def union_area(rectangles):
    xs = sorted({x for a,b,c,d in rectangles for x in (a,c)})
    total = 0.
    for left,right in zip(xs,xs[1:]):
        spans = sorted((b,d) for a,b,c,d in rectangles if a < right and c > left)
        end = None
        height = 0.
        for low,high in spans:
            height += max(0., high-max(low,end if end is not None else low))
            end = high if end is None else max(end,high)
        total += (right-left)*height
    return total


def measure(plan, boxes):
    layout = plan.get("layout")
    if not layout: return {}
    rows = []
    for table in layout["tables"]:
        half = [v/2 for v in table["size"][:2]]
        rectangles, present, missing = [], [], []
        for item in plan["instances"]:
            if item.get("table_id") != table["id"]: continue
            geometry = boxes.get(item["id"], [])
            (present if geometry else missing).append(item["id"])
            if not geometry: continue
            # Per-instance conservative world bounds, then table-local bounds.
            lo = [min(b[0][k] for b in geometry) for k in (0,1)]
            hi = [max(b[1][k] for b in geometry) for k in (0,1)]
            corners = [local_xy([x,y],table) for x in (lo[0],hi[0]) for y in (lo[1],hi[1])]
            a,b = [max(-half[k],min(q[k] for q in corners)) for k in (0,1)]
            c,d = [min(half[k],max(q[k] for q in corners)) for k in (0,1)]
            if c>a and d>b: rectangles.append((a,b,c,d))
        area=table["size"][0]*table["size"][1]
        occupied=union_area(rectangles)
        rows.append({"table":table["id"],"area_m2":area,"footprint_upper_bound_m2":occupied,
                     "occupancy_upper_bound":occupied/area,"instances":present,"missing_geometry":missing,
                     "reason":table["reason"],"operation_clearance":"not_quantified"})
    world = [box for item in plan["instances"] for box in boxes.get(item["id"], [])]
    span = ((max(b[1][0] for b in world)-min(b[0][0] for b in world)) *
            (max(b[1][1] for b in world)-min(b[0][1] for b in world))) if world else None
    extra = {"workspace_span_m2":span} if span is not None and all(boxes.get(i["id"]) for i in plan["instances"]) else {}
    return {**extra, "geometry_complete":all(boxes.get(i["id"]) for i in plan["instances"]), "unoccupied_table_lower_bound_m2":sum(r["area_m2"]-r["footprint_upper_bound_m2"] for r in rows),
            "preferred_table_count":[2,3],"table_count":len(rows),"table_area_m2":sum(r["area_m2"] for r in rows),
            "room_area_m2":layout["room"]["size"][0]*layout["room"]["size"][1],"tables":rows,
            "extra_tables_need_review":len(rows)>3,
            "qualification":"几何投影使用保守包围盒并合并重叠；不是精确利用率；未占用投影下界不等于可删除的操作空间。操作/维护留空尚未量化，不能将其当作浪费；桌数目标不是拒绝场景的门槛。"}
