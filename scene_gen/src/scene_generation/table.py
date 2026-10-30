"""桌面局部加工契约及实体分块；模型没有实验台源码写权限。"""

from asset_generation.contracts import require, finite


POLICY = {"max_cuts":8, "max_area_fraction":.10, "edge_margin_m":.08,
          "leg_corner_keep_m":.12,
          "robot_protected_half_size_m":.25, "min_pocket_floor_m":.01,
          "max_instance_margin_m":.03,
          "rule":"只允许关联桌面实例下方的轴对齐矩形凹槽或通孔；整体尺寸、桌腿和机器人安装区固定。新增底座和嵌件由fixture资产生成，不关闭碰撞。"}


def validate_cuts(cuts, plan, env=None):
    require(isinstance(cuts, list) and len(cuts) <= POLICY["max_cuts"], "桌面局部修改数量超限")
    instances = {i["id"]:i for i in plan["instances"]}
    seen = set()
    for cut in cuts:
        require(isinstance(cut, dict) and set(cut) == {"id","instance","center","size","depth","reason"}, "桌面修改字段无效")
        require(isinstance(cut["id"], str) and cut["id"] and len(cut["id"]) <= 48 and cut["id"] not in seen, "桌面修改编号重复或无效")
        seen.add(cut["id"])
        require(isinstance(cut["instance"], str) and cut["instance"] in instances
                and instances[cut["instance"]]["placement"] == "table", "桌面修改须关联桌面实例")
        for key in ("center", "size"):
            require(isinstance(cut[key], list) and len(cut[key]) == 2 and all(finite(x) for x in cut[key]), "桌面加工坐标无效")
        require(all(x > .0001 for x in cut["size"]) and finite(cut["depth"]) and cut["depth"] > 0, "桌面加工尺寸无效")
        require(isinstance(cut["reason"], str) and 0 < len(cut["reason"]) <= 4000, "桌面修改须说明安装依据")
        if env is None:
            continue
        from .layout import table_for
        sx, sy, thickness = table_for(instances[cut["instance"]], plan, env)["size"]
        x, y = cut["center"]; w, h = cut["size"]
        require(abs(x)+w/2 <= sx/2-POLICY["edge_margin_m"] and
                abs(y)+h/2 <= sy/2-POLICY["edge_margin_m"], "加工区域侵入桌边或桌腿保护区")
        require(not (abs(x)+w/2 > sx/2-POLICY["leg_corner_keep_m"] and
                     abs(y)+h/2 > sy/2-POLICY["leg_corner_keep_m"]), "加工区域侵入桌腿安装保护区")
        keep = POLICY["robot_protected_half_size_m"]
        require("layout" in plan or abs(x)-w/2 >= keep or abs(y)-h/2 >= keep, "加工区域侵入机器人安装保护区")
        require(cut["depth"] <= thickness and (abs(cut["depth"]-thickness) < 1e-9 or
                cut["depth"] <= thickness-POLICY["min_pocket_floor_m"]), "凹槽底板过薄；通孔深度须等于桌面厚度")
    if env is not None:
        from .layout import tables
        for table in tables(plan, env):
            related = [c for c in cuts if instances[c["instance"]].get("table_id", "main") == table["id"]]
            require(sum(c["size"][0]*c["size"][1] for c in related) <=
                    table["size"][0]*table["size"][1]*POLICY["max_area_fraction"], "单桌加工总面积超限")
    return cuts


def table_cells(table, cuts):
    """精确矩形差集：保留实体板块及凹槽底板，通孔处没有虚假支撑。"""
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
