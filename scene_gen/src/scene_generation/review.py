"""阶段审查契约：允许获取后续证据，但不得把待验证项记为通过。"""

from asset_generation.contracts import require, visual_result
from asset_prep.errors import ValidationError

ASSEMBLABLE = {"candidate_ready", "assembly_ready"}

REQUIREMENTS_POLICY = '''
要求来源必须区分：原文、资料包及只读fixed_contract是用户输入的约束；job.requirements及plan由规划模型生成。
规划模型提出的暂定位置、间距、高度、占地等属于设计假设，不自动升级为用户硬规格。
当资料包明确允许协同确定尺寸、或计划称为暂定/建模布局值时，允许依据实际依赖几何调整，
在清单中记录依据与新接口即可；不需要人工批准模型自己的设计值变更。
审查不能仅因偏离这样的暂定数值要求repair/uncertain，必须指出原始资料或fixed_contract中的明确冲突。
同时不得借设计自由度改动资料已固定的仪器尺寸、缩放依赖模型、伪造测量或豁免工程检查。
数值/术语有歧义时结合原始上下文和依赖的实际局部坐标定义，不把管实例原点与法兰承托面混为一谈。
'''

ASSET_SCOPE = '''
当前阶段是单资产/装置的本体审查：外形、本体内部配合与全部本体运动机构都必须在本阶段判定完毕，不能延后。
verdict只能是accept或repair。本体问题（缺件、外形或机构错误、可见穿模或悬空、与资料冲突、本体内部配合不当、
机构缺失或运动不合理）一律repair。只有本资产内部已全部核对无误才accept。
跨实例事项（与其他资产的插接、放入其他仪器、桌面支撑与整体遮挡）不属于本阶段，写入
deferred:[{"stage":"scene|operation","detail":"具体待验证内容"}]（每项只有这两个键）：跨实例配合用stage="scene"，
机器人抓取、操作序列与轨迹用stage="operation"。这些记录可与accept共存，交由对应阶段完成，
但不能把本体缺陷写进deferred。
必须返回motion_coverage:[{"requirement":"资料中的一条运动/交互要求原文","mechanism":"interfaces.json中实现它的机构id或null",
"verdict":"ok|missing|wrong","evidence":"依据：机构扫描/驱动/锁止/释放检查编号与机构两端渲染图"}]，
逐条覆盖资料Required interactions与任务requirements中属于本资产本体的全部运动要求（打开/关闭、滑动、旋转、摆动、按压、锁止/解锁、取下/装回等）。
任何missing或wrong都必须repair并写入findings。数值检查中mechanism_*、sampled_joint_sweep:*、release_probe:*、numerical_stability为fail时必须repair。
未绑定精确数值检查的尺寸列入unverified，不能仅凭图像认定精确合规。
reference_policy与evidence明确说明已传入哪些图片，不要把依赖单独渲染当作组合装配图。
只有图片无法打开或资料自相矛盾导致完全无法判断时才用uncertain，并增加blocking:["具体原因"]。
保持verdict/findings/unverified三个基础字段；motion_coverage必填；blocking、deferred为按需附加字段。
如果instance_contract.expected_model_units非空，必须增加prototype_units字段，按当前实际几何/渲染报告单个模型内的完整资产单元数量。
零件、盖、内部机构不是独立资产单元。多支完整管是多个单元；场景负责复制，不能在每个模型内再次复制整套数量。
单模型数量不匹配必须repair；图像或几何证据不足无法计数则prototype_units=null并uncertain及blocking，不能默认满足。
'''

SCENE_SCOPE = '''
当前是完整候选场景的审查，应处理deferred_checks中stage="scene"的全部事项。
增加deferred_results:[{"id":"原样复制待验证编号","status":"resolved|unresolved","evidence":"依据具体渲染、检查条目；不足时说明原因"}]。
每个scene事项须恰好回答一次。只有全部有证据解决且本轮工程检查通过，才可accept。
仍有具体缺陷用repair，增加affected_instances:["实际实例编号"]，明确列出需要调整或修复的实例。
布局本身的缺陷使用affected_layout:["robot_home"|"table:实际工作台编号"|"dock:实际停靠编号"]，
仅可引用layout_review_targets列出的对象。此字段授权布局协调，不授权修改机器人或仪器几何。
repair至少给出一个实例或布局对象；其他结论两个列表都应为空。
只能使用plan.instances中的完整id，不使用资产编号或子串；findings说明缺陷，不从文字猜测修复权限。
accept/uncertain时affected_instances应省略或为空。证据仍不足用uncertain并给blocking。
不可仅凭图片证明精确尺寸、被跳过的动力学、完整操作和轨迹。stage="operation"的事项保留为未验证，不得伪称已解决，且不阻塞当前静态场景。
本阶段不能再把scene事项延后到scene，也不能删除或改名逃避检查。
例外仅适用于review_scope.validation_mode="static"：本轮交付的是当前初始几何布局预览。
当前可见缺件、资产数量错误、错误外形、穿透、悬空和明确静态尺寸冲突必须处理，不能以例外跳过。
需要配置规程初始关节/内容物状态、切换为其他装配姿态、接触位置或执行操作才能证明的事项，
可在deferred_results中用status="out_of_scope"，增加target_stage="initial_state|static_fit|operation"，并明确evidence说明尚缺什么。
这些项保留为未验证，绝不标为resolved。只要当前布局外观合格、工程检查没有fail且剩余项都已如此明确归属，可accept交付静态预览。
工程检查中任何status="fail"或"partial"的条目都必须处理：能修复的repair；确属后续阶段的项按上面规则归属并说明。
不要要求先装管入仪器、抓取、拆盖停放或配置内容物才能查看当前静态布局；完整模式不允许此例外。
'''


def scoped_review(value, stage, deferred=(), expected_units=None, preview=False, instance_ids=None, layout_targets=()):
    try:
        return _scoped_review(value,stage,deferred,expected_units,preview,instance_ids,layout_targets)
    except ValidationError as exc:
        if stage=="scene" and isinstance(value,dict):
            expected={d["id"] for d in deferred if d["stage"]=="scene"}
            accepted,errors=resolution_feedback(value.get("deferred_results",[]),expected,stage,preview)
            exc.accepted_results=list(accepted.values())
            exc.review_feedback={"invalid":errors,"missing_ids":sorted(expected-set(accepted)),"retained_ids":sorted(accepted)}
        raise


def _scoped_review(value, stage, deferred=(), expected_units=None, preview=False, instance_ids=None, layout_targets=()):
    require(isinstance(value, dict), "阶段审查须为对象")
    required = {"verdict", "findings", "unverified"}
    allowed = required | {"deferred", "blocking", "deferred_results", "prototype_units", "motion_coverage"}
    if stage == "scene":
        allowed.update({"affected_instances", "affected_layout"})
    require(required <= set(value) <= allowed, "阶段审查字段无效：缺少="+str(sorted(required-set(value)))+"；未知="+str(sorted(set(value)-allowed)))
    require(stage in {"asset", "scene"} and isinstance(value["verdict"], str), "审查阶段或结论无效")
    require(value["verdict"] != "defer", "资产与场景审查不再使用defer：本体问题repair，跨实例事项写入deferred并给出accept或repair")
    base = {k: value[k] for k in required}
    visual_result(base)  # 缺陷列表不能与accept共存。
    if stage == "asset":
        motion_coverage(value)
    if expected_units is not None:
        units = value.get("prototype_units")
        # A multi-configuration model may report one count per configuration.
        counts = list(units.values()) if isinstance(units, dict) and units else [units]
        require("prototype_units" in value and (all(type(u) is int and u > 0 for u in counts)
                                                or units is None and value["verdict"] == "uncertain"),
                "Repeated-instance jobs must report prototype_units: the number of complete asset units in one model "
                "(an integer, or {configuration: integer})")
        require(value["verdict"] in {"repair", "uncertain"} or all(u == expected_units for u in counts),
                "The number of asset units in one model does not match; cannot accept")
    pending = value.get("deferred", [])
    require(isinstance(pending, list) and len(pending) <= 64, "待验证项必须为有界列表")
    pending = value["deferred"] = [deferred_item(item) for item in pending]
    for item in pending:
        require(set(item) == {"stage", "detail"}
                and isinstance(item["stage"], str) and item["stage"] in {"scene", "operation"}
                and isinstance(item["detail"], str) and 0 < len(item["detail"].strip()) <= 4000,
                '待验证项格式应为{"stage":"scene|operation","detail":"具体内容"}，收到：'+str(item)[:200])
    require(not pending or stage == "asset", "场景阶段不能再次延后事项")
    blocking = value.get("blocking", [])
    require(isinstance(blocking, list) and all(isinstance(x, str) and x.strip() for x in blocking), "阻断原因须为文本列表")
    require(value["verdict"] != "uncertain" or blocking,
            "uncertain须明确本阶段blocking原因；跨实例事项写入deferred，操作未验证写入unverified")
    require(value["verdict"] == "uncertain" or not blocking, "非uncertain结论不能包含阻断原因")
    require(value["verdict"] != "repair" or value["findings"], "repair必须列出本阶段具体缺陷")
    if stage == "asset":
        # The author edits from these lines alone: each must name the part, the evidence and the change.
        vague = [f for f in value["findings"] if not ("：" in f and "依据" in f and "建议" in f)]
        require(not vague, "每条finding须写成“部件/几何名：问题；依据（哪张图或哪条检查）；建议修改（方向、数值或结构）”，不合格："+str(vague[:3])[:600])
    if stage == "scene":
        affected = value.get("affected_instances",[])
        require(isinstance(affected,list) and len(affected) <= 128 and all(isinstance(i,str) for i in affected)
                and len(affected) == len(set(affected)), "修复实例编号须为不重复的有界列表")
        layout = value.get("affected_layout", [])
        require(isinstance(layout,list) and len(layout)<=289 and all(isinstance(i,str) for i in layout)
                and len(layout)==len(set(layout)) and set(layout)<=set(layout_targets), "布局修复引用未知或重复对象")
        require(value["verdict"] != "repair" or affected or layout, "场景修复必须给出affected_instances或affected_layout")
        require(value["verdict"] == "repair" or not layout, "非修复结论不能申请布局修改权限")
        require(value["verdict"] == "repair" or not affected, "非修复结论不能申请实例修改权限")
        require(not affected or instance_ids is not None and set(affected) <= set(instance_ids), "修复引用不存在的实例编号")
    resolutions = value.get("deferred_results", [])
    expected = {d["id"] for d in deferred if d["stage"] == "scene"}
    accepted, errors = resolution_feedback(resolutions, expected, stage, preview)
    missing = sorted(expected - set(accepted)) if stage == "scene" else []
    if errors or missing:
        raise ValidationError("待验证处理结果错误：" + str({"invalid":errors,"missing_ids":missing}))
    require(value["verdict"] != "accept" or all(r["status"] in {"resolved","out_of_scope"} for r in resolutions),
            "仍有未解决的场景事项，不能接受")
    return value


def deferred_item(item):
    """One deferred item as {stage, detail}; a quoted requirement is folded into the detail text."""
    if not isinstance(item, dict) or not set(item) <= {"stage", "detail", "requirement"}:
        return item
    parts = [item[k].strip() for k in ("requirement", "detail") if isinstance(item.get(k), str) and item[k].strip()]
    return {"stage": item.get("stage"), "detail": "：".join(parts)}


def motion_coverage(value):
    """Per-requirement motion verdicts; any gap forces repair with a finding."""
    rows = value.get("motion_coverage")
    if rows is None and value["verdict"] != "accept":
        return []
    require(isinstance(rows, list) and len(rows) <= 128, "资产审查accept时必须返回motion_coverage列表（无运动要求时为空列表）")
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"requirement", "mechanism", "verdict", "evidence"}
                and isinstance(row["requirement"], str) and row["requirement"].strip()
                and (row["mechanism"] is None or isinstance(row["mechanism"], str))
                and row["verdict"] in {"ok", "missing", "wrong"} and isinstance(row["evidence"], str) and row["evidence"].strip(),
                "motion_coverage条目须含requirement、mechanism、verdict(ok|missing|wrong)与evidence")
    bad = [r for r in rows if r["verdict"] != "ok"]
    require(not bad or (value["verdict"] == "repair" and value["findings"]),
            "存在缺失或错误的运动机构时必须repair并写入findings")
    return rows


def deferred_records(review, job, version):
    return [{"id": f"{job}:{version}:{i}", "job": job, "stage": item["stage"],
             "detail": item["detail"], "status": "pending"}
            for i, item in enumerate(review.get("deferred", []), 1)]


def resolution_feedback(resolutions, expected, stage="scene", preview=False):
    """Validate independent answers; an ambiguous duplicate is never retained."""
    if not isinstance(resolutions,list):
        return {}, [{"field":"deferred_results","error":"必须为列表"}]
    accepted, errors, seen = {}, [], set()
    for index, row in enumerate(resolutions):
        key = row.get("id") if isinstance(row,dict) else None
        try:
            require(stage=="scene" and isinstance(row,dict), "只有场景审查允许事项处理结果")
            require({"id","status","evidence"} <= set(row) <= {"id","status","evidence","target_stage"}, "事项字段应为id/status/evidence及按需target_stage")
            require(isinstance(key,str) and key in expected, "未知事项编号；只能使用本批待验证编号")
            require(key not in seen, "同一响应重复编号；须重新给出唯一结论")
            require(row["status"] in ("resolved","unresolved","out_of_scope"), "状态只能为resolved/unresolved/out_of_scope")
            require(isinstance(row["evidence"],str) and row["evidence"].strip(), "必须给出具体证据或未解决原因")
            if row["status"]=="out_of_scope":
                require(preview and row.get("target_stage") in ("initial_state","static_fit","operation"),
                        "仅静态范围允许明确移交initial_state/static_fit/operation，不能标为已解决")
            else:
                require("target_stage" not in row, "只有out_of_scope才允许target_stage")
            accepted[key]=row
        except ValidationError as exc:
            if isinstance(key,str): accepted.pop(key,None)
            errors.append({"index":index,"id":key,"error":str(exc)})
        if isinstance(key,str): seen.add(key)
    return accepted, errors
