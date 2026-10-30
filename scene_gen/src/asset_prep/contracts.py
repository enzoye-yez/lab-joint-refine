"""固定校验器：验证证据引用和结构，不让生成模型修改检查规则。"""

import re

from .errors import ValidationError


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(obj, names):
    require(isinstance(obj, dict), "期望对象")
    require(set(obj) == set(names.split()), f"字段必须为：{names}")


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "期望非空文本")


def texts(values):
    require(isinstance(values, list), "期望列表")
    for value in values:
        text(value)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value), "标识只能使用小写字母、数字和下划线")


def evidence(refs, sources, *, required=False, kind=None):
    require(isinstance(refs, list) and (bool(refs) or not required), "缺少证据列表")
    for ref in refs:
        fields(ref, "source_id quote")
        text(ref["quote"])
        require(ref["source_id"] in sources, "引用未知文档来源")
        source = sources[ref["source_id"]]
        require(ref["quote"] in source["text"], "引用原文未出现在指定来源中")
        if kind:
            require(source["kind"] == kind, "冲突引用来源类型错误")


def origin(obj, sources):
    require(obj["origin"] in {"explicit", "inferred"}, "依据类型必须为 explicit 或 inferred")
    text(obj["basis"])
    evidence(obj["evidence"], sources, required=obj["origin"] == "explicit")


def validate_inventory(value, sources, max_assets=64, *, allow_empty=False):
    sources = {s["id"]: s for s in sources}
    fields(value, "assets steps relations conflicts assumptions issues")
    require(isinstance(value["assets"], list) and (bool(value["assets"]) or allow_empty)
            and len(value["assets"]) <= max_assets, "资产数量为空或超过上限")
    ids = set()
    for asset in value["assets"]:
        fields(asset, "id name category aliases quantity requirements operations evidence parent_id queries")
        identifier(asset["id"])
        require(asset["id"] not in ids, "资产标识重复")
        ids.add(asset["id"])
        text(asset["name"])
        require(asset["category"] in {"instrument", "container", "consumable", "accessory"}, "不允许试剂或未支持的资产类别")
        for key in ("aliases", "operations", "queries"):
            texts(asset[key])
        require(bool(asset["queries"]), "资产缺少检索词")
        quantity = asset["quantity"]
        fields(quantity, "value origin basis evidence")
        require(type(quantity["value"]) is int and 1 <= quantity["value"] <= 100000, "数量须为 1 至 100000 的整数")
        origin(quantity, sources)
        evidence(asset["evidence"], sources, required=True)
        require(isinstance(asset["requirements"], list), "要求须为列表")
        keys = set()
        for item in asset["requirements"]:
            fields(item, "key value unit origin basis evidence")
            text(item["key"])
            require(item["key"] not in keys, "同一资产要求名称重复")
            keys.add(item["key"])
            text(item["value"])
            require(isinstance(item["unit"], str), "单位须为文本，可为空")
            origin(item, sources)
    parents = {a["id"]: a["parent_id"] for a in value["assets"]}
    for asset in value["assets"]:
        parent = asset["parent_id"]
        require(parent is None or parent in ids, "父资产不存在")
        seen = {asset["id"]}
        while parent is not None:
            require(parent not in seen, "部件关系形成环")
            seen.add(parent)
            parent = parents[parent]
    require(isinstance(value["steps"], list) and (bool(value["steps"]) or allow_empty), "缺少步骤资产对应关系")
    step_ids, used = set(), set()
    for step in value["steps"]:
        fields(step, "id description asset_ids evidence")
        identifier(step["id"])
        require(step["id"] not in step_ids, "步骤标识重复")
        step_ids.add(step["id"])
        text(step["description"])
        texts(step["asset_ids"])
        require(set(step["asset_ids"]) <= ids, "步骤引用未知资产")
        used.update(step["asset_ids"])
        evidence(step["evidence"], sources, required=True)
    require(used == ids, "存在没有关联任何步骤的资产")
    require(isinstance(value["relations"], list), "关系列表格式错误")
    for rel in value["relations"]:
        fields(rel, "subject relation object basis")
        require(rel["subject"] in ids and rel["object"] in ids, "关系引用未知资产")
        text(rel["relation"])
        text(rel["basis"])
    require(isinstance(value["conflicts"], list), "冲突列表格式错误")
    for conflict in value["conflicts"]:
        fields(conflict, "topic asset_id target_key pdf_evidence xml_evidence chosen_source resolution")
        text(conflict["topic"])
        text(conflict["resolution"])
        evidence(conflict["pdf_evidence"], sources, required=True, kind="pdf")
        evidence(conflict["xml_evidence"], sources, required=True, kind="xml")
        require(conflict["chosen_source"] == "pdf", "已识别的文档冲突必须以 PDF 为准")
        require(conflict["asset_id"] in ids, "冲突引用未知资产")
        asset = next(a for a in value["assets"] if a["id"] == conflict["asset_id"])
        target = asset["quantity"] if conflict["target_key"] == "quantity" else next(
            (r for r in asset["requirements"] if r["key"] == conflict["target_key"]), None)
        require(target is not None, "冲突必须关联数量或具体要求")
        require(target["origin"] == "explicit", "冲突后必须采用有原文依据的 PDF 要求")
        evidence(target["evidence"], sources, required=True, kind="pdf")
        require(any(r in conflict["pdf_evidence"] for r in target["evidence"]), "最终要求未使用冲突中的 PDF 依据")
    texts(value["assumptions"])
    texts(value["issues"])
    return value


def validate_selection(value, asset, references):
    fields(value, "mode product_name source_ids selection_reason specifications adaptations image_ids unknowns")
    require(value["mode"] in {"reference", "reference_adapted", "requirements_only"}, "未知选型模式")
    text(value["selection_reason"])
    texts(value["source_ids"])
    texts(value["image_ids"])
    texts(value["unknowns"])
    sources = {s["id"]: s for s in references["sources"]}
    images = {i["id"]: i for i in references["images"]}
    require(set(value["source_ids"]) <= sources.keys(), "产品引用不在实际检索结果中")
    require(set(value["image_ids"]) <= images.keys(), "图片不在实际检索结果中")
    if value["mode"] == "requirements_only":
        require(value["product_name"] is None and not value["source_ids"] and not value["image_ids"], "无参考模式不能冒充检索到产品")
    else:
        text(value["product_name"])
        require(bool(value["source_ids"]), "产品选择必须有检索来源")
    require(isinstance(value["specifications"], list), "产品规格须为列表")
    for spec in value["specifications"]:
        fields(spec, "key value unit source_id quote")
        text(spec["key"])
        text(spec["value"])
        text(spec["quote"])
        require(isinstance(spec["unit"], str), "规格单位须为文本")
        require(spec["source_id"] in value["source_ids"], "规格须来自选定产品的来源")
        require(spec["quote"] in sources[spec["source_id"]]["content"], "产品规格原文引用不成立")
    require(isinstance(value["adaptations"], list), "适配要求须为列表")
    requirement_keys = {r["key"] for r in asset["requirements"]}
    for item in value["adaptations"]:
        fields(item, "requirement_key change reason")
        require(item["requirement_key"] in requirement_keys, "适配要求引用未知目标规格")
        text(item["change"])
        text(item["reason"])
    if value["mode"] == "reference_adapted":
        require(bool(value["adaptations"]), "改造参考产品时须说明改动")
    if value["mode"] == "reference":
        require(not value["adaptations"], "存在改动时必须标记 reference_adapted")
    # 同名规格的直接差异不能隐藏为“完全匹配”。单位换算等需由模型解释为未知或适配。
    adapted = {a["requirement_key"] for a in value["adaptations"]}
    for requirement in asset["requirements"]:
        for spec in value["specifications"]:
            if requirement["key"] == spec["key"] and (requirement["value"], requirement["unit"]) != (spec["value"], spec["unit"]):
                require(requirement["key"] in adapted, "参考产品规格不同于目标，须明确适配，不能直接判为匹配")
    for image_id in value["image_ids"]:
        source_id = images[image_id]["source_id"]
        linked = images[image_id].get("source_ids", [source_id] if source_id else [])
        require(not linked or bool(set(linked) & set(value["source_ids"])), "图片不属于所选参考来源")
    return value


def fallback_selection(reason):
    return {"mode": "requirements_only", "product_name": None, "source_ids": [],
            "selection_reason": reason, "specifications": [], "adaptations": [],
            "image_ids": [], "unknowns": ["没有已确认的参考产品；按目标要求生成，保留未验证假设"]}
