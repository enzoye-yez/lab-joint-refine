"""Fixed validators: check evidence citations and structure; the generating model cannot change the checking rules."""

import re

from .errors import ValidationError


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(obj, names):
    require(isinstance(obj, dict), "expected an object")
    require(set(obj) == set(names.split()), f"fields must be exactly: {names}")


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "expected non-empty text")


def texts(values):
    require(isinstance(values, list), "expected a list")
    for value in values:
        text(value)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value), "identifiers may contain only lowercase letters, digits and underscores")


def evidence(refs, sources, *, required=False, kind=None):
    require(isinstance(refs, list) and (bool(refs) or not required), "missing evidence list")
    for ref in refs:
        fields(ref, "source_id quote")
        text(ref["quote"])
        require(ref["source_id"] in sources, "citation references an unknown document source")
        source = sources[ref["source_id"]]
        require(ref["quote"] in source["text"], "cited quote does not appear in the specified source")
        if kind:
            require(source["kind"] == kind, "conflict citation has the wrong source kind")


def origin(obj, sources):
    require(obj["origin"] in {"explicit", "inferred"}, "origin must be explicit or inferred")
    text(obj["basis"])
    evidence(obj["evidence"], sources, required=obj["origin"] == "explicit")


def validate_inventory(value, sources, max_assets=64, *, allow_empty=False):
    sources = {s["id"]: s for s in sources}
    fields(value, "assets steps relations conflicts assumptions issues")
    require(isinstance(value["assets"], list) and (bool(value["assets"]) or allow_empty)
            and len(value["assets"]) <= max_assets, "asset list is empty or exceeds the limit")
    ids = set()
    for asset in value["assets"]:
        fields(asset, "id name category aliases quantity requirements operations evidence parent_id queries")
        identifier(asset["id"])
        require(asset["id"] not in ids, "duplicate asset id")
        ids.add(asset["id"])
        text(asset["name"])
        require(asset["category"] in {"instrument", "container", "consumable", "accessory"}, "reagents and unsupported asset categories are not allowed")
        for key in ("aliases", "operations", "queries"):
            texts(asset[key])
        require(bool(asset["queries"]), "asset is missing search queries")
        quantity = asset["quantity"]
        fields(quantity, "value origin basis evidence")
        require(type(quantity["value"]) is int and 1 <= quantity["value"] <= 100000, "quantity must be an integer from 1 to 100000")
        origin(quantity, sources)
        evidence(asset["evidence"], sources, required=True)
        require(isinstance(asset["requirements"], list), "requirements must be a list")
        keys = set()
        for item in asset["requirements"]:
            fields(item, "key value unit origin basis evidence")
            text(item["key"])
            require(item["key"] not in keys, "duplicate requirement key within the same asset")
            keys.add(item["key"])
            text(item["value"])
            require(isinstance(item["unit"], str), "unit must be text (may be empty)")
            origin(item, sources)
    parents = {a["id"]: a["parent_id"] for a in value["assets"]}
    for asset in value["assets"]:
        parent = asset["parent_id"]
        require(parent is None or parent in ids, "parent asset does not exist")
        seen = {asset["id"]}
        while parent is not None:
            require(parent not in seen, "part-of relations form a cycle")
            seen.add(parent)
            parent = parents[parent]
    require(isinstance(value["steps"], list) and (bool(value["steps"]) or allow_empty), "missing step-to-asset mapping")
    step_ids, used = set(), set()
    for step in value["steps"]:
        fields(step, "id description asset_ids evidence")
        identifier(step["id"])
        require(step["id"] not in step_ids, "duplicate step id")
        step_ids.add(step["id"])
        text(step["description"])
        texts(step["asset_ids"])
        require(set(step["asset_ids"]) <= ids, "step references an unknown asset")
        used.update(step["asset_ids"])
        evidence(step["evidence"], sources, required=True)
    require(used == ids, "some assets are not linked to any step")
    require(isinstance(value["relations"], list), "relations list is malformed")
    for rel in value["relations"]:
        fields(rel, "subject relation object basis")
        require(rel["subject"] in ids and rel["object"] in ids, "relation references an unknown asset")
        text(rel["relation"])
        text(rel["basis"])
    require(isinstance(value["conflicts"], list), "conflicts list is malformed")
    for conflict in value["conflicts"]:
        fields(conflict, "topic asset_id target_key pdf_evidence xml_evidence chosen_source resolution")
        text(conflict["topic"])
        text(conflict["resolution"])
        evidence(conflict["pdf_evidence"], sources, required=True, kind="pdf")
        evidence(conflict["xml_evidence"], sources, required=True, kind="xml")
        require(conflict["chosen_source"] == "pdf", "an identified document conflict must be resolved in favor of the PDF")
        require(conflict["asset_id"] in ids, "conflict references an unknown asset")
        asset = next(a for a in value["assets"] if a["id"] == conflict["asset_id"])
        target = asset["quantity"] if conflict["target_key"] == "quantity" else next(
            (r for r in asset["requirements"] if r["key"] == conflict["target_key"]), None)
        require(target is not None, "a conflict must be linked to the quantity or a specific requirement")
        require(target["origin"] == "explicit", "after a conflict, the PDF requirement with source evidence must be adopted")
        evidence(target["evidence"], sources, required=True, kind="pdf")
        require(any(r in conflict["pdf_evidence"] for r in target["evidence"]), "the final requirement does not use the PDF evidence from the conflict")
    texts(value["assumptions"])
    texts(value["issues"])
    return value


def validate_selection(value, asset, references):
    fields(value, "mode product_name source_ids selection_reason specifications adaptations image_ids unknowns")
    require(value["mode"] in {"reference", "reference_adapted", "requirements_only"}, "unknown selection mode")
    text(value["selection_reason"])
    texts(value["source_ids"])
    texts(value["image_ids"])
    texts(value["unknowns"])
    sources = {s["id"]: s for s in references["sources"]}
    images = {i["id"]: i for i in references["images"]}
    require(set(value["source_ids"]) <= sources.keys(), "product citation is not among the actual search results")
    require(set(value["image_ids"]) <= images.keys(), "image is not among the actual search results")
    if value["mode"] == "requirements_only":
        require(value["product_name"] is None and not value["source_ids"] and not value["image_ids"], "requirements-only mode must not pretend a product was found")
    else:
        text(value["product_name"])
        require(bool(value["source_ids"]), "a product selection must have search sources")
    require(isinstance(value["specifications"], list), "product specifications must be a list")
    for spec in value["specifications"]:
        fields(spec, "key value unit source_id quote")
        text(spec["key"])
        text(spec["value"])
        text(spec["quote"])
        require(isinstance(spec["unit"], str), "specification unit must be text")
        require(spec["source_id"] in value["source_ids"], "specifications must come from the selected product's sources")
        require(spec["quote"] in sources[spec["source_id"]]["content"], "product specification quote does not hold in its source")
    require(isinstance(value["adaptations"], list), "adaptations must be a list")
    requirement_keys = {r["key"] for r in asset["requirements"]}
    for item in value["adaptations"]:
        fields(item, "requirement_key change reason")
        require(item["requirement_key"] in requirement_keys, "adaptation references an unknown target requirement")
        text(item["change"])
        text(item["reason"])
    if value["mode"] == "reference_adapted":
        require(bool(value["adaptations"]), "adapting a reference product requires stating the changes")
    if value["mode"] == "reference":
        require(not value["adaptations"], "when there are changes, mode must be reference_adapted")
    # A direct difference in a same-named specification must not be hidden as an "exact match". Unit conversions and the like must be explained by the model as unknown or adapted.
    adapted = {a["requirement_key"] for a in value["adaptations"]}
    for requirement in asset["requirements"]:
        for spec in value["specifications"]:
            if requirement["key"] == spec["key"] and (requirement["value"], requirement["unit"]) != (spec["value"], spec["unit"]):
                require(requirement["key"] in adapted, "reference product specification differs from the target; it must be explicitly adapted, not judged a match")
    for image_id in value["image_ids"]:
        source_id = images[image_id]["source_id"]
        linked = images[image_id].get("source_ids", [source_id] if source_id else [])
        require(not linked or bool(set(linked) & set(value["source_ids"])), "image does not belong to the selected reference sources")
    return value


def fallback_selection(reason):
    return {"mode": "requirements_only", "product_name": None, "source_ids": [],
            "selection_reason": reason, "specifications": [], "adaptations": [],
            "image_ids": [], "unknowns": ["no confirmed reference product; generate from the target requirements and keep unverified assumptions"]}
