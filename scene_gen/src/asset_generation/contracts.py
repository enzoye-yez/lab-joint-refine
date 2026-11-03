"""A few delivery conventions and the read-only acceptance config; does not wrap the model's geometric design methods."""

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from asset_prep.errors import ValidationError
from asset_prep.storage import encoded


@dataclass(frozen=True)
class GenerationConfig:
    max_iterations: int = 3
    max_model_calls: int = 6
    max_reserved_tokens: int = 600000
    max_transport_retries: int = 5
    max_unknown_http_retries: int = 0
    transport_cooldown_seconds: int = 300
    max_output_tokens: int = 16000
    max_context_tokens: int = 922000
    context_margin_tokens: int = 32768
    max_context_reads: int = 8
    review_context_bytes: int = 120000
    max_request_bytes: int = 24000000
    max_image_bytes: int = 8000000
    max_images: int = 24
    max_script_bytes: int = 500000
    build_timeout_seconds: int = 120
    max_artifact_bytes: int = 64000000
    max_artifact_files: int = 128
    image: str = "harness-asset-worker:0.1"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "image":
                if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]{0,200}", value):
                    raise ValidationError("Invalid build image name")
            elif type(value) is not int or value < (0 if name in {"max_transport_retries", "max_unknown_http_retries"} else 1):
                raise ValidationError(f"{name} must be a positive integer")

    @classmethod
    def load(cls, path):
        try:
            return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))
        except (ValueError, TypeError, OSError) as exc:
            raise ValidationError("Invalid generation config") from exc

    def identity(self):
        return {k: v for k, v in asdict(self).items() if k not in {
            "max_iterations", "max_model_calls", "max_reserved_tokens", "build_timeout_seconds",
            "max_transport_retries", "max_unknown_http_retries", "transport_cooldown_seconds"}}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_json(path, max_bytes=2000000):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= max_bytes, "File missing or exceeds the size limit")
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        encoded(result)  # Reject non-finite numbers.
        return result
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("Invalid structured file") from exc


def check_profile(value):
    require(isinstance(value, dict), "Acceptance config must be an object")
    allowed = {"dimensions", "required_interfaces", "joints", "settle_steps", "penetration_tolerance_m", "validation_mode"}
    require(not set(value) - allowed, "Acceptance config contains unknown fields")
    result = {"dimensions": [], "required_interfaces": [], "joints": [],
              "settle_steps": 100, "penetration_tolerance_m": 0.0005, "validation_mode": "full", **value}
    require(isinstance(result['validation_mode'],str) and result["validation_mode"] in {"full", "static"}, "Invalid validation mode")
    require(type(result["settle_steps"]) is int and 0 <= result["settle_steps"] <= 5000, "Invalid number of settle steps")
    require(finite(result["penetration_tolerance_m"]) and 0 <= result["penetration_tolerance_m"] <= 0.01,
            "Invalid penetration tolerance")
    for key in ("dimensions", "required_interfaces", "joints"):
        require(isinstance(result[key], list) and len(result[key]) <= 256, "Check entries must be bounded lists")
    for item in result["dimensions"]:
        require(isinstance(item, dict) and set(item) == {"id", "geoms", "axis", "expected_m", "tolerance_m"},
                "Invalid dimension check fields")
        require(isinstance(item["id"], str) and item["id"] and isinstance(item["geoms"], list)
                and item["geoms"] and all(isinstance(g, str) and g for g in item["geoms"]), "Invalid dimension check targets")
        require(type(item["axis"]) is int and item["axis"] in (0, 1, 2), "Invalid dimension axis")
        require(finite(item["expected_m"]) and item["expected_m"] > 0 and finite(item["tolerance_m"])
                and item["tolerance_m"] >= 0, "Invalid dimension target or tolerance")
    require(all(isinstance(i, str) and i for i in result["required_interfaces"]), "Invalid required interface id")
    for item in result["joints"]:
        require(isinstance(item, dict) and set(item) == {"name", "type", "range", "tolerance"}, "Invalid joint check fields")
        require(isinstance(item["name"], str) and isinstance(item['type'],str) and item["type"] in {"hinge", "slide", "ball", "free"}, "Invalid joint type")
        require(isinstance(item["range"], list) and len(item["range"]) == 2
                and all(finite(x) for x in item["range"]) and item["range"][0] <= item["range"][1], "Invalid joint range")
        require(finite(item["tolerance"]) and item["tolerance"] >= 0, "Invalid joint tolerance")
    require(len({i['id'] for i in result['dimensions']})==len(result['dimensions']), 'Duplicate dimension check ids')
    require(len({i['name'] for i in result['joints']})==len(result['joints']), 'Duplicate joint check names')
    return result


def author_result(value, previous, max_bytes):
    """A complete script the first time; repairs accept only unique text replacements on the current script. Not an execution sandbox."""
    require(isinstance(value, dict), "Generation response must be an object")
    if set(value)=={"deferred"}:
        item=value["deferred"]
        require(isinstance(item,dict) and set(item)=={"reason","missing"}
                and isinstance(item["reason"],str) and 0<len(item["reason"])<=2000
                and isinstance(item["missing"],list) and 1<=len(item["missing"])<=32
                and all(isinstance(v,str) and 0<len(v)<=500 for v in item["missing"]), "Deferral must state the specific reason and the missing information")
        from asset_prep.errors import ContextLimitError
        raise ContextLimitError("Author explicitly deferred: "+item["reason"]+"; missing: "+"; ".join(item["missing"]))
    if previous is None:
        require(set(value) == {"model_py"} and isinstance(value["model_py"], str), "First response must contain only model_py")
        source = value["model_py"]
    else:
        require(set(value) == {"edits"} and isinstance(value["edits"], list)
                and 1 <= len(value["edits"]) <= 32, "Repair response must contain only a non-empty edits list")
        source = previous
        for edit in value["edits"]:
            require(isinstance(edit, dict) and set(edit) == {"old", "new"}
                    and all(isinstance(x, str) for x in edit.values()), "Invalid patch format")
            require(edit["old"] and source.count(edit["old"]) == 1, "Patch old text must match the current script exactly once")
            source = source.replace(edit["old"], edit["new"], 1)
        require(source != previous, "Repair did not change the script")
    require(len(source.encode("utf-8")) <= max_bytes, "Model script exceeds the size limit")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValidationError(f"Script syntax error, line {exc.lineno}") from exc
    require(any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body), "build(output_dir) must be defined")
    return source


def visual_result(value):
    require(isinstance(value, dict) and {"verdict", "findings", "unverified"} <= set(value)
            <= {"verdict", "findings", "unverified", "motion_coverage"}, "Invalid visual review format")
    require(isinstance(value['verdict'],str) and value["verdict"] in {"accept", "repair", "uncertain"}, "Invalid visual review verdict")
    for key in ("findings", "unverified"):
        require(isinstance(value[key], list) and all(isinstance(x, str) for x in value[key]), "Review items must be lists of text")
    require(value["verdict"] != "accept" or not value["findings"], "Cannot accept while there are problems to repair")
    return value
