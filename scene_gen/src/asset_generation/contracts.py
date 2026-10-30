"""少量交付约定与只读验收配置，不封装模型的几何设计方法。"""

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
                    raise ValidationError("构建镜像名称无效")
            elif type(value) is not int or value < (0 if name in {"max_transport_retries", "max_unknown_http_retries"} else 1):
                raise ValidationError(f"{name} 必须为正整数")

    @classmethod
    def load(cls, path):
        try:
            return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))
        except (ValueError, TypeError, OSError) as exc:
            raise ValidationError("生成配置无效") from exc

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
    require(path.is_file() and path.stat().st_size <= max_bytes, "缺少文件或文件超过大小上限")
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        encoded(result)  # 拒绝非有限数值。
        return result
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("无效的结构化文件") from exc


def check_profile(value):
    require(isinstance(value, dict), "验收配置须为对象")
    allowed = {"dimensions", "required_interfaces", "joints", "settle_steps", "penetration_tolerance_m", "validation_mode"}
    require(not set(value) - allowed, "验收配置包含未知字段")
    result = {"dimensions": [], "required_interfaces": [], "joints": [],
              "settle_steps": 100, "penetration_tolerance_m": 0.0005, "validation_mode": "full", **value}
    require(isinstance(result['validation_mode'],str) and result["validation_mode"] in {"full", "static"}, "无效验证模式")
    require(type(result["settle_steps"]) is int and 0 <= result["settle_steps"] <= 5000, "稳定检查步数无效")
    require(finite(result["penetration_tolerance_m"]) and 0 <= result["penetration_tolerance_m"] <= 0.01,
            "穿透公差无效")
    for key in ("dimensions", "required_interfaces", "joints"):
        require(isinstance(result[key], list) and len(result[key]) <= 256, "检查条目必须为有界列表")
    for item in result["dimensions"]:
        require(isinstance(item, dict) and set(item) == {"id", "geoms", "axis", "expected_m", "tolerance_m"},
                "尺寸检查字段无效")
        require(isinstance(item["id"], str) and item["id"] and isinstance(item["geoms"], list)
                and item["geoms"] and all(isinstance(g, str) and g for g in item["geoms"]), "尺寸检查对象无效")
        require(type(item["axis"]) is int and item["axis"] in (0, 1, 2), "尺寸轴无效")
        require(finite(item["expected_m"]) and item["expected_m"] > 0 and finite(item["tolerance_m"])
                and item["tolerance_m"] >= 0, "尺寸目标或公差无效")
    require(all(isinstance(i, str) and i for i in result["required_interfaces"]), "必需接口编号无效")
    for item in result["joints"]:
        require(isinstance(item, dict) and set(item) == {"name", "type", "range", "tolerance"}, "关节检查字段无效")
        require(isinstance(item["name"], str) and isinstance(item['type'],str) and item["type"] in {"hinge", "slide", "ball", "free"}, "关节类型无效")
        require(isinstance(item["range"], list) and len(item["range"]) == 2
                and all(finite(x) for x in item["range"]) and item["range"][0] <= item["range"][1], "关节范围无效")
        require(finite(item["tolerance"]) and item["tolerance"] >= 0, "关节公差无效")
    require(len({i['id'] for i in result['dimensions']})==len(result['dimensions']), '尺寸检查编号重复')
    require(len({i['name'] for i in result['joints']})==len(result['joints']), '关节检查名称重复')
    return result


def author_result(value, previous, max_bytes):
    """首次完整脚本；修复只接受当前脚本上的唯一文本替换。并非执行沙箱。"""
    require(isinstance(value, dict), "生成响应必须是对象")
    if set(value)=={"deferred"}:
        item=value["deferred"]
        require(isinstance(item,dict) and set(item)=={"reason","missing"}
                and isinstance(item["reason"],str) and 0<len(item["reason"])<=2000
                and isinstance(item["missing"],list) and 1<=len(item["missing"])<=32
                and all(isinstance(v,str) and 0<len(v)<=500 for v in item["missing"]), "暂挂须说明具体原因及缺少的信息")
        from asset_prep.errors import ContextLimitError
        raise ContextLimitError("作者明确暂挂："+item["reason"]+"；缺少："+"；".join(item["missing"]))
    if previous is None:
        require(set(value) == {"model_py"} and isinstance(value["model_py"], str), "首次响应只包含 model_py")
        source = value["model_py"]
    else:
        require(set(value) == {"edits"} and isinstance(value["edits"], list)
                and 1 <= len(value["edits"]) <= 32, "修复响应只包含非空 edits 列表")
        source = previous
        for edit in value["edits"]:
            require(isinstance(edit, dict) and set(edit) == {"old", "new"}
                    and all(isinstance(x, str) for x in edit.values()), "补丁格式无效")
            require(edit["old"] and source.count(edit["old"]) == 1, "补丁旧文本须唯一匹配当前脚本")
            source = source.replace(edit["old"], edit["new"], 1)
        require(source != previous, "修复未改变脚本")
    require(len(source.encode("utf-8")) <= max_bytes, "模型脚本超过上限")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValidationError(f"脚本语法错误，行 {exc.lineno}") from exc
    require(any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body), "须定义 build(output_dir)")
    return source


def visual_result(value):
    require(isinstance(value, dict) and {"verdict", "findings", "unverified"} <= set(value)
            <= {"verdict", "findings", "unverified", "motion_coverage"}, "视觉审查格式无效")
    require(isinstance(value['verdict'],str) and value["verdict"] in {"accept", "repair", "uncertain"}, "视觉审查结论无效")
    for key in ("findings", "unverified"):
        require(isinstance(value[key], list) and all(isinstance(x, str) for x in value[key]), "审查项须为文本列表")
    require(value["verdict"] != "accept" or not value["findings"], "存在待修复问题时不能接受")
    return value
