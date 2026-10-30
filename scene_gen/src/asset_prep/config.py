"""非敏感配置与仅限项目根目录的凭据读取。"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ValidationError, ProviderError


@dataclass(frozen=True)
class Config:
    provider: str = "gpugeek"
    base_url: str = "https://api.gpugeek.com/v1"
    model: str = "GPT-6-astra"
    api_key_env: str = "HARNESS_MODEL_API_KEY"
    search_provider: str = "model_web"
    search_url: str = "https://api.tavily.com/search"
    scope: str = "根据输入的完整实验规程提取仪器、容器、耗材和必要配件及其要求，不把试剂作为待生成资产。不预设实验类型、结束步骤或排除的实验阶段。"
    chunk_chars: int = 4000
    max_document_chars: int = 1000000
    max_input_bytes: int = 160000
    max_assets: int = 64
    max_output_tokens: int = 12000
    extraction_max_assets: int = 8
    extraction_max_chars: int = 16000
    extraction_output_tokens: int = 6000
    output_token_parameter: str = 'max_completion_tokens'
    repair_attempts: int = 2
    queries_per_asset: int = 2
    results_per_query: int = 5
    images_per_asset: int = 2
    timeout_seconds: int = 600
    max_model_calls: int = 40
    max_search_calls: int = 64
    search_max_output_tokens: int = 2048
    search_context_reserve_tokens: int = 16000
    search_max_tool_calls: int = 1
    pages_per_asset: int = 3
    max_page_fetches: int = 192
    max_image_downloads: int = 128
    max_reserved_tokens: int = 1200000
    download_images: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "download_images":
                if type(value) is not bool:
                    raise ValidationError("download_images 必须是布尔值")
            elif isinstance(getattr(type(self), name, None), int):
                if type(value) is not int or value < (0 if name == "repair_attempts" else 1):
                    raise ValidationError(f"配置 {name} 必须是有效整数")
            elif not isinstance(value, str) or not value.strip():
                raise ValidationError(f"配置 {name} 必须是非空字符串")
        if self.api_key_env not in {"HARNESS_MODEL_API_KEY", "HARNESS_MODEL_API_KEY2", "OPENAI_API_KEY"}:
            raise ValidationError("模型密钥变量必须是项目支持的凭据名称")
        if self.search_provider not in {"model_web", "tavily"}:
            raise ValidationError("检索适配器只支持 model_web 或 tavily")
        if self.output_token_parameter not in {'max_tokens', 'max_completion_tokens'}:
            raise ValidationError('不支持的输出令牌参数名称')
        for value in (self.base_url, self.search_url):
            parsed = urlsplit(value)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment):
                raise ValidationError("服务地址必须为不含凭据、查询参数的 HTTPS 地址")

    @classmethod
    def load(cls, path: Path):
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8-sig")))
        except (TypeError, ValueError, OSError) as exc:
            raise ValidationError("无法读取非敏感配置或存在未知配置字段") from exc

    def identity(self):
        # 只允许恢复时提高调用预算；更换模型、范围、提示参数必须新建运行。
        result = {k: v for k, v in asdict(self).items() if k not in {
            "max_model_calls", "max_search_calls", "max_image_downloads",
            "max_reserved_tokens", "max_page_fetches", "timeout_seconds"}}
        # Preserve historical identities using the original credential slot.
        if self.api_key_env == "HARNESS_MODEL_API_KEY":
            result.pop("api_key_env")
        return result

    def model_key(self, values):
        key = values.get(self.api_key_env)
        if not key or not key.strip():
            raise ProviderError(f"缺少配置指定的 {self.api_key_env}；不会回退使用其他服务密钥")
        return key


def credentials(root: Path) -> dict[str, str]:
    names = {"HARNESS_MODEL_API_KEY", "HARNESS_MODEL_API_KEY2", "OPENAI_API_KEY", "TAVILY_API_KEY"}
    values = {}
    path = root / ".env.local"
    if path.is_symlink() or (path.exists() and path.resolve().parent != root.resolve()):
        raise ValidationError("拒绝读取指向项目外的凭据文件")
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key in names:
                values[key] = value.strip().strip("\"'")
    for name in names:
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values
