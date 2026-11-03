"""Non-sensitive configuration, and credential reading restricted to the project root."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ValidationError, ProviderError


@dataclass(frozen=True)
class Config:
    provider: str = "codex"
    base_url: str = "https://chatgpt.com/backend-api/codex"
    model: str = "gpt-6.1-sol"
    api_key_env: str = "CODEX_LOGIN"
    search_provider: str = "model_web"
    search_url: str = "https://api.tavily.com/search"
    scope: str = "From the complete input experimental protocol, extract the instruments, containers, consumables and necessary accessories together with their requirements; reagents are not assets to be generated. Do not presuppose the experiment type, the final step, or any excluded experimental stage."
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
    # codex only: author requests run as a coding-agent session with local self-checks.
    codex_agent_author: bool = False

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in {"download_images", "codex_agent_author"}:
                if type(value) is not bool:
                    raise ValidationError(f"{name} must be a boolean")
            elif isinstance(getattr(type(self), name, None), int):
                if type(value) is not int or value < (0 if name == "repair_attempts" else 1):
                    raise ValidationError(f"config {name} must be a valid integer")
            elif not isinstance(value, str) or not value.strip():
                raise ValidationError(f"config {name} must be a non-empty string")
        if self.codex_agent_author and self.provider != "codex":
            raise ValidationError("codex_agent_author applies only to the codex provider")
        if self.provider == "codex":
            # Codex CLI uses the local ChatGPT login session and reads no API key.
            if self.api_key_env != "CODEX_LOGIN":
                raise ValidationError("the codex provider uses the local login session; api_key_env must be CODEX_LOGIN")
        elif self.api_key_env not in {"HARNESS_MODEL_API_KEY", "HARNESS_MODEL_API_KEY2", "OPENAI_API_KEY"}:
            raise ValidationError("the model key variable must be a credential name supported by the project")
        if self.search_provider not in {"model_web", "tavily"}:
            raise ValidationError("the search adapter supports only model_web or tavily")
        if self.output_token_parameter not in {'max_tokens', 'max_completion_tokens'}:
            raise ValidationError('unsupported output-token parameter name')
        for value in (self.base_url, self.search_url):
            parsed = urlsplit(value)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment):
                raise ValidationError("the service URL must be an HTTPS URL without credentials or query parameters")

    @classmethod
    def load(cls, path: Path):
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8-sig")))
        except (TypeError, ValueError, OSError) as exc:
            raise ValidationError("cannot read the non-sensitive configuration, or it contains unknown fields") from exc

    def identity(self):
        # On resume only the call budget may be raised; changing the model, scope or prompt parameters requires a new run.
        result = {k: v for k, v in asdict(self).items() if k not in {
            "max_model_calls", "max_search_calls", "max_image_downloads",
            "max_reserved_tokens", "max_page_fetches", "timeout_seconds"}}
        # Preserve historical identities using the original credential slot.
        if self.api_key_env == "HARNESS_MODEL_API_KEY":
            result.pop("api_key_env")
        if not self.codex_agent_author:
            result.pop("codex_agent_author")  # Keep identities of runs made before the switch existed.
        return result

    def model_key(self, values):
        if self.provider == "codex":
            return "codex-login"
        key = values.get(self.api_key_env)
        if not key or not key.strip():
            raise ProviderError(f"missing {self.api_key_env} required by the configuration; will not fall back to another service key")
        return key


def credentials(root: Path) -> dict[str, str]:
    names = {"HARNESS_MODEL_API_KEY", "HARNESS_MODEL_API_KEY2", "OPENAI_API_KEY", "TAVILY_API_KEY"}
    values = {}
    path = root / ".env.local"
    if path.is_symlink() or (path.exists() and path.resolve().parent != root.resolve()):
        raise ValidationError("refusing to read a credential file that points outside the project")
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key in names:
                values[key] = value.strip().strip("\"'")
    for name in names:
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values
