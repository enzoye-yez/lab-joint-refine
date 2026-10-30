"""用户执行的单请求服务诊断；不读取实验输入，不生成资产，无自动重试。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from asset_generation.contracts import GenerationConfig
from asset_generation.provider_diagnostics import safe_details
from asset_generation.transport import GenerationTransport
from asset_prep.config import Config, credentials
from asset_prep.errors import PrepError, ProviderError
from asset_prep.storage import atomic_json


def main(argv=None, transport=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="发送一次真实请求，可能计费")
    parser.add_argument("--provider-config", type=Path, default=project / "config/default.json")
    parser.add_argument("--generation-config", type=Path, default=project / "config/generation.json")
    args = parser.parse_args(argv)
    config = Config.load(args.provider_config)
    generation = GenerationConfig.load(args.generation_config)
    if not args.execute:
        print("未发送请求；加 --execute 才执行单次服务诊断。")
        return 0
    keys = credentials(project)
    key = config.model_key(keys)
    if not key:
        raise PrepError("缺少项目模型服务凭据")
    # 保留正式生成的模型、参数名、输出上限和图文消息结构，输入仅为无敏感内容的短文本。
    payload = {"model": config.model, "stream": False,
               config.output_token_parameter: generation.max_output_tokens,
               "messages": [{"role": "system", "content": "Reply with the JSON object {\"ok\": true} only."},
                            {"role": "user", "content": [{"type": "text", "text": "Connectivity check."}]}]}
    report = {"provider": config.provider, "model": config.model, "attempts": 1,
              "output_token_parameter": config.output_token_parameter,
              "max_output_tokens": generation.max_output_tokens,
              "scope": "short_text_request_only", "status": "pending"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = project / ".cache" / "provider_probes" / (stamp + ".json")
    atomic_json(path, report)
    print("发送一次短文本服务诊断；不会重试，也不生成实验资产。", flush=True)
    try:
        response = (transport or GenerationTransport()).post(
            config.base_url.rstrip("/") + "/chat/completions", payload, key, config.timeout_seconds)
        if not isinstance(response.get("choices"), list) or not response["choices"]:
            report.update(status="invalid_response", provider_error=safe_details(response.get("error"), keys.values()))
        else:
            report.update(status="response_received", http_status=200)
    except ProviderError as exc:
        report.update(status="request_failed", outcome_unknown=exc.uncertain,
                      http_status=exc.diagnostics.get("http_status"), failure=exc.diagnostics.get("failure"),
                      provider_error=safe_details(exc.diagnostics.get("provider_error"), keys.values()))
    except KeyboardInterrupt:
        report.update(status="interrupted", outcome_unknown=True)
        raise
    finally:
        atomic_json(path, report)
        print(json.dumps(report, ensure_ascii=False))
        print("诊断文件：" + str(path))
    return 0 if report["status"] == "response_received" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepError as exc:
        print(str(exc))
        raise SystemExit(2)
