"""User-run single-request provider diagnostic; does not read experiment inputs, generate assets or retry automatically."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from asset_generation.contracts import GenerationConfig
from asset_generation.provider_diagnostics import safe_details
from asset_generation.transport import transport_for
from asset_prep.config import Config, credentials
from asset_prep.errors import PrepError, ProviderError
from asset_prep.storage import atomic_json


def main(argv=None, transport=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="send one real request; may be billed")
    parser.add_argument("--provider-config", type=Path, default=project / "config/default.json")
    parser.add_argument("--generation-config", type=Path, default=project / "config/generation.json")
    args = parser.parse_args(argv)
    config = Config.load(args.provider_config)
    generation = GenerationConfig.load(args.generation_config)
    if not args.execute:
        print("No request sent; add --execute to run the single provider diagnostic.")
        return 0
    keys = credentials(project)
    key = config.model_key(keys)
    if not key:
        raise PrepError("missing project model-provider credentials")
    # Keep the production model, parameter names, output limit and text+image message structure; the input is only a short non-sensitive text.
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
    print("Sending one short-text provider diagnostic; no retries, no experiment assets generated.", flush=True)
    try:
        response = (transport or transport_for(config)).post(
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
        print("diagnostic file: " + str(path))
    return 0 if report["status"] == "response_received" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepError as exc:
        print(str(exc))
        raise SystemExit(2)
