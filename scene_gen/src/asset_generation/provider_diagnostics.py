"""Keep only bounded, redacted service error fields; never store the full response or request content."""

import http.client
import json
import re
import time

from asset_prep.errors import ProviderError
from asset_prep.network import Transport


FIELDS = {"code", "type", "message", "param", "request_id", "retry_after",
          "error_body_bytes", "error_body_status"}


def safe_details(value, secrets=()):
    result = {}
    if not isinstance(value, dict):
        return result
    for name, item in value.items():
        if name not in FIELDS or type(item) not in (str, int):
            continue
        text = str(item)
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"(?i)\bBearer\s+[^\s\"',;]+", "Bearer [REDACTED]", text)
        text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
        result[name] = " ".join(text.split())[:1024]
    return result


class DiagnosticTransport(Transport):
    def http_error(self, response, connection, key, deadline):
        # Read at most 32 KiB / 2 s of the error body; a slow or corrupt body must not hide an HTTP status already received.
        details, data = {}, bytearray()
        deadline = min(deadline, time.monotonic() + 2)
        for header, field in (("x-request-id", "request_id"), ("retry-after", "retry_after")):
            value = response.getheader(header)
            if isinstance(value, str):
                details[field] = value
        try:
            while len(data) <= 32768:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                if connection.sock:
                    connection.sock.settimeout(remaining)
                part = response.read1(min(4096, 32769 - len(data)))
                if not part:
                    break
                data.extend(part)
            if len(data) > 32768:
                details["error_body_status"] = "too_large"
            else:
                body = json.loads(data)
                error = body.get("error", body) if isinstance(body, dict) else None
                if isinstance(error, dict):
                    details.update({k: error[k] for k in ("code", "type", "message", "param") if k in error})
                    details["error_body_status"] = "json_fields"
                elif isinstance(error, str):
                    details.update(message=error, error_body_status="json_message")
                else:
                    details["error_body_status"] = "no_error_fields"
        except (OSError, http.client.HTTPException):
            details["error_body_status"] = "read_failed"
        except (ValueError, UnicodeError):
            details["error_body_status"] = "not_json"
        details["error_body_bytes"] = len(data)
        details = safe_details(details, [key])
        message = f"Service returned HTTP status {response.status}"
        if details.get("message"):
            message += "; service message: " + details["message"]
        return ProviderError(message, uncertain=response.status >= 500, diagnostics={"provider_error": details})
