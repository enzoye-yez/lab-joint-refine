"""有界 HTTPS 传输：公共地址检查、解析地址固定、禁止自动重定向。"""

import http.client
import ipaddress
import json
import socket
import ssl
import time
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass

from .errors import ProviderError, ValidationError


def public_target(url, resolver=socket.getaddrinfo):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment):
        raise ValidationError("网络请求仅允许不含凭据的 HTTPS 地址")
    try:
        addresses = resolver(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ValidationError("拒绝访问回环、内网、链路本地或保留网络地址")
    except (OSError, ValueError) as exc:
        raise ProviderError("无法解析服务地址") from exc
    return parsed, addresses[0][4][0]


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, hostname, address, port, timeout):
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        sock = socket.create_connection((self.address, self.port), self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


class ProxiedPinnedHTTPS(PinnedHTTPS):
    """HTTP CONNECT 隧道只承载 TLS 密文；目标 IP、证书主机名仍固定。"""
    def __init__(self, hostname, address, port, timeout, proxy):
        super().__init__(hostname, address, port, timeout)
        self.proxy = proxy

    def connect(self):
        started = time.monotonic()
        tunnel = http.client.HTTPConnection(*self.proxy, timeout=self.timeout)
        tunnel.set_tunnel(self.address, self.port)
        try:
            tunnel.connect()
            remaining = self.timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError()
            tunnel.sock.settimeout(remaining)
            self.sock = self._context.wrap_socket(tunnel.sock, server_hostname=self.host)
            tunnel.sock = None
        finally:
            tunnel.close()


def connection_for(parsed, address, timeout):
    # 仅官方 API 采用用户已配置的系统代理；其他供应商和资产下载保持原行为。
    if parsed.hostname == "api.openai.com" and (parsed.port or 443) == 443:
        proxy = getproxies().get("https")
        if proxy and not proxy_bypass(parsed.hostname):
            try:
                target = urlsplit(proxy if "://" in proxy else "http://" + proxy)
                port = target.port or 80
                valid = (target.scheme == "http" and target.hostname and
                         not target.username and not target.password and
                         target.path in ("", "/") and not target.query and not target.fragment)
            except ValueError:
                valid = False
            if not valid:
                raise ProviderError("官方接口的系统代理配置不受支持；需要不含凭据的 HTTP 隧道代理",
                                    diagnostics={"failure": "proxy_configuration", "phase": "connect"})
            return ProxiedPinnedHTTPS(parsed.hostname, address, 443, timeout, (target.hostname, port))
    return PinnedHTTPS(parsed.hostname, address, parsed.port or 443, timeout)


class Transport:
    def http_error(self, response, connection, key, deadline):
        """默认不读取错误正文；生成模块可扩展为有界、脱敏诊断。"""
        return ProviderError(f"服务返回 HTTP 状态 {response.status}", uncertain=response.status >= 500)

    def request(self, url, *, payload=None, key=None, timeout=120, max_bytes=8_000_000, accept=None):
        parsed, address = public_target(url)
        connection = connection_for(parsed, address, timeout)
        headers = {"Accept": accept or ("application/json" if payload is not None else "image/*"),
                   "User-Agent": "WetlabAssetPreparation/0.1"}
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if key:
            headers["Authorization"] = "Bearer " + key
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        started = time.monotonic()
        phase, status, received = 'sending', None, 0
        try:
            connection.request("POST" if payload is not None else "GET", target, body, headers)
            phase = 'waiting_headers'
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError()
            if connection.sock:
                connection.sock.settimeout(remaining)
            response = connection.getresponse()
            status = response.status
            phase = 'reading_body'
            if response.status != 200:
                # 不输出响应正文，防止服务回显凭据；重定向也不会携带密钥跳转。
                raise self.http_error(response, connection, key, started + timeout)
            data = bytearray()
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError()
                # read1 避免对端持续滴流使一次 read 长期不返回。
                if connection.sock:
                    connection.sock.settimeout(remaining)
                part = response.read1(min(65536, max_bytes + 1 - len(data)))
                if not part:
                    break
                data.extend(part)
                received = len(data)
                if len(data) > max_bytes:
                    raise ProviderError("响应超过大小限制", uncertain=payload is not None)
            return bytes(data), response.getheader("Content-Type", "")
        except ProviderError as exc:
            exc.diagnostics.update({'phase': phase, 'elapsed_seconds': round(time.monotonic() - started, 3),
                               'timeout_seconds': timeout, 'http_status': status, 'received_bytes': received,
                               'failure': 'http_status' if status != 200 and status is not None else 'response_limit'})
            raise
        except (OSError, http.client.HTTPException, ValueError) as exc:
            failure = ('timeout' if isinstance(exc, TimeoutError) else
                       'tls_error' if isinstance(exc, ssl.SSLError) else
                       'connection_error' if isinstance(exc, OSError) else 'protocol_error')
            detail = {'phase': phase, 'elapsed_seconds': round(time.monotonic() - started, 3),
                      'timeout_seconds': timeout, 'http_status': status, 'received_bytes': received,
                      'failure': failure, 'exception_type': type(exc).__name__}
            for code in ('errno', 'winerror'):
                if type(getattr(exc, code, None)) is int:
                    detail[code] = getattr(exc, code)
            label = {'timeout': '等待超时', 'tls_error': '加密连接失败',
                     'connection_error': '网络连接中断', 'protocol_error': '响应协议错误'}[failure]
            raise ProviderError(f"{label}；阶段={phase}，耗时={detail['elapsed_seconds']}秒，状态码={status}",
                                uncertain=payload is not None, diagnostics=detail) from exc
        finally:
            connection.close()

    def post(self, url, payload, key, timeout):
        raw, _ = self.request(url, payload=payload, key=key, timeout=timeout)
        try:
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError()
            return result
        except (ValueError, UnicodeError) as exc:
            raise ProviderError("服务未返回有效 JSON 对象", uncertain=True,
                                diagnostics={'phase': 'decode_json', 'failure': 'invalid_json',
                                             'received_bytes': len(raw)}) from exc
