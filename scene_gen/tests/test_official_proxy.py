"""开发者合成测试：代理路由、TLS 身份及失败边界，不调用模型。"""
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit
from asset_prep.network import connection_for, ProxiedPinnedHTTPS, Transport
from asset_prep.errors import ProviderError, ValidationError

class OfficialProxyTests(unittest.TestCase):
    def route(self, url="https://api.openai.com/v1/models", proxy="http://127.0.0.1:7890", bypass=False):
        with patch("asset_prep.network.getproxies", return_value={"https":proxy} if proxy else {}), patch("asset_prep.network.proxy_bypass",return_value=bypass):
            return connection_for(urlsplit(url), "8.8.8.8", 10)

    def test_system_proxy_only_official(self):
        c=self.route()
        self.assertIsInstance(c,ProxiedPinnedHTTPS)
        self.assertEqual(c.proxy,("127.0.0.1",7890))
        for url in ("https://example.org/v1/models", "https://api.openai.com.evil.org/", "https://api.openai.com:444/"):
            self.assertNotIsInstance(self.route(url),ProxiedPinnedHTTPS)

    def test_absent_or_bypassed_proxy_keeps_direct(self):
        self.assertNotIsInstance(self.route(proxy=None),ProxiedPinnedHTTPS)
        self.assertNotIsInstance(self.route(bypass=True),ProxiedPinnedHTTPS)

    def test_invalid_proxy_fails_without_echo_or_fallback(self):
        for proxy in ("socks5://localhost:1", "http://user:secret@localhost:1", "http://localhost:nope", "http://localhost/path", "http://localhost?secret"):
            with self.assertRaises(ProviderError) as caught:self.route(proxy=proxy)
            self.assertNotIn("secret",str(caught.exception))
            self.assertFalse(caught.exception.uncertain)

    def test_tunnel_pins_ip_but_tls_checks_original_hostname(self):
        c=self.route(); tunnel=Mock(); raw=Mock(); tunnel.sock=raw
        context=Mock(); c._context=context
        with patch("asset_prep.network.http.client.HTTPConnection",return_value=tunnel):c.connect()
        tunnel.set_tunnel.assert_called_once_with("8.8.8.8",443)
        context.wrap_socket.assert_called_once_with(raw,server_hostname="api.openai.com")
        self.assertIs(c.sock,context.wrap_socket.return_value)
        tunnel.close.assert_called_once()
        self.assertIsNone(tunnel.sock)

    def test_tls_failure_closes_tunnel(self):
        c=self.route();tunnel=Mock();c._context=Mock()
        c._context.wrap_socket.side_effect=OSError("TLS failed")
        with patch("asset_prep.network.http.client.HTTPConnection",return_value=tunnel),self.assertRaises(OSError):c.connect()
        tunnel.close.assert_called_once()
        self.assertIsNone(c.sock)

    def test_connect_failure_no_retry_or_direct_fallback(self):
        c=self.route();tunnel=Mock();tunnel.connect.side_effect=TimeoutError()
        with patch("asset_prep.network.http.client.HTTPConnection",return_value=tunnel) as factory,self.assertRaises(TimeoutError):c.connect()
        self.assertEqual(factory.call_count,1);tunnel.close.assert_called_once()

    def test_public_validation_precedes_proxy(self):
        with patch("asset_prep.network.public_target",side_effect=ValidationError("private")),patch("asset_prep.network.connection_for") as route,self.assertRaises(ValidationError):
            Transport().request("https://api.openai.com/v1/models")
        route.assert_not_called()

    def test_redirect_not_followed_and_key_only_inside_request(self):
        c=Mock();r=Mock(status=302);c.getresponse.return_value=r
        with patch("asset_prep.network.public_target",return_value=(urlsplit("https://api.openai.com/v1/models"),"8.8.8.8")),patch("asset_prep.network.connection_for",return_value=c),self.assertRaises(ProviderError):
            Transport().request("https://api.openai.com/v1/models",key="fixture-key")
        c.request.assert_called_once()
        self.assertEqual(c.request.call_args.args[3]["Authorization"],"Bearer fixture-key")
        c.close.assert_called_once()

    def test_expired_connect_budget_closes_socket_before_tls(self):
        c=self.route();c._context=Mock();tunnel=Mock()
        with patch("asset_prep.network.http.client.HTTPConnection",return_value=tunnel),patch("asset_prep.network.time.monotonic",side_effect=[0,11]),self.assertRaises(TimeoutError):c.connect()
        c._context.wrap_socket.assert_not_called();tunnel.close.assert_called_once()
