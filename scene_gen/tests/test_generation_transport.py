import json
import io
from pathlib import Path
import subprocess
import socket
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from asset_generation.contracts import GenerationConfig
from asset_generation.model import GenerationModel
from asset_generation.transport import GenerationTransport
from asset_prep.config import Config
from asset_prep.errors import BudgetExceeded,ProviderError,UncertainRequest
from asset_prep.storage import Store


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store=Store(Path(self.temp.name),{},Config(max_model_calls=10,max_reserved_tokens=1000000))

    def model(self,transport):
        return GenerationModel(Config(),GenerationConfig(),{"HARNESS_MODEL_API_KEY":"synthetic-secret"},self.store,transport)

    def test_agent_author_gets_the_whole_context_in_one_request(self):
        from dataclasses import replace
        class Agent:
            sent=[]
            def whole_context(self,phase):return phase.startswith("asset:")
            def post(self,url,payload,key,timeout):
                self.sent.append(json.loads(payload["messages"][1]["content"][0]["text"]))
                return {"choices":[{"message":{"content":"{\"model_py\":\"x\"}"},"finish_reason":"stop"}],
                        "usage":{"prompt_tokens":1,"completion_tokens":1}}
        model=self.model(Agent())
        model.generation=replace(model.generation,max_context_tokens=50000,max_request_bytes=60000)
        context={"dependencies":{"A01":"x"*200000},"current_model_py":None,"profile":{}}
        model.ask("author",context,[],"asset:tool_right")
        self.assertEqual(len(Agent.sent),1)
        self.assertEqual(Agent.sent[0]["dependencies"],context["dependencies"])

    def test_429_retries_are_bounded_and_recorded(self):
        class Limited:
            def post(self,*args):
                raise ProviderError("limited",diagnostics={"http_status":429})
        with patch("asset_generation.model.time.sleep") as sleep:
            with self.assertRaises(ProviderError):
                self.model(Limited()).ask("synthetic",{},[],"test")
        self.assertEqual(len(self.store.state["calls"]),6)
        self.assertTrue(all(c["status"] == "failed" for c in self.store.state["calls"]))
        self.assertEqual(sum(c.args[0] for c in sleep.call_args_list),225)
        self.assertTrue(all(c.args[0]<=1 for c in sleep.call_args_list))

    def test_known_transient_failures_can_recover_after_eight_rejections(self):
        from dataclasses import replace
        from asset_generation.model import request_summary
        class Eventually:
            count = 0
            def post(self, *args):
                self.count += 1
                if self.count <= 8:
                    raise ProviderError("synthetic rejection", diagnostics={"http_status":503})
                return {"choices":[{"message":{"content":"{}"}}]}
        transport = Eventually()
        model = self.model(transport)
        model.generation = replace(model.generation, max_transport_retries=30)
        with patch("asset_generation.model.time.sleep"):
            self.assertEqual(model.ask("synthetic", {}, [], "asset:test")[0], {})
        calls = self.store.state["calls"]
        self.assertEqual(transport.count, 9)
        self.assertEqual(len({c["request_sha256"] for c in calls}), 1)
        self.assertEqual(request_summary(calls), {"attempts":9, "successful_responses":1,
            "failed_attempts":8, "unknown_attempts":0, "retry_attempts":8, "not_sent":0, "logical_requests":1})

    def test_unknown_400_has_separate_small_retry_limit(self):
        from dataclasses import replace
        class Rejected:
            def post(self, *args):
                raise ProviderError("unknown rejection", diagnostics={"http_status":400})
        model = self.model(Rejected())
        model.generation = replace(model.generation, max_transport_retries=30, max_unknown_http_retries=2)
        with patch("asset_generation.model.time.sleep") as sleep, self.assertRaises(ProviderError):
            model.ask("synthetic", {}, [], "test")
        self.assertEqual(len(self.store.state["calls"]), 3)
        self.assertEqual(sum(c.args[0] for c in sleep.call_args_list),45)

    def test_explicit_quota_or_invalid_key_does_not_enter_long_retries(self):
        from asset_generation.model import request_failure_kind
        for status, code in ((429,"insufficient_quota"), (400,"invalid_api_key"), (400,"invalid_request_error")):
            error = ProviderError("rejected", diagnostics={"http_status":status, "provider_error":{"code":code}})
            self.assertEqual(request_failure_kind(error), "permanent")

    def test_transient_unknown_requires_explicit_retry_then_success_is_cached(self):
        for failure in ("timeout", "process_deadline", "connection_error", "protocol_error",
                        "network_worker_exit", "invalid_json"):
            with self.subTest(failure=failure):
                class Eventually:
                    count=0
                    def post(self,*args):
                        self.count+=1
                        if self.count == 1:
                            raise ProviderError("unknown",uncertain=True,diagnostics={"failure":failure})
                        return {"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}
                transport=Eventually()
                with patch("asset_generation.model.time.sleep"):
                    model=self.model(transport)
                    with self.assertRaises(UncertainRequest):
                        model.ask("synthetic",{},[],failure)
                    self.assertEqual(transport.count,1)
                    with self.assertRaises(UncertainRequest):
                        model.ask("changed context",{},[],failure)
                    self.store.retry_uncertain=True
                    self.assertEqual(model.ask("synthetic",{},[],failure)[0],{})
                    self.store.retry_uncertain=False
                    self.assertEqual(model.ask("synthetic",{},[],failure)[0],{})
                self.assertEqual(transport.count,2)
                prior,done=self.store.state["calls"][-2:]
                self.assertEqual(prior["status"],"unknown_retried")
                self.assertTrue(prior["outcome_unknown"])
                self.assertEqual(done["retry_of"],prior["index"])
                self.assertEqual(prior["retried_by"],done["index"])
                self.assertEqual(prior["request_sha256"],done["request_sha256"])
                self.assertEqual(done["status"],"done")
                # 每个子用例独立使用同样预算。
                self.store.state["calls"]=[]
                self.store.save()

    def test_exhausted_timeout_retains_pending_and_full_budget(self):
        class Unknown:
            def post(self,*args):
                raise ProviderError("timeout",uncertain=True,diagnostics={"failure":"timeout"})
        with patch("asset_generation.model.time.sleep") as sleep:
            with self.assertRaisesRegex(UncertainRequest,"不自动重发"):
                self.model(Unknown()).ask("synthetic",{},[],"test")
        calls=self.store.state["calls"]
        self.assertEqual([c["status"] for c in calls],["pending"])
        sleep.assert_not_called()
        self.assertEqual(self.store.totals()["model_calls"],1)
        self.assertEqual(self.store.totals()["accounted_tokens"],sum(c["reserved_tokens"] for c in calls))

    def test_budget_exhaustion_prevents_resend_and_keeps_unknown(self):
        from dataclasses import replace
        self.store.config=replace(self.store.config,max_model_calls=1)
        class Unknown:
            count=0
            def post(self,*args):
                self.count+=1
                raise ProviderError("timeout",uncertain=True,diagnostics={"failure":"timeout"})
        transport=Unknown()
        with patch("asset_generation.model.time.sleep"):
            with self.assertRaises(UncertainRequest):
                self.model(transport).ask("synthetic",{},[],"test")
            self.store.retry_uncertain=True
            with self.assertRaises(BudgetExceeded):
                self.model(transport).ask("synthetic",{},[],"test")
        self.assertEqual(transport.count,1)
        self.assertEqual(self.store.state["calls"][0]["status"],"pending")

    def test_http_retry_classification(self):
        from asset_generation.model import transient_error
        self.assertTrue(transient_error(ProviderError("body timeout",uncertain=True,
                         diagnostics={"http_status":200,"failure":"timeout","phase":"reading_body"})))
        for status in (408,429,500,502,503,504,529):
            self.assertTrue(transient_error(ProviderError("synthetic",diagnostics={"http_status":status})))
        for status in (400,401,403,404,413,422,501,505):
            class Rejected:
                def post(self,*args):
                    raise ProviderError("rejected",diagnostics={"http_status":status})
            with patch("asset_generation.model.time.sleep") as sleep:
                with self.assertRaises(ProviderError):
                    self.model(Rejected()).ask("synthetic",{},[],str(status))
            sleep.assert_not_called()

    def test_success_after_503(self):
        class Eventually:
            count=0
            def post(self,*args):
                self.count+=1
                if self.count == 1:
                    raise ProviderError("unavailable",diagnostics={"http_status":503})
                return {"choices":[{"message":{"content":"{}"}}]}
        with patch("asset_generation.model.time.sleep"):
            self.model(Eventually()).ask("synthetic",{},[],"test")
        self.assertEqual([c["status"] for c in self.store.state["calls"]],["failed","done"])

    def test_unknown_never_enters_backoff(self):
        class Unknown:
            def post(self,*args):
                raise ProviderError("timeout",uncertain=True,diagnostics={"failure":"timeout"})
        with patch("asset_generation.model.time.sleep",side_effect=KeyboardInterrupt):
            with self.assertRaises(UncertainRequest):
                self.model(Unknown()).ask("synthetic",{},[],"test")
        self.assertEqual(self.store.state["calls"][0]["status"],"pending")

    def test_unknown_result_never_auto_retries(self):
        class Unknown:
            def post(self,*args):
                raise ProviderError("unknown",uncertain=True)
        with patch("asset_generation.model.time.sleep") as sleep:
            with self.assertRaises(UncertainRequest):
                self.model(Unknown()).ask("synthetic",{},[],"test")
        self.assertEqual(len(self.store.state["calls"]),1)
        self.assertEqual(self.store.state["calls"][0]["status"],"pending")
        sleep.assert_not_called()

    def test_success_after_429_keeps_failed_reservation(self):
        class Eventually:
            count=0
            def post(self,*args):
                self.count+=1
                if self.count == 1:
                    raise ProviderError("limited",diagnostics={"http_status":429})
                return {"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}],"usage":{"total_tokens":5,"completion_tokens":2}}
        with patch("asset_generation.model.time.sleep"):
            self.assertEqual(self.model(Eventually()).ask("synthetic",{},[],"test")[0],{})
        self.assertEqual([c["status"] for c in self.store.state["calls"]],["failed","done"])

    def test_deadline_raises_unknown_without_secret_in_command(self):
        from unittest.mock import MagicMock
        child=MagicMock();child.args=[]
        child.communicate.side_effect=subprocess.TimeoutExpired([],1)
        def spawn(command,**kwargs):
            self.assertNotIn("synthetic-secret",str(command))
            self.assertNotIn("synthetic-secret",str(kwargs["env"]))
            return child
        with patch("asset_generation.transport.subprocess.Popen",side_effect=spawn), \
             patch("asset_generation.process_tree.ProcessTree") as tree, \
             patch("asset_generation.transport.time.monotonic",side_effect=[0,0,9,9]):
            with self.assertRaises(ProviderError) as error:
                GenerationTransport().post("https://example.invalid",{},"synthetic-secret",3)
        self.assertEqual(json.loads(child.communicate.call_args_list[0].kwargs["input"])["key"],"synthetic-secret")
        self.assertTrue(error.exception.uncertain)
        self.assertEqual(error.exception.diagnostics["failure"],"process_deadline")
        tree.return_value.stop.assert_called_once_with(child)

    def test_worker_http_failure_survives_process_boundary(self):
        from unittest.mock import MagicMock
        response={"ok":False,"error":{"message":"limited","uncertain":False,"diagnostics":{"http_status":429}}}
        child=MagicMock();child.returncode=0;child.args=[]
        child.communicate.return_value=(json.dumps(response).encode(),b"")
        with patch("asset_generation.transport.subprocess.Popen",return_value=child), \
             patch("asset_generation.process_tree.ProcessTree"):
            with self.assertRaises(ProviderError) as error:
                GenerationTransport().post("https://example.invalid",{},"synthetic-secret",3)
        self.assertFalse(error.exception.uncertain)
        self.assertEqual(error.exception.diagnostics["http_status"],429)

    def test_provider_details_survive_ledger_without_secrets(self):
        class Rejected:
            def post(self, *args):
                raise ProviderError("rejected", diagnostics={"http_status":400,
                    "provider_error":{"code":"invalid_request", "message":"bad synthetic-secret"}})
        with self.assertRaises(ProviderError):
            self.model(Rejected()).ask("synthetic", {}, [], "scene_plan")
        record = self.store.state["calls"][-1]
        self.assertEqual(record["provider_error"]["code"], "invalid_request")
        self.assertNotIn("synthetic-secret", (self.store.root / "state.json").read_text())

    def test_worker_dns_failure_is_retryable_without_secret_output(self):
        from asset_generation.transport import worker
        from asset_generation.model import transient_error
        def dns_failure(*args):
            try:
                raise socket.gaierror("synthetic lookup failure")
            except socket.gaierror as exc:
                raise ProviderError("无法解析服务地址") from exc
        request={"url":"https://example.invalid","payload":{},"key":"synthetic-secret","timeout":3}
        stdin=SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode()))
        stdout=SimpleNamespace(buffer=io.BytesIO())
        with patch("asset_generation.transport.sys.stdin",stdin), \
             patch("asset_generation.transport.sys.stdout",stdout), \
             patch("asset_prep.network.Transport.post",side_effect=dns_failure):
            worker()
        raw=stdout.buffer.getvalue()
        self.assertNotIn(b"synthetic-secret",raw)
        error=json.loads(raw)["error"]
        self.assertFalse(error["uncertain"])
        self.assertTrue(transient_error(ProviderError(error["message"],diagnostics=error["diagnostics"])))

    def test_nontransient_unknown_failures_are_not_retried(self):
        from asset_generation.model import transient_error
        for failure in ("tls_error","response_limit","unclassified"):
            self.assertFalse(transient_error(ProviderError("synthetic",uncertain=True,
                                                         diagnostics={"failure":failure})))


if __name__ == "__main__":
    unittest.main()
