"""Transport process dedicated to generation: kills the network process at the deadline; credentials are passed only via stdin."""

import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import time

from asset_prep.errors import ProviderError


class GenerationTransport:
    def __init__(self):
        self.check_cancel = lambda: None

    def post(self, url, payload, key, timeout):
        started = time.monotonic()
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        # Never pass this request's key via the command line, process environment or temporary files.
        for name in ("HARNESS_MODEL_API_KEY", "HARNESS_MODEL_API_KEY2", "TAVILY_API_KEY", "OPENAI_API_KEY"):
            env.pop(name,None)
        request = json.dumps({"url":url,"payload":payload,"key":key,"timeout":timeout},ensure_ascii=False).encode("utf-8")
        from .process_tree import ProcessTree
        tree=ProcessTree()
        process=None
        try:
            self.check_cancel()
            process=subprocess.Popen([sys.executable,"-m","asset_generation.transport"],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=os.name!="nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            tree.attach(process)
            # communicate retains its input buffer across TimeoutExpired. The request
            # is written once; polling cancellation never resends it.
            pending=request
            while True:
                self.check_cancel()
                left=timeout+5-(time.monotonic()-started)
                if left<=0:
                    raise ProviderError("Generation request exceeded the process-level deadline; network process killed, request outcome still unknown",uncertain=True,
                        diagnostics={"failure":"process_deadline","phase":"network_worker",
                            "elapsed_seconds":round(time.monotonic()-started,3),"timeout_seconds":timeout})
                try:
                    stdout,stderr=process.communicate(input=pending,timeout=min(.25,left))
                    result=subprocess.CompletedProcess(process.args,process.returncode,stdout,stderr)
                    break
                except subprocess.TimeoutExpired:
                    pending=None
        finally:
            if process is not None:tree.stop(process)
            else:tree.close()
        if result.returncode != 0 or len(result.stdout) > 16000000:
            raise ProviderError("Generation network process exited abnormally; request outcome unknown",uncertain=True,
                                diagnostics={"failure":"network_worker_exit","exit_code":result.returncode})
        try:
            response = json.loads(result.stdout)
            if response["ok"]:
                return response["response"]
            error = response["error"]
            raise ProviderError(error["message"],uncertain=error["uncertain"],diagnostics=error["diagnostics"])
        except (ValueError,KeyError,TypeError):
            raise ProviderError("Generation network process returned an invalid format; request outcome unknown",uncertain=True) from None


def transport_for(config, workspace=None):
    """Choose the transport from the provider config: codex uses the locally logged-in Codex CLI, everything else uses the HTTPS network process.

    workspace holds the coding agent session directory (usually the run directory) for later audit.
    """
    if config.provider == "codex":
        from .codex_transport import CodexTransport
        return CodexTransport(agent=config.codex_agent_author, workspace_root=workspace)
    return GenerationTransport()


def worker():
    from asset_generation.provider_diagnostics import DiagnosticTransport as Transport
    try:
        request = json.loads(sys.stdin.buffer.read(40000000))
        result = Transport().post(request["url"],request["payload"],request["key"],request["timeout"])
        output = {"ok":True,"response":result}
    except ProviderError as exc:
        if isinstance(exc.__cause__, socket.gaierror) and not exc.diagnostics:
            exc.diagnostics = {"failure": "dns_error", "phase": "resolve"}
        output = {"ok":False,"error":{"message":str(exc),"uncertain":exc.uncertain,"diagnostics":exc.diagnostics}}
    except Exception:
        # Never echo the request, key or exception tracebacks containing request variables.
        output = {"ok":False,"error":{"message":"Unclassified error in the network process","uncertain":True,"diagnostics":{}}}
    sys.stdout.buffer.write(json.dumps(output,ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    worker()
