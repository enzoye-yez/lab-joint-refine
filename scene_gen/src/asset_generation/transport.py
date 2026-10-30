"""生成模块专用传输进程：截止时终止网络进程，凭据仅通过标准输入传递。"""

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
        # 不通过命令行、进程环境或临时文件传递本次密钥。
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
                    raise ProviderError("生成请求超过进程级截止时间；网络进程已终止，请求结果仍不明",uncertain=True,
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
            raise ProviderError("生成网络进程异常退出；请求结果不明",uncertain=True,
                                diagnostics={"failure":"network_worker_exit","exit_code":result.returncode})
        try:
            response = json.loads(result.stdout)
            if response["ok"]:
                return response["response"]
            error = response["error"]
            raise ProviderError(error["message"],uncertain=error["uncertain"],diagnostics=error["diagnostics"])
        except (ValueError,KeyError,TypeError):
            raise ProviderError("生成网络进程返回格式无效；请求结果不明",uncertain=True) from None


def transport_for(config, workspace=None):
    """按服务配置选择传输：codex 走本机登录的 Codex CLI，其余走 HTTPS 网络进程。

    workspace 保存编码 agent 会话目录（通常是运行目录），便于事后审计。
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
        # 不回显请求、密钥或含请求变量的异常堆栈。
        output = {"ok":False,"error":{"message":"网络进程发生未分类错误","uncertain":True,"diagnostics":{}}}
    sys.stdout.buffer.write(json.dumps(output,ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    worker()
