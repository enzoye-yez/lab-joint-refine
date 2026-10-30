"""通过 run.ps1 generate 调用；默认离线预检，正式调用须显式选择。"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from asset_prep.config import Config, credentials
from asset_prep.errors import PrepError, BudgetExceeded, UncertainRequest, ProviderLimitError
from .transport import transport_for
from asset_prep.storage import atomic_json, run_directory, run_lock
from .contracts import GenerationConfig, check_profile, read_json
from .inputs import load_package, public_bundle
from .pipeline import execute
from .sandbox import DockerSandbox, SandboxUnavailable
from .local import LocalBackend


def main(argv=None):
    parser = argparse.ArgumentParser(description="交互资产生成；不调用资产提取，不运行正式机器人实验")
    parser.add_argument("--package", type=Path, required=True, help="修订资料包Markdown或资产提取JSON输出")
    parser.add_argument("--asset", action="append", required=True, help="资产编号；可重复指定紧密配合的资产组")
    parser.add_argument("--run", required=True, help="独立运行名称")
    parser.add_argument("--checks", type=Path, help="可选的只读数值验收配置")
    parser.add_argument("--config", type=Path, help="生成配置")
    parser.add_argument("--provider-config", type=Path, help="复用资产提取服务配置")
    parser.add_argument("--execute", action="store_true", help="正式模型调用，由用户执行；默认仅离线预检")
    parser.add_argument("--backend", choices=("docker", "local"), default="docker",
                        help="执行后端；local 为显式本机进程模式，不隔离宿主文件和网络")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-uncertain", action="store_true", help="确认愿意承担重复请求后显式重试")
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--max-reserved-tokens", type=int)
    parser.add_argument("--max-iterations", type=int)
    args = parser.parse_args(argv)
    if (args.resume or args.retry_uncertain) and not args.execute:
        parser.error("恢复和未知请求重试仅用于正式执行")
    project = Path(__file__).resolve().parents[2]
    try:
        generation = GenerationConfig.load(args.config or project/"config/generation.json")
        changes = {name: getattr(args, name) for name in ("max_model_calls", "max_reserved_tokens", "max_iterations")
                   if getattr(args, name) is not None}
        generation = replace(generation, **changes)
        config = Config.load(args.provider_config or project/"config/default.json")
        from scene_generation.benchmark_input import load_public, attach_bundle
        public = load_public(args.package)
        bundle = attach_bundle(load_package(args.package, args.asset, generation), public)
        profile = check_profile(read_json(args.checks) if args.checks else {})
        root = run_directory(project, args.run)
        sandbox = (LocalBackend if args.backend == "local" else DockerSandbox)(generation, project)
        with run_lock(root):
            if not args.execute:
                if any(p.name != ".lock" for p in root.iterdir()):
                    raise PrepError("预检目录已有产物，请使用新运行名称")
                try:
                    backend = sandbox.preflight()
                except SandboxUnavailable as exc:
                    backend = {"available": False, "reason": str(exc)}
                report = {"status": "input_checked", "input": public_bundle(bundle), "profile": profile,
                          "sandbox": backend, "model_calls": 0, "max_model_calls": generation.max_model_calls,
                          "max_iterations": generation.max_iterations,
                          "note": "不证明资产语义完整或生成质量；未读取凭据或调用模型"}
                atomic_json(root/"preflight.json", report)
                print(json.dumps({"status": report["status"], "output": str(root/"preflight.json"),
                                  "images": len(bundle["images"]), "model_calls": 0, "sandbox": backend}, ensure_ascii=False))
                return 0
            sandbox.preflight()
            result = execute(bundle, profile, config, generation, credentials(project), root, transport_for(config),
                             sandbox, project, resume=args.resume, retry_uncertain=args.retry_uncertain)
            print(json.dumps({k: result[k] for k in ("status", "iterations", "latest_revision", "budget")}, ensure_ascii=False))
            return 0 if result["status"] == "candidate_ready" else 3
    except (BudgetExceeded, UncertainRequest, ProviderLimitError, SandboxUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (PrepError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("用户中断；已完成请求和版本保留", file=sys.stderr)
        return 130
