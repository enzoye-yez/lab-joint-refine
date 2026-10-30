"""完整场景入口。默认离线预检；正式模型执行须用户显式选择。"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from asset_generation.contracts import GenerationConfig, read_json
from asset_generation.sandbox import DockerSandbox, SandboxUnavailable
from asset_generation.local import LocalBackend
from asset_generation.checks import load_model
from asset_prep.config import Config, credentials
from asset_prep.errors import PrepError, BudgetExceeded, UncertainRequest, ProviderLimitError
from asset_generation.transport import transport_for
from asset_prep.storage import run_directory, run_lock, atomic_json, safe_path
from asset_prep import implementation
from .contracts import SceneConfig, environment, numerical_contract
from .inputs import catalog
from .pipeline import SceneRun


def main(argv=None):
    parser = argparse.ArgumentParser(description="完整场景生成与受限修复；评测跳过资产提取与搜索")
    parser.add_argument("--prepare-robot",action="store_true",help="按环境核验整机资源；旧版环境下载固定FR3，不调用模型")
    parser.add_argument("--package",type=Path)
    parser.add_argument("--protocol-pdf",type=Path,help="原文PDF：按页完整解析作为场景上下文，不调用资产提取")
    parser.add_argument("--run")
    parser.add_argument("--environment",type=Path)
    parser.add_argument("--checks",type=Path,help="用户固定的数值/配合/操作姿态验收")
    parser.add_argument("--config",type=Path,help="场景预算与检查限制")
    parser.add_argument("--generation-config",type=Path)
    parser.add_argument("--provider-config",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--expected-implementation", help="正式执行必须指定离线验证清单中的实现摘要")
    parser.add_argument("--backend", choices=("docker", "local"), default="docker",
                        help="执行后端；local 为显式本机进程模式，不隔离宿主文件和网络")
    parser.add_argument("--static-only",action="store_true",help="仅验收静态场景；跳过动态交互、重力步进和操作姿态求解")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-uncertain",action="store_true")
    parser.add_argument("--feedback",type=Path,help="绑定场景摘要的规划器失败反馈")
    for key in ("max-model-calls","max-reserved-tokens","max-iterations","max-plan-attempts","max-repair-plan-attempts","max-evidence-rounds","max-evidence-plan-attempts"):
        parser.add_argument("--"+key,type=int)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    try:
        if args.execute:
            active = getattr(sys, '_harness_runtime', None)
            if not active or not args.expected_implementation:
                raise PrepError('正式执行须使用 run.ps1，并提供 --expected-implementation 已验证源码摘要')
            implementation.verify(args.expected_implementation, project=project, where='正式执行启动')
        if args.prepare_robot:
            if args.execute or args.package or args.protocol_pdf or args.run or args.resume or args.feedback:
                parser.error("机器人准备须单独运行")
            from .robot import prepare_environment
            env = environment(read_json(args.environment or project/"config/scene_environment.json"))
            print(json.dumps({"robot_directory":str(prepare_environment(project,env))},ensure_ascii=False))
            return 0
        if not args.package or not args.run:
            parser.error("需要--package和--run")
        if (args.resume or args.retry_uncertain or args.feedback) and not args.execute:
            parser.error("恢复与反馈需要正式执行")
        if args.feedback and not args.resume:
            parser.error("规划反馈须恢复已有场景")
        provider = Config.load(args.provider_config or project/"config/default.json")
        generation = GenerationConfig.load(args.generation_config or project/"config/generation.json")
        limits = SceneConfig(**read_json(args.config or project/"config/scene.json"))
        limits = replace(limits,**{k:getattr(args,k) for k in ("max_model_calls","max_reserved_tokens","max_iterations","max_plan_attempts","max_repair_plan_attempts","max_evidence_rounds","max_evidence_plan_attempts") if getattr(args,k) is not None})
        if args.static_only:
            limits = replace(limits,validation_mode="static")
        if args.feedback and limits.validation_mode == "static":
            parser.error("静态预览不接受操作规划反馈；完整验证请使用新的运行目录")
        env = environment(read_json(args.environment or project/"config/scene_environment.json"))
        contract = numerical_contract(read_json(args.checks) if args.checks else {})
        catalogue = catalog(args.package,generation,args.protocol_pdf,provider.max_document_chars)
        robot = safe_path(project,env["robot"]["directory"])
        from .benchmark_input import bind_environment
        bind_environment(catalogue, env, robot)
        sandbox = (LocalBackend if args.backend == "local" else DockerSandbox)(generation,project)
        root = run_directory(project,args.run)
        with run_lock(root):
            if not args.execute:
                if any(p.name != ".lock" for p in root.iterdir()):
                    raise PrepError("预检目录已有产物，请换新名称")
                try:
                    backend = sandbox.preflight()
                except SandboxUnavailable as exc:
                    backend = {"available":False,"reason":str(exc)}
                robot_ready = False
                if robot.is_dir():
                    load_model(robot)
                    robot_ready = True
                report = {"status":"input_checked","asset_ids":catalogue["asset_ids"],"identity":catalogue["identity"],
                          "implementation":implementation.runtime_record(project, inputs={}, configuration={})['implementation'],
                          "robot_ready":robot_ready,"sandbox":backend,"provider":provider.provider,"model":provider.model,
                          "model_calls":0,"extraction_calls":0,"search_calls":0,"environment":env,
                          "validation_mode":limits.validation_mode,
                          "protocol":({"files":catalogue["protocol"]["files"],
                                       "pages":len(catalogue["protocol"]["sources"]),
                                       "characters":catalogue["protocol"]["parsing"]["total_characters"],
                                       "warnings":catalogue["protocol"]["warnings"]} if catalogue["protocol"] else None),
                          "note":"只读预检不认证生成质量；未读取凭据或调用模型"}
                atomic_json(root/"preflight.json",report)
                print(json.dumps({k:report[k] for k in ("status","asset_ids","protocol","validation_mode","robot_ready","sandbox","model_calls")},ensure_ascii=False))
                return 0
            sandbox.preflight()
            load_model(robot)
            run = SceneRun(catalogue,env,contract,provider,generation,limits,credentials(project),root,transport_for(provider),
                           sandbox,robot,project,args.resume,args.retry_uncertain,
                           expected_implementation=args.expected_implementation)
            result = run.run(read_json(args.feedback) if args.feedback else None)
            print(json.dumps(result,ensure_ascii=False))
            return 0 if result["status"] in {"candidate_ready", "static_candidate_ready", "static_preview_ready"} else 3
    except (BudgetExceeded,UncertainRequest,ProviderLimitError,SandboxUnavailable) as exc:
        print(str(exc),file=sys.stderr)
        return 3
    except (PrepError,OSError,ValueError,KeyError,TypeError) as exc:
        print(str(exc),file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("用户中断；调用账本和已归档版本保留",file=sys.stderr)
        return 130
