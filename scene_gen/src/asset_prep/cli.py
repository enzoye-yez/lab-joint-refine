"""命令行入口；不从旧工程导入任何代码或配置。"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import Config, credentials
from .documents import load_documents
from .errors import BudgetExceeded, PrepError, UncertainRequest, ProviderLimitError
from .network import Transport
from .pipeline import execute, run_identity
from .preflight import inspect_inputs
from .storage import atomic_json, run_directory, run_lock, safe_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="从实验规程提取资产并检索参考资料")
    parser.add_argument("--pdf", type=Path, help="主要事实来源")
    parser.add_argument("--xml", type=Path, help="可选备用文件；默认不读取")
    parser.add_argument("--xml-reason", help="明确说明 PDF 中待补证的信息缺口")
    parser.add_argument("--xml-query", action='append', default=[], help="补证检索词，可重复；仅选取命中的 XML 文本块")
    parser.add_argument("--preflight", action="store_true", help="仅本地检查输入及提取分段，不读取密钥或调用模型")
    parser.add_argument('--extract-only', action='store_true', help='只完成资产提取及合并，保存后续预算计划，不检索')
    parser.add_argument('--probe-output-limit', action='store_true', help='付费小规模接口探测，最多4次，不读取规程')
    parser.add_argument('--probe-parameter', choices=('max_tokens', 'max_completion_tokens'),
                        help='仅探测指定输出参数，最多2次；须配合 --probe-output-limit')
    parser.add_argument("--dataset", type=Path, help="仅用于预检：含案例子目录及 source.pdf/source.xml 的集合")
    parser.add_argument("--run", required=True, help="独立运行名称")
    parser.add_argument("--config", type=Path)
    parser.add_argument('--max-model-calls', type=int, help='本次显式模型调用预算，恢复时可提高')
    parser.add_argument('--max-reserved-tokens', type=int, help='本次显式保守令牌占用预算，非真实费用上限')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upgrade-run", action="store_true", help="仅离线升级已支持版本的暂停检查点，不调用模型")
    parser.add_argument("--retry-uncertain", action="store_true", help="确认后重发结果不明的请求，可能重复计费")
    args = parser.parse_args(argv)
    if args.probe_parameter and not args.probe_output_limit:
        parser.error('--probe-parameter 仅用于 --probe-output-limit')
    if args.probe_output_limit and any((args.pdf, args.xml, args.dataset, args.preflight, args.upgrade_run, args.extract_only)):
        parser.error('接口探测不能与规程输入或其他模式组合')
    if args.extract_only and (args.preflight or args.upgrade_run):
        parser.error('仅提取不能与离线检查模式组合')
    if args.xml_reason or args.xml_query:
        if not (args.xml and args.xml_reason and args.xml_query) or args.dataset:
            parser.error('定向补证须同时指定 --xml、--xml-reason 和 --xml-query，且仅用于单份输入')
    if args.upgrade_run and (args.resume or args.retry_uncertain or args.preflight or args.dataset):
        parser.error('--upgrade-run 只做离线升级，不能与执行或预检选项混用')
    if args.dataset and (not args.preflight or args.pdf or args.xml):
        parser.error("--dataset 仅用于 --preflight，不能同时指定 --pdf/--xml")
    if not args.dataset and not args.pdf and not args.probe_output_limit:
        parser.error("请指定 --pdf，或使用 --preflight --dataset；XML 为可选备用文件")
    if args.preflight and (args.resume or args.retry_uncertain):
        parser.error("离线预检使用新的运行名称，不使用恢复参数")
    root = Path(__file__).resolve().parents[2]
    try:
        config = Config.load(args.config or root / "config/default.json")
        overrides = {name:getattr(args, name) for name in ('max_model_calls', 'max_reserved_tokens')
                     if getattr(args, name) is not None}
        if overrides:
            config = replace(config, **overrides)
        if args.probe_output_limit:
            from .probe import probe_limits
            run_root = run_directory(root, args.run)
            with run_lock(run_root):
                report = probe_limits(config, credentials(root), run_root, Transport(),
                                      resume=args.resume, retry_uncertain=args.retry_uncertain, parameter=args.probe_parameter)
            print(json.dumps({'status':report['status'], 'output':str(run_root/'output_limit_probe.json'),
                              'budget':report['budget']}, ensure_ascii=False))
            return 0 if report['status']=='finished_requires_review' else 3
        if args.preflight:
            run_root = run_directory(root, args.run)
            with run_lock(run_root):
                if any(p.name != ".lock" for p in run_root.iterdir()):
                    raise PrepError("运行目录已有产物；预检请使用新的运行名称")
                report = inspect_inputs(config, dataset=args.dataset, pdf=args.pdf, xml=args.xml,
                                        xml_reason=args.xml_reason, xml_queries=args.xml_query)
                output = safe_path(run_root, "preflight.json")
                atomic_json(output, report)
            print(json.dumps({"output": str(output), "total": report["total"],
                              "input_checked": report["input_checked"], "blocked": report["blocked"],
                              "model_calls": 0}, ensure_ascii=False))
            return 2 if report["blocked"] else 0
        documents = load_documents(args.pdf.resolve(), args.xml.resolve() if args.xml else None,
                                   config.max_document_chars, xml_reason=args.xml_reason, xml_queries=args.xml_query)
        if args.upgrade_run:
            from .upgrade import upgrade_run
            run_root = run_directory(root, args.run)
            with run_lock(run_root):
                print(json.dumps(upgrade_run(run_root, run_identity(documents, config), documents, config), ensure_ascii=False))
            return 0
        keys = credentials(root)
        run_root = run_directory(root, args.run)
        with run_lock(run_root):
            package = execute(documents, config, keys, run_root, Transport(), resume=args.resume,
                              retry_uncertain=args.retry_uncertain, extract_only=args.extract_only)
        print(json.dumps({"status": package["status"], "output": str(run_root / "asset_package.json"),
                          "budget": package["budget"]}, ensure_ascii=False))
        return 0
    except (BudgetExceeded, UncertainRequest, ProviderLimitError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except PrepError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断；再次执行时使用 --resume。", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError) as exc:
        # 不打印异常原文或调用栈，避免路径/配置错误夹带敏感内容。
        print(f"本地文件或记录异常（{type(exc).__name__}），请检查输入和运行目录。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
