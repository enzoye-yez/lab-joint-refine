"""任务资料准备入口；预检不加载凭据或创建网络客户端。"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .blocks import plan
from .config import Config, credentials
from .documents import load_documents
from .errors import PrepError
from .storage import atomic_json, encoded, run_directory, run_lock
from .task_contracts import BOUNDARY, public_environment


def inspect(documents, config, environment):
    public = public_environment(environment)
    chunks = plan(documents['sources'], config.chunk_chars)
    from .blocks import excerpts
    from .providers import model_messages
    from .prepare_prompts import INDEX
    from .workflow import plan_tasks
    extraction_blocks = len(plan_tasks(documents,config))
    # 尚无任务时无法估算实例、交互和检索的实际数量；只计算确定会遍历的阶段。
    requests = [len(encoded(model_messages(config,INDEX,{'operation':'protocol_index',
        'sources':excerpts(c['units'],documents,context=0)}))) for c in chunks]
    base_calls = 4*len(chunks)+extraction_blocks
    warnings = list(documents['warnings'])
    if base_calls > config.max_model_calls:
        warnings.append('当前调用上限小于全文遍历的基础需求；完成实例分配、身份匹配和审查还需额外调用。')
    if max(requests,default=0)>min(config.max_input_bytes,96000):
        warnings.append('至少一个索引请求超过输入上限，需要减小分段配置。')
    return {'status':'input_checked', 'workflow':'asset_workflow_v3', 'network_calls':0,
            'provider':config.provider, 'model':config.model, 'base_url':config.base_url,
            'scope':BOUNDARY, 'environment_sha256':public['sha256'],
            'robot':public['profile']['robot'].get('name'), 'source_blocks':len(chunks),
            'source_characters':sum(len(s['text']) for s in documents['sources']),
            'initial_index_calls':len(chunks), 'extraction_blocks':extraction_blocks,
            'baseline_document_calls':base_calls, 'largest_index_request_bytes':max(requests,default=0),
            'initial_index_token_reserve':sum(n+512+config.max_output_tokens for n in requests),
            'warnings':warnings,
            'budget':{k:getattr(config,k) for k in ('max_model_calls','max_reserved_tokens','max_search_calls')},
            'estimate_basis':'基础调用=4×索引块数+初始提取块数；不含细分、身份及字段匹配、实例分配、交互、终检、检索和修订。词元预留为字节上界估计，不是费用。'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='选择规程执行流程并准备移动双臂任务资料')
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--xml', type=Path)
    parser.add_argument('--xml-reason')
    parser.add_argument('--xml-query', action='append', default=[])
    parser.add_argument('--public-environment', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--requirements-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-uncertain', action='store_true')
    for name in ('max-model-calls','max-reserved-tokens','max-search-calls'):
        parser.add_argument('--'+name,type=int)
    args = parser.parse_args(argv)
    if args.preflight and (args.resume or args.retry_uncertain): parser.error('预检使用新运行，不恢复网络请求')
    if (args.xml_reason or args.xml_query) and not (args.xml and args.xml_reason and args.xml_query):
        parser.error('XML 定向补证须同时提供文件、原因及检索词')
    project = Path(__file__).resolve().parents[2]
    try:
        config = Config.load(args.config or project/'config/default.json')
        config = replace(config, **{n:getattr(args,n) for n in ('max_model_calls','max_reserved_tokens','max_search_calls') if getattr(args,n) is not None})
        path = args.public_environment or project/'config/scene_environment.json'
        environment = json.loads(path.read_text(encoding='utf-8-sig')); public_environment(environment)
        documents = load_documents(args.pdf.resolve(), args.xml.resolve() if args.xml else None,
            config.max_document_chars, xml_reason=args.xml_reason, xml_queries=args.xml_query)
        root = run_directory(project, args.run)
        with run_lock(root):
            if args.preflight:
                if any(p.name!='.lock' for p in root.iterdir()): raise PrepError('预检目录已存在，请使用新名称')
                result = inspect(documents, config, environment)
                atomic_json(root/'preflight.json', result)
            else:
                from .prepare_pipeline import execute_preparation
                from asset_generation.provider_diagnostics import DiagnosticTransport
                result = execute_preparation(documents, config, credentials(project), root, DiagnosticTransport(),
                    environment, resume=args.resume, retry_uncertain=args.retry_uncertain,
                    requirements_only=args.requirements_only)
        print(json.dumps(result,ensure_ascii=False)); return 0
    except PrepError as error:
        print(str(error)); return 2
    except KeyboardInterrupt:
        print('任务准备已中断，进度与请求预算保留。'); return 130
    except (OSError, ValueError, KeyError) as error:
        print('任务资料读取或保存失败：'+type(error).__name__); return 2


if __name__ == '__main__':
    raise SystemExit(main())
