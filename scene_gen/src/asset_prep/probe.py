"""用户显式执行的小规模参数探测；不读规程，不自动选择模型参数。"""

import sys

from .errors import ProviderError, UncertainRequest, ValidationError
from .limits import output_audit
from .providers import redact
from .storage import Store, atomic_json, digest, encoded, safe_path


def probe_limits(config, credentials, root, transport, *, resume=False, retry_uncertain=False, parameter=None):
    allowed = ('max_tokens', 'max_completion_tokens')
    if parameter is not None and parameter not in allowed:
        raise ValidationError('不支持的探测参数')
    parameters = (parameter,) if parameter else allowed
    key = config.model_key(credentials)
    if not key:
        raise ProviderError('缺少模型密钥')
    identity = {'mode':'output_limit_probe_v1','base_url':config.base_url,'model':config.model}
    if parameter:
        identity['parameters'] = list(parameters)
    store = Store(root, identity, config,
                  resume=resume, retry_uncertain=retry_uncertain)
    report = {'mode':'output_limit_probe', 'parameters':list(parameters), 'results':[], 'status':'in_progress',
              'limitations':'只对本次小样本及服务自报用量进行比较；缺失用量不可验证，不证明长请求或真实计费上限。'}
    def save():
        report['budget'] = store.totals()
        atomic_json(safe_path(root, 'output_limit_probe.json'), report)
    try:
        for parameter in parameters:
            for limit in (64, 256):
                payload = {'model':config.model, 'stream':False, parameter:limit,
                           'messages':[{'role':'user','content':'Output the word lab exactly 200 times, separated by spaces. No explanation.'}]}
                request_key = digest({'probe':payload})
                response = store.cached(request_key)
                if response is None:
                    report['current_request'] = {'parameter':parameter,'requested':limit}
                    save()
                    call = store.reserve(request_key, 'model', len(encoded(payload)) + 256 + limit)
                    print(f'接口探测：参数 {parameter}，请求输出上限 {limit}，等待上限 {config.timeout_seconds} 秒', file=sys.stderr, flush=True)
                    try:
                        response = redact(transport.post(config.base_url.rstrip('/')+'/chat/completions', payload,
                                                          key, config.timeout_seconds), credentials.values())
                        store.finish(call, response)
                    except ProviderError as exc:
                        store.record_error(call, exc)
                        if exc.uncertain:
                            raise UncertainRequest('探测请求结果不明；已保留预算，核对诊断后再决定重试') from exc
                        store.fail(call, 'ProviderError')
                        report['results'].append({'parameter':parameter,'requested':limit,'status':'request_failed',
                                                  'diagnostics':exc.diagnostics})
                        save()
                        if exc.diagnostics.get('http_status') in {401, 403, 429}:
                            report['status'] = 'request_failed'
                            save()
                            return report
                        break
                store.check_token_budget()
                audit = output_audit(response, limit, parameter)
                choices = response.get('choices')
                choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
                message = choice.get('message')
                content = message.get('content') if isinstance(message, dict) else None
                audit.update(finish_reason=choice.get('finish_reason'), returned_model=response.get('model'),
                             response_characters=len(content) if isinstance(content, str) else None)
                audit['content_valid'] = isinstance(content, str)
                if not isinstance(content, str) and audit['status'] != 'reported_violation':
                    audit['status'] = 'invalid_response'
                report['results'].append(audit)
                save()
                # 已发现超限时不继续浪费请求；另一种参数可在人工检查后单独确认。
                if audit['status'] == 'reported_violation':
                    report['status'] = 'violation_detected'
                    save()
                    return report
        report['status'] = 'finished_requires_review'
        save()
        return report
    except Exception:
        report['status'] = 'paused_or_failed'
        save()
        raise
