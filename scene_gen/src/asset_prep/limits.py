"""核对服务报告的输出用量；不是独立分词或计费认证。"""

from .errors import ProviderLimitError


def output_audit(response, limit, parameter):
    usage = response.get('usage')
    reported = usage.get('completion_tokens', usage.get('output_tokens')) if isinstance(usage, dict) else None
    valid = type(reported) is int and reported >= 0
    details = usage.get('completion_tokens_details', usage.get('output_tokens_details')) if isinstance(usage, dict) else None
    reasoning = details.get('reasoning_tokens') if isinstance(details, dict) else None
    valid_reasoning = valid and type(reasoning) is int and 0 <= reasoning <= reported
    return {'kind': 'output_limit', 'parameter': parameter, 'requested': limit,
            'reported': reported if valid else None,
            'reported_reasoning_tokens': reasoning if valid_reasoning else None,
            'reported_non_reasoning_tokens': reported - reasoning if valid_reasoning else None,
            'status': ('reported_violation' if reported > limit else 'reported_within_limit') if valid else 'unverifiable'}


def check_output_limit(response, limit, parameter, store, key):
    audit = output_audit(response, limit, parameter)
    store.record_validation(key, audit)
    if audit['status'] == 'reported_violation':
        raise ProviderLimitError(f"服务报告输出 {audit['reported']} 个令牌，超过请求上限 {limit}；响应已保存，暂停后续请求。请先运行输出上限探测")
    return audit
