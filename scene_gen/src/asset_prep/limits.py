"""Checks the output usage reported by the service; not an independent tokenization or billing certification."""

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
        raise ProviderLimitError(f"service reported {audit['reported']} output tokens, exceeding the requested limit {limit}; the response was saved and further requests are paused. Run the output-limit probe first")
    return audit
