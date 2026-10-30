"""可信执行故障的范围、证据解除与有限重试；不接受生成模型修改这些约定。"""

import errno
import subprocess
from copy import deepcopy

from asset_prep.storage import digest
from .recovery import report_status
from asset_generation.check_report import complete


TEMPORARY_ERRNOS = {errno.EAGAIN, errno.EBUSY, errno.EINTR, errno.ETIMEDOUT}
TEMPORARY_WINERRORS = {32, 33, 121}


def exception_failure(exc, fallback='*'):
    """仅根据异常类型/系统错误码分类，不匹配自由文本中的“超时”等词。"""
    chain, current = [], exc
    while current is not None and id(current) not in {id(e) for e in chain}:
        chain.append(current)
        current = current.__cause__
    temporary = any(isinstance(e,(TimeoutError,subprocess.TimeoutExpired)) or
                    isinstance(e,OSError) and (e.errno in TEMPORARY_ERRNOS or getattr(e,'winerror',None) in TEMPORARY_WINERRORS)
                    or getattr(e,'retryable',False) is True for e in chain)
    origin = chain[-1]
    environment = isinstance(origin,OSError) or type(origin).__name__ in {'BackendExecutionError','SandboxUnavailable'}
    kind = 'temporary' if temporary else 'environment' if environment else 'program'
    components = [fallback]
    if getattr(origin,'component',None):
        components = [origin.component]
    elif kind == 'program':
        # 找到实际出错模块；共享模块出错时，其他同样依赖它的算法不能洗掉故障。
        tb = origin.__traceback__
        frames = []
        while tb:
            frames.append(tb)
            tb = tb.tb_next
        # require 是公共抛错助手，故障归属应是实际调用它的边界。
        tb = next((frame for frame in reversed(frames)
                   if not (frame.tb_frame.f_globals.get('__name__')=='asset_generation.contracts'
                           and frame.tb_frame.f_code.co_name=='require')),None)
        if tb:
            module = tb.tb_frame.f_globals.get('__name__','')
            method = tb.tb_frame.f_code.co_qualname
            if module.startswith(('scene_generation.','asset_generation.','asset_prep.')):
                components = [fallback if module=='scene_generation.capabilities' and method=='FullRecheck.execute' else module]
    return {'kind':kind,'retryable':temporary,'components':components,
            'error_kind':'check_execution' if kind=='program' else 'backend_execution',
            'error_type':type(exc).__name__,'detail':str(exc)}


def dependencies(capability):
    return set(getattr(capability,'dependencies',('*',))) | {capability.id,
        'scene_generation.capabilities','scene_generation.dispatch','scene_generation.faults',
        'asset_generation.check_report'}


def add_failure(report, failure, *, input_key, action_key, attempt, capability,
                request, scope='checks', environment=None):
    result = deepcopy(report)
    faults = result.setdefault('recovery_failures',[])
    identity = digest({'input':input_key,'action':action_key,'attempt':attempt})
    if not any(f['id']==identity for f in faults):
        kind = failure.get('kind','program')
        components = failure.get('components') or ['*']
        faults.append({'id':identity,'input_key':input_key,'action_key':action_key,'attempt':attempt,
            'capability':capability,'kind':kind,'retryable':kind=='temporary' and failure.get('retryable') is True,
            'components':components,'scope':scope,'affected_checks':list(request['checks']),
            'remaining_checks':list(request['checks']),
            'remaining_pairs':deepcopy(request.get('pairs',[])) if scope=='pairs' else [],
            'status':'open','resolutions':[],'environment':environment,'diagnostic':deepcopy(failure)})
    result['fault_status_managed'] = True
    return refresh(result)


def initialize(report, input_key):
    """无范围的历史/主检查异常按整场景保守记录，局部成功不能解除它。"""
    if report.get('status')!='error' or report.get('error_kind')=='measurement_error' or report.get('fault_status_managed'):
        return deepcopy(report)
    failure = report.get('execution_failure',{'kind':'program','retryable':False,'components':['*'],
                                             'detail':report.get('detail','未提供故障来源')})
    return add_failure(report,failure,input_key=input_key,action_key='main_check',attempt=1,
                       capability='main_check',request={'checks':[r['id'] for r in report.get('checks',[])]},scope='scene')


def refresh(report):
    if not report.get('fault_status_managed'):
        return report
    pending = [f for f in report.get('recovery_failures',[]) if f['status']=='open']
    if pending:
        report['status'] = 'error'
        report['error_kind'] = 'check_execution' if any(f['kind']=='program' for f in pending) else 'backend_execution'
    else:
        report['status'] = report_status(report.get('checks',[]))
        report.pop('error_kind',None)
        report.pop('recovery_failure',None)
        report.pop('execution_failure',None)
    return report


def independent(fault, capability):
    if fault['kind']!='program':
        return True
    used = dependencies(capability)
    return '*' not in fault['components'] and '*' not in used and not set(fault['components']).intersection(used)


def resolve(report, result, request, capability, action_key, input_key):
    """只消费已通过合并校验、同版本且覆盖完整的成功检查；历史故障不删除。"""
    if result.get('input_key')!=input_key or result.get('failure'):
        return report
    scope = result.get('scope')
    covered_checks = {r['id'] for r in result.get('checks',[]) if complete(r)}
    excluded = {r['id'] for r in result.get('checks',[]) if r.get('status')=='skipped'}
    # 部分碰撞失败证明缺陷存在，却不能证明整个检查范围已覆盖。
    for fault in report.get('recovery_failures',[]):
        if fault['status']!='open' or fault['input_key']!=input_key or not independent(fault,capability):
            continue
        resolved = []
        if fault['scope']=='scene':
            if (scope=='scene' and result.get('complete_scene') is True and covered_checks
                    and len(covered_checks | excluded)==len(result['checks'])
                    and set(fault['remaining_checks']) <= covered_checks | excluded):
                fault['remaining_checks'] = []
                resolved = ['scene']
        elif fault['scope']=='localization':
            if scope=='localization':
                resolved = [r['id'] for r in result.get('checks',[]) if r.get('instances') and r['id'] in fault['remaining_checks']]
        elif fault['scope']=='pairs':
            if scope in {'checks','scene'} and set(fault['remaining_checks']) <= covered_checks:
                fault['remaining_pairs'] = []
                resolved = list(fault['remaining_checks'])
            elif scope=='pairs' and set(fault['remaining_checks']) <= covered_checks:
                covered = {tuple(sorted(p)) for p in result.get('covered_pairs',[])}
                fault['remaining_pairs'] = [p for p in fault['remaining_pairs'] if tuple(sorted(p)) not in covered]
                if not fault['remaining_pairs']:
                    resolved = list(fault['remaining_checks'])
        elif scope in {'checks','scene'}:
            resolved = sorted(covered_checks.intersection(fault['remaining_checks']))
        if resolved:
            fault['remaining_checks'] = [c for c in fault['remaining_checks'] if c not in resolved]
            fault['resolutions'].append({'action_key':action_key,'capability':capability.id,'checks':resolved,
                                         'input_key':input_key,'scope':scope})
            if not fault['remaining_checks'] and not fault['remaining_pairs']:
                fault['status'] = 'resolved'
    return refresh(report)


def eligibility(previous, faults, environment, retry_limit, has_receipt):
    """返回可执行依据及是否确实耗尽了重试预算；更换环境不清零总执行预算。"""
    if previous is None:
        return True,'untried',False
    if previous['status']=='completed':
        return False,'already_completed',False
    if previous['status']=='started':
        interrupted = sum(e['status'] in {'started','interrupted'} for e in previous.get('executions',[]))
        allowed = has_receipt or (interrupted or previous['attempts'])<2
        return allowed,'receipt_replay' if has_receipt else 'interrupted_retry' if allowed else 'interruption_retry_budget',not allowed
    pending = [f for f in faults if f['action_key']==previous['key'] and f['status']=='open']
    if not pending:
        return False,'covered_by_alternative',False
    failure = previous.get('failure',{})
    if failure.get('kind')=='program' or not failure.get('kind'):
        return False,'implementation_fix_required',False
    if previous.get('environment')!=environment:
        return True,'observed_environment_change',False
    if failure.get('retryable') is not True:
        return False,'environment_change_required',False
    used = sum(e.get('environment')==environment for e in previous.get('executions',[])) or previous['attempts']
    allowed = used < 1+retry_limit
    return allowed,'temporary_retry' if allowed else 'temporary_retry_budget',not allowed
