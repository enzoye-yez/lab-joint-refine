"""由未解决的证据义务和故障义务调度恢复；状态是执行事实的派生结果。"""

from copy import deepcopy

from asset_generation.contracts import require, read_json
from asset_generation.pipeline import snapshot
from asset_prep.errors import StaleEvidence
from asset_prep.storage import digest, atomic_json, safe_path
from .capabilities import Context, REGISTRY, merge, semantic, validate_result, validate_request
from .recovery import evidence_key, gaps, located_repairs
from .faults import initialize, add_failure, resolve, eligibility, exception_failure
from .progress import evidence_progress


def consume(report, result, request, capability, key, fingerprint, mode=None):
    updated = merge(report,result,request,fingerprint,mode)
    return resolve(updated,result,request,capability,key,fingerprint)


def recover(run, report, artifact, destination):
    source = snapshot(artifact)
    fingerprint = digest({'scene':evidence_key(run.state['plan'],source),
                          'limits':run.limits.identity(),'contract':run.contract,
                          'implementation':getattr(getattr(run,'implementation_guard',None),'expected',None)})
    ctx = Context(artifact,run.sandbox,run.limits,run.contract,run.state['plan'],fingerprint)
    environment = (run.sandbox.recovery_environment() if hasattr(run.sandbox,'recovery_environment')
                   else {'backend':type(run.sandbox).__name__})
    environment_key = digest(environment)
    history = run.state.setdefault('check_recovery',{})
    run.state['active_check_recovery'] = fingerprint
    from asset_prep.state_archive import ArchivedRecord
    if isinstance(history.get(fingerprint), ArchivedRecord):
        history[fingerprint] = dict(history[fingerprint])
    session = history.setdefault(fingerprint, {'key':fingerprint,'actions':[], 'status':'started','attempts':0})
    actions = session.setdefault('actions',[])
    base_key = digest({k:v for k,v in report.items() if k!='renders'})
    if session.get('base_key') == base_key and session.get('report') is not None:
        report = {**deepcopy(session['report']),'renders':report.get('renders',[])}
    else:
        session['base_key'] = base_key
    report = initialize(report,fingerprint)
    def publish(event):
        if hasattr(run, 'check_progress'):
            run.check_progress(report, artifact, event)
    publish('evidence_recovery_started')
    while True:
        run.check_pause()
        # 同版本的独立已完成证据也可以解除后来发现的故障，不需要重新执行。
        for action in actions:
            cap = REGISTRY.get(action['capability'])
            if cap and action['status']=='completed' and action.get('result'):
                trial = resolve(deepcopy(report),action['result'],action['parameters'],cap,action['key'],fingerprint)
                if trial.get('recovery_failures')!=report.get('recovery_failures'):
                    report = consume(report,action['result'],action['parameters'],cap,action['key'],fingerprint,run.limits.validation_mode)
        # Evidence completeness is an acceptance condition, not a prerequisite
        # for repairing an independently confirmed and localized defect.
        ready = located_repairs(report, run.state['plan'])
        if ready:
            session.update(status='repair_ready', report=deepcopy(report),
                handoff={'checks':[row['id'] for row in ready],
                         'remaining_checks':[row['id'] for row in gaps(report)]})
            run.store.save()
            publish('confirmed_defects_ready_for_repair')
            return report
        candidates, audit = [], []
        retry_exhausted = False
        new_query_fault = False
        query_failures = session.setdefault('query_failures',{})
        for cap in REGISTRY.values():
            if cap.id in query_failures:
                audit.append({'capability':cap.id,'proposals':0,'excluded':[{'reason':'proposal_execution_failure',
                              'failure':query_failures[cap.id]}]})
                continue
            try:
                proposed = list(cap.propose(ctx,deepcopy(report)))
                for request in proposed:
                    validate_request(request)
                    require(request.get('full_scene') or set(request['checks']) <= {r['id'] for r in report.get('checks',[])},
                            '补查请求包含不存在的检查')
            except Exception as exc:
                failure = exception_failure(exc,cap.id)
                query_failures[cap.id] = failure
                report = add_failure(report,failure,input_key=fingerprint,action_key='proposal:'+cap.id,attempt=1,
                    capability=cap.id,request={'checks':[r['id'] for r in report.get('checks',[])]},scope='scene')
                new_query_fault = True
                session['report'] = deepcopy(report)
                run.store.save()
                continue
            inspected = {'capability':cap.id,'proposals':len(proposed),'excluded':[]}
            audit.append(inspected)
            for request in proposed:
                validate_request(request)
                key = digest({'input':fingerprint,'capability':cap.id,'parameters':semantic(request)})
                previous = next((a for a in actions if a['key']==key),None)
                has_receipt = bool(previous and previous.get('receipt') and safe_path(run.root,previous['receipt']).is_file())
                allowed,reason,exhausted = eligibility(previous,report.get('recovery_failures',[]),environment_key,
                                                       run.limits.max_check_retries,has_receipt)
                if not allowed:
                    inspected['excluded'].append({'key':key,'reason':reason})
                    retry_exhausted |= exhausted
                    continue
                # 未执行的备用能力优先于失败方法的重试；不会被反复失败的方法饿死。
                candidates.append((1 if previous and previous['status']=='failed' else 0,
                                   cap,request,key,previous,reason))
        if not candidates:
            if new_query_fault:
                continue  # 给其他能力一次消费新故障范围并重新提案的机会。
            session.update(status='budget_exhausted' if retry_exhausted else 'completed', report=deepcopy(report),
                exhaustion={'reason':'recovery_retry_budget' if retry_exhausted else 'no_executable_action',
                            'capabilities':audit,'action_keys':[a['key'] for a in actions],
                            'open_failures':[f['id'] for f in report.get('recovery_failures',[]) if f['status']=='open'],
                            'remaining_checks':[r['id'] for r in gaps(report)]})
            run.store.save()
            return report
        _,cap,request,key,action,reason = min(candidates,key=lambda c:c[0])
        saved_receipt = safe_path(run.root,action['receipt']) if action and action.get('receipt') else None
        replay = bool(action and action['status']=='started' and saved_receipt and saved_receipt.is_file())
        if sum(a['attempts'] for a in actions) >= run.limits.max_check_actions and not replay:
            session.update(status='budget_exhausted',report=deepcopy(report),
                           exhaustion={'reason':'check_action_budget','remaining_actions':len(candidates),'capabilities':audit})
            run.store.save()
            return report
        run.check_pause()
        if action is None:
            action = {'key':key,'capability':cap.id,'parameters':request,'attempts':0,'executions':[]}
            actions.append(action)
        if not replay:
            action['attempts'] += 1
            output = destination/'main_check_recovery'/f'{key}_{action["attempts"]}'
            action.update(directory=str(output.relative_to(run.root)),
                          receipt=str((output/'receipt.json').relative_to(run.root)),environment=environment_key)
            action.setdefault('executions',[]).append({'attempt':action['attempts'],'status':'started',
                'environment':environment_key,'environment_observation':deepcopy(environment),
                'eligibility':reason,'directory':action['directory'],'receipt':action['receipt']})
        else:
            output = safe_path(run.root,action['directory'])
        action.update(status='started',base_report_key=digest(semantic(report)))
        action.pop('failure',None)
        action.pop('result',None)
        # 进度基准跟随动作本身：中断恢复重放时也不能错用别的动作的前态。
        if not replay or 'before_report' not in action:
            action['before_report'] = deepcopy(report)
        session['attempts'] = max(session.get('attempts',0),action['attempts'])
        run.store.save()
        publish('check_started:'+cap.id)
        receipt = safe_path(run.root,action['receipt'])
        if replay:
            result = read_json(receipt,8000000)
        else:
            try:
                result = cap.execute(ctx,deepcopy(request),output)
            except StaleEvidence:
                raise
            except Exception as exc:
                result = {'input_key':fingerprint,'scope':'checks','checks':[],
                          'failure':exception_failure(exc,cap.id)}
            finally:
                if snapshot(artifact)!=source:
                    raise StaleEvidence('只读恢复修改了原场景，拒绝消费')
            # 保存故障收据与成功收据使用同一协议。磁盘写入故障上抛，不能误算工具重试。
            output.mkdir(parents=True,exist_ok=True)
            atomic_json(receipt,result)
        if snapshot(artifact)!=source:
            raise StaleEvidence('恢复输入已变化')
        try:
            validate_result(result,request,fingerprint,run.limits.validation_mode)
            failure = result.get('failure')
            if failure is None:
                updated = consume(report,result,request,cap,key,fingerprint,run.limits.validation_mode)
                action.update(status='completed',result=result)
                report = updated
        except Exception as exc:
            failure = exception_failure(exc,'scene_generation.capabilities')
        if failure is not None:
            action.update(status='failed',failure=deepcopy(failure))
            report = add_failure(report,failure,input_key=fingerprint,action_key=key,attempt=action['attempts'],
                capability=cap.id,request=request,environment=environment_key,
                scope='scene' if request.get('full_scene') else 'pairs' if 'pairs' in request else getattr(cap,'failure_scope','checks'))
        execution = action['executions'][-1]
        execution.update(status=action['status'])
        # 动作状态与证据进展分开记录：摘要变化不等于问题解决。
        progress = evidence_progress(action.get('before_report', report), report, cap, request,
                                     status=action['status'], failure=deepcopy(failure), result=result)
        action['effective'] = progress['evidence_changed']
        execution.update({'failure':deepcopy(failure)} if failure is not None else {'effective':action['effective']})
        execution['progress'] = progress
        action.setdefault('progress', []).append(progress)
        session.setdefault('progress', []).append(progress)
        session['report'] = deepcopy(report)
        run.store.save()
        publish('check_completed:'+cap.id)
