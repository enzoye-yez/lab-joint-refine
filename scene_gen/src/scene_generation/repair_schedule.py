"""Scheduling is a derived view of repair history, never a quality verdict."""
from asset_prep.storage import digest, file_hash, safe_path


def task_binding(run, task):
    items={i['id']:i for i in run.state['plan']['instances']}
    jobs={j['id']:j for j in run.state['plan']['jobs']}
    selected={items[i]['job'] for i in task['instances'] if i in items}
    pending=list(selected)
    while pending:
        for dependency in jobs[pending.pop()]['depends_on']:
            if dependency not in selected:selected.add(dependency);pending.append(dependency)
    versions={}
    for key in sorted(selected):
        current=run.state['jobs'].get(key,{})
        versions[key]={'job':jobs[key], 'artifact':current.get('assembly_readiness',{}).get('artifact_sha256'),
            'source':file_hash(safe_path(run.root,current['source'])) if current.get('source') else None}
    layout=task.get('repair_level')!='asset_structure'
    return digest({'task':task['id'],'versions':versions,'implementation':run.implementation_guard.expected,
        'contract':run.contract,'environment':run.env,
        'evidence':task.get('targets',{}),
        'layout':run.state['plan'] if layout else None})


def schedule(run, tasks):
    history=run.state.get('repair_history',[])
    for task in tasks:
        task['binding']=task_binding(run,task)
        key=task.get('check_id',task['id'])
        relevant=[h for h in history if h.get('status')=='evaluated'
            and h.get('task_bindings',{}).get(key)==task['binding']]
        failures=[]
        for h in reversed(relevant):
            comparison=h.get('comparison') or {}
            gained=set(comparison.get('resolved_checks',[])) | {
                r['check'] for field in ('improved_checks','partially_improved_checks') for r in comparison.get(field,[])}
            if h.get('selection')=='adopted' and key in gained:break
            failures.append(h)
        attempts=sum(h.get('status')=='evaluated' and key in h.get('target_checks',h.get('task_bindings',{})) for h in history)
        task['scheduling']={'attempts':attempts,'no_progress':len(failures),
            'status':'deferred' if len(failures)>=3 else 'ready',
            'change_mechanism':len(failures)>=2,
            'failed_mechanisms':list(dict.fromkeys(h.get('mechanism') for h in failures if h.get('mechanism'))),
            'reason':'连续三次修改与验证未改善，保留问题并处理独立任务' if len(failures)>=3 else None,
            'reopen_when':'相关资产、依赖、检查实现或有效证据变化；恢复与目录变化不重置'}
    # Every eligible issue gets a turn before an equally bound issue repeats.
    tasks.sort(key=lambda t:(t['scheduling']['attempts'],not t.get('operation_blocking',False),t['id']))
    run.state['repair_schedule']=tasks
    return [t for t in tasks if t['scheduling']['status']=='ready']


def action_tasks(tasks, action, plan, updated):
    changed=set(action['jobs'])
    old={i['id']:i for i in plan['instances']}
    moved={i['id'] for i in updated['instances'] if i!=old.get(i['id'])}
    layout_changed=any(plan.get(k)!=updated.get(k) for k in
        ('layout','table_modifications','edge_mounts','end_effectors','robot_posture'))
    return [t for t in tasks if any(old.get(i,{}).get('job') in changed or i in moved for i in t['instances'])
        or layout_changed and t.get('repair_level')!='asset_structure']
