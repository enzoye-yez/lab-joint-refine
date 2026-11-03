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


def gained(comparison, key):
    """Whether a compared round made progress on task key. A per-pair collision task
    (scene_collisions:a|b) gains when the row passed or a hit between a and b went away
    or got shallower."""
    resolved=set(comparison.get('resolved_checks',[]))
    rows=[r for field in ('improved_checks','partially_improved_checks') for r in comparison.get(field,[])]
    if key in resolved or any(r.get('check')==key for r in rows):return True
    check,_,pair=key.partition(':')
    if check!='scene_collisions' or not pair:return False
    if check in resolved:return True
    owners=set(pair.split('|'))
    def between(unit):
        names=unit.removeprefix('pair:').split('|')
        return len(names)==2 and {n.split('__',1)[0] for n in names}==owners
    return any(r.get('check')==check and any(between(u) for u in r.get('removed',[])+[r.get('reduced','')])
               for r in rows)


def touched(history_entry, jobs, task):
    """A round counts against a task only when it could have fixed it: it edited one of the
    task's jobs, or it changed the layout of a task that a layout edit can address."""
    action=history_entry.get('action') or {}
    if set(action.get('jobs',[]))&jobs:return True
    layout=any(action.get(k) for k in ('placements','layout','table_modifications','edge_mounts','end_effectors','robot_posture'))
    return layout and task.get('repair_level')!='asset_structure'


def schedule(run, tasks):
    history=run.state.get('repair_history',[])
    items={i['id']:i for i in run.state['plan']['instances']}
    for task in tasks:
        task['binding']=task_binding(run,task)
        key=task.get('check_id',task['id'])
        jobs={items[i]['job'] for i in task.get('instances',[]) if i in items}
        relevant=[h for h in history if h.get('status')=='evaluated'
            and h.get('task_bindings',{}).get(key)==task['binding'] and touched(h,jobs,task)]
        failures=[]
        for h in reversed(relevant):
            if h.get('selection')=='adopted' and gained(h.get('comparison') or {},key):break
            failures.append(h)
        attempts=sum(h.get('status')=='evaluated' and key in h.get('target_checks',h.get('task_bindings',{})) for h in history)
        task['scheduling']={'attempts':attempts,'no_progress':len(failures),
            'status':'deferred' if len(failures)>=3 else 'ready',
            'change_mechanism':len(failures)>=2,
            'failed_mechanisms':list(dict.fromkeys(h.get('mechanism') for h in failures if h.get('mechanism'))),
            'reason':'连续三次修改与验证未改善，保留问题并处理独立任务' if len(failures)>=3 else None,
            'reopen_when':'相关资产、依赖、检查实现或有效证据变化；恢复与目录变化不重置'}
    # Every eligible issue gets a turn before an equally bound issue repeats. Turns are
    # counted by assets the repair actually changed: target_checks lists every open
    # task each round, so counting it gave all tasks the same number and the order
    # fell back to report order, which kept the first planned assets in front.
    edits=job_edits(history)
    jobs_of=lambda t:{i['job'] for i in run.state['plan']['instances'] if i['id'] in t.get('instances',[])}
    for task in tasks:
        task['scheduling']['asset_edits']=min((edits[j] for j in jobs_of(task)),default=0)
        task['scheduling']['severity']=severity(task)
    tasks.sort(key=lambda t:(t['scheduling']['asset_edits'],t['scheduling']['attempts'],not t.get('operation_blocking',False),
                             -t['scheduling']['severity'],t['id']))
    run.state['repair_schedule']=tasks
    return [t for t in tasks if t['scheduling']['status']=='ready']


def job_edits(history):
    """How many applied repairs changed each asset job."""
    from collections import Counter
    return Counter(j for h in history if h.get('status') in {'applied','evaluated'}
                   for j in (h.get('action') or {}).get('jobs',[]))


def severity(task):
    """Confirmed named defects in the task's evidence; review-reported defects rank by finding count."""
    targets=task.get('targets',{})
    coverage=task.get('target_coverage',{})
    count=sum(coverage.get(k,{}).get('recorded',len(v)) for k,v in targets.items())
    return count or len(task.get('findings',[]))


def job_order(run, tasks, items, affected):
    """Jobs in task priority; inside a multi-instance task, least-edited assets first."""
    edits=job_edits(run.state.get('repair_history',[]))
    order=[]
    for task in tasks:
        jobs={items[i]['job'] for i in task['instances'] if i in affected}
        order.extend(sorted(jobs-set(order),key=lambda j:(edits[j],j)))
    return order


def action_tasks(tasks, action, plan, updated):
    changed=set(action['jobs'])
    old={i['id']:i for i in plan['instances']}
    moved={i['id'] for i in updated['instances'] if i!=old.get(i['id'])}
    layout_changed=any(plan.get(k)!=updated.get(k) for k in
        ('layout','table_modifications','edge_mounts','end_effectors','robot_posture'))
    return [t for t in tasks if any(old.get(i,{}).get('job') in changed or i in moved for i in t['instances'])
        or layout_changed and t.get('repair_level')!='asset_structure']
