"""Check only the capabilities/objects used by a node; global faults stay global."""
from scene_generation.handoff import quality_handoff
from .common import need, OperationError


def references(value):
    if isinstance(value,dict):return set().union(*(references(v) for v in value.values())) if value else set()
    if isinstance(value,list):return set().union(*(references(v) for v in value)) if value else set()
    return {value} if isinstance(value,str) else set()


def check_node(session,node):
    names=references(node)|set(session.holding.values())
    interfaces=session.manifest.get('interfaces',[])
    used={i['id'] for i in interfaces if any(i.get(k) in names for k in ('id','body','site'))}
    names.update(i[k] for i in interfaces if i['id'] in used for k in ('body','site') if i.get(k))
    missing=[e for e in session.manifest.get('mechanical_events',[])
        if e['id'] in session.unbound_events and (e['interface'] in used or e['id'] in names)]
    need(not missing,'当前操作接口缺少机械事件检测器','capability_missing')
    instances={i['id'] for i in session.meta['plan']['instances']
        if any(n==i['id'] or n.startswith(i['id']+'__') for n in names)}
    issues=quality_handoff(getattr(session,'quality_report',{}),None)['issues']
    blocked=[r for r in issues if r['global_fault'] or r['execution_requirement'] and instances.intersection(r['instances'])]
    if blocked:raise OperationError('quality_prerequisite','当前操作依赖尚未通过的质量前置条件',{'issues':blocked})
