"""版本化、可持久化的局部事实—身份—字段—提交状态机。"""

import copy
from . import blocks
from .contracts import fields, require, validate_inventory
from .errors import OutputTooLarge
from .extraction import planning_limit, request_size, validate_unique_assets
from .incremental import apply_decisions, finalize_catalog
from .prompts import INVENTORY
from .resolution import decision_for
from .storage import digest


VERSION='asset_workflow_v2'
LOCAL = INVENTORY + """
本次只提取 units 标定的局部原文事实，sources 中 focus 区间外的文字只帮助理解，不能据此新增区间外对象。
不负责分配全局资产身份，id只在本次候选清单内唯一；后续身份解析负责复用同一器具。
返回 {"inventory":上述清单,"consumed_units":本次完整处理的前缀单元数}。
单次最多 asset_limit 项候选；内容较密时，只处理较短的完整单元前缀，下一次会继续处理剩余单元。
consumed_units 必须大于0且不超过给定单元数。不能把超过本次前缀的内容当作已处理。
若首个单元本身无法在上限内完整提取，返回 {"split_unit":true}；不要因为后续单元过多而要求拆分。
候选为空也应明确处理的单元数。不根据历史资产推断本段未出现的事实。"""


def empty_catalog():
    return {k:[] for k in ('assets','steps','relations','conflicts','assumptions','issues')}


def new_workflow(documents,config):
    return {'version':VERSION,'sources_sha256':digest(documents['sources']),
            'catalog':empty_catalog(),'queue':plan_tasks(documents,config),
            'active':None,'commits':[],'revision':0,'status':'running','operation_audit':[]}


def local_data(task,documents,config,task_context=None):
    data = {'operation':'local_extract','task_id':task['id'],'units':[
        {'source_id':p['id'],'offset':p.get('offset',0),'length':len(p['text'])} for p in task['units']],
        'sources':blocks.excerpts(task['units'],documents), 'asset_limit':config.extraction_max_assets}
    if task_context is not None:
        data['selected_task'] = task_context
        source_ids = {s['id'] for s in data['sources']}
        data['related_stage_ids'] = [s['id'] for s in task_context.get('stages',[])
                                     if source_ids.intersection(s.get('source_ids',[]))]
        data['scope_rule'] = 'Only selected execution stages and their physical dependencies. No excluded branches or equipment used only for completed preparation. Shared geometry types, not one type per role. English descriptions; verbatim source quotes.'
    return data


def plan_tasks(documents,config,*,sources=None):
    pending=blocks.plan(documents['sources'] if sources is None else sources,config.chunk_chars); result=[]
    while pending:
        block=pending.pop(0)
        if request_size(config,LOCAL,local_data(block,documents,config))>min(24000,planning_limit(config)):
            pending[0:0]=blocks.split_task(block)
        else: result.append(block)
    require(bool(result),'没有可提取的结构块')
    return result


def progress_of(state):
    return {'revision':state['revision'],'applied_segments':len(state['commits']),
            'completed_segments':len(state['commits']), 'total_segments':len(state['commits'])+len(state['queue'])+(1 if state['active'] else 0),
            'phase':state['active']['stage'] if state['active'] else state['status'],
            'updates':copy.deepcopy(state['commits']), 'workflow_version':VERSION}


def seed_active(task,draft,catalog):
    return {'task':task,'stage':'identity','draft':copy.deepcopy(draft),
            'assets':copy.deepcopy(catalog['assets']),'cursor':0,'mapping':{},'keys':[],
            'decisions':[],'candidate_state':{},'audit':[]}


def confirmed_field_aliases(state):
    """只复用同一稳定资产中一致的已完成映射；冲突映射留给模型重新判断。"""
    observed = {}
    decisions = [decision for commit in state['commits'] for decision in commit['decisions']]
    decisions.extend((state.get('active') or {}).get('decisions', []))
    for decision in decisions:
        asset = observed.setdefault(decision['target_id'], {})
        for field in decision['requirements']:
            asset.setdefault(field['incoming_key'], set()).add(field['target_key'])
    return {
        asset_id: {alias: next(iter(targets)) for alias, targets in fields.items() if len(targets) == 1}
        for asset_id, fields in observed.items()
    }


def run_workflow(documents,config,model,state=None,persist=None,publish=None,*,task_context=None):
    state=copy.deepcopy(state) if state is not None else new_workflow(documents,config)
    require(state['version']==VERSION and state['sources_sha256']==digest(documents['sources']), '工作流版本或原文不一致')
    if task_context is not None:
        require(state.setdefault('task_sha256',digest(task_context))==digest(task_context), '资产工作流的任务定义已变化')
    def save():
        if persist: persist(copy.deepcopy(state))
    def emit():
        if publish: publish(state['catalog'],progress_of(state))
    save(); emit()
    while state['active'] or state['queue']:
        if state['active'] is None:
            state['active']={'task':state['queue'].pop(0),'stage':'extract'}; save()
        active=state['active']
        if active['stage']=='extract':
            task=active['task']; data=local_data(task,documents,config,task_context)
            if request_size(config,LOCAL,data)>min(64000 if task_context else 24000,planning_limit(config)):
                children=blocks.split_task(task)
                state['queue'][0:0]=children; state['active']=None; save(); continue
            def validate(value):
                fields(value,'inventory consumed_units')
                count=value['consumed_units']
                require(type(count) is int and 0<count<=len(task['units']), '候选分页必须推进连续原文单元')
                require(len(value['inventory']['assets'])<=config.extraction_max_assets,'候选页超过资产上限')
                validate_inventory(value['inventory'],blocks.excerpts(task['units'][:count],documents),
                                   config.max_assets,allow_empty=True)
                return value
            try:
                response=model.structured(LOCAL,data,validate)
            except OutputTooLarge as error:
                children=blocks.split_task(task)
                state['operation_audit'].append({'task_id':task['id'],'action':'split','reason':str(error)})
                state['queue'][0:0]=children; state['active']=None; save(); continue
            count=response['consumed_units']
            if count<len(task['units']): state['queue'].insert(0,blocks.task(task['units'][count:]))
            state['active']=seed_active(blocks.task(task['units'][:count]),response['inventory'],state['catalog'])
            active=state['active']; save(); emit()
        draft=active['draft']
        while active['cursor']<len(draft['assets']):
            candidate=draft['assets'][active['cursor']]
            # 身份尚未确认时由解析器在得到 target_id 后选择该资产的别名表。
            decision=decision_for(candidate,active['assets'],documents['sources'],model,config,active['candidate_state'],save,
                                  aliases=confirmed_field_aliases(state))
            updated,mapping,keys=apply_decisions({'decisions':[decision]},active['assets'],[candidate],
                                                documents['sources'],resolved_identity=True)
            require(len(updated)<=config.max_assets,'资产数量超过配置上限；已保存待提交候选')
            old=next((a for a in active['assets'] if a['id']==decision['target_id']),None)
            if old and old['category']!=candidate['category']:
                draft['issues'].append('身份解析跨类别复用 '+old['id']+'；保留已有类别，需语义复核')
            active['assets']=updated; active['mapping'].update(mapping)
            active['keys'].extend([[aid,key,value] for (aid,key),value in keys.items()])
            active['decisions'].append(decision); active['audit'].append(copy.deepcopy(active['candidate_state']))
            active['cursor']+=1; active['candidate_state']={}; save()
        active['stage']='commit'; save()
        keys={(aid,key):value for aid,key,value in active['keys']}
        previous=state['catalog']
        updated=finalize_catalog(previous,draft,active['assets'],active['mapping'],keys,documents['sources'],config,state['revision'])
        validate_unique_assets(updated)
        state['revision']+=1
        before={a['id']:a for a in previous['assets']}; after={a['id']:a for a in updated['assets']}
        state['commits'].append({'revision':state['revision'],'task_id':active['task']['id'],
            'source_spans':[{'source_id':p['id'],'start':p.get('offset',0),'end':p.get('offset',0)+len(p['text'])} for p in active['task']['units']],
            'decisions':active['decisions'],'resolution_audit':active['audit'],
            'added_ids':sorted(after.keys()-before.keys()),
            'updated_ids':sorted(k for k in before.keys()&after.keys() if before[k]!=after[k])})
        state['catalog']=updated; state['active']=None; save(); emit()
    validate_inventory(state['catalog'],documents['sources'],config.max_assets)
    state['status']='extraction_completed'; save(); emit()
    return state['catalog']
