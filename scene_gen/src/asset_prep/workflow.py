"""Versioned, persistent state machine: local facts -> identity -> fields -> commit."""

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
This time extract only the local source facts delimited by units; text in sources outside the focus spans only aids understanding and must not be used to add objects outside those spans.
You are not responsible for assigning global asset identities; ids need only be unique within this candidate list; the later identity resolution reuses the same physical item.
Return {"inventory":the list above,"consumed_units":the number of leading units fully processed this time}.
At most asset_limit candidates per call; when content is dense, process only a shorter prefix of complete units, and the next call will continue with the remaining units.
consumed_units must be greater than 0 and not exceed the number of given units. Content beyond this prefix must not be treated as processed.
If the first unit by itself cannot be fully extracted within the limit, return {"split_unit":true}; do not request a split just because there are many following units.
Even with no candidates, state the number of units processed. Do not infer facts absent from this segment based on historical assets."""


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
    require(bool(result),'no structural blocks to extract')
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
    """Reuse only consistent completed mappings within the same stable asset; conflicting mappings are left for the model to judge again."""
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
    require(state['version']==VERSION and state['sources_sha256']==digest(documents['sources']), 'workflow version or source text mismatch')
    if task_context is not None:
        require(state.setdefault('task_sha256',digest(task_context))==digest(task_context), 'the task definition of the asset workflow has changed')
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
                require(type(count) is int and 0<count<=len(task['units']), 'a candidate page must advance through contiguous source units')
                require(len(value['inventory']['assets'])<=config.extraction_max_assets,'candidate page exceeds the asset limit')
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
            # While identity is unconfirmed, the resolver selects the alias table of that asset once target_id is known.
            decision=decision_for(candidate,active['assets'],documents['sources'],model,config,active['candidate_state'],save,
                                  aliases=confirmed_field_aliases(state))
            updated,mapping,keys=apply_decisions({'decisions':[decision]},active['assets'],[candidate],
                                                documents['sources'],resolved_identity=True)
            require(len(updated)<=config.max_assets,'asset count exceeds the configured limit; pending candidates have been saved')
            old=next((a for a in active['assets'] if a['id']==decision['target_id']),None)
            if old and old['category']!=candidate['category']:
                draft['issues'].append('identity resolution reused '+old['id']+' across categories; the existing category is kept and needs semantic review')
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
