"""Identity and fields are paged separately; a single request is decoupled from the total number of registry entries."""

import copy
from .contracts import fields, require, text
from .extraction import request_size, planning_limit
from .incremental import choose


MATCH = """Compare the current candidate with the objects on this page and pick the one most likely to represent the same physical asset / the same specification type.
previous is the tentative object from earlier pages and may be kept or replaced; null means not matched yet, and the program will not add a new object before all pages are done.
Category is only a hint; a different category alone does not rule out the same object. Different specifications may mean a different object, or a later supplement/correction; judge from name, aliases and specifications.
Return only {"target": "id of a given object or null", "reason":"reason for the judgment"}. Never invent ids; covering every page does not mean the semantic judgment is correct."""
FIELD_MATCH = """For the current specification, choose the semantically identical field among the existing specifications on this page. previous is the tentative field from earlier pages.
Synonymous names may map to the same key; conflicting values do not necessarily mean different fields; unit case and decimal points are significant.
Return only {"target":"id of a given field or null","reason":"reason for the judgment"}. Return null when nothing matches; never invent field ids."""
SELECT = """These two fields are the same property of the same asset. Decide whether the final value takes existing or incoming.
Prefer an explicit field with PDF source text, then other explicit fields, then inferred ones; at equal priority choose based on the evidence; never add up quantities or rewrite values.
Return only {"take":"existing or incoming","reason":"reason for the choice"}. The original evidence of both sides is kept locally."""


def identity_card(asset):
    return {k:copy.deepcopy(asset[k]) for k in ('id','name','category','aliases')} | {
        'requirements':[{k:r[k] for k in ('key','value','unit')} for r in asset['requirements']]}


def field_card(value):
    return {'id':value['key'],'value':value['value'],'unit':value['unit']}


def bounded_scan(candidate, cards, state, model, config, save, *, operation='identity'):
    """Fold tentative matches page by page; the cursor and decisions are persisted per page, and nothing new is added before the scan finishes."""
    instruction = MATCH if operation=='identity' else FIELD_MATCH
    limit=min(24000,planning_limit(config))
    # Identity cards omit full evidence, so more objects fit within the same byte limit, reducing repeated scan calls.
    # Field pages keep a smaller granularity. The cursor is an object offset, so existing checkpoints with 12-item pages need no recomputation.
    page_items=24 if operation=='identity' else 12
    state.setdefault('cursor',0); state.setdefault('winner',None); state.setdefault('audit',[])
    index={c['id']:c for c in cards}
    while state['cursor']<len(cards):
        start=state['cursor']; previous=index.get(state['winner'])
        data={'operation':operation,'candidate':candidate,'previous':previous,'page':[]}
        for card in cards[start:start+page_items]:
            trial=dict(data,page=data['page']+[card])
            if request_size(config,instruction,trial)>limit: break
            data=trial
        require(bool(data['page']), 'a single identity or field description exceeds the page request limit; specifications are not truncated; inspect this object')
        allowed={c['id'] for c in data['page']} | ({previous['id']} if previous else set())
        def validate(value):
            fields(value,'target reason'); text(value['reason'])
            require(value['target'] is None or value['target'] in allowed,'identity/field match references an object outside the page')
            return value
        result=model.structured(instruction,data,validate)
        state['winner']=result['target']; state['cursor']+=len(data['page'])
        state['audit'].append({'start':start,'end':state['cursor'],**result})
        save()
    return state['winner']


def field_view(value,sources):
    kinds={s['id']:s['kind'] for s in sources}
    # The conflict comparison contains only these two fields, so the actual quotes are kept; do not judge only from the generated basis summary.
    return copy.deepcopy(value) | {
        'source_kinds':sorted({kinds[r['source_id']] for r in value['evidence']})}


def resolve_value(old,new,sources,model,config):
    if old is None: return 'incoming','new field'
    def rank(v):
        by_id={s['id']:s for s in sources}
        return (2 if any(by_id[r['source_id']]['kind']=='pdf' for r in v['evidence']) else 1) if v['origin']=='explicit' else 0
    if rank(old)!=rank(new): return ('existing' if rank(old)>rank(new) else 'incoming'),'local source priority'
    if (old['value'],old.get('unit'))==(new['value'],new.get('unit')): return 'existing','identical field value; kept locally with evidence merged'
    data={'operation':'field_value','existing':field_view(old,sources),'incoming':field_view(new,sources)}
    require(request_size(config,SELECT,data)<=min(24000,planning_limit(config)), 'a single field pair exceeds the comparison limit; evidence is not truncated')
    def validate(v):
        fields(v,'take reason'); require(v['take'] in {'existing','incoming'},'invalid field source'); text(v['reason'])
        choose(old,new,v['take'],sources)
        return v
    result=model.structured(SELECT,data,validate)
    return result['take'],result['reason']


def decision_for(asset,assets,sources,model,config,state,save,*,aliases=None):
    if 'target_id' not in state:
        match=state.setdefault('identity',{})
        winner=bounded_scan(identity_card(asset),[identity_card(a) for a in assets],match,model,config,save)
        if winner is None:
            used={a['id'] for a in assets}; winner=asset['id']; n=1
            while winner in used:
                winner=asset['id'][:54]+'_'+str(n); n+=1
        state['target_id']=winner
        state['decision']={'asset_id':asset['id'],'target_id':winner,'reason':
            match['audit'][-1]['reason'] if match.get('audit') else 'registry is empty; assigning a new id',
            'requirements':[]}
        save()
    old=next((a for a in assets if a['id']==state['target_id']),None)
    known_aliases=(aliases or {}).get(state['target_id'],{})
    decision=state['decision']
    if 'quantity_from' not in decision:
        decision['quantity_from'],state['quantity_reason']=resolve_value(old['quantity'] if old else None,asset['quantity'],sources,model,config)
        save()
    existing={r['key']:r for r in old['requirements']} if old else {}
    incoming_by_key={r['key']:r for r in asset['requirements']}
    for entry in decision['requirements']:
        selected=choose(existing.get(entry['target_key']),incoming_by_key[entry['incoming_key']],entry['take'],sources)
        selected['key']=entry['target_key']; existing[entry['target_key']]=selected
    while len(decision['requirements'])<len(asset['requirements']):
        incoming=asset['requirements'][len(decision['requirements'])]
        field_state=state.setdefault('field',{})
        if 'target_key' not in field_state:
            if old is None or incoming['key'] in existing:
                target=incoming['key']
            elif known_aliases.get(incoming['key']) in existing:
                target=known_aliases[incoming['key']]
                field_state['identity_source']='prior_confirmed_alias'
            else:
                target=bounded_scan(field_card(incoming),[field_card(r) for r in existing.values()],
                    field_state.setdefault('match',{}),model,config,save,operation='field_identity')
            field_state['target_key']=target or incoming['key']; save()
        target=field_state['target_key']
        take,reason=resolve_value(existing.get(target),incoming,sources,model,config)
        decision['requirements'].append({'incoming_key':incoming['key'],'target_key':target,'take':take})
        selected=choose(existing.get(target),incoming,take,sources); selected['key']=target; existing[target]=selected
        state.setdefault('field_audit',[]).append({'key':incoming['key'],'reason':reason,**copy.deepcopy(field_state)})
        state.pop('field'); save()
    return copy.deepcopy(decision)
