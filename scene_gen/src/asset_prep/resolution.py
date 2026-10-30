"""身份与字段分别分页；单次请求与登记表总项数解耦。"""

import copy
from .contracts import fields, require, text
from .extraction import request_size, planning_limit
from .incremental import choose


MATCH = """比较当前候选与这一页对象，选出最可能代表同一物理资产/同一规格类型的对象。
previous 是此前页的暂定对象，可以保留或替换；null 表示尚未匹配，所有页完成前程序不会新增。
类别仅是线索，不能单凭类别不同排除同一物体。规格不同可能是不同对象，也可能是后文补充/纠正，结合名称、别名和规格判断。
只返回 {"target": "给定对象的id或null", "reason":"判断理由"}。不得发明id；分页覆盖不等于语义判断正确。"""
FIELD_MATCH = """为当前规格选择这一页已有规格中语义相同的字段。previous 为之前页的暂定字段。
同义名称可映射到同一个key，数值冲突不表示一定是不同字段；单位大小写及小数点有意义。
只返回 {"target":"给定字段id或null","reason":"判断理由"}。未匹配返回null，不发明字段id。"""
SELECT = """这两个字段属于同一资产的同一属性。决定最终取 existing 或 incoming。
优先有PDF原文的显式字段，其次其他显式字段，再次推断；同优先级结合依据选择，不能累加数量或改写值。
只返回 {"take":"existing或incoming","reason":"取舍理由"}。原始双方证据由本地保留。"""


def identity_card(asset):
    return {k:copy.deepcopy(asset[k]) for k in ('id','name','category','aliases')} | {
        'requirements':[{k:r[k] for k in ('key','value','unit')} for r in asset['requirements']]}


def field_card(value):
    return {'id':value['key'],'value':value['value'],'unit':value['unit']}


def bounded_scan(candidate, cards, state, model, config, save, *, operation='identity'):
    """逐页归并暂定匹配；游标及决定每页持久化，未扫描完不能新增。"""
    instruction = MATCH if operation=='identity' else FIELD_MATCH
    limit=min(24000,planning_limit(config))
    # 身份卡不含完整依据；在同一字节上限内装入更多对象，减少重复扫描调用。
    # 字段页仍保持较小粒度。游标是对象偏移，已有 12 项页的检查点无需重算。
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
        require(bool(data['page']), '单项身份或字段描述超过分页请求上限；不截断规格，需检查该对象')
        allowed={c['id'] for c in data['page']} | ({previous['id']} if previous else set())
        def validate(value):
            fields(value,'target reason'); text(value['reason'])
            require(value['target'] is None or value['target'] in allowed,'身份/字段匹配引用页外对象')
            return value
        result=model.structured(instruction,data,validate)
        state['winner']=result['target']; state['cursor']+=len(data['page'])
        state['audit'].append({'start':start,'end':state['cursor'],**result})
        save()
    return state['winner']


def field_view(value,sources):
    kinds={s['id']:s['kind'] for s in sources}
    # 冲突比较只含这两个字段，因此保留实际引文，不能仅凭生成的依据摘要判断。
    return copy.deepcopy(value) | {
        'source_kinds':sorted({kinds[r['source_id']] for r in value['evidence']})}


def resolve_value(old,new,sources,model,config):
    if old is None: return 'incoming','新增字段'
    def rank(v):
        by_id={s['id']:s for s in sources}
        return (2 if any(by_id[r['source_id']]['kind']=='pdf' for r in v['evidence']) else 1) if v['origin']=='explicit' else 0
    if rank(old)!=rank(new): return ('existing' if rank(old)>rank(new) else 'incoming'),'本地来源优先级'
    if (old['value'],old.get('unit'))==(new['value'],new.get('unit')): return 'existing','相同字段值，本地保留并合并证据'
    data={'operation':'field_value','existing':field_view(old,sources),'incoming':field_view(new,sources)}
    require(request_size(config,SELECT,data)<=min(24000,planning_limit(config)), '单个字段对超过比较上限；不截断依据')
    def validate(v):
        fields(v,'take reason'); require(v['take'] in {'existing','incoming'},'字段来源非法'); text(v['reason'])
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
            match['audit'][-1]['reason'] if match.get('audit') else '登记表为空，分配新标识',
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
