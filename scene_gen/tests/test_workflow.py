"""有界请求与持久化状态验收；全部为开发者合成数据、禁止真实接口。"""

import copy
import json
from dataclasses import replace
from unittest.mock import patch
from asset_prep import blocks
from asset_prep.config import Config
from asset_prep.errors import BudgetExceeded, StaleEvidence, ValidationError
from asset_prep.extraction import request_size
from asset_prep.pipeline import execute
from asset_prep.resolution import bounded_scan, decision_for, identity_card
from asset_prep.storage import atomic_json, digest, Store
from asset_prep.workflow import LOCAL, local_data, new_workflow, run_workflow, seed_active
from fixtures import documents, inventory, PROVENANCE
from test_boundaries import TempCase


class SyntheticModel:
    def __init__(self,draft=None,target=None,stop_at=None):
        self.draft=draft or inventory(); self.target=target; self.stop_at=stop_at; self.requests=[]
    def structured(self,instruction,data,validate):
        if self.stop_at is not None and len(self.requests)==self.stop_at: raise BudgetExceeded('synthetic pause')
        self.requests.append(copy.deepcopy(data))
        operation=data['operation']
        if operation=='local_extract':
            result={'inventory':copy.deepcopy(self.draft),'consumed_units':len(data['units'])}
        elif operation in {'identity','field_identity'}:
            current=data['previous']['id'] if data['previous'] else None
            wanted=self.target or data['candidate']['id']
            for card in data['page']:
                if card['id']==wanted: current=card['id']
            result={'target':current,'reason':'开发者合成匹配'}
        else: result={'take':'incoming','reason':'开发者合成冲突取舍'}
        return validate(result)


class WorkflowTransport:
    def __init__(self,draft=None,target=None): self.model=SyntheticModel(draft,target); self.calls=0
    def post(self,url,payload,key,timeout):
        self.calls+=1
        data=json.loads(payload['messages'][1]['content'].split('\n以下是任务数据：\n',1)[1])
        result=self.model.structured('',data,lambda v:v)
        return {'model':'fixture','choices':[{'finish_reason':'stop','message':{'content':json.dumps(result)}}],
                'usage':{'total_tokens':50}}


class WorkflowTests(TempCase):
    def catalog(self,count,mixed=False):
        result=inventory(); result['assets']=[]; result['conflicts']=[]
        for i in range(count):
            asset=copy.deepcopy(inventory()['assets'][0]); asset.update(id=f'asset{i}',name=f'独立器具{i}',aliases=[])
            if mixed: asset['category']=('instrument','container','consumable','accessory')[i%4]
            result['assets'].append(asset)
        result['steps'][0]['asset_ids']=[a['id'] for a in result['assets']]
        return result

    def test_identity_10_100_1000_same_and_mixed_category_is_bounded_and_covers_all(self):
        config=Config()
        for count in (10,100,1000):
            for mixed in (False,True):
                with self.subTest(count=count,mixed=mixed):
                    assets=self.catalog(count,mixed)['assets']; candidate=copy.deepcopy(assets[-1])
                    model=SyntheticModel(target=assets[-1]['id']); state={}
                    target=bounded_scan(identity_card(candidate),[identity_card(a) for a in assets],state,model,config,lambda:None)
                    self.assertEqual(target,assets[-1]['id']); self.assertEqual(state['cursor'],count)
                    self.assertEqual(sum(len(d['page']) for d in model.requests),count)
                    from asset_prep.resolution import MATCH
                    self.assertTrue(all(request_size(config,MATCH,d)<=24000 for d in model.requests))
                    self.assertTrue(all(len(d['page'])<=24 for d in model.requests))
                    self.assertEqual(len(model.requests),(count+23)//24)

    def test_local_extraction_payload_is_independent_of_registry_size(self):
        docs=documents(); config=Config(); block=blocks.plan(docs['sources'],4000)[0]
        before=local_data(block,docs,config)
        for count in (10,100,1000):
            state=new_workflow(docs,config); state['catalog']=self.catalog(count)
            self.assertEqual(local_data(state['queue'][0],docs,config),before)
        self.assertNotIn('existing_assets',before)

    def test_balanced_split_preserves_every_character_and_offset_without_singleton(self):
        for count in (1677,453,4001,8001):
            source={'id':'pdf:p1','kind':'pdf','text':'x'*count,'offset':35}
            parts=blocks.balanced_parts(source,(count+1)//2)
            self.assertEqual(''.join(p['text'] for p in parts),source['text'])
            self.assertTrue(all(len(p['text'])>1 for p in parts))
            cursor=35
            for part in parts:
                self.assertEqual(part['offset'],cursor); cursor+=len(part['text'])

    def test_short_line_is_retained_in_adjacent_unit(self):
        text='Material row\n1\nNext material row\n'
        items=blocks.units([{'id':'pdf:p1','kind':'pdf','text':text}],4000)
        self.assertEqual(''.join(p['text'] for p in items),text)
        self.assertFalse(any(p['text'].strip()=='1' for p in items))

    def test_identity_resume_starts_at_saved_page(self):
        assets=self.catalog(100)['assets']; candidate=copy.deepcopy(assets[-1]); state={}; saved=[]
        model=SyntheticModel(target=assets[-1]['id'],stop_at=2)
        with self.assertRaises(BudgetExceeded):
            bounded_scan(identity_card(candidate),[identity_card(a) for a in assets],state,model,Config(),lambda:saved.append(copy.deepcopy(state)))
        self.assertEqual(saved[-1]['cursor'],48)
        next_model=SyntheticModel(target=assets[-1]['id'])
        bounded_scan(identity_card(candidate),[identity_card(a) for a in assets],saved[-1],next_model,Config(),lambda:None)
        self.assertEqual(next_model.requests[0]['page'][0]['id'],'asset48')

    def test_old_page_cursor_resumes_with_larger_pages_without_rescanning(self):
        assets=self.catalog(100)['assets']
        state={'cursor':12,'winner':'asset0','audit':[{'start':0,'end':12,'target':'asset0','reason':'saved earlier page'}]}
        model=SyntheticModel(target='asset99')
        result=bounded_scan(identity_card(assets[-1]),[identity_card(a) for a in assets],state,model,Config(),lambda:None)
        self.assertEqual(result,'asset99')
        self.assertEqual(model.requests[0]['page'][0]['id'],'asset12')
        self.assertEqual([c['id'] for d in model.requests for c in d['page']],[a['id'] for a in assets[12:]])

    def test_confirmed_field_alias_skips_repeated_matching_but_not_value_conflicts(self):
        old=inventory()['assets'][0]; incoming=copy.deepcopy(old)
        incoming['requirements']=incoming['requirements'][:1]
        incoming['requirements'][0].update(key='volume_alias',value='0.5')
        state={'target_id':'tubes','decision':{'asset_id':'tubes','target_id':'tubes','reason':'confirmed','requirements':[]}}
        model=SyntheticModel()
        decision=decision_for(incoming,[old],documents()['sources'],model,Config(),state,lambda:None,
                              aliases={'tubes':{'volume_alias':'capacity'}})
        self.assertEqual([d['operation'] for d in model.requests],['field_value'])
        self.assertEqual(decision['requirements'][0]['target_key'],'capacity')
        self.assertEqual(state['field_audit'][0]['identity_source'],'prior_confirmed_alias')

    def test_missing_alias_target_falls_back_to_model(self):
        old=inventory()['assets'][0]; incoming=copy.deepcopy(old)
        incoming['requirements']=incoming['requirements'][:1]; incoming['requirements'][0]['key']='volume_alias'
        state={'target_id':'tubes','decision':{'asset_id':'tubes','target_id':'tubes','reason':'confirmed','requirements':[]}}
        model=SyntheticModel(target='capacity')
        decision_for(incoming,[old],documents()['sources'],model,Config(),state,lambda:None,
                     aliases={'tubes':{'volume_alias':'removed_field'}})
        self.assertEqual([d['operation'] for d in model.requests],['field_identity'])

    def test_alias_history_excludes_conflicts_and_separates_asset_identity(self):
        from asset_prep.workflow import confirmed_field_aliases
        def entry(aid,target):
            return {'target_id':aid,'requirements':[{'incoming_key':'size','target_key':target}]}
        state={'commits':[{'decisions':[entry('a','capacity'),entry('b','diameter')]}],
               'active':{'decisions':[entry('a','length')]}}
        self.assertEqual(confirmed_field_aliases(state),{'a':{},'b':{'size':'diameter'}})

    def test_new_asset_not_committed_until_all_identity_pages_checked(self):
        docs=documents(); config=replace(Config(),max_assets=200)
        state=new_workflow(docs,config); state['catalog']=self.catalog(100)
        model=SyntheticModel(target='unknown',stop_at=3); saved=[]
        with self.assertRaises(BudgetExceeded):
            run_workflow(docs,config,model,state,lambda value:saved.append(value))
        self.assertEqual(len(saved[-1]['catalog']['assets']),100)
        self.assertEqual(saved[-1]['active']['cursor'],0)
        self.assertNotIn('target_id',saved[-1]['active']['candidate_state'])

    def test_cross_category_identity_reuses_id_and_records_disagreement(self):
        docs=documents(); state=new_workflow(docs,Config()); old=inventory(); old['assets'][0]['category']='accessory'
        state['catalog']=old
        result=run_workflow(docs,Config(),SyntheticModel(),state)
        self.assertEqual(len(result['assets']),1)
        self.assertEqual(result['assets'][0]['category'],'accessory')
        self.assertTrue(any('跨类别' in i for i in result['issues']))

    def test_field_pagination_and_same_rank_conflict_are_independent(self):
        docs=documents(); asset=inventory()['assets'][0]; old=copy.deepcopy(asset)
        old['requirements']=[]
        for i in range(100):
            field=copy.deepcopy(asset['requirements'][0]); field['key']=f'field{i}'; old['requirements'].append(field)
        asset['requirements']=asset['requirements'][:1]; asset['requirements'][0].update(key='new_alias',value='0.5')
        state={'target_id':'tubes','decision':{'asset_id':'tubes','target_id':'tubes','reason':'已确认身份','requirements':[]}}
        model=SyntheticModel(target='field99')
        decision=decision_for(asset,[old],docs['sources'],model,Config(),state,lambda:None)
        self.assertEqual(decision['requirements'][0]['target_key'],'field99')
        self.assertEqual(decision['requirements'][0]['take'],'incoming')
        self.assertEqual(state['field_audit'][0]['match']['cursor'],100)
        self.assertTrue(all(d['operation']!='identity' for d in model.requests))

    def test_local_pagination_advances_only_consumed_units(self):
        docs=documents(); docs['sources']=docs['sources'][:1]
        docs['sources'][0]['text']='tube first\ntube second\n'
        draft=inventory(); draft['conflicts']=[]
        from asset_prep.extraction import all_evidence
        for ref in all_evidence(draft): ref.update(source_id='pdf:p1',quote='tube')
        class Partial(SyntheticModel):
            def structured(self,instruction,data,validate):
                if data['operation']=='local_extract':
                    self.requests.append(copy.deepcopy(data))
                    return validate({'inventory':copy.deepcopy(draft),'consumed_units':1})
                return super().structured(instruction,data,validate)
        model=Partial(draft); saved=[]
        result=run_workflow(docs,Config(),model,persist=lambda v:saved.append(v))
        extracts=[d for d in model.requests if d['operation']=='local_extract']
        self.assertEqual(len(extracts),2)
        self.assertEqual(extracts[1]['units'][0]['offset'],11)
        self.assertEqual(len(result['assets']),1)
        self.assertEqual(len(saved[-1]['commits']),2)

    def test_persistent_pipeline_resume_needs_no_extraction_requests(self):
        config=replace(Config(),search_provider='tavily'); keys={'HARNESS_MODEL_API_KEY':'offline-secret'}
        transport=WorkflowTransport()
        result=execute(documents(),config,keys,self.root,transport,extract_only=True,provenance=PROVENANCE)
        self.assertEqual(transport.calls,1)
        resumed=WorkflowTransport()
        again=execute(documents(),config,keys,self.root,resumed,resume=True,extract_only=True,provenance=PROVENANCE)
        self.assertEqual(resumed.calls,0); self.assertEqual(again['inventory'],result['inventory'])

    def test_workflow_tamper_is_rejected(self):
        config=Config(); keys={'HARNESS_MODEL_API_KEY':'offline-secret'}
        execute(documents(),config,keys,self.root,WorkflowTransport(),extract_only=True,provenance=PROVENANCE)
        path=self.root/'state.json'; state=json.loads(path.read_text(encoding='utf-8'))
        state['workflow']['revision']+=1; atomic_json(path,state)
        with self.assertRaises(StaleEvidence):
            execute(documents(),config,keys,self.root,WorkflowTransport(),resume=True,extract_only=True,provenance=PROVENANCE)

    def test_every_state_save_can_resume_without_repeating_completed_operations(self):
        docs=documents(); config=Config(); state=new_workflow(docs,config)
        state['catalog']=inventory()
        draft=inventory(); draft['conflicts']=[]
        draft['assets'][0]['quantity']['value']=3
        draft['assets'][0]['requirements'][0].update(key='capacity_alias',value='0.5')
        class Matching(SyntheticModel):
            def structured(self,instruction,data,validate):
                if data['operation']=='field_identity':
                    self.requests.append(copy.deepcopy(data))
                    return validate({'target':'capacity','reason':'same synthetic property'})
                return super().structured(instruction,data,validate)
        # 明确覆盖身份、数量冲突、字段匹配/冲突、候选应用、总表提交的每次持久化。
        model=Matching(draft); snapshots=[]
        result=run_workflow(docs,config,model,state,lambda s:snapshots.append(s))
        self.assertIn('field_value',{d['operation'] for d in model.requests})
        class Crash(BaseException): pass
        for boundary in range(1,len(snapshots)+1):
            with self.subTest(boundary=boundary):
                saved=[]; first=Matching(draft)
                def persist(s):
                    saved.append(s)
                    if len(saved)==boundary: raise Crash()
                with self.assertRaises(Crash): run_workflow(docs,config,first,state,persist)
                second=Matching(draft)
                resumed=run_workflow(docs,config,second,saved[-1])
                self.assertEqual(resumed,result)
                self.assertEqual(first.requests+second.requests,model.requests)

    def test_publication_crashes_recover_committed_registry_without_new_requests(self):
        config=Config(); keys={'HARNESS_MODEL_API_KEY':'offline-secret'}
        class Crash(BaseException): pass
        for boundary in ('intent','documents','package','committed_state'):
            with self.subTest(boundary=boundary):
                root=self.root/boundary; stopped=[]
                def write(path,value,**kwargs):
                    atomic_json(path,value,**kwargs)
                    if stopped: return
                    if path.name=='state.json':
                        revision=value.get('workflow',{}).get('revision',0)
                    else:
                        state=json.loads((root/'state.json').read_text(encoding='utf-8'))
                        revision=state.get('workflow',{}).get('revision',0)
                    if revision!=1: return
                    hit=((boundary=='intent' and path.name=='state.json' and 'publication_intent' in value)
                         or (boundary=='committed_state' and path.name=='state.json' and 'publication_intent' not in value)
                         or (boundary=='documents' and path.name=='documents.json')
                         or (boundary=='package' and path.name=='asset_package.json'))
                    if hit: stopped.append(True); raise Crash()
                first=WorkflowTransport()
                with patch('asset_prep.storage.atomic_json',side_effect=write), patch('asset_prep.pipeline.atomic_json',side_effect=write), patch('asset_prep.state_archive.atomic_json',side_effect=write):
                    with self.assertRaises(Crash):
                        execute(documents(),config,keys,root,first,extract_only=True,provenance=PROVENANCE)
                second=WorkflowTransport()
                result=execute(documents(),config,keys,root,second,resume=True,extract_only=True,provenance=PROVENANCE)
                self.assertEqual(second.calls,0)
                self.assertEqual(result['extraction']['revision'],1)
                self.assertEqual(len(result['inventory']['assets']),1)

    def test_compatible_code_change_resumes_state_but_contract_change_is_rejected(self):
        config=Config(); keys={'HARNESS_MODEL_API_KEY':'offline-secret'}
        with patch('asset_prep.pipeline.code_identity',return_value='implementation_a'):
            execute(documents(),config,keys,self.root,WorkflowTransport(),extract_only=True,provenance=PROVENANCE)
        second=WorkflowTransport()
        with patch('asset_prep.pipeline.code_identity',return_value='implementation_b'):
            execute(documents(),config,keys,self.root,second,resume=True,extract_only=True,provenance=PROVENANCE)
        self.assertEqual(second.calls,0)
        state=json.loads((self.root/'state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['implementation_history'][0]['from'],'implementation_a')
        changed=copy.deepcopy(state['identity']); changed['workflow_contract']='incompatible_future_version'
        with self.assertRaises(StaleEvidence): Store(self.root,changed,config,resume=True)

    def test_identical_fields_union_evidence_locally(self):
        from asset_prep.incremental import apply_decisions
        old=inventory()['assets'][0]; new=copy.deepcopy(old)
        ref={'source_id':'pdf:p1','quote':'screw cap'}
        new['quantity']['evidence']=[ref]; new['requirements'][0]['evidence']=[ref]
        model=SyntheticModel(); decision=decision_for(new,[old],documents()['sources'],model,Config(),{},lambda:None)
        self.assertEqual([r['operation'] for r in model.requests],['identity'])
        updated,_,_=apply_decisions({'decisions':[decision]},[old],[new],documents()['sources'],resolved_identity=True)
        for before,after in ((old['quantity'],updated[0]['quantity']),(old['requirements'][0],updated[0]['requirements'][0])):
            self.assertIn(ref,after['evidence'])
            for evidence in before['evidence']: self.assertIn(evidence,after['evidence'])

    def test_response_saved_before_task_progress_is_reused_without_paid_retry(self):
        config=Config(); keys={'HARNESS_MODEL_API_KEY':'offline-secret'}
        finish=Store.finish
        class Crash(BaseException): pass
        def interrupted(store,call,response):
            finish(store,call,response)
            raise Crash()
        with patch.object(Store,'finish',interrupted):
            with self.assertRaises(Crash):
                execute(documents(),config,keys,self.root,WorkflowTransport(),extract_only=True,provenance=PROVENANCE)
        transport=WorkflowTransport()
        result=execute(documents(),config,keys,self.root,transport,resume=True,extract_only=True,provenance=PROVENANCE)
        self.assertEqual(transport.calls,0)
        self.assertEqual(result['budget']['model_calls'],1)
        self.assertEqual(len(result['inventory']['assets']),1)

    def test_field_scan_is_bounded_at_1000_fields(self):
        from asset_prep.resolution import FIELD_MATCH
        cards=[{'id':f'property{i}','value':'synthetic','unit':''} for i in range(1000)]
        model=SyntheticModel(target='property999')
        result=bounded_scan({'id':'alias','value':'synthetic','unit':''},cards,{},model,Config(),lambda:None,operation='field_identity')
        self.assertEqual(result,'property999')
        self.assertEqual(sum(len(d['page']) for d in model.requests),1000)
        self.assertTrue(all(request_size(Config(),FIELD_MATCH,d)<=24000 for d in model.requests))

    def test_oversized_atomic_card_stops_without_discarding_requirements(self):
        card=identity_card(inventory()['assets'][0]); card['aliases']=['x'*30000]
        original=copy.deepcopy(card); model=SyntheticModel()
        with self.assertRaisesRegex(ValidationError,'不截断规格'):
            bounded_scan(identity_card(inventory()['assets'][0]),[card],{},model,Config(),lambda:None)
        self.assertEqual(card,original); self.assertEqual(model.requests,[])

    def test_multiple_field_aliases_can_share_one_target_without_losing_evidence(self):
        from asset_prep.incremental import apply_decisions
        old=inventory()['assets'][0]; new=copy.deepcopy(old)
        first=copy.deepcopy(old['requirements'][0]); first['key']='capacity_a'
        second=copy.deepcopy(first); second['key']='capacity_b'; second['evidence']=[{'source_id':'pdf:p1','quote':'screw cap'}]
        new['requirements']=[first,second]
        state={'target_id':'tubes','decision':{'asset_id':'tubes','target_id':'tubes','reason':'known identity','requirements':[]}}
        model=SyntheticModel(target=old['requirements'][0]['key'])
        decision=decision_for(new,[old],documents()['sources'],model,Config(),state,lambda:None)
        updated,_,_=apply_decisions({'decisions':[decision]},[old],[new],documents()['sources'],resolved_identity=True)
        self.assertEqual(len(updated[0]['requirements']),len(old['requirements']))
        self.assertIn(second['evidence'][0],updated[0]['requirements'][0]['evidence'])
