"""针对任务准备、恢复和下游信息传递的行为验证；全部离线。"""

import json
import base64
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from asset_prep.config import Config
from asset_prep.errors import ValidationError, ProviderError
from asset_prep.prepare_pipeline import execute_preparation, unsupported_search, prepare_references, placeholder
from asset_prep.prepare_workflow import apply_patches, Preparation
from asset_prep.storage import Store, digest, file_hash, atomic_json
from asset_prep.task_contracts import validate_task, validate_roles, public_environment
from asset_prep.task_delivery import load_delivery, validate_role_bindings, validate_task_coverage
from preparation_fixtures import documents, task, inventory, refs, ENV, Transport


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.config=Config(max_model_calls=100,max_reserved_tokens=3000000)
        self.transport=Transport()

    def tearDown(self): self.temp.cleanup()

    def execute(self, **options):
        return execute_preparation(documents(),self.config,{'HARNESS_MODEL_API_KEY':'synthetic'},
            self.root,self.transport,ENV,requirements_only=True,provenance='developer_fixture',**options)

    def reopen(self):
        saved=json.loads((self.root/'state.json').read_text())
        store=Store(self.root,saved['identity'],self.config,resume=True)
        return Preparation(store,documents(),self.config,{'HARNESS_MODEL_API_KEY':'synthetic'},self.transport,ENV)

    def test_complete_selected_task_and_three_consumers(self):
        result=self.execute(); package,manifest=load_delivery(result['delivery'])
        self.assertEqual(result['status'],'requirements_completed')
        self.assertEqual(len(package['inventory']['assets']),1)
        self.assertEqual(package['inventory']['assets'][0]['quantity']['value'],2)
        self.assertEqual(len(package['instance_roles']),2)
        self.assertNotIn('centrifuge',[a['id'] for a in package['inventory']['assets']])
        self.assertTrue(all(not any(s in f['path'] for s in ('state','record','rubric')) for f in manifest['files']))
        from asset_generation.inputs import load_package
        from asset_generation.contracts import GenerationConfig
        from scene_generation.inputs import catalog, task_bundle
        p=Path(result['delivery'])/'asset_package.json'; aid=package['inventory']['assets'][0]['id']
        bundle=load_package(p,[aid],GenerationConfig())
        self.assertNotIn('inventory',bundle['source'])
        self.assertEqual(bundle['source']['task']['instance_roles'],package['instance_roles'])
        c=catalog(p,GenerationConfig()); group=task_bundle(c,{'id':'group','asset_ids':[aid]})
        self.assertEqual(group['source']['task']['task_sha256'],package['task_sha256'])
        from robot_operation.workflow import request_bundle
        plan={'jobs':[{'id':'tubes','asset_ids':[aid]}], 'instances':[{'id':n,'job':'tubes'} for n in ('sample','receiver')],
              'role_bindings':[{'role_id':r['id'],'instances':[r['object_ids'][0]]} for r in package['instance_roles']]}
        scene=self.root/'scene';scene.mkdir();atomic_json(scene/'scene.json',{'plan':plan});atomic_json(scene/'interfaces.json',{})
        with patch('robot_operation.workflow.identity',return_value='synthetic-scene'):
            op=request_bundle(Path(result['delivery']),scene,'synthetic-scene',Path(result['delivery'])/'public_environment.json')
        self.assertEqual(op['prepared_task']['definition'],package['task_definition'])
        self.assertEqual(op['public_profile_sha256'],digest(ENV))
        self.assertNotIn('reviews',package); self.assertNotIn('budget',package)

    def test_saved_response_reused_after_interruption(self):
        original=Store.finish; interrupted=[False]
        def finish(store,call,response):
            original(store,call,response)
            if len(self.transport.operations)==3 and not interrupted[0]:
                interrupted[0]=True;raise KeyboardInterrupt()
        with patch.object(Store,'finish',finish):
            with self.assertRaises(KeyboardInterrupt): self.execute()
        self.assertEqual(self.transport.operations.count('task_definition'),1)
        self.execute(resume=True)
        self.assertEqual(self.transport.operations.count('task_definition'),1)
        before=len(self.transport.operations); self.execute(resume=True)
        self.assertEqual(len(self.transport.operations),before)

    def test_delivery_is_immutable_and_tampering_rejected(self):
        result=self.execute(); folder=Path(result['delivery']); before=file_hash(folder/'asset_package.json')
        self.execute(resume=True);self.assertEqual(file_hash(folder/'asset_package.json'),before)
        (folder/'public_environment.json').write_text('{}',encoding='utf-8')
        with self.assertRaises(ValidationError):load_delivery(folder)
        with self.assertRaises(ValidationError):self.execute(resume=True)

    def test_reference_service_refusal_does_not_silently_fall_back(self):
        result=self.execute(); before=len(self.transport.operations)
        with patch('asset_prep.prepare_pipeline.Search') as search:
            search.return_value.query.side_effect=ProviderError('denied',diagnostics={'http_status':403})
            with self.assertRaises(ProviderError):
                execute_preparation(documents(),self.config,{'HARNESS_MODEL_API_KEY':'synthetic'},self.root,
                    self.transport,ENV,resume=True,provenance='developer_fixture')
        self.assertEqual(len(self.transport.operations),before)
        self.assertTrue(Path(result['delivery']).is_dir())

    def test_role_duplication_and_missing_stage_fail(self):
        p,_=load_delivery(self.execute()['delivery']);roles=p['instance_roles']
        duplicate=deepcopy(roles);duplicate[1]['object_ids']=duplicate[0]['object_ids']
        with self.assertRaises(ValidationError):validate_roles(duplicate,p['task_definition'],p['inventory'])
        duplicate=deepcopy(roles);duplicate[0]['stage_ids']=['open']
        with self.assertRaises(ValidationError):validate_roles(duplicate,p['task_definition'],p['inventory'])

    def test_scene_role_allocation_and_compiled_coverage(self):
        p,_=load_delivery(self.execute()['delivery']);aid=p['inventory']['assets'][0]['id']
        plan={'jobs':[{'id':'tubes','asset_ids':[aid]}], 'instances':[{'id':'sample','job':'tubes'},{'id':'receiver','job':'tubes'}],
              'role_bindings':[{'role_id':r['id'],'instances':[r['object_ids'][0]]} for r in p['instance_roles']]}
        validate_role_bindings(plan,p)
        plan['role_bindings'][1]['instances']=['sample']
        with self.assertRaises(ValidationError):validate_role_bindings(plan,p)
        keys=['stage:open','stage:place','initial:sample','initial:receiver','final:ready']
        coverage=[{'requirement':k,'targets':[],'unresolved':'Detector needed'} for k in keys]
        self.assertEqual(len(validate_task_coverage(coverage,p['task_definition'],{'stages':[]})),5)
        with self.assertRaises(ValidationError):validate_task_coverage(coverage[:-1],p['task_definition'],{'stages':[]})

    def test_targeted_task_patch_preserves_unrelated_objects(self):
        original=task();new=deepcopy(original['objects'][0]);new['initial_state']='Closed and upright'
        reply={'patches':[{'section':'objects','id':'sample','value':new,'reason':'Source correction','evidence':refs()}],'issues':[]}
        changed=apply_patches(original,reply,documents()['sources'])
        self.assertEqual(changed['objects'][1],original['objects'][1]);self.assertNotEqual(changed,original)
        self.assertEqual(original,task())
        changed['stages'][0]['depends_on']=['place']
        with self.assertRaises(ValidationError):validate_task(changed,documents()['sources'])

    def test_only_explicit_unsupported_search_disables_capability(self):
        self.assertTrue(unsupported_search(ProviderError('unsupported',diagnostics={'http_status':400,
            'provider_error':{'message':'web_search is not supported'}})))
        self.assertFalse(unsupported_search(ProviderError('denied',diagnostics={'http_status':403})))
        self.assertEqual(public_environment(ENV)['sha256'],digest(ENV))

    def test_metadata_before_main_route_does_not_force_invented_selection(self):
        class MetadataTransport(Transport):
            def respond(inner,data):
                op=data.get('operation')
                metadata=bool(data.get('sources')) and data['sources'][0]['id']=='pdf:p0'
                if op=='protocol_index' and metadata:return {'entries':[]}
                if op=='flow_selection' and not data['index_page']:
                    return {'pending':True,'reason':'Publication metadata does not define a route.'}
                if op in {'task_definition','task_review'} and metadata:return {'patches':[],'issues':[]}
                if op=='local_extract' and metadata:
                    return {'inventory':{k:[] for k in inventory()},'consumed_units':len(data['units'])}
                return super().respond(data)
        docs=documents();docs['sources'].insert(0,{'id':'pdf:p0','kind':'pdf','text':'Publication metadata. '.ljust(4000,' ')})
        transport=MetadataTransport()
        result=execute_preparation(docs,self.config,{'HARNESS_MODEL_API_KEY':'synthetic'},self.root,transport,ENV,
                                   requirements_only=True,provenance='developer_fixture')
        p,_=load_delivery(result['delivery'])
        self.assertEqual(p['task_definition']['selected_flow'],'Main route')
        self.assertEqual(transport.operations[:2],['protocol_index','protocol_index'])
        self.assertEqual(transport.operations.count('flow_selection'),2)
        self.assertNotIn('pdf:p0',[s['id'] for s in p['sources']])

    def test_unsupported_capability_is_not_probed_for_each_type(self):
        self.execute();prep=self.reopen()
        first=prep.state['inventory']['assets'][0]
        second=deepcopy(first);second['id']='other';second['queries']=['different product']
        prep.state['inventory']['assets'].append(second)
        with patch('asset_prep.prepare_pipeline.Search') as search:
            search.return_value.query.side_effect=ProviderError('unsupported',diagnostics={'http_status':400,
                'provider_error':{'message':'web_search unsupported'}})
            prepare_references(prep,{},self.transport)
            self.assertEqual(search.return_value.query.call_count,1)
            self.assertEqual(len(prep.state['assets']),2)
            prepare_references(prep,{},self.transport)
            self.assertEqual(search.return_value.query.call_count,1)

    def test_missing_credentials_and_transport_errors_stop_references(self):
        self.execute();prep=self.reopen()
        for error in (ProviderError('missing credential'),ProviderError('DNS unavailable')):
            with self.subTest(message=str(error)),patch('asset_prep.prepare_pipeline.Search') as search:
                search.return_value.query.side_effect=error
                with self.assertRaises(ProviderError):prepare_references(prep,{},self.transport)
                self.assertFalse(prep.state['reference_cache'])

    def test_task_revision_preserves_old_delivery_and_unrelated_reference_cache(self):
        delivery=self.execute()['delivery']; original=file_hash(Path(delivery)/'asset_package.json')
        prep=self.reopen();old_task=deepcopy(prep.state['task'])
        first=prep.state['inventory']['assets'][0]
        prep.state['reference_cache'][digest(first)]=placeholder(first)
        cache=deepcopy(prep.state['reference_cache'])
        issue={'id':'order','kind':'task','target_id':'open','description':'Clarify cap handling','evidence':refs()}
        prep.state['reviews']={'open':{'issues':[issue],'input_sha256':'old'},'place':{'issues':[],'input_sha256':'keep'}}
        new_stage=deepcopy(old_task['stages'][0]);new_stage['description']='Open both caps before placement'
        reply={'task_changes':{'patches':[{'section':'stages','id':'open','value':new_stage,'reason':'Clarify source order','evidence':refs()}],
                              'issues':[]},'assets':[],'retire_assets':[],'roles':[],'links':[],'reason':'Clarify source order'}
        with patch.object(self.transport,'respond',return_value=reply):prep.repair([issue])
        self.assertEqual(prep.state['task']['stages'][1],old_task['stages'][1])
        self.assertEqual(prep.state['reference_cache'],cache)
        self.assertEqual(prep.state['reviews']['place']['input_sha256'],'keep')
        self.assertEqual(prep.state['task_history'][-1]['affected_ids'],['open'])
        self.assertEqual(file_hash(Path(delivery)/'asset_package.json'),original)
        load_delivery(delivery)

    def test_public_package_rejects_internal_record_injection(self):
        from asset_prep.task_contracts import validate_package
        p,_=load_delivery(self.execute()['delivery']);p['reviews']={'internal':'must remain private'}
        with self.assertRaises(ValidationError):validate_package(p)

    def test_full_references_use_real_adapter_path_and_reuse_downloads(self):
        class ReferenceTransport(Transport):
            def __init__(inner):super().__init__();inner.searches=0;inner.pages=0;inner.images=0
            def post(inner,url,payload,key,timeout):
                if url.endswith('/responses'):
                    inner.searches+=1
                    return {'status':'completed','model':'fixture','usage':{'output_tokens':20,'total_tokens':80},
                        'output':[{'type':'web_search_call','status':'completed','id':'fixture',
                            'action':{'sources':[{'url':'https://manufacturer.example/tube','title':'Capped tube'}]}}]}
                return super().post(url,payload,key,timeout)
            def request(inner,url,**kwargs):
                if kwargs.get('accept','').startswith('text/html'):
                    inner.pages+=1
                    return b'<html><head><meta property="og:image" content="/tube.png"></head><body>Tube cap is openable.</body></html>','text/html'
                inner.images+=1
                return base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9q8AAAAASUVORK5CYII='),'image/png'
            def respond(inner,data):
                if 'references' in data and 'target' in data:
                    return {'mode':'reference','product_name':'Capped tube','source_ids':['r0'],'selection_reason':'Matches the cap requirement',
                        'specifications':[{'key':'cap','value':'openable','unit':'','source_id':'r0','quote':'Tube cap is openable.'}],
                        'adaptations':[],'image_ids':['i0'],'unknowns':['Geometry dimensions are not specified; retain modeling freedom.']}
                return super().respond(data)
        self.transport=ReferenceTransport()
        result=execute_preparation(documents(),self.config,{'HARNESS_MODEL_API_KEY':'synthetic'},self.root,
                                   self.transport,ENV,provenance='developer_fixture')
        package,manifest=load_delivery(result['delivery'])
        self.assertEqual(result['status'],'completed_with_gaps')
        self.assertEqual((self.transport.searches,self.transport.pages,self.transport.images),(1,1,1))
        self.assertEqual(len(package['assets'][0]['downloaded_images']),1)
        self.assertFalse(package['assets'][0]['downloaded_images'][0]['visual_identity_verified'])
        self.assertEqual(package['assets'][0]['target'],package['inventory']['assets'][0])
        before=result['budget']['model_calls']
        resumed=execute_preparation(documents(),self.config,{'HARNESS_MODEL_API_KEY':'synthetic'},self.root,
                                   self.transport,ENV,provenance='developer_fixture',resume=True)
        self.assertEqual(resumed['budget']['model_calls'],before)
        self.assertEqual((self.transport.searches,self.transport.pages,self.transport.images),(1,1,1))

    def test_scene_planning_preserves_roles_and_scoped_author_context(self):
        from asset_generation.contracts import GenerationConfig
        from scene_generation.inputs import catalog,task_bundle
        from scene_generation.contracts import SceneConfig,numerical_contract
        from scene_generation.pipeline import SceneRun
        from scene_fixtures import environment,plan,SceneSandbox
        from generation_fixtures import FakeTransport
        env=environment(self.root/'robot')
        result=execute_preparation(documents(),self.config,{'HARNESS_MODEL_API_KEY':'synthetic'},self.root/'prep',
            self.transport,env,requirements_only=True,provenance='developer_fixture')
        package,_=load_delivery(result['delivery']);aid=package['inventory']['assets'][0]['id']
        planned=plan()
        for job in planned['jobs']:job['asset_ids']=[aid]
        planned['instances'].append({'id':'receiver','job':'instrument','placement':'table','position':[.7,0,.8],'yaw':0})
        planned['role_bindings']=[{'role_id':package['instance_roles'][0]['id'],'instances':['station']},
                                 {'role_id':package['instance_roles'][1]['id'],'instances':['receiver']}]
        catalogue=catalog(Path(result['delivery'])/'asset_package.json',GenerationConfig())
        transport=FakeTransport([planned])
        run=SceneRun(catalogue,env,numerical_contract({}),self.config,GenerationConfig(),SceneConfig(),
            {'HARNESS_MODEL_API_KEY':'synthetic'},self.root/'scene_plan',transport,SceneSandbox(),self.root/'robot',
            Path(__file__).resolve().parents[1],provenance='developer_fixture')
        run.initialize()
        self.assertEqual(run.state['plan']['role_bindings'],planned['role_bindings'])
        author=task_bundle(catalogue,planned['jobs'][0])
        self.assertEqual(author['source']['task']['instance_roles'],package['instance_roles'])
        self.assertNotIn('inventory',author['source'])

    def test_evidenced_duplicate_retirement_keeps_both_roles_and_counts(self):
        self.execute();prep=self.reopen();original=prep.state['inventory']['assets'][0];aid=original['id']
        duplicate=deepcopy(original);duplicate['id']='duplicate_tube';duplicate['quantity']['value']=1
        prep.state['inventory']['assets'].append(duplicate)
        prep.state['inventory']['steps'].append({'id':'duplicate_step','description':'Repeated mention','asset_ids':['duplicate_tube'],'evidence':refs()})
        expected_roles=deepcopy(prep.all_roles())
        moved=prep.state['roles_by_asset'][aid].pop();moved['asset_id']='duplicate_tube'
        prep.state['roles_by_asset']['duplicate_tube']=[moved]
        issue={'id':'duplicate','kind':'asset','target_id':'duplicate_tube','description':'Same product and geometry','evidence':refs()}
        prep.state['reviews']={'open':{'issues':[issue]}}
        reply={'task_changes':{'patches':[],'issues':[]},'assets':[],
               'retire_assets':[{'asset_id':'duplicate_tube','replacement_id':aid,'reason':'Same geometry repeated','evidence':refs()}],
               'roles':[{'asset_id':aid,'roles':expected_roles}],'links':[],'reason':'Merge geometry; retain independent roles'}
        with patch.object(self.transport,'respond',return_value=reply):prep.repair([issue])
        self.assertEqual(len(prep.state['inventory']['assets']),1)
        self.assertEqual(prep.state['inventory']['assets'][0]['quantity']['value'],2)
        self.assertEqual(len(prep.all_roles()),2)


if __name__=='__main__':unittest.main()
