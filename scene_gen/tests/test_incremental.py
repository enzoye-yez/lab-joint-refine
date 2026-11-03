"""增量决定、暂停升级和错误诊断测试；全部离线合成。"""

import copy
import json
from dataclasses import replace
from unittest.mock import Mock, patch

from asset_prep.config import Config
from asset_prep.errors import BudgetExceeded, OutputTooLarge, ProviderError, ValidationError
from asset_prep.incremental import apply_decisions, compact, merge_incrementally, scoped_data
from asset_prep.network import Transport
from asset_prep.pipeline import execute, run_identity
from asset_prep.storage import Store, atomic_json, digest, file_hash
from asset_prep.upgrade import LEGACY_CODE, SCOPED_REPLAY_CODE, legacy_upgrade_run as upgrade_run
from fixtures import documents, inventory, delta_decisions, PROVENANCE, FakeTransport
from test_boundaries import TempCase


class IncrementalTests(TempCase):
    def large_catalog(self):
        catalog=inventory(); catalog['conflicts']=[]
        other=copy.deepcopy(catalog['assets'][0])
        other.update(id='large_instrument',name='合成大型仪器',category='instrument')
        other['quantity']['basis']='合成依据' * 5000
        catalog['assets'].append(other)
        catalog['steps'][0]['asset_ids'].append(other['id'])
        return catalog

    def test_overflow_scopes_all_eligible_targets_without_mutating_catalog(self):
        from asset_prep.extraction import request_size
        catalog=self.large_catalog(); before=copy.deepcopy(catalog); draft=inventory()
        seen=[]
        class SmallModel:
            def structured(self,instruction,data,validate):
                seen.append(copy.deepcopy(data))
                assert request_size(Config(),instruction,data)<=48000
                return validate(delta_decisions(data['incoming_assets']))
        result,_,_=merge_incrementally(catalog,draft,documents()['sources'],Config(),SmallModel(),0)
        self.assertEqual(catalog,before)
        self.assertEqual(seen[0]['existing_assets'],[compact(catalog['assets'][0],documents()['sources'])])
        self.assertEqual(seen[0]['reserved_asset_ids'],['large_instrument'])
        self.assertEqual(result['assets'][1],catalog['assets'][1])
        self.assertEqual(len(result['assets']),2)

    def test_scoped_output_split_keeps_every_candidate(self):
        catalog=self.large_catalog(); draft=inventory(); draft['conflicts']=[]; draft['assets']=[]
        for i in range(4):
            asset=copy.deepcopy(inventory()['assets'][0]); asset.update(id=f'alias{i}')
            draft['assets'].append(asset)
        draft['steps'][0]['asset_ids']=[a['id'] for a in draft['assets']]
        sizes=[]
        class SmallModel:
            def structured(self,instruction,data,validate):
                sizes.append(len(data['incoming_assets']))
                assert data['reserved_asset_ids']==['large_instrument']
                if sizes[-1]>2: raise OutputTooLarge('synthetic pressure')
                return validate(delta_decisions(data['incoming_assets'],'tubes'))
        result,_,decisions=merge_incrementally(catalog,draft,documents()['sources'],Config(),SmallModel(),0)
        self.assertEqual(sizes,[4,2,2]); self.assertEqual(len(decisions),4)
        self.assertEqual(len(result['assets']),2)

    def test_scoping_preserves_all_categories_present_in_batch_and_rejects_hidden_id(self):
        catalog=self.large_catalog(); sources=documents()['sources']
        data=scoped_data(catalog['assets'],catalog['assets'],sources)
        self.assertEqual(data['existing_assets'],[compact(a,sources) for a in catalog['assets']])
        self.assertEqual(data['reserved_asset_ids'],[])
        candidate=inventory()['assets'][0]
        with self.assertRaisesRegex(ValidationError,'across categories'):
            apply_decisions(delta_decisions([candidate],'large_instrument'),catalog['assets'],[candidate],sources)

    def test_same_category_overflow_never_drops_eligible_targets(self):
        catalog=self.large_catalog(); catalog['assets'][1]['category']='container'
        model=Mock()
        with self.assertRaisesRegex(ValidationError,'all same-category'):
            merge_incrementally(catalog,inventory(),documents()['sources'],Config(),model,0)
        model.structured.assert_not_called()

    def test_oversized_decisions_split_batch_without_losing_candidates(self):
        draft = inventory(); draft['assets'] = []; draft['conflicts'] = []
        for i in range(4):
            asset = copy.deepcopy(inventory()['assets'][0])
            asset.update(id=f'asset{i}', name=f'不同设备{i}')
            draft['assets'].append(asset)
        draft['steps'][0]['asset_ids'] = [a['id'] for a in draft['assets']]
        sizes = []
        class SmallModel:
            def structured(self, instruction, data, validate):
                sizes.append(len(data['incoming_assets']))
                if sizes[-1] > 2:
                    raise OutputTooLarge('synthetic pressure')
                return validate(delta_decisions(data['incoming_assets']))
        result, calls, _ = merge_incrementally({k: [] for k in draft}, draft, documents()['sources'], Config(), SmallModel(), 0)
        self.assertEqual(sizes, [4, 2, 2])
        self.assertEqual(calls, 2)
        self.assertEqual([a['id'] for a in result['assets']], [a['id'] for a in draft['assets']])

    def test_single_oversized_decision_stops(self):
        draft = inventory()
        model = Mock()
        model.structured.side_effect = OutputTooLarge('synthetic pressure')
        with self.assertRaisesRegex(ValidationError, 'single candidate'):
            merge_incrementally({k: [] for k in draft}, draft, documents()['sources'], Config(), model, 0)
        self.assertEqual(model.structured.call_count, 1)

    def test_local_merge_retains_unchanged_assets_fields_and_evidence(self):
        old = inventory()['assets'][0]
        untouched = copy.deepcopy(old)
        untouched.update(id='other', name='独立其他资产', aliases=[])
        untouched['requirements'][0]['value'] = '10'
        new = copy.deepcopy(old)
        new['id'] = 'alias_tubes'
        new['requirements'] = new['requirements'][:1]
        new['evidence'] = [{'source_id':'pdf:p1','quote':'screw cap'}]
        old_before = copy.deepcopy(old)
        result, mapping, _ = apply_decisions(delta_decisions([new], 'tubes'), [old, untouched], [new], documents()['sources'])
        self.assertEqual(old, old_before)
        self.assertEqual(result[1], untouched)
        self.assertEqual(len(result[0]['requirements']), 2)
        self.assertIn(new['evidence'][0], result[0]['evidence'])
        self.assertIn(old['evidence'][0], result[0]['evidence'])
        self.assertEqual(result[0]['quantity']['value'], 2)
        self.assertEqual(mapping, {'alias_tubes':'tubes'})

    def test_missing_or_duplicate_decisions_and_requirements_rejected(self):
        asset = inventory()['assets'][0]
        valid = delta_decisions([asset])
        invalid = [ {'decisions': []}, {'decisions': valid['decisions'] * 2} ]
        missing = copy.deepcopy(valid); missing['decisions'][0]['requirements'].pop(); invalid.append(missing)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                apply_decisions(value, [], [asset], documents()['sources'])

    def test_xml_or_inference_cannot_overwrite_pdf_quantity(self):
        old = inventory()['assets'][0]
        new = copy.deepcopy(old)
        new['quantity'].update(value=3, evidence=[{'source_id':'xml:n1','quote':'three tubes'}])
        with self.assertRaisesRegex(ValidationError, 'PDF'):
            apply_decisions(delta_decisions([new]), [old], [new], documents()['sources'])
        decision = delta_decisions([new]); decision['decisions'][0]['quantity_from'] = 'existing'
        result, _, _ = apply_decisions(decision, [old], [new], documents()['sources'])
        self.assertEqual(result[0]['quantity']['value'], 2)
        self.assertIn(new['quantity']['evidence'][0], result[0]['evidence'])

    def test_same_batch_synonyms_reuse_new_id(self):
        first = inventory()['assets'][0]
        second = copy.deepcopy(first); second['id'] = 'alias'
        decision = delta_decisions([first, second], 'tubes')
        result, mapping, _ = apply_decisions(decision, [], [first, second], documents()['sources'])
        self.assertEqual(len(result), 1)
        self.assertEqual(mapping, {'tubes':'tubes', 'alias':'tubes'})

    def test_many_candidates_are_bounded_without_full_catalog_output(self):
        draft = inventory(); draft['assets'] = []; draft['conflicts'] = []
        for i in range(13):
            asset = copy.deepcopy(inventory()['assets'][0])
            asset.update(id=f'asset{i}', name=f'不同设备{i}')
            draft['assets'].append(asset)
        draft['steps'][0]['asset_ids'] = [a['id'] for a in draft['assets']]
        sizes = []
        class Model:
            def structured(self, instruction, data, validate):
                sizes.append(len(data['incoming_assets']))
                assert 'evidence' not in json.dumps(data)
                return validate(delta_decisions(data['incoming_assets']))
        empty = {k: [] for k in draft}
        result, calls, _ = merge_incrementally(empty, draft, documents()['sources'], Config(), Model(), 0)
        self.assertEqual(sizes, [4,4,4,1])
        self.assertEqual(calls, 4)
        self.assertEqual(len(result['assets']), 13)

    def test_budget_between_batches_reuses_paid_decisions_without_duplicates(self):
        draft = inventory(); draft['assets'] = []; draft['conflicts'] = []
        for i in range(6):
            asset = copy.deepcopy(inventory()['assets'][0])
            asset.update(id=f'asset{i}', name=f'不同设备{i}')
            draft['assets'].append(asset)
        draft['steps'][0]['asset_ids'] = [a['id'] for a in draft['assets']]
        config = replace(Config(), search_provider='tavily', max_model_calls=2)
        keys = {'HARNESS_MODEL_API_KEY':'fixture'}
        with self.assertRaises(BudgetExceeded):
            execute(documents(), config, keys, self.root,
                    FakeTransport([draft, {'target':None,'reason':'distinct synthetic object'}]), provenance=PROVENANCE, extract_only=True)
        partial = json.loads((self.root/'asset_package.json').read_text(encoding='utf-8'))
        self.assertEqual(partial['inventory']['assets'], [])
        state=json.loads((self.root/'state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['workflow']['active']['cursor'],2)
        resumed = FakeTransport([{'target':None,'reason':'distinct synthetic object'} for _ in range(4)])
        package = execute(documents(), replace(config, max_model_calls=40), keys, self.root,
                          resumed, resume=True, provenance=PROVENANCE, extract_only=True)
        self.assertEqual(resumed.model_calls, 4)
        self.assertEqual(len(package['inventory']['assets']), 6)
        self.assertEqual(package['budget']['model_calls'], 6)


class UpgradeTests(TempCase):
    def test_dynamic_split_upgrade_replays_cache_and_continues_from_correct_text(self):
        from asset_prep.extraction import all_evidence
        source={'id':'pdf:p1','kind':'pdf','text':'tube '*160}
        docs={'sources':[source],'warnings':[],'files':{'pdf':{'name':'fixture.pdf','sha256':'fixture'}}}
        draft=inventory(); draft['conflicts']=[]
        for ref in all_evidence(draft): ref.update(source_id='pdf:p1',quote='tube ')
        class Dynamic(FakeTransport):
            def post(self,url,payload,key,timeout):
                data=json.loads(payload['messages'][1]['content'].split('\nTask data follows:\n',1)[1])
                if 'segment' in data and len(data['sources'][0]['text'])>400:
                    self.model_calls+=1
                    return {'choices':[{'finish_reason':'length','message':{'content':''}}]}
                self.replies.append(draft if 'segment' in data else delta_decisions(data['incoming_assets']))
                return super().post(url,payload,key,timeout)
        from asset_prep.providers import Model
        from asset_prep.extraction import extract_inventory
        from asset_prep.migration import migrate_run
        config=replace(Config(),max_model_calls=3); keys={'HARNESS_MODEL_API_KEY':'offline-secret-not-in-input'}
        old=run_identity(docs,config,PROVENANCE); old.pop('workflow_contract'); old['code']=SCOPED_REPLAY_CODE
        store=Store(self.root,old,config); progress={}; saved={}
        def publish(catalog): saved['inventory']=copy.deepcopy(catalog)
        with self.assertRaises(BudgetExceeded):
            extract_inventory(docs,config,Model(config,keys,store,Dynamic([])),progress,publish)
        package={'identity':old,'status':'paused','inventory':saved['inventory'],'extraction':progress}
        for name,value in [('asset_package.json',package),('documents.json',docs)]:
            atomic_json(self.root/name,value)
            store.state.setdefault('outputs',{})[name]=file_hash(self.root/name)
        store.save(); original=copy.deepcopy(store.state)
        resumed_config=replace(config,max_model_calls=40)
        identity=run_identity(docs,resumed_config,PROVENANCE)
        changed=copy.deepcopy(identity); changed['parsed_sources']='different sources'
        with self.assertRaises(ValidationError): migrate_run(self.root,changed,docs,resumed_config)
        unknown=copy.deepcopy(original); unknown['identity']['code']='unsupported'
        atomic_json(self.root/'state.json',unknown)
        with self.assertRaises(ValidationError): migrate_run(self.root,identity,docs,resumed_config)
        atomic_json(self.root/'state.json',original)
        response_path=self.root/original['calls'][0]['file']; response_bytes=response_path.read_bytes()
        response_path.write_text('{}',encoding='utf-8')
        with self.assertRaises(ValidationError): migrate_run(self.root,identity,docs,resumed_config)
        response_path.write_bytes(response_bytes)
        unconsumed=copy.deepcopy(original)
        orphan=copy.deepcopy(original['calls'][0]); orphan.update(key='unconsumed_paid_response',index=len(original['calls']))
        unconsumed['calls'].append(orphan); atomic_json(self.root/'state.json',unconsumed)
        with self.assertRaisesRegex(ValidationError,'not consumed by the migration adapter'):
            migrate_run(self.root,identity,docs,resumed_config)
        atomic_json(self.root/'state.json',original)
        with patch('socket.socket',side_effect=AssertionError('offline only')):
            upgraded=migrate_run(self.root,identity,docs,resumed_config)
        self.assertEqual(upgraded['mode'],'persistent_tasks')
        state=json.loads((self.root/'state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['calls'],original['calls'])
        self.assertEqual(state['workflow']['revision'],1)
        self.assertEqual(state['workflow']['queue'][0]['units'][0]['offset'],400)
        self.assertEqual(migrate_run(self.root,identity,docs,resumed_config)['status'],'already_current')
        from test_workflow import WorkflowTransport
        transport=WorkflowTransport(draft)
        result=execute(docs,resumed_config,keys,self.root,transport,resume=True,extract_only=True,provenance=PROVENANCE)
        self.assertEqual(transport.calls,2)
        self.assertEqual(result['extraction']['applied_segments'],2)
        self.assertEqual(len(result['inventory']['assets']),1)

    def legacy(self):
        config = Config(); docs = documents(); identity = run_identity(docs, config, PROVENANCE)
        old = copy.deepcopy(identity); old['code'] = LEGACY_CODE
        store = Store(self.root, old, config)
        call = store.reserve('uncertain', 'model', 123)
        package = {'identity': old, 'status':'paused', 'inventory': inventory(),
                   'extraction': {'applied_segments':0, 'completed_segments':1, 'revision':0,
                       'total_segments':1,'merge_calls':0,'updates':[],'phase':'merging'}}
        for name, value in [('asset_package.json', package), ('documents.json', docs)]:
            atomic_json(self.root/name, value)
            store.state.setdefault('outputs', {})[name] = file_hash(self.root/name)
        store.save()
        return identity, docs, config, copy.deepcopy(store.state)

    def test_upgrade_keeps_outputs_calls_and_budget_then_is_idempotent(self):
        identity, docs, config, original = self.legacy()
        with patch('socket.socket', side_effect=AssertionError('offline only')):
            result = upgrade_run(self.root, identity, docs, config)
        current = json.loads((self.root/'state.json').read_text(encoding='utf-8'))
        self.assertEqual(result['network_calls'], 0)
        self.assertEqual(current['calls'], original['calls'])
        self.assertEqual(current['outputs'], original['outputs'])
        self.assertEqual(current['upgrades'][0]['from_identity'], original['identity'])
        self.assertEqual(upgrade_run(self.root, identity, docs, config)['status'], 'already_current')

    def test_upgrade_rejects_changed_input_output_and_unknown_code(self):
        identity, docs, config, original = self.legacy()
        changed = copy.deepcopy(identity); changed['parsed_sources'] = 'changed'
        with self.assertRaises(ValidationError): upgrade_run(self.root, changed, docs, config)
        tampered = copy.deepcopy(original); tampered['identity']['code'] = 'unknown'
        atomic_json(self.root/'state.json', tampered)
        with self.assertRaises(ValidationError): upgrade_run(self.root, identity, docs, config)
        atomic_json(self.root/'state.json', original)
        (self.root/'asset_package.json').write_text('{}', encoding='utf-8')
        with self.assertRaises(ValidationError): upgrade_run(self.root, identity, docs, config)

    def test_checkpoint_hash_checked_before_any_requests(self):
        identity, docs, config, _ = self.legacy()
        upgrade_run(self.root, identity, docs, config)
        path = self.root/'state.json'; state = json.loads(path.read_text(encoding='utf-8'))
        state['checkpoint']['inventory']['issues'].append('tampered')
        atomic_json(path, state)
        from asset_prep.errors import StaleEvidence
        with self.assertRaises(StaleEvidence):
            execute(docs, config, {}, self.root, FakeTransport([]), resume=True, provenance=PROVENANCE)


class DiagnosticTests(TempCase):
    def test_timeout_is_classified_without_saving_exception_secrets(self):
        connection = Mock(); connection.getresponse.side_effect = TimeoutError('secret-do-not-log')
        with patch('asset_prep.network.public_target', return_value=(__import__('urllib.parse', fromlist=['urlsplit']).urlsplit('https://example.org/api'),'8.8.8.8')), patch('asset_prep.network.PinnedHTTPS', return_value=connection):
            with self.assertRaises(ProviderError) as caught:
                Transport().post('https://example.org/api', {}, 'secret-key', 600)
        error = caught.exception
        self.assertEqual(error.diagnostics['failure'], 'timeout')
        self.assertEqual(error.diagnostics['phase'], 'waiting_headers')
        self.assertNotIn('secret', str(error))
        store = Store(self.root, {}, Config()); call = store.reserve('test', 'model')
        store.record_error(call, error)
        self.assertEqual(call['status'], 'pending')
        self.assertEqual(call['diagnostics'], error.diagnostics)
        self.assertNotIn('secret', (self.root/'state.json').read_text(encoding='utf-8'))

    def test_invalid_json_diagnostic_contains_only_size(self):
        transport = Transport()
        with patch.object(transport, 'request', return_value=(b'secret-invalid-json','text/plain')):
            with self.assertRaises(ProviderError) as caught:
                transport.post('https://example.org/api', {}, 'fake', 600)
        self.assertEqual(caught.exception.diagnostics['failure'], 'invalid_json')
        self.assertNotIn('secret', str(caught.exception.diagnostics))

    def test_http_error_is_distinct_from_local_timeout(self):
        connection = Mock(); response = connection.getresponse.return_value; response.status = 504
        with patch('asset_prep.network.public_target', return_value=(__import__('urllib.parse', fromlist=['urlsplit']).urlsplit('https://example.org/api'),'8.8.8.8')), patch('asset_prep.network.PinnedHTTPS', return_value=connection):
            with self.assertRaises(ProviderError) as caught:
                Transport().post('https://example.org/api', {}, 'fake', 600)
        self.assertEqual(caught.exception.diagnostics['failure'], 'http_status')
        self.assertEqual(caught.exception.diagnostics['http_status'], 504)
        self.assertTrue(caught.exception.uncertain)
