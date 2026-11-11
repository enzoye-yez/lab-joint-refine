"""长输出、局部引用修复、预算计划和接口参数的离线工程测试。"""

import copy
import json
from dataclasses import replace

from asset_prep.config import Config
from asset_prep.contracts import validate_inventory
from asset_prep.errors import BudgetExceeded, EvidenceRepairError, OutputTooLarge, ProviderLimitError, StaleEvidence, ValidationError
from asset_prep.extraction import EXTRACT, all_evidence, extract_inventory, extraction_data
from asset_prep.limits import output_audit
from asset_prep.pipeline import execute
from asset_prep.probe import probe_limits
from asset_prep.providers import Model
from asset_prep.storage import Store
from fixtures import documents, inventory, selection, FakeTransport, delta_decisions, PROVENANCE
from test_boundaries import TempCase


class RepairTests(TempCase):
    def test_numeric_requirement_format_is_local_audited_and_cacheable(self):
        from asset_prep.workflow import LOCAL, local_data, plan_tasks
        docs=documents(); value={'inventory':inventory(),'consumed_units':2}
        value['inventory']['assets'][0]['requirements'][0]['value']=0.2
        original=copy.deepcopy(value)
        model,transport=self.model([value]); data=local_data(plan_tasks(docs,Config())[0],docs,Config())
        def validate(result):
            validate_inventory(result['inventory'],docs['sources'])
            return result
        first=model.structured(LOCAL,data,validate)
        second=model.structured(LOCAL,data,validate)
        self.assertEqual(transport.model_calls,1)
        self.assertEqual(first,second)
        self.assertEqual(first['inventory']['assets'][0]['requirements'][0]['value'],'0.2')
        self.assertEqual(value,original)
        call=model.store.state['calls'][0]
        raw=json.loads((self.root/call['file']).read_text(encoding='utf-8'))
        self.assertIsInstance(json.loads(raw['choices'][0]['message']['content'])['inventory']['assets'][0]['requirements'][0]['value'],float)
        self.assertTrue(any(e['kind']=='numeric_requirement_format' for e in call['validation_events']))

    def test_requirement_normalizer_does_not_guess_non_numeric_values(self):
        from asset_prep.providers import normalize_requirement_numbers
        for invalid in (True,None,{},[]):
            value={'inventory':inventory()}; value['inventory']['assets'][0]['requirements'][0]['value']=invalid
            normalized,paths=normalize_requirement_numbers(value)
            self.assertEqual(normalized,value); self.assertEqual(paths,[])

    def model(self, replies, config=None):
        config = config or Config()
        store = Store(self.root, {}, config)
        transport = FakeTransport(replies)
        return Model(config, {'HARNESS_MODEL_API_KEY':'fixture'}, store, transport), transport

    def extract(self, model, docs=None):
        docs = docs or documents()
        return model.structured(EXTRACT, extraction_data(0, docs['sources'], [], model.config),
            lambda v: validate_inventory(v, docs['sources']))

    def test_unique_whitespace_repair_uses_no_extra_model_request(self):
        docs = documents(); docs['sources'][0]['text'] = docs['sources'][0]['text'].replace('two 0.2', 'two\n0.2')
        model, transport = self.model([inventory()])
        result = self.extract(model, docs)
        self.assertEqual(transport.model_calls, 1)
        self.assertIn('\n', result['assets'][0]['evidence'][0]['quote'])
        events = model.store.state['calls'][0]['validation_events']
        self.assertTrue(any(e['kind']=='exact_whitespace_repair' for e in events))

    def test_local_reference_patch_preserves_every_non_reference_field(self):
        value = inventory(); value['assets'][0]['evidence'][0]['quote'] = 'made up words'
        patch = {'patches':[{'id':'e0', **inventory()['assets'][0]['evidence'][0]}]}
        model, transport = self.model([value, patch])
        result = self.extract(model)
        self.assertEqual(result, inventory())
        self.assertEqual(transport.model_calls, 2)
        data = json.loads(transport.requests[1][1]['messages'][1]['content'].split('\nTask data follows:\n',1)[1])
        self.assertIn('evidence_errors', data)
        self.assertNotIn('assets', data)
        self.assertEqual(data['evidence_errors'][0]['paths'], [['assets',0,'evidence',0]])
        replay = Model(model.config, {'HARNESS_MODEL_API_KEY':'fixture'}, model.store, FakeTransport([]))
        self.assertEqual(self.extract(replay), result)

    def test_patch_cannot_edit_quantity_or_unknown_paths(self):
        value = inventory(); value['assets'][0]['evidence'][0]['quote']='invalid'
        bad = {'patches':[{'id':'e0', **inventory()['assets'][0]['evidence'][0], 'quantity':999}]}
        model, transport = self.model([value,bad,bad,bad])
        with self.assertRaises(EvidenceRepairError): self.extract(model)
        self.assertEqual(transport.model_calls,4)
        self.assertTrue(all('evidence_errors' in p['messages'][1]['content'] for _,p in transport.requests[1:]))

    def test_digit_or_unit_changes_are_not_auto_normalized(self):
        value = inventory(); value['assets'][0]['evidence'][0]['quote']='0.2 ML'
        model, transport = self.model([value], replace(Config(), repair_attempts=0))
        with self.assertRaises(EvidenceRepairError): self.extract(model)
        self.assertEqual(transport.model_calls,1)

    def test_ambiguous_whitespace_match_is_not_guessed(self):
        docs = documents(); docs['sources'][0]['text'] = 'two\n tubes and two\n tubes'
        value = inventory(); value['conflicts'] = []
        for ref in all_evidence(value): ref.update(source_id='pdf:p1',quote='two tubes')
        model, transport = self.model([value], replace(Config(), repair_attempts=0))
        with self.assertRaises(EvidenceRepairError): self.extract(model, docs)
        self.assertEqual(transport.model_calls,1)

    def test_oversize_and_split_signal_do_not_trigger_full_repair(self):
        model, transport = self.model([{'needs_split':True}])
        with self.assertRaises(OutputTooLarge): self.extract(model)
        self.assertEqual(transport.model_calls,1)


class AdaptiveTests(TempCase):
    def test_split_path_replays_from_cache_without_duplicate_paid_calls(self):
        source={'id':'pdf:p1','kind':'pdf','text':'tube '*160}
        docs={'sources':[source],'warnings':[],'files':{'pdf':{'name':'fixture.pdf','sha256':'fixture'}}}
        class Dynamic(FakeTransport):
            def post(self,url,payload,key,timeout):
                data=json.loads(payload['messages'][1]['content'].split('\nTask data follows:\n',1)[1])
                if data.get('operation')=='local_extract' and sum(u['length'] for u in data['units'])>400:
                    self.model_calls+=1
                    return {'model':'fixture','usage':{'total_tokens':10},'choices':[{'finish_reason':'length','message':{'content':'{"assets":['}}]}
                if data.get('operation')=='local_extract':
                    value=inventory(); value['conflicts']=[]
                    for ref in all_evidence(value): ref.update(source_id='pdf:p1',quote='tube ')
                else:
                    value={'target':'tubes','reason':'same synthetic object'}
                self.replies.append(value)
                return super().post(url,payload,key,timeout)
        config=Config(); keys={'HARNESS_MODEL_API_KEY':'fixture'}; first=Dynamic([])
        package=execute(docs,config,keys,self.root,first,extract_only=True,provenance=PROVENANCE)
        self.assertEqual(first.model_calls,4)
        self.assertEqual(package['extraction']['applied_segments'],2)
        second=FakeTransport([])
        resumed=execute(docs,config,keys,self.root,second,resume=True,extract_only=True,provenance=PROVENANCE)
        self.assertEqual(second.model_calls,0)
        self.assertEqual(resumed['inventory'],package['inventory'])

    def test_recursive_split_preserves_all_text_and_offsets(self):
        source = {'id':'pdf:p1','kind':'pdf','text':'tube ' * 160}
        seen=[]
        class SmallModel:
            def structured(self, instruction, data, validate):
                if 'incoming_assets' in data:
                    return validate(delta_decisions(data['incoming_assets']))
                part = data['sources'][0]
                if len(part['text']) > 300:
                    raise OutputTooLarge('synthetic output pressure')
                seen.append(copy.deepcopy(part))
                draft=inventory(); draft['conflicts']=[]
                for ref in all_evidence(draft): ref.update(source_id=part['id'],quote=part['text'][:5])
                return validate(draft)
        progress={}
        result=extract_inventory({'sources':[source]},Config(),SmallModel(),progress)
        self.assertEqual(''.join(p['text'] for p in seen),source['text'])
        self.assertEqual([p['offset'] for p in seen],[0,200,400,600])
        self.assertEqual(progress['applied_segments'],4)
        self.assertEqual(len(result['assets']),1)

    def test_minimum_segment_stops_without_unbounded_calls(self):
        class TooLarge:
            calls=0
            def structured(self,*args):
                self.calls+=1
                raise OutputTooLarge('too much')
        model=TooLarge()
        with self.assertRaisesRegex(ValidationError,'smallest'):
            extract_inventory({'sources':[{'id':'pdf:p1','kind':'pdf','text':'x'*100}]},Config(),model)
        self.assertEqual(model.calls,1)


class OutputLimitTests(TempCase):
    def test_reasoning_details_do_not_hide_total_violation(self):
        response={'usage':{'completion_tokens':522,'completion_tokens_details':{'reasoning_tokens':297}}}
        audit=output_audit(response,64,'max_tokens')
        self.assertEqual(audit['reported_non_reasoning_tokens'],225)
        self.assertEqual(audit['status'],'reported_violation')
        for invalid in (True,-1,523,'297',None):
            response['usage']['completion_tokens_details']['reasoning_tokens']=invalid
            self.assertIsNone(output_audit(response,64,'max_tokens')['reported_non_reasoning_tokens'])

    def test_selected_probe_only_sends_selected_parameter_and_replays(self):
        class Probe:
            requests=[]
            def post(self,url,payload,key,timeout):
                self.requests.append(payload)
                return {'choices':[{'finish_reason':'length','message':{'content':'lab'}}],
                        'usage':{'completion_tokens':payload['max_completion_tokens'],'total_tokens':300}}
        transport=Probe(); config=Config(); keys={'HARNESS_MODEL_API_KEY':'fixture'}
        report=probe_limits(config,keys,self.root,transport,parameter='max_completion_tokens')
        self.assertEqual([p['max_completion_tokens'] for p in transport.requests],[64,256])
        self.assertTrue(all('max_tokens' not in p for p in transport.requests))
        self.assertEqual(report['parameters'],['max_completion_tokens'])
        probe_limits(config,keys,self.root,transport,parameter='max_completion_tokens',resume=True)
        self.assertEqual(len(transport.requests),2)
        with self.assertRaises(StaleEvidence):
            probe_limits(config,keys,self.root,transport,parameter='max_tokens',resume=True)

    def test_probe_parameter_requires_probe_mode(self):
        from asset_prep.cli import main
        from contextlib import redirect_stderr
        from io import StringIO
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main(['--probe-parameter','max_completion_tokens','--run','fixture'])
        self.assertEqual(error.exception.code,2)

    def test_invalid_content_cannot_mask_probe_violation(self):
        class Probe:
            calls=0
            def post(self,*args):
                self.calls+=1
                return {'choices':[], 'usage':{'completion_tokens':999}}
        transport=Probe()
        report=probe_limits(Config(),{'HARNESS_MODEL_API_KEY':'fixture'},self.root,transport,
                            parameter='max_completion_tokens')
        self.assertEqual(report['status'],'violation_detected')
        self.assertEqual(transport.calls,1)
        self.assertFalse(report['results'][0]['content_valid'])

    def test_local_character_cap_triggers_split_without_trusting_token_usage(self):
        config=replace(Config(), extraction_max_chars=100)
        transport=FakeTransport([inventory()]); model=Model(config,{'HARNESS_MODEL_API_KEY':'fixture'},Store(self.root,{},config),transport)
        with self.assertRaises(OutputTooLarge):
            model.structured(EXTRACT,extraction_data(0,documents()['sources'],[],config),lambda v:v)
        self.assertEqual(transport.model_calls,1)

    def test_truncated_response_triggers_split_before_json_repair(self):
        class Truncated(FakeTransport):
            def post(self,*args):
                result=super().post(*args); result['choices'][0].update(finish_reason='length')
                result['choices'][0]['message']['content']='{"assets":['
                return result
        config=Config(); transport=Truncated([inventory()]); model=Model(config,{'HARNESS_MODEL_API_KEY':'fixture'},Store(self.root,{},config),transport)
        with self.assertRaises(OutputTooLarge):
            model.structured(EXTRACT,extraction_data(0,documents()['sources'],[],config),lambda v:v)
        self.assertEqual(transport.model_calls,1)

    def test_truncated_merge_triggers_smaller_batch_before_json_repair(self):
        class Truncated(FakeTransport):
            def post(self,*args):
                result=super().post(*args)
                result['choices'][0].update(finish_reason='length')
                result['choices'][0]['message']['content']='{"decisions":['
                return result
        config=Config(); transport=Truncated([inventory()])
        model=Model(config,{'HARNESS_MODEL_API_KEY':'fixture'},Store(self.root,{},config),transport)
        with self.assertRaises(OutputTooLarge):
            model.structured('synthetic merge',{'incoming_assets':[]},lambda v:v)
        self.assertEqual(transport.model_calls,1)

    def test_reported_violation_is_saved_and_never_repaired_or_retried(self):
        class Exceeded(FakeTransport):
            def post(self,*args):
                response=super().post(*args)
                response['usage'].update(completion_tokens=6001,total_tokens=6100)
                return response
        config=Config(); transport=Exceeded([inventory()]); store=Store(self.root,{},config)
        model=Model(config,{'HARNESS_MODEL_API_KEY':'fixture'},store,transport)
        for _ in range(2):
            with self.assertRaises(ProviderLimitError):
                model.structured(EXTRACT,extraction_data(0,documents()['sources'],[],config),lambda v:v)
        self.assertEqual(transport.model_calls,1)
        self.assertEqual(store.state['calls'][0]['status'],'done')
        self.assertEqual(store.state['calls'][0]['validation_events'][0]['status'],'reported_violation')

    def test_missing_usage_is_not_treated_as_verified(self):
        self.assertEqual(output_audit({},64,'max_tokens')['status'],'unverifiable')
        self.assertEqual(output_audit({'usage':{'completion_tokens':True}},64,'max_tokens')['status'],'unverifiable')

    def test_configured_parameter_is_sent_without_silent_fallback(self):
        from pathlib import Path
        default=Config.load(Path(__file__).resolve().parents[1]/'config/default.json')
        self.assertEqual(default.output_token_parameter,'max_completion_tokens')
        self.assertEqual(Config().output_token_parameter,default.output_token_parameter)
        for config in (default,replace(default,output_token_parameter='max_tokens')):
            with self.subTest(parameter=config.output_token_parameter):
                transport=FakeTransport([inventory()])
                model=Model(config,{config.api_key_env:'fixture'},
                            Store(self.root/config.output_token_parameter,{},config),transport)
                model.structured(EXTRACT,extraction_data(0,documents()['sources'],[],config),lambda v:v)
                payload=transport.requests[0][1]
                self.assertEqual(payload[config.output_token_parameter],6000)
                other='max_tokens' if config.output_token_parameter=='max_completion_tokens' else 'max_completion_tokens'
                self.assertNotIn(other,payload)

    def test_probe_reports_both_parameters_and_reuses_saved_responses(self):
        class Probe:
            calls=0
            def post(self,url,payload,key,timeout):
                self.calls+=1
                limit=payload.get('max_tokens',payload.get('max_completion_tokens'))
                return {'model':'fixture','choices':[{'finish_reason':'length','message':{'content':'lab'}}],
                        'usage':{'completion_tokens':limit,'total_tokens':limit+20}}
        transport=Probe(); config=Config(); keys={'HARNESS_MODEL_API_KEY':'fixture'}
        report=probe_limits(config,keys,self.root,transport)
        self.assertEqual(transport.calls,4)
        self.assertEqual({r['parameter'] for r in report['results']},{'max_tokens','max_completion_tokens'})
        probe_limits(config,keys,self.root,transport,resume=True)
        self.assertEqual(transport.calls,4)

    def test_probe_stops_after_reported_violation(self):
        class BadProbe:
            calls=0
            def post(self,*args):
                self.calls+=1
                return {'choices':[{'finish_reason':'stop','message':{'content':'lab'}}],
                        'usage':{'completion_tokens':999,'total_tokens':1000}}
        transport=BadProbe()
        report=probe_limits(Config(),{'HARNESS_MODEL_API_KEY':'fixture'},self.root,transport)
        self.assertEqual(transport.calls,1)
        self.assertEqual(report['status'],'violation_detected')


class PhaseBudgetTests(TempCase):
    def test_insufficient_budget_stops_before_any_search(self):
        config=replace(Config(),search_provider='tavily',max_model_calls=1)
        transport=FakeTransport(); keys={'HARNESS_MODEL_API_KEY':'fixture','TAVILY_API_KEY':'fixture'}
        with self.assertRaises(BudgetExceeded): execute(documents(),config,keys,self.root,transport,provenance=PROVENANCE)
        self.assertEqual(transport.search_calls,0)
        package=json.loads((self.root/'asset_package.json').read_text(encoding='utf-8'))
        self.assertEqual(package['reference_budget_plan']['planned_selection_calls'],1)
        self.assertEqual(package['inventory']['assets'][0]['id'],'tubes')

    def test_extract_only_then_continue_reuses_extraction(self):
        config=replace(Config(),search_provider='tavily'); keys={'HARNESS_MODEL_API_KEY':'fixture','TAVILY_API_KEY':'fixture'}
        first=FakeTransport([inventory()])
        package=execute(documents(),config,keys,self.root,first,extract_only=True,provenance=PROVENANCE)
        self.assertEqual(package['status'],'extraction_completed')
        self.assertEqual(first.search_calls,0)
        resumed=FakeTransport([selection()])
        execute(documents(),config,keys,self.root,resumed,resume=True,provenance=PROVENANCE)
        self.assertEqual(resumed.model_calls,1)
