"""期刊 XML 与长文档工程测试；均用合成材料，不调用模型服务。"""

import copy
import json
import logging
from contextlib import redirect_stdout
from io import StringIO
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from defusedxml import ElementTree

from asset_prep.config import Config
from asset_prep.cli import main
from asset_prep.documents import load_documents, xml_sources
from asset_prep.errors import BudgetExceeded, StaleEvidence, ValidationError
from asset_prep.extraction import (EXTRACT, MERGE, all_evidence, extract_inventory, merge_data,
                                   plan_segments, planning_limit, request_size, validate_merge, validate_unique_assets)
from asset_prep.pipeline import execute
from asset_prep.preflight import inspect_inputs
from fixtures import PROVENANCE, FakeTransport, documents, inventory, selection, write_pdf, delta_decisions
from test_boundaries import TempCase


def envelope(drafts, merged=None):
    merged = copy.deepcopy(merged or drafts[0])
    refs = {(r['source_id'], r['quote']) for r in all_evidence(merged['assets'][0])}
    for draft in drafts:
        for ref in all_evidence(draft['assets']):
            pair = ref['source_id'], ref['quote']
            if pair not in refs:
                merged['assets'][0]['evidence'].append(copy.deepcopy(ref))
                refs.add(pair)
    return {'inventory': merged, 'coverage': [
        {'draft_index': i, 'asset_id': a['id'], 'target_id': merged['assets'][0]['id'], 'reason': '合成同一资产映射'}
        for i, draft in enumerate(drafts) for a in draft['assets']]}


class InputTests(TempCase):
    def pair(self, xml):
        pdf, markup = self.root / 'source.pdf', self.root / 'source.xml'
        write_pdf(pdf)
        markup.write_text(xml, encoding='utf-8')
        return pdf, markup

    def test_journal_doctype_accepted_without_external_fetch(self):
        xml = '<!DOCTYPE article PUBLIC "journal" "https://127.0.0.1/never.dtd"><article><body><p>Use a tube.</p></body></article>'
        with patch('socket.socket', side_effect=AssertionError('must not connect')):
            result = load_documents(*self.pair(xml), 10000, xml_reason='核对容器', xml_queries=['tube'])
        self.assertEqual(result['parsing']['xml_root'], 'article')
        self.assertTrue(any(s['text'] == 'Use a tube.' for s in result['sources']))

    def test_internal_and_parameter_entities_still_rejected(self):
        for entity in ('<!ENTITY value "text">', '<!ENTITY % value SYSTEM "file:///private">%value;'):
            with self.subTest(entity=entity), self.assertRaisesRegex(ValidationError, 'entit'):
                load_documents(*self.pair('<!DOCTYPE article [' + entity + ']><article/>'), 10000,
                               xml_reason='补证测试', xml_queries=['tube'])

    def test_mixed_inline_text_order_and_units_preserved(self):
        root = ElementTree.fromstring('<article><body><sec><title>Materials</title><p>Use <italic>two</italic> 0.2 <bold>mL</bold> tubes at 10<sup>3</sup> rpm.</p>After.</sec></body></article>')
        sources = xml_sources(root)
        self.assertEqual([s['text'] for s in sources], ['Materials', 'Use two 0.2 mL tubes at 10^(3) rpm.', 'After.'])
        self.assertEqual(sources[1]['section'], ['Materials'])
        self.assertEqual(sources[1]['role'], 'article_body')
        self.assertIn('/p[1]', sources[1]['path'])

    def test_namespaces_tables_captions_and_references_not_lost(self):
        root = ElementTree.fromstring('<article xmlns="urn:jats"><body><sec><title>Equipment</title><table-wrap><caption><p>Required vessels</p></caption><table><tr><th>Tube</th><th>Capacity</th></tr><tr><td>X</td><td>0.2 mL</td></tr></table><table-wrap-foot><p>Use caps.</p></table-wrap-foot></table-wrap></sec></body><back><ref-list><ref><mixed-citation>Other study.</mixed-citation></ref></ref-list></back></article>')
        sources = xml_sources(root)
        self.assertEqual([s['text'] for s in sources], ['Equipment', 'Required vessels', 'Tube\tCapacity\nX\t0.2 mL', 'Use caps.', 'Other study.'])
        self.assertEqual(sources[-1]['role'], 'references')

    def test_generic_xml_not_restricted_to_journal_format(self):
        root = ElementTree.fromstring('<protocol>Use <vessel>two <type>glass</type> tubes</vessel> now.</protocol>')
        self.assertEqual(xml_sources(root)[0]['text'], 'Use two glass tubes now.')

    def test_nested_paragraph_table_and_list_do_not_join_numbers(self):
        root = ElementTree.fromstring('<p>Items:<list><list-item><p>A</p></list-item><list-item><p>B</p></list-item></list><table><tr><td>10</td><td>20</td></tr></table>End.</p>')
        value = xml_sources(root)[0]['text']
        self.assertIn('A\n', value)
        self.assertIn('B\n', value)
        self.assertIn('10\t20\t', value)
        self.assertNotIn('1020', value)
        self.assertTrue(value.endswith('End.'))

    def test_mathml_superscript_fraction_and_warning(self):
        xml = '<article xmlns:m="urn:math"><body><p>Speed <m:math><m:mfrac><m:msup><m:mn>10</m:mn><m:mn>3</m:mn></m:msup><m:mi>x</m:mi></m:mfrac></m:math>.</p></body></article>'
        result = load_documents(*self.pair(xml), 10000, xml_reason='核对公式', xml_queries=['Speed'])
        text = next(s['text'] for s in result['sources'] if s['kind'] == 'xml')
        self.assertEqual(text, 'Speed ((10)^(3))/(x).')
        self.assertTrue(any('math formulas' in w for w in result['warnings']))

    def test_subscript_preserved_and_excessive_inline_nesting_rejected(self):
        root = ElementTree.fromstring('<p>CO<sub>2</sub> at 10<sup>3</sup>.</p>')
        self.assertEqual(xml_sources(root)[0]['text'], 'CO_(2) at 10^(3).')
        root = ElementTree.fromstring('<p>' + '<italic>' * 140 + 'text' + '</italic>' * 140 + '</p>')
        with self.assertRaisesRegex(ValidationError, 'nesting'):
            xml_sources(root)

    def test_pdf_warnings_are_recorded_and_logger_restored(self):
        from pypdf import PdfReader
        logger = logging.getLogger('pypdf')
        handlers = logger.handlers[:]
        def warn(stream):
            logger.warning('synthetic font warning')
            return PdfReader(stream)
        with patch('asset_prep.documents.PdfReader', side_effect=warn):
            result = load_documents(*self.pair('<protocol>two tubes</protocol>'), 10000)
        self.assertTrue(any('1 warnings' in w for w in result['warnings']))
        self.assertEqual(logger.handlers, handlers)

    def test_parser_versions_and_unconfirmed_pair_are_visible(self):
        result = load_documents(*self.pair('<article><front><article-id pub-id-type="doi">10.1000/other</article-id></front><body><p>two tubes</p></body></article>'), 10000,
                                xml_reason='核对数量', xml_queries=['two'])
        self.assertIn('fonttools', result['parsing']['versions'])
        self.assertEqual(result['parsing']['pairing'], 'unconfirmed')
        self.assertTrue(any('correspondence' in w for w in result['warnings']))

    def test_preflight_does_not_use_credentials_or_transport(self):
        case = self.root / 'case1'
        case.mkdir()
        write_pdf(case / 'source.pdf')
        (case / 'source.xml').write_text('<!DOCTYPE article SYSTEM "unavailable.dtd"><article><p>tube</p></article>', encoding='utf-8')
        (self.root / 'case2').mkdir()
        with patch('asset_prep.config.credentials', side_effect=AssertionError('no keys')), patch('asset_prep.network.Transport.post', side_effect=AssertionError('no model')):
            report = inspect_inputs(Config(), dataset=self.root)
        self.assertEqual((report['total'], report['input_checked'], report['blocked']), (2, 1, 1))
        self.assertEqual(report['model_calls'], 0)

    def test_preflight_cli_skips_credentials_and_refuses_overwrite(self):
        pdf, xml = self.pair('<protocol>Use tubes.</protocol>')
        run_root = self.root / 'run'
        args = ['--preflight', '--pdf', str(pdf), '--xml', str(xml), '--run', 'fixture']
        with patch('asset_prep.cli.run_directory', return_value=run_root), patch('asset_prep.cli.credentials', side_effect=AssertionError('no keys')), patch('asset_prep.cli.execute', side_effect=AssertionError('no model')), redirect_stdout(StringIO()):
            self.assertEqual(main(args), 0)
            report = (run_root / 'preflight.json').read_bytes()
            self.assertEqual(main(args), 2)
            self.assertEqual((run_root / 'preflight.json').read_bytes(), report)


class ExtractionTests(TempCase):
    def test_multibyte_request_splitting_preserves_every_character(self):
        source = {'id': 'pdf:p1', 'kind': 'pdf', 'text': '测量容器容量 µL\n' * 3000}
        config = replace(Config(), max_input_bytes=22000, chunk_chars=100000)
        segments = plan_segments([source], config)
        pieces = [s for group in segments for s in group]
        self.assertEqual(''.join(s['text'] for s in pieces), source['text'])
        offset = 0
        for piece in pieces:
            self.assertEqual(piece['offset'], offset)
            offset += len(piece['text'])
        self.assertTrue(all(request_size(config, EXTRACT, {'segment':i, 'sources':s}) <= planning_limit(config)
                            for i, s in enumerate(segments)))

    def test_excerpts_are_exact_windows_not_synthesized_summary(self):
        draft = inventory()
        sources = documents()['sources']
        sources[0]['text'] = 'x' * 100000 + sources[0]['text'] + 'y' * 100000
        data = merge_data([draft], sources)
        self.assertLess(len(json.dumps(data)), 10000)
        for source in data['sources']:
            original = next(s['text'] for s in sources if s['id'] == source['id'])
            for excerpt in source['excerpts']:
                self.assertEqual(excerpt['text'], original[excerpt['offset']:excerpt['offset'] + len(excerpt['text'])])

    def test_merge_requires_every_input_asset_and_its_evidence(self):
        drafts = [inventory(), inventory()]
        valid = envelope(drafts)
        validate_merge(valid, drafts, documents()['sources'], Config())
        missing = copy.deepcopy(valid)
        missing['coverage'].pop()
        with self.assertRaisesRegex(ValidationError, 'missing'):
            validate_merge(missing, drafts, documents()['sources'], Config())
        duplicate = copy.deepcopy(valid)
        duplicate['coverage'].append(duplicate['coverage'][0])
        with self.assertRaises(ValidationError):
            validate_merge(duplicate, drafts, documents()['sources'], Config())
        drafts[0]['assets'][0]['evidence'].append({'source_id':'pdf:p1', 'quote':'0.2 mL'})
        with self.assertRaisesRegex(ValidationError, 'evidence'):
            validate_merge(valid, drafts, documents()['sources'], Config())

    def test_merge_preserves_unresolved_issues_even_if_model_omits(self):
        draft = inventory()
        draft['issues'] = ['unresolved fixture issue']
        result = envelope([draft], inventory())
        validate_merge(result, [draft], documents()['sources'], Config())
        self.assertIn('unresolved fixture issue', result['inventory']['issues'])

    def test_unbounded_draft_cannot_loop_or_silently_drop_assets(self):
        draft = inventory()
        draft['assets'][0]['aliases'] = ['x' * 20000]
        class Model:
            def structured(self, instruction, data, validate):
                return validate(copy.deepcopy(draft))
        with self.assertRaisesRegex(ValidationError, 'exceed the incremental request limit'):
            extract_inventory(documents(), replace(Config(), max_input_bytes=12000), Model())

    def test_catalog_and_draft_that_cannot_fit_stop_without_second_merge(self):
        draft = inventory()
        draft['assets'][0]['aliases'] = ['x' * 8000]
        sources = documents()['sources']
        config = Config()
        config = replace(config, max_input_bytes=19000)
        class Model:
            merges = 0
            def structured(self, instruction, data, validate):
                if 'incoming_assets' in data:
                    self.merges += 1
                    if self.merges > 1:
                        raise AssertionError('oversized pair must not call model')
                    return validate(delta_decisions(data['incoming_assets']))
                return validate(copy.deepcopy(draft))
        with patch('asset_prep.extraction.plan_segments', return_value=[sources, sources]):
            with self.assertRaisesRegex(ValidationError, 'exceed the incremental request limit'):
                extract_inventory(documents(), config, Model())

    def test_invalid_merge_mapping_cannot_reach_search(self):
        draft=inventory(); other=copy.deepcopy(draft['assets'][0]); other['id']='alias'
        draft['assets'].append(other)
        draft['steps'][0]['asset_ids'].append('alias')
        invalid={'target':'not_in_page','reason':'invalid fixture'}
        transport = FakeTransport([draft, invalid, invalid, invalid])
        with self.assertRaises(ValidationError):
            execute(documents(), replace(Config(), search_provider='tavily'),
                    {'HARNESS_MODEL_API_KEY':'fixture','TAVILY_API_KEY':'fixture'}, self.root,
                    transport, provenance=PROVENANCE)
        self.assertEqual(transport.search_calls, 0)
        self.assertEqual(transport.model_calls, 4)

    def test_incremental_merge_keeps_coverage_and_bounded_requests(self):
        sources = [{'id': f'pdf:p{i}', 'kind':'pdf', 'text':('tube ' + str(i)).ljust(1500, '.')} for i in range(8)]
        config = replace(Config(), chunk_chars=1500, max_input_bytes=19000)
        sizes, merges = [], []
        class Model:
            def structured(self, instruction, data, validate):
                sizes.append(request_size(config, instruction, data))
                if 'segment' in data:
                    draft = inventory()
                    ref = {'source_id':data['sources'][0]['id'], 'quote':data['sources'][0]['text'][:6]}
                    # 单一对象在多个片段重复出现；大段别名模拟冗长提取输出。
                    draft['conflicts'] = []
                    draft['assets'][0]['aliases'] = ['fixture-' + 'x' * 3000]
                    for evidence in all_evidence(draft):
                        evidence.update(ref)
                    return validate(draft)
                merges.append(len(data['incoming_assets']))
                return validate(delta_decisions(data['incoming_assets']))
        progress = {}
        result = extract_inventory({'sources':sources}, config, Model(), progress)
        self.assertGreater(progress['merge_calls'], 1)
        self.assertTrue(all(size <= planning_limit(config) for size in sizes))
        self.assertEqual({r['source_id'] for r in all_evidence(result['assets'])}, {s['id'] for s in sources})
        self.assertEqual(progress['completed_segments'], 8)
        self.assertEqual(progress['revision'], 8)
        self.assertEqual(len(result['assets']), 1)

    def test_duplicate_names_and_aliases_blocked_but_different_specs_allowed(self):
        value = inventory()
        other = copy.deepcopy(value['assets'][0])
        other['id'] = 'duplicate'
        other['name'] = '反 应 管'
        value['assets'].append(other)
        with self.assertRaisesRegex(ValidationError, 'duplicate'):
            validate_unique_assets(value)
        other['name'] = '另一称谓'
        other['aliases'] = ['反应管']
        with self.assertRaises(ValidationError):
            validate_unique_assets(value)
        other['requirements'][0]['value'] = '0.5'
        validate_unique_assets(value)

    def test_existing_asset_id_cannot_be_renumbered(self):
        original = inventory()
        renamed = inventory()
        renamed['assets'][0]['id'] = 'renamed'
        renamed['steps'][0]['asset_ids'] = ['renamed']
        renamed['conflicts'][0]['asset_id'] = 'renamed'
        value = envelope([original], renamed)
        with self.assertRaisesRegex(ValidationError, 'renumbered'):
            validate_merge(value, [original], documents()['sources'], Config(), existing_catalog=True)

    def test_dedup_does_not_erase_decimal_or_unit_case(self):
        for left, right in ((('1.0', 'mL'), ('10', 'mL')), (('1', 'm'), ('1', 'M'))):
            with self.subTest(left=left, right=right):
                value = inventory()
                other = copy.deepcopy(value['assets'][0])
                other['id'] = 'different_spec'
                for asset, spec in ((value['assets'][0], left), (other, right)):
                    asset['requirements'][0].update(value=spec[0], unit=spec[1])
                value['assets'].append(other)
                validate_unique_assets(value)

    def test_single_master_catalog_survives_pause_and_deduplicates_repeated_mentions(self):
        draft = inventory()
        draft['assets'][0]['id'] = 'new_tubes'
        draft['steps'][0]['asset_ids'] = ['new_tubes']
        draft['conflicts'][0]['asset_id'] = 'new_tubes'
        replies = [inventory(), draft]
        keys = {'HARNESS_MODEL_API_KEY':'fixture', 'TAVILY_API_KEY':'fixture'}
        config = replace(Config(), search_provider='tavily', max_model_calls=2)
        transport = FakeTransport(replies)
        from asset_prep.workflow import plan_tasks
        tasks=plan_tasks(documents(),config)*2
        with patch('asset_prep.workflow.plan_tasks', return_value=tasks):
            with self.assertRaises(BudgetExceeded):
                execute(documents(), config, keys, self.root, transport, provenance=PROVENANCE)
        partial = json.loads((self.root / 'asset_package.json').read_text(encoding='utf-8'))
        self.assertEqual(partial['extraction']['revision'], 1)
        self.assertEqual([a['id'] for a in partial['inventory']['assets']], ['tubes'])
        second_extract = json.loads(transport.requests[1][1]['messages'][1]['content'].split('\nTask data follows:\n', 1)[1])
        self.assertNotIn('existing_assets',second_extract)
        replay = FakeTransport([{'target':'tubes','reason':'same synthetic object'}, selection()])
        snapshots = []
        from asset_prep.storage import atomic_json
        def capture(path, value):
            if path.name == 'asset_package.json':
                snapshots.append(copy.deepcopy(value))
            atomic_json(path, value)
        with patch('asset_prep.pipeline.atomic_json', side_effect=capture):
            package = execute(documents(), replace(config, max_model_calls=40), keys, self.root, replay,
                              resume=True, provenance=PROVENANCE)
        self.assertEqual(replay.model_calls, 2)
        self.assertTrue(all(s['extraction']['revision'] >= 1 for s in snapshots))
        self.assertTrue(all(len(s['inventory']['assets']) == 1 for s in snapshots))
        self.assertEqual(package['extraction']['revision'], 2)
        self.assertEqual(package['extraction']['updates'][1]['added_ids'], [])
        self.assertEqual(len(package['assets']), 1)
        self.assertEqual(package['inventory']['assets'][0]['quantity']['value'], 2)

    def test_completed_resume_does_not_duplicate_images_or_assets(self):
        config = replace(Config(), search_provider='tavily')
        keys = {'HARNESS_MODEL_API_KEY':'fixture', 'TAVILY_API_KEY':'fixture'}
        first = execute(documents(), config, keys, self.root, FakeTransport(), provenance=PROVENANCE)
        second = execute(documents(), config, keys, self.root, FakeTransport([]), provenance=PROVENANCE, resume=True)
        self.assertEqual(second['assets'], first['assets'])

    def test_merge_budget_pause_resumes_cached_extraction(self):
        config = replace(Config(), search_provider='tavily', max_model_calls=1)
        keys = {'HARNESS_MODEL_API_KEY':'fixture', 'TAVILY_API_KEY':'fixture'}
        with self.assertRaises(BudgetExceeded):
            execute(documents(), config, keys, self.root, FakeTransport(), provenance=PROVENANCE)
        replay = FakeTransport([inventory(), selection()])
        package = execute(documents(), replace(config, max_model_calls=40), keys, self.root, replay,
                          provenance=PROVENANCE, resume=True)
        self.assertEqual(replay.model_calls, 2)
        self.assertEqual(package['extraction']['completed_segments'], 1)

    def test_changed_parsed_text_invalidates_resume_even_with_same_file_hash(self):
        keys = {'HARNESS_MODEL_API_KEY':'fixture', 'TAVILY_API_KEY':'fixture'}
        config = replace(Config(), search_provider='tavily')
        execute(documents(), config, keys, self.root, FakeTransport(), provenance=PROVENANCE)
        changed = documents()
        changed['sources'][0]['text'] += 'changed decoding'
        with self.assertRaises(StaleEvidence):
            execute(changed, config, keys, self.root, FakeTransport(), resume=True, provenance=PROVENANCE)
