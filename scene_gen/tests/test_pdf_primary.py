"""PDF 优先与定向补证边界；合成材料，不调用模型。"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from asset_prep.cli import main
from asset_prep.config import Config
from asset_prep.documents import load_documents, snapshot, comparison_text
from asset_prep.errors import ValidationError
from asset_prep.preflight import inspect_inputs
from fixtures import write_pdf, PDF_TEXT
from test_boundaries import TempCase


class PDFPrimaryTests(TempCase):
    def pair(self, xml='<broken'):
        pdf, markup = self.root/'source.pdf', self.root/'source.xml'
        write_pdf(pdf)
        markup.write_text(xml, encoding='utf-8')
        return pdf, markup

    def test_optional_xml_is_not_opened_parsed_or_hashed_by_default(self):
        pdf, xml = self.pair()
        with patch('asset_prep.documents.snapshot', wraps=snapshot) as read, patch(
                'asset_prep.documents.ElementTree.fromstring', side_effect=AssertionError('must not parse XML')):
            result = load_documents(pdf, xml, 10000)
        read.assert_called_once_with(pdf, '.pdf')
        self.assertEqual(set(result['files']), {'pdf'})
        self.assertEqual(result['parsing']['xml_status'], 'not_read')
        self.assertTrue(all(s['kind']=='pdf' for s in result['sources']))

    def test_missing_unused_xml_does_not_change_document_identity(self):
        pdf, xml = self.pair()
        first = load_documents(pdf, xml, 10000)
        xml.unlink()
        self.assertEqual(load_documents(pdf, xml, 10000), first)
        self.assertEqual(load_documents(pdf, None, 10000), first)

    def test_supplement_is_targeted_and_records_reason_and_original_source(self):
        pdf, xml = self.pair('<article><body><p>Capacity missing 5 mL.</p><p>Unrelated centrifuge.</p></body></article>')
        result = load_documents(pdf, xml, 10000, xml_reason='容量在PDF中不清晰', xml_queries=['Capacity'])
        supplements = [s for s in result['sources'] if s['kind']=='xml']
        self.assertEqual([s['text'] for s in supplements], ['Capacity missing 5 mL.'])
        self.assertEqual(result['parsing']['xml_request']['selected_ids'], ['xml:b1'])
        self.assertEqual(result['parsing']['xml_request']['reason'], '容量在PDF中不清晰')
        self.assertIn('xml', result['files'])

    def test_supplement_skips_text_already_in_pdf_without_altering_quotes(self):
        pdf, xml = self.pair('<article><p>'+PDF_TEXT+'</p><p>Extra capacity 5 mL.</p></article>')
        result = load_documents(pdf, xml, 10000, xml_reason='核对容量', xml_queries=['mL'])
        self.assertEqual(result['parsing']['xml_request']['duplicate_pdf_ids'], ['xml:b1'])
        self.assertEqual([s['text'] for s in result['sources'] if s['kind']=='xml'], ['Extra capacity 5 mL.'])
        self.assertIn(PDF_TEXT, result['sources'][0]['text'])

    def test_reason_and_query_are_both_required_before_reading_xml(self):
        pdf, xml = self.pair()
        for options in ({'xml_reason':'missing'}, {'xml_queries':['capacity']}, {'xml_reason':' ', 'xml_queries':['x']}):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                load_documents(pdf, xml, 10000, **options)

    def test_duplicate_comparison_keeps_units_exponents_and_number_boundaries(self):
        self.assertNotEqual(comparison_text('5 mM'), comparison_text('5 mm'))
        self.assertNotEqual(comparison_text('10³'), comparison_text('103'))
        self.assertNotEqual(comparison_text('1 0'), comparison_text('10'))
        self.assertEqual(comparison_text('tube\n capacity'), comparison_text('tube capacity'))

    def test_unmatched_query_never_falls_back_to_whole_xml(self):
        pdf, xml = self.pair('<article><p>unrelated</p></article>')
        with self.assertRaisesRegex(ValidationError, 'not found'):
            load_documents(pdf, xml, 10000, xml_reason='容量缺口', xml_queries=['capacity'])

    def test_overbroad_supplement_stops_instead_of_silent_truncation(self):
        pdf, xml = self.pair('<article><p>capacity '+ 'x'*24001 +'</p></article>')
        with self.assertRaisesRegex(ValidationError, 'narrow the search terms'):
            load_documents(pdf, xml, 100000, xml_reason='容量缺口', xml_queries=['capacity'])

    def test_pdf_only_dataset_and_cli_skip_credentials(self):
        case = self.root/'case'; case.mkdir(); pdf=case/'source.pdf'; write_pdf(pdf)
        report = inspect_inputs(Config(), dataset=self.root)
        self.assertEqual(report['input_checked'], 1)
        self.assertEqual(report['cases'][0]['parsing']['xml_status'], 'not_read')
        with patch('asset_prep.cli.run_directory', return_value=self.root/'preflight'), patch(
                'asset_prep.cli.credentials', side_effect=AssertionError('no keys')), redirect_stdout(StringIO()):
            self.assertEqual(main(['--pdf', str(pdf), '--preflight', '--run', 'fixture']), 0)
