"""开发者编写的合成样例，非模型自主生成，不属于正式实验数据。"""

import copy
import json
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from asset_prep.errors import ProviderError


PROVENANCE = "developer_authored_test_fixture"
PDF_TEXT = "Place two 0.2 mL tubes in the rack. Each tube must have a screw cap."
XML_TEXT = "Place three tubes in the rack."


def documents():
    return {"sources": [{"id": "pdf:p1", "kind": "pdf", "text": PDF_TEXT},
                        {"id": "xml:n1", "kind": "xml", "text": XML_TEXT}],
            "warnings": [], "files": {"pdf": {"sha256": "fixture-pdf", "name": "synthetic.pdf"},
                                      "xml": {"sha256": "fixture-xml", "name": "synthetic.xml"}}}


def ref(quote=PDF_TEXT, source="pdf:p1"):
    return {"source_id": source, "quote": quote}


def inventory():
    return {"assets": [{"id": "tubes", "name": "反应管", "category": "container", "aliases": [],
                        "quantity": {"value": 2, "origin": "explicit", "basis": "PDF specifies two", "evidence": [ref()]},
                        "requirements": [
                            {"key": "capacity", "value": "0.2", "unit": "mL", "origin": "explicit", "basis": "原文容量", "evidence": [ref()]},
                            {"key": "closure", "value": "螺旋盖", "unit": "", "origin": "explicit", "basis": "原文容器盖要求", "evidence": [ref()]}],
                        "operations": ["放置"], "evidence": [ref()], "parent_id": None, "queries": ["0.2 mL tube specifications images"]}],
            "steps": [{"id": "s1", "description": "放置试管", "asset_ids": ["tubes"], "evidence": [ref()]}],
            "relations": [], "conflicts": [{"topic": "数量", "asset_id": "tubes", "target_key": "quantity",
                "pdf_evidence": [ref()], "xml_evidence": [ref(XML_TEXT, "xml:n1")], "chosen_source": "pdf", "resolution": "采用两支"}],
            "assumptions": [], "issues": []}


def references():
    return {"sources": [{"id": "r0", "url": "https://manufacturer.example/tube", "title": "Tube X",
                         "content": "Tube X capacity 0.5 mL", "query": "tube", "content_truncated": False}],
            "images": [{"id": "i0", "url": "https://manufacturer.example/tube.png", "source_id": "r0",
                        "description": "Tube X", "association": "search_source_linked"}]}


def selection():
    return {"mode": "reference_adapted", "product_name": "Tube X", "source_ids": ["r0"],
            "selection_reason": "以相似产品为外形参考", "specifications": [
                {"key": "capacity", "value": "0.5", "unit": "mL", "source_id": "r0", "quote": "capacity 0.5 mL"}],
            "adaptations": [{"requirement_key": "capacity", "change": "生成0.2 mL目标容器", "reason": "遵守规程"}],
            "image_ids": ["i0"], "unknowns": ["外形未经过视觉核实"]}


def delta_decisions(incoming, target=None):
    return {'decisions': [{'asset_id': a['id'], 'target_id': target or a['id'],
        'reason': '开发者合成身份判断', 'quantity_from': 'incoming',
        'requirements': [{'incoming_key': r['key'], 'target_key': r['key'], 'take': 'incoming'}
                         for r in a['requirements']]} for a in incoming]}


class FakeTransport:
    """不联网、无费用；回复完全由开发者编写。"""

    def __init__(self, replies=None):
        self.replies = list(replies if replies is not None else [inventory(), selection()])
        self.model_calls = 0
        self.search_calls = 0
        self.image_calls = 0
        self.requests = []

    def post(self, url, payload, key, timeout):
        self.requests.append((url, copy.deepcopy(payload)))
        if url.endswith("/search"):
            self.search_calls += 1
            return {"results": [{"url": "https://manufacturer.example/tube", "title": "Tube X",
                                  "content": "Tube X capacity 0.5 mL", "images": [{"url": "https://manufacturer.example/tube.png"}]}]}
        self.model_calls += 1
        if not self.replies:
            raise AssertionError("Unexpected live-path call in fixture")
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        value = copy.deepcopy(value)
        # 合并接口的外层映射由合成传输构造；不是模型生成结果。
        data = json.loads(payload["messages"][1]["content"].split("\nTask data follows:\n", 1)[1])
        if data.get('operation')=='local_extract' and isinstance(value,dict) and 'assets' in value:
            value={'inventory':value,'consumed_units':len(data['units'])}
        if 'incoming_assets' in data and isinstance(value, dict) and ('assets' in value or 'inventory' in value):
            # 旧合成清单只作为决定目标的简写；空 coverage 故意保留为非法响应。
            if 'coverage' not in value or value['coverage']:
                target = (value.get('inventory') or value)['assets'][0]['id']
                value = {'decisions': [{'asset_id': a['id'], 'target_id': target,
                    'reason': '开发者合成身份判断', 'quantity_from': 'incoming',
                    'requirements': [{'incoming_key': r['key'], 'target_key': r['key'], 'take': 'incoming'}
                                     for r in a['requirements']]} for a in data['incoming_assets']]}
        if "drafts" in data and isinstance(value, dict) and "assets" in value:
            coverage = [{"draft_index": i, "asset_id": asset["id"], "target_id": asset["id"], "reason": "开发者合成同一对象映射"}
                        for i, draft in enumerate(data["drafts"]) for asset in draft["assets"]]
            value = {"inventory": value, "coverage": coverage}
        return {"model": "fixture-model-not-a-real-provider", "usage": {"total_tokens": 50},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    def request(self, url, **kwargs):
        self.image_calls += 1
        # Valid 1x1 PNG encoded fixture.
        import base64
        return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS1kAAAAASUVORK5CYII="), "image/png"


def write_pdf(path: Path, text=PDF_TEXT):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    if text:
        stream = DecodedStreamObject()
        stream.set_data(("BT /F1 10 Tf 30 700 Td (" + text + ") Tj ET").encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)
