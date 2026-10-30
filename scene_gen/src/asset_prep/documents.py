"""按页保留 PDF 证据；XML 仅作补充，不静默覆盖 PDF。"""

import hashlib
import logging
import re
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from pypdf import PdfReader

from .errors import ValidationError


def local_name(element):
    return element.tag.rsplit("}", 1)[-1]


def inline_text(element, depth=0):
    """保留上下标语义，避免 10 的三次方被拼成 103。"""
    if depth > 128:
        raise ValidationError("XML 嵌套超过 128 层，拒绝解析")
    tag = local_name(element)
    if tag in {"msup", "msub", "msubsup", "mfrac", "msqrt", "mroot"}:
        values = [inline_text(c, depth + 1) for c in element]
        if tag in {"msup", "msub", "mfrac", "mroot"} and len(values) == 2:
            first, second = values
            if tag == "msup":
                return f"({first})^({second})"
            if tag == "msub":
                return f"({first})_({second})"
            if tag == "mfrac":
                return f"({first})/({second})"
            return f"root({second}, {first})"
        if tag == "msubsup" and len(values) == 3:
            return f"({values[0]})_({values[1]})^({values[2]})"
        if tag == "msqrt":
            return "sqrt(" + "".join(values) + ")"
    parts = [element.text or ""]
    for child in element:
        value = inline_text(child, depth + 1)
        tag = local_name(child)
        if tag == "sup":
            value = "^(" + value + ")"
        elif tag == "sub":
            value = "_(" + value + ")"
        elif tag in {"p", "list-item", "tr", "title", "caption", "table", "table-wrap", "list", "sec"}:
            value = "\n" + value + "\n"
        elif tag in {"td", "th"}:
            value += "\t"
        parts.extend([value, child.tail or ""])
    return "".join(parts)


def xml_sources(root):
    """块内保留混合内容原序；每段正文或尾文本只输出一次。"""
    sources = []
    blocks = {"p", "title", "article-title", "subtitle", "label", "table", "mixed-citation",
              "element-citation", "citation", "disp-formula", "preformat"}

    def emit(text, path, section, role):
        if text and text.strip():
            sources.append({"id": f"xml:b{len(sources) + 1}", "kind": "xml", "text": text.strip(),
                            "path": path, "section": section, "role": role})

    def walk(element, path, section, role, depth=0):
        if depth > 128:
            raise ValidationError("XML 嵌套超过 128 层，拒绝解析")
        tag = local_name(element)
        if tag in {"front", "body", "back", "ref-list"}:
            role = {"front": "article_metadata", "body": "article_body", "back": "article_back",
                    "ref-list": "references"}[tag]
        if tag == "sec":
            title = next((c for c in element if local_name(c) == "title"), None)
            if title is not None:
                section = section + [inline_text(title).strip()]
        if tag in blocks:
            if tag == "table":
                text = "\n".join("\t".join(inline_text(cell).strip() for cell in row
                                           if local_name(cell) in {"td", "th"})
                                 for row in element.iter() if local_name(row) == "tr")
                if not text.strip():
                    text = inline_text(element)
            else:
                text = inline_text(element)
            emit(text, path, section, role)
            return
        # 未知容器若只有行内元素，也保持整块，避免将数字和单位拆开。
        if not any(local_name(e) in blocks or local_name(e) in {"sec", "front", "body", "back", "ref-list"}
                   for child in element for e in child.iter()):
            emit(inline_text(element), path, section, role)
            return
        emit(element.text, path + "/text()", section, role)
        counts = {}
        for child in element:
            name = local_name(child)
            counts[name] = counts.get(name, 0) + 1
            child_path = f"{path}/{name}[{counts[name]}]"
            walk(child, child_path, section, role, depth + 1)
            emit(child.tail, child_path + "/tail()", section, role)

    walk(root, f"/{local_name(root)}[1]", [], "unspecified")
    return sources


class PDFWarnings(logging.Handler):
    """将解析器警告计入输入证据，避免刷出完整字体字典。"""
    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def snapshot(path: Path, suffix: str):
    if path is None or path.suffix.lower() != suffix or not path.is_file():
        raise ValidationError(f"缺少有效的 {suffix} 输入文件")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValidationError("单份输入文件不得超过 50 MiB")
    with path.open('rb') as stream:
        data = stream.read(50 * 1024 * 1024 + 1)
    if len(data) > 50 * 1024 * 1024:
        raise ValidationError("读取时输入文件超过大小限制")
    return data


def load_documents(pdf: Path, xml: Path | None, max_chars: int, *, xml_reason=None, xml_queries=()):
    if bool(xml_reason) != bool(xml_queries):
        raise ValidationError('补读 XML 必须同时给出信息缺口原因和具体检索词')
    if xml_reason and (not isinstance(xml_reason, str) or not xml_reason.strip()
                       or not all(isinstance(q, str) and q.strip() for q in xml_queries)):
        raise ValidationError('XML 补读原因和检索词必须为非空文本')
    pdf_snapshot = snapshot(pdf, '.pdf')
    sources, warnings = [], []
    count = 0
    warning_log = PDFWarnings()
    logger = logging.getLogger("pypdf")
    old_handlers, old_propagate = logger.handlers[:], logger.propagate
    logger.handlers, logger.propagate = [warning_log], False
    try:
        reader = PdfReader(BytesIO(pdf_snapshot))
        if reader.is_encrypted:
            raise ValidationError("暂不支持加密 PDF")
        if len(reader.pages) > 300:
            raise ValidationError("PDF 页数超过首版限制 300 页")
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                warnings.append(f"pdf:p{number} 无可提取文本；未执行光学字符识别，不能确认该页覆盖完整")
            sources.append({"id": f"pdf:p{number}", "kind": "pdf", "text": text})
            count += len(text)
            if count > max_chars:
                raise ValidationError("PDF 超过文档字符限制；不会静默截断")
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("PDF 解析失败；检查格式、损坏或加密") from exc
    finally:
        logger.handlers, logger.propagate = old_handlers, old_propagate
    if warning_log.count:
        warnings.append(f"PDF 解析器报告 {warning_log.count} 条警告；文字及特殊字符需要核对")
    if not any(s["kind"] == "pdf" and s["text"].strip() for s in sources):
        raise ValidationError("PDF 没有可读文本；请先做光学字符识别，不能把 XML 当作 PDF 已验证依据")
    documents = {'sources': sources, 'warnings': warnings,
        'parsing': {'input_policy': 'pdf_primary_targeted_xml', 'total_characters': count,
                    'xml_status': 'not_read', 'xml_blocks': 0, 'pairing': 'not_checked',
                    'versions': {name: version(name) for name in ('pypdf', 'fonttools')}},
        'files': {'pdf': {'sha256': hashlib.sha256(pdf_snapshot).hexdigest(), 'name': pdf.name}}}
    if xml_reason:
        supplement_xml(documents, xml, max_chars, reason=xml_reason, queries=xml_queries)
    return documents


def comparison_text(text):
    # 只用于筛选重复内容；保存及逐字校验的原文始终不改动。
    return re.sub(r'\s+', ' ', text).strip()


def supplement_xml(documents, xml, max_chars, *, reason, queries):
    """明确缺口后才解析备用文件，只加入命中的非重复文本块。"""
    if not isinstance(reason, str) or not reason.strip() or not queries or not all(
            isinstance(q, str) and q.strip() for q in queries):
        raise ValidationError('补读 XML 必须说明具体原因及检索词')
    xml_snapshot = snapshot(xml, '.xml')
    try:
        root = ElementTree.fromstring(xml_snapshot, forbid_dtd=False, forbid_entities=True, forbid_external=True)
    except DefusedXmlException as exc:
        raise ValidationError("XML 含禁止的实体或外部实体引用；不读取外部资源") from exc
    except Exception as exc:
        raise ValidationError("XML 无法解析；检查文件是否损坏或格式错误") from exc
    markup = xml_sources(root)
    terms = [q.strip().casefold() for q in queries]
    matched = [s for s in markup if any(q in s['text'].casefold() or
               q in ' '.join(s['section']).casefold() for q in terms)]
    if not matched:
        raise ValidationError('XML 中未找到指定补证内容；不会改为提取整个文件')
    pdf_text = [comparison_text(s['text']) for s in documents['sources'] if s['kind'] == 'pdf']
    selected, duplicated = [], []
    for source in matched:
        value = comparison_text(source['text'])
        if value and any(value in page for page in pdf_text):
            duplicated.append(source['id'])
        else:
            selected.append(source)
    size = sum(len(s['text']) for s in selected)
    if size > 24000:
        raise ValidationError('XML 补证匹配超过 24000 字符；请缩小检索范围，不转为全文提取')
    if documents['parsing']['total_characters'] + size > max_chars:
        raise ValidationError('PDF 与 XML 补证合计超过字符限制；不会静默截断')
    warnings = documents['warnings']
    math_count = sum(local_name(e) == 'math' for e in root.iter())
    if math_count:
        warnings.append(f'备用 XML 含 {math_count} 个数学公式；补证中的复杂公式语义仍需核对')
    # 标识仅作配对提示，不误判没有标识的有效文件为不同论文。
    dois = ["".join(e.itertext()).strip() for e in root.iter()
            if local_name(e) == "article-id" and e.get("pub-id-type") == "doi"]
    front_text = "".join(s["text"] for s in documents['sources'][:2]).lower()
    pairing = "identifier_found" if dois and any(d.lower() in front_text for d in dois) else "unconfirmed"
    if pairing == "unconfirmed":
        warnings.append("PDF/XML 对应关系未通过首页标识确认；文件配对不等于内容一致")
    documents['sources'].extend(selected)
    documents['files']['xml'] = {'sha256': hashlib.sha256(xml_snapshot).hexdigest(), 'name': xml.name}
    documents['parsing'].update(xml_root=local_name(root), xml_blocks=len(selected),
        xml_status='supplemented' if selected else 'read_no_new_evidence', pairing=pairing, xml_dois=dois,
        total_characters=documents['parsing']['total_characters'] + size,
        xml_request={'reason': reason, 'queries': list(queries), 'selected_ids': [s['id'] for s in selected],
                     'duplicate_pdf_ids': duplicated})
    documents['parsing']['versions']['defusedxml'] = version('defusedxml')
    return documents


def chunks(sources, limit):
    result, current, size = [], [], 0
    for source in sources:
        text = source["text"]
        for offset in range(0, len(text), limit):
            part = dict(source, text=text[offset:offset + limit], offset=offset)
            if current and size + len(part["text"]) > limit:
                result.append(current)
                current, size = [], 0
            current.append(part)
            size += len(part["text"])
    if current:
        result.append(current)
    return result
