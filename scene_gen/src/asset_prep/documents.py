"""Keep PDF evidence per page; XML is only supplementary and never silently overrides the PDF."""

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
    """Preserve superscript/subscript semantics so that 10 to the third power is not concatenated into 103."""
    if depth > 128:
        raise ValidationError("XML nesting exceeds 128 levels; refusing to parse")
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
    """Keep mixed content in original order within a block; each text or tail segment is emitted exactly once."""
    sources = []
    blocks = {"p", "title", "article-title", "subtitle", "label", "table", "mixed-citation",
              "element-citation", "citation", "disp-formula", "preformat"}

    def emit(text, path, section, role):
        if text and text.strip():
            sources.append({"id": f"xml:b{len(sources) + 1}", "kind": "xml", "text": text.strip(),
                            "path": path, "section": section, "role": role})

    def walk(element, path, section, role, depth=0):
        if depth > 128:
            raise ValidationError("XML nesting exceeds 128 levels; refusing to parse")
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
        # An unknown container holding only inline elements is also kept as one block, so numbers and units are not split apart.
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
    """Record parser warnings in the input evidence instead of dumping full font dictionaries."""
    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def snapshot(path: Path, suffix: str):
    if path is None or path.suffix.lower() != suffix or not path.is_file():
        raise ValidationError(f"missing a valid {suffix} input file")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValidationError("a single input file must not exceed 50 MiB")
    with path.open('rb') as stream:
        data = stream.read(50 * 1024 * 1024 + 1)
    if len(data) > 50 * 1024 * 1024:
        raise ValidationError("input file exceeded the size limit while being read")
    return data


def load_documents(pdf: Path, xml: Path | None, max_chars: int, *, xml_reason=None, xml_queries=()):
    if bool(xml_reason) != bool(xml_queries):
        raise ValidationError('supplementary XML reading requires both the information-gap reason and specific search terms')
    if xml_reason and (not isinstance(xml_reason, str) or not xml_reason.strip()
                       or not all(isinstance(q, str) and q.strip() for q in xml_queries)):
        raise ValidationError('the XML supplement reason and search terms must be non-empty text')
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
            raise ValidationError("encrypted PDFs are not supported yet")
        if len(reader.pages) > 300:
            raise ValidationError("PDF page count exceeds the first-release limit of 300 pages")
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                warnings.append(f"pdf:p{number} has no extractable text; OCR was not run, so coverage of this page cannot be confirmed")
            sources.append({"id": f"pdf:p{number}", "kind": "pdf", "text": text})
            count += len(text)
            if count > max_chars:
                raise ValidationError("PDF exceeds the document character limit; it will not be silently truncated")
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("PDF parsing failed; check for format problems, corruption or encryption") from exc
    finally:
        logger.handlers, logger.propagate = old_handlers, old_propagate
    if warning_log.count:
        warnings.append(f"PDF parser reported {warning_log.count} warnings; text and special characters need checking")
    if not any(s["kind"] == "pdf" and s["text"].strip() for s in sources):
        raise ValidationError("PDF has no readable text; run OCR first; XML cannot be treated as verified PDF evidence")
    documents = {'sources': sources, 'warnings': warnings,
        'parsing': {'input_policy': 'pdf_primary_targeted_xml', 'total_characters': count,
                    'xml_status': 'not_read', 'xml_blocks': 0, 'pairing': 'not_checked',
                    'versions': {name: version(name) for name in ('pypdf', 'fonttools')}},
        'files': {'pdf': {'sha256': hashlib.sha256(pdf_snapshot).hexdigest(), 'name': pdf.name}}}
    if xml_reason:
        supplement_xml(documents, xml, max_chars, reason=xml_reason, queries=xml_queries)
    return documents


def comparison_text(text):
    # Used only to filter duplicate content; the stored source text used for verbatim checks is never altered.
    return re.sub(r'\s+', ' ', text).strip()


def supplement_xml(documents, xml, max_chars, *, reason, queries):
    """Parse the backup file only once a gap is stated, adding only matching non-duplicate text blocks."""
    if not isinstance(reason, str) or not reason.strip() or not queries or not all(
            isinstance(q, str) and q.strip() for q in queries):
        raise ValidationError('supplementary XML reading must state a specific reason and search terms')
    xml_snapshot = snapshot(xml, '.xml')
    try:
        root = ElementTree.fromstring(xml_snapshot, forbid_dtd=False, forbid_entities=True, forbid_external=True)
    except DefusedXmlException as exc:
        raise ValidationError("XML contains forbidden entities or external entity references; external resources are not read") from exc
    except Exception as exc:
        raise ValidationError("XML cannot be parsed; check whether the file is corrupted or malformed") from exc
    markup = xml_sources(root)
    terms = [q.strip().casefold() for q in queries]
    matched = [s for s in markup if any(q in s['text'].casefold() or
               q in ' '.join(s['section']).casefold() for q in terms)]
    if not matched:
        raise ValidationError('the specified supplementary content was not found in the XML; will not fall back to extracting the whole file')
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
        raise ValidationError('XML supplementary matches exceed 24000 characters; narrow the search terms; will not switch to full-text extraction')
    if documents['parsing']['total_characters'] + size > max_chars:
        raise ValidationError('PDF plus XML supplement together exceed the character limit; will not be silently truncated')
    warnings = documents['warnings']
    math_count = sum(local_name(e) == 'math' for e in root.iter())
    if math_count:
        warnings.append(f'backup XML contains {math_count} math formulas; the semantics of complex formulas in the supplement still need checking')
    # Identifiers are only a pairing hint; a valid file without identifiers is not misjudged as a different paper.
    dois = ["".join(e.itertext()).strip() for e in root.iter()
            if local_name(e) == "article-id" and e.get("pub-id-type") == "doi"]
    front_text = "".join(s["text"] for s in documents['sources'][:2]).lower()
    pairing = "identifier_found" if dois and any(d.lower() in front_text for d in dois) else "unconfirmed"
    if pairing == "unconfirmed":
        warnings.append("PDF/XML correspondence was not confirmed by first-page identifiers; pairing the files does not mean the content matches")
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
