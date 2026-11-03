"""Offline input check: reads no credentials, creates no model requests, and does not mean the experiment passed."""

from pathlib import Path

from .documents import load_documents
from .errors import PrepError, ValidationError
from .extraction import request_size
from .workflow import LOCAL, plan_tasks, local_data
from .storage import digest, file_hash


def inspect_pair(pdf, xml, config, *, xml_reason=None, xml_queries=()):
    documents = load_documents(pdf, xml, config.max_document_chars, xml_reason=xml_reason, xml_queries=xml_queries)
    segments = plan_tasks(documents, config)
    sizes = [request_size(config, LOCAL, local_data(segment,documents,config)) for segment in segments]
    warnings = list(documents["warnings"])
    extraction_reserve = sum(size + 512 + min(config.max_output_tokens, config.extraction_output_tokens) for size in sizes)
    if len(segments) > config.max_model_calls or extraction_reserve > config.max_reserved_tokens:
        warnings.append("current total budget is below the reserved estimate for initial local extraction; identity, field and repair steps need extra budget, so a real run may pause")
    return {"status": "input_checked", "files": documents["files"], "parsing": documents["parsing"],
            "parsed_sources_sha256": digest(documents["sources"]),
            "pdf_pages": sum(s["kind"] == "pdf" for s in documents["sources"]),
            "extraction_segments": len(segments), "max_extraction_request_bytes": max(sizes),
            "extraction_reserved_tokens_estimate": extraction_reserve,
            "minimum_model_calls_before_search": len(segments),
            "merge_calls_estimate": "Identity pages hold at most 24 items and field pages at most 12, further split by bytes; new objects are committed only after all identity pages are scanned. Field matching and conflicts are counted separately; the number of calls cannot be determined before extraction.",
            "warnings": warnings,
            "limits": "checks only local extraction requests; historical assets are not part of the extraction input. Identity-page, field-page, conflict and repair calls grow with the actual task; semantic quality, API and search are not verified"}


def inspect_inputs(config, *, dataset=None, pdf=None, xml=None, xml_reason=None, xml_queries=()):
    if dataset is not None:
        dataset = Path(dataset)
        if not dataset.is_dir():
            raise ValidationError("input set directory does not exist")
        pairs = [(p.name, p / "source.pdf", p / "source.xml") for p in sorted(dataset.iterdir()) if p.is_dir()]
        if not pairs:
            raise ValidationError("input set has no case subdirectories")
    else:
        pairs = [(Path(pdf).parent.name, Path(pdf), Path(xml) if xml else None)]
    cases = []
    for name, pdf_path, xml_path in pairs:
        case = {"case": name, "pdf": str(pdf_path.resolve()), "xml": str(xml_path.resolve()) if xml_path else None}
        try:
            case.update(inspect_pair(pdf_path, xml_path, config, xml_reason=xml_reason, xml_queries=xml_queries))
        except (PrepError, OSError, ValueError) as exc:
            case.update(status="blocked", reason=str(exc) if isinstance(exc, PrepError) else type(exc).__name__)
        cases.append(case)
    return {"mode": "offline_input_preflight", "model_calls": 0, "config": config.identity(),
            "implementation": digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}),
            "budgets": {"max_model_calls": config.max_model_calls, "max_reserved_tokens": config.max_reserved_tokens},
            "total": len(cases), "input_checked": sum(c["status"] == "input_checked" for c in cases),
            "blocked": sum(c["status"] == "blocked" for c in cases), "cases": cases}
