"""Read the evaluation material manifest directly; never call asset extraction or web search."""

import re
from pathlib import Path

from asset_generation.contracts import read_json, require
from asset_generation.inputs import load_package
from asset_prep.storage import file_hash, digest


def catalog(path, generation, protocol_pdf=None, max_protocol_chars=1000000):
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size <= 4000000, "asset package missing or too large")
    if path.suffix.lower() == ".md":
        source = path.read_text(encoding="utf-8-sig")
        ids = re.findall(r"^###\s+(A\d+)\s*[｜|].*$",source,re.M)
    else:
        source = read_json(path,4000000)
        ids = [a["asset_id"] for a in source["assets"]]
    require(ids and len(ids) <= 128 and len(set(ids)) == len(ids), "invalid asset list in the asset package")
    from .benchmark_input import load_public, summary_text, attach_bundle
    public = load_public(path, protocol_pdf)
    # Check images per item so the whole-scene image count is not mistaken for the per-request limit.
    bundles = {a:load_package(path,[a],generation) for a in ids}
    identity = {"package_sha256":file_hash(path),"assets":{a:b["identity"] for a,b in bundles.items()}}
    if public:
        identity['benchmark_public'] = public['identity']
        note = summary_text(public)
        if isinstance(source, str): source += note
        else: source = dict(source, benchmark_environment=public['summary'])
        for bundle in bundles.values():
            attach_bundle(bundle, public)
    protocol = None
    if protocol_pdf is not None:
        # Reuse only the local document parser; do not enter the extraction, workflow or search modules.
        from asset_prep.documents import load_documents
        documents = load_documents(Path(protocol_pdf).resolve(),None,max_protocol_chars)
        protocol = {"kind":"protocol_pdf_text", **documents,
                    "representation":"extractable text of all source pages; images and table layout are not certified by text parsing, instrument images use the original images from the asset package"}
        identity["protocol"] = {"files":documents["files"],"parsing":documents["parsing"],"content_sha256":digest(protocol)}
    prepared = source if isinstance(source, dict) and source.get('schema_version') == '2.0' else None
    return {"path":str(path),"asset_ids":ids,"source":source,"identity":identity,"bundles":bundles,"protocol":protocol,"benchmark_public":public,
            'prepared_task':prepared}


def task_bundle(catalogue, job):
    selected = [catalogue["bundles"][a] for a in job["asset_ids"]]
    images = list({im["sha256"]:im for b in selected for im in b["images"]}.values())
    # load_package already scopes asset sections and keeps common requirements.
    # Deduplicate common preambles without repeating the full catalogue per job.
    sources = list({digest(b["source"]): b["source"] for b in selected}.values())
    if sources and all(s.get("kind") == "curated_markdown" for s in sources):
        source = {"kind":"curated_markdown", "global_requirements":sources[0]["global_requirements"],
                  "selected_sections":"\n".join(s["selected_sections"] for s in sources)}
    elif sources and all(s.get("kind") == "asset_prep_export" for s in sources):
        source = {"kind":"asset_prep_export", "assets":[a for s in sources for a in s["assets"]],
                  "inventory":sources[0].get("inventory"), "source_status":sources[0]["source_status"]}
    elif catalogue.get('prepared_task'):
        from asset_prep.task_contracts import project_task
        source = {'kind':'prepared_task', 'task':project_task(catalogue['prepared_task'], job['asset_ids']),
                  'assets':[a for s in sources for a in s['assets']]}
    else:
        source = {"kind":"selected_asset_sources", "assets":sources}
    if catalogue.get('benchmark_public'):
        source['benchmark_environment'] = catalogue['benchmark_public']['summary']
    return {"asset_ids":job["asset_ids"],"source":source,"images":images,"protocol":catalogue.get("protocol"),
            "warnings":list(dict.fromkeys(w for b in selected for w in b["warnings"])),
            "identity":{"catalog":digest(catalogue["identity"]),"job":job}}
