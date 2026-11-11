"""Export existing renders of archived assets, labeling the status of each; does not run modeling scripts or call the model."""

import base64
import hashlib
import html
import json
import re
from pathlib import Path


def export(project, run, package=None, *, refresh=False, allow_empty=False):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run):
        raise ValueError("invalid run name")
    root = (project/"runs"/run).resolve()
    from asset_prep.state_archive import read_state
    state = read_state(root/"state.json")
    scene = state.get("scene", {})
    stages = {s["directory"].replace("\\", "/"): s for s in scene.get("stages", [])}
    titles = {}
    if package:
        for asset, title in re.findall(r"^###\s+(A\d+)\s*[｜|]\s*(.+)$", package.read_text(encoding="utf-8"), re.M):
            titles[asset] = title.strip()
    cards, records = [], []
    for job in scene.get("plan", {}).get("jobs", []):
        value = scene["jobs"].get(job["id"], {})
        if not value.get("preview_paths"):
            continue
        labels = {"candidate_ready":"single-asset candidate passed", "assembly_ready":"assemblable, items still pending verification",
                  "needs_review":"insufficient evidence, pending re-review", "failed":"failed checks", "review_pending":"awaiting re-review"}
        label = labels.get(value.get("status"), "uncertified candidate")
        if value.get("dirty") and value.get("status") in {"candidate_ready", "assembly_ready"}:
            label = "old version, pending re-verification"
        stage = stages[value.get("artifact_stage",value["stage"]).replace("\\", "/")]
        stage_root = (root/stage["directory"]).resolve()
        if not stage_root.is_relative_to(root):
            raise ValueError("asset directory escapes the root")
        heading = "、".join(f"{a} · {titles.get(a, a)}" for a in job["asset_ids"]) if job["role"] == "asset" else job["id"]
        views, hashes = [], {}
        for number, relative in enumerate(value.get("preview_paths", []), 1):
            path = (root/relative).resolve()
            image_stage = next((s for s in stages.values() if path.is_relative_to((root/s["directory"]).resolve())), None)
            if image_stage is None or path.suffix.lower() != ".png":
                raise ValueError("invalid asset image path")
            stage_root = (root/image_stage["directory"]).resolve()
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            key = path.relative_to(stage_root).as_posix()
            expected = {k.replace("\\", "/"): v for k, v in image_stage["files"].items()}
            if expected.get(key) != sha:
                raise ValueError("archived render digest mismatch: " + str(path))
            encoded = base64.b64encode(data).decode("ascii")
            views.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(heading)}, view {number}" '
                         f'onclick="showImage(this)"><figcaption>View {number} · click to enlarge</figcaption></figure>')
            hashes[relative] = sha
        if not views:
            continue
        concerns = value.get("feedback", {}).get("visual", {})
        notes = concerns.get("blocking", []) + concerns.get("findings", [])
        cards.append(f'<section><div class="heading"><h2>{html.escape(heading)}</h2><span>{html.escape(label)}</span></div>'
                     + ''.join(f'<p>{html.escape(note)}</p>' for note in notes) +
                     f'<div class="views">{"".join(views)}</div><details><summary>View provenance</summary>'
                     f'<p>Job: {html.escape(job["id"])}<br>Archive: {html.escape(value["stage"])}<br>'
                     f'These are existing model-generated renders, not retouched or regenerated.</p></details></section>')
        records.append({"job": job["id"], "stage": value["stage"], "status":value.get("status"), "images": hashes})
    if not cards and not allow_empty:
        raise ValueError("no asset with archived renders exists yet")
    output = root/"asset_preview"
    if output.is_symlink() or output.resolve().parent != root:
        raise ValueError("preview directory escapes the root")
    output.mkdir(exist_ok=refresh)
    if any(p.name not in {"index.html", "sources.json"} or p.is_symlink() for p in output.iterdir()):
        raise ValueError("preview directory contains unknown files; refusing to overwrite")
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Generated assets and check status</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10151d;color:#e6edf5;font:16px/1.6 "Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1400px;margin:auto;padding:38px 28px}h1{margin:0;font-size:32px}header p{color:#a7b4c6;max-width:1000px}
.tag{color:#8dd8ca;letter-spacing:.1em;font-size:13px}.notice{border-left:3px solid #d5ad69;padding:12px 18px;background:#23221f;margin:24px 0}
section{background:#18212d;border:1px solid #2b394b;border-radius:14px;padding:20px;margin:22px 0}
.heading{display:flex;align-items:center;justify-content:space-between;gap:16px}h2{font-size:20px;margin:0 0 16px}
.heading span{font-size:12px;color:#8dd8ca;white-space:nowrap}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
figure{margin:0}img{width:100%;display:block;border-radius:8px;cursor:zoom-in;background:#000}figcaption{color:#96a6bb;font-size:13px;padding:8px 0}
details{color:#9caec3;font-size:13px;margin-top:10px}summary{cursor:pointer}dialog{border:1px solid #576678;background:#111923;color:white;max-width:95vw;max-height:95vh;border-radius:10px}
dialog img{width:auto;max-width:88vw;max-height:78vh;cursor:auto}dialog::backdrop{background:#000c}button{background:#28394c;color:white;border:0;border-radius:6px;padding:8px 18px;cursor:pointer;float:right;margin-bottom:12px}
@media(max-width:800px){.views{grid-template-columns:1fr}main{padding:24px 14px}.heading{display:block}}
</style><main><header><div class="tag">Existing renders · single-asset preview</div><h1>Generated assets</h1>
'''+f'<p>Run: {html.escape(run)} · {len(cards)} generated assets · existing views shown for each</p>'+'''
<div class="notice">This page includes assemblable assets and assets that have not yet passed review; the per-item status is authoritative, and having images does not mean acceptance passed. Unverified size, fit and dynamic-operation items are governed by the report. Images are not shown at a uniform scale and are not retouched or regenerated.</div></header>
'''+("".join(cards) or '<p>No archived asset images yet; generation is in progress or requests have not succeeded yet. Refresh this page to see new outputs.</p>')+'''
<dialog id="zoom"><button onclick="this.closest('dialog').close()">Close</button><img id="large" alt="Enlarged asset render"></dialog>
<script>function showImage(img){document.getElementById('large').src=img.src;document.getElementById('large').alt=img.alt;document.getElementById('zoom').showModal()}</script>
</main></html>'''
    (output/"index.html").write_text(document, encoding="utf-8")
    (output/"sources.json").write_text(json.dumps({"run": run, "model_calls": 0, "assets": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output/"index.html"


