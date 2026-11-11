"""Read the revised package or the asset_prep export directly; never re-extract the experimental procedure."""

import base64
import hashlib
import io
import re
from pathlib import Path

from asset_prep.storage import digest, file_hash, safe_path
from .contracts import read_json, require


def image_record(path, root, config, expected=None):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= config.max_image_bytes, "Image missing or exceeds the size limit")
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    require(expected is None or sha == expected, "Asset extraction image digest mismatch")
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            require(im.width * im.height <= 40000000, "Image has too many pixels")
            require(getattr(im, "n_frames", 1) == 1, "Only static reference images are supported")
            fmt = im.format
            require(fmt in {"PNG", "JPEG", "WEBP", "GIF"}, "Unsupported image format")
            im.verify()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        from asset_prep.errors import ValidationError
        raise ValidationError("Invalid image content") from exc
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}[fmt]
    return {"path": str(path.relative_to(root)), "sha256": sha, "mime": mime,
            "data_url": "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")}


def load_package(path, selected, config):
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size <= 4000000, "Package missing or too large")
    root = path.parent
    require(selected and len(set(selected)) == len(selected), "Select at least one asset, with no duplicate ids")
    images = []
    if path.suffix.lower() == ".md":
        text = path.read_text(encoding="utf-8-sig")
        headings = list(re.finditer(r"^###\s+(A\d+)\s*[｜|].*$", text, re.M))
        sections, common, end = {}, [], 0
        for i, h in enumerate(headings):
            common.append(text[end:h.start()])
            end = headings[i+1].start() if i+1<len(headings) else len(text)
            shared_heading = re.search(r"^#{1,2}\s+",text[h.end():end],re.M)
            if shared_heading:
                end = h.end()+shared_heading.start()
            sections[h[1]] = text[h.start():end]
        common.append(text[end:])
        require(len(sections) == len(headings) and all(a in sections for a in selected), "Duplicate asset headings or selected id not found")
        scoped = "\n".join(sections[a] for a in selected)
        refs = list(dict.fromkeys(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", scoped)))
        for relative in refs:
            relative = relative.strip().strip("<>")
            require(not relative.startswith(("http:", "https:", "data:")), "Package images must be local relative paths")
            images.append(image_record(safe_path(root, relative), root, config))
        source = {"kind": "curated_markdown", "global_requirements": "".join(common),
                  "selected_sections": scoped}
        all_ids = list(sections)
        warnings = [] if images else ["Selected assets have no reference images"]
    else:
        package = read_json(path, 4000000)
        prepared = package.get('schema_version') == '2.0'
        if prepared:
            from asset_prep.task_delivery import load_delivery
            require(path.name=='asset_package.json', 'New-format tasks must read the publicly delivered asset package')
            package, _ = load_delivery(root)
        require(isinstance(package, dict) and isinstance(package.get("assets"), list), "Not a package exported by the asset extraction module")
        require(package.get("status") in {"completed", "completed_with_gaps", "extraction_completed", "requirements_completed"},
                "Asset extraction is not finished; partial results cannot be used as complete input")
        items = {item["asset_id"]: item for item in package["assets"]}
        require(len(items) == len(package["assets"]) and all(a in items for a in selected), "Invalid asset id")
        chosen = [items[a] for a in selected]
        for item in chosen:
            for entry in item.get("downloaded_images", []):
                images.append(image_record(safe_path(root, entry["local_path"]), root, config, entry["sha256"]))
        source = {"kind": "asset_prep_export", "assets": chosen, "inventory": package.get("inventory"),
                  "source_status": package["status"]}
        if prepared:
            from asset_prep.task_contracts import project_task
            source.pop('inventory')
            source.update(kind='prepared_task', task=project_task(package, selected))
        all_ids = list(items)
        warnings = [w for item in chosen for w in item.get("warnings", [])]
        warnings.extend(f"{item['asset_id']} has no downloaded reference images" for item in chosen if not item.get("downloaded_images"))
        if package["status"] != "completed":
            warnings.append("Upstream has gaps; generated results cannot be treated as verified manufacturer specifications")
    images = list({im["sha256"]: im for im in images}.values())
    records = [{k: v for k, v in im.items() if k != "data_url"} for im in images]
    return {"asset_ids": selected, "available_asset_ids": all_ids, "source": source, "images": images,
            "warnings": warnings, "identity": {"package_sha256": file_hash(path), "asset_ids": selected,
                                                "images": records, "source_digest": digest(source)}}


def public_bundle(bundle):
    return {k: v for k, v in bundle.items() if k != "images"} | {
        "images": [{k: v for k, v in im.items() if k != "data_url"} for im in bundle["images"]]}
