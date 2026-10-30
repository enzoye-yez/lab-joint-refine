"""直接读取修订资料包或 asset_prep 导出，不重新提取实验流程。"""

import base64
import hashlib
import io
import re
from pathlib import Path

from asset_prep.storage import digest, file_hash, safe_path
from .contracts import read_json, require


def image_record(path, root, config, expected=None):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= config.max_image_bytes, "图片缺失或超过大小上限")
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    require(expected is None or sha == expected, "资产提取图片摘要不一致")
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            require(im.width * im.height <= 40000000, "图片像素数过大")
            require(getattr(im, "n_frames", 1) == 1, "仅支持静态参考图片")
            fmt = im.format
            require(fmt in {"PNG", "JPEG", "WEBP", "GIF"}, "图片格式不支持")
            im.verify()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        from asset_prep.errors import ValidationError
        raise ValidationError("图片内容无效") from exc
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}[fmt]
    return {"path": str(path.relative_to(root)), "sha256": sha, "mime": mime,
            "data_url": "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")}


def load_package(path, selected, config):
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size <= 4000000, "资料包不存在或过大")
    root = path.parent
    require(selected and len(set(selected)) == len(selected), "至少选择一个资产，且编号不能重复")
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
        require(len(sections) == len(headings) and all(a in sections for a in selected), "资产标题重复或所选编号不存在")
        scoped = "\n".join(sections[a] for a in selected)
        refs = list(dict.fromkeys(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", scoped)))
        for relative in refs:
            relative = relative.strip().strip("<>")
            require(not relative.startswith(("http:", "https:", "data:")), "资料包图片必须为本地相对路径")
            images.append(image_record(safe_path(root, relative), root, config))
        source = {"kind": "curated_markdown", "global_requirements": "".join(common),
                  "selected_sections": scoped}
        all_ids = list(sections)
        warnings = [] if images else ["所选资产没有参考图片"]
    else:
        package = read_json(path, 4000000)
        prepared = package.get('schema_version') == '2.0'
        if prepared:
            from asset_prep.task_delivery import load_delivery
            require(path.name=='asset_package.json', '新版任务须读取公开交付的资产资料包')
            package, _ = load_delivery(root)
        require(isinstance(package, dict) and isinstance(package.get("assets"), list), "不是资产提取模块导出的资料包")
        require(package.get("status") in {"completed", "completed_with_gaps", "extraction_completed", "requirements_completed"},
                "资产提取尚未完成，不能把部分结果当作完整输入")
        items = {item["asset_id"]: item for item in package["assets"]}
        require(len(items) == len(package["assets"]) and all(a in items for a in selected), "资产编号无效")
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
        warnings.extend(f"{item['asset_id']} 没有已下载的参考图片" for item in chosen if not item.get("downloaded_images"))
        if package["status"] != "completed":
            warnings.append("上游存在缺口；生成结果不能视为已核实的厂家规格")
    images = list({im["sha256"]: im for im in images}.values())
    records = [{k: v for k, v in im.items() if k != "data_url"} for im in images]
    return {"asset_ids": selected, "available_asset_ids": all_ids, "source": source, "images": images,
            "warnings": warnings, "identity": {"package_sha256": file_hash(path), "asset_ids": selected,
                                                "images": records, "source_digest": digest(source)}}


def public_bundle(bundle):
    return {k: v for k, v in bundle.items() if k != "images"} | {
        "images": [{k: v for k, v in im.items() if k != "data_url"} for im in bundle["images"]]}
