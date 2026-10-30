"""读取引用网页的静态内容和图片引用；不执行网页脚本。"""

import hashlib
import json
import re
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin

from .errors import ProviderError, ValidationError
from .providers import redact, valid_public_url_shape
from .storage import digest


class ProductPageParser(HTMLParser):
    IMAGE_PRIORITY = {"page_product_data": 0, "page_meta": 1, "page_img": 2, "page_srcset": 3}

    def __init__(self, page_url):
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.parts, self.candidates = [], []
        self.hidden = []
        self.json_ld = None
        self.json_blocks = []
        self.warnings = []

    def candidate(self, url, description, origin):
        if not isinstance(url, str) or not url.strip():
            return
        try:
            resolved = urldefrag(urljoin(self.page_url, url.strip()))[0]
        except ValueError:
            return
        if not valid_public_url_shape(resolved):
            return
        candidate = {"url": resolved, "description": description[:1000], "discovery": origin}
        existing = next((i for i in self.candidates if i["url"] == resolved), None)
        if existing is not None:
            if self.IMAGE_PRIORITY[origin] < self.IMAGE_PRIORITY[existing["discovery"]]:
                existing.update(candidate)
            return
        self.candidates.append(candidate)
        # 结构化产品数据在正文扫描后解析，不能被先出现的装饰图片挤出候选集。
        self.candidates.sort(key=lambda i: self.IMAGE_PRIORITY[i["discovery"]])
        if len(self.candidates) > 100:
            self.candidates.pop()
            if "页面图片候选超过上限，已按来源优先级保留" not in self.warnings:
                self.warnings.append("页面图片候选超过上限，已按来源优先级保留")

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style", "noscript", "template"}:
            if tag == "script" and not self.hidden and (attrs.get("type") or "").lower() == "application/ld+json":
                self.json_ld = []
            self.hidden.append(tag)
            return
        if self.hidden:
            return
        if tag == "meta":
            name = (attrs.get("property") or attrs.get("name") or "").lower()
            if name in {"og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src"}:
                self.candidate(attrs.get("content"), "页面图片元数据", "page_meta")
        if tag in {"img", "source"}:
            description = attrs.get("alt") or attrs.get("title") or "页面图片，型号未视觉核实"
            for name in ("data-src", "data-original", "src"):
                self.candidate(attrs.get(name), description, "page_img")
            value = attrs.get("srcset") or attrs.get("data-srcset") or ""
            if "data:" not in value.lower():
                for entry in value.split(","):
                    tokens = entry.split()
                    if tokens:
                        self.candidate(tokens[0], description, "page_srcset")

    def handle_endtag(self, tag):
        if tag == "script" and self.json_ld is not None:
            self.json_blocks.append("".join(self.json_ld))
            self.json_ld = None
        if tag in self.hidden:
            # 处理不规范嵌套时保守丢弃隐藏内容，不将脚本作为正文。
            index = len(self.hidden) - 1 - self.hidden[::-1].index(tag)
            del self.hidden[index:]

    def handle_data(self, data):
        if self.json_ld is not None:
            self.json_ld.append(data)
        if not self.hidden and data.strip():
            self.parts.append(data.strip())

    def product_images(self, value, depth=0):
        if depth > 16:
            self.warnings.append("结构化产品数据嵌套过深，部分图片未提取")
            return
        if isinstance(value, list):
            for item in value:
                self.product_images(item, depth + 1)
        elif isinstance(value, dict):
            types = value.get("@type", [])
            if isinstance(types, str):
                types = [types]
            if isinstance(types, list) and any(isinstance(t, str) and t.rsplit("/", 1)[-1] == "Product" for t in types):
                images = value.get("image", [])
                if not isinstance(images, list):
                    images = [images]
                for image in images:
                    if isinstance(image, dict):
                        image = image.get("contentUrl") or image.get("url")
                    self.candidate(image, str(value.get("name") or "结构化产品图片"), "page_product_data")
            for name, item in value.items():
                if name != "image" and isinstance(item, (list, dict)):
                    self.product_images(item, depth + 1)

    def result(self):
        for block in self.json_blocks:
            try:
                self.product_images(json.loads(block))
            except (ValueError, RecursionError):
                self.warnings.append("页面结构化产品数据无法解析，已跳过")
        # 产品结构化数据/元数据优先于普通图片，普通图片可能是品牌标识或装饰。
        images = self.candidates[:30]
        if len(self.candidates) > 30:
            self.warnings.append("单页只输出优先级最高的 30 个图片候选")
        content = "\n".join(self.parts)
        return {"content": content[:12000], "content_truncated": len(content) > 12000,
                "images": images, "warnings": list(dict.fromkeys(self.warnings))}


def parse_page(html, url):
    parser = ProductPageParser(url)
    parser.feed(html)
    parser.close()
    return parser.result()


class PageFetcher:
    def __init__(self, config, credentials, store, transport):
        self.config, self.credentials, self.store, self.transport = config, credentials, store, transport

    def fetch(self, url):
        key = digest({"kind": "page", "url": url})
        record = self.store.cached(key)
        if record is None:
            call = self.store.reserve(key, "page")
            try:
                raw, content_type = self.transport.request(url, timeout=self.config.timeout_seconds,
                                                           max_bytes=2_000_000, accept="text/html,application/xhtml+xml")
                if content_type.split(";", 1)[0].strip().lower() not in {"text/html", "application/xhtml+xml"}:
                    raise ProviderError("引用来源不是静态网页，暂不提取正文或图片")
                charset = re.search(r"charset\s*=\s*[\"']?([a-zA-Z0-9_-]+)", content_type, re.I)
                try:
                    html = raw.decode(charset.group(1) if charset else "utf-8", errors="replace")
                except LookupError:
                    html = raw.decode("utf-8", errors="replace")
                html = redact(html, self.credentials.values())
                record = {"url": url, "html": html, "content_type": content_type,
                          "body_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()}
                self.store.finish(call, record)
            except (ProviderError, ValidationError):
                self.store.fail(call, "PageError")
                raise
        result = parse_page(record["html"], url)
        result.update(page_sha256=record["body_sha256"], page_request_key=key)
        return result

    def enrich(self, references):
        warnings = references.setdefault("warnings", [])
        for source in references["sources"][:self.config.pages_per_asset]:
            try:
                page = self.fetch(source["url"])
                source.update(content=page["content"], content_truncated=page["content_truncated"],
                              evidence_kind="fetched_page", page_status="fetched",
                              page_sha256=page["page_sha256"], page_request_key=page["page_request_key"])
                warnings.extend(page["warnings"])
                for candidate in page["images"]:
                    image = next((i for i in references["images"] if i["url"] == candidate["url"]), None)
                    if image:
                        linked = image.setdefault("source_ids", [image["source_id"]] if image["source_id"] else [])
                        if source["id"] not in linked:
                            linked.append(source["id"])
                    else:
                        references["images"].append(dict(candidate, id=f"i{len(references['images'])}",
                            source_id=source["id"], source_ids=[source["id"]], association="page_linked",
                            visual_identity_verified=False, page_sha256=page["page_sha256"]))
                if not page["content"]:
                    warnings.append(f"{source['id']} 没有可用的静态正文；不以模型摘要补充厂商规格")
            except (ProviderError, ValidationError) as exc:
                source.update(page_status="unavailable")
                warnings.append(f"{source['id']} 网页读取失败：{exc}")
        if len(references["sources"]) > self.config.pages_per_asset:
            warnings.append("部分引用超过每项资产的网页读取上限，只保留引用，不视作已读取正文")
        return references
