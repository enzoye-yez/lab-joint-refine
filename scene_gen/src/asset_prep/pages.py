"""Read the static content and image references of cited web pages; never execute page scripts."""

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
        # Structured product data is parsed after the body scan so earlier decorative images cannot push it out of the candidate set.
        self.candidates.sort(key=lambda i: self.IMAGE_PRIORITY[i["discovery"]])
        if len(self.candidates) > 100:
            self.candidates.pop()
            if "page image candidates exceeded the limit; kept by source priority" not in self.warnings:
                self.warnings.append("page image candidates exceeded the limit; kept by source priority")

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
                self.candidate(attrs.get("content"), "page image metadata", "page_meta")
        if tag in {"img", "source"}:
            description = attrs.get("alt") or attrs.get("title") or "page image; model not visually verified"
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
            # With malformed nesting, hidden content is conservatively dropped and scripts are never treated as body text.
            index = len(self.hidden) - 1 - self.hidden[::-1].index(tag)
            del self.hidden[index:]

    def handle_data(self, data):
        if self.json_ld is not None:
            self.json_ld.append(data)
        if not self.hidden and data.strip():
            self.parts.append(data.strip())

    def product_images(self, value, depth=0):
        if depth > 16:
            self.warnings.append("structured product data is nested too deeply; some images were not extracted")
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
                    self.candidate(image, str(value.get("name") or "structured product image"), "page_product_data")
            for name, item in value.items():
                if name != "image" and isinstance(item, (list, dict)):
                    self.product_images(item, depth + 1)

    def result(self):
        for block in self.json_blocks:
            try:
                self.product_images(json.loads(block))
            except (ValueError, RecursionError):
                self.warnings.append("page structured product data could not be parsed; skipped")
        # Structured product data/metadata takes priority over plain images, which may be logos or decoration.
        images = self.candidates[:30]
        if len(self.candidates) > 30:
            self.warnings.append("only the 30 highest-priority image candidates are output per page")
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
                    raise ProviderError("cited source is not a static web page; body text and images are not extracted for now")
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
                    warnings.append(f"{source['id']} has no usable static body text; model summaries are not used to fill in manufacturer specifications")
            except (ProviderError, ValidationError) as exc:
                source.update(page_status="unavailable")
                warnings.append(f"{source['id']} page read failed: {exc}")
        if len(references["sources"]) > self.config.pages_per_asset:
            warnings.append("some citations exceed the per-asset page read limit; only the citation is kept and its body is not considered read")
        return references
