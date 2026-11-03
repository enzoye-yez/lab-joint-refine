"""Store reference images by content digest; never execute page scripts or trust remote file names."""

import base64
import hashlib

from .errors import ProviderError, StaleEvidence, ValidationError
from .storage import digest, safe_path


def image_extension(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    raise ProviderError("image response is not an allowed raster format")


def fetch_image(image, store, transport, config):
    key = digest({"kind": "image", "url": image["url"]})
    record = store.cached(key)
    if record is None:
        call = store.reserve(key, "image")
        try:
            data, content_type = transport.request(image["url"], timeout=config.timeout_seconds, max_bytes=5_000_000)
            if not content_type.lower().startswith("image/"):
                raise ProviderError("image URL returned non-image content")
            extension = image_extension(data)
            # The record contains the content, so a cleaned-up image can be rebuilt on resume, but a modified file must never be silently accepted.
            record = {"sha256": hashlib.sha256(data).hexdigest(), "extension": extension,
                      "data_base64": base64.b64encode(data).decode("ascii")}
            store.finish(call, record)
        except (ProviderError, ValidationError):
            store.fail(call, "ImageError")
            raise
    data = base64.b64decode(record["data_base64"], validate=True)
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise StaleEvidence("image record digest mismatch")
    relative = "images/" + record["sha256"] + record["extension"]
    path = safe_path(store.root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise StaleEvidence("reference image was modified; the old evidence is invalid")
    if not path.exists():
        temp = safe_path(store.root, relative + ".tmp")
        temp.write_bytes(data)
        temp.replace(path)
    return dict(image, local_path=relative, sha256=record["sha256"], visual_identity_verified=False)
