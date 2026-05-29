"""Visual understanding of local materials.

Samples frames from videos (and uses images directly), sends them to a vision
model, and returns concise descriptions of what is actually shown. These
descriptions are used to write the narration script and to build stock-footage
search terms that match the real content.

Currently targets NVIDIA NIM vision models (OpenAI-compatible chat endpoint),
which accept inline images embedded as ``<img src="data:image/jpeg;base64,..."/>``
in the message content, with a ~180KB-per-image size limit.
"""

import base64
import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests
from loguru import logger
from PIL import Image

from app.config import config
from app.models import const
from app.utils import cache, utils
from app.utils.retry import call_with_retry

# NVIDIA NIM rejects inline images whose base64 payload exceeds ~180KB; stay
# safely under that so a single frame never trips the limit.
_MAX_IMAGE_B64_BYTES = 170_000

_CAPTION_PROMPT = (
    "Describe what is visible in this image in one short, concrete sentence: "
    "the main subject, the setting, and any action. Be literal and specific. "
    "Output only the description, no preamble."
)


def is_enabled() -> bool:
    return bool(config.app.get("vision_enabled", False))


def _api_key() -> str:
    # Dedicated key first, otherwise reuse the NVIDIA key (same endpoint).
    return config.app.get("vision_api_key") or config.app.get("nvidia_api_key") or ""


def _image_to_jpeg_b64(image: Image.Image, max_dim: int = 512) -> str:
    """Downscale + JPEG-encode a PIL image to a base64 string under the size cap."""
    image = image.convert("RGB")
    # progressively shrink/recompress until the encoded payload fits
    for dim, quality in ((max_dim, 70), (max_dim, 55), (384, 50), (320, 40)):
        work = image.copy()
        work.thumbnail((dim, dim))
        buffer = io.BytesIO()
        work.save(buffer, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        if len(encoded) <= _MAX_IMAGE_B64_BYTES:
            return encoded
    return encoded  # best effort; may still be rejected by the API


def _sample_video_frames(video_path: str) -> List[Image.Image]:
    """Grab one frame every ``vision_seconds_per_frame`` seconds, capped."""
    from moviepy.video.io.VideoFileClip import VideoFileClip

    seconds_per_frame = float(config.app.get("vision_seconds_per_frame", 5) or 5)
    max_frames = int(config.app.get("vision_max_frames", 4) or 4)

    frames: List[Image.Image] = []
    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration or 0
        if duration <= seconds_per_frame:
            timestamps = [max(duration * 0.5, 0)]
        else:
            timestamps = []
            t = 0.0
            while t < duration and len(timestamps) < max_frames:
                timestamps.append(t)
                t += seconds_per_frame
        for ts in timestamps:
            try:
                frame = clip.get_frame(min(ts, max(duration - 0.1, 0)))
                frames.append(Image.fromarray(frame))
            except Exception as exc:
                logger.warning(f"failed to read frame at {ts:.1f}s from {video_path}: {exc}")
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass
    return frames


def _caption_frame(image_b64: str) -> str:
    """Send a single frame to the vision model and return its description."""
    base_url = config.app.get("vision_base_url", "https://integrate.api.nvidia.com/v1")
    model = config.app.get("vision_model_name", "meta/llama-3.2-11b-vision-instruct")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Accept": "application/json",
    }
    content = f'{_CAPTION_PROMPT} <img src="data:image/jpeg;base64,{image_b64}" />'
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "temperature": 0.2,
        "stream": False,
    }
    def _post():
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            proxies=config.proxy,
            timeout=(30, 120),
        )
        r.raise_for_status()
        return r

    resp = call_with_retry(_post, description="vision.caption")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"vision model returned no choices: {data}")
    text = (choices[0].get("message") or {}).get("content") or ""
    return text.strip()


def _file_fingerprint(path: str) -> str:
    """MD5 of a file's bytes -- identifies identical content across runs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_enabled() -> bool:
    return bool(config.app.get("vision_cache", True))


def _cache_key(path: str) -> str:
    """Cache key tied to the file content + the settings that affect captions."""
    return cache.make_key(
        _file_fingerprint(path),
        config.app.get("vision_model_name", "meta/llama-3.2-11b-vision-instruct"),
        config.app.get("vision_seconds_per_frame", 5),
        config.app.get("vision_max_frames", 4),
    )


def describe_media(path: str) -> str:
    """Return a concise description of an image or video file.

    Results are cached on disk by file content (+ model/frame settings) so the
    same asset is never re-analyzed -- repeat runs are instant and free. Disable
    with ``config.app.vision_cache = false``.
    """
    cache_key = None
    if _cache_enabled():
        try:
            cache_key = _cache_key(path)
            hit = cache.get("vision", cache_key)
            if hit is not None:
                logger.info(f"vision cache hit: {os.path.basename(path)}")
                return hit
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            logger.warning(f"vision cache lookup failed for {path}: {exc}")
            cache_key = None

    ext = utils.parse_extension(path)
    if ext in const.FILE_TYPE_IMAGES:
        frames = [Image.open(path)]
    else:
        frames = _sample_video_frames(path)

    captions = []
    seen = set()
    for frame in frames:
        try:
            caption = _caption_frame(_image_to_jpeg_b64(frame))
        except Exception as exc:
            logger.warning(f"vision caption failed for {path}: {exc}")
            continue
        finally:
            try:
                frame.close()
            except Exception:
                pass
        key = caption.lower().strip(" .")
        if caption and key not in seen:
            seen.add(key)
            captions.append(caption)

    result = " ".join(captions).strip()
    if cache_key and result:
        cache.set("vision", cache_key, result)
    return result


def describe_materials(materials) -> List[str]:
    """Describe each material; returns one description string per analyzed file.

    Materials are analyzed concurrently (bounded by ``vision_concurrency``,
    default 4) since each call is an independent network round-trip; output order
    matches input order. Failures are skipped (non-fatal) so missing vision never
    breaks generation.
    """
    if not _api_key():
        logger.warning("vision is enabled but no API key is set; skipping analysis")
        return []

    paths = []
    for material in materials or []:
        path = getattr(material, "url", "") or ""
        if path and os.path.isfile(path):
            paths.append(path)

    if not paths:
        return []

    def _describe(path: str) -> str:
        logger.info(f"analyzing material with vision model: {path}")
        try:
            description = describe_media(path)
        except Exception as exc:  # noqa: BLE001 - per-file failure is non-fatal
            logger.warning(f"failed to analyze material {path}: {exc}")
            return ""
        if description:
            logger.success(f"vision: {os.path.basename(path)} -> {description}")
        return description

    workers = max(1, int(config.app.get("vision_concurrency", 4) or 4))
    workers = min(workers, len(paths))
    if workers == 1:
        results = [_describe(p) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_describe, paths))  # preserves order

    return [d for d in results if d]
