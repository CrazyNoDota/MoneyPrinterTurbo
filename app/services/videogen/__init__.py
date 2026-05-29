"""AI video-clip generation -- pluggable, cached, budget-guarded.

This is the single entry point the pipeline uses. It is fully non-fatal: if
generation is disabled, not configured, over budget, or fails, it returns an
empty list and the caller falls back to stock/local footage.

Design (see ``base.py``): backends are async (submit -> poll). ``generate_clips``
drives them, caches finished clips on disk by spec identity (so retries and
re-runs are free), and enforces ``video_gen_max_clips`` / ``video_gen_max_seconds``
so an autonomous loop can never run away with cost.
"""

import os
import time
from typing import List

import requests
from loguru import logger

from app.config import config
from app.utils import cache, utils
from app.utils.retry import call_with_retry

from .base import (
    STATUS_DONE,
    STATUS_FAILED,
    ClipSpec,
    VideoGenError,
    VideoGenerator,
)
from .providers import build_provider

_CACHE_NS = "generated"


def _provider_name() -> str:
    return str(config.app.get("video_gen_provider", "null")).strip().lower()


def is_enabled() -> bool:
    """True only when a real (non-null) provider is selected and turned on."""
    if not config.app.get("video_gen_enabled", False):
        return False
    return _provider_name() not in ("", "null")


def get_generator() -> VideoGenerator:
    """Instantiate the configured provider (Null when disabled/unknown)."""
    return build_provider(_provider_name())


def _poll_until_done(gen: VideoGenerator, job: str) -> str:
    """Poll a job to completion; return the result URL or '' on failure/timeout."""
    timeout = float(config.app.get("video_gen_timeout", 300) or 300)
    interval = float(config.app.get("video_gen_poll_interval", 5) or 5)
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, url = gen.poll(job)
        if status == STATUS_DONE:
            return url or ""
        if status == STATUS_FAILED:
            logger.warning(f"video generation job {job} failed")
            return ""
        time.sleep(interval)
    logger.warning(f"video generation job {job} timed out after {timeout:.0f}s")
    return ""


def _download(url: str, dest: str) -> str:
    """Download a finished clip to ``dest`` (with retry). Returns dest or ''."""
    resp = call_with_retry(
        requests.get,
        url,
        proxies=config.proxy,
        timeout=(60, 240),
        description="videogen.download",
    )
    resp.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(resp.content)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    return ""


def _generate_one(gen: VideoGenerator, spec: ClipSpec) -> str:
    """Generate (or reuse a cached) single clip; return a local path or ''."""
    key = cache.make_key(gen.name, config.app.get("video_gen_model", ""), *spec.cache_parts())
    cached_path = cache.get(_CACHE_NS, key)
    if cached_path and os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
        logger.info(f"videogen cache hit: {os.path.basename(cached_path)}")
        return cached_path

    job = gen.submit(spec)
    logger.info(f"videogen submitted job {job} ({gen.name})")
    url = _poll_until_done(gen, job)
    if not url:
        return ""

    save_dir = utils.storage_dir(os.path.join("cache", _CACHE_NS), create=True)
    dest = os.path.join(save_dir, f"gen-{key}.mp4")
    local = _download(url, dest)
    if local:
        cache.set(_CACHE_NS, key, local)
        logger.success(f"videogen produced {os.path.basename(local)}")
    return local


def generate_clips(specs: List[ClipSpec]) -> List[str]:
    """Generate clips for ``specs`` honoring budget caps. Always non-fatal.

    The caller decides *whether* generation is wanted; this function only needs a
    configured provider. A null/unconfigured provider safely returns ``[]`` so
    the pipeline falls back to stock footage.
    """
    if not specs:
        return []

    gen = get_generator()
    if not gen.is_configured():
        logger.warning(
            f"video_gen_provider '{gen.name}' is not configured; "
            "skipping generation and falling back to stock footage"
        )
        return []

    max_clips = int(config.app.get("video_gen_max_clips", 4) or 4)
    max_seconds = float(config.app.get("video_gen_max_seconds", 60) or 60)

    paths: List[str] = []
    spent_seconds = 0.0
    for spec in specs:
        if len(paths) >= max_clips:
            logger.info(f"videogen reached max_clips={max_clips}; stopping")
            break
        if spent_seconds + spec.duration > max_seconds:
            logger.info(
                f"videogen would exceed max_seconds={max_seconds:.0f}; stopping"
            )
            break
        try:
            local = _generate_one(gen, spec)
        except VideoGenError as exc:
            logger.warning(f"videogen disabled/misconfigured: {exc}")
            break
        except Exception as exc:  # noqa: BLE001 - non-fatal per clip
            logger.warning(f"videogen failed for one clip, skipping: {exc}")
            continue
        if local:
            paths.append(local)
            spent_seconds += spec.duration

    logger.info(f"videogen produced {len(paths)} clip(s)")
    return paths


__all__ = ["is_enabled", "get_generator", "generate_clips", "ClipSpec"]
