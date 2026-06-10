"""Acquire the media a directed/solely composition needs.

Two jobs:

* **Backgrounds** for motion-graphics scenes -- real photos pulled from the
  internet and staged inside the hyperframes project's ``assets/`` dir so the
  authored HTML can reference them as ``assets/<file>`` (a local file, not a
  network fetch, so the renderer stays deterministic).
* **Footage sources** for footage scenes -- the user's own clip (by
  ``material_ref``), else a stock video for the scene's query, else a stock
  photo as a last resort (turned into a Ken-Burns segment by the assembler).

Everything is best-effort: a miss returns ``""``/``[]`` and the caller degrades.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from app.services import material
from app.utils import utils

from . import render
from .plan import ScenePlan

_used_video_urls = set()
_used_image_paths = set()
_used_local_paths = set()


@dataclass
class Background:
    """A staged background photo, referenced from HTML as ``filename``."""

    filename: str  # project-relative, e.g. "assets/img-ab12.jpg"
    description: str


def reset() -> None:
    """Clear the project assets dir before staging a fresh composition."""
    _used_video_urls.clear()
    _used_image_paths.clear()
    _used_local_paths.clear()
    render.reset_assets()


def _min_background_dimension(params):
    """Reject photos that would upscale to a blurry full-bleed background.

    Anything below ~85% of the target frame on either axis looks soft once it
    covers the whole vertical frame, so we keep searching for a sharper one.
    """
    try:
        from app.models.schema import VideoAspect

        width, height = VideoAspect(params.video_aspect).to_resolution()
    except Exception:  # noqa: BLE001 - default to vertical
        width, height = 1080, 1920
    return int(width * 0.85), int(height * 0.85)


def fetch_backgrounds(query: str, params, source: str, count: int = 1) -> List[Background]:
    """Download up to ``count`` background photo(s) into the project assets dir."""
    if not query:
        return []
    paths = material.download_images(
        [query],
        source=source,
        video_aspect=params.video_aspect,
        count=count,
        save_dir=render.assets_dir(),
        min_dimension=_min_background_dimension(params),
    )
    return [Background(filename=f"assets/{os.path.basename(p)}", description=query) for p in paths]


def _fetch_stock_video(query: str, params, save_dir: str, source: str) -> str:
    """First not-yet-used downloadable stock clip for ``query`` (>=1s), or ""."""
    aspect = params.video_aspect
    search = material.search_videos_pexels
    if (source or "").lower() == "pixabay":
        search = material.search_videos_pixabay
    items = []
    try:
        items = search(query, minimum_duration=1, video_aspect=aspect)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"stock video search failed for '{query}': {e}")
    fallback = None
    for it in items:
        if it.url in _used_video_urls:
            if fallback is None:
                fallback = it
            continue
        try:
            path = material.save_video(it.url, save_dir=save_dir)
        except Exception:  # noqa: BLE001
            continue
        if path:
            _used_video_urls.add(it.url)
            return path
    if fallback is not None:
        try:
            path = material.save_video(fallback.url, save_dir=save_dir)
        except Exception:  # noqa: BLE001
            path = ""
        if path:
            logger.info(f"reusing stock video for '{query}' after exhausting unique results")
            return path
    return ""


def resolve_footage(
    plan: ScenePlan,
    params,
    source: str,
    save_dir: str = "",
) -> str:
    """Return a local source file (video or image) for a footage scene, or ""."""
    # 1. The user's own clip, when the planner matched one.
    ref = plan.material_ref
    materials = getattr(params, "video_materials", None) or []
    if isinstance(ref, int) and 0 <= ref < len(materials):
        url = getattr(materials[ref], "url", "") or ""
        if url and os.path.exists(url) and url not in _used_local_paths:
            _used_local_paths.add(url)
            return url
        logger.debug(f"material_ref {ref} not a local file; falling back to stock")

    for m in materials:
        url = getattr(m, "url", "") or ""
        if url and os.path.exists(url) and url not in _used_local_paths:
            _used_local_paths.add(url)
            return url

    # 2. A stock video for the scene's query.
    path = _fetch_stock_video(plan.query, params, save_dir, source)
    if path:
        return path

    # 3. Last resort: a stock photo (assembler applies a Ken-Burns zoom).
    imgs = material.download_images(
        [plan.query], source=source, video_aspect=params.video_aspect,
        count=1, save_dir=save_dir or utils.storage_dir("cache_images"),
    )
    for img in imgs:
        if img not in _used_image_paths:
            _used_image_paths.add(img)
            return img
    if imgs:
        return imgs[0]

    logger.warning(f"no footage source found for scene query '{plan.query}'")
    return ""
