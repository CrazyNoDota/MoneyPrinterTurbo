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


@dataclass
class Background:
    """A staged background photo, referenced from HTML as ``filename``."""

    filename: str  # project-relative, e.g. "assets/img-ab12.jpg"
    description: str


def reset() -> None:
    """Clear the project assets dir before staging a fresh composition."""
    render.reset_assets()


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
    )
    return [Background(filename=f"assets/{os.path.basename(p)}", description=query) for p in paths]


def _fetch_stock_video(query: str, params, save_dir: str, source: str) -> str:
    """First downloadable stock clip for ``query`` (>=1s), or ""."""
    aspect = params.video_aspect
    search = material.search_videos_pexels
    if (source or "").lower() == "pixabay":
        search = material.search_videos_pixabay
    items = []
    try:
        items = search(query, minimum_duration=1, video_aspect=aspect)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"stock video search failed for '{query}': {e}")
    for it in items:
        try:
            path = material.save_video(it.url, save_dir=save_dir)
        except Exception:  # noqa: BLE001
            continue
        if path:
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
        if url and os.path.exists(url):
            return url
        logger.debug(f"material_ref {ref} not a local file; falling back to stock")

    # 2. A stock video for the scene's query.
    path = _fetch_stock_video(plan.query, params, save_dir, source)
    if path:
        return path

    # 3. Last resort: a stock photo (assembler applies a Ken-Burns zoom).
    imgs = material.download_images(
        [plan.query], source=source, video_aspect=params.video_aspect,
        count=1, save_dir=save_dir or utils.storage_dir("cache_images"),
    )
    if imgs:
        return imgs[0]

    logger.warning(f"no footage source found for scene query '{plan.query}'")
    return ""
