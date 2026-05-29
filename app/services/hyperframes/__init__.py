"""Hyperframes motion-graphics video -- the pipeline entry points.

Hyperframes (https://github.com/heygen-com/hyperframes) renders HTML/CSS/JS +
GSAP into a deterministic MP4 via headless Chrome + ffmpeg. Unlike ``videogen``
(realistic b-roll from a diffusion API), this produces *synthetic motion graphics*
-- kinetic typography, animated numbers/lists -- now optionally over real photo
backgrounds pulled from the internet.

Three visual modes (see :func:`mode`):
- ``footage``     -- unchanged stock/local footage pipeline (hyperframes off).
- ``hyperframes`` -- "solely": the whole visual track is one authored composition
  timed to the narration (with on-demand photo backgrounds).
- ``mixed``       -- the "director": the LLM tags each scene footage vs motion
  graphics; footage scenes stay native, MG scenes are rendered and the segments
  are stitched in order.

Every path is non-fatal: any problem returns ``""`` and ``task.py`` falls back to
the stock-footage pipeline.
"""

import os
from dataclasses import dataclass
from typing import List

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.utils import utils

from . import assemble, assets, author, plan, render, scenes

ClipScene = scenes.Scene

_VALID_MODES = ("footage", "hyperframes", "mixed")


def is_enabled(params=None) -> bool:
    """Back-compat flag: solely-hyperframes mode turned on (config or per-request)."""
    if params is not None and getattr(params, "hyperframes_enabled", False):
        return True
    return bool(config.app.get("hyperframes_enabled", False))


def mode(params=None) -> str:
    """Resolve the visual mode: footage | hyperframes | mixed."""
    m = (getattr(params, "video_visual_mode", "") or "").strip().lower()
    if m in _VALID_MODES:
        return m
    m = (config.app.get("video_visual_mode", "") or "").strip().lower()
    if m in _VALID_MODES:
        return m
    return "hyperframes" if is_enabled(params) else "footage"


def is_available() -> bool:
    """Whether the local toolchain is installed (see setup-hyperframes.bat)."""
    return render.is_available()


def images_enabled() -> bool:
    """Whether to pull real photo backgrounds (config gate, default on)."""
    return bool(config.app.get("hyperframes_images", True))


def _image_source(params) -> str:
    src = (config.app.get("hyperframes_image_provider", "") or "").strip().lower()
    if src in ("pexels", "pixabay"):
        return src
    src = (getattr(params, "video_source", "") or "").strip().lower()
    return src if src in ("pexels", "pixabay") else "pexels"


def _resolution(params) -> tuple:
    aspect = getattr(params, "video_aspect", None)
    try:
        return VideoAspect(aspect).to_resolution()
    except Exception:  # noqa: BLE001 - fall back to vertical
        return 1080, 1920


def _rebase(scene_list: List[scenes.Scene]) -> List[scenes.Scene]:
    """Shift a block's scenes so the first starts at 0 (standalone segment)."""
    if not scene_list:
        return []
    offset = scene_list[0].start
    return [
        scenes.Scene(text=s.text, start=round(s.start - offset, 3), duration=s.duration)
        for s in scene_list
    ]


def _render_mg_block(task_id, params, subject, width, height, block, source, idx) -> str:
    """Author + render one contiguous block of MG scenes to an mp4, or ""."""
    rebased = _rebase([p.scene for p in block])

    backgrounds = []
    if images_enabled():
        for p in block:
            if p.use_background and p.query:
                backgrounds.extend(assets.fetch_backgrounds(p.query, params, source, count=1))

    html = author.author_block(rebased, subject, width, height, backgrounds=backgrounds)
    if not html:
        return ""
    out_path = os.path.join(utils.task_dir(task_id), f"mg-block-{idx}.mp4")
    return render.render(html, out_path)


def _footage_segment(task_id, params, plan_item, width, height, source, idx) -> str:
    """Resolve a source clip/photo for a footage scene and build a normalized segment."""
    src_path = assets.resolve_footage(plan_item, params, source)
    if not src_path:
        return ""
    out_path = os.path.join(utils.task_dir(task_id), f"seg-{idx}.mp4")
    return assemble.build_footage_segment(
        src_path, plan_item.scene.duration, width, height, out_path,
        threads=getattr(params, "n_threads", 2),
    )


def render_directed_video(
    task_id, params, video_script, audio_file, subtitle_path, audio_duration,
    video_terms=None, material_hints=None,
) -> str:
    """Director mode: plan each scene, render MG blocks, stitch with native footage."""
    if not is_available():
        logger.warning("hyperframes enabled but toolchain not installed; run setup-hyperframes.bat")
        return ""

    total = float(audio_duration or 0)
    scene_list = scenes.build_scenes(video_script, subtitle_path, total)
    if not scene_list:
        logger.warning("hyperframes director: no scenes; falling back to stock footage")
        return ""

    width, height = _resolution(params)
    subject = (getattr(params, "video_subject", "") or "").strip()
    bg_source = _image_source(params)
    footage_source = (getattr(params, "video_source", "") or "pexels").strip().lower()
    if footage_source not in ("pexels", "pixabay"):
        footage_source = "pexels"

    plans = plan.build_plan(scene_list, subject, video_terms, material_hints)
    assets.reset()

    segments: List[assemble.Segment] = []
    i, n, idx = 0, len(plans), 0
    while i < n:
        if plans[i].is_mg:
            j = i
            while j < n and plans[j].is_mg:
                j += 1
            block = plans[i:j]
            seg = _render_mg_block(task_id, params, subject, width, height, block, bg_source, idx)
            idx += 1
            if seg:
                segments.append(assemble.Segment(start=block[0].scene.start, file_path=seg))
            else:
                # MG authoring/render failed for the block: degrade to footage.
                for p in block:
                    fs = _footage_segment(task_id, params, p, width, height, footage_source, idx)
                    idx += 1
                    if fs:
                        segments.append(assemble.Segment(start=p.scene.start, file_path=fs))
            i = j
        else:
            fs = _footage_segment(task_id, params, plans[i], width, height, footage_source, idx)
            idx += 1
            if fs:
                segments.append(assemble.Segment(start=plans[i].scene.start, file_path=fs))
            i += 1

    if not segments:
        logger.warning("hyperframes director: produced no segments; falling back to stock footage")
        return ""

    combined = os.path.join(utils.task_dir(task_id), "combined-1.mp4")
    return assemble.assemble(combined, segments, threads=getattr(params, "n_threads", 2))


def _solely_backgrounds(params, video_terms, source, cap=5):
    """A small rotating set of real photos for solely mode (subject + terms)."""
    if not images_enabled():
        return []
    queries = []
    subject = (getattr(params, "video_subject", "") or "").strip()
    if subject:
        queries.append(subject)
    terms = video_terms
    if isinstance(terms, str):
        terms = [t.strip() for t in terms.split(",") if t.strip()]
    for t in terms or []:
        if t not in queries:
            queries.append(t)
    backgrounds = []
    for q in queries[:cap]:
        backgrounds.extend(assets.fetch_backgrounds(q, params, source, count=1))
        if len(backgrounds) >= cap:
            break
    return backgrounds[:cap]


def render_video(
    task_id, params, video_script, audio_file, subtitle_path, audio_duration,
    video_terms=None,
) -> str:
    """Solely mode: one motion-graphics composition (with optional photo backgrounds)."""
    if not is_available():
        logger.warning(
            "hyperframes is enabled but the toolchain is not installed; "
            "run setup-hyperframes.bat. Falling back to stock footage."
        )
        return ""

    total = float(audio_duration or 0)
    scene_list = scenes.build_scenes(video_script, subtitle_path, total)
    if not scene_list:
        logger.warning("hyperframes: no scenes could be built; falling back to stock footage")
        return ""

    width, height = _resolution(params)
    subject = (getattr(params, "video_subject", "") or "").strip()

    assets.reset()
    backgrounds = _solely_backgrounds(params, video_terms, _image_source(params))

    html = author.author_composition(scene_list, subject, width, height, backgrounds=backgrounds)
    if not html:
        logger.warning("hyperframes: composition authoring failed; falling back to stock footage")
        return ""

    out_path = os.path.join(utils.task_dir(task_id), "hyperframes.mp4")
    return render.render(html, out_path)


__all__ = [
    "is_enabled", "mode", "is_available", "images_enabled",
    "render_video", "render_directed_video", "ClipScene",
]
