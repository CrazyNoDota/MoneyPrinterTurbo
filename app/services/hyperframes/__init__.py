"""Hyperframes motion-graphics video -- the single entry point for the pipeline.

Hyperframes (https://github.com/heygen-com/hyperframes) renders HTML/CSS/JS +
GSAP into a deterministic MP4 via headless Chrome + ffmpeg. Unlike ``videogen``
(realistic b-roll from a diffusion API), this produces *synthetic motion graphics*
-- kinetic typography, animated numbers/lists -- which suit text-driven explainer
shorts (the finance niche especially).

Phase 1 = "solely-hyperframes": the whole visual track is one authored composition
timed to the existing TTS narration; ``task.py`` then muxes the audio + subtitles
over it as usual. Fully non-fatal: any problem returns ``""`` and the caller falls
back to stock footage.
"""

import os

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.utils import utils

from . import author, render, scenes

ClipScene = scenes.Scene


def is_enabled(params=None) -> bool:
    """True when solely-hyperframes mode is turned on (config or per-request)."""
    if params is not None and getattr(params, "hyperframes_enabled", False):
        return True
    return bool(config.app.get("hyperframes_enabled", False))


def is_available() -> bool:
    """Whether the local toolchain is installed (see setup-hyperframes.bat)."""
    return render.is_available()


def _resolution(params) -> tuple:
    aspect = getattr(params, "video_aspect", None)
    try:
        return VideoAspect(aspect).to_resolution()
    except Exception:  # noqa: BLE001 - fall back to vertical
        return 1080, 1920


def render_video(task_id, params, video_script, audio_file, subtitle_path, audio_duration) -> str:
    """Author + render the motion-graphics visual track. Returns an mp4 path or ``""``.

    The result is a *silent* video already at the right aspect and exactly
    ``audio_duration`` long, suitable as the ``combined-N.mp4`` the audio/subtitle
    mux step expects.
    """
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

    html = author.author_composition(scene_list, subject, width, height)
    if not html:
        logger.warning("hyperframes: composition authoring failed; falling back to stock footage")
        return ""

    out_path = os.path.join(utils.task_dir(task_id), "hyperframes.mp4")
    return render.render(html, out_path)


__all__ = ["is_enabled", "is_available", "render_video", "ClipScene"]
