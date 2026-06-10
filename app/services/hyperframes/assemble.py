"""Stitch a directed video: native footage segments + rendered MG blocks.

Each segment is normalized to exactly the target resolution and ``fps`` (so the
ffmpeg concat demuxer stays happy), then concatenated **in scene order**. Footage
sources are real clips (subclipped/looped to the scene length) or stock photos
(turned into a Ken-Burns segment). Reuses the proven encode/concat helpers in
``app.services.video``.
"""

import math
import os
import hashlib
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger
from moviepy import CompositeVideoClip, ImageClip, concatenate_videoclips, vfx

from app.models import const
from app.services import video
from app.utils import utils


@dataclass
class Segment:
    """A finished, normalized segment file positioned at ``start`` seconds."""

    start: float
    file_path: str


def _is_image(path: str) -> bool:
    return utils.parse_extension(path) in const.FILE_TYPE_IMAGES


def _write(clip, out_path: str, threads: int) -> str:
    clip.write_videofile(
        out_path, fps=video.fps, codec=video.video_codec, audio=False,
        threads=threads or 2, logger=None,
    )
    return out_path


def _cover_to_aspect(clip, width: int, height: int):
    """Resize and crop to fill the frame, avoiding publish-time letterboxing."""
    clip_w, clip_h = clip.size
    scale = max(width / clip_w, height / clip_h)
    resized = clip.resized(new_size=(math.ceil(clip_w * scale), math.ceil(clip_h * scale)))
    return resized.cropped(
        x_center=resized.w / 2,
        y_center=resized.h / 2,
        width=width,
        height=height,
    )


def _motion_direction(source_path: str) -> tuple:
    digest = hashlib.md5((source_path or "").encode("utf-8")).hexdigest()
    mode = int(digest[:2], 16) % 4
    return {
        0: (0.0, 1.0, 0.0, 1.0),
        1: (1.0, 0.0, 1.0, 0.0),
        2: (0.0, 1.0, 1.0, 0.0),
        3: (1.0, 0.0, 0.0, 1.0),
    }[mode]


def _add_camera_motion(clip, width: int, height: int, duration: float, source_path: str, stronger: bool):
    """Apply deterministic slow push/pan so every segment has visible motion."""
    zoom = 0.09 if stronger else 0.035
    x0, x1, y0, y1 = _motion_direction(source_path)

    def ease(t: float) -> float:
        p = min(max(t / max(duration, 0.01), 0.0), 1.0)
        return p * p * (3 - 2 * p)

    def scale(t: float) -> float:
        return 1.0 + zoom * ease(t)

    def position(t: float):
        p = ease(t)
        current_scale = scale(t)
        extra_x = width * (current_scale - 1.0)
        extra_y = height * (current_scale - 1.0)
        x_anchor = x0 + (x1 - x0) * p
        y_anchor = y0 + (y1 - y0) * p
        return (-extra_x * x_anchor, -extra_y * y_anchor)

    moving = clip.resized(scale).with_position(position)
    return CompositeVideoClip([moving], size=(width, height)).with_duration(duration)


def _polish_segment(clip, width: int, height: int, duration: float, source_path: str, stronger: bool):
    work = _cover_to_aspect(clip, width, height).with_duration(duration)
    work = _add_camera_motion(work, width, height, duration, source_path, stronger=stronger)
    fade = min(0.18, max(duration / 8, 0.0))
    if fade > 0.04:
        work = work.with_effects([vfx.FadeIn(fade), vfx.FadeOut(fade)])
    return work


def build_footage_segment(
    source_path: str,
    duration: float,
    width: int,
    height: int,
    out_path: str,
    threads: int = 2,
) -> str:
    """Render ``source_path`` into a silent ``duration``-second W x H segment, or ""."""
    if not source_path or not os.path.exists(source_path) or duration <= 0:
        return ""

    clip = None
    work = None
    try:
        if _is_image(source_path):
            # Still photo -> stronger Ken-Burns style motion, full-bleed.
            work = _polish_segment(
                ImageClip(source_path).with_duration(duration),
                width, height, duration, source_path, stronger=True,
            )
        else:
            clip = video._open_video_clip_quietly(source_path)
            src_dur = clip.duration or 0
            if src_dur <= 0:
                return ""
            if src_dur >= duration:
                work = clip.subclipped(0, duration)
            else:
                # Loop the clip enough times to cover the scene, then trim exactly.
                reps = max(1, math.ceil(duration / src_dur))
                work = concatenate_videoclips([clip] * reps).subclipped(0, duration)
            work = _polish_segment(
                work, width, height, duration, source_path, stronger=False,
            )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        return _write(work, out_path, threads)
    except Exception as e:  # noqa: BLE001 - a bad source must not kill the run
        logger.warning(f"failed to build footage segment from {source_path}: {e}")
        return ""
    finally:
        for c in (work, clip):
            if c is not None:
                video.close_clip(c)


def assemble(combined_path: str, segments: List[Segment], threads: int = 2) -> str:
    """Concatenate segments in ``start`` order into ``combined_path``, or ""."""
    files = [s.file_path for s in sorted(segments, key=lambda s: s.start) if s.file_path]
    files = [f for f in files if os.path.exists(f) and os.path.getsize(f) > 0]
    if not files:
        logger.warning("hyperframes assemble: no usable segments")
        return ""

    output_dir = os.path.dirname(combined_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    if len(files) == 1:
        import shutil

        shutil.copy(files[0], combined_path)
        return combined_path
    try:
        video.concat_video_clips_with_ffmpeg(
            clip_files=files, output_file=combined_path,
            threads=threads, output_dir=output_dir,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"hyperframes assemble (concat) failed: {e}")
        return ""
    if os.path.exists(combined_path) and os.path.getsize(combined_path) > 0:
        logger.success(f"hyperframes assembled {len(files)} segment(s) -> {os.path.basename(combined_path)}")
        return combined_path
    return ""
