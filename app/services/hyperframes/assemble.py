"""Stitch a directed video: native footage segments + rendered MG blocks.

Each segment is normalized to exactly the target resolution and ``fps`` (so the
ffmpeg concat demuxer stays happy), then concatenated **in scene order**. Footage
sources are real clips (subclipped/looped to the scene length) or stock photos
(turned into a Ken-Burns segment). Reuses the proven encode/concat helpers in
``app.services.video``.
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger
from moviepy import CompositeVideoClip, ImageClip, concatenate_videoclips

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
            # Still photo -> gentle Ken-Burns zoom, framed to the target aspect.
            base = video.normalize_to_aspect(
                ImageClip(source_path).with_duration(duration), width, height
            )
            zoom = base.resized(lambda t: 1 + 0.03 * (t / max(duration, 0.01)))
            work = CompositeVideoClip(
                [zoom.with_position("center")], size=(width, height)
            ).with_duration(duration)
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
            work = video.normalize_to_aspect(work, width, height)

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
