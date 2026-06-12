"""Fast preview pass: catch broken compositions before the heavy full render.

Rendering the final composition at 30fps is the slow part of the pipeline. Before
paying for it, we render a tiny **low-fps proxy** of the same authored HTML, pull
one representative frame per scene, and run cheap automated checks (a near-empty /
black frame is the usual failure). The frames are tiled into a single ``contact
sheet`` PNG so the result can be eyeballed in one glance.

Everything is best-effort and non-fatal: any problem returns a report with
``ok=True`` and an explanatory note so the caller never blocks the real render on a
preview hiccup. The proxy is faithful because it reuses the exact same renderer as
the final video -- just at a fraction of the frame rate.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple

from loguru import logger
from PIL import Image, ImageDraw, ImageStat

from app.config import config

from . import render
from .scenes import Scene

# A frame this dark *and* this flat (low colour spread) is empty -- a real scene,
# even on a dark gradient, has bright caption text that lifts the spread well above
# this. Tuned to flag black gaps without tripping on intentionally moody scenes.
_EMPTY_MEAN_LUMA = 26.0
_EMPTY_STDDEV = 9.0


@dataclass
class PreviewReport:
    """Outcome of a preview pass."""

    ok: bool
    issues: List[Tuple[int, str]] = field(default_factory=list)  # (scene_index, reason)
    contact_sheet: str = ""
    frames: List[str] = field(default_factory=list)
    note: str = ""


def _ffmpeg_binary() -> str:
    # Reuse the same resolution logic as the final encode so PATH/bundled ffmpeg
    # behave identically. Imported lazily to avoid a heavy import at module load.
    from app.services import video

    return video.get_ffmpeg_binary()


def _extract_frame(proxy: str, t: float, out_path: str) -> str:
    """Pull a single frame at ``t`` seconds from ``proxy`` into ``out_path``."""
    cmd = [
        _ffmpeg_binary(), "-y", "-ss", f"{max(t, 0):.3f}", "-i", proxy,
        "-frames:v", "1", "-q:v", "2", out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001 - extraction is best-effort
        logger.debug(f"preview frame extract error at {t:.2f}s: {e}")
        return ""
    if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return ""
    return out_path


def _is_near_empty(image_path: str) -> bool:
    """True when the frame is almost a flat dark fill (a black/empty gap)."""
    try:
        with Image.open(image_path) as im:
            stat = ImageStat.Stat(im.convert("RGB"))
    except Exception:  # noqa: BLE001
        return False
    mean = sum(stat.mean) / len(stat.mean)
    stddev = sum(stat.stddev) / len(stat.stddev)
    return mean < _EMPTY_MEAN_LUMA and stddev < _EMPTY_STDDEV


def _build_contact_sheet(frames: List[str], out_path: str, columns: int = 3) -> str:
    """Tile the per-scene frames into one labelled PNG for quick inspection."""
    thumbs = []
    for i, fp in enumerate(frames):
        if not fp or not os.path.exists(fp):
            continue
        try:
            im = Image.open(fp).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        im.thumbnail((360, 640))
        draw = ImageDraw.Draw(im)
        label = f"{i}"
        draw.rectangle([0, 0, 34, 24], fill=(0, 0, 0))
        draw.text((8, 5), label, fill=(255, 255, 255))
        thumbs.append(im)
    if not thumbs:
        return ""

    cols = max(1, min(columns, len(thumbs)))
    rows = (len(thumbs) + cols - 1) // cols
    cw = max(t.width for t in thumbs)
    ch = max(t.height for t in thumbs)
    pad = 8
    sheet = Image.new(
        "RGB", (cols * cw + pad * (cols + 1), rows * ch + pad * (rows + 1)), (17, 24, 39)
    )
    for idx, t in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = pad + c * (cw + pad)
        y = pad + r * (ch + pad)
        sheet.paste(t, (x, y))
    try:
        sheet.save(out_path)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"preview contact sheet save failed: {e}")
        return ""
    return out_path


def is_enabled() -> bool:
    """Config gate (default off -- the proxy render adds time to a run)."""
    return bool(config.app.get("hyperframes_preview", False))


def preview(html: str, scene_list: List[Scene], out_dir: str) -> PreviewReport:
    """Render a low-fps proxy, sample a frame per scene, flag near-empty frames.

    Returns a :class:`PreviewReport`. Never raises -- a failure to preview yields
    ``ok=True`` with a ``note`` so the caller proceeds to the real render.
    """
    if not scene_list:
        return PreviewReport(ok=True, note="no scenes to preview")

    fps = int(config.app.get("hyperframes_preview_fps", 3) or 3)
    proxy = os.path.join(out_dir, "preview-proxy.mp4")
    try:
        rendered = render.render(html, proxy, fps=fps)
    except Exception as e:  # noqa: BLE001
        return PreviewReport(ok=True, note=f"proxy render error: {e}")
    if not rendered:
        return PreviewReport(ok=True, note="proxy render produced nothing")

    frames: List[str] = []
    issues: List[Tuple[int, str]] = []
    for i, s in enumerate(scene_list):
        # Sample mid-scene so an entrance/exit animation isn't mistaken for a gap.
        t = s.start + s.duration / 2.0
        fp = _extract_frame(proxy, t, os.path.join(out_dir, f"preview-scene-{i}.png"))
        frames.append(fp)
        if fp and _is_near_empty(fp):
            issues.append((i, "near-empty/black frame"))

    sheet = _build_contact_sheet(frames, os.path.join(out_dir, "preview-contact-sheet.png"))
    return PreviewReport(ok=not issues, issues=issues, contact_sheet=sheet, frames=frames)
