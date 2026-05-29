"""Build a timed scene list for a hyperframes composition.

A "scene" is one chunk of narration with a start time and a duration. We prefer
the subtitle ``.srt`` (real, audio-aligned timing) when it exists; otherwise we
split the script into sentences and distribute them proportionally across the
audio duration -- the same character-length heuristic the non-edge TTS providers
use for captions.
"""

import re
from dataclasses import dataclass
from typing import List

from loguru import logger

from app.services import subtitle

# A scene shorter than this reads as a flash; merge tiny tail fragments instead.
_MIN_SCENE_SECONDS = 0.8


@dataclass
class Scene:
    """One timed line of the composition."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return round(self.start + self.duration, 3)


_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def _parse_srt_range(time_line: str):
    """Parse ``00:00:01,000 --> 00:00:05,000`` into ``(start_s, end_s)`` or None."""
    matches = _SRT_TIME.findall(time_line)
    if len(matches) < 2:
        return None

    def to_seconds(parts) -> float:
        h, m, s, ms = (int(x) for x in parts)
        return h * 3600 + m * 60 + s + ms / 1000.0

    return to_seconds(matches[0]), to_seconds(matches[1])


def from_subtitle(srt_path: str) -> List[Scene]:
    """Scenes from a subtitle file's real timing."""
    scenes: List[Scene] = []
    for _, time_line, text in subtitle.file_to_subtitles(srt_path):
        rng = _parse_srt_range(time_line)
        if not rng:
            continue
        start, end = rng
        text = " ".join((text or "").split())
        if text and end > start:
            scenes.append(Scene(text=text, start=round(start, 3), duration=round(end - start, 3)))
    return scenes


def from_script(script: str, total_duration: float) -> List[Scene]:
    """Scenes from the raw script, timed proportionally to sentence length."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(script or "") if s.strip()]
    if not sentences or total_duration <= 0:
        return []

    weights = [max(len(s), 1) for s in sentences]
    total_w = sum(weights)

    scenes: List[Scene] = []
    cursor = 0.0
    for i, (text, w) in enumerate(zip(sentences, weights)):
        # Absorb rounding drift into the last scene so the timeline sums exactly.
        if i == len(sentences) - 1:
            dur = total_duration - cursor
        else:
            dur = total_duration * (w / total_w)
        scenes.append(Scene(text=text, start=round(cursor, 3), duration=round(dur, 3)))
        cursor += dur
    return scenes


def build_scenes(script: str, srt_path: str, total_duration: float) -> List[Scene]:
    """Preferred entry point: subtitle timing first, script split as fallback."""
    scenes = from_subtitle(srt_path) if srt_path else []
    source = "subtitle timing"
    if not scenes:
        scenes = from_script(script, total_duration)
        source = "script (proportional timing)"

    if scenes:
        logger.info(f"hyperframes: {len(scenes)} scene(s) from {source}")
    return scenes
