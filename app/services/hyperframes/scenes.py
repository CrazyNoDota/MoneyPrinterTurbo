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

# A scene shorter than this reads as a flash in short-form video; merge tiny
# fragments so the visual track has time to breathe.
_MIN_SCENE_SECONDS = 1.2


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


def _merge_short_scenes(items: List[Scene]) -> List[Scene]:
    """Merge very short subtitle fragments into their nearest neighbor."""
    merged: List[Scene] = []
    for s in items:
        if not merged:
            merged.append(s)
            continue
        if s.duration < _MIN_SCENE_SECONDS:
            prev = merged[-1]
            merged[-1] = Scene(
                text=f"{prev.text} {s.text}".strip(),
                start=prev.start,
                duration=round((s.start + s.duration) - prev.start, 3),
            )
        else:
            merged.append(s)
    return merged


def _cover_total_duration(items: List[Scene], total_duration: float) -> List[Scene]:
    """Normalize scenes into a contiguous timeline that reaches the audio end."""
    if not items or total_duration <= 0:
        return items

    source = sorted(
        [s for s in items if s.text and s.duration > 0 and s.start < total_duration],
        key=lambda s: s.start,
    )
    if not source:
        return []

    normalized: List[Scene] = []
    cursor = 0.0
    for i, s in enumerate(source):
        if i + 1 < len(source):
            target_end = min(max(s.end, source[i + 1].start), total_duration)
        else:
            target_end = total_duration
        expanded_duration = max(s.duration, target_end - s.start)
        normalized.append(
            Scene(
                text=s.text,
                start=round(cursor, 3),
                duration=round(max(expanded_duration, 0.01), 3),
            )
        )
        cursor += expanded_duration

    total = sum(s.duration for s in normalized)
    drift = total_duration - total
    if drift > 0.01:
        last = normalized[-1]
        normalized[-1] = Scene(
            text=last.text,
            start=last.start,
            duration=round(last.duration + drift, 3),
        )
    elif drift < -0.01:
        scale = total_duration / max(total, 0.01)
        cursor = 0.0
        scaled = []
        for s in normalized:
            dur = round(max(s.duration * scale, 0.01), 3)
            scaled.append(Scene(text=s.text, start=round(cursor, 3), duration=dur))
            cursor += dur
        normalized = scaled

    return _merge_short_scenes(normalized)


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
    else:
        scenes = _cover_total_duration(scenes, total_duration)

    if scenes:
        logger.info(
            f"hyperframes: {len(scenes)} scene(s) from {source}, covers {scenes[-1].end:.2f}s"
        )
    return scenes
