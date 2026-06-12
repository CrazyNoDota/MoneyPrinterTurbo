"""Hyperframes motion-graphics video -- the pipeline entry points.

Hyperframes (https://github.com/heygen-com/hyperframes) renders HTML/CSS/JS +
GSAP into a deterministic MP4 via headless Chrome + ffmpeg. Unlike ``videogen``
(realistic b-roll from a diffusion API), this produces *synthetic motion graphics*
-- kinetic typography, animated numbers/lists -- now optionally over real photo
backgrounds pulled from the internet.

Four visual modes (see :func:`mode`):
- ``footage``     -- unchanged stock/local footage pipeline (hyperframes off).
- ``hyperframes`` -- "solely": the whole visual track is one authored composition
  timed to the narration (with on-demand photo backgrounds).
- ``mixed``       -- the "director": the LLM tags each scene footage vs motion
  graphics; footage scenes stay native, MG scenes are rendered and the segments
  are stitched in order.
- ``news``        -- deterministic news layout (headline plate + per-scene
  lower-third over a motion background) with an optional talking-head presenter
  (the ``avatar`` module) composited into a corner.

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

from . import assemble, assets, author, plan, preview, render, scenes

ClipScene = scenes.Scene

_VALID_MODES = ("footage", "hyperframes", "mixed", "news", "quiz", "ranking", "chat")

# The spoken countdown cue inserted between a quiz question and its answer. The
# TTS reads it (so the audio has a real ~2-3s beat), and scenes.from_script /
# the proportional timeline give it its own scene that the studio renders as the
# animated 3-2-1 visual. Kept short and digit-only so it reads as a countdown.
QUIZ_COUNTDOWN_CUE = "Three... two... one..."

# Minimum on-screen seconds for the animated countdown beat (the one hard-coded
# duration WP2 is allowed -- everything else is timed from real narration).
_COUNTDOWN_MIN_SECONDS = 2.4

# Invisible separator (U+2063) used to mark answer / rank narration so the studio
# composer can classify scenes without re-parsing the LLM JSON. It survives TTS
# (silent) and the SRT/scene round-trip (not whitespace, so not collapsed).
_MARK = "⁣"


def is_enabled(params=None) -> bool:
    """Back-compat flag: solely-hyperframes mode turned on (config or per-request)."""
    if params is not None and getattr(params, "hyperframes_enabled", False):
        return True
    return bool(config.app.get("hyperframes_enabled", False))


def mode(params=None) -> str:
    """Resolve the visual mode: footage | hyperframes | mixed | news."""
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
        count = int(config.app.get("hyperframes_bg_count", 2) or 2)
        for p in block:
            if p.use_background and p.query:
                backgrounds.extend(assets.fetch_backgrounds(p.query, params, source, count=count))

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
):
    """Director mode: plan each scene, render MG blocks, stitch with native footage.

    Returns ``(video_path, footage_ranges)`` where ``footage_ranges`` is a list of
    ``(start, end)`` seconds for scenes rendered as plain footage (no baked text).
    Captions should be burned only over those ranges -- motion-graphics scenes
    already render the narration as kinetic typography. ``("", [])`` on failure.
    """
    if not is_available():
        logger.warning("hyperframes enabled but toolchain not installed; run setup-hyperframes.bat")
        return "", []

    total = float(audio_duration or 0)
    scene_list = scenes.build_scenes(video_script, subtitle_path, total)
    if not scene_list:
        logger.warning("hyperframes director: no scenes; falling back to stock footage")
        return "", []

    width, height = _resolution(params)
    subject = (getattr(params, "video_subject", "") or "").strip()
    bg_source = _image_source(params)
    footage_source = (getattr(params, "video_source", "") or "pexels").strip().lower()
    if footage_source not in ("pexels", "pixabay"):
        footage_source = "pexels"

    plans = plan.build_plan(scene_list, subject, video_terms, material_hints)
    assets.reset()

    segments: List[assemble.Segment] = []
    # Scenes shown as plain footage (no baked text) -- captions belong only here.
    footage_ranges: List[tuple] = []
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
                        footage_ranges.append((p.scene.start, p.scene.end))
            i = j
        else:
            fs = _footage_segment(task_id, params, plans[i], width, height, footage_source, idx)
            idx += 1
            if fs:
                segments.append(assemble.Segment(start=plans[i].scene.start, file_path=fs))
                footage_ranges.append((plans[i].scene.start, plans[i].scene.end))
            i += 1

    if not segments:
        logger.warning("hyperframes director: produced no segments; falling back to stock footage")
        return "", []

    combined = os.path.join(utils.task_dir(task_id), "combined-1.mp4")
    out = assemble.assemble(combined, segments, threads=getattr(params, "n_threads", 2))
    return (out, footage_ranges) if out else ("", [])


def _news_presenter(task_id, params, video_script, audio_file, width, height) -> str:
    """Best-effort talking-head clip for the news corner overlay, or ``""``.

    The Azure path synthesizes from the script text (TTS lives inside the avatar
    request -- keep ``avatar_voice`` matched to the pipeline TTS voice so lips
    track the narration). When that yields nothing, Wav2Lip lip-syncs the actual
    narration audio onto the configured portrait, which is exact by construction.
    """
    from app.services import avatar

    if not avatar.is_enabled():
        return ""

    # The presenter occupies a corner, so request a square clip -- a full-frame
    # portrait would scale into a towering overlay.
    side = min(int(width or 1080), int(height or 1920))
    voice = str(config.app.get("avatar_voice", "") or "")
    if not voice:
        # Follow the pipeline TTS voice (generate_audio records the voice it
        # actually used back onto params) so the avatar's lips track the
        # narration. Non-Azure voices (qwen:, silero:) yield "" -> provider default.
        from app.services import voice as voice_service

        voice = voice_service.azure_voice_basename(getattr(params, "voice_name", ""))
    use_alpha = bool(config.app.get("avatar_prefer_alpha", True)) and bool(
        config.app.get("avatar_alpha_supported", False)
    )
    ext = "webm" if use_alpha else "mp4"
    out_path = os.path.join(utils.task_dir(task_id), f"presenter.{ext}")
    try:
        result = avatar.synthesize(
            video_script, presenter=voice, out_path=out_path, width=side, height=side
        )
    except Exception as exc:  # noqa: BLE001 - the head layer is always optional
        logger.warning(f"avatar synthesis failed (non-fatal): {exc}")
        result = ""
    if result:
        return result

    # Wav2Lip can't read a script; retry it with the narration audio when allowed.
    provider = str(config.app.get("avatar_provider", "auto") or "auto").strip().lower()
    if provider in ("", "auto", "wav2lip") and audio_file and os.path.isfile(audio_file):
        try:
            result = avatar.Wav2LipAvatar().synthesize(
                script_or_audio=audio_file,
                presenter="",
                out_path=os.path.join(utils.task_dir(task_id), "presenter.mp4"),
                width=side,
                height=side,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"wav2lip presenter failed (non-fatal): {exc}")
            result = ""
    return result or ""


def render_news_video(
    task_id, params, video_script, audio_file, subtitle_path, audio_duration,
    video_terms=None,
):
    """News mode: deterministic news composition + optional presenter overlay.

    Returns ``(video_path, caption_ranges)``. When the presenter covers the video
    the ranges are ``[]`` (the head + lower-thirds carry the words -- burn no
    captions); when no head could be produced they span the whole video so the
    narration stays readable. ``("", [])`` on failure -> stock-footage fallback.
    """
    if not is_available():
        logger.warning("hyperframes news enabled but toolchain not installed; run setup-hyperframes.bat")
        return "", []

    total = float(audio_duration or 0)
    scene_list = scenes.build_scenes(video_script, subtitle_path, total)
    if not scene_list:
        logger.warning("hyperframes news: no scenes; falling back to stock footage")
        return "", []

    width, height = _resolution(params)
    subject = (getattr(params, "video_subject", "") or "").strip()

    assets.reset()
    backgrounds = _solely_backgrounds(params, video_terms, _image_source(params))

    html = author.author_news(scene_list, subject, width, height, backgrounds=backgrounds)
    if not html:
        logger.warning("hyperframes news: composition failed; falling back to stock footage")
        return "", []

    base = render.render(html, os.path.join(utils.task_dir(task_id), "news-base.mp4"))
    if not base:
        return "", []

    end = round(max(total, scene_list[-1].end), 3)
    presenter = _news_presenter(task_id, params, video_script, audio_file, width, height)
    if presenter:
        out = assemble.overlay_presenter(
            base, presenter, os.path.join(utils.task_dir(task_id), "news.mp4"),
            width, height, threads=getattr(params, "n_threads", 2),
        )
        if out:
            return out, []
        logger.warning("news presenter overlay failed; continuing without the talking head")
    return base, [(0.0, end)]


def quiz_narration(quiz: dict) -> str:
    """Build the continuous narration script for a quiz.

    Per question: the question, the spoken 3-2-1 countdown cue, then the answer
    (and fun fact). The answer is prefixed with the invisible marker so the
    studio can tell answer scenes from question scenes after the SRT round-trip.
    """
    parts = []
    for item in (quiz or {}).get("questions", []):
        q = (item.get("q") or "").strip()
        a = (item.get("a") or "").strip()
        if not q or not a:
            continue
        fact = (item.get("fun_fact") or "").strip()
        answer = f"{_MARK}{a}"
        if fact:
            answer = f"{answer} — {fact}"
        parts.extend([q, QUIZ_COUNTDOWN_CUE, answer])
    return "\n".join(parts)


def ranking_narration(ranking: dict) -> str:
    """Build the continuous narration for a Top-N ranking (title, then #N..#1).

    Each ranked line is marked with ``<MARK><rank><MARK>`` so the studio renders
    the rank badge; the title line carries no marker.
    """
    title = (ranking or {}).get("title", "").strip()
    parts = [title] if title else []
    for item in (ranking or {}).get("items", []):
        name = (item.get("name") or "").strip()
        if not name:
            continue
        rank = item.get("rank")
        reason = (item.get("reason") or "").strip()
        body = f"{name} — {reason}" if reason else name
        parts.append(f"{_MARK}{rank}{_MARK}{body}")
    return "\n".join(parts)


def _quiz_scene_list(quiz: dict, total: float) -> List[scenes.Scene]:
    """Time a quiz's question/countdown/answer scenes across ``total`` seconds.

    The countdown beats are given a fixed minimum window; the rest of the audio
    is split between question and answer scenes proportionally to their text
    length (a robust stand-in for SRT cue boundaries that always aligns scene
    text with what is being narrated).
    """
    rows = []  # (text, weight, fixed_seconds_or_None)
    for item in (quiz or {}).get("questions", []):
        q = (item.get("q") or "").strip()
        a = (item.get("a") or "").strip()
        if not q or not a:
            continue
        fact = (item.get("fun_fact") or "").strip()
        answer = f"{_MARK}{a}" + (f" — {fact}" if fact else "")
        rows.append((q, max(len(q), 1), None))
        rows.append((QUIZ_COUNTDOWN_CUE, 0, _COUNTDOWN_MIN_SECONDS))
        rows.append((answer, max(len(answer), 1), None))
    return _distribute(rows, total)


def _ranking_scene_list(ranking: dict, total: float) -> List[scenes.Scene]:
    """Time a ranking's title + ranked-item scenes across ``total`` seconds."""
    rows = []
    title = (ranking or {}).get("title", "").strip()
    if title:
        rows.append((title, max(len(title), 1), None))
    for item in (ranking or {}).get("items", []):
        name = (item.get("name") or "").strip()
        if not name:
            continue
        reason = (item.get("reason") or "").strip()
        body = f"{name} — {reason}" if reason else name
        text = f"{_MARK}{item.get('rank')}{_MARK}{body}"
        rows.append((text, max(len(text), 1), None))
    return _distribute(rows, total)


def chat_narration(story: dict) -> str:
    """Build the single continuous narration for a messenger chat story.

    One TTS voice reads the whole exchange as "Name: text" lines, so the listener
    can follow who is speaking (a single-track read is what the WP2 proportional
    timing model supports). Each line is prefixed with the invisible
    ``<MARK><side><MARK>`` marker so the studio can recover each message's side
    (left/right bubble) after the TTS/SRT round-trip without re-parsing the JSON.

    NOTE (future enhancement): a true two-voice read (alternating TTS voices per
    speaker, durations measured per message and concatenated) would feel less
    robotic. It needs per-segment audio assembly the current timing model does
    not yet do; tracked in docs/VIRAL_UPGRADE_PLAN.md (WP5).
    """
    persons = (story or {}).get("persons") or ["A", "B"]
    parts = []
    for msg in (story or {}).get("messages", []):
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        side = 1 if msg.get("from") else 0
        name = (persons[side] if side < len(persons) else "") or ""
        spoken = f"{name}: {text}" if name else text
        parts.append(f"{_MARK}{side}{_MARK}{spoken}")
    return "\n".join(parts)


def _chat_scene_list(story: dict, total: float) -> List[scenes.Scene]:
    """Time a chat story's per-message scenes across ``total`` seconds.

    One scene per message, weighted by the spoken text length (the same
    proportional model the quiz/ranking lists use). Each scene's text carries the
    ``<MARK><side><MARK>`` prefix so the composer renders the correct bubble side.
    """
    persons = (story or {}).get("persons") or ["A", "B"]
    rows = []
    for msg in (story or {}).get("messages", []):
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        side = 1 if msg.get("from") else 0
        name = (persons[side] if side < len(persons) else "") or ""
        spoken = f"{name}: {text}" if name else text
        marked = f"{_MARK}{side}{_MARK}{text}"
        rows.append((marked, max(len(spoken), 1), None))
    return _distribute(rows, total)


def _distribute(rows, total: float) -> List[scenes.Scene]:
    """Lay ``rows`` (text, weight, fixed_seconds) on a contiguous timeline.

    Fixed-duration rows (the countdown beats) take their seconds off the top;
    the remainder is split among the weighted rows proportionally to their text.
    """
    if not rows or total <= 0:
        return []
    fixed_total = sum(f for _, _, f in rows if f)
    flexible = max(total - fixed_total, 0.01)
    weight_total = sum(w for _, w, f in rows if not f) or 1
    out: List[scenes.Scene] = []
    cursor = 0.0
    flexible_seen = 0.0
    flex_rows = [r for r in rows if not r[2]]
    for text, weight, fixed in rows:
        if fixed:
            dur = fixed
        else:
            flexible_seen += 1
            if flexible_seen == len(flex_rows):
                # absorb rounding drift into the last flexible scene
                dur = max(total - cursor, 0.01)
            else:
                dur = max(flexible * (weight / weight_total), 0.01)
        out.append(scenes.Scene(text=text, start=round(cursor, 3), duration=round(dur, 3)))
        cursor += dur
    return out


def render_quiz_video(
    task_id, params, quiz, audio_file, subtitle_path, audio_duration,
    video_terms=None,
):
    """Quiz mode: question -> 3-2-1 countdown -> answer reveal, per question.

    Returns ``(video_path, caption_ranges)``; ``caption_ranges`` is ``[]`` (the
    cards carry all the text, so no burned captions). ``("", [])`` on any
    problem -> stock-footage fallback. ``quiz`` is the structured dict from
    ``llm.generate_quiz``.
    """
    if not is_available():
        logger.warning("hyperframes quiz enabled but toolchain not installed; run setup-hyperframes.bat")
        return "", []
    total = float(audio_duration or 0)
    scene_list = _quiz_scene_list(quiz, total)
    if not scene_list:
        logger.warning("hyperframes quiz: no scenes; falling back to stock footage")
        return "", []
    width, height = _resolution(params)
    subject = (getattr(params, "video_subject", "") or "").strip()
    assets.reset()
    backgrounds = _solely_backgrounds(params, video_terms, _image_source(params))
    from . import studio

    html = studio.compose_quiz(scene_list, subject, width, height, total, backgrounds=backgrounds)
    if not html:
        logger.warning("hyperframes quiz: composition failed; falling back to stock footage")
        return "", []
    out = render.render(html, os.path.join(utils.task_dir(task_id), "quiz.mp4"))
    return (out, []) if out else ("", [])


def render_ranking_video(
    task_id, params, ranking, audio_file, subtitle_path, audio_duration,
    video_terms=None,
):
    """Ranking mode: "Top N" countdown #N -> #1 with slamming rank badges.

    Returns ``(video_path, caption_ranges)`` (``[]`` -> cards carry the text).
    ``("", [])`` on any problem. ``ranking`` is the dict from
    ``llm.generate_ranking``.
    """
    if not is_available():
        logger.warning("hyperframes ranking enabled but toolchain not installed; run setup-hyperframes.bat")
        return "", []
    total = float(audio_duration or 0)
    scene_list = _ranking_scene_list(ranking, total)
    if not scene_list:
        logger.warning("hyperframes ranking: no scenes; falling back to stock footage")
        return "", []
    width, height = _resolution(params)
    subject = (getattr(params, "video_subject", "") or "").strip()
    title = (ranking or {}).get("title", "")
    assets.reset()
    backgrounds = _ranking_backgrounds(params, ranking, video_terms, _image_source(params))
    from . import studio

    html = studio.compose_ranking(
        scene_list, subject, width, height, total, backgrounds=backgrounds, title=title
    )
    if not html:
        logger.warning("hyperframes ranking: composition failed; falling back to stock footage")
        return "", []
    out = render.render(html, os.path.join(utils.task_dir(task_id), "ranking.mp4"))
    return (out, []) if out else ("", [])


def render_chat_video(
    task_id, params, story, audio_file, subtitle_path, audio_duration,
    video_terms=None,
):
    """Chat mode: a messenger-style two-person story, bubbles popping in synced.

    Returns ``(video_path, caption_ranges)``; ``caption_ranges`` is ``[]`` (the
    bubbles carry all the text, so no burned captions). ``("", [])`` on any
    problem -> stock-footage fallback. ``story`` is the dict from
    ``llm.generate_chat_story``.
    """
    if not is_available():
        logger.warning("hyperframes chat enabled but toolchain not installed; run setup-hyperframes.bat")
        return "", []
    total = float(audio_duration or 0)
    scene_list = _chat_scene_list(story, total)
    if not scene_list:
        logger.warning("hyperframes chat: no scenes; falling back to stock footage")
        return "", []
    width, height = _resolution(params)
    subject = (getattr(params, "video_subject", "") or "").strip()
    persons = (story or {}).get("persons") or []
    title = (story or {}).get("title", "")
    assets.reset()
    from . import studio

    html = studio.compose_chat(
        scene_list, subject, width, height, total, persons=persons, title=title
    )
    if not html:
        logger.warning("hyperframes chat: composition failed; falling back to stock footage")
        return "", []
    out = render.render(html, os.path.join(utils.task_dir(task_id), "chat.mp4"))
    return (out, []) if out else ("", [])


def _ranking_backgrounds(params, ranking, video_terms, source):
    """One photo per ranking scene, item-matched.

    The title card gets a subject/terms photo and each rank card a photo of the
    item itself (a falcon card should not sit on a cheetah photo). The list is
    aligned 1:1 with ``_ranking_scene_list``'s scene order (same skip rules), so
    the composer's ``bgs[i % len(bgs)]`` mapping becomes the identity; a failed
    fetch falls back to the generic photo (or ``None`` -> gradient base).
    """
    if not images_enabled():
        return []
    generic = _solely_backgrounds(params, video_terms, source, cap=1)
    fallback = generic[0] if generic else None
    bgs = []
    title = (ranking or {}).get("title", "").strip()
    if title:
        bgs.append(fallback)
    for item in (ranking or {}).get("items", []):
        name = (item.get("name") or "").strip()
        if not name:
            continue
        fetched = assets.fetch_backgrounds(name, params, source, count=1)
        bgs.append(fetched[0] if fetched else fallback)
    return bgs if any(bgs) else []


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

    # Optional fast preview pass: render a low-fps proxy, flag broken (black /
    # empty) scenes, and retry the composition once before paying for the
    # full-rate render. Non-blocking by design.
    html = _preview_retry(
        task_id, html, scene_list,
        # A dud background photo is the usual cause of a near-empty frame; the
        # retry recomposes on the guaranteed gradient base (no photos). For the
        # freeform engine this is also simply a fresh authoring attempt.
        lambda: author.author_composition(scene_list, subject, width, height),
    )

    out_path = os.path.join(utils.task_dir(task_id), "hyperframes.mp4")
    return render.render(html, out_path)


def _preview_retry(task_id, html: str, scene_list, recompose) -> str:
    """Preview ``html``; when scenes are flagged, try ``recompose()`` once.

    Returns whichever variant previews cleaner (ties keep the original). Any
    preview/recompose hiccup returns the original ``html`` -- this loop can
    only ever improve the composition, never lose it.
    """
    if not preview.is_enabled():
        return html
    try:
        report = preview.preview(html, scene_list, utils.task_dir(task_id))
    except Exception as exc:  # noqa: BLE001 - preview must never block the render
        logger.warning(f"hyperframes preview pass failed (non-fatal): {exc}")
        return html
    if not report.issues:
        logger.info(
            f"hyperframes preview clean ({len(scene_list)} scenes). "
            f"Contact sheet: {report.contact_sheet or 'n/a'}"
        )
        return html
    logger.warning(
        f"hyperframes preview flagged {len(report.issues)} scene(s): "
        f"{report.issues}. Contact sheet: {report.contact_sheet}. Retrying composition."
    )
    try:
        retry_html = recompose() or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"hyperframes preview retry recompose failed: {exc}")
        return html
    if not retry_html or retry_html == html:
        return html
    try:
        retry_report = preview.preview(retry_html, scene_list, utils.task_dir(task_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"hyperframes preview of retry failed: {exc}")
        return html
    if len(retry_report.issues) < len(report.issues):
        logger.info(
            f"hyperframes preview retry improved: {len(report.issues)} -> "
            f"{len(retry_report.issues)} flagged scene(s); using the retry"
        )
        return retry_html
    logger.info("hyperframes preview retry did not improve; keeping the original")
    return html


__all__ = [
    "is_enabled", "mode", "is_available", "images_enabled",
    "render_video", "render_directed_video", "render_news_video",
    "render_quiz_video", "render_ranking_video", "render_chat_video",
    "quiz_narration", "ranking_narration", "chat_narration", "ClipScene",
]
