"""LLM authoring of a hyperframes composition (a single ``index.html``).

The pipeline already has narration broken into timed :class:`~app.services.hyperframes.scenes.Scene`
objects. Here we ask the configured LLM (via :func:`app.services.llm._generate_response`)
to turn them into a self-contained, deterministic GSAP composition that obeys the
hyperframes contract, then validate the result. Authoring is best-effort: a
failure returns ``""`` and the caller falls back to stock footage.
"""

import re
from typing import List, Tuple

from loguru import logger

from app.config import config
from app.services import llm

from . import studio
from .scenes import Scene


def _engine() -> str:
    """Which authoring engine to use: 'studio' (deterministic templates) or 'freeform'."""
    e = (config.app.get("hyperframes_engine", "studio") or "studio").strip().lower()
    return e if e in ("studio", "freeform") else "studio"

# Hard rules the renderer enforces (mirrors .hyperframes/CLAUDE.md). Kept in the
# prompt AND re-checked by _validate so a chatty/al model can't break the render.
_CONTRACT = """\
HYPERFRAMES COMPOSITION CONTRACT (follow exactly):
- Output ONE complete HTML document and NOTHING else. No markdown fences, no prose.
- <html> must include the GSAP CDN:
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
- A single root wrapper:
  <div id="root" data-composition-id="main" data-start="0" data-duration="{total}"
       data-width="{width}" data-height="{height}"> ... </div>
- Every visible, timed element MUST have: class="clip", data-start, data-duration,
  data-track-index (integer z-order). Place each scene's text in its own clip whose
  data-start/data-duration match the scene timing given below.
- Register a PAUSED GSAP timeline on window.__timelines["main"]:
    window.__timelines = window.__timelines || {};
    const tl = gsap.timeline({ paused: true });
    /* tl.from(...) / tl.to(...) tweens */
    window.__timelines["main"] = tl;
- DETERMINISTIC ONLY: no Date.now(), no Math.random(), no fetch()/XHR. Self-contained
  inline CSS only. The ONLY external references allowed are the GSAP CDN above and the
  local files listed under AVAILABLE ASSETS (referenced as src="assets/<file>").
- body must be exactly {width}px by {height}px, overflow hidden, dark background.

NO BLACK FRAMES (critical):
- The VERY FIRST child of #root must be a permanent full-frame background layer that
  is visible for the WHOLE video: <div class="clip" data-start="0"
  data-duration="{total}" data-track-index="0"> styled position:absolute; inset:0;
  with a rich dark gradient (e.g. linear-gradient(160deg,#0f172a,#020617)). Never
  fade this layer to nothing -- something must always fill the frame, so a gap
  between scenes can never show black.
- Photo backgrounds (see BACKGROUNDS) sit ABOVE this base layer but are optional;
  the base gradient guarantees the frame is never empty.

ONE CENTERED TEXT COLUMN PER SCENE (critical -- prevents overlapping text):
- Each scene's text lives in ONE clip that is a vertically + horizontally CENTERED
  flex column: position:absolute; inset:0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; text-align:center; padding:0 7%.
- ALL of that scene's words -- the big number/percentage/stat AND the caption -- go
  INSIDE that one centered column as stacked children (e.g. a large number on top,
  the sentence below it). NEVER position a number or counter with absolute top/left
  coordinates that can land on top of the caption. Stacking in one flex column makes
  overlap impossible.
- Exactly ONE scene's text column is visible at any instant. Scenes must NOT overlap
  in time: scene N's text must finish animating OUT before scene N+1 animates IN
  (set its opacity to 0 / autoAlpha:0 by its data-start+data-duration). Use the
  scene data-start/data-duration EXACTLY as given below.
- Typographic safety: heading font-size clamp(40px, 8vw, 110px); line-height 1.12;
  max-width 86%; word-wrap:break-word. The big number can be larger
  (clamp(90px, 22vw, 320px)) but keep it on its OWN line above the caption with
  margin-bottom so glyphs never touch the caption. Long lines must wrap, never clip.

BACKGROUNDS: when AVAILABLE ASSETS are listed, use a relevant photo as a full-bleed
  background for that scene -- an <img class="clip" src="assets/<file>"> on a
  data-track-index ABOVE the base layer (1) but BELOW the text column, covering the
  frame (object-fit:cover; width:100%; height:100%; position:absolute; inset:0), with
  a dark gradient/scrim overlay (e.g. a sibling div with
  background:linear-gradient(180deg,rgba(0,0,0,.35),rgba(0,0,0,.65))) so text stays
  legible. Match each background's data-start/data-duration to its scene. Only
  reference files that appear under AVAILABLE ASSETS -- never invent file names.

MOTION (required, and make it tasteful):
- The timeline must contain GSAP from()/to()/fromTo() tweens for every scene.
- Animate each scene IN, hold it readable for most of its duration, then animate it
  OUT before the next scene starts.
- Use smooth eases ("power3.out" for entrances, "power2.in" for exits), short
  durations (0.45-0.7s), and a subtle transform -- rise + fade
  (y:40 -> 0, autoAlpha:0 -> 1) or a gentle scale (0.92 -> 1). Stagger the words /
  lines of a caption by ~0.05s for a kinetic feel.
- Numbers/percentages should COUNT UP: tween a JS object {v:0} to the target and
  write Math.round(v) into the element on each onUpdate (deterministic, no random).
- Give the background a slow continuous drift (scale 1.0 -> 1.08 across its scene)
  for depth. Do not return a static layout with only timeline registration.

STYLE: modern, premium short-form motion-graphics for a "{subject}" video -- bold
kinetic typography, confident spacing, real photo backgrounds where provided. One
scene visible at a time, animated in and out with GSAP. Aim for a clean,
production-ready TikTok look -- never two captions stacked on each other, never a
blank/black moment.
"""


def _scene_table(scenes: List[Scene]) -> str:
    lines = ["SCENES (index | start s | duration s | text):"]
    for i, s in enumerate(scenes):
        lines.append(f"  {i} | {s.start:.2f} | {s.duration:.2f} | {s.text}")
    return "\n".join(lines)


def _assets_block(backgrounds) -> str:
    if not backgrounds:
        return ""
    lines = ["AVAILABLE ASSETS (use as full-bleed scene backgrounds via src=\"assets/<file>\"):"]
    for b in backgrounds:
        lines.append(f"  assets/{_asset_name(b)} -- {_asset_desc(b)}")
    return "\n".join(lines)


def _asset_name(b) -> str:
    # Accept either a Background dataclass (filename "assets/x.jpg") or a bare name.
    name = getattr(b, "filename", b)
    return str(name).split("/")[-1]


def _asset_desc(b) -> str:
    return str(getattr(b, "description", "") or "")


def _build_prompt(scenes: List[Scene], subject: str, width: int, height: int, total: float, feedback: str = "", backgrounds=None) -> str:
    # .replace (not .format) because the contract contains literal JS braces.
    contract = (
        _CONTRACT.replace("{total}", str(round(total, 3)))
        .replace("{width}", str(width))
        .replace("{height}", str(height))
        .replace("{subject}", subject or "video")
    )
    parts = [
        "You are a motion-graphics author. Produce a hyperframes HTML composition.",
        contract,
        _scene_table(scenes),
        _assets_block(backgrounds),
        f"\nThe root data-duration must equal {round(total, 3)} seconds.",
    ]
    if feedback:
        parts.append(f"\nYour previous attempt was rejected: {feedback}\nReturn a corrected full HTML document.")
    parts.append("\nReturn ONLY the HTML document now.")
    return "\n\n".join(p for p in parts if p)


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text)
        text = re.sub(r"```$", "", text).strip()
    # Trim any prose the model prepended before the document's first tag.
    lower = text.lower()
    positions = [p for p in (lower.find("<!doctype"), lower.find("<html")) if p >= 0]
    if positions:
        idx = min(positions)
        if idx > 0:
            return text[idx:].strip()
    return text


_FORBIDDEN = ("math.random", "date.now(", "fetch(", "xmlhttprequest")
_ASSET_REF = re.compile(r'(?:src|href)\s*=\s*["\']assets/([^"\']+)["\']')
_TWEEN = re.compile(r"\b(?:tl|timeline|gsap)\s*\.\s*(?:fromTo|from|to)\s*\(")


def _validate(html: str, scene_count: int, total: float, asset_files=None) -> Tuple[bool, str]:
    """Cheap structural check. Returns ``(ok, reason)``; reason guides the retry."""
    if not html or "Error: " in html[:64]:
        return False, "empty or error response"
    low = html.lower()
    if "<html" not in low or "</html>" not in low:
        return False, "missing <html>...</html>"
    if 'data-composition-id="main"' not in html:
        return False, 'missing root data-composition-id="main"'
    if "window.__timelines" not in html:
        return False, "missing window.__timelines registration"
    if "gsap" not in low:
        return False, "missing GSAP"
    clips = len(re.findall(r'class="[^"]*\bclip\b', html))
    if clips < 1:
        return False, "no elements with class=\"clip\""
    # Expect roughly one clip per scene (allow extra decorative clips, not fewer).
    if clips < scene_count:
        return False, f"only {clips} clip(s) for {scene_count} scene(s); each scene needs its own clip"
    for bad in _FORBIDDEN:
        if bad in low:
            return False, f"forbidden non-deterministic call: {bad}"
    if "data-duration" not in html:
        return False, "clips missing data-duration"
    # A persistent full-frame layer (a clip spanning ~the whole video) guarantees the
    # frame is never black between scenes. Require one whose data-duration ~= total.
    if total > 0:
        durations = [float(d) for d in re.findall(r'data-duration="([0-9.]+)"', html)]
        if not durations or max(durations) < total * 0.9:
            return False, (
                "no persistent full-frame background layer; add a base clip with "
                f'data-start="0" data-duration="{round(total, 3)}" so the frame is '
                "never black between scenes"
            )
    tweens = len(_TWEEN.findall(html))
    if tweens < scene_count:
        return False, (
            f"only {tweens} GSAP tween(s) for {scene_count} scene(s); "
            "animate every scene with from/to/fromTo"
        )
    # Any referenced asset must actually have been staged, or the render 404s.
    allowed = {str(a).split("/")[-1] for a in (asset_files or [])}
    for ref in _ASSET_REF.findall(html):
        if ref.split("/")[-1] not in allowed:
            return False, f"references assets/{ref} which was not provided; only use listed AVAILABLE ASSETS"
    return True, ""


def author_composition(scenes: List[Scene], subject: str, width: int, height: int, backgrounds=None) -> str:
    """Author + validate a composition. Returns full HTML, or ``""`` on failure.

    ``backgrounds`` is an optional list of staged assets (Background dataclass or
    bare file names) the model may reference as ``assets/<file>``. One retry with
    the validation error fed back to the model.
    """
    if not scenes:
        return ""
    total = max((s.end for s in scenes), default=0.0)
    if total <= 0:
        return ""

    asset_files = [_asset_name(b) for b in (backgrounds or [])]

    # Preferred path: the deterministic studio engine. It composes a hardened
    # layout (no overlap, no black frames, consistent animation) without an LLM in
    # the layout loop. Still run it through _validate as a contract safety net; on
    # any miss we fall through to the freeform LLM author below.
    if _engine() == "studio":
        html = studio.compose(scenes, subject, width, height, total, backgrounds=backgrounds)
        if html:
            ok, reason = _validate(html, len(scenes), total, asset_files)
            if ok:
                logger.success(f"hyperframes: studio composition ({len(html)} chars)")
                return html
            logger.warning(f"hyperframes studio output rejected ({reason}); trying freeform author")
        else:
            logger.warning("hyperframes studio produced nothing; trying freeform author")

    feedback = ""
    for attempt in (1, 2):
        prompt = _build_prompt(scenes, subject, width, height, total, feedback, backgrounds)
        try:
            raw = llm._generate_response(prompt=prompt)
        except Exception as exc:  # noqa: BLE001 - authoring is best-effort
            logger.warning(f"hyperframes authoring LLM call failed (attempt {attempt}): {exc}")
            return ""
        html = _strip_fences(raw)
        ok, reason = _validate(html, len(scenes), total, asset_files)
        if ok:
            logger.success(f"hyperframes: composition authored on attempt {attempt} ({len(html)} chars)")
            return html
        logger.warning(f"hyperframes: composition rejected (attempt {attempt}): {reason}")
        feedback = reason
    return ""


def author_block(scenes: List[Scene], subject: str, width: int, height: int, backgrounds=None) -> str:
    """Author a composition for one contiguous block of MG scenes.

    The caller re-bases the block's scene start times to begin at 0 so the block
    renders as a standalone segment; otherwise identical to ``author_composition``.
    """
    return author_composition(scenes, subject, width, height, backgrounds=backgrounds)
