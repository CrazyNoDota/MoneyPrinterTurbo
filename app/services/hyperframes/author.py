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

from app.services import llm

from .scenes import Scene

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
- Make text large and legible for vertical video; keep within safe margins
  (use max-width / responsive font-size so long lines never overflow the frame).
- BACKGROUNDS: when AVAILABLE ASSETS are listed, use a relevant photo as a full-bleed
  background behind the text for that scene -- an <img class="clip" src="assets/<file>">
  on a LOWER data-track-index than the text, covering the frame (object-fit: cover,
  width/height 100%), with a dark gradient/overlay so the text stays legible. Match each
  background's data-start/data-duration to the scene it belongs to. Only reference files
  that appear under AVAILABLE ASSETS -- never invent file names.

STYLE: modern motion-graphics for a short-form "{subject}" video -- kinetic
typography, bold headings, optional animated numbers/lists, real photo backgrounds
where provided. One scene visible at a time (sequential data-start/data-duration),
animated in and out with GSAP.
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
