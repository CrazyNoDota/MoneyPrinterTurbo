"""Deterministic template engine for hyperframes compositions (the "studio").

Asking the LLM to author a whole freeform HTML/CSS/GSAP document gave us
overlapping text, black gaps, and inconsistent animation -- the model was in the
layout loop. The studio engine takes the model *out* of that loop: it classifies
each timed :class:`~app.services.hyperframes.scenes.Scene` into a small set of
hand-built **archetypes** and emits the composition from fixed, hardened templates.

Guaranteed by construction (no model, no randomness):
- ONE centered text column per scene -- a number and its caption are stacked in the
  same flex column, so they can never collide.
- a persistent full-frame gradient layer -- the frame is never black between scenes.
- tasteful, consistent GSAP entrances/exits and number count-ups.
- real photo backgrounds (when provided) with a legibility scrim.

``compose`` is best-effort: any problem returns ``""`` and the caller falls back to
the freeform author (and then to stock footage), so this can never hard-fail a run.
The output deliberately satisfies the same hyperframes contract the freeform path
is validated against.
"""

import html as _html
import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from .scenes import Scene

# GSAP, pinned to match the freeform contract / .hyperframes runtime.
_GSAP = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"

# Brand accent for the hero number (the gold the punchy caption style uses).
_ACCENT = "#ffd400"

# A "hero" number: optional currency, digits (with separators/decimals), optional
# magnitude/percent suffix. Used to decide stat-vs-statement and to drive count-ups.
# Single-letter magnitudes must be attached to the digits and NOT followed by more
# letters, so "5 men" / "5kids" don't become "5M" / "5K"; words/percent are matched
# attached too. A trailing \b is deliberately avoided -- it forced backtracking that
# dropped the "%" off "78%".
_HERO_NUM = re.compile(
    r"(?P<prefix>[$€£])?\s?"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?P<suffix>%|percent|bn|billion|million|trillion|[kmbtKMBT](?![a-zA-Z]))?",
    re.IGNORECASE,
)


@dataclass
class SceneSpec:
    """One scene mapped onto an archetype with its rendered content."""

    scene: Scene
    archetype: str  # "stat" | "statement"
    caption: str
    value: str = ""            # rendered hero string, e.g. "78%" or "$2.4B"
    number: Optional[float] = None  # numeric target for the count-up, if any
    prefix: str = ""
    suffix: str = ""
    decimals: int = 0
    background: Optional[object] = None  # a Background, or None


def _esc(text: str) -> str:
    return _html.escape((text or "").strip())


def _asset_name(b) -> str:
    name = getattr(b, "filename", b)
    return str(name).split("/")[-1] if name else ""


def _extract_hero(text: str):
    """Pick the most headline-worthy number in ``text``.

    Returns ``(value_str, number, prefix, suffix, decimals, span)`` or ``None``.
    Prefers percentages and currency, then larger magnitudes -- those are what read
    as a "stat". Plain years (1900-2099) on their own are ignored so a sentence like
    "between 2019 and 2023" doesn't become a stat card.
    """
    best = None
    best_score = -1.0
    for m in _HERO_NUM.finditer(text or ""):
        raw_num = m.group("num").replace(",", "")
        try:
            number = float(raw_num)
        except ValueError:
            continue
        prefix = m.group("prefix") or ""
        suffix = (m.group("suffix") or "")
        is_pct = suffix == "%" or suffix.lower() == "percent"
        is_year = (not prefix and not suffix and number.is_integer()
                   and 1900 <= number <= 2099)
        if is_year:
            continue
        score = number
        if is_pct:
            score += 1e6        # percentages are the strongest stat signal
        if prefix:
            score += 5e5        # money next
        if suffix and not is_pct:
            score += 1e5        # 2.4B / 50k etc.
        if score <= best_score:
            continue
        decimals = len(raw_num.split(".")[1]) if "." in raw_num else 0
        # Normalize the suffix's display form (keep "%" / uppercase magnitudes).
        disp_suffix = "%" if is_pct else suffix.upper() if suffix else ""
        value = f"{prefix}{m.group('num')}{disp_suffix}"
        best = (value, number, prefix, disp_suffix, decimals, m.span())
        best_score = score
    return best


def build_specs(scenes: List[Scene], backgrounds=None) -> List[SceneSpec]:
    """Classify every scene into an archetype and attach a rotating background."""
    bgs = list(backgrounds or [])
    specs: List[SceneSpec] = []
    for i, s in enumerate(scenes):
        hero = _extract_hero(s.text)
        bg = bgs[i % len(bgs)] if bgs else None
        if hero:
            value, number, prefix, suffix, decimals, _ = hero
            specs.append(SceneSpec(
                scene=s, archetype="stat", caption=s.text, value=value,
                number=number, prefix=prefix, suffix=suffix, decimals=decimals,
                background=bg,
            ))
        else:
            specs.append(SceneSpec(
                scene=s, archetype="statement", caption=s.text, background=bg,
            ))
    return specs


_CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
html, body { width:__W__px; height:__H__px; overflow:hidden;
  background:#020617; font-family:'Arial Black','Helvetica Neue Bold',Impact,system-ui,sans-serif; }
#root { position:relative; width:__W__px; height:__H__px; }
.clip { position:absolute; inset:0; }
.bg-base { background:linear-gradient(160deg,#0f172a 0%,#020617 70%,#000 100%); }
.bg-photo { width:100%; height:100%; object-fit:cover; will-change:transform; }
.scrim { background:linear-gradient(180deg,rgba(2,6,23,.35) 0%,rgba(2,6,23,.55) 45%,rgba(2,6,23,.8) 100%); }
.col { display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; padding:0 8%; gap:2.2vh; }
.stat-value { color:__ACCENT__; font-size:clamp(120px,26vw,360px); line-height:.92;
  letter-spacing:-.02em; text-shadow:0 8px 40px rgba(0,0,0,.55); }
.stat-caption { color:#fff; font-size:clamp(40px,7.2vw,96px); line-height:1.12;
  max-width:88%; text-shadow:0 3px 18px rgba(0,0,0,.7); }
.headline { color:#fff; font-size:clamp(52px,9vw,128px); line-height:1.08;
  letter-spacing:-.01em; max-width:90%; text-shadow:0 4px 22px rgba(0,0,0,.7); }
.headline .line { display:block; }
.line, .stat-value, .stat-caption { will-change:transform,opacity; }
"""


def _scene_html(spec: SceneSpec, idx: int) -> str:
    s = spec.scene
    start = round(s.start, 3)
    dur = round(s.duration, 3)
    parts = []

    if spec.background is not None:
        name = _asset_name(spec.background)
        if name:
            parts.append(
                f'<img id="bg-{idx}" class="clip bg-photo" data-start="{start}" '
                f'data-duration="{dur}" data-track-index="1" src="assets/{_esc(name)}">'
            )
            parts.append(
                f'<div class="clip scrim" data-start="{start}" data-duration="{dur}" '
                f'data-track-index="2"></div>'
            )

    if spec.archetype == "stat":
        body = (
            f'<div id="val-{idx}" class="line stat-value">{_esc(spec.prefix)}0'
            f'{_esc(spec.suffix)}</div>'
            f'<div class="line stat-caption">{_esc(spec.caption)}</div>'
        )
    else:
        body = (
            '<div class="headline">'
            + "".join(f'<span class="line">{_esc(w)}</span>' for w in _wrap_lines(spec.caption))
            + "</div>"
        )

    parts.append(
        f'<div id="scene-{idx}" class="clip col" data-start="{start}" '
        f'data-duration="{dur}" data-track-index="3">{body}</div>'
    )
    return "\n    ".join(parts)


def _wrap_lines(text: str, max_chars: int = 22) -> List[str]:
    """Greedy word-wrap so a headline breaks into a few balanced lines."""
    words = (text or "").split()
    lines, cur = [], ""
    for w in words:
        candidate = f"{cur} {w}".strip()
        if cur and len(candidate) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines or [text or ""]


def _scene_tweens(spec: SceneSpec, idx: int) -> str:
    s = spec.scene
    start = round(s.start, 3)
    end = round(s.end, 3)
    dur = round(s.duration, 3)
    # Hold the scene readable, then exit shortly before the next one begins.
    exit_at = round(max(start + 0.2, end - 0.45), 3)
    js = []
    if spec.background is not None and _asset_name(spec.background):
        js.append(
            f'tl.fromTo("#bg-{idx}",{{scale:1.0}},'
            f'{{scale:1.09,duration:{dur},ease:"none"}},{start});'
        )
    # Column in (rise + fade), staggered children for a kinetic feel.
    js.append(
        f'tl.fromTo("#scene-{idx} .line",{{autoAlpha:0,y:48}},'
        f'{{autoAlpha:1,y:0,duration:0.55,stagger:0.06,ease:"power3.out"}},{round(start + 0.08, 3)});'
    )
    if spec.archetype == "stat" and spec.number is not None:
        obj = f"c{idx}"
        fmt = (
            f"Math.round(o.v).toLocaleString('en-US')" if spec.decimals == 0
            else f"o.v.toFixed({spec.decimals})"
        )
        js.append(
            f'var {obj}={{v:0}};'
            f'tl.to({obj},{{v:{spec.number},duration:0.9,ease:"power1.out",'
            f'onUpdate:function(){{var o=this.targets()[0];'
            f'document.getElementById("val-{idx}").textContent='
            f'"{_js(spec.prefix)}"+({fmt})+"{_js(spec.suffix)}";}}}},{round(start + 0.12, 3)});'
        )
    # Exit (fade + slight lift) so nothing lingers into the next scene.
    js.append(
        f'tl.to("#scene-{idx}",{{autoAlpha:0,y:-28,duration:0.4,ease:"power2.in"}},{exit_at});'
    )
    return "\n    ".join(js)


def _js(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')


def compose(scenes: List[Scene], subject: str, width: int, height: int,
            total: float, backgrounds=None) -> str:
    """Compose a complete, contract-compliant composition. ``""`` on any problem."""
    try:
        if not scenes or total <= 0:
            return ""
        specs = build_specs(scenes, backgrounds)
        css = _CSS.replace("__W__", str(width)).replace("__H__", str(height)).replace("__ACCENT__", _ACCENT)
        total_r = round(total, 3)

        clips = [
            f'<div class="clip bg-base" data-start="0" data-duration="{total_r}" data-track-index="0"></div>'
        ]
        tweens = []
        for i, spec in enumerate(specs):
            clips.append(_scene_html(spec, i))
            tweens.append(_scene_tweens(spec, i))

        html = (
            "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
            f'<script src="{_GSAP}"></script>\n<style>{css}</style>\n</head>\n<body>\n'
            f'<div id="root" data-composition-id="main" data-start="0" '
            f'data-duration="{total_r}" data-width="{width}" data-height="{height}">\n  '
            + "\n  ".join(clips)
            + "\n</div>\n<script>\nwindow.__timelines = window.__timelines || {};\n"
            "const tl = gsap.timeline({ paused: true });\n    "
            + "\n    ".join(tweens)
            + '\nwindow.__timelines["main"] = tl;\n</script>\n</body>\n</html>\n'
        )
        logger.success(
            f"hyperframes studio: composed {len(specs)} scene(s) "
            f"({sum(1 for s in specs if s.archetype == 'stat')} stat, "
            f"{sum(1 for s in specs if s.archetype == 'statement')} statement)"
        )
        return html
    except Exception as exc:  # noqa: BLE001 - composition must never hard-fail a run
        logger.warning(f"hyperframes studio compose failed: {exc}")
        return ""
