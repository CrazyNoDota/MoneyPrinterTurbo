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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

from . import fonts
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
    archetype: str  # "stat" | "statement" | "quote" | "list" | "comparison" | "chart"
    caption: str
    value: str = ""            # rendered hero string, e.g. "78%" or "$2.4B"
    number: Optional[float] = None  # numeric target for the count-up, if any
    prefix: str = ""
    suffix: str = ""
    decimals: int = 0
    background: Optional[object] = None  # a Background, or None
    items: List[str] = field(default_factory=list)  # list / comparison entries
    attr: str = ""             # quote attribution ("" = none)
    bars: List[Tuple[str, float]] = field(default_factory=list)  # chart (label, pct)


def _esc(text: str) -> str:
    return _html.escape((text or "").strip())


def _fill_css(template: str, width: int, height: int) -> str:
    """Resolve the CSS template: dimensions, accent, embedded font faces.

    ``stage_fonts`` copies the bundled Anton/Oswald files into the project's
    assets dir and returns their @font-face rules; on any failure it returns
    ``""`` and the family stack degrades to the system fallbacks.
    """
    return (
        template.replace("__W__", str(width))
        .replace("__H__", str(height))
        .replace("__ACCENT__", _ACCENT)
        .replace("__FONTS__", fonts.stage_fonts())
        .replace("__FAMILY__", fonts.FAMILY)
    )


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


# --- archetype detectors (conservative: when unsure, fall back to stat/statement)

# Opening/closing quote characters around a quoted line, optional attribution
# after a dash: «Цитата» — Имя  /  "Quote" - Name.
_QUOTE_RE = re.compile(
    r'^\s*[«"“„‘\']\s*(?P<q>.{8,220}?)\s*[»"”“’\']'
    r"\s*(?:[—–-]\s*(?P<attr>.{2,48}))?\s*[.!]?\s*$",
    re.DOTALL,
)

# " vs " / " против " between two short halves -> a comparison scene.
_VS_RE = re.compile(r"\s+(?:vs\.?|versus|против)\s+", re.IGNORECASE)

# "1. item 2. item" / "1) item 2) item" enumeration markers.
_ENUM_RE = re.compile(r"(?:(?<=^)|(?<=[\s:;,—–-]))(\d{1,2})[.)]\s+")

_ITEM_MAX = 64  # an entry longer than this reads as prose, not a list row


def _match_quote(text: str) -> Optional[Tuple[str, str]]:
    """``(quote, attribution)`` when the whole scene is one quoted line."""
    m = _QUOTE_RE.match(text or "")
    if not m:
        return None
    return m.group("q").strip(), (m.group("attr") or "").strip()


def _match_comparison(text: str) -> Optional[List[str]]:
    """``[left, right]`` when the scene is two short halves around a "vs"."""
    parts = _VS_RE.split(text or "", maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = (p.strip(" \t,.;:!") for p in parts)
    if not left or not right:
        return None
    if len(left) > _ITEM_MAX or len(right) > _ITEM_MAX:
        return None
    return [left, right]


def _match_list(text: str) -> Optional[Tuple[str, List[str]]]:
    """``(title, items)`` for an enumerated/semicolon scene; title may be ""."""
    text = (text or "").strip()
    markers = _ENUM_RE.findall(text)
    if len(markers) >= 2:
        chunks = _ENUM_RE.split(text)
        # split() yields [lead-in, num, item, num, item, ...]
        title = chunks[0].strip(" \t,.;:—–-")
        items = [c.strip(" \t,.;:") for c in chunks[2::2]]
        items = [i for i in items if i]
        if 2 <= len(items) <= 5 and all(len(i) <= _ITEM_MAX for i in items):
            return (title if len(title) <= _ITEM_MAX else ""), items
        return None
    if text.count(";") >= 2:
        items = [p.strip(" \t,.;:") for p in text.split(";")]
        items = [i for i in items if i]
        if 3 <= len(items) <= 5 and all(len(i) <= _ITEM_MAX for i in items):
            return "", items
    return None


def _match_chart(text: str) -> Optional[List[Tuple[str, float]]]:
    """``[(label, percent), ...]`` when the scene compares 2+ percentages.

    Labels are the few words right before each number ("онлайн 78%, офлайн 22%"
    -> [("онлайн", 78), ("офлайн", 22)]); a missing label shows the value alone.
    """
    bars: List[Tuple[str, float]] = []
    last_end = 0
    for m in _HERO_NUM.finditer(text or ""):
        suffix = (m.group("suffix") or "").lower()
        if suffix not in ("%", "percent"):
            continue
        try:
            value = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
        if not 0 < value <= 100:
            continue
        lead = text[last_end:m.start()].strip(" \t,.;:—–-")
        label = " ".join(lead.split()[-3:])[:24]
        bars.append((label, value))
        last_end = m.end()
    if 2 <= len(bars) <= 4:
        return bars
    return None


def build_specs(scenes: List[Scene], backgrounds=None) -> List[SceneSpec]:
    """Classify every scene into an archetype and attach a rotating background."""
    bgs = list(backgrounds or [])
    specs: List[SceneSpec] = []
    for i, s in enumerate(scenes):
        bg = bgs[i % len(bgs)] if bgs else None
        quote = _match_quote(s.text)
        if quote:
            q, attr = quote
            specs.append(SceneSpec(
                scene=s, archetype="quote", caption=q, attr=attr, background=bg,
            ))
            continue
        comparison = _match_comparison(s.text)
        if comparison:
            specs.append(SceneSpec(
                scene=s, archetype="comparison", caption=s.text,
                items=comparison, background=bg,
            ))
            continue
        listed = _match_list(s.text)
        if listed:
            title, items = listed
            specs.append(SceneSpec(
                scene=s, archetype="list", caption=title, items=items, background=bg,
            ))
            continue
        bars = _match_chart(s.text)
        if bars:
            specs.append(SceneSpec(
                scene=s, archetype="chart", caption=s.text, bars=bars, background=bg,
            ))
            continue
        hero = _extract_hero(s.text)
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
__FONTS__
* { margin:0; padding:0; box-sizing:border-box; }
html, body { width:__W__px; height:__H__px; overflow:hidden;
  background:#020617; font-family:__FAMILY__; font-weight:700; }
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
.quote-mark { color:__ACCENT__; font-size:clamp(120px,22vw,300px); line-height:.6;
  height:.45em; }
.quote-text { color:#fff; font-size:clamp(44px,7.6vw,104px); line-height:1.16;
  max-width:88%; text-shadow:0 3px 18px rgba(0,0,0,.7); }
.quote-attr { color:__ACCENT__; font-size:clamp(28px,4.2vw,56px);
  letter-spacing:.12em; text-transform:uppercase; }
.list-title { color:__ACCENT__; font-size:clamp(34px,5vw,68px);
  letter-spacing:.1em; text-transform:uppercase; margin-bottom:1vh; }
.list-item { display:flex; align-items:center; gap:.55em; max-width:90%;
  text-align:left; }
.list-idx { flex:0 0 auto; display:flex; align-items:center; justify-content:center;
  width:1.5em; height:1.5em; border-radius:8px; background:__ACCENT__; color:#020617;
  font-size:clamp(30px,4.6vw,60px); }
.list-txt { color:#fff; font-size:clamp(34px,5.4vw,72px); line-height:1.14;
  text-shadow:0 3px 16px rgba(0,0,0,.7); }
.vs-row { display:flex; align-items:stretch; justify-content:center; gap:3%;
  width:100%; }
.vs-side { flex:1 1 0; display:flex; align-items:center; justify-content:center;
  padding:5% 4%; border-radius:14px; background:rgba(15,23,42,.78);
  box-shadow:0 10px 36px rgba(0,0,0,.45); color:#fff;
  font-size:clamp(34px,5.2vw,72px); line-height:1.14; }
.vs-badge { align-self:center; flex:0 0 auto; padding:.3em .5em; border-radius:10px;
  background:__ACCENT__; color:#020617; font-size:clamp(30px,4.6vw,60px);
  letter-spacing:.06em; }
.chart-caption { color:#fff; font-size:clamp(34px,5.2vw,68px); line-height:1.14;
  max-width:88%; text-shadow:0 3px 16px rgba(0,0,0,.7); margin-bottom:1.4vh; }
.chart-row { display:flex; flex-direction:column; align-items:flex-start; gap:.7vh;
  width:84%; }
.bar-head { display:flex; justify-content:space-between; width:100%; color:#fff;
  font-size:clamp(26px,3.8vw,50px); }
.bar-pct { color:__ACCENT__; }
.bar-track { width:100%; height:clamp(20px,2.6vw,34px); border-radius:999px;
  background:rgba(148,163,184,.25); overflow:hidden; }
.bar-fill { height:100%; border-radius:999px; background:__ACCENT__;
  transform-origin:left center; will-change:transform; }
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

    body = _scene_body(spec, idx)

    parts.append(
        f'<div id="scene-{idx}" class="clip col" data-start="{start}" '
        f'data-duration="{dur}" data-track-index="3">{body}</div>'
    )
    return "\n    ".join(parts)


def _scene_body(spec: SceneSpec, idx: int) -> str:
    """The children of a scene's single centered column, per archetype.

    Every archetype stays inside that one flex column with stacked children, so
    the no-overlap guarantee of the original layout extends to all of them.
    """
    if spec.archetype == "stat":
        return (
            f'<div id="val-{idx}" class="line stat-value">{_esc(spec.prefix)}0'
            f'{_esc(spec.suffix)}</div>'
            f'<div class="line stat-caption">{_esc(spec.caption)}</div>'
        )
    if spec.archetype == "quote":
        attr = f'<div class="line quote-attr">— {_esc(spec.attr)}</div>' if spec.attr else ""
        return (
            '<div class="line quote-mark">“</div>'
            f'<div class="line quote-text">{_esc(spec.caption)}</div>'
            + attr
        )
    if spec.archetype == "list":
        title = f'<div class="line list-title">{_esc(spec.caption)}</div>' if spec.caption else ""
        rows = "".join(
            f'<div class="line list-item"><span class="list-idx">{n}</span>'
            f'<span class="list-txt">{_esc(item)}</span></div>'
            for n, item in enumerate(spec.items, start=1)
        )
        return title + rows
    if spec.archetype == "comparison":
        left, right = spec.items[0], spec.items[1]
        return (
            f'<div class="line vs-row">'
            f'<div class="vs-side vs-left">{_esc(left)}</div>'
            f'<div class="vs-badge">VS</div>'
            f'<div class="vs-side vs-right">{_esc(right)}</div>'
            f'</div>'
        )
    if spec.archetype == "chart":
        rows = []
        for n, (label, value) in enumerate(spec.bars):
            pct = round(value, 1)
            pct_str = f"{pct:g}%"
            rows.append(
                f'<div class="line chart-row">'
                f'<div class="bar-head"><span>{_esc(label) or "&nbsp;"}</span>'
                f'<span class="bar-pct">{pct_str}</span></div>'
                f'<div class="bar-track"><div class="bar-fill" '
                f'style="width:{pct:g}%"></div></div></div>'
            )
        return (
            f'<div class="line chart-caption">{_esc(spec.caption)}</div>'
            + "".join(rows)
        )
    return (
        '<div class="headline">'
        + "".join(f'<span class="line">{_esc(w)}</span>' for w in _wrap_lines(spec.caption))
        + "</div>"
    )


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
    elif spec.archetype == "comparison":
        # The halves converge from opposite sides; the VS badge pops between them.
        js.append(
            f'tl.fromTo("#scene-{idx} .vs-left",{{x:-56}},'
            f'{{x:0,duration:0.55,ease:"power3.out"}},{round(start + 0.08, 3)});'
        )
        js.append(
            f'tl.fromTo("#scene-{idx} .vs-right",{{x:56}},'
            f'{{x:0,duration:0.55,ease:"power3.out"}},{round(start + 0.08, 3)});'
        )
        js.append(
            f'tl.fromTo("#scene-{idx} .vs-badge",{{scale:0.4}},'
            f'{{scale:1,duration:0.45,ease:"back.out(2)"}},{round(start + 0.3, 3)});'
        )
    elif spec.archetype == "chart":
        # Bars grow to their inline width (scaleX keeps the layout static).
        js.append(
            f'tl.fromTo("#scene-{idx} .bar-fill",{{scaleX:0}},'
            f'{{scaleX:1,duration:0.8,stagger:0.12,ease:"power2.out"}},{round(start + 0.25, 3)});'
        )
    elif spec.archetype == "quote":
        js.append(
            f'tl.fromTo("#scene-{idx} .quote-mark",{{scale:0.5}},'
            f'{{scale:1,duration:0.5,ease:"back.out(1.7)"}},{round(start + 0.08, 3)});'
        )
    # Exit (fade + slight lift) so nothing lingers into the next scene.
    js.append(
        f'tl.to("#scene-{idx}",{{autoAlpha:0,y:-28,duration:0.4,ease:"power2.in"}},{exit_at});'
    )
    return "\n    ".join(js)


def _js(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# "news" archetype: persistent headline plate + per-scene lower-third плашка.
# The lower-right corner is deliberately kept clear for the talking-head
# presenter overlay (see assemble.overlay_presenter).
# ---------------------------------------------------------------------------

_NEWS_CSS = """
__FONTS__
* { margin:0; padding:0; box-sizing:border-box; }
html, body { width:__W__px; height:__H__px; overflow:hidden;
  background:#020617; font-family:__FAMILY__; font-weight:700; }
#root { position:relative; width:__W__px; height:__H__px; }
.clip { position:absolute; inset:0; }
.bg-base { background:linear-gradient(160deg,#0f172a 0%,#020617 70%,#000 100%); }
.bg-photo { width:100%; height:100%; object-fit:cover; will-change:transform; }
.scrim { background:linear-gradient(180deg,rgba(2,6,23,.4) 0%,rgba(2,6,23,.5) 45%,rgba(2,6,23,.85) 100%); }
.headline-wrap { display:flex; align-items:flex-start; justify-content:center; padding:5% 5% 0; }
.headline-plate { display:flex; flex-direction:column; align-items:center; gap:1.2vh;
  max-width:92%; will-change:transform,opacity; }
.headline-tag { background:__ACCENT__; color:#020617; font-size:clamp(20px,2.6vw,34px);
  letter-spacing:.22em; padding:.35em .9em; border-radius:4px; }
.headline-text { color:#fff; font-size:clamp(40px,6.4vw,84px); line-height:1.1;
  text-align:center; text-shadow:0 4px 22px rgba(0,0,0,.7); }
.lt-wrap { display:flex; align-items:flex-end; justify-content:flex-start;
  padding:0 0 16% 4.5%; }
.lt-plate { display:flex; align-items:stretch; max-width:52%;
  background:rgba(2,6,23,.82); border-radius:6px; overflow:hidden;
  box-shadow:0 10px 36px rgba(0,0,0,.45); will-change:transform,opacity; }
.lt-bar { width:.55em; flex:0 0 auto; background:__ACCENT__; }
.lt-text { color:#fff; font-size:clamp(30px,4.6vw,60px); line-height:1.18;
  padding:.55em .8em; }
"""


def _kicker(text: str, max_chars: int = 38) -> str:
    """A short lower-third line: the first words of the scene, word-safe trimmed."""
    words = (text or "").split()
    out = ""
    for w in words:
        candidate = f"{out} {w}".strip()
        if out and len(candidate) > max_chars:
            return out.upper()
        out = candidate
    # A single token longer than the plate still has to fit.
    return out[:max_chars].upper()


def _news_scene_html(spec: SceneSpec, idx: int) -> str:
    s = spec.scene
    start = round(s.start, 3)
    dur = round(s.duration, 3)
    parts = []
    if spec.background is not None:
        name = _asset_name(spec.background)
        if name:
            parts.append(
                f'<img id="nbg-{idx}" class="clip bg-photo" data-start="{start}" '
                f'data-duration="{dur}" data-track-index="1" src="assets/{_esc(name)}">'
            )
            parts.append(
                f'<div class="clip scrim" data-start="{start}" data-duration="{dur}" '
                f'data-track-index="2"></div>'
            )
    parts.append(
        f'<div id="lt-{idx}" class="clip lt-wrap" data-start="{start}" '
        f'data-duration="{dur}" data-track-index="4">'
        f'<div class="lt-plate"><div class="lt-bar"></div>'
        f'<div class="lt-text">{_esc(_kicker(spec.caption))}</div></div></div>'
    )
    return "\n    ".join(parts)


def _news_scene_tweens(spec: SceneSpec, idx: int) -> str:
    s = spec.scene
    start = round(s.start, 3)
    end = round(s.end, 3)
    dur = round(s.duration, 3)
    exit_at = round(max(start + 0.2, end - 0.4), 3)
    js = []
    if spec.background is not None and _asset_name(spec.background):
        js.append(
            f'tl.fromTo("#nbg-{idx}",{{scale:1.0}},'
            f'{{scale:1.08,duration:{dur},ease:"none"}},{start});'
        )
    # Lower-third slides in from the left, holds, then fades before the next scene.
    js.append(
        f'tl.fromTo("#lt-{idx} .lt-plate",{{autoAlpha:0,x:-64}},'
        f'{{autoAlpha:1,x:0,duration:0.5,ease:"power3.out"}},{round(start + 0.1, 3)});'
    )
    js.append(
        f'tl.to("#lt-{idx}",{{autoAlpha:0,duration:0.35,ease:"power2.in"}},{exit_at});'
    )
    return "\n    ".join(js)


def compose_news(scenes: List[Scene], subject: str, width: int, height: int,
                 total: float, backgrounds=None) -> str:
    """Compose the deterministic "news" layout. ``""`` on any problem.

    Layout: persistent gradient base, per-scene photo background with a scrim,
    a persistent headline plate (the video subject) at the top, and a per-scene
    lower-third with the scene's opening words. The lower-right quadrant stays
    empty so the presenter overlay never covers any text.
    """
    try:
        if not scenes or total <= 0:
            return ""
        specs = build_specs(scenes, backgrounds)
        css = _fill_css(_NEWS_CSS, width, height)
        total_r = round(total, 3)

        headline = _esc((subject or "").strip().upper() or "NEWS")
        clips = [
            f'<div class="clip bg-base" data-start="0" data-duration="{total_r}" data-track-index="0"></div>'
        ]
        for i, spec in enumerate(specs):
            clips.append(_news_scene_html(spec, i))
        clips.append(
            f'<div id="headline" class="clip headline-wrap" data-start="0" '
            f'data-duration="{total_r}" data-track-index="3">'
            f'<div class="headline-plate"><span class="headline-tag">NEWS</span>'
            f'<span class="headline-text">{headline}</span></div></div>'
        )

        tweens = [
            'tl.fromTo("#headline .headline-plate",{autoAlpha:0,y:-44},'
            '{autoAlpha:1,y:0,duration:0.6,ease:"power3.out"},0.05);'
        ]
        for i, spec in enumerate(specs):
            tweens.append(_news_scene_tweens(spec, i))

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
        logger.success(f"hyperframes studio: composed news layout ({len(specs)} scene(s))")
        return html
    except Exception as exc:  # noqa: BLE001 - composition must never hard-fail a run
        logger.warning(f"hyperframes studio compose_news failed: {exc}")
        return ""


def compose(scenes: List[Scene], subject: str, width: int, height: int,
            total: float, backgrounds=None) -> str:
    """Compose a complete, contract-compliant composition. ``""`` on any problem."""
    try:
        if not scenes or total <= 0:
            return ""
        specs = build_specs(scenes, backgrounds)
        css = _fill_css(_CSS, width, height)
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
        mix = {}
        for s in specs:
            mix[s.archetype] = mix.get(s.archetype, 0) + 1
        logger.success(
            f"hyperframes studio: composed {len(specs)} scene(s) "
            f"({', '.join(f'{v} {k}' for k, v in sorted(mix.items()))})"
        )
        return html
    except Exception as exc:  # noqa: BLE001 - composition must never hard-fail a run
        logger.warning(f"hyperframes studio compose failed: {exc}")
        return ""
