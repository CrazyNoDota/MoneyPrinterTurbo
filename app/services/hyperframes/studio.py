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
        # A bare small integer ("Дешёвые ракеты 2", "шаг 3") reads as an
        # enumeration fragment, not a headline stat -- don't blow it up to
        # 26vw. With a currency/percent/magnitude marker any size qualifies.
        if not prefix and not suffix and number < 10:
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
        # Bars are normalized to the largest value (infographic convention):
        # comparing 9% vs 15% should read as a 60/100 contrast, not two slivers
        # on an absolute 0-100 track.
        peak = max((value for _, value in spec.bars), default=0) or 1
        for n, (label, value) in enumerate(spec.bars):
            pct = round(value, 1)
            pct_str = f"{pct:g}%"
            width = round(value / peak * 100, 1)
            rows.append(
                f'<div class="line chart-row">'
                f'<div class="bar-head"><span>{_esc(label) or "&nbsp;"}</span>'
                f'<span class="bar-pct">{pct_str}</span></div>'
                f'<div class="bar-track"><div class="bar-fill" '
                f'style="width:{width:g}%"></div></div></div>'
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
    # Column in (rise + fade), staggered children for a kinetic feel. The very
    # first scene is the hook — it must be fully readable at frame 0, so it
    # skips the entrance animation entirely.
    if start <= 0.001:
        js.append(f'tl.set("#scene-{idx} .line",{{autoAlpha:1,y:0}},0);')
    else:
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

        # The headline is the hook — visible at frame 0, no drop-in delay.
        tweens = [
            'tl.set("#headline .headline-plate",{autoAlpha:1,y:0},0);'
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


# ---------------------------------------------------------------------------
# "quiz" archetype: per-question Q card -> 3-2-1 countdown beat -> answer card.
# The narration is a single continuous TTS track; the visual scene list is built
# from the script's real SRT timings (see scenes.build_scenes upstream). The
# countdown is a dedicated scene that lands on the spoken "three... two... one"
# segment, so the visual countdown is audio-synced with no per-segment audio
# surgery (see app.services.hyperframes.render_quiz_video for how the script is
# assembled with that spoken cue between every question and its answer).
# ---------------------------------------------------------------------------

# A scene whose narration is the spoken countdown cue. Detected by the digits
# pattern so the renderer can swap in the animated 3-2-1 visual. Matches lines
# like "Three, two, one" / "3 2 1" / "Три... два... один".
_COUNTDOWN_WORDS = (
    "three", "two", "one", "три", "два", "один",
)
_COUNTDOWN_RE = re.compile(
    r"^\W*(?:3\W+2\W+1|three\W+two\W+one|три\W+два\W+один)\W*$",
    re.IGNORECASE,
)


def is_countdown_text(text: str) -> bool:
    """True when a scene's narration is the quiz countdown cue."""
    return bool(_COUNTDOWN_RE.match((text or "").strip()))


_QUIZ_CSS = """
__FONTS__
* { margin:0; padding:0; box-sizing:border-box; }
html, body { width:__W__px; height:__H__px; overflow:hidden;
  background:#020617; font-family:__FAMILY__; font-weight:700; }
#root { position:relative; width:__W__px; height:__H__px; }
.clip { position:absolute; inset:0; }
.bg-base { background:linear-gradient(160deg,#0f172a 0%,#020617 70%,#000 100%); }
.bg-photo { width:100%; height:100%; object-fit:cover; will-change:transform; }
.scrim { background:linear-gradient(180deg,rgba(2,6,23,.45) 0%,rgba(2,6,23,.6) 45%,rgba(2,6,23,.85) 100%); }
.col { display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; padding:0 8%; gap:2.4vh; }
.q-tag { background:__ACCENT__; color:#020617; font-size:clamp(26px,3.6vw,46px);
  letter-spacing:.2em; padding:.35em 1em; border-radius:6px; }
.q-text { color:#fff; font-size:clamp(48px,8vw,116px); line-height:1.1;
  max-width:92%; text-shadow:0 4px 22px rgba(0,0,0,.7); }
.cd-digit { color:__ACCENT__; font-size:clamp(220px,40vw,520px); line-height:.9;
  text-shadow:0 8px 50px rgba(0,0,0,.6); will-change:transform,opacity; }
.cd-ring { position:absolute; width:clamp(360px,62vw,760px); height:clamp(360px,62vw,760px);
  border:clamp(8px,1.4vw,18px) solid rgba(255,212,0,.3); border-radius:50%;
  top:50%; left:50%; transform:translate(-50%,-50%); }
.a-tag { background:#16a34a; color:#fff; font-size:clamp(26px,3.6vw,46px);
  letter-spacing:.2em; padding:.35em 1em; border-radius:6px; }
.a-text { color:__ACCENT__; font-size:clamp(56px,9.2vw,140px); line-height:1.04;
  letter-spacing:-.01em; max-width:92%; text-shadow:0 6px 30px rgba(0,0,0,.6); }
.a-fact { color:#fff; font-size:clamp(34px,5vw,66px); line-height:1.18;
  max-width:88%; text-shadow:0 3px 16px rgba(0,0,0,.7); }
.reveal-wipe { position:absolute; inset:0; background:#16a34a; transform-origin:left center;
  will-change:transform; }
.line { will-change:transform,opacity; }
"""


def _quiz_role(text: str) -> str:
    """Classify a quiz scene: 'countdown' | 'answer' | 'question'.

    Answer scenes are tagged with a leading marker the renderer strips (see
    render_quiz_video). Countdown scenes match the spoken 3-2-1 cue.
    """
    t = (text or "").strip()
    if is_countdown_text(t):
        return "countdown"
    if t.startswith(_ANSWER_MARK):
        return "answer"
    return "question"


# Zero-width marker prefixed onto answer narration so the composer can tell an
# answer scene from a question scene without re-parsing the LLM JSON. It is a
# normal-looking token that survives TTS/SRT round-trips ("A:" reads naturally).
_ANSWER_MARK = "⁣"  # invisible separator; stripped before display


def _quiz_scene_html(spec_scene: Scene, role: str, idx: int, background) -> str:
    start = round(spec_scene.start, 3)
    dur = round(spec_scene.duration, 3)
    parts = []
    if background is not None:
        name = _asset_name(background)
        if name:
            parts.append(
                f'<img id="qbg-{idx}" class="clip bg-photo" data-start="{start}" '
                f'data-duration="{dur}" data-track-index="1" src="assets/{_esc(name)}">'
            )
            parts.append(
                f'<div class="clip scrim" data-start="{start}" data-duration="{dur}" '
                f'data-track-index="2"></div>'
            )
    text = spec_scene.text.lstrip(_ANSWER_MARK).strip()
    if role == "countdown":
        body = (
            '<div class="cd-ring"></div>'
            f'<div id="cd-{idx}" class="line cd-digit">3</div>'
        )
    elif role == "answer":
        # answer text + optional fun fact split on the first " — " / " - "
        fact = ""
        for sep in (" — ", " – ", " - "):
            if sep in text:
                text, fact = text.split(sep, 1)
                break
        fact_html = f'<div class="line a-fact">{_esc(fact)}</div>' if fact.strip() else ""
        body = (
            f'<div class="reveal-wipe" id="wipe-{idx}"></div>'
            '<div class="line a-tag">ANSWER</div>'
            f'<div class="line a-text">{_esc(text)}</div>'
            + fact_html
        )
    else:
        body = (
            '<div class="line q-tag">QUESTION</div>'
            f'<div class="line q-text">{_esc(text)}</div>'
        )
    parts.append(
        f'<div id="qscene-{idx}" class="clip col" data-start="{start}" '
        f'data-duration="{dur}" data-track-index="3">{body}</div>'
    )
    return "\n    ".join(parts)


def _quiz_scene_tweens(spec_scene: Scene, role: str, idx: int, background) -> str:
    start = round(spec_scene.start, 3)
    end = round(spec_scene.end, 3)
    dur = round(spec_scene.duration, 3)
    exit_at = round(max(start + 0.2, end - 0.4), 3)
    js = []
    if background is not None and _asset_name(background):
        js.append(
            f'tl.fromTo("#qbg-{idx}",{{scale:1.0}},'
            f'{{scale:1.08,duration:{dur},ease:"none"}},{start});'
        )
    if role == "countdown":
        # 3 -> 2 -> 1 across the scene; each digit pops then shrinks out. The
        # ring sweeps once over the whole beat.
        seg = round(dur / 3.0, 3)
        for n, digit in enumerate((3, 2, 1)):
            at = round(start + n * seg, 3)
            js.append(
                f'tl.set("#cd-{idx}",{{textContent:"{digit}"}},{at});'
            )
            js.append(
                f'tl.fromTo("#cd-{idx}",{{scale:0.4,autoAlpha:0}},'
                f'{{scale:1,autoAlpha:1,duration:{round(seg * 0.5, 3)},ease:"back.out(2)"}},{at});'
            )
            js.append(
                f'tl.to("#cd-{idx}",{{scale:1.3,autoAlpha:0,duration:{round(seg * 0.4, 3)},'
                f'ease:"power2.in"}},{round(at + seg * 0.55, 3)});'
            )
        js.append(
            f'tl.fromTo("#qscene-{idx} .cd-ring",{{rotation:0,scale:0.8,autoAlpha:0.2}},'
            f'{{rotation:360,scale:1.05,autoAlpha:0.6,duration:{dur},ease:"none"}},{start});'
        )
        return "\n    ".join(js)
    if role == "answer":
        # Green wipe sweeps across, then the answer slams in behind it.
        js.append(
            f'tl.fromTo("#wipe-{idx}",{{scaleX:0}},'
            f'{{scaleX:1,duration:0.28,ease:"power2.in"}},{start});'
        )
        js.append(
            f'tl.to("#wipe-{idx}",{{scaleX:0,transformOrigin:"right center",'
            f'duration:0.3,ease:"power2.out"}},{round(start + 0.3, 3)});'
        )
        js.append(
            f'tl.fromTo("#qscene-{idx} .line",{{autoAlpha:0,y:40}},'
            f'{{autoAlpha:1,y:0,duration:0.5,stagger:0.08,ease:"power3.out"}},{round(start + 0.34, 3)});'
        )
        js.append(
            f'tl.fromTo("#qscene-{idx} .a-text",{{scale:0.6}},'
            f'{{scale:1,duration:0.45,ease:"back.out(1.6)"}},{round(start + 0.34, 3)});'
        )
    else:
        # The opening question is the hook — fully readable at frame 0, no
        # entrance animation; later questions keep the rise-in.
        if start <= 0.001:
            js.append(f'tl.set("#qscene-{idx} .line",{{autoAlpha:1,y:0}},0);')
        else:
            js.append(
                f'tl.fromTo("#qscene-{idx} .line",{{autoAlpha:0,y:48}},'
                f'{{autoAlpha:1,y:0,duration:0.55,stagger:0.1,ease:"power3.out"}},{round(start + 0.08, 3)});'
            )
    js.append(
        f'tl.to("#qscene-{idx}",{{autoAlpha:0,y:-28,duration:0.4,ease:"power2.in"}},{exit_at});'
    )
    return "\n    ".join(js)


def compose_quiz(scenes: List[Scene], subject: str, width: int, height: int,
                 total: float, backgrounds=None) -> str:
    """Compose the deterministic "quiz" layout. ``""`` on any problem.

    Each scene is one of: a question card, a 3-2-1 countdown beat, or an answer
    reveal -- classified from its narration text (answer scenes carry an
    invisible marker, countdown scenes match the spoken 3-2-1 cue).
    """
    try:
        if not scenes or total <= 0:
            return ""
        css = _fill_css(_QUIZ_CSS, width, height)
        total_r = round(total, 3)
        bgs = list(backgrounds or [])
        clips = [
            f'<div class="clip bg-base" data-start="0" data-duration="{total_r}" data-track-index="0"></div>'
        ]
        tweens = []
        for i, s in enumerate(scenes):
            role = _quiz_role(s.text)
            bg = bgs[i % len(bgs)] if bgs else None
            clips.append(_quiz_scene_html(s, role, i, bg))
            tweens.append(_quiz_scene_tweens(s, role, i, bg))
        html = _document(css, total_r, width, height, clips, tweens)
        logger.success(f"hyperframes studio: composed quiz layout ({len(scenes)} scene(s))")
        return html
    except Exception as exc:  # noqa: BLE001 - composition must never hard-fail a run
        logger.warning(f"hyperframes studio compose_quiz failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# "ranking" archetype: "Top N" countdown #N -> #1 with a slamming rank badge.
# ---------------------------------------------------------------------------

# Rank narration carries an invisible marker + the rank number so the composer
# can render the badge without re-parsing the LLM JSON. Form: ⁣<rank>⁣<text>.

_RANK_RE = re.compile(rf"^{_ANSWER_MARK}(\d+){_ANSWER_MARK}(.*)$", re.DOTALL)


def _rank_parts(text: str):
    """``(rank:int, body:str)`` from a marked ranking scene, else ``None``."""
    m = _RANK_RE.match((text or ""))
    if not m:
        return None
    try:
        return int(m.group(1)), m.group(2).strip()
    except ValueError:
        return None


_RANK_CSS = """
__FONTS__
* { margin:0; padding:0; box-sizing:border-box; }
html, body { width:__W__px; height:__H__px; overflow:hidden;
  background:#020617; font-family:__FAMILY__; font-weight:700; }
#root { position:relative; width:__W__px; height:__H__px; }
.clip { position:absolute; inset:0; }
.bg-base { background:linear-gradient(160deg,#0f172a 0%,#020617 70%,#000 100%); }
.bg-photo { width:100%; height:100%; object-fit:cover; will-change:transform; }
.scrim { background:linear-gradient(180deg,rgba(2,6,23,.45) 0%,rgba(2,6,23,.6) 45%,rgba(2,6,23,.85) 100%); }
.col { display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; padding:0 8%; gap:2.2vh; }
.title-wrap { display:flex; align-items:center; justify-content:center; padding:0 8%; }
.title-text { color:#fff; font-size:clamp(56px,10vw,150px); line-height:1.05;
  letter-spacing:-.01em; max-width:92%; text-shadow:0 6px 30px rgba(0,0,0,.7); }
.rank-badge { display:flex; align-items:center; justify-content:center;
  width:clamp(180px,30vw,360px); height:clamp(180px,30vw,360px); border-radius:50%;
  background:__ACCENT__; color:#020617; font-size:clamp(110px,22vw,260px);
  line-height:1; box-shadow:0 12px 50px rgba(0,0,0,.5); will-change:transform; }
.rank-badge.gold { background:linear-gradient(160deg,#ffe066,#ffb300); }
.rank-name { color:#fff; font-size:clamp(50px,8.4vw,124px); line-height:1.08;
  max-width:92%; text-shadow:0 4px 22px rgba(0,0,0,.7); }
.rank-reason { color:#fff; font-size:clamp(32px,4.8vw,62px); line-height:1.2;
  max-width:88%; opacity:.92; text-shadow:0 3px 16px rgba(0,0,0,.7); }
.line { will-change:transform,opacity; }
"""


def _rank_scene_html(s: Scene, idx: int, background, title: str) -> str:
    start = round(s.start, 3)
    dur = round(s.duration, 3)
    parts = []
    if background is not None:
        name = _asset_name(background)
        if name:
            parts.append(
                f'<img id="rbg-{idx}" class="clip bg-photo" data-start="{start}" '
                f'data-duration="{dur}" data-track-index="1" src="assets/{_esc(name)}">'
            )
            parts.append(
                f'<div class="clip scrim" data-start="{start}" data-duration="{dur}" '
                f'data-track-index="2"></div>'
            )
    parsed = _rank_parts(s.text)
    if parsed is None:
        # Title / intro scene.
        body = (
            '<div class="title-wrap"><span class="line title-text">'
            f'{_esc(title or s.text)}</span></div>'
        )
    else:
        rank, body_text = parsed
        name, reason = body_text, ""
        for sep in (" — ", " – ", " - ", ": "):
            if sep in body_text:
                name, reason = body_text.split(sep, 1)
                break
        gold = " gold" if rank == 1 else ""
        reason_html = (
            f'<div class="line rank-reason">{_esc(reason)}</div>' if reason.strip() else ""
        )
        body = (
            f'<div class="line rank-badge{gold}">#{rank}</div>'
            f'<div class="line rank-name">{_esc(name)}</div>'
            + reason_html
        )
    parts.append(
        f'<div id="rscene-{idx}" class="clip col" data-start="{start}" '
        f'data-duration="{dur}" data-track-index="3">{body}</div>'
    )
    return "\n    ".join(parts)


def _rank_scene_tweens(s: Scene, idx: int, background) -> str:
    start = round(s.start, 3)
    end = round(s.end, 3)
    dur = round(s.duration, 3)
    exit_at = round(max(start + 0.2, end - 0.4), 3)
    parsed = _rank_parts(s.text)
    js = []
    if background is not None and _asset_name(background):
        js.append(
            f'tl.fromTo("#rbg-{idx}",{{scale:1.0}},'
            f'{{scale:1.08,duration:{dur},ease:"none"}},{start});'
        )
    if parsed is None:
        # The title card is the hook — fully readable at frame 0; a mid-video
        # title (shouldn't normally happen) keeps the pop-in.
        if start <= 0.001:
            js.append(f'tl.set("#rscene-{idx} .line",{{autoAlpha:1,scale:1}},0);')
        else:
            js.append(
                f'tl.fromTo("#rscene-{idx} .line",{{autoAlpha:0,scale:0.7}},'
                f'{{autoAlpha:1,scale:1,duration:0.6,ease:"back.out(1.7)"}},{round(start + 0.08, 3)});'
            )
    else:
        rank = parsed[0]
        # #1 gets a heavier, later slam (suspense beat before it lands).
        slam_at = round(start + (0.55 if rank == 1 else 0.1), 3)
        js.append(
            f'tl.fromTo("#rscene-{idx} .rank-badge",{{scale:2.2,autoAlpha:0,rotation:-12}},'
            f'{{scale:1,autoAlpha:1,rotation:0,duration:0.45,ease:"back.out(2.2)"}},{slam_at});'
        )
        js.append(
            f'tl.fromTo("#rscene-{idx} .rank-name, #rscene-{idx} .rank-reason",'
            f'{{autoAlpha:0,y:40}},{{autoAlpha:1,y:0,duration:0.5,stagger:0.08,'
            f'ease:"power3.out"}},{round(slam_at + 0.2, 3)});'
        )
    js.append(
        f'tl.to("#rscene-{idx}",{{autoAlpha:0,y:-28,duration:0.4,ease:"power2.in"}},{exit_at});'
    )
    return "\n    ".join(js)


def compose_ranking(scenes: List[Scene], subject: str, width: int, height: int,
                    total: float, backgrounds=None, title: str = "") -> str:
    """Compose the deterministic "ranking" (Top-N countdown) layout. ``""`` on any problem."""
    try:
        if not scenes or total <= 0:
            return ""
        css = _fill_css(_RANK_CSS, width, height)
        total_r = round(total, 3)
        bgs = list(backgrounds or [])
        clips = [
            f'<div class="clip bg-base" data-start="0" data-duration="{total_r}" data-track-index="0"></div>'
        ]
        tweens = []
        for i, s in enumerate(scenes):
            bg = bgs[i % len(bgs)] if bgs else None
            clips.append(_rank_scene_html(s, i, bg, title))
            tweens.append(_rank_scene_tweens(s, i, bg))
        html = _document(css, total_r, width, height, clips, tweens)
        logger.success(f"hyperframes studio: composed ranking layout ({len(scenes)} scene(s))")
        return html
    except Exception as exc:  # noqa: BLE001 - composition must never hard-fail a run
        logger.warning(f"hyperframes studio compose_ranking failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# "chat" archetype: messenger-style story. ONE persistent composition -- a phone
# frame (header with avatar + contact name) over a scrollable message column.
# Bubbles pop in one at a time, synced to the narration; a typing indicator
# flashes briefly before each incoming bubble; the column auto-scrolls so the
# newest bubble stays in view.
#
# Unlike the per-scene formats (quiz/ranking), the chat does NOT clear between
# scenes -- bubbles accumulate. So this is built as a single timeline driving
# per-message tweens, not a scene-clears-scene sequence. Timing comes from the
# proportional scene list (one Scene per message; see _chat_scene_list upstream),
# so each bubble lands on the segment of narration that reads it.
# ---------------------------------------------------------------------------

# Invisible per-message marker: ⁣<side>⁣<text>, side 0/1 (left/right). Lets the
# composer recover each message's side after the TTS/SRT scene round-trip without
# re-parsing the LLM JSON (same trick as the quiz answer / rank markers).
_CHAT_RE = re.compile(rf"^{_ANSWER_MARK}([01]){_ANSWER_MARK}(.*)$", re.DOTALL)


def _chat_parts(text: str):
    """``(side:int 0|1, body:str)`` from a marked chat scene, else ``None``."""
    m = _CHAT_RE.match(text or "")
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip()


_CHAT_CSS = """
__FONTS__
* { margin:0; padding:0; box-sizing:border-box; }
html, body { width:__W__px; height:__H__px; overflow:hidden;
  background:#0b141a; font-family:__FAMILY__; font-weight:600; }
#root { position:relative; width:__W__px; height:__H__px; }
.clip { position:absolute; inset:0; }
.bg-base { background:linear-gradient(160deg,#101b22 0%,#0b141a 70%,#05080a 100%); }
.bg-photo { width:100%; height:100%; object-fit:cover; will-change:transform; }
.scrim { background:rgba(7,12,15,.78); }
.chat-header { position:absolute; top:0; left:0; right:0; height:11%;
  display:flex; align-items:center; gap:3.4%; padding:0 5%;
  background:rgba(13,24,30,.96); box-shadow:0 4px 20px rgba(0,0,0,.45);
  border-bottom:1px solid rgba(255,255,255,.06); z-index:5; }
.avatar { flex:0 0 auto; display:flex; align-items:center; justify-content:center;
  width:clamp(96px,13vw,150px); height:clamp(96px,13vw,150px); border-radius:50%;
  background:linear-gradient(150deg,#25d366,#128c7e); color:#fff;
  font-size:clamp(44px,6vw,72px); }
.contact { display:flex; flex-direction:column; }
.contact-name { color:#fff; font-size:clamp(40px,5.6vw,68px); line-height:1.05; }
.contact-status { color:#8aa0aa; font-size:clamp(24px,3.2vw,38px); line-height:1.2; }
.chat-window { position:absolute; top:11%; left:0; right:0; bottom:0;
  overflow:hidden; }
.msg-col { position:absolute; left:0; right:0; bottom:0;
  display:flex; flex-direction:column; gap:2.2vh; padding:4% 5% 6%;
  will-change:transform; }
.bubble { max-width:78%; padding:.62em .8em; border-radius:26px;
  font-size:clamp(42px,5.8vw,72px); line-height:1.22; color:#fff;
  box-shadow:0 4px 16px rgba(0,0,0,.4); will-change:transform,opacity;
  word-wrap:break-word; }
.bubble.left { align-self:flex-start; background:#202c33;
  border-bottom-left-radius:8px; }
.bubble.right { align-self:flex-end; background:#005c4b;
  border-bottom-right-radius:8px; }
.typing { align-self:flex-start; display:flex; align-items:center; gap:.32em;
  padding:.9em 1em; border-radius:26px; border-bottom-left-radius:8px;
  background:#202c33; will-change:opacity; }
.typing.right { align-self:flex-end; border-bottom-left-radius:26px;
  border-bottom-right-radius:8px; background:#005c4b; }
.dot { width:clamp(14px,1.8vw,22px); height:clamp(14px,1.8vw,22px);
  border-radius:50%; background:#8aa0aa; }
"""

# Rough per-bubble height estimate (px) used to auto-scroll deterministically
# without measuring the DOM: a base row plus one extra line per ~22 chars.
_CHAT_ROW_BASE = 130
_CHAT_ROW_PER_LINE = 96
_CHAT_LINE_CHARS = 22


def _chat_bubble_height(text: str) -> int:
    lines = max(1, (len(text or "") + _CHAT_LINE_CHARS - 1) // _CHAT_LINE_CHARS)
    return _CHAT_ROW_BASE + (lines - 1) * _CHAT_ROW_PER_LINE


def compose_chat(scenes: List[Scene], subject: str, width: int, height: int,
                 total: float, persons=None, title: str = "") -> str:
    """Compose the deterministic messenger "chat" layout. ``""`` on any problem.

    One persistent phone frame: a header (contact name + avatar initial) over a
    message column. Each scene is one message (its side recovered from an
    invisible marker); bubbles pop in at their narrated time, a typing indicator
    flashes before each incoming (left) bubble, and the column scrolls up so the
    latest bubble stays visible. The cards carry all the text -> no burned
    captions (the caller returns ``caption_ranges = []``).
    """
    try:
        if not scenes or total <= 0:
            return ""
        persons = list(persons or [])
        contact = (persons[0] if persons else (title or subject) or "Chat").strip() or "Chat"
        initial = _esc(contact[:1].upper() or "?")
        css = _fill_css(_CHAT_CSS, width, height)
        total_r = round(total, 3)

        bubbles = []
        tweens = []
        # Cumulative column height so we can scroll the newest bubble into view.
        cum_h = 0
        offsets = []  # (top_offset_before_this_bubble, height)
        for s in scenes:
            parsed = _chat_parts(s.text)
            text = parsed[1] if parsed else (s.text or "").strip()
            offsets.append((cum_h, _chat_bubble_height(text)))
            cum_h += _chat_bubble_height(text)

        # Visible window height (below the header) in px.
        window_h = max(int(height * 0.89), 1)

        for i, s in enumerate(scenes):
            parsed = _chat_parts(s.text)
            side = (parsed[0] if parsed else (i % 2)) and 1 or 0
            text = parsed[1] if parsed else (s.text or "").strip()
            side_cls = "right" if side == 1 else "left"
            bubbles.append(
                f'<div id="msg-{i}" class="bubble {side_cls}">{_esc(text)}</div>'
            )

            start = round(s.start, 3)
            # Incoming (left) bubbles get a short typing indicator just before.
            typing_id = ""
            if side == 0:
                typing_id = f"typing-{i}"
                bubbles.append(
                    f'<div id="{typing_id}" class="typing {side_cls}">'
                    '<span class="dot"></span><span class="dot"></span>'
                    '<span class="dot"></span></div>'
                )
                type_dur = round(min(0.6, max(0.2, s.duration * 0.3)), 3)
                tweens.append(
                    f'tl.fromTo("#{typing_id}",{{autoAlpha:0}},'
                    f'{{autoAlpha:1,duration:0.18,ease:"power1.out"}},{start});'
                )
                tweens.append(
                    f'tl.to("#{typing_id}",{{autoAlpha:0,height:0,margin:0,padding:0,'
                    f'duration:0.18,ease:"power1.in"}},{round(start + type_dur, 3)});'
                )
                pop_at = round(start + type_dur, 3)
            else:
                pop_at = start

            tweens.append(
                f'tl.fromTo("#msg-{i}",{{autoAlpha:0,scale:0.6,y:24,'
                f'transformOrigin:"{"right bottom" if side else "left bottom"}"}},'
                f'{{autoAlpha:1,scale:1,y:0,duration:0.34,ease:"back.out(1.7)"}},{pop_at});'
            )
            # Auto-scroll: translate the column up so the bottom of this bubble
            # sits at the bottom of the window (only once the stack exceeds it).
            bottom_of_bubble = offsets[i][0] + offsets[i][1]
            scroll = max(0, bottom_of_bubble - window_h)
            tweens.append(
                f'tl.to("#msg-col",{{y:{-scroll},duration:0.3,ease:"power2.out"}},{pop_at});'
            )

        col = '<div id="msg-col" class="msg-col">' + "".join(bubbles) + "</div>"
        header = (
            '<div class="chat-header">'
            f'<div class="avatar">{initial}</div>'
            f'<div class="contact"><span class="contact-name">{_esc(contact)}</span>'
            '<span class="contact-status">online</span></div></div>'
        )
        clips = [
            f'<div class="clip bg-base" data-start="0" data-duration="{total_r}" data-track-index="0"></div>',
            f'<div class="clip chat-window" data-start="0" data-duration="{total_r}" '
            f'data-track-index="3">{col}</div>',
            f'<div class="clip" data-start="0" data-duration="{total_r}" '
            f'data-track-index="4">{header}</div>',
        ]
        # Persistent typing-dot bounce loop (visual only). repeat:-1 is banned:
        # an infinite child makes the GSAP timeline's duration Infinity, which
        # breaks the renderer's frame seek (every tween then freezes at t=0).
        # Repeat a finite count that covers the whole video instead.
        cycles = max(1, int(total_r / 0.6) + 1)
        bounce = (
            f'tl.to(".dot",{{y:-10,duration:0.3,repeat:{cycles},yoyo:true,'
            'stagger:0.12,ease:"sine.inOut"},0);'
        )
        tweens.insert(0, bounce)
        html = _document(css, total_r, width, height, clips, tweens)
        logger.success(f"hyperframes studio: composed chat layout ({len(scenes)} message(s))")
        return html
    except Exception as exc:  # noqa: BLE001 - composition must never hard-fail a run
        logger.warning(f"hyperframes studio compose_chat failed: {exc}")
        return ""


def _document(css: str, total_r: float, width: int, height: int,
              clips: List[str], tweens: List[str]) -> str:
    """Assemble the shared hyperframes HTML document (head + clips + timeline)."""
    return (
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
