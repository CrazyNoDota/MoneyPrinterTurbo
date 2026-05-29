"""LLM scene planner for the hyperframes "director".

Given the timed :class:`~app.services.hyperframes.scenes.Scene` list, decide for
each scene whether it should be **real footage** (stock / the user's own clips /
future AI clips) or **motion graphics** (hyperframes), plus a search ``query`` for
the background photo / footage clip and an optional ``material_ref`` pointing at a
user-supplied clip that visually fits.

Best-effort: any LLM/JSON problem falls back to a deterministic heuristic so the
pipeline never hard-fails -- the caller can always render *something*.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from app.services import llm

from .scenes import Scene

# Numbers / money / percentages animate well as kinetic motion graphics; plain
# narrative prose is better served by real footage.
_NUMERIC = re.compile(r"\d|%|\$|€|£")

_VALID_KINDS = ("footage", "motiongraphics")


@dataclass
class ScenePlan:
    """One scene plus the director's decision for it."""

    scene: Scene
    kind: str  # "footage" | "motiongraphics"
    query: str = ""
    use_background: bool = False
    material_ref: Optional[int] = None

    @property
    def is_mg(self) -> bool:
        return self.kind == "motiongraphics"


def _heuristic_kind(text: str) -> str:
    return "motiongraphics" if _NUMERIC.search(text or "") else "footage"


def _heuristic_plan(scenes: List[Scene], subject: str) -> List[ScenePlan]:
    """Deterministic fallback used when the LLM is unavailable or invalid."""
    plans: List[ScenePlan] = []
    for s in scenes:
        kind = _heuristic_kind(s.text)
        plans.append(
            ScenePlan(
                scene=s,
                kind=kind,
                query=(subject or s.text[:60]).strip(),
                use_background=(kind == "motiongraphics"),
            )
        )
    return plans


def _scene_table(scenes: List[Scene]) -> str:
    lines = ["SCENES (index | start s | duration s | text):"]
    for i, s in enumerate(scenes):
        lines.append(f"  {i} | {s.start:.2f} | {s.duration:.2f} | {s.text}")
    return "\n".join(lines)


def _materials_block(material_hints: List[str]) -> str:
    if not material_hints:
        return ""
    lines = ["\nUSER MATERIALS (index | what the clip shows) -- reference by material_ref:"]
    for i, desc in enumerate(material_hints):
        lines.append(f"  {i} | {desc}")
    return "\n".join(lines)


def _build_prompt(scenes, subject, video_terms, material_hints) -> str:
    terms = video_terms
    if isinstance(terms, (list, tuple)):
        terms = ", ".join(str(t) for t in terms)
    parts = [
        "You are a short-form video director. For EACH scene decide whether it is "
        'best shown as real footage ("footage") or as animated motion graphics '
        '("motiongraphics"). Use motion graphics for numbers, statistics, money, '
        "comparisons, and lists; use footage for people, places, actions, and mood.",
        f"Video subject: {subject or 'general'}",
        f"Suggested stock search terms: {terms or '(none)'}",
        _scene_table(scenes),
        _materials_block(material_hints),
        (
            "\nReturn ONLY a JSON array with exactly one object per scene, in order. "
            "Each object: "
            '{"kind": "footage"|"motiongraphics", '
            '"query": "<1-4 word image/footage search>", '
            '"use_background": true|false, '
            '"material_ref": <integer index of a USER MATERIAL that fits, or null>}. '
            "use_background means a motion-graphics scene should show a real photo "
            "behind the animated text. material_ref is only for footage scenes that "
            "match a user material; otherwise null. No prose, no markdown fences."
        ),
    ]
    return "\n\n".join(p for p in parts if p)


_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def _parse(raw: str, count: int):
    """Parse the LLM response into a list of ``count`` dicts, or None."""
    if not raw or "Error: " in raw[:64]:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        m = _ARRAY.search(raw)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except Exception:
            return None
    if not isinstance(data, list) or len(data) != count:
        return None
    if not all(isinstance(d, dict) for d in data):
        return None
    return data


def _coerce(scenes, data, subject) -> List[ScenePlan]:
    plans: List[ScenePlan] = []
    for s, d in zip(scenes, data):
        kind = str(d.get("kind", "")).strip().lower()
        if kind not in _VALID_KINDS:
            kind = _heuristic_kind(s.text)
        query = (str(d.get("query", "")).strip() or subject or s.text[:60]).strip()
        ref = d.get("material_ref", None)
        if not isinstance(ref, int):
            ref = None
        plans.append(
            ScenePlan(
                scene=s,
                kind=kind,
                query=query,
                use_background=bool(d.get("use_background", kind == "motiongraphics")),
                material_ref=ref,
            )
        )
    return plans


def build_plan(
    scenes: List[Scene],
    subject: str,
    video_terms=None,
    material_hints: Optional[List[str]] = None,
) -> List[ScenePlan]:
    """Plan every scene. Falls back to a heuristic on any LLM/parse failure."""
    if not scenes:
        return []

    prompt = _build_prompt(scenes, subject, video_terms, material_hints or [])
    try:
        raw = llm._generate_response(prompt=prompt)
    except Exception as exc:  # noqa: BLE001 - planning is best-effort
        logger.warning(f"hyperframes planner LLM call failed: {exc}; using heuristic")
        return _heuristic_plan(scenes, subject)

    data = _parse(raw, len(scenes))
    if data is None:
        logger.warning("hyperframes planner: invalid LLM plan; using heuristic")
        return _heuristic_plan(scenes, subject)

    plans = _coerce(scenes, data, subject)
    mg = sum(1 for p in plans if p.is_mg)
    logger.success(
        f"hyperframes plan: {len(plans)} scene(s), {mg} motion-graphics, "
        f"{len(plans) - mg} footage"
    )
    return plans
