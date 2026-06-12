"""Bundled display fonts for studio compositions.

Headless Chrome on a clean machine has no guarantee of any particular display
font: the old ``'Arial Black', Impact`` stack rendered differently per machine
and had no proper Cyrillic bold, so RU captions silently fell back to whatever
the OS picked. We bundle two open fonts (OFL) next to this module and stage
them into the hyperframes project's ``assets/`` dir alongside the background
photos, so the composition is rendered with the exact same glyphs everywhere:

- **Anton** -- the condensed poster face for Latin headlines (no Cyrillic).
- **Oswald** (variable) -- covers Cyrillic; листается per-glyph: a font-family
  stack of ``'Anton','Oswald',...`` gives Anton for Latin text and Oswald for
  RU automatically, because CSS font fallback is per missing glyph.

Everything is best-effort: when staging fails the returned CSS is ``""`` and
the composition still renders with the system fallback stack.
"""

import os
import shutil

from loguru import logger

from . import render

_FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")

# (file name, @font-face rule). Oswald is a variable font; declaring the full
# weight range lets `font-weight:700` pick the bold instance in Chrome.
_FONTS = (
    (
        "Anton-Regular.ttf",
        "@font-face { font-family:'Anton'; src:url('assets/Anton-Regular.ttf'); "
        "font-weight:400 700; font-display:block; }",
    ),
    (
        "Oswald-Variable.ttf",
        "@font-face { font-family:'Oswald'; src:url('assets/Oswald-Variable.ttf'); "
        "font-weight:200 700; font-display:block; }",
    ),
)

# The studio CSS references this stack; Anton first so Latin gets the poster
# face, Oswald catches Cyrillic per-glyph. No system fallbacks: the hyperframes
# compiler fetches unknown families from Google Fonts at render time (a network
# call that breaks offline determinism), and 'oswald' is also in its built-in
# deterministic map, so even without our staged files the stack stays offline.
FAMILY = "'Anton','Oswald',sans-serif"


def stage_fonts() -> str:
    """Copy the bundled fonts into the project assets dir; the @font-face CSS.

    Returns ``""`` when nothing could be staged (missing toolchain dir, missing
    bundled files) -- callers just embed the empty string and the fallback
    stack applies.
    """
    rules = []
    try:
        dest_dir = render.assets_dir()
        os.makedirs(dest_dir, exist_ok=True)
    except Exception as e:  # noqa: BLE001 - fonts are an enhancement, never a blocker
        logger.debug(f"hyperframes fonts: assets dir unavailable: {e}")
        return ""
    for name, rule in _FONTS:
        src = os.path.join(_FONTS_DIR, name)
        if not os.path.isfile(src):
            logger.debug(f"hyperframes fonts: bundled font missing: {src}")
            continue
        try:
            shutil.copyfile(src, os.path.join(dest_dir, name))
            rules.append(rule)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"hyperframes fonts: failed to stage {name}: {e}")
    return "\n".join(rules)
