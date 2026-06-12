"""
app/services/sfx.py — tiny helper for resolving sound-effect asset paths.

Usage:
    from app.services.sfx import get_sfx_file
    path = get_sfx_file("whoosh")   # returns absolute path or "" if missing
"""

import os

from loguru import logger

from app.utils import utils


def _sfx_dir() -> str:
    """Return the absolute path to resource/sfx/ (may not exist yet)."""
    return utils.resource_dir("sfx")


def get_sfx_file(name: str) -> str:
    """
    Return the absolute path to the SFX mp3 for *name*.

    *name* may be given with or without the .mp3 extension, e.g.
    ``"whoosh"`` or ``"whoosh.mp3"``.

    Returns ``""`` — never raises — when:
    - the resource/sfx/ directory does not exist
    - the requested file does not exist inside it
    - any other unexpected error occurs
    """
    try:
        sfx_dir = _sfx_dir()
        if not os.path.isdir(sfx_dir):
            logger.warning(f"sfx dir not found: {sfx_dir}")
            return ""

        # normalise: add .mp3 if the caller omitted the extension
        filename = name if name.lower().endswith(".mp3") else f"{name}.mp3"
        path = os.path.join(sfx_dir, filename)

        if not os.path.isfile(path):
            logger.warning(f"sfx file not found: {path}")
            return ""

        return path
    except Exception as exc:
        logger.warning(f"get_sfx_file({name!r}) failed unexpectedly: {exc}")
        return ""
