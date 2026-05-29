"""A tiny JSON file cache under ``storage/cache/<namespace>/``.

Used to avoid re-paying for deterministic-but-expensive results across runs --
notably vision descriptions of the same image and AI-generated clips for the
same prompt. The caller is responsible for building a stable key (typically via
``utils.md5`` of the inputs).

Values must be JSON-serializable. The cache is best-effort: any read/write error
is swallowed and logged, so a corrupt or unwritable cache never breaks
generation.
"""

import json
import os
import re
from typing import Any, Optional

from loguru import logger

from app.utils import utils

_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]")


def _safe(name: str) -> str:
    """Sanitize a namespace/key fragment so it is a safe file/dir name."""
    return _SAFE_KEY.sub("_", str(name))[:128] or "_"


def _entry_path(namespace: str, key: str) -> str:
    cache_dir = utils.storage_dir(os.path.join("cache", _safe(namespace)), create=True)
    return os.path.join(cache_dir, f"{_safe(key)}.json")


def get(namespace: str, key: str) -> Optional[Any]:
    """Return the cached value, or ``None`` on miss / error."""
    path = _entry_path(namespace, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("value")
    except Exception as exc:  # noqa: BLE001 - cache is best-effort
        logger.warning(f"cache read failed for {namespace}/{key}: {exc}")
        return None


def set(namespace: str, key: str, value: Any) -> None:
    """Store a JSON-serializable value. Best effort; errors are logged."""
    path = _entry_path(namespace, key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"value": value}, f, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 - cache is best-effort
        logger.warning(f"cache write failed for {namespace}/{key}: {exc}")


def make_key(*parts: Any) -> str:
    """Build a stable cache key from arbitrary parts."""
    return utils.md5("|".join("" if p is None else str(p) for p in parts))


__all__ = ["get", "set", "make_key"]
