"""Shared contract for news sources.

A source turns an external feed (Threads profile, RSS feed, ...) into a list
of ``NewsItem``. Sources are best-effort by design: any network/parse failure
must degrade to an empty list, never raise out of ``latest()`` -- the caller
falls through to the next source.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Drop tags and collapse whitespace -- RSS descriptions often embed HTML."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()


def make_item_id(*parts: str) -> str:
    """Stable id for dedupe across sources/runs."""
    digest = hashlib.md5("|".join(p or "" for p in parts).encode("utf-8"))
    return digest.hexdigest()[:16]


@dataclass
class NewsItem:
    title: str
    text: str = ""
    url: str = ""
    source: str = ""  # which source produced it: "threads", "rss", ...
    published: str = ""  # ISO 8601 when the feed provides it, else ""
    media: List[str] = field(default_factory=list)  # image/video URLs
    id: str = ""

    def __post_init__(self):
        self.title = strip_html(self.title)
        self.text = strip_html(self.text) or self.title
        if not self.id:
            self.id = make_item_id(self.url, self.title)


class NewsSource:
    """Base news source. Subclasses override ``is_configured`` and ``latest``."""

    name = "base"

    def is_configured(self) -> bool:
        return False

    def latest(self, limit: int = 10) -> List[NewsItem]:
        return []
