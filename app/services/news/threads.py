"""Threads (threads.net) news source.

Threads has no public read API without an app token, so this scrapes the JSON
that the profile page embeds in ``<script type="application/json">`` blobs and
walks it for ``thread_items`` -- the same payload the web client renders from.
That structure is Meta-internal and may change at any time, which is exactly
why the pipeline always keeps RSS as the fallback source: every failure here
degrades to an empty list.

Profiles come from ``news_threads_profiles`` in config (usernames without @).
"""

import json
import re
from datetime import datetime, timezone
from typing import List

import requests
from loguru import logger

from app.config import config
from app.services.news.base import NewsItem, NewsSource

# Tolerates single/double quotes and a charset suffix in the type attribute.
_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/json[^"\']*["\'][^>]*>(.*?)</script>', re.S
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _walk_thread_items(obj, found: list):
    """Collect every ``thread_items`` list anywhere in the decoded JSON."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "thread_items" and isinstance(value, list):
                found.extend(value)
            else:
                _walk_thread_items(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _walk_thread_items(value, found)


def _post_media(post: dict) -> List[str]:
    urls = []
    candidates = (post.get("image_versions2") or {}).get("candidates") or []
    if candidates:
        url = (candidates[0] or {}).get("url") or ""
        if url:
            urls.append(url)
    for carousel in post.get("carousel_media") or []:
        for cand in ((carousel or {}).get("image_versions2") or {}).get("candidates") or []:
            url = (cand or {}).get("url") or ""
            if url:
                urls.append(url)
            break
    for video in post.get("video_versions") or []:
        url = (video or {}).get("url") or ""
        if url:
            urls.append(url)
            break
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


class ThreadsSource(NewsSource):
    name = "threads"

    def __init__(self, profiles=None, timeout=None):
        raw = profiles if profiles is not None else config.app.get("news_threads_profiles", [])
        self.profiles = [p.strip().lstrip("@") for p in raw if p and p.strip()]
        self.timeout = int(
            timeout if timeout is not None else config.app.get("news_fetch_timeout", 15)
        )

    def is_configured(self) -> bool:
        return bool(self.profiles)

    def latest(self, limit: int = 10) -> List[NewsItem]:
        items: List[NewsItem] = []
        for profile in self.profiles:
            try:
                items.extend(self._fetch_profile(profile, limit))
            except Exception as e:  # noqa: BLE001 - scraping is best-effort
                logger.warning(f"threads profile failed (@{profile}): {e}")
        return items[:limit]

    def _fetch_profile(self, username: str, limit: int) -> List[NewsItem]:
        url = f"https://www.threads.net/@{username}"
        resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
        if resp.status_code != 200:
            logger.warning(f"threads returned {resp.status_code} for @{username}")
            return []
        return self.parse(resp.text, username)[:limit]

    def parse(self, html: str, username: str) -> List[NewsItem]:
        """Extract posts from the embedded JSON of a profile page; bad HTML -> []."""
        thread_items: list = []
        for blob in _SCRIPT_RE.findall(html or ""):
            if "thread_items" not in blob:
                continue
            try:
                _walk_thread_items(json.loads(blob), thread_items)
            except (ValueError, RecursionError):
                continue

        items = []
        seen_codes = set()
        for entry in thread_items:
            post = (entry or {}).get("post") or {}
            text = ((post.get("caption") or {}).get("text") or "").strip()
            code = (post.get("code") or "").strip()
            if not text or code in seen_codes:
                continue
            seen_codes.add(code)
            published = ""
            taken_at = post.get("taken_at")
            if isinstance(taken_at, (int, float)) and taken_at > 0:
                published = datetime.fromtimestamp(taken_at, tz=timezone.utc).isoformat()
            items.append(
                NewsItem(
                    # First line as the headline; the full text rides along.
                    title=text.splitlines()[0],
                    text=text,
                    url=f"https://www.threads.net/@{username}/post/{code}" if code else "",
                    source=self.name,
                    published=published,
                    media=_post_media(post),
                )
            )
        if not items:
            logger.warning(f"threads: no posts extracted for @{username} (markup change?)")
        return items
