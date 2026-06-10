"""RSS 2.0 news source (stdlib XML, no extra dependencies).

Feeds come from ``news_rss_feeds`` in config. Media URLs are taken from
``<enclosure>`` and Media-RSS ``<media:content>``/``<media:thumbnail>``
elements so downstream can use real article imagery as backgrounds.
"""

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import List
from urllib.parse import urljoin

import requests
from loguru import logger

from app.config import config
from app.services.news.base import NewsItem, NewsSource

_MEDIA_NS = "{http://search.yahoo.com/mrss/}"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def _iso_date(raw: str) -> str:
    """RFC 822 pubDate -> ISO 8601, or "" when unparseable."""
    try:
        return parsedate_to_datetime(raw).isoformat()
    except Exception:  # noqa: BLE001 - feeds put anything in pubDate
        return ""


def _item_media(node) -> List[str]:
    urls = []
    for child in node:
        tag = child.tag
        if tag == "enclosure" or tag in (f"{_MEDIA_NS}content", f"{_MEDIA_NS}thumbnail"):
            url = (child.get("url") or "").strip()
            kind = (child.get("type") or "").strip().lower()
            if url and (not kind or kind.startswith(("image/", "video/"))):
                urls.append(url)
    # de-dup preserving order (thumbnail often repeats content)
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


class RSSSource(NewsSource):
    name = "rss"

    def __init__(self, feeds=None, timeout=None):
        self.feeds = [
            f.strip()
            for f in (feeds if feeds is not None else config.app.get("news_rss_feeds", []))
            if f and f.strip()
        ]
        self.timeout = int(
            timeout if timeout is not None else config.app.get("news_fetch_timeout", 15)
        )

    def is_configured(self) -> bool:
        return bool(self.feeds)

    def latest(self, limit: int = 10) -> List[NewsItem]:
        items: List[NewsItem] = []
        for feed in self.feeds:
            try:
                items.extend(self._fetch_feed(feed, limit))
            except Exception as e:  # noqa: BLE001 - a dead feed must not kill the run
                logger.warning(f"rss feed failed ({feed}): {e}")
        return items[:limit]

    def _fetch_feed(self, feed_url: str, limit: int) -> List[NewsItem]:
        resp = requests.get(feed_url, headers=_HEADERS, timeout=self.timeout)
        if resp.status_code != 200:
            logger.warning(f"rss feed returned {resp.status_code}: {feed_url}")
            return []
        return self.parse(resp.content, feed_url)[:limit]

    def parse(self, xml_bytes, feed_url: str = "") -> List[NewsItem]:
        """Parse RSS 2.0 bytes into items; malformed XML -> []."""
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            logger.warning(f"rss parse error ({feed_url}): {e}")
            return []

        items = []
        for node in root.iter("item"):
            title = (node.findtext("title") or "").strip()
            if not title:
                continue
            link = (node.findtext("link") or "").strip()
            items.append(
                NewsItem(
                    title=title,
                    text=node.findtext("description") or "",
                    # urljoin absolutizes the rare relative URL; absolute ones pass through.
                    url=urljoin(feed_url, link) if link else "",
                    source=self.name,
                    published=_iso_date(node.findtext("pubDate") or ""),
                    media=[urljoin(feed_url, u) for u in _item_media(node)],
                )
            )
        if not items:
            # Parsed fine but zero <item> nodes -- usually an Atom feed
            # (<entry>), which this RSS-2.0 parser does not read.
            logger.warning(f"rss feed has no <item> entries (Atom feed?): {feed_url}")
        return items
