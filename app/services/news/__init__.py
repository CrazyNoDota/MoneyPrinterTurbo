"""News aggregation: Threads first, RSS fallback.

``latest()`` walks the configured sources in priority order and returns the
first non-empty batch -- so when Threads is unreachable (or scraping breaks)
the pipeline silently continues on RSS. All failures degrade to ``[]``.
"""

from typing import List

from loguru import logger

from app.services.news.base import NewsItem, NewsSource  # noqa: F401 - re-export
from app.services.news.rss import RSSSource
from app.services.news.threads import ThreadsSource

__all__ = ["NewsItem", "NewsSource", "RSSSource", "ThreadsSource", "latest"]


def _sources() -> List[NewsSource]:
    return [ThreadsSource(), RSSSource()]


def latest(limit: int = 10) -> List[NewsItem]:
    """Fresh news items from the first source that yields any, deduped, or []."""
    for source in _sources():
        if not source.is_configured():
            continue
        try:
            items = source.latest(limit)
        except Exception as e:  # noqa: BLE001 - sources are best-effort
            logger.warning(f"news source {source.name} failed: {e}")
            continue
        if not items:
            logger.warning(f"news source {source.name} returned nothing; trying next")
            continue
        seen = set()
        deduped = [i for i in items if not (i.id in seen or seen.add(i.id))]
        logger.info(f"news: {len(deduped)} item(s) from {source.name}")
        return deduped[:limit]
    logger.warning("news: no configured source produced items")
    return []
