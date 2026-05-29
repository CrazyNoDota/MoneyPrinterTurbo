"""Scrape themed photos from Pinterest to enrich the video.

Pinterest has no free official API for searching other people's pins (the v5
API only manages your own boards), so this targets the site's unofficial
``BaseSearchResource`` JSON endpoint used by the web client. It needs no API
key, but it is not a stable contract: Pinterest can change or block it at any
time. Every failure here is therefore non-fatal and logged -- the caller falls
back to stock footage / local materials so generation never breaks.

The downloaded images are saved into the ``local_videos`` storage directory so
they pass ``video.preprocess_video``'s path-security and resolution checks, and
they are analyzed by the vision model just like uploaded materials.
"""

import json
import os
from typing import List
from urllib.parse import quote, urlencode

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo
from app.utils import utils

_SEARCH_URL = "https://www.pinterest.com/resource/BaseSearchResource/get/"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json, text/javascript, */*, q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

# Pinterest tags every image with several rendered sizes; prefer the original.
_PREFERRED_SIZES = ("orig", "originals", "736x", "564x", "474x")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_enabled() -> bool:
    return bool(config.app.get("pinterest_enabled", False))


def _get_tls_verify() -> bool:
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")
    return bool(tls_verify)


def _pick_image_url(images: dict) -> str:
    """Return the URL of the largest usable image variant, or ''."""
    if not isinstance(images, dict) or not images:
        return ""
    # Prefer known large sizes in order.
    for key in _PREFERRED_SIZES:
        variant = images.get(key)
        if isinstance(variant, dict) and variant.get("url"):
            return variant["url"]
    # Otherwise fall back to the widest variant we can find.
    best_url, best_width = "", -1
    for variant in images.values():
        if isinstance(variant, dict) and variant.get("url"):
            width = variant.get("width") or 0
            if width >= best_width:
                best_url, best_width = variant["url"], width
    return best_url


def parse_results(payload: dict) -> List[str]:
    """Extract image URLs from a BaseSearchResource JSON response.

    Kept separate from the HTTP call so the fragile parsing is unit-testable
    without the network.
    """
    urls: List[str] = []
    try:
        results = (
            payload.get("resource_response", {}).get("data", {}).get("results")
        )
    except AttributeError:
        results = None
    for item in results or []:
        if not isinstance(item, dict):
            continue
        url = _pick_image_url(item.get("images") or {})
        if url and url not in urls:
            urls.append(url)
    return urls


def search_images(query: str, limit: int = 6) -> List[str]:
    """Query Pinterest and return up to ``limit`` image URLs (best effort).

    Pinterest's resource endpoint rejects bare requests ("Invalid Resource
    Request"). It needs (1) session cookies incl. a ``csrftoken`` obtained by
    first loading the search page, and (2) the web client's PWS-handler headers.
    """
    query = (query or "").strip()
    if not query:
        return []

    referer = f"https://www.pinterest.com/search/pins/?q={quote(query)}"
    session = requests.Session()
    session.headers.update(
        {"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    )

    logger.info(f"searching pinterest images for '{query}'")
    try:
        # 1. Prime the session: the search page sets csrftoken / session cookies.
        session.get(
            referer,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )

        # Ask for extra so the resolution filter downstream still leaves enough.
        page_size = max(int(limit) * 2, int(limit))
        data = {
            "options": {"query": query, "scope": "pins", "page_size": page_size},
            "context": {},
        }
        params = {
            "source_url": f"/search/pins/?q={query}&rs=typed",
            "data": json.dumps(data, separators=(",", ":")),
        }
        url = f"{_SEARCH_URL}?{urlencode(params)}"
        headers = dict(
            _HEADERS,
            Referer=referer,
            **{
                "X-Pinterest-PWS-Handler": "www/search/[scope].js",
                "X-Pinterest-AppState": "active",
                "X-APP-VERSION": "a8f6e8c",
                "X-CSRFToken": session.cookies.get("csrftoken", ""),
            },
        )
        r = session.get(
            url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        r.raise_for_status()
        urls = parse_results(r.json())
    except Exception as e:
        logger.warning(f"pinterest search failed for '{query}': {e}")
        return []
    finally:
        session.close()

    logger.info(f"pinterest returned {len(urls)} image(s) for '{query}'")
    return urls[: int(limit)]


def _save_image(image_url: str, query: str) -> str:
    """Download one image into the local_videos cache; return its local path."""
    save_dir = utils.storage_dir("local_videos", create=True)
    url_without_query = image_url.split("?")[0]
    ext = os.path.splitext(url_without_query)[1].lower()
    if ext not in _IMAGE_EXTS:
        ext = ".jpg"
    image_path = os.path.join(save_dir, f"pinterest-{utils.md5(url_without_query)}{ext}")

    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        return image_path

    resp = requests.get(
        image_url,
        headers=_HEADERS,
        proxies=config.proxy,
        verify=_get_tls_verify(),
        timeout=(60, 120),
    )
    resp.raise_for_status()
    with open(image_path, "wb") as f:
        f.write(resp.content)

    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        return image_path
    return ""


def download_images(query: str, limit: int = 6) -> List[MaterialInfo]:
    """Search + download themed Pinterest photos as local materials.

    Returns ``MaterialInfo`` objects pointing at local files (so they flow
    through vision analysis and the normal preprocessing pipeline). The clean
    ``name`` doubles as a file-name content hint for the script.
    """
    materials: List[MaterialInfo] = []
    image_urls = search_images(query, limit=limit)
    for image_url in image_urls:
        try:
            local_path = _save_image(image_url, query)
        except Exception as e:
            logger.warning(f"failed to download pinterest image {image_url}: {e}")
            continue
        if not local_path:
            continue
        m = MaterialInfo()
        m.provider = "pinterest"
        m.url = local_path
        # The query is a clean, human-readable content hint for the LLM.
        m.name = f"{query}.jpg"
        materials.append(m)

    logger.success(f"downloaded {len(materials)} pinterest image(s) for '{query}'")
    return materials


__all__ = ["is_enabled", "search_images", "download_images", "parse_results"]
