import os
import random
import threading
from typing import List
from urllib.parse import urlencode

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

from app.config import config
from app.models import const
from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode
from app.utils import utils
from app.utils.retry import call_with_retry

# Extensions accepted when scanning a local asset folder. Superset of the
# WebUI uploader (adds avi/flv) reusing the canonical lists in const.
LOCAL_MATERIAL_EXTENSIONS = sorted(
    set(const.FILE_TYPE_VIDEOS + const.FILE_TYPE_IMAGES + ["avi", "flv"])
)


def list_local_materials(directory: str, recursive: bool = True) -> List[MaterialInfo]:
    """Scan a folder and return every supported photo/video as a MaterialInfo.

    Files are returned sorted by path so the resulting video order is stable.
    """
    materials: List[MaterialInfo] = []
    if not directory or not os.path.isdir(directory):
        logger.warning(f"local materials folder not found: {directory}")
        return materials

    allowed = {f".{ext.lower()}" for ext in LOCAL_MATERIAL_EXTENSIONS}
    if recursive:
        candidates = [
            os.path.join(root, name)
            for root, _, files in os.walk(directory)
            for name in files
        ]
    else:
        candidates = [os.path.join(directory, name) for name in os.listdir(directory)]

    for file_path in sorted(candidates):
        if not os.path.isfile(file_path):
            continue
        if os.path.splitext(file_path)[1].lower() not in allowed:
            continue
        m = MaterialInfo()
        m.provider = "local"
        m.url = file_path
        m.name = os.path.basename(file_path)
        materials.append(m)

    logger.info(f"found {len(materials)} local materials in {directory}")
    return materials

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()


def _get_tls_verify() -> bool:
    # 默认开启 TLS 证书校验，防止素材搜索和下载过程被中间人篡改。
    # 仅在企业代理、自签证书等明确需要的场景下，允许用户通过
    # `config.toml` 显式设置 `tls_verify = false` 临时关闭。
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )

    return bool(tls_verify)


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\nPlease set it in the config.toml file: {config.config_file}\n\n"
            f"{utils.to_json(config.app)}"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_orientation = aspect.name
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20, "orientation": video_orientation}
    query_url = f"https://api.pexels.com/videos/search?{urlencode(params)}"
    logger.info(f"searching videos: {query_url}, with proxies: {config.proxy}")

    try:
        def _search():
            resp = requests.get(
                query_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 60),
            )
            resp.raise_for_status()
            return resp

        r = call_with_retry(_search, description="pexels.search")
        response = r.json()
        video_items = []
        if "videos" not in response:
            logger.error(f"search videos failed: {response}")
            return video_items
        videos = response["videos"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["video_files"]
            # loop through each url to determine the best quality
            for video in video_files:
                w = int(video["width"])
                h = int(video["height"])
                if w == video_width and h == video_height:
                    item = MaterialInfo()
                    item.provider = "pexels"
                    item.url = video["link"]
                    item.duration = duration
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(f"searching videos: {query_url}, with proxies: {config.proxy}")

    try:
        def _search():
            resp = requests.get(
                query_url,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 60),
            )
            resp.raise_for_status()
            return resp

        r = call_with_retry(_search, description="pixabay.search")
        response = r.json()
        video_items = []
        if "hits" not in response:
            logger.error(f"search videos failed: {response}")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["videos"]
            # loop through each url to determine the best quality
            for video_type in video_files:
                video = video_files[video_type]
                w = int(video["width"])
                # h = int(video["height"])
                if w >= video_width:
                    item = MaterialInfo()
                    item.provider = "pixabay"
                    item.url = video["url"]
                    item.duration = duration
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


# --- Photo search -----------------------------------------------------------
# Hyperframes can show real photos as backgrounds behind motion graphics. These
# mirror the video search above (same key rotation / retry / TLS / proxy), but
# hit the photo endpoints and return MaterialInfo whose ``url`` is a remote
# image URL to be downloaded with ``save_image``.

# Pixabay's image API only accepts all/horizontal/vertical; map the aspect name.
_PIXABAY_ORIENTATION = {"portrait": "vertical", "landscape": "horizontal", "square": "all"}


def search_images_pexels(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    params = {"query": search_term, "per_page": per_page, "orientation": aspect.name}
    query_url = f"https://api.pexels.com/v1/search?{urlencode(params)}"
    logger.info(f"searching images: {query_url}, with proxies: {config.proxy}")

    try:
        def _search():
            resp = requests.get(
                query_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 60),
            )
            resp.raise_for_status()
            return resp

        r = call_with_retry(_search, description="pexels.search_images")
        response = r.json()
        items = []
        for p in response.get("photos", []):
            src = p.get("src", {}) or {}
            # Prefer a large rendition; fall back through what Pexels returns.
            url = src.get("large2x") or src.get("original") or src.get("large")
            if not url:
                continue
            item = MaterialInfo()
            item.provider = "pexels"
            item.url = url
            item.name = (p.get("alt") or search_term).strip()
            items.append(item)
        return items
    except Exception as e:
        logger.error(f"search images failed: {str(e)}")
    return []


def search_images_pixabay(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("pixabay_api_keys")
    params = {
        "q": search_term,
        "image_type": "photo",
        "orientation": _PIXABAY_ORIENTATION.get(aspect.name, "all"),
        "per_page": per_page,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/?{urlencode(params)}"
    logger.info(f"searching images: {query_url}, with proxies: {config.proxy}")

    try:
        def _search():
            resp = requests.get(
                query_url,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 60),
            )
            resp.raise_for_status()
            return resp

        r = call_with_retry(_search, description="pixabay.search_images")
        response = r.json()
        items = []
        for h in response.get("hits", []):
            url = h.get("largeImageURL") or h.get("webformatURL")
            if not url:
                continue
            item = MaterialInfo()
            item.provider = "pixabay"
            item.url = url
            item.name = (h.get("tags") or search_term).strip()
            items.append(item)
        return items
    except Exception as e:
        logger.error(f"search images failed: {str(e)}")
    return []


def search_images(
    search_term: str,
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[MaterialInfo]:
    """Search photos from the chosen provider, falling back to the other one."""
    source = (source or "pexels").strip().lower()
    primary, secondary = (
        (search_images_pixabay, search_images_pexels)
        if source == "pixabay"
        else (search_images_pexels, search_images_pixabay)
    )
    items = primary(search_term, video_aspect=video_aspect, per_page=per_page)
    if items:
        return items
    # Fall back to the other provider when the primary has no key / no hits.
    try:
        return secondary(search_term, video_aspect=video_aspect, per_page=per_page)
    except Exception:  # noqa: BLE001 - fallback is best-effort
        return []


def save_image(image_url: str, save_dir: str = "") -> str:
    """Download an image, validate it with Pillow, and return the local path.

    Mirrors ``save_video``: cached by url-hash so repeat runs are cheap, and a
    corrupt download is discarded so callers can fall back.
    """
    if not save_dir:
        save_dir = utils.storage_dir("cache_images")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url_without_query = image_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    ext = os.path.splitext(url_without_query)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    image_path = f"{save_dir}/img-{url_hash}{ext}"

    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        logger.info(f"image already exists: {image_path}")
        return image_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    def _download():
        resp = requests.get(
            image_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 120),
        )
        resp.raise_for_status()
        return resp

    with open(image_path, "wb") as f:
        f.write(call_with_retry(_download, description="material.download_image").content)

    try:
        with Image.open(image_path) as im:
            im.verify()
        return image_path
    except Exception as e:  # noqa: BLE001 - discard a corrupt download
        logger.warning(f"invalid image file: {image_path} => {str(e)}")
        try:
            os.remove(image_path)
        except Exception:
            pass
    return ""


def download_images(
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    count: int = 1,
    save_dir: str = "",
) -> List[str]:
    """Search + download up to ``count`` photos across ``search_terms``.

    Returns local file paths (deduplicated). Best-effort: returns whatever it
    could fetch, possibly empty.
    """
    seen_urls = set()
    paths: List[str] = []
    for term in search_terms or []:
        if len(paths) >= count:
            break
        for item in search_images(term, source=source, video_aspect=video_aspect):
            if len(paths) >= count:
                break
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            try:
                local = save_image(item.url, save_dir=save_dir)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"failed to download image {item.url}: {e}")
                continue
            if local:
                paths.append(local)
    logger.info(f"downloaded {len(paths)} image(s) for {search_terms}")
    return paths


def save_video(video_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url_without_query = video_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    # if video already exists, return the path
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        logger.info(f"video already exists: {video_path}")
        return video_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # if video does not exist, download it
    def _download():
        resp = requests.get(
            video_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(60, 240),
        )
        resp.raise_for_status()
        return resp

    with open(video_path, "wb") as f:
        f.write(call_with_retry(_download, description="material.download").content)

    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        clip = None
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            fps = clip.fps
            if duration > 0 and fps > 0:
                return video_path
        except Exception as e:
            logger.warning(f"invalid video file: {video_path} => {str(e)}")
            try:
                os.remove(video_path)
            except Exception as remove_error:
                logger.warning(
                    f"failed to remove invalid video file: {video_path}, error: {str(remove_error)}"
                )
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception as close_error:
                    logger.warning(
                        f"failed to close video clip: {video_path}, error: {str(close_error)}"
                    )
    return ""


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_contact_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
) -> List[str]:
    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    search_videos = search_videos_pexels
    if source == "pixabay":
        search_videos = search_videos_pixabay

    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if video_contact_mode.value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    total_duration = 0.0
    for item in valid_video_items:
        try:
            logger.info(f"downloading video: {item.url}")
            saved_video_path = save_video(
                video_url=item.url, save_dir=material_directory
            )
            if saved_video_path:
                logger.info(f"video saved: {saved_video_path}")
                video_paths.append(saved_video_path)
                seconds = min(max_clip_duration, item.duration)
                total_duration += seconds
                if total_duration > audio_duration:
                    logger.info(
                        f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                    )
                    break
        except Exception as e:
            logger.error(f"failed to download video: {utils.to_json(item)} => {str(e)}")
    logger.success(f"downloaded {len(video_paths)} videos")
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )
