import json
import math
import os.path
import re
from os import path

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import (
    llm,
    material,
    pinterest,
    subtitle,
    video,
    videogen,
    vision,
    voice,
    upload_post,
)
from app.services import state as sm
from app.utils import utils


# streamlit prefixes uploaded files with a UUID file_id: "<uuid>_original-name.ext"
_FILE_ID_PREFIX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_"
)


def get_material_keywords(materials):
    """Derive human-readable content hints from uploaded material file names.

    e.g. "a1b2..._sunset_over_the_sea.mp4" -> "sunset over the sea".
    Returns a de-duplicated, order-preserving list.
    """
    hints = []
    for m in materials or []:
        name = (getattr(m, "name", "") or "").strip()
        if not name:
            name = os.path.basename(m.url or "")
            name = _FILE_ID_PREFIX.sub("", name)
        name = os.path.splitext(name)[0]
        # normalize separators (._-) and collapse whitespace into spaces
        name = re.sub(r"[._\-]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        if name:
            hints.append(name)

    seen = set()
    unique = []
    for h in hints:
        key = h.lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


def _pinterest_enabled(params) -> bool:
    return bool(getattr(params, "pinterest_enabled", False)) or pinterest.is_enabled()


def augment_with_pinterest(params):
    """Scrape themed Pinterest photos and prepend them to the local materials.

    These images then flow through the normal pipeline: the vision model
    describes them (informing the script + search terms) and they are blended
    into the final video. Any failure is non-fatal -- generation continues with
    whatever stock/local materials are available.
    """
    if not _pinterest_enabled(params):
        return

    query = (params.video_subject or "").strip()
    if not query:
        logger.warning("pinterest is enabled but video_subject is empty; skipping scrape")
        return

    count = getattr(params, "pinterest_count", None) or config.app.get(
        "pinterest_count", 6
    )
    try:
        scraped = pinterest.download_images(query, limit=int(count))
    except Exception as e:
        logger.warning(f"pinterest scrape failed, continuing without it: {e}")
        return

    if not scraped:
        return

    existing = list(params.video_materials or [])
    # Prepend so the themed photos are analyzed first and lead the material list.
    params.video_materials = scraped + existing
    logger.info(f"added {len(scraped)} pinterest image(s) to the materials")


def _resume_enabled() -> bool:
    """Whether to reuse artifacts from a previous run of the same task_id."""
    return bool(config.app.get("resume", True))


def load_resumable_script(task_id):
    """Return ``(script, terms)`` cached from a previous run, else ``(None, None)``.

    Lets a re-run with the same task_id skip the expensive vision + LLM steps.
    """
    script_file = path.join(utils.task_dir(task_id), "script.json")
    if not os.path.exists(script_file):
        return None, None
    try:
        with open(script_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"failed to read cached script.json: {e}")
        return None, None
    script = (data.get("script") or "").strip()
    if script:
        return script, data.get("search_terms") or ""
    return None, None


def _videogen_enabled(params) -> bool:
    return bool(getattr(params, "video_gen_enabled", False)) or videogen.is_enabled()


def _build_clip_specs(params, video_terms):
    """Build AI-clip specs from on-theme images (image2video) or terms (text2video)."""
    aspect = getattr(params.video_aspect, "value", params.video_aspect) or "9:16"
    duration = float(params.video_clip_duration or 5)
    subject = (params.video_subject or "").strip()
    mode = str(config.app.get("video_gen_mode", "image2video")).strip().lower()

    image_exts = {f".{e.lower()}" for e in const.FILE_TYPE_IMAGES}
    images = []
    for m in params.video_materials or []:
        url = getattr(m, "url", "") or ""
        if url and os.path.splitext(url)[1].lower() in image_exts:
            hint = (getattr(m, "name", "") or "").rsplit(".", 1)[0]
            images.append((url, hint))

    specs = []
    if mode != "text2video" and images:
        for url, hint in images:
            prompt = subject or re.sub(r"[._\-]+", " ", hint).strip() or "cinematic scene"
            specs.append(
                videogen.ClipSpec(
                    prompt=prompt, init_image=url, duration=duration, aspect=aspect
                )
            )
    else:
        terms = video_terms
        if isinstance(terms, str):
            terms = [t.strip() for t in re.split(r"[,，]", terms) if t.strip()]
        for term in terms or ([subject] if subject else []):
            prompt = f"{subject} {term}".strip() if subject else term
            specs.append(
                videogen.ClipSpec(prompt=prompt, duration=duration, aspect=aspect)
            )
    return specs


def generate_ai_clips(params, video_terms):
    """Generate AI video clips (best effort). Returns local paths, or [] if off."""
    if not _videogen_enabled(params):
        return []
    specs = _build_clip_specs(params, video_terms)
    if not specs:
        return []
    count = getattr(params, "video_gen_count", None) or config.app.get(
        "video_gen_count", 2
    )
    specs = specs[: int(count)]
    try:
        clips = videogen.generate_clips(specs)
    except Exception as e:  # noqa: BLE001 - generation is best-effort
        logger.warning(f"AI video generation failed, continuing without it: {e}")
        return []
    if clips:
        logger.info(f"added {len(clips)} AI-generated clip(s) to the materials")
    return clips


def build_material_hints(params):
    """Content hints describing the local materials.

    Uses the vision model (actual scene descriptions) when enabled, otherwise
    falls back to readable keywords derived from the file names.
    """
    if not params.video_materials:
        return []

    if vision.is_enabled():
        try:
            descriptions = vision.describe_materials(params.video_materials)
            if descriptions:
                logger.info(
                    f"using {len(descriptions)} vision description(s) as content hints"
                )
                return descriptions
            logger.warning("vision returned no descriptions; falling back to file names")
        except Exception as e:
            logger.warning(f"vision analysis failed, falling back to file names: {e}")

    return get_material_keywords(params.video_materials)


def generate_script(task_id, params, material_hints=None):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        if material_hints is None:
            material_hints = get_material_keywords(params.video_materials)
        if material_hints:
            logger.info(f"using {len(material_hints)} material content hint(s)")
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            material_names=material_hints,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video script.")
        return None

    return video_script


def generate_terms(task_id, params, video_script, material_hints=None):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        video_terms = llm.generate_terms(
            video_subject=params.video_subject,
            video_script=video_script,
            amount=5,
            material_descriptions=material_hints,
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video terms.")
        return None

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(utils.to_json(script_data))


def generate_audio(task_id, params, video_script):
    '''
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    '''
    logger.info("\n\n## generating audio")
    # /audio 和 /subtitle 请求模型不包含 custom_audio_file，
    # 这里统一做兼容读取，避免直调接口时抛属性错误。
    custom_audio_file = getattr(params, "custom_audio_file", None)
    if not custom_audio_file or not os.path.exists(custom_audio_file):
        if custom_audio_file:
            logger.warning(
                f"custom audio file not found: {custom_audio_file}, using TTS to generate audio."
            )
        else:
            logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        subtitle_file = path.join(utils.task_dir(task_id), "subtitle.srt")
        # Resume: reuse a previously synthesized audio.mp3 (skips TTS). Only safe
        # when subtitles are disabled or already on disk, because reusing the file
        # yields no sub_maker for edge-based subtitle timing.
        if (
            _resume_enabled()
            and os.path.exists(audio_file)
            and os.path.getsize(audio_file) > 0
            and (not params.subtitle_enabled or os.path.exists(subtitle_file))
        ):
            reused_duration = math.ceil(voice.get_audio_duration(audio_file))
            if reused_duration > 0:
                logger.info("resuming: reusing existing audio.mp3 (skipping TTS)")
                return audio_file, reused_duration, None
            logger.warning("existing audio.mp3 had zero duration; regenerating")
        sub_maker = voice.tts(
            text=video_script,
            voice_name=voice.parse_voice_name(params.voice_name),
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                """failed to generate audio:
1. check if the language of the voice matches the language of the video script.
2. check if the network is available. If you are in China, it is recommended to use a VPN and enable the global traffic mode.
            """.strip()
            )
            return None, None, None
        audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
        if audio_duration == 0:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to get audio duration.")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to get audio duration from custom audio file.")
            return None, None, None
        return custom_audio_file, audio_duration, None

def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    '''
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    '''
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")

    # Resume: reuse a valid subtitle.srt from a previous run. This also covers
    # the case where audio.mp3 was reused (sub_maker is None) but subtitles
    # already exist on disk.
    if _resume_enabled() and os.path.exists(subtitle_path):
        if subtitle.file_to_subtitles(subtitle_path):
            logger.info("resuming: reusing existing subtitle.srt")
            return subtitle_path

    if sub_maker is None:
        return ""
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    subtitle_fallback = False
    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            subtitle_fallback = True
            logger.warning("subtitle file not found, fallback to whisper")

    if subtitle_provider == "whisper" or subtitle_fallback:
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def _download_stock_videos(task_id, params, video_terms, required_duration):
    """Download stock footage to cover ``required_duration`` seconds.

    Used both for pure stock sources and to top up local uploads. Only the
    pexels/pixabay providers support keyword search; other providers fall back
    to pexels.
    """
    source = (params.stock_source or "pexels").strip().lower()
    if source not in ("pexels", "pixabay"):
        source = "pexels"

    logger.info(f"\n\n## downloading videos from {source}")
    return material.download_videos(
        task_id=task_id,
        search_terms=video_terms,
        source=source,
        video_aspect=params.video_aspect,
        video_contact_mode=params.video_concat_mode,
        # never request less than a single clip so the search still returns hits
        audio_duration=max(required_duration, params.video_clip_duration),
        max_clip_duration=params.video_clip_duration,
    )


def _preprocess_materials(params):
    """Turn ``params.video_materials`` into validated clip paths.

    Covers both uploaded/local-folder assets and scraped images (e.g.
    Pinterest). Returns ``(material_paths, contributed_duration)``.
    """
    material_paths = []
    local_duration = 0.0
    if params.video_materials:
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials,
            clip_duration=params.video_clip_duration,
            allowed_dirs=[params.local_materials_dir]
            if params.local_materials_dir
            else None,
        )
        for material_info in materials:
            material_paths.append(material_info.url)
        # each preprocessed clip contributes up to one clip_duration to the video
        local_duration = len(materials) * params.video_clip_duration
    return material_paths, local_duration


def get_video_materials(task_id, params, video_terms, audio_duration):
    """Gather clip paths and blend in AI-generated clips when enabled.

    AI clips lead the list (most on-theme), then the stock/local footage. If the
    base gathering fails but AI clips were produced, those alone still make a
    video -- generation never hard-fails just because stock was unavailable.
    """
    generated = generate_ai_clips(params, video_terms)
    base = _gather_base_materials(task_id, params, video_terms, audio_duration)

    if generated:
        return generated + (base or [])
    return base


def _gather_base_materials(task_id, params, video_terms, audio_duration):
    required_duration = audio_duration * params.video_count

    # A non-local source uses the chosen stock provider as the primary footage,
    # but any scraped images (e.g. Pinterest in video_materials) are blended in
    # and reduce how much stock we need to download.
    if params.video_source != "local":
        material_paths, local_duration = _preprocess_materials(params)
        remaining = max(required_duration - local_duration, 0.0)
        # When scraped images already cover the duration, still pull at least one
        # stock clip so the video keeps some motion.
        fill_duration = required_duration if not material_paths else remaining

        downloaded_videos = []
        if fill_duration > 0 or not material_paths:
            logger.info(f"\n\n## downloading videos from {params.video_source}")
            downloaded_videos = material.download_videos(
                task_id=task_id,
                search_terms=video_terms,
                source=params.video_source,
                video_aspect=params.video_aspect,
                video_contact_mode=params.video_concat_mode,
                audio_duration=fill_duration,
                max_clip_duration=params.video_clip_duration,
            )
        # Lead with the scraped imagery, then the stock footage.
        all_paths = material_paths + (downloaded_videos or [])
        if not all_paths:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
            )
            return None
        return all_paths

    # Local source: use all uploaded/scraped materials first, then optionally
    # fill the remaining audio duration with stock footage. With no materials at
    # all, fall back to stock for the full duration so a video can still be made.
    # 若只提供了素材文件夹（常见于 API 调用），扫描文件夹生成素材列表。
    if not params.video_materials and params.local_materials_dir:
        params.video_materials = material.list_local_materials(
            params.local_materials_dir
        )

    material_paths, local_duration = _preprocess_materials(params)

    remaining = required_duration - local_duration
    no_uploads = not material_paths
    should_fill = no_uploads or (params.fill_with_stock and remaining > 0)

    if should_fill and video_terms:
        fill_duration = required_duration if no_uploads else remaining
        try:
            downloaded_videos = _download_stock_videos(
                task_id, params, video_terms, fill_duration
            )
            if downloaded_videos:
                logger.info(
                    f"added {len(downloaded_videos)} stock clips to fill ~{fill_duration:.0f}s"
                )
                material_paths.extend(downloaded_videos)
            elif no_uploads:
                logger.error(
                    "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
                )
        except Exception as e:
            # If we already have local clips, a failed top-up is non-fatal.
            logger.warning(f"failed to fill with stock footage: {str(e)}")
    elif should_fill and not video_terms:
        logger.warning("no video terms available, skip stock footage fill")

    if not material_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error(
            "no valid materials found, please check the materials and try again."
        )
        return None
    return material_paths


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path
):
    final_video_paths = []
    combined_video_paths = []
    video_concat_mode = (
        params.video_concat_mode if params.video_count == 1 else VideoConcatMode.random
    )
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths


def start(task_id, params: VideoParams, stop_at: str = "video"):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    # 提前扫描素材文件夹，让脚本生成也能基于文件名理解素材内容。
    if (
        params.video_source == "local"
        and not params.video_materials
        and params.local_materials_dir
    ):
        params.video_materials = material.list_local_materials(
            params.local_materials_dir
        )

    # Resume: a previous run of this task_id may have already produced the
    # script + terms. Reuse them to skip the costly vision + LLM steps.
    cached_script, cached_terms = (None, None)
    if _resume_enabled():
        cached_script, cached_terms = load_resumable_script(task_id)

    # Scrape themed Pinterest photos (if enabled) so they are understood and
    # blended in just like uploaded materials. Always done -- the materials are
    # needed for video assembly even when the script is reused.
    augment_with_pinterest(params)

    # 1. Generate script (or reuse the cached one).
    if cached_script:
        logger.info("resuming: reusing cached script + terms from a previous run")
        video_script = cached_script
        material_hints = None
    else:
        # Understand the local materials once (vision descriptions or file names)
        # and reuse the hints for both the script and the stock search terms.
        material_hints = build_material_hints(params)
        video_script = generate_script(task_id, params, material_hints=material_hints)

    if not video_script or "Error: " in video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms (or reuse the cached ones).
    # Stock footage (used as primary source, or to top up local uploads) needs
    # search keywords. Generate them whenever stock footage may be used.
    if cached_script:
        video_terms = cached_terms
    else:
        video_terms = ""
        local_may_use_stock = params.video_source == "local" and (
            params.fill_with_stock or not params.video_materials
        )
        if params.video_source != "local":
            video_terms = generate_terms(
                task_id, params, video_script, material_hints=material_hints
            )
            if not video_terms:
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                return
        elif local_may_use_stock:
            # In local mode a terms failure is non-fatal: we can still fall back
            # to the uploaded materials only.
            video_terms = generate_terms(
                task_id, params, video_script, material_hints=material_hints
            )
            if not video_terms:
                logger.warning(
                    "failed to generate video terms; stock footage fill will be skipped"
                )

        save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id, params, video_script
    )
    if not audio_file:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id, params, video_terms, audio_duration
    )
    if not downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 仅完整视频生成流程才需要处理视频拼接模式；
    # 这样可以避免 /subtitle 和 /audio 这类请求访问不存在的字段。
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    final_video_paths, combined_video_paths = generate_final_videos(
        task_id, params, downloaded_videos, audio_file, subtitle_path
    )

    if not final_video_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 7. Cross-post to TikTok/Instagram (if enabled)
    cross_post_results = []
    if upload_post.upload_post_service.is_configured() and upload_post.upload_post_service.auto_upload:
        logger.info("\n\n## cross-posting videos to TikTok/Instagram")
        for video_path in final_video_paths:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=params.video_subject or "Check out this video! #shorts #viral"
            )
            cross_post_results.append(result)
            if result.get('success'):
                logger.info(f"✅ Cross-posted: {video_path}")
            else:
                logger.warning(f"⚠️ Failed to cross-post: {video_path} - {result.get('error', 'Unknown error')}")

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_results": cross_post_results if cross_post_results else None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )
    return kwargs


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
