import glob
import itertools
import io
import os
import random
import gc
import shutil
import subprocess
from contextlib import redirect_stdout
from typing import List
from loguru import logger
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageFont

from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.config import config
from app.services.utils import video_effects
from app.utils import file_security, utils

class SubClippedVideoClip:
    def __init__(self, file_path, start_time=None, end_time=None, width=None, height=None, duration=None):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# Docker 里的 ffmpeg/AAC 组合在默认配置下更容易出现音频质量波动，
# 这里显式抬高音频码率，避免成片阶段因为默认值过低而引入明显失真。
audio_bitrate = "192k"
video_codec = "libx264"
fps = 30
_BGM_EXTENSIONS = (".mp3",)


def get_ffmpeg_binary():
    # 优先复用用户在 config.toml / 环境变量里显式指定的 ffmpeg，可避免
    # Windows 便携包、Docker、自定义安装目录等场景下 PATH 不一致。
    configured_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured_ffmpeg:
        return configured_ffmpeg

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_ffmpeg:
            return bundled_ffmpeg
    except Exception as exc:
        logger.warning(f"failed to resolve bundled ffmpeg binary: {str(exc)}")

    return "ffmpeg"


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # concat demuxer 使用单引号包裹路径，路径中的单引号需要先转义。
    return file_path.replace("'", "'\\''")


def concat_video_clips_with_ffmpeg(
    clip_files: List[str], output_file: str, threads: int, output_dir: str
):
    concat_list_file = os.path.join(output_dir, "ffmpeg-concat-list.txt")
    with open(concat_list_file, "w", encoding="utf-8") as fp:
        for clip_file in clip_files:
            absolute_path = os.path.abspath(clip_file)
            fp.write(f"file '{_escape_ffmpeg_concat_path(absolute_path)}'\n")

    command = [
        get_ffmpeg_binary(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_file,
        "-c:v",
        video_codec,
        "-threads",
        str(threads or 2),
        "-pix_fmt",
        "yuv420p",
        output_file,
    ]

    try:
        # 使用 ffmpeg 只做一次串联与编码，避免 MoviePy 逐段合并时反复重编码，
        # 从而降低画质劣化与颜色偏移风险。
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
    finally:
        delete_files(concat_list_file)


def _sanitize_image_file(image_path: str, output_dir: str = None) -> str:
    # 某些本地图片虽然能被 Pillow 打开，但会因为损坏的 EXIF/eXIf 元数据导致
    # ImageClip 在解析阶段直接抛异常。这里重新导出一份“干净图片”，把坏元数据剥离掉。
    if output_dir:
        # 当源文件在用户自选的素材目录时，把清洗副本写到缓存目录，避免污染原目录。
        base = os.path.splitext(os.path.basename(image_path))[0]
        sanitized_path = os.path.join(output_dir, f"{base}.sanitized.png")
    else:
        image_root, _ = os.path.splitext(image_path)
        sanitized_path = f"{image_root}.sanitized.png"

    with Image.open(image_path) as image:
        image.load()
        # 统一导出为 PNG，避免 JPEG/PNG 不同元数据路径继续把坏块带过去。
        cleaned_image = Image.new(image.mode, image.size)
        cleaned_image.putdata(list(image.getdata()))
        cleaned_image.save(sanitized_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str, output_dir: str = None):
    # 优先直接打开原始图片；如果因为损坏元数据失败，再尝试生成无元数据副本。
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path, output_dir=output_dir)
        return ImageClip(sanitized_path), sanitized_path


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    安静地打开视频文件，避免 MoviePy 2.1.x 把 ffmpeg 探测信息直接打印到 stdout。

    背景：
    当前依赖版本的 `FFMPEG_VideoReader` 内部存在 `print(self.infos)` 和
    `print(ffmpeg command)`，读取无音轨的中间视频时会输出
    `audio_found: False`。这只是输入素材 metadata，不代表最终成片没有音频，
    但会误导 WebUI/终端用户以为生成失败。

    实现：
    1. 只在打开 VideoFileClip 的短窗口内重定向 stdout；
    2. 默认 `audio=False`，因为项目视频素材阶段不需要保留素材原声，
       最终音频会在 `generate_video()` 阶段统一挂载；
    3. 如果依赖库确实输出了内容，降级为 debug 日志，便于必要时排查。
    """
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            logger.debug(f"failed to delete file {file}: {str(e)}")

def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        song_dir = utils.song_dir()
        try:
            resolved_bgm_file = file_security.resolve_path_within_directory(
                song_dir, bgm_file
            )
        except ValueError as exc:
            # API 请求里的 bgm_file 来自用户输入，不能直接把任意绝对路径交给
            # MoviePy 打开。这里强制限制到 resource/songs 目录，阻止读取
            # /etc/passwd、配置文件、密钥等非背景音乐文件。
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, song_dir: {song_dir}, error: {str(exc)}"
            )
            return ""

        if not resolved_bgm_file.lower().endswith(_BGM_EXTENSIONS):
            logger.warning(f"reject unsupported bgm file extension: {resolved_bgm_file}")
            return ""

        return resolved_bgm_file

    if bgm_type == "random":
        suffix = "*.mp3"
        song_dir = utils.song_dir()
        files = glob.glob(os.path.join(song_dir, suffix))
        # 当背景音乐目录为空时，直接回退为“不使用 BGM”，避免 random.choice([]) 抛异常。
        if not files:
            logger.warning(f"no bgm files found in song directory: {song_dir}")
            return ""
        return random.choice(files)

    return ""


def normalize_to_aspect(clip, video_width: int, video_height: int):
    """Resize a clip to exactly ``video_width`` x ``video_height``.

    If the aspect ratios match it scales directly; otherwise it scales to fit
    and letterboxes the remainder with black. Shared by ``combine_videos`` and
    the hyperframes assembler so every stitched segment is the same size (the
    ffmpeg concat demuxer requires uniform resolution).
    """
    clip_w, clip_h = clip.size
    if clip_w == video_width and clip_h == video_height:
        return clip

    clip_ratio = clip_w / clip_h
    video_ratio = video_width / video_height
    if clip_ratio == video_ratio:
        return clip.resized(new_size=(video_width, video_height))

    if clip_ratio > video_ratio:
        scale_factor = video_width / clip_w
    else:
        scale_factor = video_height / clip_h
    new_width = int(clip_w * scale_factor)
    new_height = int(clip_h * scale_factor)

    background = ColorClip(
        size=(video_width, video_height), color=(0, 0, 0)
    ).with_duration(clip.duration)
    clip_resized = clip.resized(new_size=(new_width, new_height)).with_position("center")
    return CompositeVideoClip([background, clip_resized])


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
) -> str:
    audio_clip = AudioFileClip(audio_file)
    try:
        # 这里只需要读取旁白音频时长来决定素材视频拼接长度；后续不会再使用
        # audio_clip。读取完成后立即关闭，避免早退或异常路径泄漏文件句柄。
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")

    # 兼容 API 直接调用时未传转场模式的情况，避免后续访问 .value 时崩溃。
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = _open_video_clip_quietly(video_path)
        clip_duration = clip.duration
        clip_w, clip_h = clip.size
        close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + max_clip_duration, clip_duration)

            # 保留所有有效分段。
            # 这样既不会丢掉“整段视频本身就短于 max_clip_duration”的素材，
            # 也不会吞掉长视频最后剩下的一小段尾部内容。
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    # random subclipped_items order
    if video_concat_mode.value == VideoConcatMode.random.value:
        random.shuffle(subclipped_items)
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    for i, subclipped_item in enumerate(subclipped_items):
        if video_duration > audio_duration:
            break
        
        logger.debug(f"processing clip {i+1}: {subclipped_item.width}x{subclipped_item.height}, current duration: {video_duration:.2f}s, remaining: {audio_duration - video_duration:.2f}s")
        
        try:
            clip = _open_video_clip_quietly(subclipped_item.file_path).subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            clip_duration = clip.duration
            # Not all videos are same size, so we need to resize them
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                logger.debug(f"resizing clip, source: {clip_w}x{clip_h}, target: {video_width}x{video_height}")
                clip = normalize_to_aspect(clip, video_width, video_height)

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            if transition_value in (None, VideoTransitionMode.none.value):
                clip = clip
            elif transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, 1)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, 1, shuffle_side),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)
                
            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i+1}.mp4"
            clip.write_videofile(clip_file, logger=None, fps=fps, codec=video_codec)

            # Store clip duration before closing
            clip_duration_saved = clip.duration
            close_clip(clip)

            processed_clips.append(SubClippedVideoClip(file_path=clip_file, duration=clip_duration_saved, width=clip_w, height=clip_h))
            video_duration += clip_duration_saved
            
        except Exception as e:
            logger.error(f"failed to process clip: {str(e)}")
    
    # loop processed clips until the video duration matches or exceeds the audio duration.
    if video_duration < audio_duration:
        logger.warning(f"video duration ({video_duration:.2f}s) is shorter than audio duration ({audio_duration:.2f}s), looping clips to match audio length.")
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= audio_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, looped {len(processed_clips)-len(base_clips)} clips")
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        return combined_video_path
    
    # if there is only one clip, use it directly
    if len(processed_clips) == 1:
        logger.info("using single clip directly")
        shutil.copy(processed_clips[0].file_path, combined_video_path)
        delete_files([processed_clips[0].file_path])
        logger.info("video combining completed")
        return combined_video_path

    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    concat_video_clips_with_ffmpeg(
        clip_files=clip_files,
        output_file=combined_video_path,
        threads=threads,
        output_dir=output_dir,
    )
    
    # clean temp files
    delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


_KARAOKE_PUNCTUATION = ".!?,;:。！？，；：…"


def group_words_into_chunks(
    words,
    max_words: int = 4,
    pause_threshold: float = 0.5,
):
    """Group per-word timings into short karaoke caption chunks.

    ``words`` is a list of ``{"text", "start", "end"}`` dicts (seconds). Chunks
    break on (a) reaching ``max_words``, (b) a gap > ``pause_threshold`` seconds
    between consecutive words, or (c) sentence-ending punctuation on a word.
    Returns a list of chunks, each ``{"start", "end", "words": [...]}`` where the
    inner words preserve their individual ``{text, start, end}``.
    """
    chunks = []
    current = []

    def flush():
        if not current:
            return
        chunks.append(
            {
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "words": list(current),
            }
        )
        current.clear()

    prev_end = None
    for w in words:
        text = (w.get("text") or "").strip()
        if not text:
            continue
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start))
        if end < start:
            end = start

        if current:
            gap = start - prev_end if prev_end is not None else 0.0
            if len(current) >= max_words or gap > pause_threshold:
                flush()

        current.append({"text": text, "start": start, "end": end})
        prev_end = end

        # Break after sentence-ending punctuation so chunks read naturally.
        if text[-1] in _KARAOKE_PUNCTUATION:
            flush()

    flush()
    return chunks


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # Create ImageFont
    font = ImageFont.truetype(font, fontsize)

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    processed = True

    _wrapped_lines_ = []
    words = text.split(" ")
    _txt_ = ""
    for word in words:
        _before = _txt_
        _txt_ += f"{word} "
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            if _txt_.strip() == word.strip():
                processed = False
                break
            _wrapped_lines_.append(_before)
            _txt_ = f"{word} "
    _wrapped_lines_.append(_txt_)
    if processed:
        _wrapped_lines_ = [line.strip() for line in _wrapped_lines_]
        result = "\n".join(_wrapped_lines_).strip()
        height = len(_wrapped_lines_) * height
        return result, height

    _wrapped_lines_ = []
    chars = list(text)
    _txt_ = ""
    for word in chars:
        _txt_ += word
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            _wrapped_lines_.append(_txt_)
            _txt_ = ""
    _wrapped_lines_.append(_txt_)
    result = "\n".join(_wrapped_lines_).strip()
    height = len(_wrapped_lines_) * height
    return result, height


# ---------------------------------------------------------------------------
# WP4: BGM ducking + transition SFX (config-gated, non-fatal)
# ---------------------------------------------------------------------------

# Gap (seconds) below which two adjacent word/phrase intervals are treated as a
# single continuous speech span (avoids the BGM swelling for tiny inter-word
# silences). Ramp length is the linear fade applied at each speech edge.
_SPEECH_MERGE_GAP = 0.7
_DUCK_RAMP = 0.2  # ~200 ms linear ramp


def _srt_timestamp_to_seconds(ts: str) -> float:
    """Parse an SRT ``HH:MM:SS,mmm`` timestamp into float seconds."""
    ts = ts.strip().replace(".", ",")
    hms, _, ms = ts.partition(",")
    parts = hms.split(":")
    if len(parts) != 3:
        raise ValueError(f"bad srt timestamp: {ts!r}")
    h, m, s = (int(p) for p in parts)
    millis = int(ms) if ms else 0
    return h * 3600 + m * 60 + s + millis / 1000.0


def _intervals_from_words(words: list) -> list:
    """Return raw ``(start, end)`` tuples from a word-timing sidecar list."""
    out = []
    for w in words or []:
        try:
            start = float(w["start"])
            end = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            out.append((start, end))
    return out


def _intervals_from_srt(subtitle_path: str) -> list:
    """Return raw ``(start, end)`` phrase tuples parsed from an SRT file."""
    out = []
    try:
        from app.services import subtitle as subtitle_service

        for _idx, times, _text in subtitle_service.file_to_subtitles(subtitle_path):
            # ``times`` looks like "00:00:01,000 --> 00:00:03,500"
            if "-->" not in times:
                continue
            lo_s, _, hi_s = times.partition("-->")
            start = _srt_timestamp_to_seconds(lo_s)
            end = _srt_timestamp_to_seconds(hi_s)
            if end > start:
                out.append((start, end))
    except Exception as e:
        logger.warning(f"failed to parse srt for ducking envelope: {str(e)}")
        return []
    return out


def merge_speech_spans(intervals: list, gap: float = _SPEECH_MERGE_GAP) -> list:
    """Merge ``(start, end)`` intervals into speech spans.

    Intervals are sorted; any two whose gap is < ``gap`` seconds are merged
    into one continuous span. Returns a sorted list of non-overlapping
    ``(start, end)`` tuples.
    """
    cleaned = [(float(a), float(b)) for a, b in intervals if b > a]
    if not cleaned:
        return []
    cleaned.sort(key=lambda iv: iv[0])
    spans = [list(cleaned[0])]
    for start, end in cleaned[1:]:
        last = spans[-1]
        if start - last[1] < gap:
            if end > last[1]:
                last[1] = end
        else:
            spans.append([start, end])
    return [(s, e) for s, e in spans]


def build_speech_spans(subtitle_path: str, karaoke_words: list = None) -> list:
    """Build the voice-activity speech spans for BGM ducking.

    Prefers the word-timing sidecar (passed in or loaded next to the SRT);
    falls back to SRT phrase ranges. Returns ``[]`` when no timing data is
    available so callers can keep today's flat-mix behaviour.
    """
    intervals = []
    words = karaoke_words
    if not words and subtitle_path:
        try:
            from app.services import subtitle as subtitle_service

            words = subtitle_service.load_words_sidecar(subtitle_path)
        except Exception as e:
            logger.warning(f"failed to load words sidecar for ducking: {str(e)}")
            words = []
    if words:
        intervals = _intervals_from_words(words)
    if not intervals and subtitle_path and os.path.exists(subtitle_path):
        intervals = _intervals_from_srt(subtitle_path)
    return merge_speech_spans(intervals)


def make_duck_gain_fn(spans: list, base_volume: float, duck_volume: float,
                      ramp: float = _DUCK_RAMP):
    """Return ``g(t)`` mapping time → BGM gain, with linear ramps at span edges.

    ``t`` may be a scalar or a numpy array. Gain is ``duck_volume`` while
    speech is active, ``base_volume`` in gaps/before/after, with a linear
    ramp of length ``ramp`` seconds on each side of every speech edge
    (ramp centred on the edge: half before, half after).
    """
    import numpy as np

    starts = np.array([s for s, _ in spans], dtype=float)
    ends = np.array([e for _, e in spans], dtype=float)
    half = max(ramp, 1e-6) / 2.0

    def g(t):
        t_arr = np.asarray(t, dtype=float)
        # speech_level(t): 1.0 fully inside a speech span, 0.0 fully outside,
        # linearly ramping across the +/- half window at each edge.
        level = np.zeros(t_arr.shape, dtype=float)
        for s, e in zip(starts, ends):
            # rising edge centred at s, falling edge centred at e
            rise = np.clip((t_arr - (s - half)) / (2 * half), 0.0, 1.0)
            fall = np.clip(((e + half) - t_arr) / (2 * half), 0.0, 1.0)
            contrib = np.minimum(rise, fall)
            level = np.maximum(level, contrib)
        gain = base_volume + (duck_volume - base_volume) * level
        if np.isscalar(t) or np.ndim(t) == 0:
            return float(gain)
        return gain

    return g


def apply_bgm_ducking(bgm_clip, spans: list, base_volume: float,
                      duck_volume: float):
    """Apply a time-varying ducking gain to ``bgm_clip``.

    Returns a new clip whose per-sample amplitude is scaled by the gain
    function. Falls back (returns the input unchanged) on any error.
    """
    import numpy as np

    if not spans:
        return bgm_clip
    g = make_duck_gain_fn(spans, base_volume, duck_volume)

    def _apply(get_frame, t):
        frame = get_frame(t)
        gain = g(t)
        gain_arr = np.asarray(gain, dtype=float)
        if frame.ndim == 2:
            gain_arr = gain_arr.reshape(-1, 1)
        return frame * gain_arr

    return bgm_clip.transform(_apply, keep_duration=True)


def compute_sfx_cut_times(spans: list, total_duration: float = None,
                          min_gap: float = 0.8) -> list:
    """Derive transition-SFX cut points from speech spans.

    A cut is placed at the start of each speech span (a new narration
    beat / scene). ``t=0`` is skipped, cuts within ``min_gap`` seconds of
    each other are de-duplicated, and anything past ``total_duration`` is
    dropped. Returns a sorted list of cut times in seconds.
    """
    cuts = []
    for s, _e in spans:
        if s <= 0.05:
            continue
        if total_duration is not None and s >= total_duration:
            continue
        if cuts and s - cuts[-1] < min_gap:
            continue
        cuts.append(s)
    return cuts


def _overlay_transition_sfx(audio_clip, spans: list, total_duration: float):
    """Overlay a whoosh SFX at each scene-cut point (config-gated, non-fatal).

    Returns a new composite audio clip with the SFX mixed in, or the
    original ``audio_clip`` unchanged when disabled, when no cut points
    exist, or when the SFX asset is missing.
    """
    if not bool(config.app.get("sfx_enabled", True)):
        return audio_clip
    cuts = compute_sfx_cut_times(spans, total_duration=total_duration)
    if not cuts:
        logger.debug("transition sfx: no cut points; skipping")
        return audio_clip

    from app.services.sfx import get_sfx_file

    whoosh_path = get_sfx_file("whoosh")
    if not whoosh_path:
        logger.debug("transition sfx: whoosh asset missing; skipping")
        return audio_clip

    sfx_volume = float(config.app.get("sfx_volume", 0.6))
    layers = [audio_clip]
    placed = 0
    for t in cuts:
        try:
            clip = (
                AudioFileClip(whoosh_path)
                .with_effects([afx.MultiplyVolume(sfx_volume)])
                .with_start(t)
            )
            layers.append(clip)
            placed += 1
        except Exception as e:
            logger.debug(f"transition sfx: failed at t={t:.2f}s: {str(e)}")
    if placed == 0:
        return audio_clip
    logger.info(f"overlaid {placed} transition sfx cut(s)")
    return CompositeAudioClip(layers)


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    subtitle_ranges: List[tuple] = None,
):
    """Compose the final video.

    ``subtitle_ranges`` (optional) is a list of ``(start, end)`` seconds; when
    given, only subtitle lines whose timing falls inside one of those ranges are
    burned in. Used by hyperframes *mixed* mode to caption footage scenes while
    leaving motion-graphics scenes (which already render the narration) untouched.
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "Anton-Regular.ttf"
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # 兼容历史参数：API 里 `text_background_color` 既可能是布尔值，
        # 也可能是实际颜色字符串。统一在这里归一化，避免把 True/False
        # 直接传给 TextClip 后出现不可预期的渲染结果。
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    # Subtitle look:
    #   karaoke -> CapCut/TikTok word-by-word highlight (default)
    #   tiktok  -> ALL-CAPS, heavy outline + drop shadow (loud viral "shorts" look)
    #   outline -> no box, thick outline (clean modern caption)
    #   shadow  -> outline + a soft drop shadow for extra pop
    #   box     -> the legacy solid background box (uses text_background_color)
    subtitle_style = (getattr(params, "subtitle_style", "") or "tiktok").strip().lower()
    # Karaoke loads a word-timing sidecar; if it is missing/malformed we degrade
    # to the "tiktok" phrase captions. ``render_style`` is what the per-line
    # renderer below actually draws (karaoke uses its own dedicated renderer).
    render_style = "tiktok" if subtitle_style == "karaoke" else subtitle_style

    def split_long_subtitle(subtitle_item):
        """Split a long subtitle range into sequential readable caption chunks."""
        start, end = subtitle_item[0][0], subtitle_item[0][1]
        phrase = subtitle_item[1]
        words = phrase.split()
        if len(words) <= 9:
            return [subtitle_item]

        chunks = []
        current = []
        current_chars = 0
        for word in words:
            next_chars = current_chars + len(word) + (1 if current else 0)
            if current and (len(current) >= 8 or next_chars > 46):
                chunks.append(" ".join(current))
                current = [word]
                current_chars = len(word)
            else:
                current.append(word)
                current_chars = next_chars
        if current:
            chunks.append(" ".join(current))
        if len(chunks) <= 1:
            return [subtitle_item]

        total_words = sum(len(c.split()) for c in chunks)
        total_duration = max(end - start, 0.01)
        cursor = start
        split_items = []
        for i, chunk in enumerate(chunks):
            if i == len(chunks) - 1:
                chunk_end = end
            else:
                share = len(chunk.split()) / max(total_words, 1)
                chunk_end = min(end, cursor + total_duration * share)
            split_items.append(((cursor, chunk_end), chunk))
            cursor = chunk_end
        return split_items

    def create_text_clips(subtitle_item):
        """Return the positioned clip(s) for one subtitle line.

        Returns a list because the ``shadow`` style stacks a shadow clip behind
        the text.
        """
        font_size = int(params.font_size)
        phrase = subtitle_item[1]
        # The punchy "tiktok" look is all-caps. Uppercase before wrapping so the
        # line-width math uses the (wider) capital glyphs and nothing overflows.
        if render_style == "tiktok":
            phrase = phrase.upper()
        max_width = video_width * 0.9
        max_caption_height = video_height * 0.28
        min_font_size = max(28, int(params.font_size * 0.55))

        while True:
            wrapped_txt, txt_height = wrap_text(
                phrase, max_width=max_width, font=font_path, fontsize=font_size
            )
            interline = int(font_size * 0.25)
            line_count = wrapped_txt.count("\n") + 1
            vertical_padding = int(font_size * 0.35)
            rendered_height = txt_height + vertical_padding + (interline * line_count)
            if rendered_height <= max_caption_height or font_size <= min_font_size:
                break
            font_size = max(min_font_size, font_size - 4)

        # MoviePy 在 `method=label` 下会自动收缩文本框高度，遇到多行字幕、
        # 描边或背景色时，容易把最后一行的下半部分裁掉。这里显式传入
        # 一个更保守的高度，把行间距和额外上下留白一并算进去，保证字幕
        # 背景框与文字本身都能完整渲染出来。
        size = (
            int(max_width),
            int(rendered_height),
        )

        # Only the legacy "box" style draws a background rectangle. Without a box
        # the text needs a strong outline to stay legible over any footage, so we
        # honor a larger user-set stroke but enforce a sensible minimum.
        provided_stroke = int(params.stroke_width)
        if render_style == "box":
            bg_color = resolve_subtitle_background_color()
            stroke_width = provided_stroke
        elif render_style == "tiktok":
            # Loud captions need a heavy outline so the words punch off any footage.
            bg_color = None
            stroke_width = max(provided_stroke, max(4, round(font_size * 0.09)))
        else:
            bg_color = None
            stroke_width = max(provided_stroke, max(2, round(font_size * 0.05)))

        common = dict(
            text=wrapped_txt,
            font=font_path,
            font_size=font_size,
            interline=interline,
            size=size,
            text_align="center",
        )

        main_clip = TextClip(
            color=params.text_fore_color,
            bg_color=bg_color,
            stroke_color=params.stroke_color,
            stroke_width=stroke_width,
            **common,
        )

        start, end = subtitle_item[0][0], subtitle_item[0][1]
        duration = end - start
        clip_h, clip_w = main_clip.h, main_clip.w

        if params.subtitle_position == "bottom":
            y = video_height * 0.95 - clip_h
        elif params.subtitle_position == "top":
            y = video_height * 0.05
        elif params.subtitle_position == "custom":
            margin = 10  # Additional margin, in pixels
            max_y = video_height - clip_h - margin
            min_y = margin
            y = (video_height - clip_h) * (params.custom_position / 100)
            y = max(min_y, min(y, max_y))  # constrain within screen bounds
        else:  # center
            y = (video_height - clip_h) / 2
        x = (video_width - clip_w) / 2

        def _timed(clip, pos):
            return (
                clip.with_start(start)
                .with_end(end)
                .with_duration(duration)
                .with_position(pos)
            )

        clips = []
        if render_style in ("shadow", "tiktok"):
            offset = max(2, round(font_size * 0.06))
            shadow_clip = TextClip(
                color="#000000",
                bg_color=None,
                stroke_color="#000000",
                stroke_width=stroke_width,
                **common,
            )
            clips.append(_timed(shadow_clip, (x + offset, y + offset)))
        clips.append(_timed(main_clip, (x, y)))
        return clips

    highlight_color = (
        getattr(params, "subtitle_highlight_color", "") or "#FFE600"
    ).strip()

    def create_karaoke_clips(chunk):
        """Render one karaoke caption chunk as word-by-word highlighted clips.

        Each word is rendered twice (base color + highlight color); the base
        glyphs are laid out as a centered single row, then for every word
        interval the active word is swapped for its slightly-scaled highlight
        copy. Heavy outline + drop shadow are kept so text stays legible. Reuses
        already-rendered TextClips via lightweight ``.with_*`` views, so the cost
        is ~2 renders per word, not per (word x interval).
        """
        words = chunk.get("words") or []
        if not words:
            return []

        chunk_start = chunk["start"]
        chunk_end = chunk["end"]

        # Shrink the font until the row fits the safe width.
        max_width = video_width * 0.92
        font_size = int(params.font_size)
        min_font_size = max(28, int(params.font_size * 0.55))
        gap = 0
        base_clips = []
        highlight_clips = []
        widths = []

        provided_stroke = int(params.stroke_width)

        def _build(size_px):
            stroke_width = max(provided_stroke, max(4, round(size_px * 0.09)))
            bases, highs, ws = [], [], []
            for w in words:
                token = w["text"].upper()
                common = dict(
                    text=token,
                    font=font_path,
                    font_size=size_px,
                    stroke_width=stroke_width,
                )
                base = TextClip(
                    color=params.text_fore_color,
                    stroke_color=params.stroke_color,
                    **common,
                )
                high = TextClip(
                    color=highlight_color,
                    stroke_color=params.stroke_color,
                    **common,
                )
                bases.append(base)
                highs.append(high)
                ws.append(base.w)
            return bases, highs, ws, stroke_width

        while True:
            base_clips, highlight_clips, widths, stroke_width = _build(font_size)
            gap = max(6, int(font_size * 0.18))
            total_width = sum(widths) + gap * (len(words) - 1)
            if total_width <= max_width or font_size <= min_font_size:
                break
            # Dispose of the oversized renders before retrying smaller.
            for c in base_clips + highlight_clips:
                try:
                    c.close()
                except Exception:
                    pass
            font_size = max(min_font_size, font_size - 4)

        row_height = max((c.h for c in base_clips), default=font_size)
        total_width = sum(widths) + gap * (len(words) - 1)

        # Vertical placement mirrors the phrase-caption logic.
        if params.subtitle_position == "bottom":
            y = video_height * 0.95 - row_height
        elif params.subtitle_position == "top":
            y = video_height * 0.05
        elif params.subtitle_position == "custom":
            margin = 10
            max_y = video_height - row_height - margin
            y = (video_height - row_height) * (params.custom_position / 100)
            y = max(margin, min(y, max_y))
        else:  # center
            y = (video_height - row_height) / 2

        x0 = (video_width - total_width) / 2
        # Pre-compute each word's left edge in the centered row.
        x_positions = []
        cursor = x0
        for w in widths:
            x_positions.append(cursor)
            cursor += w + gap

        out = []
        for i, w in enumerate(words):
            w_start = max(w["start"], chunk_start)
            w_end = w["end"]
            if i == len(words) - 1:
                w_end = max(w_end, chunk_end)
            if w_end <= w_start:
                w_end = w_start + 0.05

            # Render the whole row for this word interval; only word ``i`` is
            # highlighted (and slightly enlarged). The heavy per-glyph stroke
            # keeps the text legible over any footage.
            for j in range(len(words)):
                active = j == i
                src = highlight_clips[j] if active else base_clips[j]
                wy = y
                px = x_positions[j]
                if active:
                    src = src.resized(1.15)
                    wy = y - (src.h - row_height) / 2
                    px = x_positions[j] - (src.w - widths[j]) / 2
                out.append(
                    src.with_position((px, wy)).with_start(w_start).with_end(w_end)
                )
        return out

    video_clip = _open_video_clip_quietly(video_path)
    audio_clip = AudioFileClip(audio_path).with_effects(
        [afx.MultiplyVolume(params.voice_volume)]
    )

    def make_textclip(text):
        return TextClip(
            text=text,
            font=font_path,
            font_size=params.font_size,
        )

    def _in_subtitle_ranges(item) -> bool:
        # No filter => caption everything (default footage pipeline). Otherwise keep
        # only lines whose midpoint lands inside an allowed (footage) range.
        if subtitle_ranges is None:
            return True
        start, end = item[0][0], item[0][1]
        mid = (start + end) / 2
        return any(lo <= mid <= hi for lo, hi in subtitle_ranges)

    def _chunk_in_subtitle_ranges(chunk) -> bool:
        if subtitle_ranges is None:
            return True
        mid = (chunk["start"] + chunk["end"]) / 2
        return any(lo <= mid <= hi for lo, hi in subtitle_ranges)

    karaoke_words = []
    if subtitle_style == "karaoke" and subtitle_path:
        try:
            from app.services import subtitle as subtitle_service

            karaoke_words = subtitle_service.load_words_sidecar(subtitle_path)
        except Exception as e:
            logger.warning(f"failed to load karaoke word timings: {str(e)}")
            karaoke_words = []
        if not karaoke_words:
            logger.warning(
                "karaoke style requested but no usable word-timing sidecar; "
                "falling back to phrase captions"
            )

    if subtitle_style == "karaoke" and karaoke_words:
        text_clips = []
        chunks = group_words_into_chunks(karaoke_words)
        for chunk in chunks:
            if not _chunk_in_subtitle_ranges(chunk):
                continue
            try:
                text_clips.extend(create_karaoke_clips(chunk))
            except Exception as e:
                logger.warning(f"failed to render karaoke chunk: {str(e)}")
        if text_clips:
            video_clip = CompositeVideoClip([video_clip, *text_clips])
    elif subtitle_path and os.path.exists(subtitle_path):
        sub = SubtitlesClip(
            subtitles=subtitle_path, encoding="utf-8", make_textclip=make_textclip
        )
        text_clips = []
        for item in sub.subtitles:
            if not _in_subtitle_ranges(item):
                continue
            for split_item in split_long_subtitle(item):
                text_clips.extend(create_text_clips(subtitle_item=split_item))
        if text_clips:
            video_clip = CompositeVideoClip([video_clip, *text_clips])

    # WP4: voice-activity speech spans drive both BGM ducking and the
    # transition-SFX cut points. Built once; empty list => no timing data
    # available, so we keep today's flat-mix behaviour.
    speech_spans = []
    try:
        speech_spans = build_speech_spans(subtitle_path, karaoke_words)
    except Exception as e:
        logger.warning(f"failed to build speech spans (ducking disabled): {str(e)}")
        speech_spans = []

    bgm_ducking_enabled = bool(config.app.get("bgm_ducking_enabled", True))
    bgm_duck_volume = float(config.app.get("bgm_duck_volume", 0.15))

    bgm_file = get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)
    if bgm_file:
        try:
            bgm_clip = AudioFileClip(bgm_file).with_effects(
                [
                    afx.AudioFadeOut(3),
                    afx.AudioLoop(duration=video_clip.duration),
                ]
            )
            if bgm_ducking_enabled and speech_spans:
                # Time-varying ducking gain (params.bgm_volume in gaps,
                # bgm_duck_volume under speech, ~200 ms ramps).
                bgm_clip = apply_bgm_ducking(
                    bgm_clip, speech_spans, params.bgm_volume, bgm_duck_volume
                )
                logger.info(
                    f"bgm ducking active over {len(speech_spans)} speech span(s)"
                )
            else:
                # Flat fallback: no timing data or ducking disabled.
                bgm_clip = bgm_clip.with_effects(
                    [afx.MultiplyVolume(params.bgm_volume)]
                )
            audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
        except Exception as e:
            logger.error(f"failed to add bgm: {str(e)}")

    # WP4: overlay transition SFX (whoosh) at scene-cut points derived from
    # the speech spans. Config-gated and fully non-fatal.
    try:
        audio_clip = _overlay_transition_sfx(
            audio_clip, speech_spans, video_clip.duration
        )
    except Exception as e:
        logger.warning(f"failed to overlay transition sfx: {str(e)}")

    video_clip = video_clip.with_audio(audio_clip)
    # 显式沿用输入音频的采样率；如果取不到，再回退到 MoviePy 默认的 44100Hz。
    # 这样可以减少不同运行环境，尤其是 Docker 环境中再次重采样带来的音质波动。
    output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
    video_clip.write_videofile(
        output_file,
        audio_codec=audio_codec,
        audio_fps=output_audio_fps,
        audio_bitrate=audio_bitrate,
        temp_audiofile_path=output_dir,
        threads=params.n_threads or 2,
        logger=None,
        fps=fps,
    )
    video_clip.close()
    del video_clip


def preprocess_video(materials: List[MaterialInfo], clip_duration=4, allowed_dirs=None):
    # WebUI 在某些二次生成场景下可能传入空素材列表，这里直接返回空结果，避免抛出 NoneType 异常。
    if not materials:
        return []

    # 仅返回通过预处理校验的素材，避免低分辨率图片继续进入后续的视频合成流程。
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    # 允许的素材根目录：默认上传缓存目录，外加用户自选的素材文件夹。
    search_dirs = [os.path.realpath(local_videos_dir)]
    for extra in allowed_dirs or []:
        extra = (extra or "").strip()
        if extra and os.path.isdir(extra):
            search_dirs.append(os.path.realpath(extra))

    for material in materials:
        if not material.url:
            continue

        # 依次在各允许目录内解析素材路径，命中任意一个即视为安全。
        material_source_path = None
        for base_dir in search_dirs:
            try:
                material_source_path = file_security.resolve_path_within_directory(
                    base_dir, material.url
                )
                break
            except ValueError:
                continue

        if not material_source_path:
            # local video_source 的素材路径来自 API 参数，必须限制在允许目录内。
            # 允许用户传文件名/绝对路径，但不允许逃逸到其他目录，避免任意文件读取
            # 或通过 MoviePy 探测本地敏感文件。
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"allowed_dirs: {search_dirs}"
            )
            continue

        # 源文件在缓存目录之外（用户自选文件夹）时，把生成的产物写到缓存目录，
        # 既保持安全模型，又不会在用户的素材文件夹里留下转码文件。
        is_external = (
            os.path.commonpath([search_dirs[0], material_source_path]) != search_dirs[0]
        )
        artifact_dir = local_videos_dir if is_external else None

        ext = utils.parse_extension(material_source_path)
        try:
            # 图片素材直接按图片方式读取，避免先走 VideoFileClip 误判后触发不稳定的回退分支。
            if ext in const.FILE_TYPE_IMAGES:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path, output_dir=artifact_dir
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # 非标准扩展名或探测失败时再回退到图片模式，兼容历史上直接传本地图片路径的情况。
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path, output_dir=artifact_dir
                )
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if width < 480 or height < 480:
                logger.warning(f"low resolution material: {width}x{height}, minimum 480x480 required")
                # 探测到低分辨率素材后立即关闭资源，并且不要把该素材返回给后续流程。
                close_clip(clip)
                continue

            if ext in const.FILE_TYPE_IMAGES:
                logger.info(f"processing image: {material_source_path}")
                # 探测尺寸时已经打开过一次素材，这里先释放探测句柄，再重新创建用于导出的图片 clip。
                close_clip(clip)
                # Create an image clip and set its duration to 3 seconds
                clip = (
                    ImageClip(material_source_path)
                    .with_duration(clip_duration)
                    .with_position("center")
                )
                # Apply a zoom effect using the resize method.
                # A lambda function is used to make the zoom effect dynamic over time.
                # The zoom effect starts from the original size and gradually scales up to 120%.
                # t represents the current time, and clip.duration is the total duration of the clip (3 seconds).
                # Note: 1 represents 100% size, so 1.2 represents 120% size.
                zoom_clip = clip.resized(
                    lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
                )

                # Optionally, create a composite video clip containing the zoomed clip.
                # This is useful when you want to add other elements to the video.
                final_clip = CompositeVideoClip([zoom_clip])

                # Output the video to a file. 外部素材目录的图片转码结果写到缓存目录，
                # 避免在用户的素材文件夹里产生 .mp4 文件。
                if artifact_dir:
                    image_base = os.path.basename(material_source_path)
                    video_file = os.path.join(artifact_dir, f"{image_base}.mp4")
                else:
                    video_file = f"{material_source_path}.mp4"
                final_clip.write_videofile(video_file, fps=30, logger=None)
                close_clip(clip)
                close_clip(final_clip)
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # 普通视频素材只需要读取尺寸做校验，校验完成后立即释放句柄即可。
                close_clip(clip)
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
