import os
import sys

from loguru import logger

from app.config import config
from app.utils import utils


_APP_DEFAULTS = {
    "avatar_enabled": False,
    "avatar_provider": "auto",
    "avatar_character": "lisa",
    "avatar_style": "casual-sitting",
    "avatar_voice": "",  # "" -> follow the pipeline TTS voice (news mode lip sync)
    "avatar_portrait": "",
    "avatar_prefer_alpha": True,
    "avatar_alpha_supported": False,
    "avatar_fps": 25,
    "avatar_timeout": 900,
    "avatar_poll_interval": 5,
    "avatar_azure_api_version": "3.1-preview1",
    "avatar_azure_endpoint": "",
    "wav2lip_python": "",
    "wav2lip_repo": "",
    "wav2lip_script": "",
    "wav2lip_weights": "",
    "wav2lip_timeout": 1800,
    "wav2lip_resize_factor": "",
    "news_presenter_scale": 0.38,
    "news_presenter_position": "bottom-right",
    "news_presenter_margin": 0.04,
    "news_threads_profiles": [],
    "news_rss_feeds": ["https://feeds.bbci.co.uk/news/world/rss.xml"],
    "news_fetch_timeout": 15,
    "voice_ru_primary": "ru-RU-DmitryNeural-V2-Male",
    "voice_ru_fallback": "qwen:Russian:ryan-Male",
    "voice_en_primary": "en-US-AndrewMultilingualNeural-V2-Male",
    "voice_en_fallback": "en-US-AndrewNeural-Male",
}

for _key, _value in _APP_DEFAULTS.items():
    config.app.setdefault(_key, _value)


def __init_logger():
    # _log_file = utils.storage_dir("logs/server.log")
    _lvl = config.log_level
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )

    def format_record(record):
        # 获取日志记录中的文件全路径
        file_path = record["file"].path
        # 将绝对路径转换为相对于项目根目录的路径
        relative_path = os.path.relpath(file_path, root_dir)
        # 更新记录中的文件路径
        record["file"].path = f"./{relative_path}"
        # 返回修改后的格式字符串
        # 您可以根据需要调整这里的格式
        _format = (
            "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
            + "<level>{level}</> | "
            + '"{file.path}:{line}":<blue> {function}</> '
            + "- <level>{message}</>"
            + "\n"
        )
        return _format

    # On Windows the console defaults to a legacy codepage (e.g. cp1252) that
    # cannot encode non-Latin text, so logging a Cyrillic/CJK script crashes
    # the handler with UnicodeEncodeError. Force UTF-8 on the std streams.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    logger.remove()

    logger.add(
        sys.stdout,
        level=_lvl,
        format=format_record,
        colorize=True,
    )

    # logger.add(
    #     _log_file,
    #     level=_lvl,
    #     format=format_record,
    #     rotation="00:00",
    #     retention="3 days",
    #     backtrace=True,
    #     diagnose=True,
    #     enqueue=True,
    # )


__init_logger()
