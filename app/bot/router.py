"""Update routing: commands, media intake, allowlist.

Security model: only chats in ``[telegram].allowed_chats`` may use the bot.
The default is deny-all -- an unknown chat gets one reply with its chat id so
the owner can add it to config deliberately. Renders cost money/compute, so
no chat is ever auto-approved.
"""

import os
import re
from loguru import logger

from app.config import config
from app.bot.jobs import Job
from app.utils import utils

_SAFE_NAME_RE = re.compile(r"[^\w.\- ]+", re.UNICODE)

_HELP = (
    "Команды:\n"
    "/make <тема> — собрать видео на тему\n"
    "/news — видео из свежей новости\n"
    "/status — очередь и прогресс\n"
    "Пришли фото/видео — добавлю в материалы для роликов.\n"
    "Режимы видео (video_visual_mode в config.toml): quiz, ranking, chat, news, hyperframes."
)


def _safe_filename(name: str, default: str) -> str:
    name = _SAFE_NAME_RE.sub("_", os.path.basename(name or "")).strip(" .")
    return name or default


def _uniquify(path: str) -> str:
    """Suffix -1, -2, ... when the name is taken; uploads must not overwrite."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    for i in range(1, 1000):
        candidate = f"{stem}-{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
    return path


class Router:
    def __init__(self, api, jobs, allowed_chats=None, materials_dir: str = ""):
        self.api = api
        self.jobs = jobs
        raw = (
            allowed_chats
            if allowed_chats is not None
            else config.telegram.get("allowed_chats", [])
        )
        self.allowed_chats = {str(c).strip() for c in raw if str(c).strip()}
        self.materials_dir = materials_dir or utils.storage_dir("local_videos")
        # One unauthorized-chat reply per chat per process -- no reply spam.
        self._denied_notified = set()

    def handle_update(self, update: dict):
        """Route one update; exceptions are logged, never propagated."""
        try:
            message = (update or {}).get("message") or {}
            chat_id = ((message.get("chat") or {}).get("id"))
            if chat_id is None:
                return
            if not self._allowed(chat_id):
                return
            self._handle_message(chat_id, message)
        except Exception as e:  # noqa: BLE001 - one bad update must not stop polling
            logger.error(f"bot update handling failed: {e}")

    def _allowed(self, chat_id) -> bool:
        if str(chat_id) in self.allowed_chats:
            return True
        if chat_id not in self._denied_notified:
            self._denied_notified.add(chat_id)
            logger.warning(f"telegram chat {chat_id} not in allowed_chats; ignoring")
            self.api.send_message(
                chat_id,
                f"⛔ Этот чат не авторизован. chat id: {chat_id} — "
                "добавь его в [telegram].allowed_chats в config.toml.",
            )
        return False

    def _handle_message(self, chat_id, message: dict):
        text = (message.get("text") or "").strip()
        if text.startswith("/"):
            self._handle_command(chat_id, text)
            return
        if any(k in message for k in ("photo", "video", "document")):
            self._handle_media(chat_id, message)
            return
        if text:
            self.api.send_message(chat_id, _HELP)

    def _handle_command(self, chat_id, text: str):
        # "/make@MyBot topic" -> command "make", args "topic"
        head, _, args = text.partition(" ")
        command = head[1:].split("@")[0].lower()
        args = args.strip()

        if command == "make":
            if not args:
                self.api.send_message(chat_id, "Использование: /make <тема видео>")
                return
            position = self.jobs.submit(Job(chat_id=chat_id, kind="make", topic=args))
            self.api.send_message(chat_id, f"📋 В очереди (№{position}): {args}")
        elif command == "news":
            # The news fetch is network-bound; it runs inside the job worker
            # so a slow source never blocks this poll loop.
            position = self.jobs.submit(Job(chat_id=chat_id, kind="news"))
            self.api.send_message(
                chat_id, f"📋 В очереди (№{position}): видео из свежей новости"
            )
        elif command == "status":
            self.api.send_message(chat_id, self.jobs.status())
        else:  # /help, /start, anything unknown
            self.api.send_message(chat_id, _HELP)

    def _handle_media(self, chat_id, message: dict):
        file_id, name = "", ""
        if message.get("photo"):
            sizes = message["photo"]
            file_id = (sizes[-1] or {}).get("file_id", "")  # last entry = largest
            name = f"{(sizes[-1] or {}).get('file_unique_id', 'photo')}.jpg"
        elif message.get("video"):
            video = message["video"]
            file_id = video.get("file_id", "")
            name = video.get("file_name") or f"{video.get('file_unique_id', 'video')}.mp4"
        elif message.get("document"):
            doc = message["document"]
            mime = (doc.get("mime_type") or "").lower()
            if not mime.startswith(("image/", "video/")):
                self.api.send_message(chat_id, "Поддерживаю только фото и видео.")
                return
            file_id = doc.get("file_id", "")
            name = doc.get("file_name") or f"{doc.get('file_unique_id', 'file')}"
        if not file_id:
            return

        dest = _uniquify(
            os.path.join(self.materials_dir, _safe_filename(name, "upload.bin"))
        )
        saved = self.api.download_file(file_id, dest)
        if saved:
            self.api.send_message(
                chat_id, f"📥 Добавил в материалы: {os.path.basename(saved)}"
            )
        else:
            self.api.send_message(chat_id, "⚠️ Не получилось скачать файл, попробуй ещё раз.")
