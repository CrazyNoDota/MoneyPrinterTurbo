"""Thin Telegram Bot API client (long polling, no extra dependencies).

Token comes from ``[telegram].bot_token`` in config (or the TELEGRAM_BOT_TOKEN
env var via .env). This must be the product bot's OWN token. Every call is
non-fatal: network/API errors are logged (without the token) and return None,
so the worker loop and handlers never crash on Telegram hiccups.
"""

import os
from typing import List, Optional, Union

import requests
from loguru import logger

from app.config import config


class TelegramAPI:
    def __init__(self, token: str = None, timeout: int = None):
        self.token = (token if token is not None else config.telegram.get("bot_token", "")) or ""
        self.token = self.token.strip()
        self.timeout = int(
            timeout if timeout is not None else config.telegram.get("timeout", 30)
        )

    def is_configured(self) -> bool:
        return bool(self.token)

    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def _scrub(self, text: str) -> str:
        """Strip the token from any string before it can reach a log sink.

        requests exceptions embed the full request URL (token included) in
        str(e), so every logged exception must pass through here.
        """
        text = str(text)
        return text.replace(self.token, "***") if self.token else text

    def _call(self, method: str, payload: dict = None, files: dict = None,
              timeout: int = None) -> Optional[Union[dict, list, bool]]:
        """POST one Bot API method; the ``result`` field on success, None otherwise."""
        try:
            resp = requests.post(
                self._url(method),
                data=payload or {},
                files=files,
                timeout=timeout or self.timeout,
            )
            body = resp.json()
            if not body.get("ok"):
                # description is safe to log; the URL (with token) is not.
                logger.warning(
                    f"telegram {method} failed: {body.get('description', resp.status_code)}"
                )
                return None
            return body.get("result")
        except Exception as e:  # noqa: BLE001 - the bot must survive any API hiccup
            logger.warning(
                f"telegram {method} error: {type(e).__name__}: {self._scrub(e)}"
            )
            return None

    def get_updates(self, offset: int = 0, poll_timeout: int = 50) -> List[dict]:
        result = self._call(
            "getUpdates",
            {"offset": offset, "timeout": poll_timeout, "allowed_updates": '["message"]'},
            # HTTP timeout must outlive the long-poll hold time.
            timeout=poll_timeout + 10,
        )
        return result if isinstance(result, list) else []

    def send_message(self, chat_id, text: str, reply_to: int = None) -> Optional[dict]:
        payload = {"chat_id": chat_id, "text": text}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
            payload["allow_sending_without_reply"] = True
        return self._call("sendMessage", payload)

    def edit_message(self, chat_id, message_id: int, text: str) -> Optional[dict]:
        return self._call(
            "editMessageText",
            {"chat_id": chat_id, "message_id": message_id, "text": text},
        )

    def send_chat_action(self, chat_id, action: str = "upload_video"):
        return self._call("sendChatAction", {"chat_id": chat_id, "action": action})

    # Bot API rejects uploads over 50 MB; checking up front gives a clear log
    # instead of a cryptic API error after a long upload.
    MAX_UPLOAD_BYTES = 50 * 1024 * 1024

    def send_video(self, chat_id, video_path: str, caption: str = "") -> Optional[dict]:
        if not video_path or not os.path.isfile(video_path):
            logger.warning(f"telegram send_video: missing file {video_path}")
            return None
        size = os.path.getsize(video_path)
        if size > self.MAX_UPLOAD_BYTES:
            logger.warning(
                f"telegram send_video: {os.path.basename(video_path)} is "
                f"{size / 1024 / 1024:.0f}MB, over the 50MB Bot API limit"
            )
            return None
        try:
            with open(video_path, "rb") as f:
                return self._call(
                    "sendVideo",
                    {"chat_id": chat_id, "caption": caption[:1024], "supports_streaming": True},
                    files={"video": (os.path.basename(video_path), f, "video/mp4")},
                    # Uploads of multi-minute renders need far more than 30s.
                    timeout=max(self.timeout, 300),
                )
        except OSError as e:
            logger.warning(f"telegram send_video read error: {e}")
            return None

    def download_file(self, file_id: str, dest_path: str) -> str:
        """Download a user upload into ``dest_path``; "" on any failure."""
        info = self._call("getFile", {"file_id": file_id})
        file_path = (info or {}).get("file_path", "")
        if not file_path:
            return ""
        try:
            url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
            resp = requests.get(url, timeout=max(self.timeout, 120))
            if resp.status_code != 200 or not resp.content:
                logger.warning(f"telegram file download failed: {resp.status_code}")
                return ""
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return dest_path
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"telegram file download error: {type(e).__name__}: {self._scrub(e)}"
            )
            return ""
