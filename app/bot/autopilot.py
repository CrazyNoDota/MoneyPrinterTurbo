"""Scheduled autonomous news videos: the full cycle with no manual step.

Every ``autopilot_interval_hours`` the autopilot pulls the freshest news item
and, unless it was already rendered, queues one render per configured language
(``autopilot_languages``); the finished mp4s arrive in ``autopilot_chat_id``
like any bot delivery. Rendered item ids persist across restarts so a reboot
never re-burns compute on yesterday's headline.

Off by default: requires ``[telegram].autopilot_enabled`` and a target chat.
Every cycle is best-effort -- a failed news fetch just waits for the next tick.
"""

import os
import threading
from typing import List, Optional

from loguru import logger

from app.config import config
from app.bot.jobs import Job
from app.utils import utils

# Remember this many rendered news ids (a few weeks of cycles is plenty).
_SEEN_CAP = 200


def _seen_file() -> str:
    return os.path.join(utils.storage_dir("temp"), "autopilot_seen.txt")


class Autopilot:
    """Periodic news->render scheduler feeding the bot's job queue."""

    def __init__(self, jobs, chat_id, languages: List[str], interval_hours: float):
        self.jobs = jobs
        self.chat_id = chat_id
        self.languages = [str(l).strip().lower() for l in languages if str(l).strip()] or [""]
        self.interval = max(0.25, float(interval_hours)) * 3600.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="bot-autopilot")
        self._seen = self._load_seen()

    @staticmethod
    def from_config(jobs) -> Optional["Autopilot"]:
        """Build from ``[telegram]`` config, or None when not enabled/configured."""
        tg = config.telegram
        if not tg.get("autopilot_enabled", False):
            return None
        chat_id = str(tg.get("autopilot_chat_id", "") or "").strip()
        if not chat_id:
            logger.warning("autopilot enabled but autopilot_chat_id is not set; staying off")
            return None
        languages = tg.get("autopilot_languages", []) or [tg.get("language", "")]
        interval = float(tg.get("autopilot_interval_hours", 6) or 6)
        return Autopilot(jobs, chat_id, languages, interval)

    def start(self):
        self._thread.start()
        logger.info(
            f"autopilot started: every {self.interval / 3600:.1f}h -> chat {self.chat_id} "
            f"({', '.join(l or 'default' for l in self.languages)})"
        )

    def stop(self):
        self._stop.set()

    def _loop(self):
        # First tick immediately on startup, then every interval.
        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - a bad cycle must not kill the schedule
                logger.error(f"autopilot tick failed: {e}")
            if self._stop.wait(self.interval):
                return

    def tick(self) -> int:
        """One poll: fetch the freshest item, queue unseen renders. # queued."""
        from app.services import news

        items = news.latest(limit=1)
        if not items:
            logger.info("autopilot: no news available from any source this cycle")
            return 0
        item = items[0]
        if item.id in self._seen:
            logger.debug(f"autopilot: '{item.title}' already rendered; skipping")
            return 0
        queued = 0
        for lang in self.languages:
            self.jobs.submit(Job(
                chat_id=self.chat_id, kind="news", topic=item.title,
                news_item=item, language=lang,
            ))
            queued += 1
        logger.info(f"autopilot: queued '{item.title}' in {queued} language(s)")
        self._remember(item.id)
        return queued

    # -- seen-id persistence ------------------------------------------------

    def _load_seen(self) -> List[str]:
        try:
            with open(_seen_file(), "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()][-_SEEN_CAP:]
        except OSError:
            return []

    def _remember(self, item_id: str):
        self._seen.append(item_id)
        self._seen = self._seen[-_SEEN_CAP:]
        try:
            with open(_seen_file(), "w", encoding="utf-8") as f:
                f.write("\n".join(self._seen))
        except OSError as e:
            logger.warning(f"autopilot: failed to persist seen ids: {e}")
