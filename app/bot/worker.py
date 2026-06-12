"""Bot worker entry point: long-poll loop dispatching updates.

Run with ``python -m app.bot``. The loop itself never dies on errors: failed
polls back off briefly and continue; handler errors are contained in Router.
"""

import os
import time

from loguru import logger

from app.bot.api import TelegramAPI
from app.bot.autopilot import Autopilot
from app.bot.jobs import JobQueue
from app.bot.router import Router
from app.utils import utils


def _offset_file() -> str:
    return os.path.join(utils.storage_dir("temp"), "telegram_bot_offset.txt")


def load_offset() -> int:
    try:
        with open(_offset_file(), "r", encoding="utf-8") as f:
            return max(0, int(f.read().strip() or 0))
    except (OSError, ValueError):
        return 0


def save_offset(offset: int):
    try:
        with open(_offset_file(), "w", encoding="utf-8") as f:
            f.write(str(int(offset)))
    except OSError as e:
        logger.warning(f"failed to persist bot offset: {e}")


def run(api: TelegramAPI = None, jobs: JobQueue = None, router: Router = None,
        max_iterations: int = None):
    """Start polling. ``max_iterations`` exists for tests; None = forever."""
    api = api or TelegramAPI()
    if not api.is_configured():
        logger.error(
            "telegram bot token is not set ([telegram].bot_token in config.toml "
            "or TELEGRAM_BOT_TOKEN in .env); bot worker not started"
        )
        return

    jobs = jobs or JobQueue(api)
    jobs.start()
    router = router or Router(api, jobs)
    logger.info("telegram bot worker started (long polling)")

    # Optional scheduled news->video->Telegram cycle (Phase 8 autonomy).
    autopilot = Autopilot.from_config(jobs)
    if autopilot:
        autopilot.start()

    # Persisted across restarts so already-handled updates are not redelivered
    # (a replayed /make would silently burn a whole render's compute/credits).
    offset = load_offset()
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        try:
            updates = api.get_updates(offset=offset)
        except Exception as e:  # noqa: BLE001 - defensive; api already swallows
            logger.warning(f"poll failed: {e}")
            updates = []
        if not updates:
            # get_updates returns [] both on idle and on network errors;
            # a short pause avoids a hot loop when Telegram is unreachable.
            time.sleep(2)
            continue
        for update in updates:
            offset = max(offset, int(update.get("update_id", 0)) + 1)
            router.handle_update(update)
        save_offset(offset)
