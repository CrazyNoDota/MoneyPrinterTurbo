"""Telegram bot: chat control over the video pipeline.

``python -m app.bot`` starts the worker (long polling). See router.py for
commands, jobs.py for the render queue, api.py for the Bot API client.
"""

from app.bot.api import TelegramAPI  # noqa: F401
from app.bot.autopilot import Autopilot  # noqa: F401
from app.bot.jobs import Job, JobQueue  # noqa: F401
from app.bot.router import Router  # noqa: F401
from app.bot.worker import run  # noqa: F401

__all__ = ["TelegramAPI", "Autopilot", "Job", "JobQueue", "Router", "run"]
