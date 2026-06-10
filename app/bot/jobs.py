"""Render job queue for the Telegram bot.

One render at a time (the pipeline is CPU/GPU heavy); jobs queue up FIFO.
The worker thread reports progress by editing a status message in the chat,
sends the finished mp4, and turns every failure into a chat message instead
of a crash.
"""

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoParams
from app.services import llm
from app.services import state as sm


@dataclass
class Job:
    chat_id: int
    kind: str  # "make" | "news"
    topic: str = ""
    news_item: object = None  # NewsItem for kind == "news"
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _build_params(job: Job) -> Optional[VideoParams]:
    """VideoParams for a bot job, or None when there is nothing to render.

    For "news" jobs the news fetch happens HERE, on the job worker thread --
    never in the poll loop -- and the fetched item is pinned onto the job so
    the status label can show the headline.
    """
    language = str(config.telegram.get("language", "") or "")
    subject = job.topic
    script = ""
    if job.kind == "news":
        if job.news_item is None:
            from app.services import news

            items = news.latest(limit=1)
            if not items:
                logger.warning("bot news job: no news available from any source")
                return None
            job.news_item = items[0]
        subject = (getattr(job.news_item, "title", "") or "").strip()
        job.topic = job.topic or subject
        script = llm.generate_news_script(job.news_item, language=language) or ""
        # No script is non-fatal: the pipeline writes its own from the subject.
    if not subject:
        return None
    return VideoParams(
        video_subject=subject,
        video_script=script,
        video_language=language,
        # "" lets the Phase 3 language switch pick the voice for the language.
        voice_name=str(config.telegram.get("voice_name", "") or ""),
        video_source=str(config.telegram.get("video_source", "pexels") or "pexels"),
        video_count=1,
        paragraph_number=int(config.telegram.get("paragraph_number", 1) or 1),
    )


def _run_pipeline(job: Job) -> Optional[dict]:
    """Default job runner: build params and execute the full pipeline."""
    from app.services import task as task_service

    params = _build_params(job)
    if params is None:
        return None
    return task_service.start(task_id=job.task_id, params=params)


class JobQueue:
    """FIFO queue with a single daemon worker thread."""

    def __init__(self, api, runner=None, progress_interval: float = 10.0):
        self.api = api
        self.runner = runner or _run_pipeline
        self.progress_interval = progress_interval
        self._queue = queue.Queue()
        self._current: Optional[Job] = None
        self._worker = threading.Thread(target=self._loop, daemon=True, name="bot-jobs")
        self._started = False

    def start(self):
        if not self._started:
            self._started = True
            self._worker.start()

    def submit(self, job: Job) -> int:
        """Enqueue and return the 1-based queue position (1 = next/now)."""
        self._queue.put(job)
        return self._queue.qsize() + (1 if self._current else 0)

    def status(self) -> str:
        lines = []
        if self._current is not None:
            task = sm.state.get_task(self._current.task_id) or {}
            lines.append(
                f"rendering: {self._current.topic or self._current.kind} "
                f"({task.get('progress', 0)}%)"
            )
        lines.append(f"queued: {self._queue.qsize()}")
        return "\n".join(lines)

    def _loop(self):
        while True:
            job = self._queue.get()
            self._current = job
            try:
                self._execute(job)
            except Exception as e:  # noqa: BLE001 - a failed job must not kill the worker
                logger.error(f"bot job {job.task_id} crashed: {e}")
                self.api.send_message(job.chat_id, f"❌ Ошибка рендера: {e}")
            finally:
                self._current = None
                self._queue.task_done()

    def _execute(self, job: Job):
        label = job.topic or getattr(job.news_item, "title", "") or job.kind
        status = self.api.send_message(job.chat_id, f"🎬 Рендерю: {label}") or {}
        status_id = status.get("message_id")

        stop = threading.Event()
        reporter = threading.Thread(
            target=self._report_progress,
            args=(job, status_id, stop),
            daemon=True,
            name="bot-progress",
        )
        if status_id:
            reporter.start()
        started = time.time()
        try:
            result = self.runner(job)
        finally:
            stop.set()
        if reporter.is_alive():
            reporter.join(timeout=2)

        videos = (result or {}).get("videos") or []
        if not videos:
            self.api.send_message(
                job.chat_id, f"❌ Не получилось собрать видео: {label}"
            )
            return
        elapsed = int(time.time() - started)
        if status_id:
            self.api.edit_message(job.chat_id, status_id, f"✅ Готово за {elapsed}с, загружаю…")
        self.api.send_chat_action(job.chat_id, "upload_video")
        sent = self.api.send_video(job.chat_id, videos[0], caption=label[:200])
        if not sent:
            self.api.send_message(
                job.chat_id,
                f"⚠️ Видео готово, но не загрузилось в Telegram. Файл: {videos[0]}",
            )

    def _report_progress(self, job: Job, status_id: int, stop: threading.Event):
        last = -1
        label = job.topic or job.kind
        while not stop.wait(self.progress_interval):
            task = sm.state.get_task(job.task_id) or {}
            progress = int(task.get("progress", 0) or 0)
            if task.get("state") in (const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED):
                return
            if progress != last:
                last = progress
                self.api.edit_message(
                    job.chat_id, status_id, f"⏳ {label}: {progress}%"
                )
