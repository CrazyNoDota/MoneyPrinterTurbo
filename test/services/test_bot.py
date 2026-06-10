import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.bot import jobs as bot_jobs
from app.bot import router as bot_router
from app.bot import worker as bot_worker
from app.bot.jobs import Job, JobQueue
from app.bot.router import Router


def _api():
    api = mock.Mock()
    api.is_configured.return_value = True
    api.send_message.return_value = {"message_id": 5}
    api.edit_message.return_value = {"message_id": 5}
    api.send_video.return_value = {"message_id": 6}
    api.download_file.side_effect = lambda file_id, dest: dest
    return api


def _message(chat_id=1, **fields):
    return {"message": {"chat": {"id": chat_id}, **fields}}


class TestRouterAuth(unittest.TestCase):
    def test_unknown_chat_denied_with_single_notice(self):
        api = _api()
        jobs = mock.Mock()
        router = Router(api, jobs, allowed_chats=["42"])
        router.handle_update(_message(chat_id=7, text="/make topic"))
        router.handle_update(_message(chat_id=7, text="/make topic"))
        jobs.submit.assert_not_called()
        # exactly one denial notice, containing the chat id to whitelist
        self.assertEqual(api.send_message.call_count, 1)
        self.assertIn("7", api.send_message.call_args.args[1])

    def test_allowed_chat_passes_int_or_str(self):
        api = _api()
        jobs = mock.Mock()
        jobs.submit.return_value = 1
        router = Router(api, jobs, allowed_chats=[42])
        router.handle_update(_message(chat_id=42, text="/make a topic"))
        jobs.submit.assert_called_once()


class TestRouterCommands(unittest.TestCase):
    def setUp(self):
        self.api = _api()
        self.jobs = mock.Mock()
        self.jobs.submit.return_value = 2
        self.jobs.status.return_value = "queued: 0"
        self.router = Router(self.api, self.jobs, allowed_chats=["1"])

    def test_make_enqueues_topic(self):
        self.router.handle_update(_message(text="/make AI news today"))
        job = self.jobs.submit.call_args.args[0]
        self.assertEqual(job.kind, "make")
        self.assertEqual(job.topic, "AI news today")
        self.assertIn("№2", self.api.send_message.call_args.args[1])

    def test_make_with_bot_suffix(self):
        self.router.handle_update(_message(text="/make@MyBot topic"))
        self.assertEqual(self.jobs.submit.call_args.args[0].topic, "topic")

    def test_make_without_topic_shows_usage(self):
        self.router.handle_update(_message(text="/make"))
        self.jobs.submit.assert_not_called()
        self.assertIn("/make", self.api.send_message.call_args.args[1])

    def test_news_enqueues_without_fetching(self):
        # The fetch must happen on the job worker, never in the poll loop.
        with mock.patch("app.services.news.latest") as latest:
            self.router.handle_update(_message(text="/news"))
        latest.assert_not_called()
        job = self.jobs.submit.call_args.args[0]
        self.assertEqual(job.kind, "news")
        self.assertIsNone(job.news_item)

    def test_status_and_help(self):
        self.router.handle_update(_message(text="/status"))
        self.assertEqual(self.api.send_message.call_args.args[1], "queued: 0")
        self.router.handle_update(_message(text="/definitely-unknown"))
        self.assertIn("/make", self.api.send_message.call_args.args[1])
        self.router.handle_update(_message(text="just chatting"))
        self.assertIn("/make", self.api.send_message.call_args.args[1])

    def test_handler_errors_do_not_propagate(self):
        self.jobs.submit.side_effect = RuntimeError("boom")
        self.router.handle_update(_message(text="/make topic"))  # must not raise


class TestRouterMedia(unittest.TestCase):
    def setUp(self):
        self.api = _api()
        self.router = Router(
            self.api, mock.Mock(), allowed_chats=["1"], materials_dir="X:/mats"
        )

    def test_photo_takes_largest_size(self):
        self.router.handle_update(
            _message(photo=[{"file_id": "small"}, {"file_id": "big", "file_unique_id": "u1"}])
        )
        file_id, dest = self.api.download_file.call_args.args
        self.assertEqual(file_id, "big")
        self.assertTrue(dest.endswith("u1.jpg"))
        self.assertIn("Добавил", self.api.send_message.call_args.args[1])

    def test_video_uses_original_name_sanitized(self):
        self.router.handle_update(
            _message(video={"file_id": "v1", "file_name": "my/../clip?.mp4"})
        )
        _, dest = self.api.download_file.call_args.args
        self.assertNotIn("?", dest)
        self.assertNotIn("..", Path(dest).name)

    def test_non_media_document_rejected(self):
        self.router.handle_update(
            _message(document={"file_id": "d", "file_name": "x.exe", "mime_type": "application/x-msdownload"})
        )
        self.api.download_file.assert_not_called()

    def test_failed_download_reports(self):
        self.api.download_file.side_effect = lambda f, d: ""
        self.router.handle_update(_message(photo=[{"file_id": "p", "file_unique_id": "u"}]))
        self.assertIn("⚠️", self.api.send_message.call_args.args[1])


class TestBuildParams(unittest.TestCase):
    def test_make_job_params(self):
        with mock.patch.dict(
            bot_jobs.config.telegram, {"language": "ru", "voice_name": ""}
        ):
            params = bot_jobs._build_params(Job(chat_id=1, kind="make", topic="Тема"))
        self.assertEqual(params.video_subject, "Тема")
        self.assertEqual(params.video_language, "ru")
        self.assertEqual(params.voice_name, "")
        self.assertEqual(params.video_count, 1)

    def test_news_job_uses_generated_script(self):
        item = SimpleNamespace(title="Story", text="text", url="u")
        with mock.patch.object(
            bot_jobs.llm, "generate_news_script", return_value="A script."
        ):
            params = bot_jobs._build_params(
                Job(chat_id=1, kind="news", news_item=item)
            )
        self.assertEqual(params.video_subject, "Story")
        self.assertEqual(params.video_script, "A script.")

    def test_news_script_failure_is_non_fatal(self):
        item = SimpleNamespace(title="Story", text="t", url="u")
        with mock.patch.object(bot_jobs.llm, "generate_news_script", return_value=""):
            params = bot_jobs._build_params(Job(chat_id=1, kind="news", news_item=item))
        self.assertEqual(params.video_script, "")  # pipeline writes its own

    def test_empty_job_returns_none(self):
        self.assertIsNone(bot_jobs._build_params(Job(chat_id=1, kind="make", topic="")))

    def test_news_job_fetches_item_on_worker(self):
        item = SimpleNamespace(title="Fetched story", text="t", url="u")
        job = Job(chat_id=1, kind="news")
        with mock.patch("app.services.news.latest", return_value=[item]), \
             mock.patch.object(bot_jobs.llm, "generate_news_script", return_value="S"):
            params = bot_jobs._build_params(job)
        self.assertEqual(params.video_subject, "Fetched story")
        self.assertIs(job.news_item, item)  # pinned for the status label
        self.assertEqual(job.topic, "Fetched story")

    def test_news_job_without_news_returns_none(self):
        with mock.patch("app.services.news.latest", return_value=[]):
            self.assertIsNone(bot_jobs._build_params(Job(chat_id=1, kind="news")))


class TestTokenNeverLogged(unittest.TestCase):
    TOKEN = "1234567:SECRET-abcDEF"

    def _logged(self, logger_mock):
        return " | ".join(str(c.args[0]) for c in logger_mock.call_args_list)

    def test_call_exception_is_scrubbed(self):
        from app.bot import api as bot_api

        client = bot_api.TelegramAPI(token=self.TOKEN)
        err = RuntimeError(
            f"Max retries exceeded with url: /bot{self.TOKEN}/sendMessage"
        )
        with mock.patch.object(bot_api.requests, "post", side_effect=err), \
             mock.patch.object(bot_api.logger, "warning") as warn:
            self.assertIsNone(client.send_message(1, "hi"))
        logged = self._logged(warn)
        self.assertNotIn(self.TOKEN, logged)
        self.assertIn("***", logged)

    def test_download_exception_is_scrubbed(self):
        from app.bot import api as bot_api

        client = bot_api.TelegramAPI(token=self.TOKEN)
        ok = mock.Mock()
        ok.json.return_value = {"ok": True, "result": {"file_path": "photos/x.jpg"}}
        err = RuntimeError(f"boom https://api.telegram.org/file/bot{self.TOKEN}/x")
        with mock.patch.object(
            bot_api.requests, "post", return_value=ok
        ), mock.patch.object(
            bot_api.requests, "get", side_effect=err
        ), mock.patch.object(bot_api.logger, "warning") as warn:
            self.assertEqual(client.download_file("fid", "X:/dest.jpg"), "")
        self.assertNotIn(self.TOKEN, self._logged(warn))

    def test_oversized_video_rejected_before_upload(self):
        from app.bot import api as bot_api

        client = bot_api.TelegramAPI(token=self.TOKEN)
        with mock.patch.object(bot_api.os.path, "isfile", return_value=True), \
             mock.patch.object(
                 bot_api.os.path, "getsize", return_value=51 * 1024 * 1024
             ), mock.patch.object(bot_api.requests, "post") as post:
            self.assertIsNone(client.send_video(1, "X:/big.mp4"))
        post.assert_not_called()


class TestJobQueue(unittest.TestCase):
    def test_success_sends_video(self):
        api = _api()
        q = JobQueue(api, runner=lambda job: {"videos": ["X:/out/final-1.mp4"]})
        q._execute(Job(chat_id=1, kind="make", topic="T"))
        api.send_video.assert_called_once()
        self.assertEqual(api.send_video.call_args.args[1], "X:/out/final-1.mp4")

    def test_no_videos_reports_failure(self):
        api = _api()
        q = JobQueue(api, runner=lambda job: None)
        q._execute(Job(chat_id=1, kind="make", topic="T"))
        api.send_video.assert_not_called()
        self.assertIn("❌", api.send_message.call_args.args[1])

    def test_upload_failure_reports_path(self):
        api = _api()
        api.send_video.return_value = None
        q = JobQueue(api, runner=lambda job: {"videos": ["X:/v.mp4"]})
        q._execute(Job(chat_id=1, kind="make", topic="T"))
        self.assertIn("X:/v.mp4", api.send_message.call_args.args[1])

    def test_crashing_runner_does_not_kill_worker(self):
        api = _api()
        calls = []

        def runner(job):
            calls.append(job.topic)
            if job.topic == "bad":
                raise RuntimeError("boom")
            return {"videos": []}

        q = JobQueue(api, runner=runner)
        q.start()
        q.submit(Job(chat_id=1, kind="make", topic="bad"))
        q.submit(Job(chat_id=1, kind="make", topic="good"))
        deadline = time.time() + 5
        while len(calls) < 2 and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(calls, ["bad", "good"])  # survived the crash
        texts = [c.args[1] for c in api.send_message.call_args_list]
        self.assertTrue(any("Ошибка" in t for t in texts))


class TestWorkerLoop(unittest.TestCase):
    def test_unconfigured_token_aborts(self):
        api = mock.Mock()
        api.is_configured.return_value = False
        bot_worker.run(api=api)  # returns immediately
        api.get_updates.assert_not_called()

    def test_polls_and_advances_offset(self):
        api = _api()
        api.get_updates.side_effect = [
            [{"update_id": 10, "message": {"chat": {"id": 1}, "text": "hi"}}],
            [],
        ]
        router = mock.Mock()
        jobs = mock.Mock()
        with mock.patch.object(bot_worker.time, "sleep"), \
             mock.patch.object(bot_worker, "load_offset", return_value=0), \
             mock.patch.object(bot_worker, "save_offset") as save:
            bot_worker.run(api=api, jobs=jobs, router=router, max_iterations=2)
        router.handle_update.assert_called_once()
        self.assertEqual(api.get_updates.call_args.kwargs["offset"], 11)
        save.assert_called_with(11)

    def test_offset_persisted_across_restarts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "offset.txt"
            with mock.patch.object(bot_worker, "_offset_file", return_value=str(path)):
                self.assertEqual(bot_worker.load_offset(), 0)  # no file yet
                bot_worker.save_offset(42)
                self.assertEqual(bot_worker.load_offset(), 42)

    def test_resumes_from_persisted_offset(self):
        api = _api()
        api.get_updates.return_value = []
        with mock.patch.object(bot_worker.time, "sleep"), \
             mock.patch.object(bot_worker, "load_offset", return_value=100):
            bot_worker.run(api=api, jobs=mock.Mock(), router=mock.Mock(), max_iterations=1)
        self.assertEqual(api.get_updates.call_args.kwargs["offset"], 100)


class TestAutopilot(unittest.TestCase):
    def _item(self, title="Big news", item_id="abc123"):
        return SimpleNamespace(title=title, id=item_id)

    def _autopilot(self, d, languages=("ru", "en")):
        from app.bot import autopilot as ap

        jobs = mock.Mock()
        with mock.patch.object(ap, "_seen_file",
                               return_value=str(Path(d) / "seen.txt")):
            pilot = ap.Autopilot(jobs, chat_id="99", languages=list(languages),
                                 interval_hours=6)
        return ap, pilot, jobs

    def test_from_config_disabled_returns_none(self):
        from app.bot import autopilot as ap

        with mock.patch.dict(ap.config.telegram, {"autopilot_enabled": False}):
            self.assertIsNone(ap.Autopilot.from_config(mock.Mock()))

    def test_from_config_requires_chat_id(self):
        from app.bot import autopilot as ap

        with mock.patch.dict(
            ap.config.telegram,
            {"autopilot_enabled": True, "autopilot_chat_id": ""},
        ):
            self.assertIsNone(ap.Autopilot.from_config(mock.Mock()))

    def test_tick_queues_one_job_per_language(self):
        import tempfile

        from app.services import news

        with tempfile.TemporaryDirectory() as d:
            ap, pilot, jobs = self._autopilot(d)
            with mock.patch.object(ap, "_seen_file",
                                   return_value=str(Path(d) / "seen.txt")), \
                 mock.patch.object(news, "latest", return_value=[self._item()]):
                queued = pilot.tick()
        self.assertEqual(queued, 2)
        langs = [c.args[0].language for c in jobs.submit.call_args_list]
        self.assertEqual(langs, ["ru", "en"])
        job = jobs.submit.call_args_list[0].args[0]
        self.assertEqual(job.kind, "news")
        self.assertEqual(job.chat_id, "99")
        self.assertEqual(job.topic, "Big news")

    def test_tick_dedupes_seen_item_across_instances(self):
        import tempfile

        from app.services import news

        with tempfile.TemporaryDirectory() as d:
            ap, pilot, jobs = self._autopilot(d)
            seen = str(Path(d) / "seen.txt")
            with mock.patch.object(ap, "_seen_file", return_value=seen), \
                 mock.patch.object(news, "latest", return_value=[self._item()]):
                self.assertEqual(pilot.tick(), 2)
                self.assertEqual(pilot.tick(), 0)  # same item, same instance
                # A fresh instance (restart) must load the persisted ids.
                _, pilot2, jobs2 = self._autopilot(d)
                with mock.patch.object(ap, "_seen_file", return_value=seen):
                    self.assertEqual(pilot2.tick(), 0)
            jobs2.submit.assert_not_called()

    def test_tick_no_news_is_quiet(self):
        import tempfile

        from app.services import news

        with tempfile.TemporaryDirectory() as d:
            ap, pilot, jobs = self._autopilot(d)
            with mock.patch.object(news, "latest", return_value=[]):
                self.assertEqual(pilot.tick(), 0)
        jobs.submit.assert_not_called()

    def test_job_language_overrides_config_default(self):
        with mock.patch.dict(
            bot_jobs.config.telegram, {"language": "ru", "voice_name": ""}
        ):
            params = bot_jobs._build_params(
                Job(chat_id=1, kind="make", topic="space", language="en")
            )
        self.assertEqual(params.video_language, "en")


if __name__ == "__main__":
    unittest.main()
