import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import videogen
from app.services.videogen import providers
from app.services.videogen.base import (
    STATUS_DONE,
    STATUS_PENDING,
    ClipSpec,
    VideoGenerator,
)


class _FakeGen(VideoGenerator):
    name = "fake"

    def __init__(self):
        self.submits = 0

    def is_configured(self):
        return True

    def submit(self, spec):
        self.submits += 1
        return f"job-{self.submits}"

    def poll(self, job):
        return STATUS_DONE, f"http://example/{job}.mp4"


def _resp(payload):
    r = mock.Mock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


class TestVideogenConfig(unittest.TestCase):
    def setUp(self):
        self.original = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original)

    def test_disabled_by_default(self):
        config.app.pop("video_gen_enabled", None)
        config.app["video_gen_provider"] = "null"
        self.assertFalse(videogen.is_enabled())

    def test_enabled_requires_real_provider(self):
        config.app["video_gen_enabled"] = True
        config.app["video_gen_provider"] = "null"
        self.assertFalse(videogen.is_enabled())
        config.app["video_gen_provider"] = "replicate"
        self.assertTrue(videogen.is_enabled())

    def test_factory_selects_provider(self):
        config.app["video_gen_provider"] = "fal"
        self.assertEqual(videogen.get_generator().name, "fal")
        config.app["video_gen_provider"] = "bogus"
        self.assertEqual(videogen.get_generator().name, "null")  # unknown -> null


class TestGenerateClips(unittest.TestCase):
    def setUp(self):
        self.original = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original)

    def test_null_provider_returns_empty(self):
        config.app["video_gen_provider"] = "null"
        self.assertEqual(videogen.generate_clips([ClipSpec(prompt="a")]), [])

    def test_caches_generated_clip(self):
        fake = _FakeGen()
        spec = ClipSpec(prompt="unique-prompt-xyz")
        store = {}  # in-memory cache so the test is isolated from disk state

        def fake_download(url, dest, headers=None):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(b"x")
            return dest

        with mock.patch.object(videogen, "get_generator", return_value=fake), \
            mock.patch.object(videogen, "_download", side_effect=fake_download), \
            mock.patch.object(videogen.cache, "get", side_effect=lambda ns, k: store.get(k)), \
            mock.patch.object(videogen.cache, "set", side_effect=lambda ns, k, v: store.__setitem__(k, v)):
            first = videogen.generate_clips([spec])
            second = videogen.generate_clips([spec])

        self.assertEqual(len(first), 1)
        self.assertEqual(first, second)
        self.assertEqual(fake.submits, 1)  # second call served from cache

    def test_enforces_max_clips(self):
        config.app["video_gen_max_clips"] = 1
        config.app["video_gen_max_seconds"] = 999
        fake = _FakeGen()
        specs = [ClipSpec(prompt=f"p{i}") for i in range(3)]

        def fake_download(url, dest, headers=None):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(b"x")
            return dest

        with mock.patch.object(videogen, "get_generator", return_value=fake), \
            mock.patch.object(videogen, "_download", side_effect=fake_download):
            result = videogen.generate_clips(specs)
        self.assertEqual(len(result), 1)


class TestProviders(unittest.TestCase):
    def setUp(self):
        self.original = dict(config.app)
        config.app["video_gen_api_key"] = "k"
        config.app["video_gen_model"] = "model/version"
        config.app["video_gen_endpoint"] = "https://gpu.example"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original)

    def test_replicate_submit_and_poll(self):
        gen = providers.ReplicateVideoGenerator()
        with mock.patch.object(providers, "requests") as rq:
            rq.post.return_value = _resp({"id": "pred1"})
            rq.get.return_value = _resp(
                {"status": "succeeded", "output": ["http://x/out.mp4"]}
            )
            self.assertEqual(gen.submit(ClipSpec(prompt="hi")), "pred1")
            self.assertEqual(gen.poll("pred1"), (STATUS_DONE, "http://x/out.mp4"))

    def test_fal_submit_and_poll(self):
        gen = providers.FalVideoGenerator()
        with mock.patch.object(providers, "requests") as rq:
            rq.post.return_value = _resp({"request_id": "r1"})
            rq.get.side_effect = [
                _resp({"status": "COMPLETED"}),
                _resp({"video": {"url": "http://x/v.mp4"}}),
            ]
            self.assertEqual(gen.submit(ClipSpec(prompt="hi")), "r1")
            self.assertEqual(gen.poll("r1"), (STATUS_DONE, "http://x/v.mp4"))

    def test_http_submit_and_poll(self):
        gen = providers.HttpVideoGenerator()
        with mock.patch.object(providers, "requests") as rq:
            rq.post.return_value = _resp({"job": "j1"})
            rq.get.return_value = _resp({"status": "pending"})
            self.assertEqual(gen.submit(ClipSpec(prompt="hi")), "j1")
            self.assertEqual(gen.poll("j1"), (STATUS_PENDING, ""))

    def test_unconfigured_provider_is_not_configured(self):
        config.app["video_gen_api_key"] = ""
        config.app["video_gen_model"] = ""
        self.assertFalse(providers.ReplicateVideoGenerator().is_configured())
        self.assertFalse(providers.FalVideoGenerator().is_configured())


class TestAzureSora(unittest.TestCase):
    def setUp(self):
        from app.services.videogen import azure_sora

        self.azure_sora = azure_sora
        self.original = dict(config.app)
        config.app["video_gen_api_key"] = "azkey"
        config.app["video_gen_endpoint"] = "https://res.openai.azure.com"
        config.app["video_gen_model"] = ""

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original)

    def test_build_provider_resolves(self):
        gen = providers.build_provider("azure-sora")
        self.assertEqual(gen.name, "azure-sora")
        self.assertTrue(gen.is_configured())

    def test_submit_payload_and_poll_done(self):
        gen = self.azure_sora.AzureSoraVideoGenerator()
        with mock.patch.object(self.azure_sora, "requests") as rq:
            rq.post.return_value = _resp({"id": "job1"})
            job = gen.submit(ClipSpec(prompt="a drone shot", duration=4.2, aspect="9:16"))
            self.assertEqual(job, "job1")
            url = rq.post.call_args.args[0]
            self.assertIn("/openai/v1/video/generations/jobs", url)
            self.assertIn("api-version=preview", url)
            payload = rq.post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "sora")  # default when unset
            self.assertEqual((payload["width"], payload["height"]), (720, 1280))
            self.assertEqual(payload["n_seconds"], 4)
            self.assertEqual(rq.post.call_args.kwargs["headers"]["api-key"], "azkey")

            rq.get.return_value = _resp(
                {"status": "succeeded", "generations": [{"id": "gen9"}]}
            )
            status, result = gen.poll("job1")
        self.assertEqual(status, STATUS_DONE)
        self.assertIn("/openai/v1/video/generations/gen9/content/video", result)

    def test_poll_pending_and_failed(self):
        from app.services.videogen.base import STATUS_FAILED

        gen = self.azure_sora.AzureSoraVideoGenerator()
        with mock.patch.object(self.azure_sora, "requests") as rq:
            rq.get.return_value = _resp({"status": "running"})
            self.assertEqual(gen.poll("j"), (STATUS_PENDING, ""))
            rq.get.return_value = _resp({"status": "failed", "failure_reason": "nope"})
            self.assertEqual(gen.poll("j"), (STATUS_FAILED, ""))
            # Succeeded but empty generations must fail closed, not loop forever.
            rq.get.return_value = _resp({"status": "succeeded", "generations": []})
            self.assertEqual(gen.poll("j"), (STATUS_FAILED, ""))

    def test_duration_clamped_to_api_range(self):
        gen = self.azure_sora.AzureSoraVideoGenerator()
        with mock.patch.object(self.azure_sora, "requests") as rq:
            rq.post.return_value = _resp({"id": "j"})
            gen.submit(ClipSpec(prompt="x", duration=90.0))
            self.assertEqual(rq.post.call_args.kwargs["json"]["n_seconds"], 20)

    def test_unconfigured_raises_videogen_error(self):
        from app.services.videogen.base import VideoGenError

        config.app["video_gen_api_key"] = ""
        gen = self.azure_sora.AzureSoraVideoGenerator()
        self.assertFalse(gen.is_configured())
        with self.assertRaises(VideoGenError):
            gen.submit(ClipSpec(prompt="x"))

    def test_download_headers_carry_api_key(self):
        gen = self.azure_sora.AzureSoraVideoGenerator()
        self.assertEqual(gen.download_headers(), {"api-key": "azkey"})

    def test_orchestrator_passes_download_headers(self):
        fake = _FakeGen()
        fake.download_headers = lambda: {"api-key": "azkey"}
        captured = {}

        def fake_download(url, dest, headers=None):
            captured["headers"] = headers
            return dest

        with mock.patch.object(videogen, "get_generator", return_value=fake), \
            mock.patch.object(videogen, "_download", side_effect=fake_download), \
            mock.patch.object(videogen.cache, "get", return_value=None), \
            mock.patch.object(videogen.cache, "set"):
            videogen.generate_clips([ClipSpec(prompt="x", duration=3)])
        self.assertEqual(captured["headers"], {"api-key": "azkey"})


if __name__ == "__main__":
    unittest.main()
