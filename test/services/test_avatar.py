import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import avatar
from app.services.avatar import azure as azure_mod
from app.services.avatar import wav2lip as wav2lip_mod


def _tempdir():
    return tempfile.TemporaryDirectory()


def _resp(payload=None, content=b"", headers=None):
    r = mock.Mock()
    r.json.return_value = payload or {}
    r.content = content
    r.headers = headers or {}
    r.raise_for_status.return_value = None
    return r


class _ConfigGuard:
    def setUp(self):
        self._app = dict(config.app)
        self._azure = dict(config.azure)

    def tearDown(self):
        config.app.clear()
        config.app.update(self._app)
        config.azure.clear()
        config.azure.update(self._azure)


class TestAzureAvatar(unittest.TestCase, _ConfigGuard):
    def setUp(self):
        _ConfigGuard.setUp(self)
        config.azure["speech_key"] = "speech-key"
        config.azure["speech_region"] = "eastus"
        config.app.update(
            {
                "avatar_character": "lisa",
                "avatar_style": "casual-sitting",
                "avatar_poll_interval": 0,
                "avatar_timeout": 3,
                "avatar_alpha_supported": False,
            }
        )

    def tearDown(self):
        _ConfigGuard.tearDown(self)

    def test_submit_poll_download_success(self):
        gen = azure_mod.AzureAvatar()
        with _tempdir() as tmp_dir, mock.patch.object(
            azure_mod, "requests"
        ) as rq, mock.patch.object(azure_mod.time, "sleep"):
            out_path = os.path.join(tmp_dir, "avatar.mp4")
            rq.post.return_value = _resp(
                headers={
                    "Location": "https://eastus.customvoice.api.speech.microsoft.com/jobs/job-1"
                }
            )
            rq.get.side_effect = [
                _resp({"status": "Running"}),
                _resp({"status": "Succeeded", "outputs": {"result": "https://cdn/out.mp4"}}),
                _resp(content=b"video-bytes"),
            ]

            result = gen.synthesize(
                "Hello from Azure",
                "en-US-GuyNeural",
                out_path,
                1080,
                1920,
            )
            output_bytes = Path(out_path).read_bytes()

        self.assertEqual(result, out_path)
        self.assertEqual(output_bytes, b"video-bytes")

        submit_kwargs = rq.post.call_args.kwargs
        self.assertIn("/api/texttospeech/3.1-preview1/batchsynthesis/talkingavatar", rq.post.call_args.args[0])
        self.assertEqual(submit_kwargs["headers"]["Ocp-Apim-Subscription-Key"], "speech-key")
        payload = submit_kwargs["json"]
        self.assertEqual(payload["inputKind"], "PlainText")
        self.assertEqual(payload["synthesisConfig"]["voice"], "en-US-GuyNeural")
        self.assertEqual(payload["inputs"], [{"content": "Hello from Azure"}])
        self.assertEqual(payload["properties"]["talkingAvatarCharacter"], "lisa")
        self.assertEqual(payload["properties"]["talkingAvatarStyle"], "casual-sitting")
        self.assertEqual(payload["properties"]["videoFormat"], "mp4")
        self.assertEqual(payload["properties"]["videoCodec"], "h264")
        self.assertEqual(payload["properties"]["videoFps"], 25)
        self.assertEqual(payload["properties"]["videoWidth"], 1080)
        self.assertEqual(payload["properties"]["videoHeight"], 1920)
        self.assertEqual(rq.get.call_args_list[-1].args[0], "https://cdn/out.mp4")

    def test_ssml_and_alpha_request_shape(self):
        config.app["avatar_alpha_supported"] = True
        gen = azure_mod.AzureAvatar()
        payload = gen._payload(
            "<speak version='1.0'>Hi</speak>",
            "",
            1080,
            1920,
        )

        self.assertEqual(payload["inputKind"], "SSML")
        self.assertEqual(payload["synthesisConfig"]["voice"], "en-US-JennyNeural")
        self.assertEqual(payload["properties"]["videoFormat"], "webm")
        self.assertEqual(payload["properties"]["videoCodec"], "vp9")
        self.assertEqual(payload["properties"]["backgroundColor"], "transparent")

    def test_failed_terminal_state_returns_empty(self):
        gen = azure_mod.AzureAvatar()
        with mock.patch.object(azure_mod, "requests") as rq:
            rq.get.return_value = _resp({"status": "Failed"})
            self.assertEqual(gen.poll("job-1"), ("failed", ""))

    def test_download_failure_returns_empty(self):
        gen = azure_mod.AzureAvatar()
        with _tempdir() as tmp_dir, mock.patch.object(
            azure_mod, "requests"
        ) as rq, mock.patch.object(azure_mod.time, "sleep"):
            out_path = os.path.join(tmp_dir, "avatar.mp4")
            rq.post.return_value = _resp(headers={"Location": "https://host/jobs/job-1"})
            download = _resp(content=b"")
            download.raise_for_status.side_effect = Exception("503 from CDN")
            rq.get.side_effect = [
                _resp({"status": "Succeeded", "outputs": {"result": "https://cdn/out.mp4"}}),
                download,
            ]

            result = gen.synthesize("Hello", "en-US-GuyNeural", out_path, 1080, 1920)

        self.assertEqual(result, "")
        self.assertFalse(os.path.exists(out_path))

    def test_empty_download_body_returns_empty(self):
        gen = azure_mod.AzureAvatar()
        with _tempdir() as tmp_dir, mock.patch.object(azure_mod, "requests") as rq:
            out_path = os.path.join(tmp_dir, "avatar.mp4")
            rq.get.return_value = _resp(content=b"")
            self.assertEqual(gen._download("https://cdn/out.mp4", out_path), "")

    def test_missing_key_is_nonfatal(self):
        config.azure["speech_key"] = ""
        with _tempdir() as tmp_dir:
            out_path = os.path.join(tmp_dir, "avatar.mp4")
            self.assertEqual(
                azure_mod.AzureAvatar().synthesize("hello", "voice", out_path, 1080, 1920),
                "",
            )


class TestWav2LipAvatar(unittest.TestCase, _ConfigGuard):
    def setUp(self):
        _ConfigGuard.setUp(self)

    def tearDown(self):
        _ConfigGuard.tearDown(self)

    def test_missing_venv_or_weights_returns_empty(self):
        with _tempdir() as tmp_dir, mock.patch.object(
            wav2lip_mod.subprocess, "run"
        ) as run:
            audio = Path(tmp_dir) / "audio.wav"
            portrait = Path(tmp_dir) / "portrait.png"
            audio.write_bytes(b"a")
            portrait.write_bytes(b"p")

            config.app.update(
                {
                    "wav2lip_python": str(Path(tmp_dir) / "missing-python.exe"),
                    "wav2lip_script": str(Path(tmp_dir) / "missing-inference.py"),
                    "wav2lip_weights": str(Path(tmp_dir) / "missing.pth"),
                }
            )

            result = wav2lip_mod.Wav2LipAvatar().synthesize(
                str(audio), str(portrait), str(Path(tmp_dir) / "out.mp4"), 1080, 1920
            )

        self.assertEqual(result, "")
        run.assert_not_called()

    def test_runs_subprocess_when_configured(self):
        with _tempdir() as tmp_dir, mock.patch.object(
            wav2lip_mod.subprocess, "run"
        ) as run:
            root = Path(tmp_dir)
            py = root / "python.exe"
            repo = root / "Wav2Lip"
            script = repo / "inference.py"
            weights = root / "wav2lip_gan.pth"
            audio = root / "audio.wav"
            portrait = root / "portrait.png"
            out = root / "out.mp4"
            repo.mkdir()
            for path in (py, script, weights, audio, portrait):
                path.write_bytes(b"x")

            def fake_run(cmd, **kwargs):
                out.write_bytes(b"mp4")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            run.side_effect = fake_run
            config.app.update(
                {
                    "wav2lip_python": str(py),
                    "wav2lip_repo": str(repo),
                    "wav2lip_script": str(script),
                    "wav2lip_weights": str(weights),
                    "wav2lip_timeout": 10,
                }
            )

            result = wav2lip_mod.Wav2LipAvatar().synthesize(
                str(audio), str(portrait), str(out), 1080, 1920
            )

        self.assertEqual(result, str(out))
        cmd = run.call_args.args[0]
        self.assertIn("--checkpoint_path", cmd)
        self.assertIn(str(weights), cmd)
        self.assertIn("--face", cmd)
        self.assertIn(str(portrait), cmd)
        self.assertIn("--audio", cmd)
        self.assertIn(str(audio), cmd)
        self.assertIn("--outfile", cmd)
        self.assertIn(str(out), cmd)


class TestAvatarResolver(unittest.TestCase, _ConfigGuard):
    def setUp(self):
        _ConfigGuard.setUp(self)
        self.calls = []

    def tearDown(self):
        _ConfigGuard.tearDown(self)

    def _provider(self, name, result):
        calls = self.calls

        class Provider:
            def __init__(self):
                self.name = name

            def synthesize(self, *args, **kwargs):
                calls.append(name)
                return result

        return Provider

    def test_disabled_returns_empty_immediately(self):
        config.app["avatar_enabled"] = False
        with mock.patch.object(avatar, "AzureAvatar", self._provider("azure", "x")):
            self.assertEqual(avatar.synthesize("text", "voice", "out.mp4"), "")
        self.assertEqual(self.calls, [])

    def test_auto_falls_through_azure_to_wav2lip(self):
        config.app["avatar_enabled"] = True
        config.app["avatar_provider"] = "auto"
        with mock.patch.object(avatar, "AzureAvatar", self._provider("azure", "")), mock.patch.object(
            avatar, "Wav2LipAvatar", self._provider("wav2lip", "out.mp4")
        ):
            self.assertEqual(avatar.synthesize("text", "voice", "out.mp4"), "out.mp4")
        self.assertEqual(self.calls, ["azure", "wav2lip"])

    def test_both_fail_returns_empty(self):
        config.app["avatar_enabled"] = True
        config.app["avatar_provider"] = "auto"
        with mock.patch.object(avatar, "AzureAvatar", self._provider("azure", "")), mock.patch.object(
            avatar, "Wav2LipAvatar", self._provider("wav2lip", "")
        ):
            self.assertEqual(avatar.synthesize("text", "voice", "out.mp4"), "")
        self.assertEqual(self.calls, ["azure", "wav2lip"])

    def test_explicit_provider_pins_chain(self):
        config.app["avatar_enabled"] = True
        config.app["avatar_provider"] = "azure"
        with mock.patch.object(avatar, "AzureAvatar", self._provider("azure", "")), mock.patch.object(
            avatar, "Wav2LipAvatar", self._provider("wav2lip", "out.mp4")
        ):
            self.assertEqual(avatar.synthesize("text", "voice", "out.mp4"), "")
        self.assertEqual(self.calls, ["azure"])

        self.calls.clear()
        config.app["avatar_provider"] = "wav2lip"
        with mock.patch.object(avatar, "AzureAvatar", self._provider("azure", "azure.mp4")), mock.patch.object(
            avatar, "Wav2LipAvatar", self._provider("wav2lip", "wav.mp4")
        ):
            self.assertEqual(avatar.synthesize("text", "voice", "out.mp4"), "wav.mp4")
        self.assertEqual(self.calls, ["wav2lip"])


if __name__ == "__main__":
    unittest.main()
