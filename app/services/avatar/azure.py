"""Azure TTS Avatar batch synthesis provider."""

import os
import time
from typing import Any
from urllib.parse import urljoin

import requests
from loguru import logger

from app.config import config
from app.utils.retry import call_with_retry

from .base import TalkingHead


_DEFAULT_API_VERSION = "3.1-preview1"
_DEFAULT_VOICE = "en-US-JennyNeural"


def _cfg(key: str, default: Any = None) -> Any:
    return config.app.get(key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class AzureAvatar(TalkingHead):
    """Primary provider using Azure Speech batch avatar synthesis."""

    name = "azure"

    def is_configured(self) -> bool:
        return bool(
            config.azure.get("speech_key")
            and config.azure.get("speech_region")
            and _cfg("avatar_character", "")
        )

    def _endpoint(self) -> str:
        explicit = str(_cfg("avatar_azure_endpoint", "") or "").strip()
        if explicit:
            return explicit.rstrip("/")
        region = str(config.azure.get("speech_region", "")).strip()
        version = str(_cfg("avatar_azure_api_version", _DEFAULT_API_VERSION))
        return (
            f"https://{region}.customvoice.api.speech.microsoft.com/"
            f"api/texttospeech/{version}/batchsynthesis/talkingavatar"
        )

    def _headers(self) -> dict:
        return {
            "Ocp-Apim-Subscription-Key": config.azure.get("speech_key", ""),
            "Content-Type": "application/json",
        }

    def _payload(self, text: str, voice: str, width: int, height: int) -> dict:
        prefer_alpha = _as_bool(_cfg("avatar_prefer_alpha", True))
        use_alpha = prefer_alpha and _as_bool(_cfg("avatar_alpha_supported", False))
        is_ssml = text.lstrip().lower().startswith("<speak")
        video_format = "webm" if use_alpha else "mp4"
        video_codec = "vp9" if use_alpha else "h264"
        properties = {
            "customized": False,
            "talkingAvatarCharacter": str(_cfg("avatar_character", "lisa")),
            "talkingAvatarStyle": str(_cfg("avatar_style", "casual-sitting")),
            "videoFormat": video_format,
            "videoCodec": video_codec,
            "videoFps": int(_cfg("avatar_fps", 25) or 25),
            "videoWidth": int(width or 1080),
            "videoHeight": int(height or 1920),
            "subtitleType": "soft_embedded",
        }
        if use_alpha:
            properties["backgroundColor"] = "transparent"

        return {
            # avatar_voice defaults to "" (= follow the pipeline voice, resolved
            # by the caller); an avatar invoked directly still needs a real one.
            "synthesisConfig": {
                "voice": voice or str(_cfg("avatar_voice", "") or "") or _DEFAULT_VOICE
            },
            "inputKind": "SSML" if is_ssml else "PlainText",
            "inputs": [{"content": text}],
            "properties": properties,
        }

    def _job_id_from_response(self, response) -> str:
        location = response.headers.get("Location") or response.headers.get("location") or ""
        if location:
            return location.rstrip("/").split("/")[-1]
        try:
            data = response.json()
        except Exception:
            data = {}
        return data.get("id") or data.get("jobId") or data.get("job_id") or ""

    def submit(self, text: str, voice: str, width: int, height: int) -> str:
        response = call_with_retry(
            requests.post,
            self._endpoint(),
            headers=self._headers(),
            json=self._payload(text, voice, width, height),
            proxies=config.proxy,
            timeout=(30, 120),
            description="avatar.azure.submit",
        )
        response.raise_for_status()
        return self._job_id_from_response(response)

    def poll(self, job_id: str) -> tuple[str, str]:
        response = call_with_retry(
            requests.get,
            urljoin(self._endpoint().rstrip("/") + "/", job_id),
            headers=self._headers(),
            proxies=config.proxy,
            timeout=(30, 60),
            description="avatar.azure.poll",
        )
        response.raise_for_status()
        data = response.json()
        status = str(data.get("status", "")).lower()
        if status == "succeeded":
            outputs = data.get("outputs") or {}
            return "succeeded", (
                outputs.get("result")
                or outputs.get("video")
                or data.get("resultUrl")
                or data.get("url")
                or ""
            )
        if status in ("failed", "canceled", "cancelled"):
            return "failed", ""
        return "running", ""

    def _poll_until_done(self, job_id: str) -> str:
        timeout = float(_cfg("avatar_timeout", 900) or 900)
        interval = float(_cfg("avatar_poll_interval", 5) or 5)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, result_url = self.poll(job_id)
            if status == "succeeded":
                return result_url or ""
            if status == "failed":
                logger.warning(f"Azure avatar job {job_id} failed")
                return ""
            time.sleep(interval)
        logger.warning(f"Azure avatar job {job_id} timed out after {timeout:.0f}s")
        return ""

    def _download(self, url: str, out_path: str) -> str:
        response = call_with_retry(
            requests.get,
            url,
            proxies=config.proxy,
            timeout=(60, 240),
            description="avatar.azure.download",
        )
        response.raise_for_status()
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(response.content)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        return ""

    def synthesize(
        self,
        script_or_audio: str,
        presenter: str,
        out_path: str,
        width: int,
        height: int,
    ) -> str:
        if not self.is_configured():
            logger.warning("Azure avatar is not configured; falling back")
            return ""
        text = (script_or_audio or "").strip()
        if not text:
            logger.warning("Azure avatar requires text or SSML input")
            return ""
        try:
            job_id = self.submit(text, presenter, width, height)
            if not job_id:
                logger.warning("Azure avatar submit did not return a job id")
                return ""
            logger.info(f"Azure avatar submitted job {job_id}")
            result_url = self._poll_until_done(job_id)
            if not result_url:
                return ""
            return self._download(result_url, out_path)
        except Exception as exc:  # noqa: BLE001 - avatar failures are non-fatal
            logger.warning(f"Azure avatar failed: {exc}")
            return ""


__all__ = ["AzureAvatar"]
