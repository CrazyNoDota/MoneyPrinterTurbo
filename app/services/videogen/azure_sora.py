"""Azure OpenAI Sora backend for the videogen layer.

Speaks the Azure OpenAI video-generation jobs API (preview): submit a job,
poll its status, then download the finished mp4 from the generation-content
endpoint. Runs on the startup's Azure credits, so it sits behind the same
``video_gen_*`` flags and budget caps as every other provider.

⚠️ Sunset: the Sora API on Azure shuts down 2026-09-24. After that date this
provider starts failing closed -- the orchestrator already degrades every
failure to stock footage, so nothing else has to change; just flip
``video_gen_provider`` back to ``null``.

Unlike the other hosted providers, Azure's download URL is NOT pre-signed:
fetching the content requires the ``api-key`` header, which the orchestrator
attaches via :meth:`download_headers`.
"""

from loguru import logger
import requests

from app.config import config
from app.utils.retry import call_with_retry

from .base import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    ClipSpec,
    PollResult,
    VideoGenError,
    VideoGenerator,
)


def _cfg(key: str, default=None):
    return config.app.get(key, default)


# Sora accepts explicit pixel dimensions, not an aspect string. 720p tier is
# the b-roll sweet spot (billing scales with resolution and seconds).
_RESOLUTIONS = {
    "9:16": (720, 1280),
    "16:9": (1280, 720),
    "1:1": (720, 720),
}


class AzureSoraVideoGenerator(VideoGenerator):
    """Azure OpenAI ``/v1/video/generations/jobs`` submit/poll backend."""

    name = "azure-sora"

    def is_configured(self) -> bool:
        return bool(_cfg("video_gen_endpoint") and _cfg("video_gen_api_key"))

    def _base(self) -> str:
        # The resource endpoint, e.g. https://<resource>.openai.azure.com
        return str(_cfg("video_gen_endpoint", "")).rstrip("/")

    def _api_version(self) -> str:
        return str(_cfg("video_gen_api_version", "preview") or "preview")

    def _headers(self) -> dict:
        return {"api-key": str(_cfg("video_gen_api_key", "")), "Content-Type": "application/json"}

    def download_headers(self) -> dict:
        """Auth the orchestrator must send when fetching the finished mp4."""
        return {"api-key": str(_cfg("video_gen_api_key", ""))}

    def submit(self, spec: ClipSpec) -> str:
        if not self.is_configured():
            raise VideoGenError(
                "azure-sora: video_gen_endpoint / video_gen_api_key not set"
            )
        if spec.init_image:
            # The jobs API is text-to-video; on-theme prompts still keep b-roll
            # coherent, so we proceed rather than fail the clip.
            logger.debug("azure-sora ignores init_image (text-to-video only)")
        width, height = _RESOLUTIONS.get(spec.aspect, _RESOLUTIONS["9:16"])
        payload = {
            "model": str(_cfg("video_gen_model", "") or "sora"),
            "prompt": spec.prompt,
            "width": width,
            "height": height,
            "n_seconds": max(1, min(20, int(round(spec.duration or 5)))),
            "n_variants": 1,
        }
        resp = call_with_retry(
            requests.post,
            f"{self._base()}/openai/v1/video/generations/jobs"
            f"?api-version={self._api_version()}",
            headers=self._headers(),
            json=payload,
            proxies=config.proxy,
            timeout=(30, 120),
            description="azure-sora.submit",
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def poll(self, job: str) -> PollResult:
        resp = call_with_retry(
            requests.get,
            f"{self._base()}/openai/v1/video/generations/jobs/{job}"
            f"?api-version={self._api_version()}",
            headers=self._headers(),
            proxies=config.proxy,
            timeout=(30, 60),
            description="azure-sora.poll",
        )
        resp.raise_for_status()
        data = resp.json()
        status = (data.get("status") or "").lower()
        if status == "succeeded":
            generations = data.get("generations") or []
            gen_id = (generations[0] or {}).get("id", "") if generations else ""
            if not gen_id:
                logger.warning(f"azure-sora job {job} succeeded without generations")
                return STATUS_FAILED, ""
            return STATUS_DONE, (
                f"{self._base()}/openai/v1/video/generations/{gen_id}"
                f"/content/video?api-version={self._api_version()}"
            )
        if status in ("failed", "cancelled", "canceled"):
            logger.warning(
                f"azure-sora job {job} {status}: "
                f"{(data.get('failure_reason') or data.get('error') or '')}"
            )
            return STATUS_FAILED, ""
        # queued | preprocessing | running | processing -> keep polling.
        return STATUS_PENDING, ""


__all__ = ["AzureSoraVideoGenerator"]
