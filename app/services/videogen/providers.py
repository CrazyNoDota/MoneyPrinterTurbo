"""Concrete video-generation backends.

All providers speak plain HTTP via ``requests`` (no extra SDK dependency) and
route their network calls through ``call_with_retry`` so transient failures are
retried. Hosted providers raise :class:`VideoGenError` when they are not
configured, which the orchestrator treats as "disabled" and falls back to stock
footage.

Config is read from ``config.app``:
    video_gen_provider   null | replicate | fal | http
    video_gen_api_key    credential for the hosted provider
    video_gen_model      model id / version (provider specific)
    video_gen_endpoint   base URL for the self-hosted ``http`` provider
    video_gen_mode       image2video | text2video  (informational; init_image wins)
"""

import base64
import os
from typing import Optional

import requests
from loguru import logger

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


def _image_payload(init_image: str) -> Optional[str]:
    """Return a data URI for a local image, or the URL itself if remote/empty."""
    if not init_image:
        return None
    if init_image.startswith(("http://", "https://", "data:")):
        return init_image
    if os.path.isfile(init_image):
        with open(init_image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(init_image)[1].lstrip(".").lower() or "jpeg"
        return f"data:image/{ext};base64,{b64}"
    return None


class NullVideoGenerator(VideoGenerator):
    """Default no-op backend: generation is disabled, nothing is produced."""

    name = "null"

    def is_configured(self) -> bool:
        return False

    def submit(self, spec: ClipSpec) -> str:  # pragma: no cover - never called
        raise VideoGenError("video generation is disabled (provider=null)")

    def poll(self, job: str) -> PollResult:  # pragma: no cover - never called
        raise VideoGenError("video generation is disabled (provider=null)")


class ReplicateVideoGenerator(VideoGenerator):
    """Replicate predictions REST API (https://replicate.com/docs).

    ``video_gen_model`` is the version hash to run. Submit returns a prediction
    id; poll reads its status until ``succeeded``/``failed``.
    """

    name = "replicate"
    _BASE = "https://api.replicate.com/v1/predictions"

    def is_configured(self) -> bool:
        return bool(_cfg("video_gen_api_key") and _cfg("video_gen_model"))

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {_cfg('video_gen_api_key')}",
            "Content-Type": "application/json",
        }

    def submit(self, spec: ClipSpec) -> str:
        if not self.is_configured():
            raise VideoGenError("replicate: video_gen_api_key / video_gen_model not set")
        model_input = {"prompt": spec.prompt}
        image = _image_payload(spec.init_image)
        if image:
            model_input["image"] = image
        if spec.seed is not None:
            model_input["seed"] = spec.seed
        payload = {"version": _cfg("video_gen_model"), "input": model_input}
        resp = call_with_retry(
            requests.post,
            self._BASE,
            headers=self._headers(),
            json=payload,
            proxies=config.proxy,
            timeout=(30, 120),
            description="replicate.submit",
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def poll(self, job: str) -> PollResult:
        resp = call_with_retry(
            requests.get,
            f"{self._BASE}/{job}",
            headers=self._headers(),
            proxies=config.proxy,
            timeout=(30, 60),
            description="replicate.poll",
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "succeeded":
            output = data.get("output")
            url = output[-1] if isinstance(output, list) and output else (output or "")
            return STATUS_DONE, url or ""
        if status in ("failed", "canceled"):
            return STATUS_FAILED, ""
        return STATUS_PENDING, ""


class FalVideoGenerator(VideoGenerator):
    """fal.ai queue REST API (https://fal.ai/docs).

    ``video_gen_model`` is the model route, e.g. ``fal-ai/ltx-video``.
    """

    name = "fal"
    _QUEUE = "https://queue.fal.run"

    def is_configured(self) -> bool:
        return bool(_cfg("video_gen_api_key") and _cfg("video_gen_model"))

    def _headers(self) -> dict:
        return {
            "Authorization": f"Key {_cfg('video_gen_api_key')}",
            "Content-Type": "application/json",
        }

    def submit(self, spec: ClipSpec) -> str:
        if not self.is_configured():
            raise VideoGenError("fal: video_gen_api_key / video_gen_model not set")
        body = {"prompt": spec.prompt}
        image = _image_payload(spec.init_image)
        if image:
            body["image_url"] = image
        if spec.seed is not None:
            body["seed"] = spec.seed
        resp = call_with_retry(
            requests.post,
            f"{self._QUEUE}/{_cfg('video_gen_model')}",
            headers=self._headers(),
            json=body,
            proxies=config.proxy,
            timeout=(30, 120),
            description="fal.submit",
        )
        resp.raise_for_status()
        # fal returns a status_url + response_url; keep the request id.
        return resp.json()["request_id"]

    def poll(self, job: str) -> PollResult:
        model = _cfg("video_gen_model")
        status_url = f"{self._QUEUE}/{model}/requests/{job}/status"
        resp = call_with_retry(
            requests.get,
            status_url,
            headers=self._headers(),
            proxies=config.proxy,
            timeout=(30, 60),
            description="fal.poll",
        )
        resp.raise_for_status()
        status = resp.json().get("status")
        if status == "COMPLETED":
            result = call_with_retry(
                requests.get,
                f"{self._QUEUE}/{model}/requests/{job}",
                headers=self._headers(),
                proxies=config.proxy,
                timeout=(30, 60),
                description="fal.result",
            )
            result.raise_for_status()
            data = result.json()
            video = data.get("video") or {}
            return STATUS_DONE, video.get("url", "")
        if status in ("FAILED", "ERROR"):
            return STATUS_FAILED, ""
        return STATUS_PENDING, ""


class HttpVideoGenerator(VideoGenerator):
    """Generic self-hosted backend we control (Google Cloud Run+GPU / Modal /
    RunPod / a custom box).

    Contract:
      POST  {endpoint}/submit  {prompt, image, duration, aspect, seed} -> {"job": "<id>"}
      GET   {endpoint}/status/{job}                                    ->
            {"status": "pending|done|failed", "url": "<result mp4 url>"}

    This is the slot to wire a cheap GPU service into later -- only this small
    contract has to be honored on the server side.
    """

    name = "http"

    def is_configured(self) -> bool:
        return bool(_cfg("video_gen_endpoint"))

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        key = _cfg("video_gen_api_key")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _base(self) -> str:
        return str(_cfg("video_gen_endpoint", "")).rstrip("/")

    def submit(self, spec: ClipSpec) -> str:
        if not self.is_configured():
            raise VideoGenError("http: video_gen_endpoint not set")
        payload = {
            "prompt": spec.prompt,
            "image": _image_payload(spec.init_image) or "",
            "duration": spec.duration,
            "aspect": spec.aspect,
            "seed": spec.seed,
            "model": _cfg("video_gen_model", ""),
        }
        resp = call_with_retry(
            requests.post,
            f"{self._base()}/submit",
            headers=self._headers(),
            json=payload,
            proxies=config.proxy,
            timeout=(30, 120),
            description="http.submit",
        )
        resp.raise_for_status()
        return resp.json()["job"]

    def poll(self, job: str) -> PollResult:
        resp = call_with_retry(
            requests.get,
            f"{self._base()}/status/{job}",
            headers=self._headers(),
            proxies=config.proxy,
            timeout=(30, 60),
            description="http.poll",
        )
        resp.raise_for_status()
        data = resp.json()
        status = (data.get("status") or "").lower()
        if status in ("done", "succeeded", "completed"):
            return STATUS_DONE, data.get("url", "")
        if status in ("failed", "error", "canceled"):
            return STATUS_FAILED, ""
        return STATUS_PENDING, ""


_PROVIDERS = {
    "null": NullVideoGenerator,
    "replicate": ReplicateVideoGenerator,
    "fal": FalVideoGenerator,
    "http": HttpVideoGenerator,
}


def build_provider(name: str) -> VideoGenerator:
    """Instantiate a provider by name, defaulting to the no-op Null backend."""
    cls = _PROVIDERS.get((name or "null").strip().lower())
    if cls is None:
        logger.warning(f"unknown video_gen_provider '{name}', using null")
        cls = NullVideoGenerator
    return cls()


__all__ = [
    "NullVideoGenerator",
    "ReplicateVideoGenerator",
    "FalVideoGenerator",
    "HttpVideoGenerator",
    "build_provider",
]
