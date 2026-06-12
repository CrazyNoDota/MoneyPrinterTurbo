"""Provider-agnostic contract for AI video-clip generation.

The pipeline only ever talks to the :class:`VideoGenerator` interface, so the
actual backend (a hosted API like Replicate/fal, or our own GPU service) can be
swapped from config without touching ``task.py``.

Generation is modeled as async submit -> poll, because real text/image-to-video
models take tens of seconds to minutes per clip. ``generate`` is a synchronous
convenience that submits, polls with backoff, downloads the result, and returns
a local file path.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple


class VideoGenError(Exception):
    """Raised for non-recoverable video-generation problems (e.g. missing key)."""


@dataclass
class ClipSpec:
    """A single clip to generate.

    ``init_image`` (a local path or URL) selects image-to-video -- animating an
    already-on-theme photo we scraped/downloaded -- which is cheaper and more
    coherent than pure text-to-video. When it is empty the provider falls back
    to text-to-video from ``prompt``.
    """

    prompt: str
    init_image: str = ""
    duration: float = 5.0
    aspect: str = "9:16"
    seed: Optional[int] = None

    def cache_parts(self) -> tuple:
        """Stable identity used for on-disk caching of the generated clip."""
        return (self.prompt, self.init_image, self.duration, self.aspect, self.seed)


@dataclass
class GeneratedClip:
    """Result of a finished generation job."""

    url: str = ""
    local_path: str = ""
    meta: dict = field(default_factory=dict)


# A poll returns (status, result_url). status is one of: pending | done | failed.
PollResult = Tuple[str, str]

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class VideoGenerator(ABC):
    """Backend that turns a :class:`ClipSpec` into a video file.

    Subclasses implement ``submit`` + ``poll``; the orchestrator in
    ``videogen/__init__.py`` drives them. ``name`` is used for logging and as
    part of the cache key.
    """

    name: str = "base"

    def is_configured(self) -> bool:
        """Whether this provider has the credentials/endpoint it needs."""
        return True

    def download_headers(self) -> dict:
        """Extra HTTP headers for fetching the finished clip.

        Most providers return pre-signed URLs (no auth); Azure protects its
        content endpoint with the same api-key as the job API.
        """
        return {}

    @abstractmethod
    def submit(self, spec: ClipSpec) -> str:
        """Start a generation job and return an opaque job id/handle."""

    @abstractmethod
    def poll(self, job: str) -> PollResult:
        """Return ``(status, result_url)`` for a previously submitted job."""


__all__ = [
    "ClipSpec",
    "GeneratedClip",
    "VideoGenerator",
    "VideoGenError",
    "PollResult",
    "STATUS_PENDING",
    "STATUS_DONE",
    "STATUS_FAILED",
]
