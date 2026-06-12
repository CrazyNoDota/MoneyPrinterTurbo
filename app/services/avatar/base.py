"""Provider-agnostic contract for talking-head generation."""

from abc import ABC, abstractmethod


class TalkingHead(ABC):
    """Backend that turns text/SSML or audio+portrait into a presenter video.

    Implementations must be non-fatal: return the output path on success and
    ``""`` on any failure so callers can continue without a presenter layer.
    """

    name: str = "base"

    def is_configured(self) -> bool:
        """Whether this provider has the credentials/files it needs."""
        return True

    @abstractmethod
    def synthesize(
        self,
        script_or_audio: str,
        presenter: str,
        out_path: str,
        width: int,
        height: int,
    ) -> str:
        """Create a talking-head clip and return ``out_path`` or ``""``."""


__all__ = ["TalkingHead"]
