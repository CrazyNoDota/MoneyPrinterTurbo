"""Talking-head generation resolver.

This module is intentionally not wired into the main video pipeline yet. Phase 1
only delivers the provider layer and non-fatal fallback chain.
"""

from loguru import logger

from app.config import config

from .azure import AzureAvatar
from .base import TalkingHead
from .wav2lip import Wav2LipAvatar


def _provider_name() -> str:
    return str(config.app.get("avatar_provider", "auto") or "auto").strip().lower()


def is_enabled() -> bool:
    return bool(config.app.get("avatar_enabled", False))


def _providers_for(name: str) -> list[TalkingHead]:
    if name == "azure":
        return [AzureAvatar()]
    if name == "wav2lip":
        return [Wav2LipAvatar()]
    if name in ("", "auto"):
        return [AzureAvatar(), Wav2LipAvatar()]
    logger.warning(f"unknown avatar_provider '{name}', using auto")
    return [AzureAvatar(), Wav2LipAvatar()]


def synthesize(
    script_or_audio: str,
    presenter: str = "",
    out_path: str = "",
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Generate a talking-head clip through the configured provider chain."""
    if not is_enabled():
        return ""

    if not out_path:
        logger.warning("avatar output path is empty")
        return ""

    for provider in _providers_for(_provider_name()):
        try:
            result = provider.synthesize(
                script_or_audio=script_or_audio,
                presenter=presenter,
                out_path=out_path,
                width=width,
                height=height,
            )
        except Exception as exc:  # noqa: BLE001 - resolver must never crash pipeline
            logger.warning(f"avatar provider {provider.name} crashed: {exc}")
            result = ""
        if result:
            return result
    return ""


__all__ = ["AzureAvatar", "TalkingHead", "Wav2LipAvatar", "is_enabled", "synthesize"]
