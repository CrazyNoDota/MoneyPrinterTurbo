"""Local Wav2Lip fallback provider driven through an isolated subprocess."""

import os
import subprocess
from typing import Any

from loguru import logger

from app.config import config
from app.utils import utils

from .base import TalkingHead


def _cfg(key: str, default: Any = None) -> Any:
    return config.app.get(key, default)


class Wav2LipAvatar(TalkingHead):
    """Fallback provider for audio + portrait photo lip sync."""

    name = "wav2lip"

    def _python(self) -> str:
        override = str(_cfg("wav2lip_python", "") or "").strip()
        if override:
            return override
        root = utils.root_dir()
        if os.name == "nt":
            return os.path.join(root, ".venv-wav2lip", "Scripts", "python.exe")
        return os.path.join(root, ".venv-wav2lip", "bin", "python")

    def _repo_dir(self) -> str:
        return str(_cfg("wav2lip_repo", "") or os.path.join(utils.root_dir(), ".wav2lip", "Wav2Lip"))

    def _script(self) -> str:
        return str(_cfg("wav2lip_script", "") or os.path.join(self._repo_dir(), "inference.py"))

    def _weights(self) -> str:
        return str(
            _cfg(
                "wav2lip_weights",
                os.path.join(utils.root_dir(), "models", "wav2lip", "wav2lip_gan.pth"),
            )
        )

    def is_configured(self) -> bool:
        return (
            os.path.exists(self._python())
            and os.path.isfile(self._script())
            and os.path.isfile(self._weights())
            and os.path.getsize(self._weights()) > 0
        )

    def synthesize(
        self,
        script_or_audio: str,
        presenter: str,
        out_path: str,
        width: int,
        height: int,
    ) -> str:
        audio_path = script_or_audio or ""
        portrait_path = presenter or str(_cfg("avatar_portrait", "") or "")

        if not os.path.isfile(audio_path):
            logger.warning(f"Wav2Lip audio missing: {audio_path}")
            return ""
        if not os.path.isfile(portrait_path):
            logger.warning(f"Wav2Lip portrait missing: {portrait_path}")
            return ""
        if not self.is_configured():
            logger.warning(
                "Wav2Lip is not configured; expected .venv-wav2lip, inference.py, "
                "and wav2lip_gan.pth"
            )
            return ""

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cmd = [
            self._python(),
            self._script(),
            "--checkpoint_path",
            self._weights(),
            "--face",
            portrait_path,
            "--audio",
            audio_path,
            "--outfile",
            out_path,
        ]
        resize_factor = str(_cfg("wav2lip_resize_factor", "") or "").strip()
        if resize_factor:
            cmd.extend(["--resize_factor", resize_factor])

        timeout = int(_cfg("wav2lip_timeout", 1800) or 1800)
        try:
            logger.info(f"start Wav2Lip avatar subprocess: {os.path.basename(out_path)}")
            proc = subprocess.run(
                cmd,
                cwd=self._repo_dir(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                logger.warning(f"Wav2Lip failed: {detail}")
                return ""
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                logger.success(f"Wav2Lip avatar produced {out_path}")
                return out_path
            logger.warning("Wav2Lip completed without producing output")
            return ""
        except subprocess.TimeoutExpired:
            logger.warning(f"Wav2Lip timed out after {timeout}s")
            return ""
        except Exception as exc:  # noqa: BLE001 - avatar failures are non-fatal
            logger.warning(f"Wav2Lip failed: {exc}")
            return ""


__all__ = ["Wav2LipAvatar"]
