"""Render an authored composition to MP4 by shelling out to the hyperframes CLI.

Mirrors the isolated-subprocess pattern in ``app.services.voice.qwen_tts``: locate
the toolchain, run it with a timeout, capture output, and degrade gracefully (return
``""``) when the toolchain is missing or the render fails. Node + a headless Chrome
(fetched by ``setup-hyperframes.bat``) + ffmpeg/ffprobe must be available; a static
ffmpeg bundled under ``<project>/bin`` is prepended to PATH automatically.
"""

import os
import shutil
import subprocess

from loguru import logger

from app.config import config
from app.utils import utils

# Pin the CLI so renders are reproducible regardless of what npx would resolve.
_DEFAULT_VERSION = "0.6.58"


def project_dir() -> str:
    """Absolute path to the hyperframes project (created by setup-hyperframes.bat)."""
    d = config.app.get("hyperframes_dir", ".hyperframes")
    if not os.path.isabs(d):
        d = os.path.join(utils.root_dir(), d)
    return d


def is_available() -> bool:
    """True when the project exists and Node is reachable."""
    if not os.path.exists(os.path.join(project_dir(), "index.html")):
        return False
    return _npx() is not None


def _npx():
    node_dir = config.app.get("hyperframes_node", "")
    if node_dir:
        cand = shutil.which("npx", path=node_dir) or shutil.which("npx.cmd", path=node_dir)
        if cand:
            return cand
    return shutil.which("npx") or shutil.which("npx.cmd")


def _render_env(proj: str) -> dict:
    """A copy of os.environ with the bundled ffmpeg dir prepended to PATH."""
    env = os.environ.copy()
    bin_dir = os.path.join(proj, "bin")
    node_dir = config.app.get("hyperframes_node", "")
    extra = [p for p in (bin_dir, node_dir) if p and os.path.isdir(p)]
    if extra:
        env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
    return env


def render(html: str, out_path: str) -> str:
    """Write ``html`` into the project, render it, and return ``out_path`` or ``""``."""
    proj = project_dir()
    npx = _npx()
    if not npx or not os.path.exists(os.path.join(proj, "index.html")):
        logger.error(
            "hyperframes toolchain not found. Run setup-hyperframes.bat once "
            "(needs Node 22+, a headless Chrome, and ffmpeg/ffprobe)."
        )
        return ""

    version = config.app.get("hyperframes_version", _DEFAULT_VERSION)
    timeout = int(config.app.get("hyperframes_timeout", 600))
    fps = str(config.app.get("hyperframes_fps", 30))

    # Author into the project's index.html (the proven render entrypoint) and write
    # the MP4 outside the project so it survives re-scaffolding.
    comp_file = os.path.join(proj, "index.html")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        with open(comp_file, "w", encoding="utf-8") as f:
            f.write(html)

        cmd = [npx, "--yes", f"hyperframes@{version}", "render",
               "-o", out_path, "-f", fps, "--quiet"]
        logger.info(f"hyperframes render (subprocess): {os.path.basename(out_path)} @ {fps}fps")
        proc = subprocess.run(
            cmd, cwd=proj, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            env=_render_env(proj),
        )
        if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-800:]
            logger.error(f"hyperframes render failed (rc={proc.returncode}): {detail}")
            return ""
        logger.success(f"hyperframes rendered {os.path.basename(out_path)}")
        return out_path
    except subprocess.TimeoutExpired:
        logger.error(f"hyperframes render timed out after {timeout}s")
        return ""
    except Exception as exc:  # noqa: BLE001 - render is best-effort
        logger.error(f"hyperframes render error: {exc}")
        return ""
