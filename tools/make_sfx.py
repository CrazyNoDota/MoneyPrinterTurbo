"""
tools/make_sfx.py — synthesise the five SFX assets used by WP3/WP4.

All sounds are generated deterministically via ffmpeg filtergraphs (no network
downloads). Each file is skipped when it already exists unless --force is given.

Usage:
    python tools/make_sfx.py           # create missing files
    python tools/make_sfx.py --force   # overwrite all
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Locate ffmpeg — mirror exactly the logic in app/services/video.py so we
# reuse whatever binary the rest of the project uses.
# ---------------------------------------------------------------------------

def _get_ffmpeg_binary() -> str:
    configured = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured:
        return configured

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            return bundled
    except Exception:
        pass

    return "ffmpeg"


def _repo_root() -> Path:
    # tools/ sits one level below the repo root
    return Path(__file__).resolve().parent.parent


def _sfx_dir() -> Path:
    return _repo_root() / "resource" / "sfx"


# ---------------------------------------------------------------------------
# SFX definitions — (filename, duration_hint_s, filtergraph)
# Peak levels are kept at roughly -6 dBFS via a final volume= stage.
# ---------------------------------------------------------------------------

SFX_SPECS: list[tuple[str, float, str]] = [
    # whoosh.mp3 (~0.4 s): band-passed noise with fast downward frequency sweep
    # + volume envelope (fade in then out).
    (
        "whoosh.mp3",
        0.4,
        (
            "anoisesrc=d=0.4:c=white:a=0.9,"
            "afade=t=in:st=0:d=0.02,"
            "afade=t=out:st=0.30:d=0.10,"
            "bandpass=f=1200:width_type=o:width=3,"
            "volume=0.5"
        ),
    ),
    # pop.mp3 (~0.15 s): short sine burst at 220 Hz with fast exponential decay
    (
        "pop.mp3",
        0.15,
        (
            "sine=frequency=220:duration=0.15,"
            "afade=t=in:st=0:d=0.005,"
            "afade=t=out:st=0.05:d=0.10,"
            "volume=0.5"
        ),
    ),
    # ding.mp3 (~0.6 s): 880 Hz fundamental + 1760 Hz harmonic, exponential decay
    (
        "ding.mp3",
        0.6,
        (
            "sine=frequency=880:duration=0.6[a];"
            "sine=frequency=1760:duration=0.6[b];"
            "[a][b]amix=inputs=2:duration=first:weights=1 0.4,"
            "afade=t=in:st=0:d=0.002,"
            "afade=t=out:st=0.15:d=0.45,"
            "volume=0.5"
        ),
    ),
    # tick.mp3 (~0.1 s): very short filtered noise burst (mechanical click feel)
    (
        "tick.mp3",
        0.1,
        (
            "anoisesrc=d=0.1:c=white:a=1.0,"
            "afade=t=in:st=0:d=0.002,"
            "afade=t=out:st=0.03:d=0.07,"
            "highpass=f=3000,"
            "lowpass=f=8000,"
            "volume=0.5"
        ),
    ),
    # riser.mp3 (~1.5 s): rising sine sweep 80 Hz → 1600 Hz ending abruptly
    # (suspense riser before reveal).  Use aevalsrc for a frequency-modulated
    # sine that starts at 80 Hz and sweeps upward, then trim + abrupt cut.
    (
        "riser.mp3",
        1.5,
        (
            # aevalsrc generates a 1.5 s audio signal; the frequency ramps from
            # 80 Hz to 1600 Hz linearly over the duration.
            "aevalsrc="
            "'sin(2*PI*(80 + (1600-80)*t/1.5)*t)'"
            ":c=mono:s=44100:d=1.5,"
            "afade=t=in:st=0:d=0.1,"
            "afade=t=out:st=1.4:d=0.1,"
            "volume=0.5"
        ),
    ),
]


def _run_ffmpeg(ffmpeg: str, output_path: Path, filtergraph: str, duration: float) -> bool:
    """
    Synthesise audio via an ffmpeg filtergraph and write to output_path.
    Returns True on success, False on failure (prints the error to stderr).
    """
    cmd = [
        ffmpeg,
        "-y",                          # overwrite without asking
        "-f", "lavfi",
        "-i", filtergraph,
        "-t", str(duration),
        "-ar", "44100",
        "-ac", "1",
        "-b:a", "128k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: ffmpeg failed for {output_path.name}", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesise SFX assets into resource/sfx/")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    ffmpeg = _get_ffmpeg_binary()
    sfx_dir = _sfx_dir()
    sfx_dir.mkdir(parents=True, exist_ok=True)

    print(f"ffmpeg binary : {ffmpeg}")
    print(f"output dir    : {sfx_dir}")
    print()

    errors = 0
    for filename, duration, filtergraph in SFX_SPECS:
        out = sfx_dir / filename
        if out.exists() and not args.force:
            print(f"  skip   {filename}  (already exists; use --force to regenerate)")
            continue

        print(f"  synth  {filename}  ({duration}s)...", end=" ", flush=True)
        ok = _run_ffmpeg(ffmpeg, out, filtergraph, duration)
        if ok:
            print("OK")
        else:
            errors += 1

    print()
    if errors:
        print(f"{errors} file(s) failed to generate.", file=sys.stderr)
        return 1
    print("All SFX files present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
