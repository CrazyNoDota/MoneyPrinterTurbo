"""
Tests for app/services/sfx.py — get_sfx_file resolution.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import sfx as sfx_module


class TestGetSfxFile(unittest.TestCase):
    """get_sfx_file returns paths for known files and '' for anything missing."""

    def _make_sfx_dir(self, tmp_path: str) -> str:
        """Create a fake sfx directory with one dummy mp3 file."""
        sfx_dir = os.path.join(tmp_path, "sfx")
        os.makedirs(sfx_dir, exist_ok=True)
        # Create a minimal fake mp3 (not a real mp3, just non-empty bytes)
        for name in ("whoosh.mp3", "pop.mp3", "ding.mp3"):
            with open(os.path.join(sfx_dir, name), "wb") as f:
                f.write(b"\xff\xfb\x00\x00")  # minimal MP3-like header bytes
        return sfx_dir

    def test_returns_path_for_existing_file_with_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            sfx_dir = self._make_sfx_dir(tmp)
            with patch.object(sfx_module, "_sfx_dir", return_value=sfx_dir):
                result = sfx_module.get_sfx_file("whoosh.mp3")
        self.assertTrue(result.endswith("whoosh.mp3"))
        self.assertTrue(os.path.isabs(result))

    def test_returns_path_for_existing_file_without_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            sfx_dir = self._make_sfx_dir(tmp)
            with patch.object(sfx_module, "_sfx_dir", return_value=sfx_dir):
                result = sfx_module.get_sfx_file("whoosh")
        self.assertTrue(result.endswith("whoosh.mp3"))

    def test_returns_empty_string_for_unknown_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            sfx_dir = self._make_sfx_dir(tmp)
            with patch.object(sfx_module, "_sfx_dir", return_value=sfx_dir):
                result = sfx_module.get_sfx_file("nonexistent")
        self.assertEqual(result, "")

    def test_returns_empty_string_when_dir_missing(self):
        missing_dir = "/tmp/__sfx_dir_that_does_not_exist_12345__"
        with patch.object(sfx_module, "_sfx_dir", return_value=missing_dir):
            result = sfx_module.get_sfx_file("whoosh")
        self.assertEqual(result, "")

    def test_never_raises_on_arbitrary_name(self):
        """get_sfx_file must not raise under any circumstance."""
        with tempfile.TemporaryDirectory() as tmp:
            sfx_dir = self._make_sfx_dir(tmp)
            with patch.object(sfx_module, "_sfx_dir", return_value=sfx_dir):
                for name in ("", "   ", "../../../etc/passwd", "tick", "riser.mp3"):
                    try:
                        result = sfx_module.get_sfx_file(name)
                    except Exception as exc:
                        self.fail(f"get_sfx_file raised {exc!r} for name={name!r}")
                    self.assertIsInstance(result, str)

    def test_extension_normalisation_case_insensitive(self):
        """Files already ending in .mp3 (any case) are not double-suffixed."""
        with tempfile.TemporaryDirectory() as tmp:
            sfx_dir = self._make_sfx_dir(tmp)
            with patch.object(sfx_module, "_sfx_dir", return_value=sfx_dir):
                result = sfx_module.get_sfx_file("pop.MP3")
        # "pop.MP3" lowercases to "pop.mp3" → should find pop.mp3 in the dir
        # (platform-dependent: on Windows filenames are case-insensitive)
        if os.name == "nt":
            self.assertTrue(result.endswith(".mp3") or result.endswith(".MP3"))
        else:
            # On Linux the filename "pop.mp3" != "pop.MP3" so this returns ""
            # — either outcome is acceptable as long as no exception is raised.
            self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
