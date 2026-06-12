"""WP6 pacing: scene-duration cap splits long scenes (scenes.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.hyperframes import scenes
from app.services.hyperframes.scenes import Scene, _split_long_scenes, _split_text_at_word_boundary


def _total_duration(items):
    return round(sum(s.duration for s in items), 3)


def _contiguous(items):
    """True when each scene starts where the previous one ended (within rounding)."""
    for i in range(1, len(items)):
        if abs(items[i].start - items[i - 1].end) > 0.01:
            return False
    return True


class SplitTextTests(unittest.TestCase):
    def test_no_split_for_single_word(self):
        self.assertEqual(_split_text_at_word_boundary("Hello", 3), ["Hello"])

    def test_splits_into_requested_parts(self):
        text = "one two three four five six"
        parts = _split_text_at_word_boundary(text, 3)
        self.assertEqual(len(parts), 3)
        # No word dropped or duplicated; rejoining gives the original words.
        self.assertEqual(" ".join(parts).split(), text.split())

    def test_no_word_is_split(self):
        text = "alpha beta gamma delta"
        parts = _split_text_at_word_boundary(text, 2)
        for p in parts:
            for w in p.split():
                self.assertIn(w, text.split())

    def test_prefers_punctuation_boundary(self):
        # "first, second third fourth" -> the comma after "first," is a natural break.
        text = "first, second third fourth"
        parts = _split_text_at_word_boundary(text, 2)
        self.assertEqual(parts[0], "first,")


class SplitLongScenesTests(unittest.TestCase):
    def test_short_scene_untouched(self):
        s = Scene(text="a quick line", start=0.0, duration=3.0)
        out = _split_long_scenes([s], cap=4.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "a quick line")
        self.assertEqual(out[0].duration, 3.0)

    def test_long_scene_splits_at_cap(self):
        s = Scene(text="one two three four five six seven eight", start=0.0, duration=12.0)
        out = _split_long_scenes([s], cap=4.0)
        self.assertGreaterEqual(len(out), 3)
        for sub in out:
            self.assertLessEqual(sub.duration, 4.0 + 1e-6)

    def test_total_duration_preserved(self):
        s = Scene(text="one two three four five six seven eight nine ten", start=2.0, duration=10.0)
        out = _split_long_scenes([s], cap=4.0)
        self.assertAlmostEqual(_total_duration(out), 10.0, places=3)
        # First sub-scene starts where the original did; pieces stay contiguous.
        self.assertAlmostEqual(out[0].start, 2.0, places=3)
        self.assertTrue(_contiguous(out))
        self.assertAlmostEqual(out[-1].end, 12.0, places=2)

    def test_order_and_text_preserved(self):
        s = Scene(text="alpha bravo charlie delta echo foxtrot", start=0.0, duration=9.0)
        out = _split_long_scenes([s], cap=4.0)
        joined = " ".join(sub.text for sub in out)
        self.assertEqual(joined.split(), s.text.split())

    def test_cap_disabled_when_zero(self):
        s = Scene(text="one two three four five six", start=0.0, duration=20.0)
        out = _split_long_scenes([s], cap=0.0)
        self.assertEqual(len(out), 1)

    def test_mixed_list_only_splits_long(self):
        items = [
            Scene(text="short one", start=0.0, duration=2.0),
            Scene(text="long scene with many words to split apart", start=2.0, duration=10.0),
            Scene(text="short two", start=12.0, duration=2.0),
        ]
        out = _split_long_scenes(items, cap=4.0)
        self.assertGreater(len(out), 3)
        self.assertEqual(out[0].text, "short one")
        self.assertEqual(out[-1].text, "short two")
        self.assertAlmostEqual(_total_duration(out), _total_duration(items), places=3)


class BuildScenesCapTests(unittest.TestCase):
    def setUp(self):
        # config.app is a plain dict; remember/restore the override key.
        self._had = "scene_max_seconds" in scenes.config.app
        self._prev = scenes.config.app.get("scene_max_seconds")

    def tearDown(self):
        if self._had:
            scenes.config.app["scene_max_seconds"] = self._prev
        else:
            scenes.config.app.pop("scene_max_seconds", None)

    def test_build_scenes_respects_config_override(self):
        long_script = (
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet."
        )
        # cap = 1.0s forces aggressive splitting; default would split less.
        scenes.config.app["scene_max_seconds"] = 1.0
        out_capped = scenes.build_scenes(long_script, srt_path="", total_duration=10.0)
        # The same script with a generous cap splits into far fewer scenes, proving
        # the override drives the splitting (cap is a soft target: a single long
        # word may slightly exceed it since words are never split).
        scenes.config.app["scene_max_seconds"] = 100.0
        out_loose = scenes.build_scenes(long_script, srt_path="", total_duration=10.0)
        self.assertGreater(len(out_capped), len(out_loose))
        self.assertAlmostEqual(_total_duration(out_capped), 10.0, places=2)

    def test_build_scenes_default_cap_splits_long(self):
        # One long sentence over the ~4s default should split.
        script = "word one two three four five six seven eight nine ten eleven twelve."
        scenes.config.app.pop("scene_max_seconds", None)  # use the code default
        out = scenes.build_scenes(script, srt_path="", total_duration=12.0)
        self.assertGreater(len(out), 1)
        self.assertAlmostEqual(_total_duration(out), 12.0, places=2)


if __name__ == "__main__":
    unittest.main()
