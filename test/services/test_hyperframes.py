import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.hyperframes import author, scenes


def _valid_html(total=6.0):
    return f"""<!doctype html>
<html>
<head><script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script></head>
<body>
  <div id="root" data-composition-id="main" data-start="0" data-duration="{total}"
       data-width="1080" data-height="1920">
    <div class="clip" data-start="0" data-duration="3" data-track-index="1">One</div>
    <div class="clip" data-start="3" data-duration="3" data-track-index="2">Two</div>
  </div>
  <script>
    window.__timelines = window.__timelines || {{}};
    window.__timelines["main"] = gsap.timeline({{ paused: true }});
  </script>
</body>
</html>"""


class TestScenes(unittest.TestCase):
    def test_from_script_sums_to_total_duration(self):
        script = "First sentence here. Second one is a bit longer than the first! And a third?"
        built = scenes.from_script(script, total_duration=12.0)
        self.assertEqual(len(built), 3)
        self.assertAlmostEqual(built[0].start, 0.0)
        self.assertAlmostEqual(built[-1].end, 12.0, places=2)
        # scenes are contiguous
        for a, b in zip(built, built[1:]):
            self.assertAlmostEqual(a.end, b.start, places=2)

    def test_from_script_empty(self):
        self.assertEqual(scenes.from_script("", 10.0), [])
        self.assertEqual(scenes.from_script("hello", 0), [])

    def test_parse_srt_range(self):
        rng = scenes._parse_srt_range("00:00:01,500 --> 00:00:04,000")
        self.assertIsNotNone(rng)
        start, end = rng
        self.assertAlmostEqual(start, 1.5)
        self.assertAlmostEqual(end, 4.0)

    def test_from_subtitle_uses_srt_timing(self):
        srt = [
            (1, "00:00:00,000 --> 00:00:02,000", "Hello world"),
            (2, "00:00:02,000 --> 00:00:05,500", "Second line"),
        ]
        with mock.patch.object(scenes.subtitle, "file_to_subtitles", return_value=srt):
            built = scenes.from_subtitle("dummy.srt")
        self.assertEqual(len(built), 2)
        self.assertAlmostEqual(built[0].duration, 2.0)
        self.assertAlmostEqual(built[1].start, 2.0)
        self.assertAlmostEqual(built[1].duration, 3.5)

    def test_build_scenes_prefers_subtitle(self):
        srt = [(1, "00:00:00,000 --> 00:00:03,000", "From srt")]
        with mock.patch.object(scenes.subtitle, "file_to_subtitles", return_value=srt):
            built = scenes.build_scenes("ignored script.", "dummy.srt", 10.0)
        self.assertEqual(len(built), 1)
        self.assertEqual(built[0].text, "From srt")


class TestAuthorValidation(unittest.TestCase):
    def test_strip_fences(self):
        fenced = "```html\n<!doctype html><html></html>\n```"
        self.assertTrue(author._strip_fences(fenced).startswith("<!doctype"))
        prosed = "Here you go:\n<html></html>"
        self.assertTrue(author._strip_fences(prosed).startswith("<html"))

    def test_validate_accepts_good(self):
        ok, reason = author._validate(_valid_html(), scene_count=2, total=6.0)
        self.assertTrue(ok, reason)

    def test_validate_rejects_missing_timeline(self):
        html = _valid_html().replace("window.__timelines", "window.__nope")
        ok, reason = author._validate(html, 2, 6.0)
        self.assertFalse(ok)

    def test_validate_rejects_nondeterministic(self):
        html = _valid_html().replace("One</div>", "<span>x</span>One</div>")
        html = html.replace("paused: true", "paused: true /* Math.random() */")
        ok, reason = author._validate(html, 2, 6.0)
        self.assertFalse(ok)
        self.assertIn("random", reason)

    def test_validate_rejects_too_few_clips(self):
        ok, reason = author._validate(_valid_html(), scene_count=5, total=6.0)
        self.assertFalse(ok)

    def _scenes(self):
        return [scenes.Scene("One", 0.0, 3.0), scenes.Scene("Two", 3.0, 3.0)]

    def test_author_composition_succeeds(self):
        with mock.patch.object(author.llm, "_generate_response", return_value=_valid_html()):
            html = author.author_composition(self._scenes(), "money", 1080, 1920)
        self.assertIn('data-composition-id="main"', html)

    def test_author_composition_retries_then_succeeds(self):
        responses = ["garbage not html", _valid_html()]
        with mock.patch.object(author.llm, "_generate_response", side_effect=responses) as m:
            html = author.author_composition(self._scenes(), "money", 1080, 1920)
        self.assertEqual(m.call_count, 2)
        self.assertTrue(html)

    def test_author_composition_gives_up(self):
        with mock.patch.object(author.llm, "_generate_response", return_value="still not html"):
            html = author.author_composition(self._scenes(), "money", 1080, 1920)
        self.assertEqual(html, "")


if __name__ == "__main__":
    unittest.main()
