import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import types

from PIL import Image

from app.services import hyperframes as hf
from app.services.hyperframes import assemble, author, plan, preview, scenes, studio
from app.services.hyperframes.assets import Background


def _valid_html(total=6.0):
    return f"""<!doctype html>
<html>
<head><script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script></head>
<body>
  <div id="root" data-composition-id="main" data-start="0" data-duration="{total}"
       data-width="1080" data-height="1920">
    <div class="clip" data-start="0" data-duration="{total}" data-track-index="0">bg</div>
    <div class="clip" data-start="0" data-duration="3" data-track-index="1">One</div>
    <div class="clip" data-start="3" data-duration="3" data-track-index="2">Two</div>
  </div>
  <script>
    window.__timelines = window.__timelines || {{}};
    const tl = gsap.timeline({{ paused: true }});
    tl.fromTo(".clip:nth-of-type(1)", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 0);
    tl.fromTo(".clip:nth-of-type(2)", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 3);
    window.__timelines["main"] = tl;
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
        self.assertAlmostEqual(built[0].duration, 10.0)

    def test_build_scenes_expands_subtitle_gaps_to_audio_duration(self):
        srt = [
            (1, "00:00:01,000 --> 00:00:02,000", "First"),
            (2, "00:00:04,000 --> 00:00:05,000", "Second"),
        ]
        with mock.patch.object(scenes.subtitle, "file_to_subtitles", return_value=srt):
            built = scenes.build_scenes("ignored script.", "dummy.srt", 8.0)
        self.assertAlmostEqual(built[0].start, 0.0)
        self.assertAlmostEqual(built[-1].end, 8.0)
        self.assertAlmostEqual(sum(s.duration for s in built), 8.0)


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
        with mock.patch.object(author, "_engine", return_value="freeform"), \
             mock.patch.object(author.llm, "_generate_response", return_value=_valid_html()):
            html = author.author_composition(self._scenes(), "money", 1080, 1920)
        self.assertIn('data-composition-id="main"', html)

    def test_author_composition_retries_then_succeeds(self):
        responses = ["garbage not html", _valid_html()]
        with mock.patch.object(author, "_engine", return_value="freeform"), \
             mock.patch.object(author.llm, "_generate_response", side_effect=responses) as m:
            html = author.author_composition(self._scenes(), "money", 1080, 1920)
        self.assertEqual(m.call_count, 2)
        self.assertTrue(html)

    def test_author_composition_gives_up(self):
        with mock.patch.object(author, "_engine", return_value="freeform"), \
             mock.patch.object(author.llm, "_generate_response", return_value="still not html"):
            html = author.author_composition(self._scenes(), "money", 1080, 1920)
        self.assertEqual(html, "")


def _valid_html_bg(total=6.0, asset="bg1.jpg"):
    return f"""<!doctype html>
<html>
<head><script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script></head>
<body>
  <div id="root" data-composition-id="main" data-start="0" data-duration="{total}"
       data-width="1080" data-height="1920">
    <div class="clip" data-start="0" data-duration="{total}" data-track-index="0">bg</div>
    <img class="clip" data-start="0" data-duration="3" data-track-index="1" src="assets/{asset}">
    <div class="clip" data-start="0" data-duration="3" data-track-index="2">One</div>
    <div class="clip" data-start="3" data-duration="3" data-track-index="3">Two</div>
  </div>
  <script>
    window.__timelines = window.__timelines || {{}};
    const tl = gsap.timeline({{ paused: true }});
    tl.fromTo(".clip:nth-of-type(1)", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 0);
    tl.fromTo(".clip:nth-of-type(2)", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 3);
    window.__timelines["main"] = tl;
  </script>
</body>
</html>"""


class TestAuthorBackgrounds(unittest.TestCase):
    def test_validate_accepts_listed_asset(self):
        ok, reason = author._validate(_valid_html_bg(), 2, 6.0, asset_files=["bg1.jpg"])
        self.assertTrue(ok, reason)

    def test_validate_rejects_unlisted_asset(self):
        ok, reason = author._validate(_valid_html_bg(), 2, 6.0, asset_files=[])
        self.assertFalse(ok)
        self.assertIn("bg1.jpg", reason)

    def test_assets_block_lists_backgrounds(self):
        from app.services.hyperframes.assets import Background

        block = author._assets_block([Background(filename="assets/x.jpg", description="a lake")])
        self.assertIn("assets/x.jpg", block)
        self.assertIn("a lake", block)


class TestPlanner(unittest.TestCase):
    def _scenes(self):
        return [
            scenes.Scene("A calm forest at dawn", 0.0, 3.0),
            scenes.Scene("Revenue grew 25% in 2024", 3.0, 3.0),
        ]

    def test_heuristic_kind(self):
        self.assertEqual(plan._heuristic_kind("Revenue grew 25%"), "motiongraphics")
        self.assertEqual(plan._heuristic_kind("a quiet beach"), "footage")

    def test_parse_valid(self):
        raw = '[{"kind":"footage","query":"forest","use_background":false,"material_ref":null},'\
              '{"kind":"motiongraphics","query":"revenue","use_background":true,"material_ref":null}]'
        data = plan._parse(raw, 2)
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 2)

    def test_parse_wrong_length(self):
        self.assertIsNone(plan._parse('[{"kind":"footage"}]', 2))

    def test_parse_extracts_from_prose(self):
        raw = 'Sure! Here is the plan:\n[{"kind":"footage"},{"kind":"motiongraphics"}]\nDone.'
        data = plan._parse(raw, 2)
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 2)

    def test_build_plan_uses_llm(self):
        raw = '[{"kind":"footage","query":"forest","use_background":false,"material_ref":null},'\
              '{"kind":"motiongraphics","query":"revenue","use_background":true,"material_ref":7}]'
        with mock.patch.object(plan.llm, "_generate_response", return_value=raw):
            plans = plan.build_plan(self._scenes(), "economy")
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].kind, "footage")
        self.assertTrue(plans[1].is_mg)
        self.assertTrue(plans[1].use_background)
        self.assertEqual(plans[1].material_ref, 7)

    def test_build_plan_falls_back_on_error(self):
        with mock.patch.object(plan.llm, "_generate_response", side_effect=RuntimeError("boom")):
            plans = plan.build_plan(self._scenes(), "economy", video_terms=["forest sunrise", "market revenue"])
        # Heuristic: forest -> footage, "25%" -> motiongraphics
        self.assertEqual([p.kind for p in plans], ["footage", "motiongraphics"])
        self.assertNotEqual(plans[0].query, plans[1].query)

    def test_build_plan_falls_back_on_invalid_json(self):
        with mock.patch.object(plan.llm, "_generate_response", return_value="not json at all"):
            plans = plan.build_plan(self._scenes(), "economy")
        self.assertEqual(len(plans), 2)


class TestDirectorFootageRanges(unittest.TestCase):
    """Mixed mode must report the (start, end) of footage scenes only, so the
    pipeline can burn captions there and leave motion-graphics scenes clean."""

    def _run(self, kinds):
        scene_list = [
            scenes.Scene(text=f"s{i}", start=float(i * 3), duration=3.0)
            for i in range(len(kinds))
        ]
        plans = [
            plan.ScenePlan(scene=s, kind=k, query="q", use_background=(k == "motiongraphics"))
            for s, k in zip(scene_list, kinds)
        ]
        params = types.SimpleNamespace(
            video_aspect="portrait", video_subject="econ",
            video_source="pexels", n_threads=2,
        )
        with mock.patch.object(hf, "is_available", return_value=True), \
             mock.patch.object(hf.scenes, "build_scenes", return_value=scene_list), \
             mock.patch.object(hf.plan, "build_plan", return_value=plans), \
             mock.patch.object(hf.assets, "reset"), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(hf, "_render_mg_block", return_value="mg.mp4"), \
             mock.patch.object(hf, "_footage_segment", side_effect=lambda *a, **k: f"f{a[6]}.mp4"), \
             mock.patch.object(hf.assemble, "assemble", return_value="combined.mp4"):
            return hf.render_directed_video(
                "task", params, "script", "audio.wav", "sub.srt", float(len(kinds) * 3),
            )

    def test_footage_ranges_exclude_motion_graphics(self):
        out, ranges = self._run(["footage", "motiongraphics", "footage"])
        self.assertEqual(out, "combined.mp4")
        self.assertEqual(ranges, [(0.0, 3.0), (6.0, 9.0)])

    def test_all_motion_graphics_has_no_footage_ranges(self):
        out, ranges = self._run(["motiongraphics", "motiongraphics"])
        self.assertEqual(out, "combined.mp4")
        self.assertEqual(ranges, [])

    def test_unavailable_returns_empty_tuple(self):
        with mock.patch.object(hf, "is_available", return_value=False):
            out, ranges = hf.render_directed_video(
                "task", types.SimpleNamespace(video_aspect="portrait"),
                "script", "audio.wav", "sub.srt", 6.0,
            )
        self.assertEqual(out, "")
        self.assertEqual(ranges, [])


class TestAssemble(unittest.TestCase):
    def test_assemble_orders_segments_by_start(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                p = os.path.join(d, f"seg{i}.mp4")
                with open(p, "wb") as f:
                    f.write(b"x")
                paths.append(p)
            # Deliberately out of order: starts 6, 0, 3
            segments = [
                assemble.Segment(start=6.0, file_path=paths[0]),
                assemble.Segment(start=0.0, file_path=paths[1]),
                assemble.Segment(start=3.0, file_path=paths[2]),
            ]
            out = os.path.join(d, "combined-1.mp4")
            captured = {}

            def fake_concat(clip_files, output_file, threads, output_dir):
                captured["order"] = list(clip_files)
                with open(output_file, "wb") as f:
                    f.write(b"combined")

            with mock.patch.object(assemble.video, "concat_video_clips_with_ffmpeg", side_effect=fake_concat):
                result = assemble.assemble(out, segments)

            self.assertEqual(result, out)
            self.assertEqual(captured["order"], [paths[1], paths[2], paths[0]])

    def test_assemble_single_segment_copies(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "only.mp4")
            with open(src, "wb") as f:
                f.write(b"data")
            out = os.path.join(d, "combined-1.mp4")
            result = assemble.assemble(out, [assemble.Segment(start=0.0, file_path=src)])
            self.assertEqual(result, out)
            self.assertTrue(os.path.exists(out))

    def test_assemble_no_segments(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "combined-1.mp4")
            self.assertEqual(assemble.assemble(out, []), "")


class TestStudio(unittest.TestCase):
    def _scenes(self):
        return [
            scenes.Scene("Driving a 78% jump in Kazakhstan tourism receipts", 0.0, 4.0),
            scenes.Scene("A calm forest at dawn over the mountains", 4.0, 4.0),
        ]

    def test_extract_hero_prefers_percent(self):
        hero = studio._extract_hero("Driving a 78% jump in 2023 from $5 spending")
        self.assertIsNotNone(hero)
        value, number, prefix, suffix, decimals, _ = hero
        self.assertEqual(number, 78.0)
        self.assertEqual(suffix, "%")
        self.assertEqual(value, "78%")

    def test_extract_hero_ignores_bare_years(self):
        self.assertIsNone(studio._extract_hero("between 2019 and 2023"))

    def test_extract_hero_currency_and_magnitude(self):
        hero = studio._extract_hero("worth $2.4B today")
        value, number, prefix, suffix, decimals, _ = hero
        self.assertEqual(prefix, "$")
        self.assertEqual(number, 2.4)
        self.assertEqual(suffix, "B")
        self.assertEqual(decimals, 1)

    def test_build_specs_classifies(self):
        specs = studio.build_specs(self._scenes())
        self.assertEqual(specs[0].archetype, "stat")
        self.assertEqual(specs[1].archetype, "statement")

    def test_compose_passes_contract_validation(self):
        sc = self._scenes()
        total = max(s.end for s in sc)
        html = studio.compose(sc, "tourism", 1080, 1920, total)
        self.assertTrue(html)
        ok, reason = author._validate(html, len(sc), total)
        self.assertTrue(ok, reason)

    def test_compose_with_backgrounds_validates_assets(self):
        sc = self._scenes()
        total = max(s.end for s in sc)
        bgs = [Background(filename="assets/city.jpg", description="city")]
        html = studio.compose(sc, "tourism", 1080, 1920, total, backgrounds=bgs)
        ok, reason = author._validate(html, len(sc), total, asset_files=["city.jpg"])
        self.assertTrue(ok, reason)
        self.assertIn("assets/city.jpg", html)

    def test_compose_counts_up_the_stat(self):
        sc = [scenes.Scene("Revenue grew 25% last year", 0.0, 4.0)]
        html = studio.compose(sc, "money", 1080, 1920, 4.0)
        self.assertIn('id="val-0"', html)
        self.assertIn("toLocaleString", html)  # integer count-up
        self.assertIn("v:25.0", html.replace(" ", ""))

    def test_author_composition_uses_studio_by_default(self):
        # No LLM should be called when studio succeeds.
        with mock.patch.object(author.llm, "_generate_response",
                               side_effect=AssertionError("LLM must not be called")):
            html = author.author_composition(self._scenes(), "tourism", 1080, 1920)
        self.assertIn('data-composition-id="main"', html)
        self.assertIn("studio", "studio")  # sanity

    def test_author_falls_back_to_freeform_when_studio_disabled(self):
        with mock.patch.object(author, "_engine", return_value="freeform"), \
             mock.patch.object(author.llm, "_generate_response", return_value=_valid_html()):
            html = author.author_composition(
                [scenes.Scene("One", 0.0, 3.0), scenes.Scene("Two", 3.0, 3.0)],
                "money", 1080, 1920,
            )
        self.assertIn('data-composition-id="main"', html)


class TestPreview(unittest.TestCase):
    def _png(self, d, name, color):
        p = os.path.join(d, name)
        Image.new("RGB", (108, 192), color).save(p)
        return p

    def test_is_near_empty_flags_flat_dark_frame(self):
        with tempfile.TemporaryDirectory() as d:
            black = self._png(d, "black.png", (8, 10, 14))
            self.assertTrue(preview._is_near_empty(black))

    def test_is_near_empty_passes_frame_with_bright_content(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "content.png")
            im = Image.new("RGB", (108, 192), (15, 23, 42))  # dark gradient base
            # A bright caption block lifts the colour spread above the flat-dark gate.
            for y in range(80, 110):
                for x in range(10, 98):
                    im.putpixel((x, y), (255, 255, 255))
            im.save(p)
            self.assertFalse(preview._is_near_empty(p))

    def test_contact_sheet_built_from_frames(self):
        with tempfile.TemporaryDirectory() as d:
            frames = [
                self._png(d, "f0.png", (200, 30, 30)),
                self._png(d, "f1.png", (30, 200, 30)),
            ]
            out = os.path.join(d, "sheet.png")
            result = preview._build_contact_sheet(frames, out)
            self.assertEqual(result, out)
            self.assertTrue(os.path.exists(out))

    def test_contact_sheet_no_frames(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(preview._build_contact_sheet([], os.path.join(d, "x.png")), "")

    def test_preview_reports_black_scene(self):
        scene_list = [scenes.Scene("a", 0.0, 3.0), scenes.Scene("b", 3.0, 3.0)]
        with tempfile.TemporaryDirectory() as d:
            proxy = os.path.join(d, "preview-proxy.mp4")

            def fake_render(html, out_path, fps=None):
                with open(out_path, "wb") as f:
                    f.write(b"proxy")
                return out_path

            # scene 0 -> a real frame; scene 1 -> a flat black frame (a gap).
            def fake_extract(p, t, out_path):
                color = (12, 12, 12) if t > 3 else (240, 240, 240)
                Image.new("RGB", (108, 192), color).save(out_path)
                return out_path

            with mock.patch.object(preview.render, "render", side_effect=fake_render), \
                 mock.patch.object(preview, "_extract_frame", side_effect=fake_extract):
                report = preview.preview("<html></html>", scene_list, d)

            self.assertFalse(report.ok)
            self.assertEqual([i for i, _ in report.issues], [1])
            self.assertTrue(os.path.exists(report.contact_sheet))

    def test_preview_non_fatal_when_proxy_fails(self):
        scene_list = [scenes.Scene("a", 0.0, 3.0)]
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(preview.render, "render", return_value=""):
                report = preview.preview("<html></html>", scene_list, d)
            self.assertTrue(report.ok)
            self.assertIn("proxy", report.note)


class TestNewsStudio(unittest.TestCase):
    def _scenes(self):
        return [
            scenes.Scene("Markets opened sharply higher this morning", 0.0, 4.0),
            scenes.Scene("Tech stocks gained 12% over the quarter", 4.0, 4.0),
        ]

    def test_kicker_trims_word_safe_and_uppercases(self):
        self.assertEqual(studio._kicker("hello brave new world"), "HELLO BRAVE NEW WORLD")
        long = "an extremely long opening sentence that keeps going well past the plate"
        kicked = studio._kicker(long)
        self.assertLessEqual(len(kicked), 38)
        self.assertEqual(kicked, kicked.upper())
        self.assertFalse(kicked.endswith(" "))
        # A single token longer than the plate is hard-sliced, not returned whole.
        self.assertEqual(len(studio._kicker("x" * 80)), 38)

    def test_compose_news_passes_contract_validation(self):
        html = studio.compose_news(self._scenes(), "AI takeover", 1080, 1920, 8.0)
        self.assertTrue(html)
        ok, reason = author._validate(html, scene_count=2, total=8.0)
        self.assertTrue(ok, reason)
        # Persistent headline plate with the subject, one lower-third per scene.
        self.assertIn("AI TAKEOVER", html)
        self.assertIn('id="headline"', html)
        self.assertIn('id="lt-0"', html)
        self.assertIn('id="lt-1"', html)

    def test_compose_news_with_backgrounds_validates_assets(self):
        bgs = [Background(filename="assets/news1.jpg", description="city")]
        html = studio.compose_news(self._scenes(), "econ", 1080, 1920, 8.0, backgrounds=bgs)
        ok, reason = author._validate(html, 2, 8.0, ["news1.jpg"])
        self.assertTrue(ok, reason)
        self.assertIn('src="assets/news1.jpg"', html)

    def test_compose_news_empty_inputs(self):
        self.assertEqual(studio.compose_news([], "x", 1080, 1920, 8.0), "")
        self.assertEqual(studio.compose_news(self._scenes(), "x", 1080, 1920, 0), "")

    def test_author_news_rejects_invalid_and_does_not_fall_back(self):
        with mock.patch.object(author.studio, "compose_news", return_value="<html>broken</html>"), \
             mock.patch.object(author.llm, "_generate_response") as llm_call:
            self.assertEqual(author.author_news(self._scenes(), "x", 1080, 1920), "")
        llm_call.assert_not_called()

    def test_author_news_accepts_studio_output(self):
        html = author.author_news(self._scenes(), "econ", 1080, 1920)
        self.assertIn('data-composition-id="main"', html)


class TestOverlayPresenter(unittest.TestCase):
    def _files(self, d, presenter_name="presenter.mp4"):
        base = os.path.join(d, "base.mp4")
        pres = os.path.join(d, presenter_name)
        for p in (base, pres):
            with open(p, "wb") as f:
                f.write(b"x")
        return base, pres

    def test_missing_inputs_return_empty(self):
        with tempfile.TemporaryDirectory() as d:
            base, _ = self._files(d)
            out = os.path.join(d, "out.mp4")
            self.assertEqual(assemble.overlay_presenter("", base, out, 1080, 1920), "")
            self.assertEqual(assemble.overlay_presenter(base, os.path.join(d, "nope.mp4"), out, 1080, 1920), "")

    def test_overlay_runs_ffmpeg_and_returns_output(self):
        with tempfile.TemporaryDirectory() as d:
            base, pres = self._files(d)
            out = os.path.join(d, "out.mp4")
            captured = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                with open(out, "wb") as f:
                    f.write(b"video")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(assemble.subprocess, "run", side_effect=fake_run):
                result = assemble.overlay_presenter(base, pres, out, 1080, 1920)

            self.assertEqual(result, out)
            cmd = " ".join(captured["cmd"])
            self.assertIn("overlay=x=main_w-overlay_w-", cmd)   # default bottom-right
            self.assertIn("eof_action=pass", cmd)
            self.assertIn("scale=410:-2", cmd)                  # 1080 * 0.38 -> even 410
            self.assertNotIn("libvpx-vp9", cmd)                 # plain mp4 input

    def test_webm_presenter_uses_alpha_decoder(self):
        with tempfile.TemporaryDirectory() as d:
            base, pres = self._files(d, presenter_name="presenter.webm")
            out = os.path.join(d, "out.mp4")
            captured = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                with open(out, "wb") as f:
                    f.write(b"video")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(assemble.subprocess, "run", side_effect=fake_run):
                self.assertEqual(assemble.overlay_presenter(base, pres, out, 1080, 1920), out)
            cmd = captured["cmd"]
            # The decoder override must come before the presenter input.
            self.assertLess(cmd.index("libvpx-vp9"), cmd.index(pres))

    def test_ffmpeg_failure_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            base, pres = self._files(d)
            out = os.path.join(d, "out.mp4")
            failed = types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
            with mock.patch.object(assemble.subprocess, "run", return_value=failed):
                self.assertEqual(assemble.overlay_presenter(base, pres, out, 1080, 1920), "")
            self.assertFalse(os.path.exists(out))


class TestNewsMode(unittest.TestCase):
    """News mode = deterministic news track + optional presenter overlay.
    Caption ranges: [] with a head (lower-thirds carry the words), the whole
    video without one."""

    def test_mode_recognizes_news(self):
        params = types.SimpleNamespace(video_visual_mode="news")
        self.assertEqual(hf.mode(params), "news")
        with mock.patch.dict(hf.config.app, {"video_visual_mode": "news"}):
            self.assertEqual(hf.mode(types.SimpleNamespace(video_visual_mode="")), "news")

    def _params(self):
        return types.SimpleNamespace(
            video_aspect="portrait", video_subject="econ news",
            video_source="pexels", n_threads=2, video_visual_mode="news",
        )

    def _run(self, presenter="", overlay=""):
        scene_list = [scenes.Scene("s0", 0.0, 4.0), scenes.Scene("s1", 4.0, 4.0)]
        with mock.patch.object(hf, "is_available", return_value=True), \
             mock.patch.object(hf.scenes, "build_scenes", return_value=scene_list), \
             mock.patch.object(hf.assets, "reset"), \
             mock.patch.object(hf, "_solely_backgrounds", return_value=[]), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(hf.author, "author_news", return_value="<html>ok</html>"), \
             mock.patch.object(hf.render, "render", return_value="news-base.mp4"), \
             mock.patch.object(hf, "_news_presenter", return_value=presenter), \
             mock.patch.object(hf.assemble, "overlay_presenter", return_value=overlay) as ov:
            result = hf.render_news_video("task", self._params(), "script", "audio.mp3", "sub.srt", 8.0)
        return result, ov

    def test_presenter_success_suppresses_captions(self):
        (out, ranges), ov = self._run(presenter="presenter.mp4", overlay="news.mp4")
        self.assertEqual(out, "news.mp4")
        self.assertEqual(ranges, [])
        ov.assert_called_once()

    def test_no_presenter_captions_whole_video(self):
        (out, ranges), ov = self._run(presenter="")
        self.assertEqual(out, "news-base.mp4")
        self.assertEqual(ranges, [(0.0, 8.0)])
        ov.assert_not_called()

    def test_overlay_failure_degrades_to_headless_base(self):
        (out, ranges), _ = self._run(presenter="presenter.mp4", overlay="")
        self.assertEqual(out, "news-base.mp4")
        self.assertEqual(ranges, [(0.0, 8.0)])

    def test_unavailable_toolchain(self):
        with mock.patch.object(hf, "is_available", return_value=False):
            out, ranges = hf.render_news_video("task", self._params(), "s", "a.mp3", "s.srt", 8.0)
        self.assertEqual((out, ranges), ("", []))

    def test_composition_failure_falls_back(self):
        scene_list = [scenes.Scene("s0", 0.0, 4.0)]
        with mock.patch.object(hf, "is_available", return_value=True), \
             mock.patch.object(hf.scenes, "build_scenes", return_value=scene_list), \
             mock.patch.object(hf.assets, "reset"), \
             mock.patch.object(hf, "_solely_backgrounds", return_value=[]), \
             mock.patch.object(hf.author, "author_news", return_value=""):
            out, ranges = hf.render_news_video("task", self._params(), "s", "a.mp3", "s.srt", 4.0)
        self.assertEqual((out, ranges), ("", []))


class TestNewsPresenter(unittest.TestCase):
    def _params(self):
        return types.SimpleNamespace(video_aspect="portrait")

    def test_disabled_avatar_returns_empty(self):
        from app.services import avatar
        with mock.patch.object(avatar, "is_enabled", return_value=False):
            self.assertEqual(hf._news_presenter("task", self._params(), "script", "a.mp3", 1080, 1920), "")

    def test_requests_square_clip_and_returns_result(self):
        from app.services import avatar
        with mock.patch.object(avatar, "is_enabled", return_value=True), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(avatar, "synthesize", return_value="p.mp4") as syn:
            result = hf._news_presenter("task", self._params(), "script", "a.mp3", 1080, 1920)
        self.assertEqual(result, "p.mp4")
        kwargs = syn.call_args.kwargs
        self.assertEqual(kwargs["width"], 1080)
        self.assertEqual(kwargs["height"], 1080)

    def test_follows_pipeline_voice_when_avatar_voice_unset(self):
        from app.services import avatar
        params = types.SimpleNamespace(
            video_aspect="portrait", voice_name="ru-RU-DmitryNeural-V2-Male"
        )
        with mock.patch.dict(hf.config.app, {"avatar_voice": ""}), \
             mock.patch.object(avatar, "is_enabled", return_value=True), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(avatar, "synthesize", return_value="p.mp4") as syn:
            hf._news_presenter("task", params, "script", "a.mp3", 1080, 1920)
        self.assertEqual(syn.call_args.kwargs["presenter"], "ru-RU-DmitryNeural")

    def test_explicit_avatar_voice_wins_over_pipeline_voice(self):
        from app.services import avatar
        params = types.SimpleNamespace(
            video_aspect="portrait", voice_name="ru-RU-DmitryNeural-V2-Male"
        )
        with mock.patch.dict(hf.config.app, {"avatar_voice": "en-US-JennyNeural"}), \
             mock.patch.object(avatar, "is_enabled", return_value=True), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(avatar, "synthesize", return_value="p.mp4") as syn:
            hf._news_presenter("task", params, "script", "a.mp3", 1080, 1920)
        self.assertEqual(syn.call_args.kwargs["presenter"], "en-US-JennyNeural")

    def test_non_azure_pipeline_voice_leaves_provider_default(self):
        from app.services import avatar
        params = types.SimpleNamespace(
            video_aspect="portrait", voice_name="qwen:Russian:ryan"
        )
        with mock.patch.dict(hf.config.app, {"avatar_voice": ""}), \
             mock.patch.object(avatar, "is_enabled", return_value=True), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(avatar, "synthesize", return_value="p.mp4") as syn:
            hf._news_presenter("task", params, "script", "a.mp3", 1080, 1920)
        self.assertEqual(syn.call_args.kwargs["presenter"], "")

    def test_falls_back_to_wav2lip_with_audio(self):
        from app.services import avatar
        with tempfile.TemporaryDirectory() as d:
            audio = os.path.join(d, "audio.mp3")
            with open(audio, "wb") as f:
                f.write(b"a")
            w2l = mock.Mock()
            w2l.synthesize.return_value = "w2l.mp4"
            with mock.patch.object(avatar, "is_enabled", return_value=True), \
                 mock.patch.object(hf.utils, "task_dir", return_value=d), \
                 mock.patch.object(avatar, "synthesize", return_value=""), \
                 mock.patch.object(avatar, "Wav2LipAvatar", return_value=w2l):
                result = hf._news_presenter("task", self._params(), "script", audio, 1080, 1920)
        self.assertEqual(result, "w2l.mp4")
        self.assertEqual(w2l.synthesize.call_args.kwargs["script_or_audio"], audio)


class TestStudioArchetypes(unittest.TestCase):
    def _spec(self, text):
        return studio.build_specs([scenes.Scene(text, 0.0, 4.0)])[0]

    def test_quote_with_attribution(self):
        spec = self._spec("«Мы изменим всю индустрию» — Илон Маск")
        self.assertEqual(spec.archetype, "quote")
        self.assertEqual(spec.caption, "Мы изменим всю индустрию")
        self.assertEqual(spec.attr, "Илон Маск")

    def test_quote_without_attribution(self):
        spec = self._spec('"The best way out is always through."')
        self.assertEqual(spec.archetype, "quote")
        self.assertEqual(spec.attr, "")

    def test_comparison_vs(self):
        spec = self._spec("iPhone 16 vs Galaxy S25")
        self.assertEqual(spec.archetype, "comparison")
        self.assertEqual(spec.items, ["iPhone 16", "Galaxy S25"])

    def test_comparison_russian(self):
        spec = self._spec("Электромобили против бензиновых машин")
        self.assertEqual(spec.archetype, "comparison")
        self.assertEqual(spec.items[0], "Электромобили")

    def test_comparison_rejects_long_halves(self):
        long_half = "a very long meandering clause that keeps going on and on endlessly"
        spec = self._spec(f"{long_half} vs short")
        self.assertNotEqual(spec.archetype, "comparison")

    def test_list_numbered(self):
        spec = self._spec("Три причины: 1. Скорость 2. Цена 3. Дизайн")
        self.assertEqual(spec.archetype, "list")
        self.assertEqual(spec.caption, "Три причины")
        self.assertEqual(spec.items, ["Скорость", "Цена", "Дизайн"])

    def test_list_semicolons(self):
        spec = self._spec("Speed; price; design; support")
        self.assertEqual(spec.archetype, "list")
        self.assertEqual(len(spec.items), 4)

    def test_list_not_triggered_by_single_marker(self):
        spec = self._spec("Step 1. Open the app and look around")
        self.assertNotEqual(spec.archetype, "list")

    def test_chart_from_two_percents(self):
        spec = self._spec("Онлайн вырос до 78%, офлайн упал до 22%")
        self.assertEqual(spec.archetype, "chart")
        self.assertEqual([v for _, v in spec.bars], [78.0, 22.0])

    def test_single_percent_stays_stat(self):
        spec = self._spec("Revenue grew 25% last year")
        self.assertEqual(spec.archetype, "stat")

    def test_plain_text_stays_statement(self):
        spec = self._spec("A calm forest at dawn over the mountains")
        self.assertEqual(spec.archetype, "statement")

    def test_compose_all_archetypes_passes_contract(self):
        sc = [
            scenes.Scene("«Мы изменим индустрию» — Маск", 0.0, 3.0),
            scenes.Scene("iPhone vs Galaxy", 3.0, 3.0),
            scenes.Scene("Итоги: 1. Рост 2. Прибыль 3. Экспансия", 6.0, 3.0),
            scenes.Scene("Онлайн 78%, офлайн 22%", 9.0, 3.0),
            scenes.Scene("Будущее уже здесь", 12.0, 3.0),
        ]
        total = max(s.end for s in sc)
        html = studio.compose(sc, "тренды", 1080, 1920, total)
        self.assertTrue(html)
        ok, reason = author._validate(html, len(sc), total)
        self.assertTrue(ok, reason)
        for marker in ("quote-text", "vs-badge", "list-idx", "bar-fill", "headline"):
            self.assertIn(marker, html)

    def test_compose_embeds_font_family(self):
        html = studio.compose([scenes.Scene("Просто текст", 0.0, 4.0)],
                              "тема", 1080, 1920, 4.0)
        self.assertIn("'Anton','Oswald'", html)


class TestFonts(unittest.TestCase):
    def test_stage_fonts_copies_and_returns_css(self):
        from app.services.hyperframes import fonts
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(fonts.render, "assets_dir", return_value=d):
            css = fonts.stage_fonts()
            self.assertIn("font-family:'Anton'", css)
            self.assertIn("font-family:'Oswald'", css)
            self.assertTrue(os.path.isfile(os.path.join(d, "Anton-Regular.ttf")))
            self.assertTrue(os.path.isfile(os.path.join(d, "Oswald-Variable.ttf")))

    def test_stage_fonts_missing_bundle_is_nonfatal(self):
        from app.services.hyperframes import fonts
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(fonts.render, "assets_dir", return_value=d), \
             mock.patch.object(fonts, "_FONTS_DIR", os.path.join(d, "nope")):
            self.assertEqual(fonts.stage_fonts(), "")


class TestPreviewRetry(unittest.TestCase):
    def _report(self, issues):
        return preview.PreviewReport(ok=not issues, issues=issues)

    def test_disabled_returns_original(self):
        with mock.patch.object(hf.preview, "is_enabled", return_value=False), \
             mock.patch.object(hf.preview, "preview") as pv:
            out = hf._preview_retry("t", "<html>", [], lambda: "<other>")
        self.assertEqual(out, "<html>")
        pv.assert_not_called()

    def test_clean_preview_keeps_original(self):
        with mock.patch.object(hf.preview, "is_enabled", return_value=True), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(hf.preview, "preview", return_value=self._report([])):
            out = hf._preview_retry("t", "<html>", [], lambda: "<other>")
        self.assertEqual(out, "<html>")

    def test_retry_used_when_it_previews_cleaner(self):
        reports = [self._report([(1, "near-empty/black frame")]), self._report([])]
        with mock.patch.object(hf.preview, "is_enabled", return_value=True), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(hf.preview, "preview", side_effect=reports):
            out = hf._preview_retry("t", "<html>", [], lambda: "<better>")
        self.assertEqual(out, "<better>")

    def test_retry_discarded_when_not_better(self):
        issue = [(1, "near-empty/black frame")]
        reports = [self._report(issue), self._report(issue)]
        with mock.patch.object(hf.preview, "is_enabled", return_value=True), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(hf.preview, "preview", side_effect=reports):
            out = hf._preview_retry("t", "<html>", [], lambda: "<retry>")
        self.assertEqual(out, "<html>")

    def test_recompose_failure_keeps_original(self):
        def boom():
            raise RuntimeError("llm down")

        with mock.patch.object(hf.preview, "is_enabled", return_value=True), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(
                 hf.preview, "preview",
                 return_value=self._report([(0, "near-empty/black frame")])):
            out = hf._preview_retry("t", "<html>", [], boom)
        self.assertEqual(out, "<html>")


if __name__ == "__main__":
    unittest.main()
