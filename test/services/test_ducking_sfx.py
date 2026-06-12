"""
Tests for WP4: BGM ducking + transition SFX helpers in app/services/video.py.

These exercise the pure helper functions directly (speech-span merging, the
time-varying gain function, cut-point derivation, and the SFX overlay's
config gating). No real moviepy rendering and no task.start() — fast/headless.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import video as vd


class TestMergeSpeechSpans(unittest.TestCase):
    def test_merges_intervals_within_gap(self):
        # gap 0.2s < 0.7 default -> merged into one span
        spans = vd.merge_speech_spans([(0.0, 1.0), (1.2, 2.0)])
        self.assertEqual(spans, [(0.0, 2.0)])

    def test_keeps_intervals_separated_by_large_gap(self):
        spans = vd.merge_speech_spans([(0.0, 1.0), (5.0, 6.0)])
        self.assertEqual(spans, [(0.0, 1.0), (5.0, 6.0)])

    def test_sorts_unordered_input(self):
        spans = vd.merge_speech_spans([(5.0, 6.0), (0.0, 1.0)])
        self.assertEqual(spans, [(0.0, 1.0), (5.0, 6.0)])

    def test_overlapping_intervals_merge(self):
        spans = vd.merge_speech_spans([(0.0, 2.0), (1.0, 3.0)])
        self.assertEqual(spans, [(0.0, 3.0)])

    def test_empty_input_returns_empty(self):
        self.assertEqual(vd.merge_speech_spans([]), [])

    def test_discards_zero_and_negative_length(self):
        spans = vd.merge_speech_spans([(1.0, 1.0), (2.0, 1.5), (3.0, 4.0)])
        self.assertEqual(spans, [(3.0, 4.0)])

    def test_custom_gap_threshold(self):
        # gap of 0.5 between intervals; with gap=0.3 they stay separate
        spans = vd.merge_speech_spans([(0.0, 1.0), (1.5, 2.0)], gap=0.3)
        self.assertEqual(spans, [(0.0, 1.0), (1.5, 2.0)])


class TestDuckGainFn(unittest.TestCase):
    def setUp(self):
        self.spans = [(2.0, 5.0)]
        self.base = 0.5
        self.duck = 0.1
        self.g = vd.make_duck_gain_fn(self.spans, self.base, self.duck)

    def test_outside_speech_is_base_volume(self):
        self.assertAlmostEqual(self.g(0.0), self.base)
        self.assertAlmostEqual(self.g(10.0), self.base)

    def test_inside_speech_is_duck_volume(self):
        self.assertAlmostEqual(self.g(3.5), self.duck)

    def test_edge_midpoint_is_halfway(self):
        # falling edge centred on 5.0 -> exactly between base and duck
        self.assertAlmostEqual(self.g(5.0), (self.base + self.duck) / 2.0)
        # rising edge centred on 2.0
        self.assertAlmostEqual(self.g(2.0), (self.base + self.duck) / 2.0)

    def test_accepts_numpy_array(self):
        t = np.array([0.0, 2.0, 3.5, 5.0, 10.0])
        out = self.g(t)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(
            out,
            [self.base, (self.base + self.duck) / 2, self.duck,
             (self.base + self.duck) / 2, self.base],
        )

    def test_gain_stays_within_bounds(self):
        t = np.linspace(0, 8, 200)
        out = self.g(t)
        self.assertTrue(np.all(out <= max(self.base, self.duck) + 1e-9))
        self.assertTrue(np.all(out >= min(self.base, self.duck) - 1e-9))

    def test_empty_spans_is_flat_base(self):
        g = vd.make_duck_gain_fn([], self.base, self.duck)
        np.testing.assert_allclose(g(np.array([0.0, 5.0, 50.0])), self.base)


class TestBuildSpeechSpans(unittest.TestCase):
    def test_prefers_passed_in_words(self):
        words = [
            {"text": "a", "start": 0.0, "end": 0.5},
            {"text": "b", "start": 0.6, "end": 1.0},
            {"text": "c", "start": 5.0, "end": 5.5},
        ]
        spans = vd.build_speech_spans("", karaoke_words=words)
        # first two merge (gap 0.1), third separate
        self.assertEqual(spans, [(0.0, 1.0), (5.0, 5.5)])

    def test_falls_back_to_srt(self):
        spans = vd.build_speech_spans("/no/such/file.srt", karaoke_words=[])
        self.assertEqual(spans, [])

    def test_srt_phrase_ranges(self):
        import tempfile

        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
            "2\n00:00:02,200 --> 00:00:03,000\nWorld\n\n"
            "3\n00:00:10,000 --> 00:00:11,000\nLater\n\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub.srt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(srt)
            spans = vd.build_speech_spans(path, karaoke_words=[])
        # 1&2 merge (gap 0.2 < 0.7); 3 separate
        self.assertEqual(spans, [(1.0, 3.0), (10.0, 11.0)])

    def test_no_data_returns_empty(self):
        self.assertEqual(vd.build_speech_spans("", karaoke_words=[]), [])


class TestSrtTimestamp(unittest.TestCase):
    def test_parses_comma_millis(self):
        self.assertAlmostEqual(
            vd._srt_timestamp_to_seconds("01:02:03,500"), 3723.5
        )

    def test_parses_dot_millis(self):
        self.assertAlmostEqual(
            vd._srt_timestamp_to_seconds("00:00:01.250"), 1.25
        )


class TestComputeSfxCutTimes(unittest.TestCase):
    def test_skips_t_zero(self):
        cuts = vd.compute_sfx_cut_times([(0.0, 1.0), (3.0, 4.0)])
        self.assertEqual(cuts, [3.0])

    def test_one_cut_per_span_start(self):
        cuts = vd.compute_sfx_cut_times([(2.0, 3.0), (5.0, 6.0), (8.0, 9.0)])
        self.assertEqual(cuts, [2.0, 5.0, 8.0])

    def test_dedupes_close_cuts(self):
        cuts = vd.compute_sfx_cut_times([(2.0, 3.0), (2.5, 4.0)], min_gap=0.8)
        self.assertEqual(cuts, [2.0])

    def test_drops_cuts_past_duration(self):
        cuts = vd.compute_sfx_cut_times([(2.0, 3.0), (9.0, 10.0)], total_duration=8.0)
        self.assertEqual(cuts, [2.0])

    def test_empty_spans(self):
        self.assertEqual(vd.compute_sfx_cut_times([]), [])


class TestApplyBgmDucking(unittest.TestCase):
    def test_empty_spans_returns_input_unchanged(self):
        clip = MagicMock(name="bgm")
        out = vd.apply_bgm_ducking(clip, [], 0.5, 0.1)
        self.assertIs(out, clip)

    def test_transform_scales_frames_by_gain(self):
        # Build a real moviepy AudioClip so .transform actually runs, then
        # sample frames inside vs outside a speech span.
        from moviepy import AudioClip

        # constant stereo amplitude of 1.0
        def frame(t):
            t = np.asarray(t, dtype=float)
            ones = np.ones_like(t)
            return np.array([ones, ones]).T

        base_clip = AudioClip(frame_function=frame, duration=8.0, fps=100)
        spans = [(2.0, 5.0)]
        ducked = vd.apply_bgm_ducking(base_clip, spans, 0.5, 0.1)

        # outside speech -> ~base 0.5; inside speech -> ~duck 0.1
        outside = ducked.get_frame(np.array([0.5]))
        inside = ducked.get_frame(np.array([3.5]))
        self.assertAlmostEqual(float(outside[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(inside[0, 0]), 0.1, places=5)


class TestOverlayTransitionSfx(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_disabled_returns_clip_unchanged(self):
        config.app["sfx_enabled"] = False
        clip = MagicMock(name="audio_clip")
        spans = [(2.0, 3.0)]
        out = vd._overlay_transition_sfx(clip, spans, 10.0)
        self.assertIs(out, clip)

    def test_no_cut_points_returns_unchanged(self):
        config.app["sfx_enabled"] = True
        clip = MagicMock(name="audio_clip")
        # only a span starting at 0 -> no cuts
        out = vd._overlay_transition_sfx(clip, [(0.0, 1.0)], 10.0)
        self.assertIs(out, clip)

    def test_missing_sfx_file_returns_unchanged(self):
        config.app["sfx_enabled"] = True
        clip = MagicMock(name="audio_clip")
        with patch("app.services.sfx.get_sfx_file", return_value=""):
            out = vd._overlay_transition_sfx(clip, [(2.0, 3.0)], 10.0)
        self.assertIs(out, clip)

    def test_overlays_sfx_at_cut_points(self):
        config.app["sfx_enabled"] = True
        config.app["sfx_volume"] = 0.6
        clip = MagicMock(name="audio_clip")
        fake_sfx = MagicMock(name="sfx_clip")
        # chainable with_effects().with_start()
        fake_sfx.with_effects.return_value = fake_sfx
        fake_sfx.with_start.return_value = fake_sfx

        with patch("app.services.sfx.get_sfx_file", return_value="/fake/whoosh.mp3"), \
             patch.object(vd, "AudioFileClip", return_value=fake_sfx), \
             patch.object(vd, "CompositeAudioClip") as comp:
            comp.side_effect = lambda layers: ("composite", layers)
            spans = [(2.0, 3.0), (5.0, 6.0)]
            out = vd._overlay_transition_sfx(clip, spans, 10.0)

        # composite was built with the original clip + 2 sfx layers
        self.assertEqual(out[0], "composite")
        self.assertEqual(len(out[1]), 3)
        self.assertIs(out[1][0], clip)


if __name__ == "__main__":
    unittest.main()
