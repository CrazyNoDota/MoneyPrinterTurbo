"""Tests for WP1 karaoke word-level captions.

Covers:
- word grouping/chunking logic (group_words_into_chunks)
- sidecar write+load round-trip (subtitle.write_words_sidecar / load_words_sidecar)
- sub_maker -> sidecar extraction (voice._extract_word_timings_from_submaker)
- fallback when sidecar missing/corrupt
- karaoke clip-timeline generation (TextClip mocked for headless speed)
"""
import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

from app.services import subtitle as subtitle_service
from app.services import video as vd
from app.services import voice as voice_service


class GroupWordsTest(unittest.TestCase):
    def _w(self, text, start, end):
        return {"text": text, "start": start, "end": end}

    def test_caps_chunk_at_max_words(self):
        words = [self._w(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(10)]
        chunks = vd.group_words_into_chunks(words, max_words=4)
        self.assertTrue(all(len(c["words"]) <= 4 for c in chunks))
        # 10 words, max 4 -> 4 + 4 + 2
        self.assertEqual([len(c["words"]) for c in chunks], [4, 4, 2])

    def test_breaks_on_long_pause(self):
        words = [
            self._w("a", 0.0, 0.2),
            self._w("b", 0.3, 0.5),
            # >0.5s gap here
            self._w("c", 1.4, 1.6),
        ]
        chunks = vd.group_words_into_chunks(words, max_words=4, pause_threshold=0.5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([w["text"] for w in chunks[0]["words"]], ["a", "b"])
        self.assertEqual([w["text"] for w in chunks[1]["words"]], ["c"])

    def test_breaks_on_punctuation(self):
        words = [
            self._w("Hello", 0.0, 0.3),
            self._w("world.", 0.3, 0.6),
            self._w("Next", 0.7, 0.9),
        ]
        chunks = vd.group_words_into_chunks(words, max_words=4)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["end"], 0.6)
        self.assertEqual(chunks[1]["words"][0]["text"], "Next")

    def test_chunk_start_end_span_words(self):
        words = [self._w("a", 0.5, 0.8), self._w("b", 0.8, 1.2)]
        chunks = vd.group_words_into_chunks(words)
        self.assertEqual(chunks[0]["start"], 0.5)
        self.assertEqual(chunks[0]["end"], 1.2)

    def test_empty_and_blank_words(self):
        self.assertEqual(vd.group_words_into_chunks([]), [])
        chunks = vd.group_words_into_chunks(
            [self._w("", 0, 0.2), self._w("x", 0.2, 0.4)]
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["words"][0]["text"], "x")


class SidecarRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.srt = os.path.join(self.tmp, "subtitle.srt")

    def test_sidecar_path(self):
        self.assertEqual(
            subtitle_service.words_sidecar_path(self.srt),
            os.path.join(self.tmp, "subtitle.words.json"),
        )
        # Non-.srt basename still appends suffix.
        other = os.path.join(self.tmp, "audio.mp3")
        self.assertEqual(
            subtitle_service.words_sidecar_path(other),
            os.path.join(self.tmp, "audio.mp3.words.json"),
        )

    def test_write_then_load(self):
        words = [
            {"text": "hello", "start": 0.0, "end": 0.4},
            {"text": "world", "start": 0.4, "end": 0.9},
        ]
        path = subtitle_service.write_words_sidecar(self.srt, words)
        self.assertTrue(os.path.isfile(path))
        loaded = subtitle_service.load_words_sidecar(self.srt)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["text"], "hello")
        self.assertAlmostEqual(loaded[1]["end"], 0.9)

    def test_write_skips_blank_and_clamps(self):
        words = [
            {"text": "  ", "start": 0.0, "end": 0.2},
            {"text": "ok", "start": 1.0, "end": 0.5},  # end < start -> clamped
        ]
        subtitle_service.write_words_sidecar(self.srt, words)
        loaded = subtitle_service.load_words_sidecar(self.srt)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["text"], "ok")
        self.assertEqual(loaded[0]["end"], loaded[0]["start"])

    def test_write_empty_returns_empty(self):
        self.assertEqual(subtitle_service.write_words_sidecar(self.srt, []), "")
        self.assertFalse(
            os.path.isfile(subtitle_service.words_sidecar_path(self.srt))
        )

    def test_load_missing_returns_empty(self):
        self.assertEqual(subtitle_service.load_words_sidecar(self.srt), [])

    def test_load_corrupt_returns_empty(self):
        sidecar = subtitle_service.words_sidecar_path(self.srt)
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertEqual(subtitle_service.load_words_sidecar(self.srt), [])

    def test_load_wrong_shape_returns_empty(self):
        sidecar = subtitle_service.words_sidecar_path(self.srt)
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
        self.assertEqual(subtitle_service.load_words_sidecar(self.srt), [])


class SubMakerExtractionTest(unittest.TestCase):
    def test_extract_from_legacy_offsets(self):
        sub_maker = types.SimpleNamespace(
            subs=["hello", "world"],
            offset=[(0, 4000000), (4000000, 9000000)],  # 100-ns units
        )
        words = voice_service._extract_word_timings_from_submaker(sub_maker)
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0]["text"], "hello")
        self.assertAlmostEqual(words[0]["end"], 0.4)
        self.assertAlmostEqual(words[1]["start"], 0.4)
        self.assertAlmostEqual(words[1]["end"], 0.9)

    def test_extract_from_cues(self):
        cue = types.SimpleNamespace(
            content="hi",
            start=types.SimpleNamespace(total_seconds=lambda: 0.1),
            end=types.SimpleNamespace(total_seconds=lambda: 0.5),
        )
        sub_maker = types.SimpleNamespace(cues=[cue])
        words = voice_service._extract_word_timings_from_submaker(sub_maker)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["text"], "hi")
        self.assertAlmostEqual(words[0]["start"], 0.1)

    def test_extract_empty(self):
        sub_maker = types.SimpleNamespace(subs=[], offset=[])
        self.assertEqual(
            voice_service._extract_word_timings_from_submaker(sub_maker), []
        )


class _FakeTextClip:
    """Lightweight stand-in for moviepy TextClip used in karaoke rendering."""

    def __init__(self, *args, **kwargs):
        self.text = kwargs.get("text", "")
        self.font_size = kwargs.get("font_size", 60)
        # Width proportional to glyph count so layout math has something real.
        self.w = max(1, len(self.text) * (self.font_size // 2))
        self.h = self.font_size
        self.position = None
        self.start = None
        self.end = None
        self.color = kwargs.get("color")

    def resized(self, factor):
        clone = _FakeTextClip(text=self.text, font_size=self.font_size, color=self.color)
        clone.w = int(self.w * factor)
        clone.h = int(self.h * factor)
        return clone

    def with_position(self, pos):
        self.position = pos
        return self

    def with_start(self, t):
        self.start = t
        return self

    def with_end(self, t):
        self.end = t
        return self

    def with_duration(self, d):
        self.duration = d
        return self

    def with_opacity(self, o):
        self.opacity = o
        return self

    def close(self):
        pass


def _make_params(highlight="#FFE600", style="karaoke"):
    from app.models.schema import VideoParams

    p = VideoParams(video_subject="test")
    p.subtitle_style = style
    p.subtitle_highlight_color = highlight
    p.font_size = 60
    p.subtitle_position = "bottom"
    p.stroke_width = 1.5
    return p


class KaraokeTimelineTest(unittest.TestCase):
    """Exercise generate_video's karaoke branch with everything heavy mocked."""

    def _run_karaoke(self, words, subtitle_ranges=None):
        captured = {}

        def fake_composite(clips):
            # clips[0] is the (mock) base video; the rest are caption clips.
            captured["text_clips"] = clips[1:]
            return clips[0]

        class _FakeVideo:
            duration = 10.0

            def with_audio(self, *a, **k):
                return self

            def write_videofile(self, *a, **k):
                pass

            def close(self):
                pass

        tmp = tempfile.mkdtemp()
        srt = os.path.join(tmp, "subtitle.srt")
        with open(srt, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\nhello world\n\n")
        subtitle_service.write_words_sidecar(srt, words)
        out = os.path.join(tmp, "out.mp4")

        params = _make_params()

        with patch.object(vd, "TextClip", _FakeTextClip), patch.object(
            vd, "_open_video_clip_quietly", return_value=_FakeVideo()
        ), patch.object(vd, "AudioFileClip") as fake_audio, patch.object(
            vd, "CompositeVideoClip", side_effect=fake_composite
        ), patch.object(
            vd, "get_bgm_file", return_value=""
        ):
            fake_audio.return_value.with_effects.return_value = fake_audio.return_value
            fake_audio.return_value.fps = 44100
            vd.generate_video(
                video_path="video.mp4",
                audio_path="audio.mp3",
                subtitle_path=srt,
                output_file=out,
                params=params,
                subtitle_ranges=subtitle_ranges,
            )
        return captured.get("text_clips", [])

    def test_karaoke_emits_one_clip_per_word_per_interval(self):
        words = [
            {"text": "hello", "start": 0.0, "end": 0.4},
            {"text": "world", "start": 0.4, "end": 0.9},
        ]
        clips = self._run_karaoke(words)
        # 1 chunk of 2 words -> 2 intervals x 2 words = 4 clips.
        self.assertEqual(len(clips), 4)
        # Each clip carries a start/end time.
        self.assertTrue(all(c.start is not None and c.end is not None for c in clips))
        # Some clip uses the highlight color.
        self.assertTrue(any(c.color == "#FFE600" for c in clips))

    def test_karaoke_respects_subtitle_ranges(self):
        words = [
            {"text": "hello", "start": 0.0, "end": 0.4},
            {"text": "world", "start": 0.4, "end": 0.9},
        ]
        # Range excludes the chunk midpoint (~0.45) -> no clips.
        clips = self._run_karaoke(words, subtitle_ranges=[(5.0, 9.0)])
        self.assertEqual(len(clips), 0)

    def test_karaoke_fallback_when_sidecar_missing(self):
        # No sidecar written -> karaoke should fall back to phrase captions
        # (the SubtitlesClip path), not crash.
        tmp = tempfile.mkdtemp()
        srt = os.path.join(tmp, "subtitle.srt")
        with open(srt, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\nhello world\n\n")
        out = os.path.join(tmp, "out.mp4")
        params = _make_params()

        class _FakeVideo:
            duration = 10.0

            def with_audio(self, *a, **k):
                return self

            def write_videofile(self, *a, **k):
                pass

            def close(self):
                pass

        with patch.object(vd, "TextClip", _FakeTextClip), patch.object(
            vd, "_open_video_clip_quietly", return_value=_FakeVideo()
        ), patch.object(vd, "AudioFileClip") as fake_audio, patch.object(
            vd, "CompositeVideoClip", side_effect=lambda clips: clips[0]
        ), patch.object(
            vd, "get_bgm_file", return_value=""
        ), patch.object(
            vd, "SubtitlesClip"
        ) as fake_subs:
            fake_audio.return_value.with_effects.return_value = fake_audio.return_value
            fake_audio.return_value.fps = 44100
            fake_subs.return_value.subtitles = [((0.0, 1.0), "hello world")]
            # Should not raise.
            vd.generate_video(
                video_path="video.mp4",
                audio_path="audio.mp3",
                subtitle_path=srt,
                output_file=out,
                params=params,
            )
            # Fallback engaged the phrase-caption path.
            self.assertTrue(fake_subs.called)


if __name__ == "__main__":
    unittest.main()
