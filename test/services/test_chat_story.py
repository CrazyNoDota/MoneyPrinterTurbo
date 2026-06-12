"""WP5: fake-chat (messenger story) viral video format.

Covers the layers added for the chat format:
- llm.generate_chat_story : strict-JSON parsing + retry + fallback.
- hyperframes.chat_narration / _chat_scene_list : narration + proportional timing.
- studio.compose_chat : deterministic phone-frame HTML structure.
- hyperframes.render_chat_video dispatch + task.py mode wiring (everything mocked).
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import hyperframes as hf
from app.services import llm, task
from app.services.hyperframes import studio


# --------------------------------------------------------------------------- #
# llm.generate_chat_story
# --------------------------------------------------------------------------- #
class TestChatStoryLLM(unittest.TestCase):
    def test_parses_strict_json(self):
        payload = (
            '{"title": "The Text", "persons": ["Anna", "Ben"], "messages": ['
            '{"from": 0, "text": "You up?"},'
            '{"from": 1, "text": "Yeah why"},'
            '{"from": 0, "text": "It was me the whole time"}]}'
        )
        with mock.patch.object(llm, "_generate_response", return_value=payload):
            out = llm.generate_chat_story("breakup")
        self.assertEqual(out["persons"], ["Anna", "Ben"])
        self.assertEqual(len(out["messages"]), 3)
        self.assertEqual(out["messages"][0]["from"], 0)
        self.assertEqual(out["messages"][1]["from"], 1)
        self.assertEqual(out["messages"][0]["text"], "You up?")

    def test_strips_code_fence(self):
        fenced = (
            '```json\n{"title": "T", "persons": ["A", "B"], "messages": ['
            '{"from": 0, "text": "hi"}, {"from": 1, "text": "bye"}]}\n```'
        )
        with mock.patch.object(llm, "_generate_response", return_value=fenced):
            out = llm.generate_chat_story("x")
        self.assertEqual(out["title"], "T")
        self.assertEqual(len(out["messages"]), 2)

    def test_fills_missing_persons(self):
        payload = (
            '{"messages": [{"from": 0, "text": "one"}, {"from": 1, "text": "two"}]}'
        )
        with mock.patch.object(llm, "_generate_response", return_value=payload):
            out = llm.generate_chat_story("x")
        self.assertEqual(len(out["persons"]), 2)

    def test_coerces_bad_from_field(self):
        payload = (
            '{"persons": ["A", "B"], "messages": ['
            '{"from": "oops", "text": "one"}, {"from": 5, "text": "two"}]}'
        )
        with mock.patch.object(llm, "_generate_response", return_value=payload):
            out = llm.generate_chat_story("x")
        # non-int -> 0; any truthy int -> 1
        self.assertEqual(out["messages"][0]["from"], 0)
        self.assertEqual(out["messages"][1]["from"], 1)

    def test_retries_then_succeeds(self):
        good = (
            '{"persons": ["A", "B"], "messages": ['
            '{"from": 0, "text": "hi"}, {"from": 1, "text": "yo"}]}'
        )
        with mock.patch.object(llm, "_generate_response", side_effect=["garbage", good]), \
             mock.patch.object(llm, "_backoff_sleep"):
            out = llm.generate_chat_story("x")
        self.assertIsNotNone(out)
        self.assertEqual(len(out["messages"]), 2)

    def test_returns_none_on_persistent_failure(self):
        with mock.patch.object(llm, "_generate_response", return_value="nope"), \
             mock.patch.object(llm, "_backoff_sleep"):
            self.assertIsNone(llm.generate_chat_story("x"))

    def test_error_response_returns_none(self):
        with mock.patch.object(llm, "_generate_response", return_value="Error: boom"):
            self.assertIsNone(llm.generate_chat_story("x"))

    def test_too_few_messages_returns_none(self):
        payload = '{"persons": ["A", "B"], "messages": [{"from": 0, "text": "only"}]}'
        with mock.patch.object(llm, "_generate_response", return_value=payload), \
             mock.patch.object(llm, "_backoff_sleep"):
            self.assertIsNone(llm.generate_chat_story("x"))


# --------------------------------------------------------------------------- #
# narration + scene timing
# --------------------------------------------------------------------------- #
def _story():
    return {
        "title": "The Text",
        "persons": ["Anna", "Ben"],
        "messages": [
            {"from": 0, "text": "You up?"},
            {"from": 1, "text": "Yeah, what's wrong?"},
            {"from": 0, "text": "It was me the whole time."},
        ],
    }


class TestChatNarration(unittest.TestCase):
    def test_narration_names_speakers_and_marks_sides(self):
        script = hf.chat_narration(_story())
        lines = script.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("Anna: You up?", lines[0])
        self.assertIn("Ben: Yeah", lines[1])
        # each line carries the invisible side marker
        self.assertTrue(lines[0].startswith(hf._MARK + "0" + hf._MARK))
        self.assertTrue(lines[1].startswith(hf._MARK + "1" + hf._MARK))

    def test_narration_skips_empty_text(self):
        story = {"persons": ["A", "B"], "messages": [
            {"from": 0, "text": "hi"}, {"from": 1, "text": "  "}]}
        self.assertEqual(len(hf.chat_narration(story).splitlines()), 1)


class TestChatSceneList(unittest.TestCase):
    def test_one_scene_per_message_marked(self):
        sl = hf._chat_scene_list(_story(), total=12.0)
        self.assertEqual(len(sl), 3)
        # sides recovered from each scene's marker
        sides = [studio._chat_parts(s.text)[0] for s in sl]
        self.assertEqual(sides, [0, 1, 0])
        # scene text is the bare message (no "Name:" prefix), un-marked
        self.assertEqual(studio._chat_parts(sl[0].text)[1], "You up?")

    def test_scenes_are_contiguous_and_fill_total(self):
        total = 9.0
        sl = hf._chat_scene_list(_story(), total=total)
        self.assertAlmostEqual(sl[0].start, 0.0, places=2)
        for a, b in zip(sl, sl[1:]):
            self.assertAlmostEqual(a.end, b.start, places=2)
        self.assertAlmostEqual(sl[-1].end, total, places=1)

    def test_longer_message_gets_more_time(self):
        sl = hf._chat_scene_list(_story(), total=12.0)
        # "It was me the whole time." is longer than "You up?"
        self.assertGreater(sl[2].duration, sl[0].duration)

    def test_empty_story_no_scenes(self):
        self.assertEqual(hf._chat_scene_list({"messages": []}, total=10.0), [])


# --------------------------------------------------------------------------- #
# studio.compose_chat
# --------------------------------------------------------------------------- #
class TestComposeChat(unittest.TestCase):
    def _scenes(self):
        story = {
            "title": "T",
            "persons": ["Anna", "Ben"],
            "messages": [
                {"from": 0, "text": "Hi <b>there</b>"},
                {"from": 1, "text": "hello"},
            ],
        }
        return hf._chat_scene_list(story, total=10.0)

    def test_html_structure(self):
        html = studio.compose_chat(
            self._scenes(), "subj", 1080, 1920, 10.0, persons=["Anna", "Ben"], title="T"
        )
        self.assertIn("chat-header", html)
        self.assertIn("class=\"avatar\"", html)
        self.assertIn("Anna", html)               # contact name in header
        self.assertIn(">A<", html)                # avatar initial
        self.assertIn("msg-col", html)
        self.assertIn("bubble left", html)        # message 0 (from 0)
        self.assertIn("bubble right", html)       # message 1 (from 1)
        self.assertIn("typing", html)             # typing indicator for incoming

    def test_alternating_sides(self):
        html = studio.compose_chat(
            self._scenes(), "s", 1080, 1920, 10.0, persons=["Anna", "Ben"]
        )
        left_at = html.index("bubble left")
        right_at = html.index("bubble right")
        self.assertLess(left_at, right_at)        # msg 0 (left) before msg 1 (right)

    def test_user_text_is_escaped(self):
        html = studio.compose_chat(
            self._scenes(), "s", 1080, 1920, 10.0, persons=["Anna", "Ben"]
        )
        self.assertNotIn("<b>there</b>", html)
        self.assertIn("there", html)

    def test_has_timeline_and_pops(self):
        html = studio.compose_chat(
            self._scenes(), "s", 1080, 1920, 10.0, persons=["Anna", "Ben"]
        )
        self.assertIn('window.__timelines["main"]', html)
        self.assertIn("#msg-0", html)             # per-message tween targets
        self.assertIn("#msg-1", html)

    def test_no_infinite_repeat(self):
        # repeat:-1 makes the GSAP timeline's duration Infinity, which breaks
        # the renderer's frame seek: every tween freezes at t=0 and the video
        # comes out as an empty phone frame. The bounce must repeat finitely.
        html = studio.compose_chat(
            self._scenes(), "s", 1080, 1920, 10.0, persons=["Anna", "Ben"]
        )
        self.assertNotIn("repeat:-1", html)
        self.assertNotIn("repeat: -1", html)
        self.assertIn("repeat:", html)            # the bounce loop is still there

    def test_empty_returns_empty(self):
        self.assertEqual(studio.compose_chat([], "x", 1080, 1920, 0.0), "")


# --------------------------------------------------------------------------- #
# render dispatch + mode resolution
# --------------------------------------------------------------------------- #
class TestChatRenderDispatch(unittest.TestCase):
    def _params(self):
        return types.SimpleNamespace(
            video_aspect="portrait", video_subject="chat subject",
            video_source="pexels", n_threads=2, video_visual_mode="chat",
        )

    def test_mode_recognizes_chat(self):
        self.assertEqual(
            hf.mode(types.SimpleNamespace(video_visual_mode="chat")), "chat"
        )

    def test_render_chat_video_success(self):
        with mock.patch.object(hf, "is_available", return_value=True), \
             mock.patch.object(hf.assets, "reset"), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(hf.render, "render", return_value="chat.mp4"):
            out, ranges = hf.render_chat_video(
                "task", self._params(), _story(), "a.mp3", "s.srt", 12.0
            )
        self.assertEqual(out, "chat.mp4")
        self.assertEqual(ranges, [])

    def test_render_chat_unavailable_toolchain(self):
        with mock.patch.object(hf, "is_available", return_value=False):
            out, ranges = hf.render_chat_video(
                "task", self._params(), _story(), "a.mp3", "s.srt", 12.0
            )
        self.assertEqual((out, ranges), ("", []))

    def test_render_chat_compose_failure_falls_back(self):
        with mock.patch.object(hf, "is_available", return_value=True), \
             mock.patch.object(hf.assets, "reset"), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(studio, "compose_chat", return_value=""):
            out, ranges = hf.render_chat_video(
                "task", self._params(), _story(), "a.mp3", "s.srt", 12.0
            )
        self.assertEqual((out, ranges), ("", []))

    def test_render_chat_empty_story_falls_back(self):
        with mock.patch.object(hf, "is_available", return_value=True):
            out, ranges = hf.render_chat_video(
                "task", self._params(), {"messages": []}, "a.mp3", "s.srt", 12.0
            )
        self.assertEqual((out, ranges), ("", []))


# --------------------------------------------------------------------------- #
# task.py wiring
# --------------------------------------------------------------------------- #
class TestChatTaskWiring(unittest.TestCase):
    def _params(self):
        return types.SimpleNamespace(
            video_subject="subj", video_language="", video_script="",
            video_visual_mode="chat",
        )

    def test_generate_chat_dispatch(self):
        with mock.patch.object(task.llm, "generate_chat_story", return_value=_story()):
            script, data = task.generate_quiz_or_ranking("t", self._params(), "chat")
        self.assertTrue(script.strip())
        self.assertIn("Anna: You up?", script)
        self.assertEqual(data["persons"], ["Anna", "Ben"])

    def test_falls_back_on_none(self):
        with mock.patch.object(task.llm, "generate_chat_story", return_value=None):
            script, data = task.generate_quiz_or_ranking("t", self._params(), "chat")
        self.assertIsNone(script)
        self.assertIsNone(data)

    def test_non_fatal_on_exception(self):
        with mock.patch.object(task.llm, "generate_chat_story",
                               side_effect=RuntimeError("boom")):
            script, data = task.generate_quiz_or_ranking("t", self._params(), "chat")
        self.assertIsNone(script)
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
