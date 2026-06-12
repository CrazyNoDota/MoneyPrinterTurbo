"""WP2: quiz + Top-N ranking viral video formats.

Covers the three layers added for these formats:
- llm.generate_quiz / generate_ranking : strict-JSON parsing + retry + fallback.
- studio.compose_quiz / compose_ranking : deterministic HTML structure.
- hyperframes render dispatch + task.py mode wiring (everything mocked).
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
from app.services.hyperframes import scenes, studio


# --------------------------------------------------------------------------- #
# llm.generate_quiz / generate_ranking
# --------------------------------------------------------------------------- #
class TestQuizRankingLLM(unittest.TestCase):
    def test_generate_quiz_parses_strict_json(self):
        payload = (
            '{"questions": [{"q": "Q1?", "a": "A1", "fun_fact": "F1"}, '
            '{"q": "Q2?", "a": "A2"}]}'
        )
        with mock.patch.object(llm, "_generate_response", return_value=payload):
            out = llm.generate_quiz("space", count=2)
        self.assertEqual(len(out["questions"]), 2)
        self.assertEqual(out["questions"][0]["q"], "Q1?")
        self.assertEqual(out["questions"][1]["fun_fact"], "")

    def test_generate_quiz_strips_code_fence(self):
        fenced = '```json\n{"questions": [{"q": "Q?", "a": "A"}]}\n```'
        with mock.patch.object(llm, "_generate_response", return_value=fenced):
            out = llm.generate_quiz("dogs", count=3)
        self.assertEqual(out["questions"][0]["a"], "A")

    def test_generate_quiz_retries_then_succeeds(self):
        good = '{"questions": [{"q": "Q?", "a": "A"}]}'
        responses = ["not json at all", good]
        with mock.patch.object(llm, "_generate_response", side_effect=responses), \
             mock.patch.object(llm, "_backoff_sleep"):
            out = llm.generate_quiz("cats")
        self.assertIsNotNone(out)
        self.assertEqual(len(out["questions"]), 1)

    def test_generate_quiz_returns_none_on_persistent_failure(self):
        with mock.patch.object(llm, "_generate_response", return_value="garbage"), \
             mock.patch.object(llm, "_backoff_sleep"):
            self.assertIsNone(llm.generate_quiz("x"))

    def test_generate_quiz_error_response_returns_none(self):
        with mock.patch.object(llm, "_generate_response", return_value="Error: boom"):
            self.assertIsNone(llm.generate_quiz("x"))

    def test_generate_ranking_orders_n_to_1(self):
        payload = (
            '{"title": "Top 3 Cats", "items": ['
            '{"rank": 1, "name": "One", "reason": "best"},'
            '{"rank": 3, "name": "Three", "reason": "ok"},'
            '{"rank": 2, "name": "Two", "reason": "good"}]}'
        )
        with mock.patch.object(llm, "_generate_response", return_value=payload):
            out = llm.generate_ranking("cats", count=3)
        ranks = [it["rank"] for it in out["items"]]
        self.assertEqual(ranks, [3, 2, 1])
        self.assertEqual(out["title"], "Top 3 Cats")

    def test_generate_ranking_dedups_ranks(self):
        payload = (
            '{"title": "T", "items": ['
            '{"rank": 2, "name": "A"}, {"rank": 2, "name": "B"},'
            '{"rank": 1, "name": "C"}]}'
        )
        with mock.patch.object(llm, "_generate_response", return_value=payload):
            out = llm.generate_ranking("x", count=2)
        self.assertEqual([it["rank"] for it in out["items"]], [2, 1])

    def test_generate_ranking_returns_none_on_failure(self):
        with mock.patch.object(llm, "_generate_response", return_value="nope"), \
             mock.patch.object(llm, "_backoff_sleep"):
            self.assertIsNone(llm.generate_ranking("x"))


# --------------------------------------------------------------------------- #
# studio.compose_quiz / compose_ranking
# --------------------------------------------------------------------------- #
class TestComposeQuiz(unittest.TestCase):
    def _quiz(self):
        return {
            "questions": [
                {"q": "What is <b>2+2</b>?", "a": "Four", "fun_fact": "Basic math"},
            ]
        }

    def _scenes(self):
        return hf._quiz_scene_list(self._quiz(), total=12.0)

    def test_scene_roles_present(self):
        sl = self._scenes()
        roles = [studio._quiz_role(s.text) for s in sl]
        self.assertEqual(roles, ["question", "countdown", "answer"])

    def test_countdown_beat_min_duration(self):
        sl = self._scenes()
        countdown = [s for s in sl if studio._quiz_role(s.text) == "countdown"][0]
        self.assertGreaterEqual(countdown.duration, 2.0)

    def test_html_contains_countdown_and_answer(self):
        sl = self._scenes()
        html = studio.compose_quiz(sl, "math", 1080, 1920, 12.0)
        self.assertIn("cd-digit", html)          # animated countdown digit element
        self.assertIn(">QUESTION<", html)
        self.assertIn(">ANSWER<", html)
        self.assertIn("Four", html)              # answer text rendered
        # user text is HTML-escaped (no raw injected tag)
        self.assertNotIn("<b>2+2</b>", html)
        self.assertIn("2+2", html)

    def test_compose_quiz_empty_returns_empty(self):
        self.assertEqual(studio.compose_quiz([], "x", 1080, 1920, 0.0), "")


class TestComposeRanking(unittest.TestCase):
    def _ranking(self):
        return {
            "title": "Top 2 <Cats>",
            "items": [
                {"rank": 2, "name": "Tabby", "reason": "cute"},
                {"rank": 1, "name": "Lion", "reason": "king"},
            ],
        }

    def _scenes(self):
        return hf._ranking_scene_list(self._ranking(), total=10.0)

    def test_scene_list_has_title_and_items(self):
        sl = self._scenes()
        # title scene + 2 ranked scenes
        self.assertEqual(len(sl), 3)
        self.assertIsNone(studio._rank_parts(sl[0].text))
        self.assertEqual(studio._rank_parts(sl[1].text)[0], 2)
        self.assertEqual(studio._rank_parts(sl[2].text)[0], 1)

    def test_html_contains_rank_badges(self):
        sl = self._scenes()
        html = studio.compose_ranking(sl, "cats", 1080, 1920, 10.0,
                                      title="Top 2 <Cats>")
        self.assertIn("rank-badge", html)
        self.assertIn(">#2<", html)
        self.assertIn(">#1<", html)
        self.assertIn("gold", html)              # #1 gets the gold badge class
        self.assertIn("Lion", html)
        # title is escaped
        self.assertNotIn("<Cats>", html)

    def test_compose_ranking_empty_returns_empty(self):
        self.assertEqual(studio.compose_ranking([], "x", 1080, 1920, 0.0), "")


# --------------------------------------------------------------------------- #
# narration builders + render dispatch
# --------------------------------------------------------------------------- #
class TestNarration(unittest.TestCase):
    def test_quiz_narration_has_countdown_cue(self):
        quiz = {"questions": [{"q": "Q?", "a": "A", "fun_fact": "F"}]}
        script = hf.quiz_narration(quiz)
        self.assertIn("Q?", script)
        self.assertIn(hf.QUIZ_COUNTDOWN_CUE, script)
        self.assertIn("A", script)
        self.assertIn("F", script)

    def test_ranking_narration_orders_lines(self):
        ranking = {"title": "Top 2", "items": [
            {"rank": 2, "name": "B", "reason": "r2"},
            {"rank": 1, "name": "A", "reason": "r1"},
        ]}
        script = hf.ranking_narration(ranking)
        lines = script.splitlines()
        self.assertEqual(lines[0], "Top 2")
        self.assertIn("B", lines[1])
        self.assertIn("A", lines[2])


class TestRenderDispatch(unittest.TestCase):
    def _params(self):
        return types.SimpleNamespace(
            video_aspect="portrait", video_subject="quiz subject",
            video_source="pexels", n_threads=2, video_visual_mode="quiz",
        )

    def test_mode_recognizes_quiz_and_ranking(self):
        self.assertEqual(hf.mode(types.SimpleNamespace(video_visual_mode="quiz")), "quiz")
        self.assertEqual(
            hf.mode(types.SimpleNamespace(video_visual_mode="ranking")), "ranking"
        )

    def test_render_quiz_video_success(self):
        quiz = {"questions": [{"q": "Q?", "a": "A", "fun_fact": "F"}]}
        with mock.patch.object(hf, "is_available", return_value=True), \
             mock.patch.object(hf.assets, "reset"), \
             mock.patch.object(hf, "_solely_backgrounds", return_value=[]), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(hf.render, "render", return_value="quiz.mp4"):
            out, ranges = hf.render_quiz_video(
                "task", self._params(), quiz, "a.mp3", "s.srt", 12.0
            )
        self.assertEqual(out, "quiz.mp4")
        self.assertEqual(ranges, [])

    def test_render_quiz_unavailable_toolchain(self):
        with mock.patch.object(hf, "is_available", return_value=False):
            out, ranges = hf.render_quiz_video(
                "task", self._params(), {"questions": []}, "a.mp3", "s.srt", 12.0
            )
        self.assertEqual((out, ranges), ("", []))

    def test_render_ranking_video_success(self):
        ranking = {"title": "Top 2", "items": [
            {"rank": 2, "name": "B", "reason": "x"},
            {"rank": 1, "name": "A", "reason": "y"},
        ]}
        with mock.patch.object(hf, "is_available", return_value=True), \
             mock.patch.object(hf.assets, "reset"), \
             mock.patch.object(hf, "_solely_backgrounds", return_value=[]), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(hf.render, "render", return_value="ranking.mp4"):
            out, ranges = hf.render_ranking_video(
                "task", self._params(), ranking, "a.mp3", "s.srt", 10.0
            )
        self.assertEqual(out, "ranking.mp4")
        self.assertEqual(ranges, [])

    def test_render_quiz_compose_failure_falls_back(self):
        with mock.patch.object(hf, "is_available", return_value=True), \
             mock.patch.object(hf.assets, "reset"), \
             mock.patch.object(hf, "_solely_backgrounds", return_value=[]), \
             mock.patch.object(hf.utils, "task_dir", return_value=tempfile.gettempdir()), \
             mock.patch.object(studio, "compose_quiz", return_value=""):
            out, ranges = hf.render_quiz_video(
                "task", self._params(),
                {"questions": [{"q": "Q?", "a": "A"}]}, "a.mp3", "s.srt", 12.0,
            )
        self.assertEqual((out, ranges), ("", []))


# --------------------------------------------------------------------------- #
# task.py wiring
# --------------------------------------------------------------------------- #
class TestTaskWiring(unittest.TestCase):
    def _params(self, mode):
        return types.SimpleNamespace(
            video_subject="subj", video_language="", video_script="",
            video_visual_mode=mode,
        )

    def test_generate_quiz_or_ranking_quiz(self):
        quiz = {"questions": [{"q": "Q?", "a": "A", "fun_fact": "F"}]}
        with mock.patch.object(task.llm, "generate_quiz", return_value=quiz):
            script, data = task.generate_quiz_or_ranking("t", self._params("quiz"), "quiz")
        self.assertIn("Q?", script)
        self.assertIs(data, quiz)

    def test_generate_quiz_or_ranking_ranking(self):
        ranking = {"title": "Top 2", "items": [
            {"rank": 2, "name": "B"}, {"rank": 1, "name": "A"}]}
        with mock.patch.object(task.llm, "generate_ranking", return_value=ranking):
            script, data = task.generate_quiz_or_ranking(
                "t", self._params("ranking"), "ranking"
            )
        self.assertTrue(script.strip())
        self.assertIs(data, ranking)

    def test_generate_quiz_or_ranking_falls_back_on_none(self):
        with mock.patch.object(task.llm, "generate_quiz", return_value=None):
            script, data = task.generate_quiz_or_ranking("t", self._params("quiz"), "quiz")
        self.assertIsNone(script)
        self.assertIsNone(data)

    def test_generate_quiz_or_ranking_non_fatal_on_exception(self):
        with mock.patch.object(task.llm, "generate_quiz", side_effect=RuntimeError("boom")):
            script, data = task.generate_quiz_or_ranking("t", self._params("quiz"), "quiz")
        self.assertIsNone(script)
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
