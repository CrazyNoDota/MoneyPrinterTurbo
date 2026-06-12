"""WP6 hook-first scripting: greeting sanitizer + prompt constraints (llm.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import llm
from app.services.llm import _strip_greeting_hook


class StripGreetingHookTests(unittest.TestCase):
    def test_strips_english_welcome(self):
        script = "Welcome to my channel. Bitcoin just crashed hard."
        self.assertEqual(_strip_greeting_hook(script), "Bitcoin just crashed hard.")

    def test_strips_today_we(self):
        script = "Today we explore black holes. They bend light itself."
        self.assertEqual(_strip_greeting_hook(script), "They bend light itself.")

    def test_strips_in_this_video(self):
        script = "In this video, you will learn three tricks. Trick one is timing."
        self.assertEqual(_strip_greeting_hook(script), "Trick one is timing.")

    def test_strips_russian_privet(self):
        script = "Привет, друзья! Цена золота резко выросла."
        self.assertEqual(_strip_greeting_hook(script), "Цена золота резко выросла.")

    def test_strips_russian_segodnya_my(self):
        script = "Сегодня мы поговорим о космосе. Звёзды умирают каждый день."
        self.assertEqual(_strip_greeting_hook(script), "Звёзды умирают каждый день.")

    def test_clean_english_untouched(self):
        script = "Bitcoin just crashed hard. Here is why it matters."
        self.assertEqual(_strip_greeting_hook(script), script)

    def test_clean_russian_untouched(self):
        script = "Цена золота резко выросла. Вот почему это важно."
        self.assertEqual(_strip_greeting_hook(script), script)

    def test_banned_word_midscript_untouched(self):
        # "today we" appears later, not as the opening -- leave it alone.
        script = "Markets shook overnight. Today we see the fallout spread."
        self.assertEqual(_strip_greeting_hook(script), script)

    def test_greeting_only_kept(self):
        # Nothing follows the greeting -- never return an empty script.
        script = "Welcome to the channel."
        self.assertEqual(_strip_greeting_hook(script), script)

    def test_empty_input_safe(self):
        self.assertEqual(_strip_greeting_hook(""), "")
        self.assertEqual(_strip_greeting_hook("   "), "   ")

    def test_leading_quote_greeting_stripped(self):
        script = '"Welcome everyone." The data tells a wild story.'
        self.assertEqual(_strip_greeting_hook(script), "The data tells a wild story.")


class PromptConstraintTests(unittest.TestCase):
    """Simple string assertions that the new hook rules are wired into prompts."""

    def test_shared_hook_rules_present(self):
        rules = llm._HOOK_RULES.lower()
        self.assertIn("pattern-interrupt hook", rules)
        self.assertIn("8 words or fewer", rules)
        self.assertIn("welcome", rules)
        self.assertIn("привет", rules)
        self.assertIn("сегодня мы", rules)

    def test_generate_script_prompt_includes_hook(self):
        captured = {}

        def fake_response(prompt):
            captured["prompt"] = prompt
            return "Bitcoin crashed. Here is why."

        orig = llm._generate_response
        llm._generate_response = fake_response
        try:
            llm.generate_script("crypto news", paragraph_number=1)
        finally:
            llm._generate_response = orig
        p = captured["prompt"].lower()
        self.assertIn("pattern-interrupt hook", p)
        self.assertIn("8 words or fewer", p)
        self.assertIn("привет", p)

    def test_news_script_prompt_includes_hook(self):
        captured = {}

        def fake_response(prompt):
            captured["prompt"] = prompt
            return "Stocks plunged today. Investors are nervous."

        item = type("Item", (), {"title": "Markets fall", "text": "Big drop today."})()
        orig = llm._generate_response
        llm._generate_response = fake_response
        try:
            llm.generate_news_script(item, paragraph_number=1)
        finally:
            llm._generate_response = orig
        p = captured["prompt"].lower()
        self.assertIn("pattern-interrupt hook", p)
        self.assertIn("8 words or fewer", p)
        self.assertIn("привет", p)

    def test_generate_script_applies_sanitizer(self):
        def fake_response(prompt):
            return "Welcome to the channel. Bitcoin crashed today."

        orig = llm._generate_response
        llm._generate_response = fake_response
        try:
            out = llm.generate_script("crypto", paragraph_number=1)
        finally:
            llm._generate_response = orig
        self.assertNotIn("Welcome", out)
        self.assertIn("Bitcoin crashed today.", out)


if __name__ == "__main__":
    unittest.main()
