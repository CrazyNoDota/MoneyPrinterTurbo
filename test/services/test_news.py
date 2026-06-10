import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import news
from app.services.news import base, rss, threads


RSS_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>Example Feed</title>
    <item>
      <title>First headline</title>
      <description><![CDATA[<p>Body of the <b>first</b> story.</p>]]></description>
      <link>https://example.com/1</link>
      <pubDate>Wed, 10 Jun 2026 08:30:00 GMT</pubDate>
      <media:thumbnail url="https://example.com/1.jpg"/>
      <media:content url="https://example.com/1-big.jpg" type="image/jpeg"/>
    </item>
    <item>
      <title>Second headline</title>
      <link>https://example.com/2</link>
      <enclosure url="https://example.com/2.mp4" type="video/mp4"/>
    </item>
    <item>
      <title></title>
      <link>https://example.com/skipped-no-title</link>
    </item>
  </channel>
</rss>
"""


def _threads_html(posts):
    payload = {"data": {"nested": {"thread_items": posts}}}
    return (
        "<html><head><script type=\"application/json\" data-sjs>"
        + json.dumps(payload)
        + "</script></head><body></body></html>"
    )


def _threads_post(text, code, taken_at=1780000000, image="https://cdn/img.jpg"):
    return {
        "post": {
            "caption": {"text": text},
            "code": code,
            "taken_at": taken_at,
            "image_versions2": {"candidates": [{"url": image}]},
        }
    }


class TestNewsItem(unittest.TestCase):
    def test_strips_html_and_falls_back_to_title_text(self):
        item = base.NewsItem(title="<b>Hello</b>  world", url="u")
        self.assertEqual(item.title, "Hello world")
        self.assertEqual(item.text, "Hello world")
        self.assertTrue(item.id)

    def test_id_is_stable_for_same_url_and_title(self):
        a = base.NewsItem(title="T", url="https://x/1")
        b = base.NewsItem(title="T", url="https://x/1")
        c = base.NewsItem(title="T", url="https://x/2")
        self.assertEqual(a.id, b.id)
        self.assertNotEqual(a.id, c.id)


class TestRSSSource(unittest.TestCase):
    def test_parse_fixture(self):
        source = rss.RSSSource(feeds=["https://feed"], timeout=5)
        items = source.parse(RSS_FIXTURE, "https://feed")
        self.assertEqual(len(items), 2)  # empty-title item skipped
        first = items[0]
        self.assertEqual(first.title, "First headline")
        self.assertEqual(first.text, "Body of the first story.")
        self.assertEqual(first.url, "https://example.com/1")
        self.assertEqual(first.source, "rss")
        self.assertTrue(first.published.startswith("2026-06-10T08:30:00"))
        self.assertEqual(
            first.media, ["https://example.com/1.jpg", "https://example.com/1-big.jpg"]
        )
        self.assertEqual(items[1].media, ["https://example.com/2.mp4"])

    def test_relative_urls_resolved_against_feed(self):
        xml = (
            b'<rss version="2.0"><channel><item>'
            b"<title>T</title><link>/articles/1</link>"
            b'<enclosure url="/img/1.jpg" type="image/jpeg"/>'
            b"</item></channel></rss>"
        )
        items = rss.RSSSource(feeds=["https://feed"]).parse(
            xml, "https://example.com/rss.xml"
        )
        self.assertEqual(items[0].url, "https://example.com/articles/1")
        self.assertEqual(items[0].media, ["https://example.com/img/1.jpg"])

    def test_atom_feed_yields_empty_not_crash(self):
        atom = (
            b'<feed xmlns="http://www.w3.org/2005/Atom">'
            b"<entry><title>Atom entry</title></entry></feed>"
        )
        self.assertEqual(rss.RSSSource(feeds=["https://feed"]).parse(atom, "f"), [])

    def test_malformed_xml_returns_empty(self):
        source = rss.RSSSource(feeds=["https://feed"])
        self.assertEqual(source.parse(b"<not really xml", "https://feed"), [])

    def test_latest_fetches_and_limits(self):
        source = rss.RSSSource(feeds=["https://feed"], timeout=5)
        resp = SimpleNamespace(status_code=200, content=RSS_FIXTURE)
        with mock.patch.object(rss.requests, "get", return_value=resp) as get:
            items = source.latest(limit=1)
        self.assertEqual(len(items), 1)
        self.assertEqual(get.call_args.kwargs["timeout"], 5)

    def test_http_error_and_exception_are_non_fatal(self):
        source = rss.RSSSource(feeds=["https://a", "https://b"], timeout=5)
        bad = SimpleNamespace(status_code=503, content=b"")
        with mock.patch.object(
            rss.requests, "get", side_effect=[bad, RuntimeError("boom")]
        ):
            self.assertEqual(source.latest(), [])

    def test_not_configured_without_feeds(self):
        self.assertFalse(rss.RSSSource(feeds=[]).is_configured())
        self.assertTrue(rss.RSSSource(feeds=["https://feed"]).is_configured())


class TestThreadsSource(unittest.TestCase):
    def test_parse_embedded_json(self):
        html = _threads_html(
            [
                _threads_post("Big news today\nMore details here", "C1"),
                _threads_post("Second post", "C2", image=""),
                _threads_post("", "C3"),  # captionless -> skipped
                _threads_post("Big news today\nMore details here", "C1"),  # dup code
            ]
        )
        source = threads.ThreadsSource(profiles=["someuser"], timeout=5)
        items = source.parse(html, "someuser")
        self.assertEqual(len(items), 2)
        first = items[0]
        self.assertEqual(first.title, "Big news today")
        self.assertIn("More details here", first.text)
        self.assertEqual(first.url, "https://www.threads.net/@someuser/post/C1")
        self.assertEqual(first.source, "threads")
        self.assertEqual(first.media, ["https://cdn/img.jpg"])
        self.assertTrue(first.published)  # taken_at -> ISO

    def test_parse_single_quoted_and_charset_script_tags(self):
        payload = json.dumps({"thread_items": [_threads_post("Quoted post", "Q1")]})
        for tag in (
            f"<script type='application/json'>{payload}</script>",
            f'<script nonce="x" type="application/json; charset=utf-8">{payload}</script>',
        ):
            items = threads.ThreadsSource(profiles=["u"], timeout=5).parse(
                f"<html>{tag}</html>", "u"
            )
            self.assertEqual([i.title for i in items], ["Quoted post"])

    def test_parse_garbage_html_returns_empty(self):
        source = threads.ThreadsSource(profiles=["x"], timeout=5)
        self.assertEqual(source.parse("<html>nothing here</html>", "x"), [])
        self.assertEqual(
            source.parse(
                '<script type="application/json">{"thread_items": not-json</script>', "x"
            ),
            [],
        )

    def test_latest_swallows_network_errors(self):
        source = threads.ThreadsSource(profiles=["x"], timeout=5)
        with mock.patch.object(threads.requests, "get", side_effect=RuntimeError("net")):
            self.assertEqual(source.latest(), [])

    def test_profiles_normalized_and_configured(self):
        source = threads.ThreadsSource(profiles=["@user", " other ", ""])
        self.assertEqual(source.profiles, ["user", "other"])
        self.assertTrue(source.is_configured())
        self.assertFalse(threads.ThreadsSource(profiles=[]).is_configured())


class TestLatest(unittest.TestCase):
    def _source(self, name, items, configured=True):
        source = mock.Mock(spec=news.NewsSource)
        source.name = name
        source.is_configured.return_value = configured
        if isinstance(items, Exception):
            source.latest.side_effect = items
        else:
            source.latest.return_value = items
        return source

    def test_threads_unavailable_falls_back_to_rss(self):
        rss_items = [base.NewsItem(title="From RSS", url="https://r/1")]
        with mock.patch.object(
            news,
            "_sources",
            return_value=[self._source("threads", []), self._source("rss", rss_items)],
        ):
            items = news.latest()
        self.assertEqual([i.title for i in items], ["From RSS"])

    def test_threads_exception_falls_back_to_rss(self):
        rss_items = [base.NewsItem(title="From RSS", url="https://r/1")]
        with mock.patch.object(
            news,
            "_sources",
            return_value=[
                self._source("threads", RuntimeError("scrape broke")),
                self._source("rss", rss_items),
            ],
        ):
            items = news.latest()
        self.assertEqual(len(items), 1)

    def test_first_source_with_items_wins_and_dedupes(self):
        dup = base.NewsItem(title="Same", url="https://t/1")
        threads_items = [dup, base.NewsItem(title="Same", url="https://t/1")]
        with mock.patch.object(
            news,
            "_sources",
            return_value=[
                self._source("threads", threads_items),
                self._source("rss", [base.NewsItem(title="rss", url="r")]),
            ],
        ):
            items = news.latest(limit=5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "")

    def test_unconfigured_sources_are_skipped(self):
        with mock.patch.object(
            news,
            "_sources",
            return_value=[
                self._source("threads", [], configured=False),
                self._source("rss", [], configured=False),
            ],
        ):
            self.assertEqual(news.latest(), [])


class TestGenerateNewsScript(unittest.TestCase):
    def _item(self, **overrides):
        defaults = dict(
            title="Markets rally",
            text="Stocks rose sharply on Tuesday.",
            url="https://example.com/1",
            source="rss",
            published="2026-06-10T08:30:00+00:00",
            media=[],
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_prompt_contains_item_and_language(self):
        from app.services import llm

        with mock.patch.object(
            llm, "_generate_response", return_value="The script."
        ) as gen:
            script = llm.generate_news_script(self._item(), language="ru-RU")
        self.assertEqual(script, "The script.")
        prompt = gen.call_args.kwargs["prompt"]
        self.assertIn("Markets rally", prompt)
        self.assertIn("Stocks rose sharply", prompt)
        self.assertIn("ru-RU", prompt)

    def test_empty_item_returns_empty_without_llm_call(self):
        from app.services import llm

        with mock.patch.object(llm, "_generate_response") as gen:
            script = llm.generate_news_script(self._item(title="", text=""))
        self.assertEqual(script, "")
        gen.assert_not_called()

    def test_llm_failure_returns_empty(self):
        from app.services import llm

        with mock.patch.object(
            llm, "_generate_response", return_value=""
        ), mock.patch.object(llm, "_backoff_sleep"):
            script = llm.generate_news_script(self._item())
        self.assertEqual(script, "")

    def test_markdown_is_stripped(self):
        from app.services import llm

        with mock.patch.object(
            llm, "_generate_response", return_value="## Title\n**Bold** script"
        ):
            script = llm.generate_news_script(self._item())
        self.assertNotIn("#", script)
        self.assertNotIn("*", script)


if __name__ == "__main__":
    unittest.main()
