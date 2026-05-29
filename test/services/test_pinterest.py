import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import pinterest


def _result(images):
    return {"images": images}


class TestPinterestParse(unittest.TestCase):
    def test_parse_results_prefers_original_size(self):
        payload = {
            "resource_response": {
                "data": {
                    "results": [
                        _result(
                            {
                                "236x": {"url": "https://i.pinimg.com/236x/a.jpg"},
                                "orig": {"url": "https://i.pinimg.com/orig/a.jpg"},
                            }
                        )
                    ]
                }
            }
        }
        self.assertEqual(
            pinterest.parse_results(payload),
            ["https://i.pinimg.com/orig/a.jpg"],
        )

    def test_parse_results_falls_back_to_widest_variant(self):
        payload = {
            "resource_response": {
                "data": {
                    "results": [
                        _result(
                            {
                                "small": {"url": "https://x/s.jpg", "width": 200},
                                "big": {"url": "https://x/b.jpg", "width": 900},
                            }
                        )
                    ]
                }
            }
        }
        self.assertEqual(pinterest.parse_results(payload), ["https://x/b.jpg"])

    def test_parse_results_dedupes_and_skips_imageless(self):
        payload = {
            "resource_response": {
                "data": {
                    "results": [
                        _result({"orig": {"url": "https://x/a.jpg"}}),
                        {"id": "no-images-story-pin"},
                        _result({"orig": {"url": "https://x/a.jpg"}}),  # duplicate
                        _result({"orig": {"url": "https://x/b.jpg"}}),
                    ]
                }
            }
        }
        self.assertEqual(
            pinterest.parse_results(payload),
            ["https://x/a.jpg", "https://x/b.jpg"],
        )

    def test_parse_results_handles_malformed_payload(self):
        self.assertEqual(pinterest.parse_results({}), [])
        self.assertEqual(pinterest.parse_results({"resource_response": {}}), [])
        self.assertEqual(
            pinterest.parse_results({"resource_response": {"data": {"results": None}}}),
            [],
        )


class TestPinterestSearch(unittest.TestCase):
    def test_search_images_empty_query_skips_network(self):
        with mock.patch.object(pinterest.requests, "get") as get:
            self.assertEqual(pinterest.search_images("   "), [])
        get.assert_not_called()

    def test_search_images_returns_capped_urls(self):
        payload = {
            "resource_response": {
                "data": {
                    "results": [
                        _result({"orig": {"url": f"https://x/{i}.jpg"}})
                        for i in range(10)
                    ]
                }
            }
        }
        fake_resp = mock.Mock()
        fake_resp.json.return_value = payload
        fake_resp.raise_for_status.return_value = None
        with mock.patch.object(pinterest.requests, "get", return_value=fake_resp):
            urls = pinterest.search_images("money", limit=3)
        self.assertEqual(urls, ["https://x/0.jpg", "https://x/1.jpg", "https://x/2.jpg"])

    def test_search_images_swallows_network_errors(self):
        with mock.patch.object(
            pinterest.requests, "get", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(pinterest.search_images("money"), [])


if __name__ == "__main__":
    unittest.main()
