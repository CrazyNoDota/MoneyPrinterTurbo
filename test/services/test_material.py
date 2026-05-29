import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import material


class TestMaterialTlsVerification(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    def test_search_pexels_uses_tls_verification_by_default(self):
        """
        默认路径必须开启 TLS 校验，避免素材 API key 和返回的素材 URL
        在公共网络或不可信代理环境中被中间人攻击截获或篡改。
        """
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "duration": 8,
                        "video_files": [
                            {
                                "width": 1080,
                                "height": 1920,
                                "link": "https://example.com/video.mp4",
                            }
                        ],
                    }
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response) as get:
            results = material.search_videos_pexels("cat", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertTrue(get.call_args.kwargs["verify"])

    def test_search_pixabay_allows_explicit_tls_disable_for_proxy(self):
        """
        少数企业代理会使用自签证书。该场景必须显式配置关闭 TLS 校验，
        不能再由代码硬编码默认关闭。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["tls_verify"] = False
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "url": "https://example.com/video.mp4",
                            }
                        },
                    }
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response) as get:
            results = material.search_videos_pixabay("cat", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertFalse(get.call_args.kwargs["verify"])

    def test_save_video_uses_tls_verification_by_default(self):
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(content=b"fake-video")

        class FakeVideoFileClip:
            duration = 1
            fps = 24

            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "app.services.material.requests.get", return_value=fake_response
            ) as get, patch("app.services.material.VideoFileClip", FakeVideoFileClip):
                video_path = material.save_video(
                    "https://example.com/video.mp4?token=abc", save_dir=temp_dir
                )

            self.assertTrue(os.path.exists(video_path))
            self.assertTrue(get.call_args.kwargs["verify"])


class TestListLocalMaterials(unittest.TestCase):
    def test_scans_supported_files_sorted_with_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # supported media
            for fname in ["b.mp4", "a.jpg", "c.png"]:
                Path(temp_dir, fname).write_bytes(b"x")
            # unsupported files are ignored
            Path(temp_dir, "notes.txt").write_bytes(b"x")
            Path(temp_dir, "archive.zip").write_bytes(b"x")

            materials = material.list_local_materials(temp_dir, recursive=False)

        names = [m.name for m in materials]
        self.assertEqual(names, ["a.jpg", "b.mp4", "c.png"])  # sorted by path
        self.assertTrue(all(m.provider == "local" for m in materials))
        self.assertTrue(all(os.path.isabs(m.url) for m in materials))

    def test_recursive_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sub = os.path.join(temp_dir, "sub")
            os.makedirs(sub)
            Path(temp_dir, "top.jpg").write_bytes(b"x")
            Path(sub, "nested.mov").write_bytes(b"x")

            materials = material.list_local_materials(temp_dir, recursive=True)

        self.assertEqual({m.name for m in materials}, {"top.jpg", "nested.mov"})

    def test_missing_directory_returns_empty(self):
        self.assertEqual(material.list_local_materials("/no/such/dir"), [])
        self.assertEqual(material.list_local_materials(""), [])


if __name__ == "__main__":
    unittest.main()
