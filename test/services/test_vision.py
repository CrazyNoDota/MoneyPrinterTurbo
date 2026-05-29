import base64
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

from app.config import config
from app.models.schema import MaterialInfo
from app.services import vision

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")


class TestVision(unittest.TestCase):
    def setUp(self):
        self.original_app = dict(config.app)
        self.test_img = os.path.join(resources_dir, "1.png")
        # The on-disk vision cache persists across tests; disable it by default
        # so cases don't contaminate each other. The cache test re-enables it
        # with an in-memory stub.
        config.app["vision_cache"] = False

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app)

    def test_image_to_jpeg_b64_stays_under_size_cap(self):
        image = Image.new("RGB", (1500, 1500), (123, 200, 50))
        encoded = vision._image_to_jpeg_b64(image)
        self.assertLessEqual(len(encoded), vision._MAX_IMAGE_B64_BYTES)
        # must be valid base64 of a JPEG
        raw = base64.b64decode(encoded)
        self.assertTrue(raw.startswith(b"\xff\xd8"))  # JPEG SOI marker

    def test_describe_media_image_uses_caption(self):
        with mock.patch.object(vision, "_caption_frame", return_value="a fluffy cat on a sofa"):
            desc = vision.describe_media(self.test_img)
        self.assertEqual(desc, "a fluffy cat on a sofa")

    def test_describe_media_dedupes_identical_frame_captions(self):
        with mock.patch.object(vision, "_caption_frame", return_value="same scene"):
            # two frames returning identical captions collapse into one
            with mock.patch.object(
                vision, "_sample_video_frames",
                return_value=[Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))],
            ):
                desc = vision.describe_media("/fake/video.mp4")
        self.assertEqual(desc, "same scene")

    def test_describe_media_survives_caption_failure(self):
        with mock.patch.object(vision, "_caption_frame", side_effect=RuntimeError("boom")):
            desc = vision.describe_media(self.test_img)
        self.assertEqual(desc, "")

    def test_describe_materials_skips_when_no_key(self):
        config.app.pop("vision_api_key", None)
        config.app.pop("nvidia_api_key", None)
        materials = [MaterialInfo(provider="local", url=self.test_img)]
        self.assertEqual(vision.describe_materials(materials), [])

    def test_describe_materials_returns_descriptions(self):
        config.app["vision_api_key"] = "test-key"
        materials = [MaterialInfo(provider="local", url=self.test_img)]
        with mock.patch.object(vision, "describe_media", return_value="a scene"):
            result = vision.describe_materials(materials)
        self.assertEqual(result, ["a scene"])

    def test_describe_materials_skips_missing_files(self):
        config.app["vision_api_key"] = "test-key"
        materials = [MaterialInfo(provider="local", url="/no/such/file.png")]
        with mock.patch.object(vision, "describe_media", return_value="x") as dm:
            result = vision.describe_materials(materials)
        self.assertEqual(result, [])
        dm.assert_not_called()

    def test_describe_media_caches_result(self):
        # cache hit on the second call avoids a second vision API round-trip
        config.app["vision_cache"] = True
        store = {}
        with mock.patch.object(
            vision.cache, "get", side_effect=lambda ns, k: store.get(k)
        ), mock.patch.object(
            vision.cache, "set", side_effect=lambda ns, k, v: store.__setitem__(k, v)
        ), mock.patch.object(
            vision, "_caption_frame", return_value="a cat"
        ) as cap:
            first = vision.describe_media(self.test_img)
            second = vision.describe_media(self.test_img)
        self.assertEqual(first, "a cat")
        self.assertEqual(second, "a cat")
        cap.assert_called_once()  # second call served from cache

    def test_describe_materials_runs_in_parallel_preserving_order(self):
        config.app["vision_api_key"] = "test-key"
        config.app["vision_concurrency"] = 4
        materials = [
            MaterialInfo(provider="local", url=os.path.join(resources_dir, f"{i}.png"))
            for i in range(1, 4)
        ]
        with mock.patch.object(
            vision, "describe_media", side_effect=lambda p: f"desc-{os.path.basename(p)}"
        ):
            result = vision.describe_materials(materials)
        self.assertEqual(result, ["desc-1.png", "desc-2.png", "desc-3.png"])


if __name__ == "__main__":
    unittest.main()
