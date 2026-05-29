import unittest
import os
import sys
from pathlib import Path
from unittest import mock

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm
from app.models.schema import MaterialInfo, VideoParams

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

class TestTaskService(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_task_local_materials(self):
        task_id = "00000000-0000-0000-0000-000000000000"
        video_materials=[]
        for i in range(1, 4):
            video_materials.append(MaterialInfo(
                provider="local",
                url=os.path.join(resources_dir, f"{i}.png"),
                duration=0
            ))

        params = VideoParams(
            video_subject="金钱的作用",
            video_script="金钱不仅是交换媒介，更是社会资源的分配工具。它能满足基本生存需求，如食物和住房，也能提供教育、医疗等提升生活品质的机会。拥有足够的金钱意味着更多选择权，比如职业自由或创业可能。但金钱的作用也有边界，它无法直接购买幸福、健康或真诚的人际关系。过度追逐财富可能导致价值观扭曲，忽视精神层面的需求。理想的状态是理性看待金钱，将其作为实现目标的工具而非终极目的。",
            video_terms="money importance, wealth and society, financial freedom, money and happiness, role of money",
            video_aspect="9:16",
            video_concat_mode="random",
            video_transition_mode="None",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
            video_materials=video_materials,
            video_language="",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            voice_volume=1.0,
            voice_rate=1.0,
            bgm_type="random",
            bgm_file="",
            bgm_volume=0.2,
            subtitle_enabled=True,
            subtitle_position="bottom",
            custom_position=70.0,
            font_name="MicrosoftYaHeiBold.ttc",
            text_fore_color="#FFFFFF",
            text_background_color=True,
            font_size=60,
            stroke_color="#000000",
            stroke_width=1.5,
            n_threads=2,
            paragraph_number=1
        )
        result = tm.start(task_id=task_id, params=params)
        print(result)


class TestGetVideoMaterials(unittest.TestCase):
    """Hybrid material gathering: uploads first, then fill gaps with stock."""

    TASK_ID = "00000000-0000-0000-0000-000000000001"

    def _params(self, **overrides):
        defaults = dict(
            video_subject="money",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
        )
        defaults.update(overrides)
        return VideoParams(**defaults)

    def test_local_fill_gap_appends_stock_after_uploads(self):
        # 1 local clip (3s) but 10s of audio + fill enabled -> top up with stock.
        params = self._params(
            fill_with_stock=True,
            video_materials=[MaterialInfo(provider="local", url="local1.mp4")],
        )
        with mock.patch.object(
            tm.video, "preprocess_video",
            return_value=[MaterialInfo(provider="local", url="local1.mp4")],
        ), mock.patch.object(
            tm.material, "download_videos",
            return_value=["stock1.mp4", "stock2.mp4"],
        ) as dl:
            result = tm.get_video_materials(
                self.TASK_ID, params, ["money"], audio_duration=10
            )
        self.assertEqual(result, ["local1.mp4", "stock1.mp4", "stock2.mp4"])
        dl.assert_called_once()

    def test_local_no_fill_uses_uploads_only(self):
        params = self._params(
            fill_with_stock=False,
            video_materials=[MaterialInfo(provider="local", url="local1.mp4")],
        )
        with mock.patch.object(
            tm.video, "preprocess_video",
            return_value=[MaterialInfo(provider="local", url="local1.mp4")],
        ), mock.patch.object(tm.material, "download_videos") as dl:
            result = tm.get_video_materials(
                self.TASK_ID, params, ["money"], audio_duration=10
            )
        self.assertEqual(result, ["local1.mp4"])
        dl.assert_not_called()

    def test_local_no_uploads_falls_back_to_stock(self):
        # No uploads at all -> use stock for the full duration even without flag.
        params = self._params(fill_with_stock=False, video_materials=None)
        with mock.patch.object(tm.video, "preprocess_video") as pp, \
            mock.patch.object(
                tm.material, "download_videos", return_value=["s1.mp4"]
            ) as dl:
            result = tm.get_video_materials(
                self.TASK_ID, params, ["money"], audio_duration=10
            )
        self.assertEqual(result, ["s1.mp4"])
        pp.assert_not_called()
        dl.assert_called_once()

    def test_non_local_source_unchanged(self):
        params = self._params(video_source="pexels")
        with mock.patch.object(
            tm.material, "download_videos", return_value=["p1.mp4"]
        ) as dl:
            result = tm.get_video_materials(
                self.TASK_ID, params, ["money"], audio_duration=10
            )
        self.assertEqual(result, ["p1.mp4"])
        # full duration requested from the chosen source
        self.assertEqual(dl.call_args.kwargs["source"], "pexels")


class TestMaterialKeywords(unittest.TestCase):
    """Derive LLM content hints from uploaded file names."""

    def test_uses_name_field_and_normalizes(self):
        materials = [
            MaterialInfo(provider="local", url="/x/abc.mp4", name="Sunset_Over-The.Sea.mp4"),
        ]
        self.assertEqual(tm.get_material_keywords(materials), ["Sunset Over The Sea"])

    def test_strips_streamlit_file_id_prefix_from_url(self):
        url = "/storage/local_videos/123e4567-e89b-12d3-a456-426614174000_my_cat.jpg"
        materials = [MaterialInfo(provider="local", url=url)]
        self.assertEqual(tm.get_material_keywords(materials), ["my cat"])

    def test_dedupes_case_insensitively(self):
        materials = [
            MaterialInfo(provider="local", url="/a.mp4", name="Beach.mp4"),
            MaterialInfo(provider="local", url="/b.mp4", name="beach.mp4"),
        ]
        self.assertEqual(tm.get_material_keywords(materials), ["Beach"])

    def test_empty_and_none(self):
        self.assertEqual(tm.get_material_keywords(None), [])
        self.assertEqual(tm.get_material_keywords([]), [])

    def test_generate_script_passes_hints_to_llm(self):
        params = VideoParams(
            video_subject="cats",
            video_script="",
            video_materials=[
                MaterialInfo(provider="local", url="/a.mp4", name="fluffy_cat.mp4"),
            ],
        )
        with mock.patch.object(
            tm.llm, "generate_script", return_value="a script"
        ) as gen:
            result = tm.generate_script("tid", params)
        self.assertEqual(result, "a script")
        self.assertEqual(gen.call_args.kwargs["material_names"], ["fluffy cat"])


if __name__ == "__main__":
    unittest.main()