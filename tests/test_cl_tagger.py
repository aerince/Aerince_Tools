import unittest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cl_tagger
from cl_tagger import AerinceCLTaggerV2, _preprocess, _preprocess_wd, _select_tags, _wd_vocabulary


class CLTaggerTests(unittest.TestCase):
    def test_preprocess_and_tag_selection(self):
        image = np.zeros((8, 4, 3), dtype=np.float32)
        model_input = _preprocess(image)
        self.assertEqual(model_input.shape, (1, 3, 384, 384))
        self.assertEqual(model_input.dtype, np.float32)
        vocabulary = {
            "idx_to_tag": {"0": "1girl", "1": "blue_hair", "2": "red_hair"}
        }
        text = _select_tags(
            np.array([2.0, 1.0, -2.0], dtype=np.float32),
            vocabulary,
            0.55,
            True,
            "1girl",
        )
        self.assertEqual(text, "blue hair")

    def test_wd_preprocess_pads_white_and_converts_to_bgr(self):
        from timm.data import create_transform

        transform = create_transform(
            input_size=(3, 448, 448), interpolation="bicubic", crop_pct=1.0,
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        )
        image = np.zeros((8, 4, 3), dtype=np.float32)
        image[:, :, 0] = 1.0
        model_input = _preprocess_wd(image, transform)
        self.assertEqual(tuple(model_input.shape), (1, 3, 448, 448))
        np.testing.assert_allclose(model_input[0, :, 224, 224].numpy(), [-1.0, -1.0, 1.0])
        np.testing.assert_allclose(model_input[0, :, 224, 0].numpy(), [1.0, 1.0, 1.0])

        rgba = np.zeros((8, 8, 4), dtype=np.float32)
        np.testing.assert_allclose(_preprocess_wd(rgba, transform).numpy(), 1.0)

    def test_wd_thresholds_skip_ratings_and_keep_csv_order(self):
        vocabulary = {
            "idx_to_tag": {"0": "general", "1": "character_low", "2": "blue_hair", "3": "character_high"},
            "categories": [9, 4, 0, 4],
        }
        text = _select_tags(
            np.array([5.0, 0.8, 1.0, 2.0], dtype=np.float32), vocabulary,
            0.6094, True, "", 0.75,
        )
        self.assertEqual(text, "character high, blue hair")
        self.assertEqual(
            _select_tags(np.array([5.0, 0.8, 1.0, 2.0]), vocabulary, 0.6094, False, "hair", 0.75),
            "character_high",
        )

    def test_wd_vocabulary_uses_row_order_not_tag_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_tags.csv"
            path.write_text("tag_id,name,category\n999,general,9\n123,blue_hair,0\n", encoding="utf-8")
            vocabulary = _wd_vocabulary(path, 2)
            self.assertEqual(vocabulary["idx_to_tag"], {"0": "general", "1": "blue_hair"})
            self.assertEqual(vocabulary["categories"], [9, 0])
            with self.assertRaisesRegex(RuntimeError, "model and vocabulary versions do not match"):
                _wd_vocabulary(path, 3)

    def test_model_menu_keeps_cl_and_offers_wd_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "v2_01a"
            model_dir.mkdir()
            for filename in ("model.onnx", "model.onnx.data", "model_vocabulary.json"):
                (model_dir / filename).touch()
            with patch.object(cl_tagger, "MODEL_DIR", root):
                names = list(cl_tagger._models())
                self.assertEqual(names, [cl_tagger.WD_MODEL_NAME, "v2_01a/model.onnx"])
        self.assertEqual(cl_tagger.NODE_DISPLAY_NAME_MAPPINGS["AerinceCLTaggerV2"], "Aerince Tagger")
        self.assertEqual(AerinceCLTaggerV2.RETURN_NAMES, ("tags", "tags_list"))

    def test_old_cl_workflow_does_not_need_new_character_threshold(self):
        import torch

        class Session:
            def run(self, outputs, inputs):
                self.outputs = outputs
                self.inputs = inputs
                return [np.array([[2.0, 1.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            vocabulary = {"num_tags": 2, "idx_to_tag": {"0": "1girl", "1": "blue_hair"}}
            (path.parent / "model_vocabulary.json").write_text(cl_tagger.json.dumps(vocabulary), encoding="utf-8")
            session = Session()
            with patch.object(cl_tagger, "_models", return_value={"v2_01a/model.onnx": path}), \
                    patch.object(cl_tagger, "_session", return_value=session):
                text, tags_list = AerinceCLTaggerV2().tag(
                    torch.zeros((2, 8, 4, 3)), "v2_01a/model.onnx", 0.55, "CPU", True, "",
                )
            self.assertEqual(tags_list, ["1girl, blue hair", "1girl, blue hair"])
            self.assertEqual(text, "\n".join(tags_list))
            self.assertEqual(session.outputs, ["logits"])
            self.assertEqual(session.inputs["pixel_values"].shape, (1, 3, 384, 384))


if __name__ == "__main__":
    unittest.main()
