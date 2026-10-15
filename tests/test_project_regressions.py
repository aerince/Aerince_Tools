import ast
import base64
import hashlib
import io
import json
import os
import re
import tempfile
import unittest
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request

import numpy as np
import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = ROOT / "nodes.py"
JS_PATH = ROOT / "js" / "lora_recipe_manager.js"


class ProjectRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = NODES_PATH.read_text(encoding="utf-8")
        cls.js = JS_PATH.read_text(encoding="utf-8")
        tree = ast.parse(cls.source)
        wanted = {
            "_pixel_fingerprint",
            "_data_url_fingerprint",
            "_maybe_json",
            "_split_trigger_text",
            "_dedupe_words",
            "_normalize_tag_key",
            "_filter_common_trigger_words",
            "_civitai_image_url",
            "_download_lora_preview",
            "_tensor_png_size",
            "_compress_tensor_to_byte_limit",
        }
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "base64": base64,
            "hashlib": hashlib,
            "io": io,
            "json": json,
            "re": re,
            "np": np,
            "torch": torch,
            "Image": Image,
            "ImageOps": ImageOps,
            "Request": Request,
            "os": os,
            "urlparse": urlparse,
            "uuid": uuid,
            "COMMON_NON_TRIGGER_TAGS": {"1girl"},
            "MAX_RECEIVER_IMAGE_SIZE": 8 * 1024 * 1024,
        }
        exec(compile(ast.Module(functions, type_ignores=[]), str(NODES_PATH), "exec"), namespace)
        cls.funcs = namespace

    def test_manual_image_hash_matches_rgb_pixels(self):
        pixels = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        self.assertEqual(
            self.funcs["_data_url_fingerprint"](data_url),
            self.funcs["_pixel_fingerprint"](pixels),
        )

    def test_receiver_compresses_image_under_limit(self):
        rng = np.random.default_rng(42)
        pixels = rng.random((1, 512, 512, 3), dtype=np.float32)
        image = torch.from_numpy(pixels)
        mask = torch.zeros((1, 512, 512), dtype=torch.float32)

        compressed, compressed_mask, encoded_size, changed = self.funcs["_compress_tensor_to_byte_limit"](
            image, mask, 128 * 1024
        )

        self.assertTrue(changed)
        self.assertLessEqual(encoded_size, 128 * 1024)
        self.assertLess(compressed.shape[1], image.shape[1])
        self.assertEqual(compressed.shape[1:3], compressed_mask.shape[1:3])
        self.assertEqual(torch.count_nonzero(compressed_mask).item(), 0)

    def test_frequency_tags_are_not_used_as_trigger_words(self):
        self.assertNotIn("def _collect_frequency_tags", self.source)
        self.assertNotIn('source = "tag_frequency"', self.source)
        self.assertNotIn('metadata.get("ss_tag_frequency")', self.source)

    def test_lora_api_and_cover_base_path_contracts(self):
        self.assertIn('"loras": names, "lora_details": details', self.source)
        self.assertIn("data.lora_details || data.loras", self.js)
        self.assertIn("api.apiURL(url)", self.js)
        # The lora list refresh lives in the chooser modal, not on the node.
        self.assertIn('"刷新列表"', self.js)
        self.assertNotIn('"刷新 LoRA/封面"', self.js)
        self.assertIn('"排序：创建时间（最新）", "排序：创建时间（最早）"', self.js)
        self.assertIn('"created_at": _lora_created_at(name)', self.source)

    def test_chooser_sort_mode_persists(self):
        self.assertIn("loadChooserSortMode(sortLabels.length)", self.js)
        self.assertIn("saveChooserSortMode(sortMode)", self.js)
        self.assertIn("aerince_lora_chooser_sort", self.js)

    def test_empty_trigger_words_are_preserved(self):
        self.assertIn('output_triggers = str(trigger_words or "").strip()', self.source)
        self.assertNotIn('or auto_triggers', self.source)

    def test_civitai_cover_is_cached_as_local_png(self):
        png = io.BytesIO()
        Image.new("RGB", (8, 4), "red").save(png, format="PNG")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self, _limit):
                return png.getvalue()

        original = self.funcs.get("urlopen")
        self.funcs["urlopen"] = lambda *_args, **_kwargs: Response()
        try:
            with tempfile.TemporaryDirectory() as directory:
                target = os.path.join(directory, "model.preview.png")
                result = self.funcs["_download_lora_preview"](
                    "https://image.civitai.com/example.png", target, 1
                )
                self.assertEqual(result, target)
                with Image.open(target) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.size, (8, 4))
        finally:
            if original is None:
                self.funcs.pop("urlopen", None)
            else:
                self.funcs["urlopen"] = original

    def test_civitai_cover_rejects_non_civitai_url(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported cover URL"):
                self.funcs["_download_lora_preview"](
                    "https://example.com/cover.png",
                    os.path.join(directory, "model.preview.png"),
                    1,
                )

    def test_civitai_image_url_skips_video(self):
        data = {"images": [
            {"type": "video", "url": "https://civitai.com/video.mp4"},
            {"type": "image", "url": "https://image.civitai.com/cover.jpeg"},
        ]}
        self.assertEqual(
            self.funcs["_civitai_image_url"](data),
            "https://image.civitai.com/cover.jpeg",
        )

    def test_recipe_overwrite_inherits_cover_and_meta(self):
        # Overwrite save must inherit the saved cover, never reset it.
        self.assertIn('if not recipe["cover"]:', self.source)
        self.assertIn('recipe["cover"] = str(existing.get("cover") or "")', self.source)
        # The save dialog must resolve the saved recipe by recipe_id before prefilling,
        # so a workflow reload cannot wipe description/tags with empty values.
        self.assertIn("currentRecipe.id !== currentId", self.js)
        self.assertIn("coverInfo.cover || currentRecipe.cover", self.js)

    def test_manager_has_new_recipe_action(self):
        # 新建画风 is a node button, not part of the manager modal.
        self.assertIn('addManagedButton(node, "新建画风"', self.js)
        self.assertIn("async function startNewRecipe", self.js)
        self.assertNotIn("aerince-manager-toolbar", self.js)

    def test_cover_route_uses_hash_fallback_and_local_cache(self):
        self.assertIn("_find_or_cache_lora_preview", self.source)
        self.assertIn("_cached_file_sha256(lora_path)", self.source)
        self.assertIn('os.path.splitext(lora_path)[0] + ".preview.png"', self.source)


if __name__ == "__main__":
    unittest.main()
