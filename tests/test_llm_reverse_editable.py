import ast
import unittest
from pathlib import Path


NODES_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
JS_PATH = Path(__file__).resolve().parents[1] / "js" / "bridge_receiver.js"


class LLMReverseEditableStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = NODES_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(NODES_PATH))
        cls.js = JS_PATH.read_text(encoding="utf-8")

    def _class(self, name):
        node = next(
            (
                item
                for item in self.tree.body
                if isinstance(item, ast.ClassDef) and item.name == name
            ),
            None,
        )
        self.assertIsNotNone(node, f"missing class {name}")
        return node

    def _function(self, name):
        node = next(
            (
                item
                for item in self.tree.body
                if isinstance(item, ast.FunctionDef) and item.name == name
            ),
            None,
        )
        self.assertIsNotNone(node, f"missing function {name}")
        return node

    def test_helpers_are_module_level(self):
        self._function("_llm_reverse_image_prompt_from_data_url")
        self._function("_llm_reverse_image_prompt_from_tensor")

    def test_editable_node_exists_with_expected_fields(self):
        node = self._class("LLMImagePromptReverseEditable")
        method_names = {
            item.name for item in node.body if isinstance(item, ast.FunctionDef)
        }
        self.assertIn("INPUT_TYPES", method_names)
        self.assertIn("output_prompt", method_names)

        # Keep the key contract strings present in source.
        self.assertIn('"prompt_text"', self.source)
        self.assertIn('"auto_reverse"', self.source)
        self.assertIn('"source_image_hash"', self.source)
        self.assertIn('_image_tensor_fingerprint', self.source)
        self.assertIn('CATEGORY = "Aerince Tools/LLM"', self.source)
        self.assertIn(
            '"LLMImagePromptReverseEditable": LLMImagePromptReverseEditable',
            self.source,
        )
        self.assertIn(
            '"LLMImagePromptReverseEditable": "Aerince LLM 图像反推（可编辑）"',
            self.source,
        )

    def test_reverse_api_route_registered(self):
        self.assertIn('/aerince/llm/reverse-image', self.source)

    def test_frontend_has_manual_reverse_button_path(self):
        for needle in (
            "LLMImagePromptReverseEditable",
            "reverseImagePromptEditable",
            "reverse_now",
            "/aerince/llm/reverse-image",
            "setWidgetValue",
            "prompt_text",
        ):
            self.assertIn(needle, self.js)

    def test_nodes_py_compiles(self):
        compile(self.source, str(NODES_PATH), "exec")


if __name__ == "__main__":
    unittest.main()
