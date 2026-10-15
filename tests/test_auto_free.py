import ast
import sys
import types
import unittest
from pathlib import Path


NODES_PATH = Path(__file__).resolve().parents[1] / "nodes.py"


def load_auto_free_class():
    tree = ast.parse(NODES_PATH.read_text(encoding="utf-8"), filename=str(NODES_PATH))
    class_node = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AerinceAutoFreeModelAndNodeCache"),
        None,
    )
    if class_node is None:
        raise AssertionError("AerinceAutoFreeModelAndNodeCache is missing from nodes.py")

    module = ast.Module(body=[class_node], type_ignores=[])
    namespace = {"any_type": "*"}
    exec(compile(module, str(NODES_PATH), "exec"), namespace)
    return namespace["AerinceAutoFreeModelAndNodeCache"]


class FakePromptQueue:
    def __init__(self):
        self.flags = {}

    def set_flag(self, name, value):
        self.flags[name] = value


class AutoFreeNodeTests(unittest.TestCase):
    def setUp(self):
        self.queue = FakePromptQueue()
        fake_server = types.ModuleType("server")
        fake_server.PromptServer = types.SimpleNamespace(
            instance=types.SimpleNamespace(prompt_queue=self.queue)
        )
        self.old_server = sys.modules.get("server")
        sys.modules["server"] = fake_server

    def tearDown(self):
        if self.old_server is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = self.old_server

    def test_enabled_requests_same_cleanup_as_free_model_and_node_cache(self):
        node_class = load_auto_free_class()
        result = node_class().free_after_prompt(enabled=True, trigger="done")

        self.assertEqual({"unload_models": True, "free_memory": True}, self.queue.flags)
        self.assertEqual(("done",), result)
        self.assertTrue(node_class.OUTPUT_NODE)

    def test_disabled_is_passthrough_without_cleanup_request(self):
        node_class = load_auto_free_class()
        result = node_class().free_after_prompt(enabled=False, trigger=123)

        self.assertEqual({}, self.queue.flags)
        self.assertEqual((123,), result)

    def test_is_changed_forces_execution_on_every_prompt(self):
        node_class = load_auto_free_class()
        first = node_class.IS_CHANGED()
        second = node_class.IS_CHANGED()

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
