"""Offline lifecycle checks: python -B tests/test_local_llm.py."""

import ast
import base64
import importlib.util
import io
import json
import re
import sys
import tempfile
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch


def fake_server():
    def option(name):
        return sys.argv[sys.argv.index(name) + 1]

    mode = Path(option("--model")).stem
    if mode == "exit":
        print("FAKE_LOAD_FAILURE", flush=True)
        return

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_json({"status": "loading" if mode == "loading" else "ok"},
                           503 if mode == "loading" else 200)

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + option("--api-key"):
                self.send_json({"error": "bad key"}, 401)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            assert payload["stream"] is False
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            assert payload["max_tokens"] == 1024
            user_content = payload["messages"][1]["content"]
            if isinstance(user_content, list):
                assert "--mmproj" in sys.argv
                data_url = user_content[0]["image_url"]["url"]
                assert base64.b64decode(data_url.split(",", 1)[1]).startswith(b"\x89PNG")
                answer = "Prompt: A red square on white."
            else:
                assert "--mmproj" not in sys.argv
                assert isinstance(user_content, str)
                answer = "处理结果：" + user_content
            if mode == "slow":
                print("FAKE_INFERENCE_STARTED", flush=True)
                time.sleep(10)
            content = "" if mode == "empty" else "<think>PRIVATE</think>\n" + answer
            self.send_json({"choices": [{"index": 0, "message": {"content": content},
                                        "finish_reason": "length" if mode == "truncated" else "stop"}]})

    HTTPServer(("127.0.0.1", int(option("--port"))), Handler).serve_forever()


if __name__ == "__main__" and "--fake-server" in sys.argv:
    fake_server()
    sys.exit(0)


import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "aerince_local_llm_test"


class FakeTensor:
    def __init__(self, pixels):
        self.pixels = pixels
        self.shape = pixels.shape

    def __getitem__(self, key):
        return FakeTensor(self.pixels[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.pixels


class Interrupted(BaseException):
    pass


class LocalLLMTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path("F:/pip-temp/opencode")
        self.temp = tempfile.TemporaryDirectory(dir=str(temp_root) if temp_root.is_dir() else None)
        self.addCleanup(self.temp.cleanup)
        self.models = Path(self.temp.name) / "LLM"
        self.models.mkdir()
        for name in ("qwen.gguf", "mmproj-qwen.gguf"):
            (self.models / name).touch()
        folders = types.ModuleType("folder_paths")
        folders.models_dir = self.temp.name
        folders.get_folder_paths = lambda name: [str(self.models)]
        folders.recursive_search = lambda root, **kwargs: (
            [str(path.relative_to(root)) for path in Path(root).rglob("*") if path.is_file()], {}
        )
        management = types.ModuleType("comfy.model_management")
        management.throw_exception_if_processing_interrupted = lambda: None
        management.unload_all_models = lambda: None
        management.soft_empty_cache = lambda: None
        comfy = types.ModuleType("comfy")
        comfy.model_management = management
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        helpers = types.ModuleType(PACKAGE + ".nodes")
        helpers.__dict__.update(base64=base64, io=io, np=np, Image=Image, re=re)
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        names = {"_image_tensor_to_data_url", "_extract_chat_text", "_chat_choice",
                 "_chat_content_text", "_clean_llm_prompt_text"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "nodes.py", "exec"), helpers.__dict__)
        modules = {"folder_paths": folders, "comfy": comfy, "comfy.model_management": management,
                   PACKAGE: package, PACKAGE + ".nodes": helpers}
        stub_modules = patch.dict(sys.modules, modules)
        stub_modules.start()
        self.addCleanup(stub_modules.stop)
        spec = importlib.util.spec_from_file_location(PACKAGE + ".local_llm", ROOT / "local_llm.py")
        self.local = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.local)
        self.server = self.local._SERVER
        self.addCleanup(self.server.stop)
        self.processes = []
        self.commands = []
        real_popen = self.local.subprocess.Popen

        def launch(command, **options):
            self.commands.append(command)
            process = real_popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                  "--fake-server", *command[1:]], **options)
            self.processes.append(process)
            return process

        mocked_popen = patch.object(self.local.subprocess, "Popen", launch)
        mocked_popen.start()
        self.addCleanup(mocked_popen.stop)
        self.image = FakeTensor(np.ones((1, 4, 4, 3), dtype=np.float32))

    def call(self, model="qwen.gguf", **options):
        return self.local.AerinceLocalLLMImagePromptReverse().reverse_prompt(
            self.image, model, "mmproj-qwen.gguf", llama_server_path=sys.executable,
            **options,
        )

    def call_text(self, text="这是待处理文本。", model="qwen.gguf", **options):
        return self.local.AerinceLocalLLMTextProcess().process_text(
            text, model, llama_server_path=sys.executable, **options,
        )

    def assert_stopped(self):
        self.assertIsNone(self.server.process)
        self.assertTrue(all(process.poll() is not None for process in self.processes))

    def test_dropdowns_include_extra_paths_and_separate_projectors(self):
        extra = Path(self.temp.name) / "extra"
        extra.mkdir()
        (extra / "nested").mkdir()
        for name in ("nested/other.GGUF", "nested/mmproj-other.GGUF",
                     "split-00001-of-00002.gguf", "split-00002-of-00002.gguf"):
            (extra / name).touch()
        self.local.folder_paths.get_folder_paths = lambda name: [str(self.models), str(extra)]
        inputs = self.local.AerinceLocalLLMImagePromptReverse.INPUT_TYPES()["required"]
        self.assertIn("nested/other.GGUF", inputs["model"][0])
        self.assertIn("nested/mmproj-other.GGUF", inputs["mmproj"][0])
        self.assertNotIn("mmproj-qwen.gguf", inputs["model"][0])
        self.assertNotIn("split-00002-of-00002.gguf", inputs["model"][0])
        self.assertTrue(inputs["release_after_run"][1]["default"])

    def test_default_release_and_clean_final_prompt(self):
        self.assertEqual(self.call(), ("A red square on white.",))
        self.assert_stopped()
        command = self.commands[0]
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertEqual(command[command.index("--reasoning") + 1], "off")
        self.assertEqual(command[command.index("--mmproj") + 1], str(self.models / "mmproj-qwen.gguf"))

    def test_unsigned_32_bit_fixed_seeds_pass_through_and_reuse_cache(self):
        limits = self.local.AerinceLocalLLMImagePromptReverse.INPUT_TYPES()["optional"]["seed"][1]
        self.assertEqual(limits["min"], 0)
        self.assertEqual(limits["max"], 4294967294)
        response = {"choices": [{"index": 0, "message": {"content": "A red square."},
                                 "finish_reason": "stop"}]}
        with patch.object(self.server, "ensure"), patch.object(self.server, "chat", return_value=response) as chat:
            for seed in (2310270724, 4294967294):
                with self.subTest(seed=seed):
                    self.assertLessEqual(seed, limits["max"])
                    previous = chat.call_count
                    self.call(seed=seed)
                    self.assertEqual(chat.call_count, previous + 1)
                    self.assertEqual(chat.call_args.args[0]["seed"], seed)
                    self.call(seed=seed)
                    self.assertEqual(chat.call_count, previous + 1)

    def test_resident_reuse_then_release(self):
        self.call(release_after_run=False)
        process = self.server.process
        self.call(release_after_run=False)
        self.assertIs(self.server.process, process)
        self.assertEqual(len(self.processes), 1)
        self.call(release_after_run=True)
        self.assertEqual(len(self.processes), 1)
        self.assert_stopped()

    def test_identical_pixels_reuse_caption_after_release_and_new_node_instance(self):
        with patch.object(self.server, "chat", wraps=self.server.chat) as chat:
            first = self.call()
            self.assert_stopped()
            self.image = FakeTensor(self.image.pixels.copy())
            self.assertEqual(self.call(), first)
            self.assertEqual(chat.call_count, 1)
        self.assertEqual(len(self.processes), 1)
        self.assert_stopped()

    def test_changed_pixels_invalidate_caption_cache(self):
        self.call()
        self.image.pixels[0, 0, 0, 0] = 0
        self.call()
        self.assertEqual(len(self.processes), 2)
        self.assert_stopped()

    def test_caption_parameters_invalidate_cache_but_timeout_does_not(self):
        response = {"choices": [{"index": 0, "message": {"content": "A red square."},
                                 "finish_reason": "stop"}]}
        with patch.object(self.server, "ensure"), patch.object(self.server, "chat", return_value=response) as chat:
            self.call()
            self.call(timeout=30)
            self.assertEqual(chat.call_count, 1)
            for options in (
                {"system_prompt": "Describe the visible subject."},
                {"user_prompt": "Describe the lighting."},
                {"ctx_size": 4096}, {"max_tokens": 512},
                {"temperature": 0.5}, {"seed": 1}, {"gpu_layers": 0},
            ):
                with self.subTest(options=options):
                    self.call()
                    previous = chat.call_count
                    self.call(**options)
                    self.assertEqual(chat.call_count, previous + 1)
                    self.call(**options)
                    self.assertEqual(chat.call_count, previous + 1)

    def test_replaced_model_or_projector_invalidates_caption_cache(self):
        for filename in ("qwen.gguf", "mmproj-qwen.gguf"):
            with self.subTest(filename=filename):
                self.call()
                previous = len(self.processes)
                (self.models / filename).write_bytes(b"updated weights")
                self.call()
                self.assertEqual(len(self.processes), previous + 1)
        self.assert_stopped()

    def test_failed_caption_is_not_cached_and_keeps_last_success(self):
        first = self.call()
        (self.models / "empty.gguf").touch()
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "empty"):
                self.call(model="empty.gguf")
            self.assert_stopped()
        self.assertEqual(self.call(), first)
        self.assertEqual(len(self.processes), 3)
        self.assert_stopped()

    def test_loading_parameter_change_restarts_server(self):
        self.call(release_after_run=False)
        old = self.server.process
        self.call(ctx_size=4096, release_after_run=False)
        self.assertIsNotNone(old.poll())
        self.assertIsNot(old, self.server.process)
        self.assertEqual(len(self.processes), 2)

    def test_switching_model_restarts_server(self):
        (self.models / "other.gguf").touch()
        self.call(release_after_run=False)
        old = self.server.process
        self.call(model="other.gguf", release_after_run=False)
        self.assertIsNotNone(old.poll())
        self.assertEqual(len(self.processes), 2)

    def test_partial_or_empty_response_releases_even_in_resident_mode(self):
        for name, message in (("truncated", "truncated"), ("empty", "empty")):
            with self.subTest(name=name):
                (self.models / f"{name}.gguf").touch()
                with self.assertRaisesRegex(RuntimeError, message):
                    self.call(model=f"{name}.gguf", release_after_run=False)
                self.assert_stopped()

    def test_loading_failure_and_timeout_release(self):
        for name, error, message in (("exit", RuntimeError, "FAKE_LOAD_FAILURE"),
                                     ("loading", TimeoutError, "loading timed out")):
            with self.subTest(name=name):
                (self.models / f"{name}.gguf").touch()
                with self.assertRaisesRegex(error, message):
                    self.call(model=f"{name}.gguf", timeout=1, release_after_run=False)
                self.assert_stopped()

    def test_inference_timeout_releases(self):
        (self.models / "slow.gguf").touch()
        with self.assertRaisesRegex((TimeoutError, RuntimeError), "timed out|TimeoutError"):
            self.call(model="slow.gguf", timeout=1, release_after_run=False)
        self.assert_stopped()

    def test_cancel_during_inference_releases(self):
        (self.models / "slow.gguf").touch()

        def check_interrupt():
            if "FAKE_INFERENCE_STARTED" in self.server.log_tail:
                raise Interrupted()

        self.local.model_management.throw_exception_if_processing_interrupted = check_interrupt
        with self.assertRaises(Interrupted):
            self.call(model="slow.gguf", timeout=5, release_after_run=False)
        self.assert_stopped()

    def test_text_node_schema_has_no_image_or_projector(self):
        (self.models / "mmproj-qwen.gguf").unlink()
        cls = self.local.AerinceLocalLLMTextProcess
        inputs = cls.INPUT_TYPES()
        self.assertEqual(inputs["required"]["text"][0], "STRING")
        self.assertTrue(inputs["required"]["text"][1]["multiline"])
        self.assertFalse(inputs["required"]["text"][1]["dynamicPrompts"])
        self.assertEqual(inputs["required"]["model"][0], ["qwen.gguf"])
        self.assertTrue(inputs["required"]["release_after_run"][1]["default"])
        self.assertNotIn("image", inputs["required"])
        self.assertNotIn("mmproj", inputs["required"])
        self.assertNotIn("mmproj", inputs["optional"])
        self.assertEqual(cls.RETURN_TYPES, ("STRING",))
        self.assertEqual(cls.RETURN_NAMES, ("text",))
        self.assertEqual(cls.FUNCTION, "process_text")
        self.assertEqual(cls.CATEGORY, "Aerince Tools/LLM")
        self.assertEqual(inputs["optional"]["seed"][1]["max"], 4294967294)

    def test_text_node_is_imported_and_registered(self):
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        imported = next(node for node in tree.body if isinstance(node, ast.ImportFrom)
                        and node.module == "local_llm")
        self.assertIn("AerinceLocalLLMTextProcess", [name.name for name in imported.names])
        mappings = {node.targets[0].id: node.value for node in tree.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")}
        classes = {key.value: value for key, value in zip(
            mappings["NODE_CLASS_MAPPINGS"].keys, mappings["NODE_CLASS_MAPPINGS"].values)}
        self.assertEqual(classes["AerinceLocalLLMTextProcess"].id, "AerinceLocalLLMTextProcess")
        display = ast.literal_eval(mappings["NODE_DISPLAY_NAME_MAPPINGS"])
        self.assertEqual(display["AerinceLocalLLMTextProcess"], "Aerince 本地大模型文本处理")

    def test_text_default_release_without_projector(self):
        (self.models / "mmproj-qwen.gguf").unlink()
        expected = "处理结果：" + self.local.TEXT_USER_PROMPT + "\n\n这是待处理文本。"
        self.assertEqual(self.call_text(), (expected,))
        self.assert_stopped()
        command = self.commands[0]
        self.assertNotIn("--mmproj", command)
        self.assertEqual(command[command.index("--model") + 1], str(self.models / "qwen.gguf"))
        self.assertEqual(command[command.index("--reasoning") + 1], "off")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")

    def test_text_unicode_and_instructions_payload(self):
        response = {"choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}]}
        text = "原文：中文、日本語、emoji 😊\n保留 {花括号} 与换行"
        with patch.object(self.server, "ensure") as ensure, patch.object(
                self.server, "chat", return_value=response) as chat:
            self.assertEqual(self.call_text(text=text, system_prompt="你是翻译助手。",
                                           user_prompt="翻译为英文。", seed=4294967294), ("译文",))
            payload = chat.call_args.args[0]
            self.assertEqual(payload["messages"], [
                {"role": "system", "content": "你是翻译助手。"},
                {"role": "user", "content": "翻译为英文。\n\n" + text},
            ])
            self.assertEqual(payload["seed"], 4294967294)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertFalse(payload["stream"])
            self.assertIsNone(ensure.call_args.args[2])

    def test_text_blank_instructions_send_text_directly(self):
        response = {"choices": [{"message": {"content": "回答"}, "finish_reason": "stop"}]}
        with patch.object(self.server, "ensure"), patch.object(
                self.server, "chat", return_value=response) as chat:
            self.call_text(text="直接提问", system_prompt="", user_prompt="")
            self.assertEqual(chat.call_args.args[0]["messages"], [
                {"role": "system", "content": ""}, {"role": "user", "content": "直接提问"},
            ])

    def test_text_preserves_markdown_labels_and_indentation(self):
        outputs = [
            "Prompt: 保留此标签。", "Final answer: 保留此标签。", "分析：这是需要的分析结果。",
            "Analysis: requested analysis", "```text\n带格式的正文\n```",
            "```python\n    print('保留缩进')\n```", "    缩进文本\n\n第二段\n",
            "正文中的 <think>标签示例</think> 不应被改写。",
        ]
        with patch.object(self.server, "ensure"), patch.object(self.server, "chat") as chat:
            for seed, output in enumerate(outputs):
                with self.subTest(output=output):
                    chat.return_value = {"choices": [{"message": {"content": output}, "finish_reason": "stop"}]}
                    self.assertEqual(self.call_text(seed=seed), (output,))

    def test_text_uses_final_content_not_reasoning(self):
        responses = [
            ({"content": "<think>PRIVATE</think>\n<reasoning>PRIVATE</reasoning>\nPrompt: 正文"}, "Prompt: 正文"),
            ({"content": "<think>PRIVATE</think>\n    保留缩进"}, "    保留缩进"),
            ({"content": [{"type": "reasoning", "text": "PRIVATE"},
                          {"type": "text", "text": "第一段\n"},
                          {"type": "output_text", "text": "第二段"}],
              "reasoning_content": "PRIVATE", "analysis": "PRIVATE"}, "第一段\n第二段"),
        ]
        with patch.object(self.server, "ensure"), patch.object(self.server, "chat") as chat:
            for seed, (message, expected) in enumerate(responses):
                chat.return_value = {"choices": [{"message": message, "finish_reason": "stop"}]}
                self.assertEqual(self.call_text(seed=seed), (expected,))

    def test_text_invalid_input_never_launches_server(self):
        for text in (None, "", " \n\t", 123):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "不能为空"):
                self.call_text(text=text)
        for model in ("missing.gguf", "mmproj-qwen.gguf"):
            with self.subTest(model=model), self.assertRaises(FileNotFoundError):
                self.call_text(model=model)
        for max_tokens in (8192, 16384):
            with self.subTest(max_tokens=max_tokens), self.assertRaisesRegex(ValueError, "max_tokens"):
                self.call_text(ctx_size=8192, max_tokens=max_tokens)
        self.assertEqual(self.commands, [])
        self.assert_stopped()

    def test_text_rejects_incomplete_malformed_and_nontext_answers(self):
        responses = [None, [], {}, {"error": "failed"}, {"choices": []},
                     {"choices": [{"finish_reason": "stop"}]}]
        for content, finish, fields in (
            ("", "stop", {}), (" \n", "stop", {}), ("partial", "length", {}),
            ("partial", "max_tokens", {}), ("answer", None, {}),
            ("answer", "tool_calls", {}), ("answer", "stop", {"tool_calls": [{}]}),
            ("answer", "stop", {"function_call": {"name": "tool"}}),
            ("answer", "stop", {"refusal": "no"}), ("answer", "content_filter", {}),
            ([{"type": "refusal", "text": "no"}, {"type": "text", "text": "partial"}], "stop", {}),
            ("", "stop", {"reasoning_content": "PRIVATE"}),
            ([{"type": "reasoning", "text": "PRIVATE"}], "stop", {}),
            ("<think>PRIVATE</think>", "stop", {}), ("<think>unfinished", "stop", {}),
        ):
            responses.append({"choices": [{"message": {"content": content, **fields}, "finish_reason": finish}]})
        with patch.object(self.server, "ensure"), patch.object(self.server, "chat") as chat, patch.object(
                self.server, "stop", wraps=self.server.stop) as stop:
            for response in responses:
                with self.subTest(response=response):
                    chat.return_value = response
                    with self.assertRaises(RuntimeError):
                        self.call_text(release_after_run=False)
                    self.assertIsNone(self.local._TEXT_CACHE)
            self.assertEqual(stop.call_count, len(responses))

    def test_text_cache_reuses_after_release_and_new_node_instance(self):
        with patch.object(self.server, "chat", wraps=self.server.chat) as chat:
            first = self.call_text()
            self.assert_stopped()
            self.assertEqual(self.call_text(), first)
            self.assertEqual(chat.call_count, 1)
        self.assertEqual(len(self.processes), 1)
        self.assert_stopped()

    def test_text_resident_reuse_then_release(self):
        self.call_text(release_after_run=False)
        process = self.server.process
        self.call_text(text="另一段文本", release_after_run=False)
        self.assertIs(self.server.process, process)
        self.assertEqual(len(self.processes), 1)
        self.call_text(text="另一段文本", release_after_run=True)
        self.assertEqual(len(self.processes), 1)
        self.assert_stopped()

    def test_text_parameters_invalidate_cache_but_timeout_does_not(self):
        response = {"choices": [{"message": {"content": "正文"}, "finish_reason": "stop"}]}
        with patch.object(self.server, "ensure"), patch.object(self.server, "chat", return_value=response) as chat:
            self.call_text()
            self.call_text(timeout=30)
            self.assertEqual(chat.call_count, 1)
            for options in (
                {"text": "另一段文本"}, {"system_prompt": "你是翻译助手。"}, {"user_prompt": "翻译为英文。"},
                {"ctx_size": 4096}, {"max_tokens": 512}, {"temperature": 0.5},
                {"seed": 2310270724}, {"seed": 4294967294}, {"gpu_layers": 0},
            ):
                with self.subTest(options=options):
                    self.call_text()
                    previous = chat.call_count
                    self.call_text(**options)
                    self.assertEqual(chat.call_count, previous + 1)
                    self.call_text(**options)
                    self.assertEqual(chat.call_count, previous + 1)

    def test_text_replaced_model_invalidates_cache(self):
        self.call_text()
        (self.models / "qwen.gguf").write_bytes(b"updated weights")
        self.call_text()
        self.assertEqual(len(self.processes), 2)
        self.assert_stopped()

    def test_text_cache_is_independent_of_image_cache(self):
        self.call()
        caption_cache = self.local._CAPTION_CACHE
        self.call_text()
        text_cache = self.local._TEXT_CACHE
        self.call()
        self.call_text()
        self.assertIs(self.local._CAPTION_CACHE, caption_cache)
        self.assertIs(self.local._TEXT_CACHE, text_cache)
        self.assertEqual(len(self.processes), 2)
        self.assert_stopped()

    def test_text_and_image_modes_share_one_server(self):
        self.call(release_after_run=False)
        image_process = self.server.process
        self.call_text(release_after_run=False)
        text_process = self.server.process
        self.assertIsNotNone(image_process.poll())
        self.assertNotIn("--mmproj", self.commands[-1])
        self.image.pixels[0, 0, 0, 0] = 0
        self.call(release_after_run=False)
        self.assertIsNotNone(text_process.poll())
        self.assertIn("--mmproj", self.commands[-1])
        self.assertEqual(sum(process.poll() is None for process in self.processes), 1)
        self.server.stop()
        self.assert_stopped()

    def test_text_failed_answer_is_not_cached_and_keeps_last_success(self):
        first = self.call_text()
        (self.models / "empty.gguf").touch()
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "文本为空"):
                self.call_text(model="empty.gguf", release_after_run=False)
            self.assert_stopped()
        self.assertEqual(self.call_text(), first)
        self.assertEqual(len(self.processes), 3)
        self.assert_stopped()

    def test_text_partial_or_empty_answer_releases_resident_server(self):
        for name, message in (("truncated", "截断"), ("empty", "文本为空")):
            with self.subTest(name=name):
                (self.models / f"{name}.gguf").touch()
                with self.assertRaisesRegex(RuntimeError, message):
                    self.call_text(model=f"{name}.gguf", release_after_run=False)
                self.assert_stopped()

    def test_text_loading_failure_and_timeout_release(self):
        for name, error, message in (("exit", RuntimeError, "FAKE_LOAD_FAILURE"),
                                     ("loading", TimeoutError, "loading timed out")):
            with self.subTest(name=name):
                (self.models / f"{name}.gguf").touch()
                with self.assertRaisesRegex(error, message):
                    self.call_text(model=f"{name}.gguf", timeout=1, release_after_run=False)
                self.assert_stopped()

    def test_text_inference_timeout_releases(self):
        (self.models / "slow.gguf").touch()
        with self.assertRaisesRegex((TimeoutError, RuntimeError), "timed out|TimeoutError"):
            self.call_text(model="slow.gguf", timeout=1, release_after_run=False)
        self.assert_stopped()

    def test_text_cancellation_releases(self):
        (self.models / "slow.gguf").touch()

        def check_interrupt():
            if "FAKE_INFERENCE_STARTED" in self.server.log_tail:
                raise Interrupted()

        self.local.model_management.throw_exception_if_processing_interrupted = check_interrupt
        with self.assertRaises(Interrupted):
            self.call_text(model="slow.gguf", timeout=5, release_after_run=False)
        self.assert_stopped()


if __name__ == "__main__":
    unittest.main()
