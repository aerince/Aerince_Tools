"""Offline checks: python -B tests/test_llm_responses.py (no ComfyUI/API needed)."""
import ast
import copy
import io
import json
import re
import time
import threading
import unittest
from contextlib import redirect_stdout
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
NAMES = {
    "_openai_headers", "_apply_thinking_options", "_request_chat_json",
    "_chat_error_message", "_chat_choice", "_chat_content_text",
    "_chat_response_summary", "_read_chat_response", "_extract_chat_text",
    "_clean_llm_prompt_text", "LLMImagePromptReverse", "LLMImagePromptReverseEditable",
    "LLMPromptOptimizer",
}


class Response(io.BytesIO):
    def __init__(self, value, content_type="application/json"):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())
        self.headers = {"Content-Type": content_type}


def answer(content="A quiet mountain lake.", finish="stop", **fields):
    return {"choices": [{"index": 0, "message": {"content": content, **fields}, "finish_reason": finish}]}


def event(delta=None, finish=None, **fields):
    chunk = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **fields}
    return ("data: " + json.dumps(chunk, ensure_ascii=False) + "\r\n\r\n").encode()


def http_error(message, status=400):
    return HTTPError("https://example.invalid/v1/chat/completions", status, "error", {},
                     io.BytesIO(json.dumps({"error": {"message": message}}).encode()))


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(globals(), THINKING_STRENGTH_OPTIONS=("auto", "off", "low", "medium", "high"))
        definitions = [node for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "nodes.py", "exec"), self.ns)
        self.requests = []
        self.logs = io.StringIO()

    def call(self, responses, **options):
        pending = iter(responses)

        def open_mock(request, timeout):
            self.requests.append(json.loads(request.data))
            result = next(pending)
            if isinstance(result, Exception):
                raise result
            return result

        self.ns["urlopen"] = open_mock
        payload = {"model": "test-vision", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE"}}
        ]}], **options}
        with redirect_stdout(self.logs):
            return self.ns["_request_chat_json"]("https://example.invalid/chat/completions", "secret-key", payload, 10)

    def test_json_and_removal_of_token_cap(self):
        result = self.call([Response(answer())], max_tokens=1)
        self.assertEqual(self.ns["_extract_chat_text"](result), "A quiet mountain lake.")
        self.assertNotIn("max_tokens", self.requests[0])
        self.assertTrue(self.requests[0]["stream"])

    def test_stream_reasoning_utf8_and_usage(self):
        stream = b": heartbeat\r\n\r\n" + event({"reasoning_content": "PRIVATE reasoning"})
        stream += event({"content": "山间 "}) + event({"content": [{"type": "text", "text": "lake."}]}, "stop")
        stream += b'data: {"choices": [], "usage": {"completion_tokens": 12}}\n\n'
        stream += b"data: [DONE]\n\n"
        result = self.call([Response(stream, "text/event-stream; charset=utf-8")])
        self.assertEqual(self.ns["_extract_chat_text"](result), "山间 lake.")
        self.assertEqual(result["usage"]["completion_tokens"], 12)
        self.assertNotIn("PRIVATE", self.logs.getvalue())
        self.assertNotIn("reasoning_content", result["choices"][0]["message"])

    def test_sse_header_missing_and_multiline_event(self):
        raw = b'data: {"choices": [\ndata: {"delta": {"content": "lake"}, "finish_reason": "stop"}]}\n\n'
        result = self.call([Response(raw)])
        self.assertEqual(self.ns["_extract_chat_text"](result), "lake")

    def test_empty_retries_original_image_once(self):
        result = self.call([Response(answer("", reasoning_content="Prompt: PRIVATE")), Response(answer())])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0], self.requests[1])
        self.assertTrue(self.ns["_extract_chat_text"](result))

    def test_repeated_empty_fails_without_reasoning_leak(self):
        with self.assertRaisesRegex(RuntimeError, "最终正文为空") as caught:
            self.call([Response(answer("", reasoning_content="Final prompt: PRIVATE")), Response(answer(""))])
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn("PRIVATE", str(caught.exception) + self.logs.getvalue())

    def test_truncation_refusal_and_tools_are_not_retried(self):
        for value in (answer("partial", "length"), answer("", refusal="no"),
                      answer("", "content_filter"), answer("", "tool_calls")):
            with self.subTest(value=value):
                self.requests.clear()
                with self.assertRaises(RuntimeError):
                    self.call([Response(value)])
                self.assertEqual(len(self.requests), 1)

    def test_incomplete_stream_never_returns_partial(self):
        with self.assertRaisesRegex(RuntimeError, "结束标记前中断"):
            self.call([Response(event({"content": "partial"}), "text/event-stream")])
        self.assertEqual(len(self.requests), 1)

    def test_stream_fallback_only_on_explicit_parameter_error(self):
        result = self.call([http_error("stream is not supported"), Response(answer())])
        self.assertFalse(self.requests[1]["stream"])
        self.assertTrue(self.ns["_extract_chat_text"](result))
        for message, status in (("invalid image", 400), ("stream unsupported", 401), ("invalid model", 400)):
            self.requests.clear()
            with self.assertRaisesRegex(RuntimeError, "HTTP"):
                self.call([http_error(message, status)])
            self.assertEqual(len(self.requests), 1)

    def test_thinking_fallback_preserves_explicit_control(self):
        self.call([http_error("unknown enable_thinking"), Response(answer())],
                  enable_thinking=False, thinking={"type": "disabled"})
        self.assertNotIn("enable_thinking", self.requests[1])
        self.assertIn("thinking", self.requests[1])
        self.requests.clear()
        with self.assertRaisesRegex(RuntimeError, "未静默切回 auto"):
            self.call([http_error("unknown enable_thinking")], enable_thinking=False)
        self.assertEqual(len(self.requests), 1)

    def test_shared_attempt_limit(self):
        with self.assertRaisesRegex(RuntimeError, "最终正文为空"):
            self.call([http_error("unknown enable_thinking"), Response(answer("")), Response(answer(""))],
                      enable_thinking=False, thinking={"type": "disabled"})
        self.assertEqual(len(self.requests), 3)

    def test_malformed_and_http200_errors(self):
        values = [Response(b"<html>bad gateway</html>"), Response({"error": {"message": "upstream failure"}}),
                  Response({"choices": []}), Response(event({"content": "x"}) + b"data: broken\n\n", "text/event-stream"),
                  Response(b'data: {"error": {"message": "upstream failure"}}\n\n', "text/event-stream")]
        for response in values:
            self.requests.clear()
            with self.assertRaises(RuntimeError):
                self.call([response])
            self.assertEqual(len(self.requests), 1)

    def test_only_final_content_is_extracted(self):
        extract = self.ns["_extract_chat_text"]
        self.assertEqual(extract(answer("", reasoning_content="Final prompt: fake")), "")
        self.assertEqual(extract(answer("<think>unfinished")), "")
        self.assertEqual(extract(answer("Thinking Process:\nPrompt: fake")), "")
        self.assertEqual(extract(answer("<think>private</think>\nPrompt: lake")), "lake")
        self.assertEqual(extract(answer([{"type": "reasoning", "text": "private"}, {"type": "text", "text": "lake"}])), "lake")
        self.assertEqual(extract(answer('A sign reading "Prompt: hello".')), 'A sign reading "Prompt: hello".')

    def test_all_llm_nodes_remove_widget_and_accept_legacy_keyword(self):
        for name in ("LLMImagePromptReverse", "LLMPromptOptimizer", "LLMImagePromptReverseEditable"):
            cls = self.ns[name]
            self.assertNotIn("max_tokens", cls.INPUT_TYPES()["optional"])
        self.ns["_llm_reverse_image_prompt_from_tensor"] = lambda *args, **kwargs: "lake"
        node = self.ns["LLMImagePromptReverse"]()
        self.assertEqual(node.reverse_prompt(None, "url", "key", "model", "", "", max_tokens=1), ("lake",))

    def test_auto_failure_does_not_return_old_prompt(self):
        self.ns["_image_tensor_fingerprint"] = lambda image: "new-hash"
        def fail(*args, **kwargs):
            raise RuntimeError("empty")
        self.ns["_llm_reverse_image_prompt_from_tensor"] = fail
        node = self.ns["LLMImagePromptReverseEditable"]()
        with self.assertRaisesRegex(RuntimeError, "empty"):
            node.output_prompt(None, "url", "key", "model", "", "", "old prompt",
                               auto_reverse=True, source_image_hash="old-hash")
        self.assertEqual(node.output_prompt(None, "url", "key", "model", "", "", "old prompt",
                                           auto_reverse=True, source_image_hash="new-hash"), ("old prompt",))

    def test_safe_error_message(self):
        message = self.ns["_chat_error_message"]({"message": "secret-key data:image/png;base64,AAAA Bearer another-key"}, "secret-key")
        self.assertNotIn("secret-key", message)
        self.assertNotIn("AAAA", message)
        self.assertNotIn("another-key", message)

    def test_real_local_http_stream(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in (b": heartbeat\n\n", event({"reasoning_content": "private"}),
                              event({"content": "Local "}), event({"content": "lake"}, "stop"), b"data: [DONE]\n\n"):
                    self.wfile.write(chunk)
                    self.wfile.flush()

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            server.timeout = 5
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                with redirect_stdout(self.logs):
                    result = self.ns["_request_chat_json"](
                        f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "",
                        {"model": "local-test", "messages": []}, 5)
                self.assertEqual(self.ns["_extract_chat_text"](result), "Local lake")
                self.assertNotIn("max_tokens", captured[0])
                self.assertTrue(captured[0]["stream"])
            finally:
                thread.join(timeout=6)

    def test_stream_deadline(self):
        with self.assertRaisesRegex(RuntimeError, "超时"):
            self.ns["_read_chat_response"](Response(event({"content": "late"}), "text/event-stream"), time.monotonic() - 1)


if __name__ == "__main__":
    unittest.main()
