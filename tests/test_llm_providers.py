"""Offline persistence and node wiring check: python -B tests/test_llm_providers.py."""
import ast
import base64
import importlib.util
import io
import json
import tempfile
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("llm_providers", ROOT / "llm_providers.py")
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)


def check():
    with tempfile.TemporaryDirectory(prefix="llm-providers-", dir=ROOT / "tests") as temporary:
        store.PROVIDER_FILE = Path(temporary) / "providers.json"
        assert store.list_llm_providers() == []
        first = store.save_llm_provider({"name": "Provider A", "base_url": "https://a.invalid/v1/", "api_key": "test-secret-a"})
        second = store.save_llm_provider({"name": "Provider B", "base_url": "http://127.0.0.1:1234/v1", "api_key": "test-secret-b"})
        ref = store.PROVIDER_PREFIX + first["id"]
        assert first["has_key"] and "api_key" not in first
        assert "test-secret" not in json.dumps(store.list_llm_providers())
        assert store.resolve_llm_provider(ref, "stale-key") == ("https://a.invalid/v1", "test-secret-a")
        assert store.resolve_llm_provider(store.PROVIDER_PREFIX + second["id"]) == ("http://127.0.0.1:1234/v1", "test-secret-b")
        assert store.resolve_llm_provider("https://legacy.invalid", "legacy-key") == ("https://legacy.invalid", "legacy-key")
        spec.loader.exec_module(reloaded := importlib.util.module_from_spec(spec))
        reloaded.PROVIDER_FILE = store.PROVIDER_FILE
        assert reloaded.list_llm_providers() == [first, second], "survives module reload"
        saved = store.save_llm_provider({"id": first["id"], "name": "Renamed A", "base_url": "https://a.invalid/v1"})
        assert saved["has_key"] and store.resolve_llm_provider(ref)[1] == "test-secret-a"
        store.save_llm_provider({**saved, "api_key": ""})
        assert store.resolve_llm_provider(ref)[1] == ""
        store.save_llm_provider({**saved, "api_key": "test-secret-new"})
        original = store.PROVIDER_FILE.read_bytes()
        for bad in ([], {}, {**saved, "id": "missing"}, {**saved, "name": "Provider B"},
                    {**saved, "base_url": "file:///secret"}, {**saved, "base_url": "https://x:99999"},
                    {**saved, "base_url": "https://user:secret@x"}, {**saved, "api_key": []}):
            try:
                store.save_llm_provider(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid provider accepted")
            assert store.PROVIDER_FILE.read_bytes() == original
        try:
            store.resolve_llm_provider(store.PROVIDER_PREFIX + "missing")
        except ValueError as error:
            assert "重新选择" in str(error)
        else:
            raise AssertionError("missing reference accepted")
        with patch.object(store.os, "replace", side_effect=OSError("simulated write failure")):
            try:
                store.save_llm_provider({**saved, "api_key": "must-not-save"})
            except OSError:
                pass
            else:
                raise AssertionError("failed write reported success")
        assert store.PROVIDER_FILE.read_bytes() == original
        assert not list(Path(temporary).glob("*.tmp"))

        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        names = {"_fetch_llm_models", "_normalize_openai_base_url", "_llm_reverse_image_prompt_from_data_url",
                 "_llm_reverse_image_prompt_from_tensor", "LLMPromptOptimizer", "LLMImagePromptReverse",
                 "LLMImagePromptReverseEditable"}
        calls = []
        def capture(url, api_key, **kwargs):
            calls.append((url, api_key))
            return {"data": [{"id": "test-model"}]}
        ns = {
            "resolve_llm_provider": store.resolve_llm_provider,
            "_request_json": capture, "_request_chat_json": capture,
            "_model_supports_thinking": lambda *args: False,
            "_apply_thinking_options": lambda *args: None,
            "_extract_chat_text": lambda data: "test prompt",
            "base64": base64, "io": io, "json": json,
            "Image": SimpleNamespace(open=lambda data: nullcontext(SimpleNamespace(format="PNG", width=1, height=1, verify=lambda: None))),
            "_image_tensor_to_data_url": lambda image: "data:image/png;base64,AA==",
            "_image_tensor_fingerprint": lambda image: "fresh-hash",
        }
        definitions = [item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in names]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "nodes.py", "exec"), ns)
        assert ns["_fetch_llm_models"](ref)[0] == ["test-model"]
        assert calls[-1] == ("https://a.invalid/v1/models", "test-secret-new")
        with redirect_stdout(io.StringIO()):
            assert ns["_llm_reverse_image_prompt_from_data_url"]("data:image/png;base64,AA==", ref, "", "m", "", "") == "test prompt"
            assert ns["LLMImagePromptReverse"]().reverse_prompt(None, ref, "", "m", "", "") == ("test prompt",)
            result = ns["LLMImagePromptReverseEditable"]().output_prompt(None, ref, "", "m", "", "", "old", auto_reverse=True)
            assert result["result"] == ("test prompt",)
        assert ns["LLMPromptOptimizer"]().optimize_prompt("input", ref, "", "m", "", "") == ("test prompt",)
        assert calls[1:] == [("https://a.invalid/v1/chat/completions", "test-secret-new")] * 4

        store.PROVIDER_FILE.write_text("{broken", encoding="utf-8")
        try:
            store.save_llm_provider({"name": "C", "base_url": "https://c.invalid"})
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt store was overwritten")
        assert store.PROVIDER_FILE.read_text(encoding="utf-8") == "{broken"
    print("PASS: persistence/reload, redaction, update/clear, validation, atomic failure, legacy inputs, all four LLM paths, corrupt store preservation")


if __name__ == "__main__":
    check()
