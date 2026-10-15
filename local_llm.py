"""Local GGUF vision captioning and text processing with a node-owned llama.cpp server."""

import atexit
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import folder_paths
import comfy.model_management as model_management


PLUGIN_DIR = Path(__file__).resolve().parent
MODEL_ALIAS = "aerince-local-vision"
NO_MODELS = "GGUF models not found"
NO_PROJECTORS = "mmproj files not found"
SYSTEM_PROMPT = (
    "You are an expert image prompt engineer. Describe only what is visible: "
    "subject, appearance, pose, composition, style, lighting, colors, background "
    "and notable details. Return only a detailed English image generation prompt, "
    "without thinking, reasoning, explanations, labels or markdown."
)
TEXT_SYSTEM_PROMPT = (
    "你是一名文本处理助手。根据用户的处理指令对输入文本进行翻译、润色、总结、扩写、改写或格式整理。"
    "保留原意，除非指令另有要求，否则保留原文语言与格式。"
    "只输出处理结果，不输出思考过程、前言或额外解释。"
)
TEXT_USER_PROMPT = "润色以下文本，保留原意，只输出处理后的文本。"


def _model_roots():
    try:
        roots = folder_paths.get_folder_paths("LLM")
    except KeyError:
        roots = []
    roots = [*roots, str(Path(folder_paths.models_dir) / "LLM")]
    unique = {}
    for root in roots:
        path = os.path.abspath(root)
        unique.setdefault(os.path.normcase(path), path)
    return list(unique.values())


def _models():
    models, projectors = {}, {}
    for root in _model_roots():
        files, _ = folder_paths.recursive_search(root, excluded_dir_names=[".git"])
        for name in sorted(files):
            if Path(name).suffix.casefold() != ".gguf":
                continue
            shard = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name, re.I)
            if shard and shard.group(1) != "00001":
                continue
            filename = Path(name).name.casefold()
            target = projectors if "mmproj" in filename or "projector" in filename else models
            target.setdefault(Path(name).as_posix(), str(Path(root) / name))
    return models, projectors


def _server_executable(value):
    value = (value or os.environ.get("AERINCE_LLAMA_SERVER", "")).strip().strip('"')
    candidates = [value] if value else [
        str(PLUGIN_DIR / "bin" / "llama-server.exe"),
        str(PLUGIN_DIR / "bin" / "llama-server"),
        shutil.which("llama-server.exe") or shutil.which("llama-server") or "",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "llama-server executable not found. Set llama_server_path to a recent "
        "llama.cpp CUDA build with Qwen3.5 vision support, or put its complete "
        f"binary distribution in {PLUGIN_DIR / 'bin'}."
    )


def _request_json(url, api_key, payload=None, timeout=2):
    headers = {"Authorization": f"Bearer {api_key}"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers)
    # Loopback inference must not pass through the user's HTTP proxy.
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


class _LocalServer:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.key = None
        self.port = None
        self.api_key = ""
        self.log_tail = deque(maxlen=40)
        self.log_thread = None

    def stop(self):
        with self.lock:
            process = self.process
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if self.log_thread is not None:
                self.log_thread.join(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
            self.process = self.key = self.port = None
            self.log_thread = None
            self.api_key = ""
            print("[Aerince Local LLM] Server released.")

    def _read_logs(self, process):
        for line in process.stdout:
            self.log_tail.append(line.rstrip())

    def _diagnostic(self):
        if self.log_thread is not None and self.process.poll() is not None:
            self.log_thread.join(timeout=1)
        return "\n".join(self.log_tail)[-6000:].replace(self.api_key, "[redacted]")

    def ensure(self, executable, model, mmproj, ctx_size, gpu_layers, deadline):
        key = (executable, model, mmproj, int(ctx_size), int(gpu_layers))
        if self.key == key and self.process is not None and self.process.poll() is None:
            return
        self.stop()
        model_management.throw_exception_if_processing_interrupted()
        if gpu_layers != 0:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.api_key = secrets.token_urlsafe(24)
        command = [
            executable, "--model", model,
            "--host", "127.0.0.1", "--port", str(self.port),
            "--ctx-size", str(ctx_size), "--n-gpu-layers", str(gpu_layers),
            "--parallel", "1", "--jinja", "--reasoning", "off",
            "--alias", MODEL_ALIAS, "--api-key", self.api_key,
        ]
        self.log_tail.clear()
        if mmproj:
            command.extend(["--mmproj", mmproj])
        model_label = Path(model).name + (f" + {Path(mmproj).name}" if mmproj else "")
        print(f"[Aerince Local LLM] Loading {model_label}")
        self.process = subprocess.Popen(
            command, cwd=str(Path(executable).parent), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.log_thread = threading.Thread(target=self._read_logs, args=(self.process,), daemon=True)
        self.log_thread.start()
        while True:
            model_management.throw_exception_if_processing_interrupted()
            if self.process.poll() is not None:
                raise RuntimeError("llama-server exited during loading:\n" + self._diagnostic())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Local LLM model loading timed out.\n" + self._diagnostic())
            try:
                health = _request_json(
                    f"http://127.0.0.1:{self.port}/health", self.api_key,
                    timeout=min(1, remaining),
                )
                if health.get("status") == "ok":
                    self.key = key
                    return
            except (HTTPError, URLError, OSError, ValueError):
                pass
            time.sleep(min(0.1, remaining))

    def chat(self, payload, deadline):
        completed = threading.Event()
        result = {}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Local LLM inference timed out.")
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        api_key = self.api_key

        def request():
            try:
                result["data"] = _request_json(url, api_key, payload, timeout=remaining)
            except Exception as error:
                result["error"] = error
            finally:
                completed.set()

        # Poll cancellation while the non-streaming HTTP request runs in a worker.
        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        while not completed.wait(0.1):
            model_management.throw_exception_if_processing_interrupted()
            if time.monotonic() >= deadline:
                raise TimeoutError("Local LLM inference timed out.")
            if self.process.poll() is not None:
                raise RuntimeError("llama-server exited during inference:\n" + self._diagnostic())
        model_management.throw_exception_if_processing_interrupted()
        if time.monotonic() >= deadline:
            raise TimeoutError("Local LLM inference timed out.")
        error = result.get("error")
        if isinstance(error, HTTPError):
            error.close()
            raise RuntimeError(f"Local llama-server HTTP {error.code}.\n" + self._diagnostic()) from None
        if error is not None:
            raise RuntimeError(
                f"Local LLM request failed ({type(error).__name__}); no partial output was used.\n"
                + self._diagnostic()
            ) from None
        return result["data"]


_SERVER = _LocalServer()
atexit.register(_SERVER.stop)
# Keep only text and fingerprints, independently of ComfyUI's node cache.
_CAPTION_CACHE = None
_TEXT_CACHE = None


def _extract_text_answer(data):
    from .nodes import _chat_choice, _chat_content_text

    if not isinstance(data, dict) or data.get("error"):
        raise RuntimeError("本地 LLM 未返回有效的文本回答。")
    choice, message = _chat_choice(data)
    if choice.get("finish_reason") in ("length", "max_tokens"):
        raise RuntimeError("本地 LLM 输出被截断，请增大 max_tokens 或缩短输入文本。")
    content = message.get("content")
    refusal = isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") == "refusal" for part in content
    )
    if (choice.get("finish_reason") != "stop" or message.get("tool_calls")
            or message.get("function_call") or message.get("refusal") or refusal):
        raise RuntimeError("本地 LLM 未正常完成文本回答。")
    text = _chat_content_text(content)
    # Only remove a leading reasoning block; keep labels, Markdown and indentation.
    while re.match(r"(?is)^\s*<(think|reasoning)>", text):
        block = re.match(r"(?is)^\s*<(think|reasoning)>.*?</\1>(?:\r?\n)*", text)
        if block is None:
            raise RuntimeError("本地 LLM 只返回了未完成的思考过程，没有最终文本。")
        text = text[block.end():]
    if not text.strip():
        raise RuntimeError("本地 LLM 最终文本为空，未将思考过程用作输出。")
    return text


class AerinceLocalLLMImagePromptReverse:
    @classmethod
    def INPUT_TYPES(cls):
        models, projectors = _models()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(models) or [NO_MODELS],),
                "mmproj": (list(projectors) or [NO_PROJECTORS],),
                "llama_server_path": ("STRING", {"default": "", "multiline": False}),
                "release_after_run": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release the local model after captioning. Disable to keep one server resident.",
                }),
                "system_prompt": ("STRING", {"default": SYSTEM_PROMPT, "multiline": True}),
                "user_prompt": ("STRING", {
                    "default": "Reverse this image into a detailed image generation prompt.",
                    "multiline": True,
                }),
            },
            "optional": {
                "ctx_size": ("INT", {"default": 8192, "min": 2048, "max": 131072, "step": 1024}),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFE,
                    "tooltip": "Fixed unsigned 32-bit seed. 4294967295 is reserved by llama.cpp. Keep this value fixed to reuse captions."}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200,
                    "tooltip": "-1: all model layers on GPU; 0: CPU text model. mmproj uses GPU by default."}),
                "timeout": ("INT", {"default": 300, "min": 5, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "reverse_prompt"
    CATEGORY = "Aerince Tools/LLM"
    DESCRIPTION = "Local GGUF + mmproj vision captioning. One shared, node-owned llama.cpp server."

    def reverse_prompt(
        self, image, model, mmproj, llama_server_path="", release_after_run=True,
        system_prompt=SYSTEM_PROMPT, user_prompt="", ctx_size=8192, max_tokens=1024,
        temperature=0.2, seed=0, gpu_layers=-1, timeout=300,
    ):
        global _CAPTION_CACHE
        from .nodes import _image_tensor_to_data_url, _extract_chat_text, _chat_choice

        with _SERVER.lock:
            succeeded = False
            try:
                model_management.throw_exception_if_processing_interrupted()
                models, projectors = _models()
                if model not in models or mmproj not in projectors:
                    raise FileNotFoundError(
                        "Select a GGUF model and its matching mmproj from models/LLM "
                        "or a configured LLM folder, then refresh the ComfyUI model list. "
                        "Searched: " + ", ".join(_model_roots())
                    )
                if image is None or image.shape[0] == 0:
                    raise ValueError("image input is required")
                if max_tokens >= ctx_size:
                    raise ValueError("max_tokens must be smaller than ctx_size to leave room for the image.")
                executable = _server_executable(llama_server_path)
                sources = []
                for path in (executable, models[model], projectors[mmproj]):
                    info = Path(path).stat()
                    sources.append((path, info.st_size, info.st_mtime_ns))
                images = tuple(
                    _image_tensor_to_data_url(image[index:index + 1])
                    for index in range(image.shape[0])
                )
                cache_key = (
                    tuple(hashlib.sha256(data_url.encode("ascii")).digest() for data_url in images),
                    tuple(sources), system_prompt or "", user_prompt or "Describe this image.",
                    int(ctx_size), int(max_tokens), float(temperature), int(seed), int(gpu_layers),
                )
                if _CAPTION_CACHE is not None and _CAPTION_CACHE[0] == cache_key:
                    succeeded = True
                    print("[Aerince Local LLM] Reusing cached caption; inference skipped.")
                    return _CAPTION_CACHE[1]
                deadline = time.monotonic() + int(timeout)
                _SERVER.ensure(executable, models[model], projectors[mmproj], ctx_size, gpu_layers, deadline)
                prompts = []
                for data_url in images:
                    payload = {
                        "model": MODEL_ALIAS,
                        "messages": [
                            {"role": "system", "content": system_prompt or ""},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {
                                    "url": data_url,
                                }},
                                {"type": "text", "text": user_prompt or "Describe this image."},
                            ]},
                        ],
                        "stream": False,
                        "max_tokens": int(max_tokens),
                        "temperature": float(temperature),
                        "seed": int(seed),
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    data = _SERVER.chat(payload, deadline)
                    if not isinstance(data, dict) or data.get("error"):
                        raise RuntimeError("Local LLM did not return a valid chat completion.")
                    choice, message = _chat_choice(data)
                    if choice.get("finish_reason") in ("length", "max_tokens"):
                        raise RuntimeError("Local LLM output was truncated. Increase max_tokens or shorten the prompt.")
                    if choice.get("finish_reason") != "stop" or message.get("tool_calls") or message.get("refusal"):
                        raise RuntimeError("Local LLM did not finish a text answer normally.")
                    text = _extract_chat_text(data)
                    if not text:
                        raise RuntimeError("Local LLM final prompt is empty; reasoning was not used as output.")
                    prompts.append(text)
                result = ("\n\n".join(prompts),)
                _CAPTION_CACHE = (cache_key, result)
                succeeded = True
                if not release_after_run:
                    print("[Aerince Local LLM] Caption complete; server remains resident.")
                return result
            finally:
                # ComfyUI cancellation is a BaseException, so cleanup must be in finally.
                if release_after_run or not succeeded:
                    _SERVER.stop()



class AerinceLocalLLMTextProcess:
    @classmethod
    def INPUT_TYPES(cls):
        models, _ = _models()
        return {
            "required": {
                "text": ("STRING", {
                    "default": "", "multiline": True, "dynamicPrompts": False,
                    "tooltip": "待处理文本。可直接填写，也可将此控件转换为输入以连接其他 STRING 节点。",
                }),
                "model": (list(models) or [NO_MODELS],),
                "llama_server_path": ("STRING", {"default": "", "multiline": False}),
                "release_after_run": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "处理完成后释放本地模型。关闭后保留一个共享服务，便于连续处理文本。",
                }),
                "system_prompt": ("STRING", {
                    "default": TEXT_SYSTEM_PROMPT, "multiline": True, "dynamicPrompts": False,
                    "tooltip": "模型的角色与通用处理规则。",
                }),
                "user_prompt": ("STRING", {
                    "default": TEXT_USER_PROMPT, "multiline": True, "dynamicPrompts": False,
                    "tooltip": "本次处理指令，例如翻译、润色、扩写或总结。留空则将输入文本直接发送给模型。",
                }),
            },
            "optional": {
                "ctx_size": ("INT", {"default": 8192, "min": 2048, "max": 131072, "step": 1024}),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFE,
                    "tooltip": "固定种子可复用相同输入的处理结果；修改种子可重新生成。"}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200,
                    "tooltip": "-1：尽可能将模型层加载到 GPU；0：使用 CPU。纯文本处理不加载 mmproj。"}),
                "timeout": ("INT", {"default": 300, "min": 5, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process_text"
    CATEGORY = "Aerince Tools/LLM"
    DESCRIPTION = "输入文本和处理指令，由本地 GGUF 模型输出处理后的文本。无需图片或 mmproj，与本地图像反推共用一个 llama.cpp 服务。"

    def process_text(
        self, text, model, llama_server_path="", release_after_run=True,
        system_prompt=TEXT_SYSTEM_PROMPT, user_prompt=TEXT_USER_PROMPT,
        ctx_size=8192, max_tokens=1024, temperature=0.2, seed=0,
        gpu_layers=-1, timeout=300,
    ):
        global _TEXT_CACHE

        with _SERVER.lock:
            succeeded = False
            try:
                model_management.throw_exception_if_processing_interrupted()
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("输入文本不能为空。")
                models, _ = _models()
                if model not in models:
                    raise FileNotFoundError(
                        "请选择 models/LLM 或已配置 LLM 目录中的 GGUF 模型，并刷新 ComfyUI 模型列表。"
                        "已搜索：" + ", ".join(_model_roots())
                    )
                if max_tokens >= ctx_size:
                    raise ValueError("max_tokens 必须小于 ctx_size，以便为输入文本和处理指令留出上下文空间。")
                executable = _server_executable(llama_server_path)
                sources = []
                for path in (executable, models[model]):
                    info = Path(path).stat()
                    sources.append((path, info.st_size, info.st_mtime_ns))
                user_content = f"{user_prompt}\n\n{text}" if user_prompt and user_prompt.strip() else text
                messages = [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_content},
                ]
                cache_key = (
                    hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).digest(),
                    tuple(sources), int(ctx_size), int(max_tokens),
                    float(temperature), int(seed), int(gpu_layers),
                )
                if _TEXT_CACHE is not None and _TEXT_CACHE[0] == cache_key:
                    succeeded = True
                    print("[Aerince 本地 LLM] 复用已缓存的文本处理结果，跳过推理。")
                    return _TEXT_CACHE[1]
                deadline = time.monotonic() + int(timeout)
                _SERVER.ensure(executable, models[model], None, ctx_size, gpu_layers, deadline)
                payload = {
                    "model": MODEL_ALIAS,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "seed": int(seed),
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                result = (_extract_text_answer(_SERVER.chat(payload, deadline)),)
                _TEXT_CACHE = (cache_key, result)
                succeeded = True
                if not release_after_run:
                    print("[Aerince 本地 LLM] 文本处理完成，服务保持常驻。")
                return result
            finally:
                # Cancellation is a BaseException; always release on failure.
                if release_after_run or not succeeded:
                    _SERVER.stop()
